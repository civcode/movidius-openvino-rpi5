import copy
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.cnn_ctc import acoustic_output_length, load_spec, model_resource_estimate
from speech_asr.orchestration import (
    V3_ARCHITECTURE,
    V6_ARCHITECTURE,
    compatibility_probe_command,
    training_command,
    validate_model_executor_request,
)

SPEC_PATH = SPEECH / "models" / "cnn_ctc_v6" / "model_spec.json"


def experiment_model_spec():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v6",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v1"},
        "architecture": copy.deepcopy(V6_ARCHITECTURE),
        "export": {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True},
    }


def train_config():
    ref = {"id": "quality", "path": "work/manifest.jsonl", "sha256": "a" * 64}
    return {
        "schema": "speech-asr/train-config",
        "version": 1,
        "seed": 1337,
        "device": "cuda",
        "epochs": 32,
        "batch_size": 1,
        "max_samples": None,
        "checkpoint_selection": "validation_cer",
        "ctc_objective": {
            "kind": "intermediate-ctc-v1",
            "intermediate_ctc_weight": 0.3,
        },
        "optimizer": {"kind": "adam", "learning_rate": 0.0003},
        "training_manifest": dict(ref),
        "validation_manifest": dict(ref),
    }


class CnnCtcV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(SPEC_PATH)

    def test_deployed_geometry_matches_v3(self):
        estimate = model_resource_estimate(self.spec)
        self.assertEqual(acoustic_output_length(512, self.spec), 128)
        self.assertEqual(estimate["parameters"], 1346343)
        self.assertEqual(estimate["macs_fixed_input"], 174804992)
        self.assertEqual(estimate["receptive_field_feature_frames"], 533)
        v6_without_objective = dict(V6_ARCHITECTURE)
        del v6_without_objective["kind"]
        del v6_without_objective["intermediate_ctc_after_block"]
        v3_without_kind = dict(V3_ARCHITECTURE)
        del v3_without_kind["kind"]
        self.assertEqual(v6_without_objective, v3_without_kind)

    def test_reviewed_request_and_commands_are_bound(self):
        validate_model_executor_request(experiment_model_spec(), train_config())
        command = training_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/v6"),
            train_config=train_config(),
            model_spec=experiment_model_spec(),
        )
        self.assertEqual(command[0], "/repo/scripts/train-cnn-ctc-v6.sh")
        self.assertIn(
            "--intermediate-ctc-weight 0.3",
            " ".join(command),
        )
        probe = compatibility_probe_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/v6"),
            model_spec=experiment_model_spec(),
        )
        self.assertEqual(probe[0], "/repo/scripts/probe-cnn-ctc-v6.sh")

    def test_deployed_projection_is_initialized_before_auxiliary_projection(self):
        source = (SPEECH / "training" / "cnn_ctc_v6.py").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            source.index("self.projection = nn.Conv1d"),
            source.index("self.intermediate_projection = nn.Conv1d"),
        )
        model_source = source.split("class CnnCtcV6", 1)[1]
        forward = model_source.split("    def forward(self, features):", 1)[1].split(
            "    def forward_with_intermediate", 1
        )[0]
        self.assertNotIn("intermediate_projection", forward)

    def test_entry_points_are_executable(self):
        for name in (
            "init-cnn-ctc-v6-interctc.sh",
            "train-cnn-ctc-v6.sh",
            "probe-cnn-ctc-v6.sh",
            "prepare-cnn-ctc-v6.sh",
            "evaluate-cnn-ctc-v6.sh",
        ):
            path = ROOT / "scripts" / name
            self.assertTrue(path.is_file(), name)
            self.assertNotEqual(path.stat().st_mode & 0o111, 0, name)
