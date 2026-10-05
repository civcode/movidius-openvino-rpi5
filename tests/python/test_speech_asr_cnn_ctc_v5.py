import copy
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.cnn_ctc import (
    acoustic_output_length,
    load_spec,
    model_resource_estimate,
)
from speech_asr.contracts import ContractValidationError, validate_train_config
from speech_asr.orchestration import (
    V5_ARCHITECTURE,
    compatibility_probe_command,
    training_command,
    validate_model_executor_request,
)

SPEC_PATH = SPEECH / "models" / "cnn_ctc_v5" / "model_spec.json"


def experiment_model_spec():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v5",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v1"},
        "architecture": copy.deepcopy(V5_ARCHITECTURE),
        "export": {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True},
    }


def train_config():
    ref = {
        "id": "ami-model-quality-v1",
        "path": "work/manifest.jsonl",
        "sha256": "a" * 64,
    }
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


class CnnCtcV5ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(SPEC_PATH)

    def test_fixed_tensor_contract_and_resource_estimate(self):
        self.assertEqual(self.spec["input_contract"]["shape"], [1, 64, 512])
        self.assertEqual(self.spec["output_contract"]["shape"], [1, 128, 39])
        self.assertEqual(acoustic_output_length(512, self.spec), 128)
        estimate = model_resource_estimate(self.spec)
        self.assertEqual(estimate["parameters"], 304343)
        self.assertEqual(estimate["macs_fixed_input"], 40820736)
        self.assertEqual(estimate["receptive_field_feature_frames"], 237)
        self.assertEqual(
            estimate["parameters"],
            self.spec["network"]["estimated_deployed_parameters"],
        )

    def test_compact_architecture_and_intermediate_head_are_frozen(self):
        network = self.spec["network"]
        self.assertEqual([x["channels"] for x in network["stem"]], [48, 64])
        self.assertEqual(
            [x["kernel"] for x in network["residual_blocks"]],
            [9, 9, 13, 13, 17],
        )
        self.assertEqual(network["intermediate_ctc_after_block"], 3)
        self.assertEqual(self.spec["training"]["intermediate_ctc_weight"], 0.3)
        self.assertEqual(
            self.spec["training"]["checkpoint_selection"],
            "best_validation_cer",
        )

class CnnCtcV5ExecutorTests(unittest.TestCase):
    def test_reviewed_request_is_supported(self):
        validate_model_executor_request(experiment_model_spec(), train_config())

    def test_missing_or_mutated_objective_is_rejected(self):
        missing = train_config()
        del missing["ctc_objective"]
        with self.assertRaisesRegex(ValueError, "requires intermediate"):
            validate_model_executor_request(experiment_model_spec(), missing)
        changed = train_config()
        changed["ctc_objective"]["intermediate_ctc_weight"] = 1.0
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            validate_model_executor_request(experiment_model_spec(), changed)

    def test_contract_accepts_intermediate_ctc_and_rejects_invalid_weight(self):
        self.assertEqual(
            validate_train_config(train_config())["ctc_objective"]["kind"],
            "intermediate-ctc-v1",
        )
        invalid = train_config()
        invalid["ctc_objective"]["intermediate_ctc_weight"] = 0
        with self.assertRaisesRegex(ContractValidationError, "intermediate_ctc_weight"):
            validate_train_config(invalid)

    def test_commands_select_v5_and_propagate_objective(self):
        command = training_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v5"),
            train_config=train_config(),
            model_spec=experiment_model_spec(),
        )
        self.assertEqual(command[0], "/repo/scripts/train-cnn-ctc-v5.sh")
        text = " ".join(command)
        self.assertIn("--ctc-objective-kind intermediate-ctc-v1", text)
        self.assertIn("--intermediate-ctc-weight 0.3", text)
        probe = compatibility_probe_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/build/cnn_ctc_v5"),
            model_spec=experiment_model_spec(),
        )
        self.assertEqual(probe[0], "/repo/scripts/probe-cnn-ctc-v5.sh")

    def test_architecture_mutation_is_rejected(self):
        model = experiment_model_spec()
        model["architecture"]["residual_channels"] = 96
        with self.assertRaisesRegex(ValueError, "exact frozen architecture"):
            validate_model_executor_request(model, train_config())


class CnnCtcV5SourceTests(unittest.TestCase):
    def test_export_forward_does_not_call_intermediate_head(self):
        source = (SPEECH / "training" / "cnn_ctc_v5.py").read_text(
            encoding="utf-8"
        )
        model_source = source.split("class CnnCtcV5", 1)[1]
        forward = model_source.split("    def forward(self, features):", 1)[1].split(
            "    def forward_with_intermediate", 1
        )[0]
        self.assertNotIn("intermediate_projection", forward)
        trainer = (SPEECH / "training" / "train_cnn_ctc_v5.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("model.forward_with_intermediate", trainer)
        self.assertIn("intermediate_ctc_weight * intermediate_loss", trainer)

    def test_entry_points_exist_and_are_executable(self):
        for name in (
            "init-cnn-ctc-v5-interctc.sh",
            "train-cnn-ctc-v5.sh",
            "probe-cnn-ctc-v5.sh",
            "prepare-cnn-ctc-v5.sh",
            "evaluate-cnn-ctc-v5.sh",
        ):
            path = ROOT / "scripts" / name
            self.assertTrue(path.is_file(), name)
            self.assertNotEqual(path.stat().st_mode & 0o111, 0, name)
