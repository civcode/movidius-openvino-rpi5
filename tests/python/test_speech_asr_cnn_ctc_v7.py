import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.cnn_ctc import load_spec, model_resource_estimate
from speech_asr.orchestration import (
    V7_ARCHITECTURE,
    compatibility_probe_command,
    training_command,
    validate_model_executor_request,
)

SPEC_PATH = SPEECH / "models" / "cnn_ctc_v7" / "model_spec.json"


def experiment_model_spec():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v7",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v1"},
        "architecture": dict(V7_ARCHITECTURE),
        "export": {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True},
    }


def train_config():
    return {
        "schema": "speech-asr/train-config",
        "version": 1,
        "seed": 1337,
        "device": "cuda",
        "epochs": 32,
        "batch_size": 1,
        "max_samples": None,
        "checkpoint_selection": "validation_cer",
        "optimizer": {"kind": "adam", "learning_rate": 0.0003},
        "training_manifest": {
            "id": "ami-model-quality-v4-train",
            "path": "work/speech-asr/ami/model-quality-v4/train.manifest.jsonl",
            "sha256": "6025d17f08c1815d1710c3ab56cea51a34365f398e9877b4aefec234e64e4096",
        },
        "validation_manifest": {
            "id": "ami-model-quality-v4-es2011-validation",
            "path": "work/speech-asr/ami/model-quality-v4/validation.manifest.jsonl",
            "sha256": "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088",
        },
    }


class CnnCtcV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(SPEC_PATH)

    def test_width_only_resource_contract(self):
        estimate = model_resource_estimate(self.spec)
        self.assertEqual([x["channels"] for x in self.spec["network"]["stem"]], [64, 112])
        self.assertTrue(
            all(x["channels"] == 112 for x in self.spec["network"]["residual_blocks"])
        )
        self.assertEqual(
            [x["kernel"] for x in self.spec["network"]["residual_blocks"]],
            [11, 19, 27, 35, 43],
        )
        self.assertEqual(estimate["parameters"], 1818183)
        self.assertEqual(estimate["macs_fixed_input"], 235177984)
        self.assertEqual(estimate["receptive_field_feature_frames"], 533)
        self.assertEqual(estimate["output_frames"], 128)

    def test_request_and_commands_are_registered(self):
        model = experiment_model_spec()
        config = train_config()
        validate_model_executor_request(model, config)
        train = training_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v7"),
            train_config=config,
            model_spec=model,
        )
        self.assertEqual(train[0], "/repo/scripts/train-cnn-ctc-v7.sh")
        probe = compatibility_probe_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v7"),
            model_spec=model,
        )
        self.assertIsNotNone(probe)
        self.assertEqual(probe[0], "/repo/scripts/probe-cnn-ctc-v7.sh")

    def test_initializer_freezes_same_manifest_parent_and_cer_gate(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v7_wide.py"
        ).read_text(encoding="utf-8")
        self.assertIn('DEFAULT_PARENT = "exp-63fdb8d218673527"', source)
        self.assertIn('PARENT_CER = 0.7588550365720209', source)
        self.assertIn('"max_cer": PARENT_CER', source)
        self.assertIn('"model_id": "cnn_ctc_v7"', source)
        self.assertIn('"checkpoint_selection": "validation_cer"', source)
        self.assertNotIn('"augmentation": {', source)
        self.assertNotIn('"ctc_objective": {', source)
        self.assertIn("sealed held-out metrics must remain forbidden", source)

    def test_v7_entrypoints_exist(self):
        for name in (
            "init-cnn-ctc-v7-wide.sh",
            "train-cnn-ctc-v7.sh",
            "probe-cnn-ctc-v7.sh",
            "prepare-cnn-ctc-v7.sh",
            "evaluate-cnn-ctc-v7.sh",
        ):
            self.assertTrue((ROOT / "scripts" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
