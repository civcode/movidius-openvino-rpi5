import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.cnn_ctc import load_spec, model_resource_estimate
from speech_asr.orchestration import (
    V12_ARCHITECTURE,
    compatibility_probe_command,
    training_command,
    validate_model_executor_request,
)

SPEC = SPEECH / "models" / "cnn_ctc_v12" / "model_spec.json"


def experiment_model():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v12",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v1"},
        "architecture": dict(V12_ARCHITECTURE),
        "export": {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True},
    }


def train_config():
    return {
        "schema": "speech-asr/train-config",
        "version": 1,
        "seed": 1337,
        "device": "cuda",
        "epochs": 12,
        "batch_size": 1,
        "max_samples": None,
        "checkpoint_selection": "validation_cer",
        "optimizer": {"kind": "adam", "learning_rate": 0.0003},
        "training_manifest": {
            "id": "ami-model-quality-v4-architecture-screen-v1-train",
            "path": "work/speech-asr/ami/model-quality-v4-architecture-screen-v1/train.manifest.jsonl",
            "sha256": "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b",
        },
        "validation_manifest": {
            "id": "ami-model-quality-v4-es2011-validation",
            "path": "work/speech-asr/ami/model-quality-v4/validation.manifest.jsonl",
            "sha256": "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088",
        },
    }


class CnnCtcV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(SPEC)

    def test_wider_deep_dilated_resource_contract(self):
        estimate = model_resource_estimate(self.spec)
        self.assertEqual([x["channels"] for x in self.spec["network"]["stem"]], [128, 448])
        self.assertEqual([x["stride"] for x in self.spec["network"]["stem"]], [2, 2])
        self.assertEqual(len(self.spec["network"]["residual_blocks"]), 8)
        self.assertEqual(
            [x["kernel"] for x in self.spec["network"]["residual_blocks"]],
            [5] * 16,
        )
        self.assertEqual(
            [x["dilation"] for x in self.spec["network"]["residual_blocks"]],
            [1, 2, 3, 4, 4, 3, 2, 1] * 2,
        )
        self.assertEqual(estimate["parameters"], 19627687)
        self.assertEqual(estimate["macs_fixed_input"], 2515673088)
        self.assertEqual(estimate["receptive_field_feature_frames"], 653)
        self.assertEqual(estimate["output_frames"], 128)
        self.assertEqual(self.spec["output_contract"]["shape"], [1, 128, 39])

    def test_request_and_commands_are_registered(self):
        model = experiment_model()
        config = train_config()
        validate_model_executor_request(model, config)
        train = training_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v12"),
            train_config=config,
            model_spec=model,
        )
        self.assertEqual(train[0], "/repo/scripts/train-cnn-ctc-v12.sh")
        probe = compatibility_probe_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v12"),
            model_spec=model,
        )
        self.assertIsNotNone(probe)
        self.assertEqual(probe[0], "/repo/scripts/probe-cnn-ctc-v12.sh")

    def test_initializer_freezes_screen_parent_and_budget(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v12_capacity_probe.py"
        ).read_text(encoding="utf-8")
        self.assertIn('DEFAULT_PARENT = "exp-00e6b1e434d187d8"', source)
        self.assertIn('SCREEN_MANIFEST_SHA256 = "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b"', source)
        self.assertIn('MODEL_SPEC_SHA256 = "0a63c74a2414a26a04bf06da1cc8493b2886257077b451281e0d1da2b107f73a"', source)
        self.assertIn('SCREEN_EPOCHS = 12', source)
        self.assertIn('PARENT_CER = 0.7984219316938317', source)
        self.assertIn('PROMOTION_CER = 0.7744692737430167', source)
        self.assertIn('"max_cer": PARENT_CER', source)
        self.assertIn('"model_id": "cnn_ctc_v12"', source)
        self.assertNotIn('"augmentation": {', source)
        self.assertNotIn('"ctc_objective": {', source)

    def test_entrypoints_exist(self):
        for name in (
            "init-cnn-ctc-v12-capacity-probe.sh",
            "train-cnn-ctc-v12.sh",
            "probe-cnn-ctc-v12.sh",
            "prepare-cnn-ctc-v12.sh",
            "evaluate-cnn-ctc-v12.sh",
        ):
            self.assertTrue((ROOT / "scripts" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
