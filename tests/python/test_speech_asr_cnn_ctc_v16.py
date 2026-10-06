import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.cnn_ctc import load_spec, model_resource_estimate
from speech_asr.orchestration import (
    V16_ARCHITECTURE,
    compatibility_probe_command,
    training_command,
    validate_model_executor_request,
)

SPEC = SPEECH / "models" / "cnn_ctc_v16" / "model_spec.json"


def experiment_model():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v16",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v1"},
        "architecture": dict(V16_ARCHITECTURE),
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


class CnnCtcV16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(SPEC)

    def test_quartznet_resource_contract(self):
        network = self.spec["network"]
        estimate = model_resource_estimate(self.spec)
        self.assertEqual(network["kind"], "quartznet-15x5-v1")
        self.assertEqual(network["normalization"], "batchnorm")
        self.assertEqual(network["separable_convolution"], "depthwise-pointwise")
        self.assertEqual(network["prologue"]["channels"], 256)
        self.assertEqual(network["prologue"]["kernel"], 33)
        self.assertEqual(network["prologue"]["stride"], 2)
        self.assertEqual(
            [group["channels"] for group in network["block_groups"]],
            [256, 256, 512, 512, 512],
        )
        self.assertEqual(
            [group["kernel"] for group in network["block_groups"]],
            [33, 39, 51, 63, 75],
        )
        self.assertTrue(
            all(group["block_repeats"] == 3 for group in network["block_groups"])
        )
        self.assertTrue(
            all(group["module_repeats"] == 5 for group in network["block_groups"])
        )
        self.assertEqual(network["epilogue"][0]["kernel"], 87)
        self.assertEqual(network["epilogue"][0]["dilation"], 2)
        self.assertEqual(network["epilogue"][1]["channels"], 1024)
        self.assertEqual(estimate["parameters"], 18934631)
        self.assertEqual(estimate["macs_fixed_input"], 4827463680)
        self.assertEqual(estimate["receptive_field_feature_frames"], 8057)
        self.assertEqual(estimate["fp16_weight_bytes"], 37869262)
        self.assertEqual(estimate["output_frames"], 256)
        self.assertEqual(self.spec["output_contract"]["shape"], [1, 256, 39])

    def test_request_and_commands_are_registered(self):
        model = experiment_model()
        config = train_config()
        validate_model_executor_request(model, config)
        train = training_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v16"),
            train_config=config,
            model_spec=model,
        )
        self.assertEqual(train[0], "/repo/scripts/train-cnn-ctc-v16.sh")
        probe = compatibility_probe_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v16"),
            model_spec=model,
        )
        self.assertIsNotNone(probe)
        self.assertEqual(probe[0], "/repo/scripts/probe-cnn-ctc-v16.sh")

    def test_initializer_freezes_parent_and_screen(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v16_quartznet.py"
        ).read_text(encoding="utf-8")
        self.assertIn('DEFAULT_PARENT = "exp-5d1209f5120388f8"', source)
        self.assertIn(
            'SCREEN_MANIFEST_SHA256 = "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b"',
            source,
        )
        self.assertIn(
            'MODEL_SPEC_SHA256 = "70320bc878891f843312f0d8c1ae41ad7f414dbe2a0537e6766af8bf15f70a93"',
            source,
        )
        self.assertIn("SCREEN_EPOCHS = 12", source)
        self.assertIn("PARENT_CER = 0.8428267004549905", source)
        self.assertIn('"max_cer": None', source)
        self.assertIn('"max_realtime_factor": None', source)
        self.assertIn('"max_latency_p95_ms": None', source)
        self.assertIn('"model_id": "cnn_ctc_v16"', source)
        self.assertIn('"learning_rate": float(package["training"]["learning_rate"])', source)
        self.assertNotIn('"augmentation": {', source)
        self.assertNotIn('"ctc_objective": {', source)

    def test_vpu_cpu_split_is_explicit(self):
        split = self.spec["deployment"]["split"]
        self.assertIn("CTC logits", split["vpu"])
        self.assertIn("prefix beam", split["cpu"])

    def test_entrypoints_exist(self):
        for name in (
            "init-cnn-ctc-v16-quartznet.sh",
            "train-cnn-ctc-v16.sh",
            "probe-cnn-ctc-v16.sh",
            "prepare-cnn-ctc-v16.sh",
            "evaluate-cnn-ctc-v16.sh",
        ):
            self.assertTrue((ROOT / "scripts" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
