import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.cnn_ctc import load_spec, model_resource_estimate
from speech_asr.orchestration import (
    V17_ARCHITECTURE,
    compatibility_probe_command,
    training_command,
    validate_model_executor_request,
)

SPEC = SPEECH / "models" / "cnn_ctc_v17" / "model_spec.json"


def experiment_model():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v17",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v1"},
        "architecture": dict(V17_ARCHITECTURE),
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
        "optimizer": {"kind": "novograd", "learning_rate": 0.01},
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


class CnnCtcV17Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(SPEC)

    def test_quartznet_inference_contract_is_preserved(self):
        network = self.spec["network"]
        estimate = model_resource_estimate(self.spec)
        self.assertEqual(network["kind"], "quartznet-15x5-v1")
        self.assertEqual(network["dropout"], 0)
        self.assertEqual(
            [group["channels"] for group in network["block_groups"]],
            [256, 256, 512, 512, 512],
        )
        self.assertEqual(
            [group["kernel"] for group in network["block_groups"]],
            [33, 39, 51, 63, 75],
        )
        self.assertEqual(estimate["parameters"], 18934631)
        self.assertEqual(estimate["macs_fixed_input"], 4827463680)
        self.assertEqual(estimate["receptive_field_feature_frames"], 8057)
        self.assertEqual(estimate["output_frames"], 256)
        self.assertEqual(self.spec["output_contract"]["shape"], [1, 256, 39])

    def test_reference_training_recipe_is_frozen(self):
        training = self.spec["training"]
        self.assertEqual(training["optimizer"], "novograd")
        self.assertEqual(training["learning_rate"], 0.01)
        self.assertEqual(training["optimizer_betas"], [0.8, 0.5])
        self.assertEqual(training["optimizer_eps"], 1e-8)
        self.assertEqual(training["weight_decay"], 0.001)
        self.assertEqual(training["lr_schedule"], "warmup_cosine")
        self.assertEqual(training["warmup_ratio"], 0.12)
        self.assertEqual(training["min_learning_rate"], 0.00001)
        self.assertEqual(training["gradient_clip_norm"], 5)

    def test_request_and_commands_are_registered(self):
        model = experiment_model()
        config = train_config()
        validate_model_executor_request(model, config)
        train = training_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v17"),
            train_config=config,
            model_spec=model,
        )
        self.assertEqual(train[0], "/repo/scripts/train-cnn-ctc-v17.sh")
        probe = compatibility_probe_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v17"),
            model_spec=model,
        )
        self.assertIsNotNone(probe)
        self.assertEqual(probe[0], "/repo/scripts/probe-cnn-ctc-v17.sh")

    def test_initializer_freezes_v16_evidence_and_screen(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v17_quartznet_reference_training.py"
        ).read_text(encoding="utf-8")
        self.assertIn('DEFAULT_PARENT = "exp-4d7f92c301064c45"', source)
        self.assertIn(
            'MODEL_SPEC_SHA256 = "766a3851160bc4d6d1a5be8f27d45066accce9391683efc99ea5e9833dfb0fa1"',
            source,
        )
        self.assertIn("V16_BEST_VALIDATION_CER = 0.9297356447618499", source)
        self.assertIn('"failure_class") != "hardware_execution"', source)
        self.assertIn("SCREEN_EPOCHS = 12", source)
        self.assertIn('"kind": "novograd"', source)
        self.assertIn('"max_cer": None', source)
        self.assertIn('"max_latency_p95_ms": None', source)

    def test_trainer_and_evaluator_include_stability_fixes(self):
        trainer = (
            SPEECH / "training" / "train_cnn_ctc_v17.py"
        ).read_text(encoding="utf-8")
        evaluator = (
            SPEECH / "evaluation" / "evaluate_cnn_ctc_v17.py"
        ).read_text(encoding="utf-8")
        self.assertIn("NovoGrad(", trainer)
        self.assertIn("warmup_cosine_learning_rate(", trainer)
        self.assertIn("MAX_SERVER_RESTARTS = 4", evaluator)
        self.assertIn("infer_with_restart(", evaluator)

    def test_entrypoints_exist(self):
        for name in (
            "init-cnn-ctc-v17-quartznet-reference-training.sh",
            "train-cnn-ctc-v17.sh",
            "probe-cnn-ctc-v17.sh",
            "prepare-cnn-ctc-v17.sh",
            "evaluate-cnn-ctc-v17.sh",
        ):
            self.assertTrue((ROOT / "scripts" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
