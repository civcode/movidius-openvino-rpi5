import importlib.util
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
from speech_asr.orchestration import (
    V3_ARCHITECTURE,
    compatibility_probe_command,
    pretraining_myriad_compatibility,
    training_command,
    validate_model_executor_request,
)

SPEC_PATH = SPEECH / "models" / "cnn_ctc_v3" / "model_spec.json"


def experiment_model_spec():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v3",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v1"},
        "architecture": dict(V3_ARCHITECTURE),
        "export": {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True},
    }


def train_config():
    sha = "a" * 64
    ref = {"id": "ami-smoke-v1", "path": "work/manifest.jsonl", "sha256": sha}
    return {
        "schema": "speech-asr/train-config",
        "version": 1,
        "seed": 1337,
        "device": "cuda",
        "epochs": 32,
        "batch_size": 1,
        "max_samples": None,
        "optimizer": {"kind": "adam", "learning_rate": 0.0003},
        "training_manifest": dict(ref),
        "validation_manifest": dict(ref),
    }


class CnnCtcV3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(SPEC_PATH)

    def test_fixed_contract_matches_v2_geometry(self):
        self.assertEqual(self.spec["input_contract"]["shape"], [1, 64, 512])
        self.assertEqual(self.spec["output_contract"]["shape"], [1, 128, 39])
        self.assertEqual(acoustic_output_length(512, self.spec), 128)

    def test_resource_estimate_matches_frozen_spec(self):
        estimate = model_resource_estimate(self.spec)
        self.assertEqual(estimate["parameters"], 1346343)
        self.assertEqual(estimate["macs_fixed_input"], 174804992)
        self.assertEqual(estimate["receptive_field_feature_frames"], 533)
        self.assertEqual(estimate["output_frames"], 128)
        self.assertEqual(
            estimate["parameters"],
            self.spec["network"]["estimated_parameters"],
        )
        self.assertEqual(
            estimate["macs_fixed_input"],
            self.spec["network"]["estimated_macs_fixed_input"],
        )

    def test_inference_graph_policy_is_normalization_free(self):
        network = self.spec["network"]
        self.assertEqual(network["normalization"], "none")
        self.assertEqual(
            network["residual_projection_init"],
            "kaiming_scaled_0.01",
        )
        source = (SPEECH / "training" / "cnn_ctc_v3.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("nn.Conv1d", source)
        self.assertIn("value = value + residual", source)
        self.assertIn("nn.init.kaiming_normal_", source)
        self.assertIn("self.projection.weight.mul_(0.01)", source)
        self.assertNotIn("BatchNorm", source)
        self.assertNotIn("groups=", source)
        self.assertNotIn("MultiheadAttention", source)
        self.assertNotIn("LayerNorm", source)

    def test_training_budget_repairs_generation_one_undertraining(self):
        training = self.spec["training"]
        self.assertEqual(training["epochs"], 32)
        self.assertEqual(training["batch_size"], 1)
        self.assertEqual(training["learning_rate"], 0.0003)
        self.assertEqual(training["lr_schedule"], "cosine")
        self.assertEqual(training["min_learning_rate"], 0.00003)
        self.assertEqual(training["gradient_clip_norm"], 5.0)
        self.assertEqual(
            training["checkpoint_selection"],
            "best_validation_loss",
        )


class CnnCtcV3ExecutorTests(unittest.TestCase):
    def test_reviewed_v3_request_is_supported(self):
        validate_model_executor_request(experiment_model_spec(), train_config())

    def test_v3_architecture_mutation_is_rejected(self):
        value = experiment_model_spec()
        value["architecture"]["normalization"] = "batchnorm"
        with self.assertRaisesRegex(ValueError, "exact frozen architecture"):
            validate_model_executor_request(value, train_config())

    def test_training_command_selects_v3_pipeline(self):
        command = training_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v3"),
            train_config=train_config(),
            model_spec=experiment_model_spec(),
        )
        self.assertEqual(command[0], "/repo/scripts/train-cnn-ctc-v3.sh")
        text = " ".join(command)
        self.assertIn("--epochs 32", text)
        self.assertIn("--batch-size 1", text)
        self.assertIn("--learning-rate 0.0003", text)

    def test_v3_has_pretraining_compatibility_probe(self):
        command = compatibility_probe_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/build/cnn_ctc_v3"),
            model_spec=experiment_model_spec(),
        )
        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(command[0], "/repo/scripts/probe-cnn-ctc-v3.sh")


class CnnCtcV3PretrainingGateTests(unittest.TestCase):
    def test_low_argmax_agreement_can_pass_when_logits_are_numerically_close(self):
        result = pretraining_myriad_compatibility(
            {
                "max_abs_error": 0.002,
                "frame_argmax_agreement": 0.984375,
                "frame_argmax_mismatches": 2,
                "min_reference_top2_margin": 0.0001,
            }
        )
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["frame_argmax_agreement"], 0.984375)

    def test_large_numerical_error_still_fails_pretraining_gate(self):
        result = pretraining_myriad_compatibility(
            {
                "max_abs_error": 0.011,
                "frame_argmax_agreement": 1.0,
                "frame_argmax_mismatches": 0,
                "min_reference_top2_margin": 0.1,
            }
        )
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(result["reasons"])


class CnnCtcV3IrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = SPEECH / "tools" / "validate_cnn_ctc_ir.py"
        module_spec = importlib.util.spec_from_file_location(
            "validate_cnn_ctc_ir_phase11_v3",
            path,
        )
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError("could not load IR validator")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        cls.tool = module
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def test_v3_ir_contract_is_accepted(self):
        contract = {
            "schema": "speech-asr/openvino-ir-contract",
            "version": 1,
            "ir_version": "10",
            "canonical_graph_sha256": "a" * 64,
            "inputs": [
                {"name": "features", "ports": [{"shape": [1, 64, 512]}]}
            ],
            "outputs": [
                {
                    "name": "/Transpose",
                    "result_name": "/Transpose/sink_port_0",
                    "ports": [{"shape": [1, 128, 39]}],
                }
            ],
        }
        result = self.tool.validate(self.spec, contract)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["model_id"], "cnn_ctc_v3")


class Phase11Generation2SourceTests(unittest.TestCase):
    def test_training_records_optimizer_step_and_best_checkpoint_evidence(self):
        source = (
            SPEECH / "training" / "train_cnn_ctc_v3.py"
        ).read_text(encoding="utf-8")
        for marker in (
            "optimizer_steps",
            "best_validation_loss",
            "best_epoch",
            "clip_grad_norm_",
            "CosineAnnealingLR",
        ):
            self.assertIn(marker, source)

    def test_edge_worker_registers_v3_evaluator(self):
        source = (SPEECH / "agent" / "edge_worker.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"cnn_ctc_v3": "evaluate-cnn-ctc-v3.sh"', source)

    def test_v3_probe_uses_v3_conversion_and_checks_artifacts(self):
        source = (ROOT / "scripts" / "probe-cnn-ctc-v3.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("prepare-cnn-ctc-v3.sh", source)
        self.assertNotIn("prepare-cnn-ctc-v2.sh", source)
        self.assertIn('test -s "$IR/cnn_ctc_v3.xml"', source)
        self.assertIn('test -s "$IR/cnn_ctc_v3.bin"', source)

    def test_initializer_uses_generation_one_parent_and_tighter_hw_gates(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v3_experiment.py"
        ).read_text(encoding="utf-8")
        self.assertIn('DEFAULT_PARENT = "exp-1c682f4eda475a01"', source)
        self.assertIn('"max_cer": 0.913043', source)
        self.assertIn('"max_realtime_factor": 0.01', source)
        self.assertIn('"max_latency_p95_ms": 25.0', source)


if __name__ == "__main__":
    unittest.main()
