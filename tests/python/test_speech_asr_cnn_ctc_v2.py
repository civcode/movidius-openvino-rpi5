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
    V2_ARCHITECTURE,
    compatibility_probe_command,
    training_command,
    validate_model_executor_request,
)

SPEC_PATH = SPEECH / "models" / "cnn_ctc_v2" / "model_spec.json"


def experiment_model_spec():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v2",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v1"},
        "architecture": dict(V2_ARCHITECTURE),
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
        "epochs": 1,
        "batch_size": 2,
        "max_samples": None,
        "optimizer": {"kind": "adam", "learning_rate": 0.001},
        "training_manifest": dict(ref),
        "validation_manifest": dict(ref),
    }


class CnnCtcV2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(SPEC_PATH)

    def test_fixed_tensor_contract_and_output_rate(self):
        self.assertEqual(self.spec["input_contract"]["shape"], [1, 64, 512])
        self.assertEqual(self.spec["output_contract"]["shape"], [1, 128, 39])
        self.assertEqual(acoustic_output_length(512, self.spec), 128)

    def test_resource_estimate_matches_frozen_spec(self):
        estimate = model_resource_estimate(self.spec)
        self.assertEqual(estimate["parameters"], 1347463)
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

    def test_generation_one_operator_policy_is_simple(self):
        network = self.spec["network"]
        self.assertEqual(network["normalization"], "batchnorm")
        self.assertEqual(network["activation"], "relu")
        self.assertEqual(
            [block["kernel"] for block in network["residual_blocks"]],
            [11, 19, 27, 35, 43],
        )
        source = (SPEECH / "training" / "cnn_ctc_v2.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("nn.Conv1d", source)
        self.assertIn("nn.BatchNorm1d", source)
        self.assertIn("value = value + residual", source)
        self.assertNotIn("groups=", source)
        self.assertNotIn("MultiheadAttention", source)
        self.assertNotIn("LayerNorm", source)

    def test_model_spec_resource_numbers_are_json_integers(self):
        raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        self.assertIsInstance(raw["network"]["estimated_parameters"], int)
        self.assertIsInstance(raw["network"]["estimated_macs_fixed_input"], int)


class CnnCtcV2ExecutorTests(unittest.TestCase):
    def test_reviewed_v2_request_is_supported(self):
        validate_model_executor_request(experiment_model_spec(), train_config())

    def test_architecture_mutation_is_rejected(self):
        value = experiment_model_spec()
        value["architecture"]["residual_kernels"] = [11, 19, 27, 35, 45]
        with self.assertRaisesRegex(ValueError, "exact frozen architecture"):
            validate_model_executor_request(value, train_config())

    def test_training_command_selects_v2_pipeline(self):
        command = training_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v2"),
            train_config=train_config(),
            model_spec=experiment_model_spec(),
        )
        self.assertEqual(command[0], "/repo/scripts/train-cnn-ctc-v2.sh")
        self.assertIn("--work-dir", command)

    def test_v2_has_pretraining_compatibility_probe(self):
        command = compatibility_probe_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/build/cnn_ctc_v2"),
            model_spec=experiment_model_spec(),
        )
        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(command[0], "/repo/scripts/probe-cnn-ctc-v2.sh")


class CnnCtcV2IrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = SPEECH / "tools" / "validate_cnn_ctc_ir.py"
        module_spec = importlib.util.spec_from_file_location(
            "validate_cnn_ctc_ir_phase11",
            path,
        )
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError("could not load IR validator")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        cls.tool = module
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def test_v2_ir_contract_is_accepted(self):
        contract = {
            "schema": "speech-asr/openvino-ir-contract",
            "version": 1,
            "ir_version": "10",
            "canonical_graph_sha256": "a" * 64,
            "inputs": [
                {
                    "name": "features",
                    "ports": [{"shape": [1, 64, 512]}],
                }
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
        self.assertEqual(result["model_id"], "cnn_ctc_v2")


class Phase11SourceInvariantTests(unittest.TestCase):
    def test_controller_physically_probes_before_training(self):
        source = (SPEECH / "agent" / "run_experiment.py").read_text(
            encoding="utf-8"
        )
        probe_index = source.index("physical_compatibility_probe(")
        execute_index = source.index('"--state",\n            "EXECUTE"')
        self.assertLess(probe_index, execute_index)

    def test_edge_worker_registers_v2_evaluator(self):
        source = (SPEECH / "agent" / "edge_worker.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"cnn_ctc_v2": "evaluate-cnn-ctc-v2.sh"', source)

    def test_initializer_keeps_v1_lineage_and_hardware_gates(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v2_experiment.py"
        ).read_text(encoding="utf-8")
        self.assertIn('DEFAULT_PARENT = "exp-f915ec624a63caf6"', source)
        self.assertIn('"max_realtime_factor": 0.05', source)
        self.assertIn('"max_latency_p95_ms": 100.0', source)
        self.assertIn('"max_cer": 0.913043', source)


if __name__ == "__main__":
    unittest.main()
