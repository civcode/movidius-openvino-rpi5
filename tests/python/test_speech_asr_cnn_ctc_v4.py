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
    V4_ARCHITECTURE,
    compatibility_probe_command,
    training_command,
    validate_model_executor_request,
)

SPEC_PATH = SPEECH / "models" / "cnn_ctc_v4" / "model_spec.json"


def experiment_model_spec():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v4",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v2"},
        "architecture": dict(V4_ARCHITECTURE),
        "export": {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True},
    }


def train_config():
    sha = "a" * 64
    ref = {
        "id": "ami-model-quality-v1",
        "path": "work/manifest.jsonl",
        "sha256": sha,
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
        "optimizer": {"kind": "adam", "learning_rate": 0.0003},
        "training_manifest": dict(ref),
        "validation_manifest": dict(ref),
    }


class CnnCtcV4ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(SPEC_PATH)

    def test_frontend_is_valid_frame_cmvn(self):
        frontend = self.spec["frontend"]
        self.assertEqual(frontend["kind"], "logmel-v2")
        self.assertEqual(
            frontend["normalization"],
            "per_mel_bin_mean_valid_zero_pad",
        )

    def test_graph_contract_matches_v3(self):
        v3 = load_spec(SPEECH / "models" / "cnn_ctc_v3" / "model_spec.json")
        self.assertEqual(self.spec["input_contract"], v3["input_contract"])
        self.assertEqual(self.spec["output_contract"], v3["output_contract"])
        self.assertEqual(self.spec["network"], v3["network"])
        self.assertEqual(acoustic_output_length(512, self.spec), 128)
        self.assertEqual(
            model_resource_estimate(self.spec),
            model_resource_estimate(v3),
        )

    def test_frontend_source_excludes_padding_from_cmvn(self):
        source = (
            SPEECH / "python" / "speech_asr" / "cnn_ctc_frontend.py"
        ).read_text(encoding="utf-8")
        self.assertIn('logged[:, :valid_frames].mean', source)
        self.assertIn('logged[:, valid_frames:] = 0.0', source)
        self.assertIn('"per_mel_bin_mean"', source)
        self.assertIn('"per_mel_bin_mean_valid_zero_pad"', source)


class CnnCtcV4ExecutorTests(unittest.TestCase):
    def test_exact_v4_request_is_supported(self):
        validate_model_executor_request(
            experiment_model_spec(),
            train_config(),
        )

    def test_v4_rejects_old_frontend_declaration(self):
        value = experiment_model_spec()
        value["frontend"] = {"kind": "logmel-v1"}
        with self.assertRaisesRegex(ValueError, "exact frontend"):
            validate_model_executor_request(value, train_config())

    def test_training_command_selects_v4_pipeline(self):
        command = training_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v4"),
            train_config=train_config(),
            model_spec=experiment_model_spec(),
        )
        self.assertEqual(command[0], "/repo/scripts/train-cnn-ctc-v4.sh")
        self.assertIn(
            "--checkpoint-selection validation_cer",
            " ".join(command),
        )

    def test_v4_has_pretraining_graph_probe(self):
        command = compatibility_probe_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/build/cnn_ctc_v4"),
            model_spec=experiment_model_spec(),
        )
        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(command[0], "/repo/scripts/probe-cnn-ctc-v4.sh")


class CnnCtcV4IrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = SPEECH / "tools" / "validate_cnn_ctc_ir.py"
        module_spec = importlib.util.spec_from_file_location(
            "validate_cnn_ctc_ir_v4",
            path,
        )
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError("could not load IR validator")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        cls.tool = module
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def test_v4_ir_contract_is_accepted(self):
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
        self.assertEqual(result["model_id"], "cnn_ctc_v4")


class CnnCtcV4SourceTests(unittest.TestCase):
    def test_v4_pipeline_uses_only_v4_conversion(self):
        for name in ("train-cnn-ctc-v4.sh", "probe-cnn-ctc-v4.sh"):
            source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("prepare-cnn-ctc-v4.sh", source)
            self.assertNotIn("prepare-cnn-ctc-v3.sh", source)
            self.assertIn("cnn_ctc_v4.xml", source)
            self.assertIn("cnn_ctc_v4.bin", source)

    def test_edge_worker_registers_v4_evaluator(self):
        source = (SPEECH / "agent" / "edge_worker.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"cnn_ctc_v4": "evaluate-cnn-ctc-v4.sh"', source)

    def test_initializer_branches_from_best_non_augmented_parent(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v4_valid_cmvn.py"
        ).read_text(encoding="utf-8")
        self.assertIn('DEFAULT_PARENT = "exp-3c7727ca3f37ba2c"', source)
        self.assertIn('"model_id": "cnn_ctc_v4"', source)
        self.assertIn('"frontend": {"kind": "logmel-v2"}', source)
        self.assertIn('"checkpoint_selection": "validation_cer"', source)
        self.assertNotIn('"augmentation": {', source)


if __name__ == "__main__":
    unittest.main()
