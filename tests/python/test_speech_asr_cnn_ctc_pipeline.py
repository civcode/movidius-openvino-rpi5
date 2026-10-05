import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
IR_VALIDATOR = (
    ROOT / "examples" / "speech-asr" / "tools" / "validate_cnn_ctc_ir.py"
)


def load_ir_validator():
    spec = importlib.util.spec_from_file_location("validate_cnn_ctc_ir", IR_VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load cnn_ctc IR validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CnnCtcIrValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = load_ir_validator()

    @staticmethod
    def model_spec():
        return {
            "id": "cnn_ctc_v1",
            "input_contract": {
                "name": "features",
                "shape": [1, 64, 512],
            },
            "output_contract": {
                "name": "logits",
                "shape": [1, 128, 39],
            },
        }

    @staticmethod
    def ir_contract(output_name="/Transpose"):
        return {
            "schema": "speech-asr/openvino-ir-contract",
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
                    "name": output_name,
                    "type": "Transpose",
                    "ports": [{"shape": [1, 128, 39]}],
                    "result_name": "logits/sink_port_0",
                }
            ],
        }

    def test_accepts_mo_generated_single_output_name(self):
        result = self.validator.validate(self.model_spec(), self.ir_contract())
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["output"]["declared_name"], "logits")
        self.assertEqual(result["output"]["ir_name"], "/Transpose")
        self.assertFalse(result["output"]["name_preserved"])
        self.assertEqual(result["output"]["shape"], [1, 128, 39])

    def test_records_preserved_output_name_when_available(self):
        result = self.validator.validate(
            self.model_spec(),
            self.ir_contract(output_name="logits"),
        )
        self.assertTrue(result["output"]["name_preserved"])
        self.assertEqual(result["output"]["ir_name"], "logits")

    def test_still_rejects_wrong_output_shape(self):
        contract = self.ir_contract()
        contract["outputs"][0]["ports"][0]["shape"] = [1, 127, 39]
        with self.assertRaisesRegex(ValueError, "IR output shape mismatch"):
            self.validator.validate(self.model_spec(), contract)


class CnnCtcPipelineSourceTests(unittest.TestCase):
    def test_training_environment_is_uv_managed(self):
        prepare = (ROOT / "scripts" / "prepare-python-env.sh").read_text(encoding="utf-8")
        self.assertIn("work/venv-training", prepare)
        self.assertIn("requirements/training.txt", prepare)

    def test_probe_orders_compatibility_before_device(self):
        probe = (ROOT / "scripts" / "probe-cnn-ctc-v1.sh").read_text(encoding="utf-8")
        self.assertLess(
            probe.index("compare ONNX Runtime"),
            probe.index("MYRIAD load/compile/infer"),
        )
        self.assertIn("--init-only", probe)
        self.assertIn("--no-device", probe)
        self.assertNotIn('[[ -s "$MANIFEST" ]]', probe)

    def test_init_only_does_not_construct_manifest_dataset(self):
        source = (
            ROOT / "examples" / "speech-asr" / "training" / "train_cnn_ctc_v1.py"
        ).read_text(encoding="utf-8")
        guard = source.index("if not args.init_only:")
        dataset = source.index("train_dataset = ManifestCtcDataset(")
        self.assertLess(guard, dataset)
        self.assertIn('"manifest": None if args.init_only else str(args.manifest)', source)

    def test_full_pipeline_exports_before_mo(self):
        pipeline = (ROOT / "scripts" / "train-cnn-ctc-v1.sh").read_text(encoding="utf-8")
        self.assertLess(
            pipeline.index("export fixed-shape ONNX"),
            pipeline.index("convert OpenVINO"),
        )
        self.assertIn("--device", pipeline)

    def test_myriad_evaluator_uses_standard_result_contract(self):
        source = (
            ROOT / "examples" / "speech-asr" / "evaluation" / "evaluate_cnn_ctc_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"speech-asr/experiment-result"', source)
        self.assertIn("validate_experiment_result", source)
        self.assertIn('"backend": "MYRIAD"', source)


if __name__ == "__main__":
    unittest.main()
