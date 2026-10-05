import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = (
    ROOT
    / "examples"
    / "speech-asr"
    / "tools"
    / "validate_cnn_ctc_ir.py"
)


def load_tool():
    spec = importlib.util.spec_from_file_location("validate_cnn_ctc_ir", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load validate_cnn_ctc_ir")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SPEC = {
    "id": "cnn_ctc_v1",
    "input_contract": {"name": "features", "shape": [1, 64, 512]},
    "output_contract": {"name": "logits", "shape": [1, 128, 39]},
}


def contract():
    return {
        "schema": "speech-asr/openvino-ir-contract",
        "version": 1,
        "ir_version": "10",
        "canonical_graph_sha256": "1" * 64,
        "inputs": [
            {
                "name": "features",
                "ports": [{"id": 0, "shape": [1, 64, 512]}],
            }
        ],
        "outputs": [
            {
                "name": "logits",
                "result_name": "logits/sink_port_0",
                "ports": [{"id": 0, "shape": [1, 128, 39]}],
            }
        ],
    }


class CnnCtcIrValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()

    def test_accepts_declared_ir_v10_contract(self):
        result = self.tool.validate(SPEC, contract())
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["ir_version"], "10")
        self.assertEqual(result["input"]["shape"], [1, 64, 512])
        self.assertEqual(result["output"]["shape"], [1, 128, 39])

    def test_rejects_output_shape_change(self):
        value = contract()
        value["outputs"][0]["ports"][0]["shape"] = [1, 127, 39]
        with self.assertRaisesRegex(ValueError, "output shape mismatch"):
            self.tool.validate(SPEC, value)

    def test_accepts_generated_output_name_and_records_provenance(self):
        value = contract()
        value["outputs"][0]["name"] = "/Transpose"
        value["outputs"][0]["result_name"] = "/Transpose/sink_port_0"
        result = self.tool.validate(SPEC, value)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["output"]["declared_name"], "logits")
        self.assertEqual(result["output"]["ir_name"], "/Transpose")
        self.assertEqual(result["output"]["result_name"], "/Transpose/sink_port_0")
        self.assertFalse(result["output"]["name_preserved"])


if __name__ == "__main__":
    unittest.main()
