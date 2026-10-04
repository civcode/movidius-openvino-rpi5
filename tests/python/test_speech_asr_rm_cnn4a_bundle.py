import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
TOOL = SPEECH / "tools" / "verify_prepared_rm_cnn4a.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("verify_prepared_rm_cnn4a", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load rm_cnn4a verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IR = """<net name="synthetic-rm-cnn4a" version="10">
<layers>
  <layer id="0" name="input" type="Input" precision="FP32">
    <output><port id="0" precision="FP32"><dim>1</dim><dim>40</dim></port></output>
  </layer>
  <layer id="1" name="scores" type="FullyConnected" precision="FP16">
    <input><port id="0"><dim>1</dim><dim>40</dim></port></input>
    <output><port id="1" precision="FP16"><dim>1</dim><dim>10</dim></port></output>
  </layer>
</layers>
<edges><edge from-layer="0" from-port="0" to-layer="1" to-port="0"/></edges>
</net>"""


class PreparedRmCnn4aVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()

    def make_bundle(self, root: pathlib.Path):
        model_root = root / "model"
        source = model_root / "source"
        ir = model_root / "openvino" / "fp16"
        source.mkdir(parents=True)
        ir.mkdir(parents=True)

        names = [
            "rm_cnn4a.nnet",
            "rm_cnn4a.counts",
            "rm_cnn4a.mapping",
            "rm_cnn4a.md",
            "feat1_10.ark",
            "score1_10.ark",
        ]
        for index, name in enumerate(names):
            (source / name).write_bytes(f"{name}:{index}".encode("utf-8"))
        (source / "LICENSE.txt").write_text("license", encoding="utf-8")

        lock_lines = []
        for name in sorted(names + ["LICENSE.txt"]):
            digest = hashlib.sha256((source / name).read_bytes()).hexdigest()
            lock_lines.append(f"{digest}  {name}")
        (source / "SOURCE-LOCK.sha256").write_text(
            "\n".join(lock_lines) + "\n",
            encoding="utf-8",
        )

        source_spec = {
            "schema": "speech-asr/model-source",
            "version": 1,
            "id": "synthetic",
            "source_release": "synthetic",
            "files": [{"name": name, "role": "fixture"} for name in names],
        }
        source_spec_path = root / "source-v1.json"
        source_spec_path.write_text(
            json.dumps(source_spec, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        xml = ir / "rm_cnn4a_fp16.xml"
        binary = ir / "rm_cnn4a_fp16.bin"
        xml.write_text(IR, encoding="utf-8")
        binary.write_bytes(b"weights")
        contract = self.tool.inspect_ir(xml, binary)
        (ir / "ir-contract.json").write_text(
            json.dumps(contract, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        source_hashes = {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in names
        }
        model_spec = {
            "schema": "speech-asr/prepared-model",
            "version": 1,
            "id": "rm_cnn4a-fp16",
            "family": "rm_cnn4a",
            "source_spec_sha256": hashlib.sha256(
                source_spec_path.read_bytes()
            ).hexdigest(),
            "source_release": "synthetic",
            "source_artifact_sha256": source_hashes,
            "openvino": {
                "version": "2020.3.2",
                "precision": "FP16",
                "xml_sha256": contract["xml_sha256"],
                "bin_sha256": contract["bin_sha256"],
                "ir_contract": contract,
            },
            "reference_fixture": {
                "features": "source/feat1_10.ark",
                "scores": "source/score1_10.ark",
            },
        }
        (model_root / "openvino" / "model-spec.json").write_text(
            json.dumps(model_spec, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return model_root, source_spec_path

    def test_verifies_complete_bundle(self):
        with tempfile.TemporaryDirectory() as name:
            model_root, spec = self.make_bundle(pathlib.Path(name))
            result = self.tool.verify_bundle(model_root, spec)
            self.assertEqual(result["id"], "rm_cnn4a-fp16")
            self.assertEqual(result["inputs"][0]["name"], "input")
            self.assertEqual(result["outputs"][0]["name"], "scores")

    def test_detects_source_mutation(self):
        with tempfile.TemporaryDirectory() as name:
            model_root, spec = self.make_bundle(pathlib.Path(name))
            (model_root / "source" / "feat1_10.ark").write_bytes(b"mutated")
            with self.assertRaisesRegex(self.tool.VerificationError, "source hash mismatch"):
                self.tool.verify_bundle(model_root, spec)


if __name__ == "__main__":
    unittest.main()
