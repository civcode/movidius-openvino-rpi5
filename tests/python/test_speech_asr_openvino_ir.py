import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "speech-asr" / "python"))

from speech_asr.openvino_ir import canonical_graph_manifest, inspect_ir
from xml.etree import ElementTree as ET


IR = """<net name="synthetic" version="10">
<layers>
  <layer id="0" name="input" type="Input" precision="FP32">
    <output><port id="0" precision="FP32"><dim>1</dim><dim>40</dim></port></output>
  </layer>
  <layer id="1" name="fc" type="FullyConnected" precision="FP16">
    <input>
      <port id="0"><dim>1</dim><dim>40</dim></port>
      <port id="1"><dim>10</dim><dim>40</dim></port>
    </input>
    <output><port id="2" precision="FP16"><dim>1</dim><dim>10</dim></port></output>
  </layer>
  <layer id="2" name="weights" type="Const" precision="FP16">
    <output><port id="0" precision="FP16"><dim>10</dim><dim>40</dim></port></output>
  </layer>
</layers>
<edges>
  <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
  <edge from-layer="2" from-port="0" to-layer="1" to-port="1"/>
</edges>
<meta_data><cli_parameters><input_model value="/host/a/model.nnet"/></cli_parameters></meta_data>
</net>"""


class IrInspectorTests(unittest.TestCase):
    def test_identifies_input_and_sink_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            xml = root / "model.xml"
            binary = root / "model.bin"
            xml.write_text(IR, encoding="utf-8")
            binary.write_bytes(b"weights")
            result = inspect_ir(xml, binary)
            self.assertEqual(result["network_name"], "synthetic")
            self.assertEqual(len(result["inputs"]), 1)
            self.assertEqual(result["inputs"][0]["name"], "input")
            self.assertEqual(result["inputs"][0]["ports"][0]["shape"], [1, 40])
            self.assertNotIn("weights", [item["name"] for item in result["inputs"]])
            self.assertEqual(result["outputs"][0]["name"], "fc")
            self.assertEqual(result["outputs"][0]["ports"][0]["shape"], [1, 10])
            self.assertEqual(len(result["xml_sha256"]), 64)
            self.assertEqual(len(result["graph_sha256"]), 64)
            self.assertEqual(len(result["canonical_graph_sha256"]), 64)
            self.assertEqual(len(result["bin_sha256"]), 64)

    def test_canonical_graph_hash_ignores_layer_ids_and_order(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            first = root / "first.xml"
            second = root / "second.xml"
            first.write_text(IR, encoding="utf-8")
            variant = (
                IR.replace('id="0" name="input"', 'id="9" name="input"')
                  .replace('id="1" name="fc"', 'id="4" name="fc"')
                  .replace('id="2" name="weights"', 'id="7" name="weights"')
                  .replace('from-layer="0"', 'from-layer="9"')
                  .replace('from-layer="2"', 'from-layer="7"')
                  .replace('to-layer="1"', 'to-layer="4"')
            )
            # Reorder layer elements without altering names/content.
            start = variant.index("<layers>") + len("<layers>")
            end = variant.index("</layers>")
            body = variant[start:end]
            chunks = [chunk for chunk in body.split("  <layer ") if chunk.strip()]
            reordered = "  <layer " + "  <layer ".join(reversed(chunks))
            variant = variant[:start] + "\n" + reordered + variant[end:]
            second.write_text(variant, encoding="utf-8")
            a = inspect_ir(first)
            b = inspect_ir(second)
            self.assertNotEqual(a["graph_sha256"], b["graph_sha256"])
            self.assertEqual(
                a["canonical_graph_sha256"],
                b["canonical_graph_sha256"],
            )

    def test_generated_cast_regex_matches_numeric_suffix(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            xml = root / "model.xml"
            xml.write_text(
                IR.replace('name="weights"', 'name="reshape/Cast_12591_const"'),
                encoding="utf-8",
            )
            manifest = canonical_graph_manifest(ET.parse(xml).getroot())
            names = [item["name"] for item in manifest["layers"]]
            self.assertTrue(
                any("Cast_<auto>_const@" in name for name in names),
                names,
            )

    def test_canonical_graph_hash_normalizes_generated_cast_const_suffix(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            first = root / "first.xml"
            second = root / "second.xml"
            first.write_text(
                IR.replace('name="weights"', 'name="reshape/Cast_12591_const"'),
                encoding="utf-8",
            )
            second.write_text(
                IR.replace('name="weights"', 'name="reshape/Cast_12597_const"'),
                encoding="utf-8",
            )
            a = inspect_ir(first)
            b = inspect_ir(second)
            self.assertNotEqual(a["graph_sha256"], b["graph_sha256"])
            self.assertEqual(
                a["canonical_graph_sha256"],
                b["canonical_graph_sha256"],
            )

    def test_manifest_pinpoints_layer_sections(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            first = root / "first.xml"
            second = root / "second.xml"
            first.write_text(IR, encoding="utf-8")
            second.write_text(
                IR.replace("<dim>10</dim>", "<dim>11</dim>", 1),
                encoding="utf-8",
            )
            a = canonical_graph_manifest(ET.parse(first).getroot())
            b = canonical_graph_manifest(ET.parse(second).getroot())
            self.assertEqual(a["layer_count"], b["layer_count"])
            self.assertEqual(a["edge_count"], b["edge_count"])
            a_layers = {item["name"]: item for item in a["layers"]}
            b_layers = {item["name"]: item for item in b["layers"]}
            self.assertNotEqual(a_layers["fc"]["sha256"], b_layers["fc"]["sha256"])
            self.assertEqual(a_layers["input"]["sha256"], b_layers["input"]["sha256"])

    def test_graph_hash_ignores_only_top_level_model_optimizer_metadata(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            first = root / "first.xml"
            second = root / "second.xml"
            first.write_text(IR, encoding="utf-8")
            second.write_text(
                IR.replace("/host/a/model.nnet", "/other/host/model.nnet"),
                encoding="utf-8",
            )
            a = inspect_ir(first)
            b = inspect_ir(second)
            self.assertNotEqual(a["xml_sha256"], b["xml_sha256"])
            self.assertEqual(a["graph_sha256"], b["graph_sha256"])

            changed = root / "changed.xml"
            changed.write_text(
                IR.replace('precision="FP16"><dim>1</dim><dim>10</dim>',
                           'precision="FP16"><dim>1</dim><dim>11</dim>'),
                encoding="utf-8",
            )
            self.assertNotEqual(
                a["graph_sha256"],
                inspect_ir(changed)["graph_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
