import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "speech-asr" / "python"))

from speech_asr.openvino_ir import inspect_ir


IR = """<net name="synthetic" version="10">
<layers>
  <layer id="0" name="input" type="Input" precision="FP32">
    <output><port id="0" precision="FP32"><dim>1</dim><dim>40</dim></port></output>
  </layer>
  <layer id="1" name="fc" type="FullyConnected" precision="FP16">
    <input><port id="0"><dim>1</dim><dim>40</dim></port></input>
    <output><port id="1" precision="FP16"><dim>1</dim><dim>10</dim></port></output>
  </layer>
</layers>
<edges><edge from-layer="0" from-port="0" to-layer="1" to-port="0"/></edges>
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
            self.assertEqual(result["inputs"][0]["name"], "input")
            self.assertEqual(result["inputs"][0]["ports"][0]["shape"], [1, 40])
            self.assertEqual(result["outputs"][0]["name"], "fc")
            self.assertEqual(result["outputs"][0]["ports"][0]["shape"], [1, 10])
            self.assertEqual(len(result["xml_sha256"]), 64)
            self.assertEqual(len(result["bin_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
