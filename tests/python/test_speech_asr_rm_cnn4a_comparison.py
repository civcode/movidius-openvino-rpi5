import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
TOOL = SPEECH / "evaluation" / "compare_rm_cnn4a.py"
MODEL_DIR = SPEECH / "models" / "rm_cnn4a"


def load_tool():
    spec = importlib.util.spec_from_file_location("compare_rm_cnn4a", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load compare_rm_cnn4a")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RmCnn4aComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()
        cls.reference = json.loads(
            (MODEL_DIR / "cpu-reference-v1.json").read_text(encoding="utf-8")
        )
        cls.candidate = json.loads(
            (MODEL_DIR / "myriad-amd64-v1.json").read_text(encoding="utf-8")
        )

    def test_frozen_comparison_is_reproducible(self):
        actual = self.tool.compare_results(
            self.reference,
            self.candidate,
            reference_name="cpu-reference-v1.json",
            candidate_name="myriad-amd64-v1.json",
        )
        expected = json.loads(
            (MODEL_DIR / "cpu-vs-myriad-amd64-v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(actual, expected)

    def test_rejects_artifact_mismatch(self):
        candidate = json.loads(json.dumps(self.candidate))
        candidate["model"]["xml_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "model.xml_sha256"):
            self.tool.compare_results(
                self.reference,
                candidate,
                reference_name="cpu.json",
                candidate_name="myriad.json",
            )

    def test_rejects_cpu_candidate(self):
        candidate = json.loads(json.dumps(self.candidate))
        candidate["runtime"]["backend"] = "CPU"
        candidate["runtime"]["target"] = "amd64"
        with self.assertRaisesRegex(ValueError, "candidate backend must be MYRIAD"):
            self.tool.compare_results(
                self.reference,
                candidate,
                reference_name="cpu.json",
                candidate_name="candidate.json",
            )


if __name__ == "__main__":
    unittest.main()
