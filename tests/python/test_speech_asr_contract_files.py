import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.contracts import validate_experiment_result, validate_speech_sample


class ContractFileTests(unittest.TestCase):
    def test_json_contract_files_are_parseable(self):
        for name in (
            "speech-sample-v1.schema.json",
            "experiment-result-v1.schema.json",
            "text-v1.json",
        ):
            with self.subTest(name=name):
                with (SPEECH / "contracts" / name).open("r", encoding="utf-8") as handle:
                    self.assertIsInstance(json.load(handle), dict)

    def test_example_documents_validate(self):
        examples = SPEECH / "contracts" / "examples"
        with (examples / "sample-v1.json").open("r", encoding="utf-8") as handle:
            self.assertEqual(validate_speech_sample(json.load(handle))["version"], 1)
        with (examples / "result-v1.json").open("r", encoding="utf-8") as handle:
            self.assertEqual(validate_experiment_result(json.load(handle))["status"], "completed")

    def test_validator_cli_accepts_example(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(SPEECH / "tools" / "validate_contract.py"),
                str(SPEECH / "contracts" / "examples" / "sample-v1.json"),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("valid sample sha256=", proc.stdout)


if __name__ == "__main__":
    unittest.main()
