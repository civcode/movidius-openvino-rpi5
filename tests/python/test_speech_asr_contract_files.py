import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.contracts import (
    validate_audio_contract,
    validate_benchmark_contract,
    validate_experiment_result,
    validate_acoustic_regression_result,
    validate_model_contract,
    validate_speech_sample,
    validate_text_contract,
)


class ContractFileTests(unittest.TestCase):
    def test_contract_documents_are_json_parseable_yaml_compatible(self):
        for name in (
            "audio-v1.yaml",
            "benchmark-v1.yaml",
            "model-v1.yaml",
            "speech-sample-v1.schema.json",
            "experiment-result-v1.schema.json",
            "acoustic-regression-result-v1.schema.json",
            "text-v1.json",
        ):
            with self.subTest(name=name):
                with (SPEECH / "contracts" / name).open("r", encoding="utf-8") as handle:
                    self.assertIsInstance(json.load(handle), dict)

    def test_root_contracts_validate(self):
        validators = {
            "audio-v1.yaml": validate_audio_contract,
            "benchmark-v1.yaml": validate_benchmark_contract,
            "model-v1.yaml": validate_model_contract,
            "text-v1.json": validate_text_contract,
        }
        for name, validator in validators.items():
            with self.subTest(name=name):
                with (SPEECH / "contracts" / name).open("r", encoding="utf-8") as handle:
                    self.assertEqual(validator(json.load(handle))["version"], 1)

    def test_example_documents_validate(self):
        examples = SPEECH / "contracts" / "examples"
        with (examples / "sample-v1.json").open("r", encoding="utf-8") as handle:
            self.assertEqual(validate_speech_sample(json.load(handle))["version"], 1)
        with (examples / "result-v1.json").open("r", encoding="utf-8") as handle:
            self.assertEqual(validate_experiment_result(json.load(handle))["status"], "completed")
        with (
            SPEECH / "models" / "rm_cnn4a" / "cpu-reference-v1.json"
        ).open("r", encoding="utf-8") as handle:
            reference = validate_acoustic_regression_result(json.load(handle))
        self.assertEqual(reference["status"], "completed")
        self.assertEqual(reference["runtime"]["backend"], "CPU")
        self.assertEqual(reference["metrics"]["failures"], 0)
        self.assertEqual(reference["metrics"]["total_frames"], 3401)

        with (
            SPEECH / "models" / "rm_cnn4a" / "myriad-amd64-v1.json"
        ).open("r", encoding="utf-8") as handle:
            device = validate_acoustic_regression_result(json.load(handle))
        self.assertEqual(device["status"], "completed")
        self.assertEqual(device["runtime"]["backend"], "MYRIAD")
        self.assertEqual(device["runtime"]["target"], "amd64")
        self.assertEqual(device["metrics"]["failures"], 0)
        self.assertEqual(device["metrics"]["total_frames"], 3401)

        with (
            SPEECH / "models" / "rm_cnn4a" / "myriad-arm64-pi5-v1.json"
        ).open("r", encoding="utf-8") as handle:
            pi_device = validate_acoustic_regression_result(json.load(handle))
        self.assertEqual(pi_device["status"], "completed")
        self.assertEqual(pi_device["runtime"]["backend"], "MYRIAD")
        self.assertEqual(pi_device["runtime"]["target"], "arm64")
        self.assertEqual(pi_device["metrics"]["failures"], 0)
        self.assertEqual(pi_device["metrics"]["total_frames"], 3401)

    def test_validator_cli_accepts_all_root_contracts(self):
        for name in ("audio-v1.yaml", "benchmark-v1.yaml", "model-v1.yaml", "text-v1.json"):
            with self.subTest(name=name):
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(SPEECH / "tools" / "validate_contract.py"),
                        str(SPEECH / "contracts" / name),
                    ],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("sha256=", proc.stdout)


if __name__ == "__main__":
    unittest.main()
