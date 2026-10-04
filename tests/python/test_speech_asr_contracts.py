import copy
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "speech-asr" / "python"))

from speech_asr.contracts import (
    ContractValidationError,
    canonical_json_sha256,
    validate_experiment_result,
    validate_speech_sample,
)


SHA = "a" * 64


def sample_document():
    return {
        "schema": "speech-asr/sample",
        "version": 1,
        "id": "synthetic-001",
        "audio": {
            "path": "audio/synthetic.wav",
            "sample_rate_hz": 16000,
            "channels": 1,
            "sample_type": "float32",
            "start_sample": 160,
            "end_sample": 1760,
        },
        "transcript": {
            "text": "hello world",
            "normalization": "text-v1",
            "words": [
                {"text": "hello", "start_sample": 160, "end_sample": 800},
                {"text": "world", "start_sample": 960, "end_sample": 1760},
            ],
        },
        "metadata": {"source": "synthetic"},
    }


def result_document():
    return {
        "schema": "speech-asr/experiment-result",
        "version": 1,
        "experiment_id": "asr-0001",
        "status": "completed",
        "benchmark": {
            "id": "ami-smoke-v1",
            "contract_version": 1,
            "manifest_sha256": SHA,
            "text_normalization_version": "text-v1",
        },
        "model": {
            "id": "rm_cnn4a",
            "family": "rm_cnn4a",
            "spec_sha256": SHA,
            "artifact_sha256": {"model.xml": SHA, "model.bin": SHA},
        },
        "runtime": {
            "repo_commit": "7e3d0e0ce950769fa1c03f6d2bf5574de244a16c",
            "backend": "MYRIAD",
            "openvino_version": "2020.3.2",
            "target": "arm64",
            "firmware_sha256": SHA,
        },
        "metrics": {
            "wer": 0.25,
            "cer": 0.10,
            "realtime_factor": 0.6,
            "inference_latency_p50_ms": 12.0,
            "inference_latency_p95_ms": 14.0,
            "failures": 0,
        },
        "provenance": {
            "dataset_manifest_sha256": SHA,
            "model_spec_sha256": SHA,
        },
    }


class SpeechSampleContractTests(unittest.TestCase):
    def test_valid_sample(self):
        self.assertEqual(validate_speech_sample(sample_document())["id"], "synthetic-001")

    def test_rejects_noncanonical_audio(self):
        document = sample_document()
        document["audio"]["sample_rate_hz"] = 48000
        with self.assertRaisesRegex(ContractValidationError, "sample_rate_hz"):
            validate_speech_sample(document)

    def test_rejects_word_outside_sample(self):
        document = sample_document()
        document["transcript"]["words"][1]["end_sample"] = 2000
        with self.assertRaisesRegex(ContractValidationError, "exceeds sample audio range"):
            validate_speech_sample(document)

    def test_rejects_unordered_words(self):
        document = sample_document()
        document["transcript"]["words"][1]["start_sample"] = 100
        with self.assertRaisesRegex(ContractValidationError, "ordered by start_sample"):
            validate_speech_sample(document)


class ExperimentResultContractTests(unittest.TestCase):
    def test_valid_completed_result(self):
        self.assertEqual(validate_experiment_result(result_document())["status"], "completed")

    def test_completed_result_requires_metrics(self):
        document = result_document()
        del document["metrics"]
        with self.assertRaisesRegex(ContractValidationError, "metrics"):
            validate_experiment_result(document)

    def test_failed_result_requires_diagnostics(self):
        document = result_document()
        document["status"] = "failed"
        document.pop("metrics")
        with self.assertRaisesRegex(ContractValidationError, "diagnostics.summary"):
            validate_experiment_result(document)

    def test_rejects_unknown_runtime_version(self):
        document = result_document()
        document["runtime"]["openvino_version"] = "2024.0"
        with self.assertRaisesRegex(ContractValidationError, "2020.3.2"):
            validate_experiment_result(document)


class CanonicalHashTests(unittest.TestCase):
    def test_hash_is_key_order_independent(self):
        a = {"b": 2, "a": [1, 3]}
        b = {"a": [1, 3], "b": 2}
        self.assertEqual(canonical_json_sha256(a), canonical_json_sha256(b))

    def test_hash_changes_with_content(self):
        document = sample_document()
        original = canonical_json_sha256(document)
        changed = copy.deepcopy(document)
        changed["transcript"]["text"] = "hello there"
        self.assertNotEqual(original, canonical_json_sha256(changed))


if __name__ == "__main__":
    unittest.main()
