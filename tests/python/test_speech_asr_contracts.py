import copy
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "speech-asr" / "python"))

from speech_asr.contracts import (
    ContractValidationError,
    canonical_json_sha256,
    validate_acoustic_benchmark_result,
    validate_acoustic_regression_result,
    validate_benchmark_contract,
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
            "path": "audio/synthetic.f32",
            "sample_rate_hz": 16000,
            "channels": 1,
            "sample_type": "float32",
            "encoding": "f32le",
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
            "id": "example-model",
            "family": "example-family",
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


def regression_document():
    return {
        "schema": "speech-asr/acoustic-regression-result",
        "version": 1,
        "status": "completed",
        "model": {
            "id": "rm_cnn4a-fp16",
            "family": "rm_cnn4a",
            "xml_sha256": SHA,
            "bin_sha256": SHA,
        },
        "fixture": {
            "features_sha256": SHA,
            "reference_scores_sha256": SHA,
        },
        "runtime": {
            "repo_commit": "7e3d0e0ce950769fa1c03f6d2bf5574de244a16c",
            "backend": "MYRIAD",
            "target": "arm64",
            "openvino_version": "2020.3.2",
        },
        "metrics": {
            "utterances": 1,
            "total_frames": 100,
            "weighted_mean_infer_ms_per_frame": 10.0,
            "utterance_avg_infer_ms_per_frame_p50": 10.0,
            "utterance_avg_infer_ms_per_frame_p95": 10.0,
            "max_error_max": 1.0,
            "avg_error_mean": 0.1,
            "rms_error_mean": 0.2,
            "failures": 0,
        },
        "provenance": {"raw_log_sha256": SHA},
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


class BenchmarkContractTests(unittest.TestCase):
    def test_rejects_implicit_measurement_policy(self):
        document = {
            "schema": "speech-asr/benchmark",
            "version": 1,
            "result_schema": "experiment-result-v1.schema.json",
            "dataset": {
                "canonical_corpus": "AMI",
                "normalized_manifest_format": "jsonl",
                "timing_unit": "sample_index",
                "splits": {
                    "smoke": "smoke.jsonl",
                    "benchmark": "benchmark.jsonl",
                    "validation": "validation.jsonl",
                },
            },
            "audio_contract": "audio-v1",
            "scoring": {
                "text_normalization_version": "text-v1",
                "text_normalization_contract": "text-v1.json",
                "metrics": ["wer", "cer"],
            },
            "hardware_metrics": [
                "realtime_factor",
                "inference_latency_p50_ms",
                "inference_latency_p95_ms",
                "model_load_success",
                "inference_failures",
            ],
            "measurement_policy": {
                "warmup_count": "runner decides",
                "measured_iterations": "runner decides",
                "latency_percentiles": [50, 95],
                "aggregation": "corpus_weighted",
            },
        }
        with self.assertRaisesRegex(ContractValidationError, "warmup_count"):
            validate_benchmark_contract(document)


    def test_rejects_changed_frozen_iteration_count(self):
        document = {
            "schema": "speech-asr/benchmark",
            "version": 1,
            "result_schema": "experiment-result-v1.schema.json",
            "dataset": {
                "canonical_corpus": "AMI",
                "normalized_manifest_format": "jsonl",
                "timing_unit": "sample_index",
                "splits": {
                    "smoke": "smoke.jsonl",
                    "benchmark": "benchmark.jsonl",
                    "validation": "validation.jsonl",
                },
            },
            "audio_contract": "audio-v1",
            "scoring": {
                "text_normalization_version": "text-v1",
                "text_normalization_contract": "text-v1.json",
                "metrics": ["wer", "cer"],
            },
            "hardware_metrics": [
                "realtime_factor",
                "inference_latency_p50_ms",
                "inference_latency_p95_ms",
                "model_load_success",
                "inference_failures",
            ],
            "measurement_policy": {
                "warmup_count": 1,
                "measured_iterations": 3,
                "latency_percentiles": [50, 95],
                "aggregation": "corpus_weighted",
            },
        }
        with self.assertRaisesRegex(ContractValidationError, "frozen value 5"):
            validate_benchmark_contract(document)


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


class AcousticRegressionContractTests(unittest.TestCase):
    def test_valid_regression_result(self):
        self.assertEqual(
            validate_acoustic_regression_result(regression_document())["status"],
            "completed",
        )

    def test_failed_regression_requires_diagnostics(self):
        document = regression_document()
        document["status"] = "failed"
        with self.assertRaisesRegex(ContractValidationError, "diagnostics.summary"):
            validate_acoustic_regression_result(document)

    def test_cpu_reference_backend_is_valid_on_amd64(self):
        document = regression_document()
        document["runtime"]["backend"] = "CPU"
        document["runtime"]["target"] = "amd64"
        self.assertEqual(
            validate_acoustic_regression_result(document)["runtime"]["backend"],
            "CPU",
        )

    def test_cpu_reference_backend_rejects_arm_target(self):
        document = regression_document()
        document["runtime"]["backend"] = "CPU"
        document["runtime"]["target"] = "arm64"
        with self.assertRaisesRegex(ContractValidationError, "requires 'amd64'"):
            validate_acoustic_regression_result(document)



def acoustic_benchmark_document():
    stat = {
        "mean": 10.0,
        "median": 10.0,
        "min": 9.9,
        "max": 10.1,
        "range": 0.2,
        "relative_range": 0.02,
        "cv_population": 0.01,
    }
    run = {
        "index": 0,
        "status": "completed",
        "result_path": "work/run.json",
        "log_path": "work/run.log",
        "returncode": 0,
        "raw_log_sha256": SHA,
    }
    return {
        "schema": "speech-asr/acoustic-benchmark-result",
        "version": 1,
        "status": "completed",
        "benchmark": {
            "id": "rm_cnn4a-vendor-regression-v1",
            "contract_sha256": SHA,
            "measurement_policy": {
                "warmup_count": 1,
                "measured_iterations": 5,
                "within_run_aggregation": "corpus_weighted",
                "across_runs_aggregation": "unweighted_run_statistics",
                "warmup_included_in_aggregate": False,
            },
        },
        "request": {"model": "rm_cnn4a", "backend": "myriad", "platform": "arm64"},
        "model": {
            "id": "rm_cnn4a-fp16",
            "family": "rm_cnn4a",
            "xml_sha256": SHA,
            "canonical_graph_sha256": SHA,
            "bin_sha256": SHA,
        },
        "fixture": {
            "features_sha256": SHA,
            "reference_scores_sha256": SHA,
        },
        "runtime": {
            "repo_commit": "d34843a2587b0bea76d7c35137b15ca80eb6f4f4",
            "backend": "MYRIAD",
            "target": "arm64",
            "openvino_version": "2020.3.2",
            "host_machine": "aarch64",
        },
        "runs": {
            "warmup": [dict(run)],
            "measured": [{**run, "index": index} for index in range(5)],
        },
        "aggregate": {
            "measured_runs": 5,
            "counts": {"utterances": 10, "total_frames": 3401, "failures": 0},
            "metrics": {
                "weighted_mean_infer_ms_per_frame": dict(stat),
                "utterance_avg_infer_ms_per_frame_p50": dict(stat),
                "utterance_avg_infer_ms_per_frame_p95": dict(stat),
                "model_load_ms": dict(stat),
                "max_error_max": dict(stat),
                "avg_error_mean": dict(stat),
                "rms_error_mean": dict(stat),
            },
        },
        "provenance": {
            "benchmark_contract_sha256": SHA,
            "runner_sha256": SHA,
            "worker_sha256": SHA,
            "python_version": "3.11.0",
        },
    }


class AcousticBenchmarkResultContractTests(unittest.TestCase):
    def test_valid_completed_worker_result(self):
        self.assertEqual(
            validate_acoustic_benchmark_result(acoustic_benchmark_document())["status"],
            "completed",
        )

    def test_completed_worker_requires_five_measured_runs(self):
        document = acoustic_benchmark_document()
        document["runs"]["measured"].pop()
        with self.assertRaisesRegex(ContractValidationError, "exactly five"):
            validate_acoustic_benchmark_result(document)

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
