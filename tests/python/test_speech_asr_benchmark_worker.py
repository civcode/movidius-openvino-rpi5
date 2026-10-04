import importlib.util
import json
import pathlib
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKER = ROOT / "examples" / "speech-asr" / "evaluation" / "benchmark_worker.py"


def load_worker():
    spec = importlib.util.spec_from_file_location("benchmark_worker", WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load benchmark worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_result(value: float) -> dict:
    return {
        "schema": "speech-asr/acoustic-regression-result",
        "version": 1,
        "status": "completed",
        "model": {
            "id": "rm_cnn4a-fp16",
            "family": "rm_cnn4a",
            "xml_sha256": "1" * 64,
            "canonical_graph_sha256": "2" * 64,
            "bin_sha256": "3" * 64,
        },
        "fixture": {
            "features_sha256": "4" * 64,
            "reference_scores_sha256": "5" * 64,
        },
        "runtime": {
            "repo_commit": "d34843a2587b0bea76d7c35137b15ca80eb6f4f4",
            "backend": "MYRIAD",
            "target": "arm64",
            "openvino_version": "2020.3.2",
        },
        "metrics": {
            "utterances": 10,
            "total_frames": 3401,
            "weighted_mean_infer_ms_per_frame": value,
            "utterance_avg_infer_ms_per_frame_p50": value + 0.1,
            "utterance_avg_infer_ms_per_frame_p95": value + 0.2,
            "max_error_max": 0.09,
            "avg_error_mean": 0.006,
            "rms_error_mean": 0.008,
            "failures": 0,
        },
        "provenance": {
            "raw_log_sha256": "6" * 64,
            "model_load_ms": 1900.0 + value,
        },
    }


class BenchmarkWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = load_worker()

    def test_native_target(self):
        self.assertEqual(self.worker.native_target("x86_64"), "amd64")
        self.assertEqual(self.worker.native_target("aarch64"), "arm64")
        self.assertEqual(self.worker.native_target("armv7l"), "armv7")
        self.assertIsNone(self.worker.native_target("mips64"))

    def test_frozen_policy_is_one_plus_five(self):
        policy, digest = self.worker.load_measurement_policy()
        self.assertEqual(policy["warmup_count"], 1)
        self.assertEqual(policy["measured_iterations"], 5)
        self.assertFalse(policy["warmup_included_in_aggregate"])
        self.assertEqual(len(digest), 64)

    def test_summary_stats(self):
        stats = self.worker.summary_stats([1.0, 2.0, 3.0])
        self.assertEqual(stats["mean"], 2.0)
        self.assertEqual(stats["median"], 2.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)
        self.assertEqual(stats["range"], 2.0)
        self.assertEqual(stats["relative_range"], 1.0)

    def test_aggregate_requires_five_runs(self):
        with self.assertRaisesRegex(ValueError, "exactly five"):
            self.worker.aggregate_results([run_result(10.0)] * 4)

    def test_aggregate_reports_repeatability(self):
        values = [10.0, 10.1, 9.9, 10.0, 10.05]
        result = self.worker.aggregate_results([run_result(value) for value in values])
        self.assertEqual(result["measured_runs"], 5)
        self.assertEqual(result["counts"]["failures"], 0)
        stats = result["metrics"]["weighted_mean_infer_ms_per_frame"]
        self.assertAlmostEqual(stats["min"], 9.9)
        self.assertAlmostEqual(stats["max"], 10.1)
        self.assertGreater(stats["cv_population"], 0.0)

    def test_identity_change_is_rejected(self):
        reference = run_result(10.0)
        candidate = run_result(10.0)
        candidate["model"]["bin_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "model.bin_sha256"):
            self.worker.require_same_identity(reference, candidate)

    def test_summary_is_derived_from_result(self):
        measured = [run_result(10.0)] * 5
        aggregate = self.worker.aggregate_results(measured)
        result = {
            "status": "completed",
            "benchmark": {
                "id": self.worker.BENCHMARK_ID,
                "measurement_policy": {
                    "warmup_count": 1,
                    "measured_iterations": 5,
                },
            },
            "runtime": measured[0]["runtime"],
            "aggregate": aggregate,
        }
        text = self.worker.render_summary(result)
        self.assertIn("1 warmup (excluded)", text)
        self.assertIn("5 measured", text)
        self.assertIn("weighted infer ms/frame", text)


if __name__ == "__main__":
    unittest.main()
