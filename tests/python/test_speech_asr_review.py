import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "examples" / "speech-asr" / "agent" / "review_experiment.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("review_experiment", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load review_experiment")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReviewSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()

    def test_metric_delta_reports_candidate_parent_ratio(self):
        candidate = {
            "metrics": {
                "wer": 0.8,
                "cer": 0.4,
                "realtime_factor": 0.02,
                "inference_latency_p50_ms": 12.0,
                "inference_latency_p95_ms": 20.0,
            }
        }
        parent = {
            "metrics": {
                "wer": 1.0,
                "cer": 0.5,
                "realtime_factor": 0.01,
                "inference_latency_p50_ms": 4.0,
                "inference_latency_p95_ms": 5.0,
            }
        }
        result = self.tool.delta(candidate, parent)
        self.assertAlmostEqual(result["wer"]["delta"], -0.2)
        self.assertAlmostEqual(result["wer"]["ratio"], 0.8)
        self.assertAlmostEqual(result["inference_latency_p95_ms"]["ratio"], 4.0)

    def test_metric_delta_does_not_compare_accuracy_across_benchmarks(self):
        candidate = {
            "benchmark": {"manifest_sha256": "a" * 64},
            "metrics": {
                "wer": 0.5,
                "cer": 0.2,
                "realtime_factor": 0.02,
                "inference_latency_p50_ms": 12.0,
                "inference_latency_p95_ms": 20.0,
            },
        }
        parent = {
            "benchmark": {"manifest_sha256": "b" * 64},
            "metrics": {
                "wer": 1.0,
                "cer": 0.5,
                "realtime_factor": 0.01,
                "inference_latency_p50_ms": 10.0,
                "inference_latency_p95_ms": 18.0,
            },
        }
        result = self.tool.delta(candidate, parent)
        self.assertFalse(result["same_benchmark_manifest"])
        self.assertEqual(
            result["not_compared"],
            ["wer", "cer", "realtime_factor"],
        )
        self.assertNotIn("wer", result)
        self.assertNotIn("cer", result)
        self.assertNotIn("realtime_factor", result)
        self.assertAlmostEqual(
            result["inference_latency_p95_ms"]["delta"],
            2.0,
        )

    def test_latest_attempt_uses_highest_number(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            attempts = root / "attempts"
            (attempts / "attempt-0001").mkdir(parents=True)
            (attempts / "attempt-0003").mkdir()
            (attempts / "attempt-0002").mkdir()
            self.assertEqual(
                self.tool.latest_attempt(root).name,
                "attempt-0003",
            )

    def test_missing_parent_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            exp = root / "exp-0123456789abcdef"
            request = exp / "request"
            request.mkdir(parents=True)
            (request / "experiment.json").write_text(
                '{"parent_experiment_id":"exp-fedcba9876543210"}\n',
                encoding="utf-8",
            )
            result = self.tool.parent_summary(exp)
            self.assertEqual(result["status"], "missing-local-parent")


class ReviewSourceTests(unittest.TestCase):
    def test_controller_surfaces_acceptance_reasons(self):
        source = (
            ROOT / "examples" / "speech-asr" / "agent" / "run_experiment.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"acceptance_reasons": acceptance_result["reasons"]', source)
        self.assertIn('"threshold_checks": acceptance_result["threshold_checks"]', source)


if __name__ == "__main__":
    unittest.main()
