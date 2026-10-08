"""Dependency-free tests for QuartzNet ROCm affinity/timing diagnostics."""

import os
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "speech-asr" / "python"))
from speech_asr.rocm_diagnostics import (  # noqa: E402
    inspect_rocm_affinity,
    parse_cpuset,
    summarize_model_timings,
)


class RocmDiagnosticsTests(unittest.TestCase):
    def test_cpuset_supports_ranges_and_disjoint_cpus(self):
        self.assertEqual(parse_cpuset("0-2,4,7-8"), {0, 1, 2, 4, 7, 8})

    def test_cpuset_rejects_invalid_ranges(self):
        for value in ("", "0-", "3-1", "-2", "2,,4", "a"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_cpuset(value)

    @mock.patch.dict(os.environ, {"SPEECH_ROCM_AFFINITY_RESTORED": "1"})
    @mock.patch("os.listdir", return_value=["123", "124"])
    @mock.patch("os.sched_getaffinity", return_value={0, 1, 2})
    def test_affinity_passes_when_every_task_has_expected_mask(self, affinity, listed):
        snapshot = inspect_rocm_affinity({0, 1, 2})
        self.assertEqual(snapshot["thread_count_observed"], 2)
        self.assertEqual(snapshot["thread_affinity_mask_counts"], {"0,1,2": 2})

    @mock.patch.dict(os.environ, {"SPEECH_ROCM_AFFINITY_RESTORED": "1"})
    @mock.patch("os.listdir", return_value=["123", "124"])
    def test_affinity_allows_pinned_worker_and_reports_mask(self, listed):
        def mask(tid):
            return {1} if tid == 124 else {0, 1}
        with mock.patch("os.sched_getaffinity", side_effect=mask):
            snapshot = inspect_rocm_affinity({0, 1})
        self.assertEqual(snapshot["narrow_thread_count"], 1)
        self.assertEqual(snapshot["thread_affinity_mask_counts"], {"0,1": 1, "1": 1})
        self.assertEqual(snapshot["narrow_thread_examples"], [{"tid": 124, "cpus": [1]}])

    @mock.patch.dict(os.environ, {"SPEECH_ROCM_AFFINITY_RESTORED": "1"})
    @mock.patch("os.listdir", return_value=["123", "124"])
    def test_affinity_rejects_worker_outside_requested_cpuset(self, listed):
        def mask(tid):
            return {16} if tid == 124 else {0, 1}
        with mock.patch("os.sched_getaffinity", side_effect=mask):
            with self.assertRaisesRegex(RuntimeError, "tid=124"):
                inspect_rocm_affinity({0, 1})

    @mock.patch.dict(os.environ, {"SPEECH_ROCM_AFFINITY_RESTORED": "1"})
    @mock.patch("os.sched_getaffinity", return_value={0})
    def test_affinity_detects_main_thread_drift(self, affinity):
        with self.assertRaisesRegex(RuntimeError, "main-thread affinity drift"):
            inspect_rocm_affinity({0, 1})

    @mock.patch.dict(os.environ, {"SPEECH_ROCM_AFFINITY_RESTORED": "0"})
    def test_affinity_requires_bootstrap(self):
        with self.assertRaisesRegex(RuntimeError, "bootstrap"):
            inspect_rocm_affinity({0, 1})

    def test_timing_summary_handles_single_utterance(self):
        summary = summarize_model_timings([25.0], 2.0)
        self.assertEqual(summary["model_inference_ms_p95"], 25.0)
        self.assertAlmostEqual(summary["model_inference_rtf"], 0.0125)

    def test_timing_summary_uses_interpolated_p95(self):
        summary = summarize_model_timings([10.0, 30.0, 20.0], 3.0)
        self.assertEqual(summary["model_inference_ms_p50"], 20.0)
        self.assertAlmostEqual(summary["model_inference_ms_p95"], 29.0)
        self.assertEqual(summary["utterances"], 3)

    def test_timing_summary_rejects_empty_samples(self):
        with self.assertRaises(ValueError):
            summarize_model_timings([], 2.0)


if __name__ == "__main__":
    unittest.main()
