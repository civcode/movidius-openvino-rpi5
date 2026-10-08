"""Dependency-free tests for ROCm accelerator probe affinity diagnostics."""

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE = ROOT / "examples" / "speech-asr" / "tools" / "probe_torch_accelerator.py"
spec = importlib.util.spec_from_file_location("probe_torch_accelerator", PROBE)
probe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(probe)


class RocmAffinityTests(unittest.TestCase):
    def test_all_threads_retain_requested_affinity(self):
        self.assertEqual(
            probe.rocm_affinity_issues(
                [0, 1, 2], [0, 1, 2],
                {"thread_affinity_mask_counts": {"0,1,2": 4}}, "1"
            ),
            [],
        )

    def test_missing_bootstrap_is_a_failure(self):
        issues = probe.rocm_affinity_issues(
            [0], [0], {"thread_affinity_mask_counts": {"0": 4}}, None
        )
        self.assertTrue(any("bootstrap" in issue for issue in issues))

    def test_main_thread_constrained_after_rocm(self):
        issues = probe.rocm_affinity_issues(
            [0, 1, 2], [0],
            {"thread_affinity_mask_counts": {"0": 4}}, "1"
        )
        self.assertTrue(any("main thread" in issue for issue in issues))
        self.assertTrue(any("process threads" in issue for issue in issues))

    def test_helper_thread_constrained_after_rocm(self):
        issues = probe.rocm_affinity_issues(
            [0, 1], [0, 1],
            {"thread_affinity_mask_counts": {"0,1": 3, "0": 1}}, "1"
        )
        self.assertTrue(any("process threads" in issue for issue in issues))

    def test_missing_thread_affinity_sample_is_not_a_pass(self):
        issues = probe.rocm_affinity_issues(
            [0, 1], [0, 1], {"thread_affinity_mask_counts": {}}, "1"
        )
        self.assertEqual(len(issues), 1)


if __name__ == "__main__":
    unittest.main()
