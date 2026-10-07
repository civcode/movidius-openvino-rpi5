import json
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.quartznet_fixed512 import (  # noqa: E402
    DEFAULT_HOP_OUTPUT_FRAMES,
    FIXED_TENSOR_FRAMES,
    FULL_OUTPUT_FRAMES,
    FULL_VALID_FEATURE_FRAMES,
    FULL_WINDOW_SAMPLES,
    Fixed512LogitStitcher,
    fixed512_policy_dict,
    plan_fixed512_windows,
    quartznet_output_frames,
    validate_fixed512_geometry,
)
from speech_asr.quartznet_reference_frontend import reference_feature_lengths  # noqa: E402


class QuartzNetFixed512Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(
            (
                SPEECH
                / "models"
                / "quartznet15x5_nvidia_ref"
                / "model_spec.json"
            ).read_text(encoding="utf-8")
        )

    def test_full_window_geometry_is_exactly_512(self):
        validate_fixed512_geometry(self.spec)
        valid, tensor = reference_feature_lengths(FULL_WINDOW_SAMPLES, self.spec)
        self.assertEqual(valid, FULL_VALID_FEATURE_FRAMES)
        self.assertEqual(tensor, FIXED_TENSOR_FRAMES)
        self.assertEqual(
            quartznet_output_frames(valid, self.spec),
            FULL_OUTPUT_FRAMES,
        )

    def test_default_policy_is_half_output_overlap(self):
        policy = fixed512_policy_dict()
        self.assertEqual(policy["hop_output_frames"], DEFAULT_HOP_OUTPUT_FRAMES)
        self.assertEqual(policy["hop_samples"], 40960)
        self.assertEqual(policy["full_window_samples"], 81760)
        self.assertEqual(policy["overlap_samples"], 40800)

    def test_planner_covers_long_utterance_without_output_gaps(self):
        sample_count = 200000
        windows = plan_fixed512_windows(sample_count, self.spec)
        valid, _ = reference_feature_lengths(sample_count, self.spec)
        total_output = quartznet_output_frames(valid, self.spec)
        coverage = [0] * total_output
        for window in windows:
            for index in range(window.start_output_frame, window.end_output_frame):
                coverage[index] += 1
        self.assertTrue(all(value >= 1 for value in coverage))
        self.assertEqual(windows[0].start_output_frame, 0)
        self.assertLessEqual(windows[-1].end_output_frame, total_output)

    def test_stitcher_prefers_window_center(self):
        sample_count = 150000
        windows = plan_fixed512_windows(sample_count, self.spec)
        self.assertGreaterEqual(len(windows), 2)
        valid, _ = reference_feature_lengths(sample_count, self.spec)
        total_output = quartznet_output_frames(valid, self.spec)
        stitcher = Fixed512LogitStitcher(total_output, 2)

        for window in windows:
            logits = np.zeros((window.valid_output_frames, 2), dtype=np.float32)
            logits[:, 1] = float(window.index + 1)
            stitcher.add(window, logits)

        logits, owners = stitcher.finish()
        self.assertEqual(logits.shape, (total_output, 2))
        self.assertTrue(np.all(owners >= 0))
        self.assertEqual(int(owners[0]), 0)
        overlap_global = windows[1].start_output_frame + 100
        self.assertEqual(int(owners[overlap_global]), 1)


if __name__ == "__main__":
    unittest.main()
