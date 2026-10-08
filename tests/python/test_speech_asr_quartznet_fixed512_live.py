import json
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.cnn_ctc import greedy_decode_logits  # noqa: E402
from speech_asr.quartznet_fixed512 import (  # noqa: E402
    DEFAULT_HOP_OUTPUT_FRAMES,
    Fixed512LogitStitcher,
    plan_fixed512_windows,
    quartznet_output_frames,
)
from speech_asr.quartznet_fixed512_live import (  # noqa: E402
    IncrementalCtcDecoder,
    OnlineFixed512LogitStitcher,
)
from speech_asr.quartznet_reference_frontend import reference_feature_lengths  # noqa: E402


class QuartzNetFixed512LiveTests(unittest.TestCase):
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
        cls.vocab = json.loads(
            (
                SPEECH
                / "models"
                / "quartznet15x5_nvidia_ref"
                / "vocab.json"
            ).read_text(encoding="utf-8")
        )

    def test_incremental_ctc_preserves_cross_commit_repeat_state(self):
        decoder = IncrementalCtcDecoder(self.vocab)
        a = self.vocab["tokens"].index("a")
        blank = self.vocab["blank_index"]
        decoder.push([a, a])
        self.assertEqual(decoder.text, "a")
        decoder.push([a, blank, a])
        self.assertEqual(decoder.text, "aa")

    def test_online_final_text_matches_offline_center_owned_stitch(self):
        sample_count = 150000
        windows = plan_fixed512_windows(sample_count, self.spec)
        valid_features, _ = reference_feature_lengths(sample_count, self.spec)
        output_frames = quartznet_output_frames(valid_features, self.spec)

        rng = np.random.default_rng(12345)
        offline = Fixed512LogitStitcher(output_frames, 29)
        online = OnlineFixed512LogitStitcher(self.vocab)

        for window in windows:
            logits = rng.normal(
                size=(window.valid_output_frames, 29),
            ).astype(np.float32)
            offline.add(window, logits)
            online.add_window(window, logits)
            live_limit = min(
                window.start_output_frame + DEFAULT_HOP_OUTPUT_FRAMES,
                window.end_output_frame,
                output_frames,
            )
            if live_limit > online.next_commit_frame:
                online.commit_before(live_limit)

        offline_logits, _ = offline.finish()
        expected = greedy_decode_logits(offline_logits, self.vocab)
        online.commit_before(output_frames)
        self.assertEqual(online.committed_text, expected)
        self.assertEqual(online.partial_text, expected)

    def test_partial_text_can_include_uncommitted_overlap(self):
        stitcher = OnlineFixed512LogitStitcher(self.vocab)
        windows = plan_fixed512_windows(100000, self.spec)
        window = windows[0]
        logits = np.zeros((window.valid_output_frames, 29), dtype=np.float32)
        logits[:, self.vocab["blank_index"]] = 1.0
        logits[200, self.vocab["tokens"].index("x")] = 5.0
        stitcher.add_window(window, logits)
        stitcher.commit_before(DEFAULT_HOP_OUTPUT_FRAMES)
        self.assertEqual(stitcher.committed_text, "")
        self.assertEqual(stitcher.partial_text, "x")


if __name__ == "__main__":
    unittest.main()
