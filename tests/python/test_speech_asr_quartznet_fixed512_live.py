import json
import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.cnn_ctc import greedy_decode_logits  # noqa: E402
from speech_asr.quartznet_fixed512 import (  # noqa: E402
    DEFAULT_HOP_OUTPUT_FRAMES,
    FULL_WINDOW_SAMPLES,
    Fixed512LogitStitcher,
    plan_fixed512_windows,
    quartznet_output_frames,
)
from speech_asr.quartznet_fixed512_live import (  # noqa: E402
    Fixed512StreamingRecognizer,
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

    def test_pause_spacing_is_display_only(self):
        a = self.vocab["tokens"].index("a")
        b = self.vocab["tokens"].index("b")
        blank = self.vocab["blank_index"]
        decoder = IncrementalCtcDecoder(self.vocab, pause_space_frames=30)
        decoder.push([a] + [blank] * 30 + [b])
        self.assertEqual(decoder.text, "ab")
        self.assertEqual(decoder.display_text, "a b")
        self.assertEqual(decoder.preview([], display=True), "a b")

    def test_pause_spacing_survives_commit_and_preview_boundaries(self):
        a = self.vocab["tokens"].index("a")
        b = self.vocab["tokens"].index("b")
        blank = self.vocab["blank_index"]
        decoder = IncrementalCtcDecoder(self.vocab, pause_space_frames=30)
        decoder.push([a] + [blank] * 10)
        self.assertEqual(decoder.preview([blank] * 20 + [b]), "ab")
        self.assertEqual(
            decoder.preview([blank] * 20 + [b], display=True), "a b"
        )
        self.assertEqual(decoder.display_text, "a")
        decoder.push([blank] * 20)
        self.assertEqual(decoder.display_text, "a")
        decoder.push([b])
        self.assertEqual(decoder.display_text, "a b")
        self.assertEqual(decoder.text, "ab")

    def test_pause_spacing_does_not_add_duplicate_or_trailing_spaces(self):
        a = self.vocab["tokens"].index("a")
        b = self.vocab["tokens"].index("b")
        space = self.vocab["tokens"].index(" ")
        blank = self.vocab["blank_index"]
        decoder = IncrementalCtcDecoder(self.vocab, pause_space_frames=20)
        decoder.push([blank] * 30 + [a] + [blank] * 20 + [space])
        decoder.push([blank] * 20 + [b] + [blank] * 30)
        self.assertEqual(decoder.display_text, "a b")
        self.assertEqual(decoder.text, "a b")

    def test_short_ctc_blank_gap_does_not_create_space(self):
        a = self.vocab["tokens"].index("a")
        b = self.vocab["tokens"].index("b")
        blank = self.vocab["blank_index"]
        decoder = IncrementalCtcDecoder(self.vocab, pause_space_frames=30)
        decoder.push([a] + [blank] * 29 + [b])
        self.assertEqual(decoder.display_text, "ab")

    def test_pause_spacing_disabled_preserves_previous_behavior(self):
        a = self.vocab["tokens"].index("a")
        b = self.vocab["tokens"].index("b")
        blank = self.vocab["blank_index"]
        decoder = IncrementalCtcDecoder(self.vocab)
        decoder.push([a] + [blank] * 60 + [b])
        self.assertEqual(decoder.text, "ab")
        self.assertEqual(decoder.display_text, "ab")

    def test_pause_spacing_rejects_negative_threshold(self):
        with self.assertRaisesRegex(ValueError, "pause_space_frames"):
            IncrementalCtcDecoder(self.vocab, pause_space_frames=-1)

    def test_pause_delimiter_preserves_custom_characters_and_strings(self):
        a = self.vocab["tokens"].index("a")
        b = self.vocab["tokens"].index("b")
        blank = self.vocab["blank_index"]
        for delimiter in ("$", "#", "[ref_delimiter]"):
            with self.subTest(delimiter=delimiter):
                decoder = IncrementalCtcDecoder(
                    self.vocab,
                    pause_space_frames=30,
                    pause_delimiter=delimiter,
                )
                self.assertEqual(decoder.push([a] + [blank] * 30), "a")
                self.assertEqual(decoder.display_text, "a")
                self.assertEqual(
                    decoder.preview([b], display=True),
                    "a" + delimiter + "b",
                )
                # Preview is tentative; no marker should be committed yet.
                self.assertEqual(decoder.display_text, "a")
                self.assertEqual(decoder.push([b]), "ab")
                self.assertEqual(decoder.display_text, "a" + delimiter + "b")
                self.assertEqual(decoder.text, "ab")

    def test_pause_delimiter_replaces_predicted_space_at_long_pause(self):
        a = self.vocab["tokens"].index("a")
        b = self.vocab["tokens"].index("b")
        space = self.vocab["tokens"].index(" ")
        blank = self.vocab["blank_index"]
        decoder = IncrementalCtcDecoder(
            self.vocab,
            pause_space_frames=30,
            pause_delimiter="[ref_delimiter]",
        )
        decoder.push([a] + [blank] * 30 + [space, b])
        self.assertEqual(decoder.text, "a b")
        self.assertEqual(decoder.display_text, "a[ref_delimiter]b")

    def test_pause_delimiter_after_native_space_and_blank_run(self):
        a = self.vocab["tokens"].index("a")
        b = self.vocab["tokens"].index("b")
        space = self.vocab["tokens"].index(" ")
        blank = self.vocab["blank_index"]
        decoder = IncrementalCtcDecoder(
            self.vocab,
            pause_space_frames=30,
            pause_delimiter="$",
        )
        decoder.push([a, space] + [blank] * 30 + [b])
        self.assertEqual(decoder.display_text, "a$b")
        self.assertEqual(decoder.text, "a b")

    def test_pause_delimiter_does_not_change_short_utterances(self):
        a = self.vocab["tokens"].index("a")
        b = self.vocab["tokens"].index("b")
        blank = self.vocab["blank_index"]
        decoder = IncrementalCtcDecoder(
            self.vocab, pause_space_frames=30,
            pause_delimiter="[ref_delimiter]",
        )
        decoder.push([a] + [blank] * 29 + [b])
        self.assertEqual(decoder.text, "ab")
        self.assertEqual(decoder.display_text, "ab")

    def test_pause_delimiter_is_ignored_if_pause_spacing_disabled(self):
        a = self.vocab["tokens"].index("a")
        b = self.vocab["tokens"].index("b")
        blank = self.vocab["blank_index"]
        decoder = IncrementalCtcDecoder(
            self.vocab, pause_space_frames=0, pause_delimiter="[ref_delimiter]"
        )
        decoder.push([a] + [blank] * 40 + [b])
        self.assertEqual(decoder.display_text, "ab")

    def test_pause_delimiter_rejects_empty_or_nonprintable_string(self):
        for invalid in ("", "hello\nworld"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "printable string"):
                    IncrementalCtcDecoder(
                        self.vocab, pause_space_frames=30,
                        pause_delimiter=invalid,
                    )

    def test_first_live_window_commits_maximal_safe_prefix(self):
        blank = self.vocab["blank_index"]
        letter = self.vocab["tokens"].index("a")
        logits = np.zeros((256, 29), dtype=np.float32)
        logits[:, blank] = 2.0
        logits[150, letter] = 4.0

        def infer(_features):
            return logits, 1.0

        with patch(
            "speech_asr.quartznet_fixed512_live.fixed512_features",
            return_value=(np.zeros((1, 64, 512), dtype=np.float32), 511, 256),
        ):
            for hop, safe_frames in ((64, 160), (128, 192)):
                with self.subTest(hop=hop):
                    recognizer = Fixed512StreamingRecognizer(
                        spec=self.spec, vocab=self.vocab, infer=infer,
                        hop_output_frames=hop,
                    )
                    first = recognizer.push_audio(
                        np.zeros((FULL_WINDOW_SAMPLES,), dtype=np.float32)
                    )[0]
                    self.assertEqual(first.committed_output_frames, safe_frames)
                    self.assertEqual(first.committed_text, "a")

    def test_early_commits_match_offline_center_owned_for_64_and_128(self):
        sample_count = 170000
        valid_features, _ = reference_feature_lengths(sample_count, self.spec)
        output_frames = quartznet_output_frames(valid_features, self.spec)
        rng = np.random.default_rng(2026)

        for hop in (64, 128):
            with self.subTest(hop=hop):
                windows = plan_fixed512_windows(
                    sample_count, self.spec, hop_output_frames=hop
                )
                offline = Fixed512LogitStitcher(output_frames, 29)
                online = OnlineFixed512LogitStitcher(self.vocab)

                for window in windows:
                    logits = rng.normal(
                        size=(window.valid_output_frames, 29),
                    ).astype(np.float32)
                    offline.add(window, logits)
                    online.add_window(window, logits)
                    if window.sample_count == FULL_WINDOW_SAMPLES:
                        # Mirror safe boundary for fully captured live windows.
                        limit = min(
                            window.start_output_frame + (256 + hop) // 2,
                            window.end_output_frame,
                        )
                        if limit > online.next_commit_frame:
                            online.commit_before(limit)

                offline_logits, _ = offline.finish()
                expected = greedy_decode_logits(offline_logits, self.vocab)
                online.commit_before(output_frames)
                self.assertEqual(online.committed_text, expected)
                self.assertEqual(online.partial_text, expected)

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
