"""Hardware-free microphone squelch regression checks."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.quartznet_fixed512 import FULL_WINDOW_SAMPLES  # noqa: E402
from speech_asr.quartznet_fixed512_live import (  # noqa: E402
    Fixed512StreamingRecognizer,
    squelch_silent_logits,
)

BLANK = 28
VOWEL = 5  # arbitrary nonblank CTC class


def vowel_logits(size=256):
    logits = np.zeros((size, 29), dtype=np.float32)
    logits[:, VOWEL] = 3.0
    return logits


class QuartzNetMicrophoneSquelchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = SPEECH / "models/quartznet15x5_nvidia_ref"
        cls.spec = json.loads((directory / "model_spec.json").read_text())
        cls.vocab = json.loads((directory / "vocab.json").read_text())

    def apply(self, audio, *, threshold=-40.0, frames=256):
        return squelch_silent_logits(
            vowel_logits(frames), audio,
            valid_output_frames=frames,
            blank_index=BLANK,
            threshold_dbfs=threshold,
        )

    def test_silence_produces_only_ctc_blanks(self):
        output = self.apply(np.zeros((81760,), dtype=np.float32))
        self.assertTrue(np.all(np.argmax(output, axis=1) == BLANK))

    def test_quiet_noise_is_squelched(self):
        # -50 dBFS sine/noise floor should be below the -40 dBFS threshold.
        x = np.arange(81760, dtype=np.float32)
        quiet = 0.004 * np.sin(2 * np.pi * x / 123.0)
        output = self.apply(quiet)
        self.assertTrue(np.all(np.argmax(output, axis=1) == BLANK))

    def test_speech_is_preserved_and_long_silent_tail_squelched(self):
        audio = np.zeros((81760,), dtype=np.float32)
        audio[80 * 320 : 110 * 320] = 0.1
        output = self.apply(audio)
        decisions = np.argmax(output, axis=1)
        self.assertEqual(int(decisions[95]), VOWEL)
        self.assertEqual(int(decisions[20]), BLANK)
        self.assertEqual(int(decisions[180]), BLANK)

    def test_short_click_does_not_open_gate(self):
        audio = np.zeros((81760,), dtype=np.float32)
        audio[90 * 320 : 91 * 320] = 0.8
        output = self.apply(audio)
        self.assertTrue(np.all(np.argmax(output, axis=1) == BLANK))

    def test_quiet_speech_threshold_can_be_adjusted(self):
        audio = np.full((81760,), 0.015, dtype=np.float32)
        strict = self.apply(audio, threshold=-30.0)
        permissive = self.apply(audio, threshold=-45.0)
        self.assertTrue(np.all(np.argmax(strict, axis=1) == BLANK))
        self.assertTrue(np.all(np.argmax(permissive, axis=1) == VOWEL))

    def test_short_final_segment_no_shape_error(self):
        output = self.apply(
            np.zeros((100,), dtype=np.float32), frames=1
        )
        self.assertEqual(output.shape, (1, 29))
        self.assertEqual(int(np.argmax(output[0])), BLANK)

    def test_invalid_threshold_rejected(self):
        with self.assertRaisesRegex(ValueError, "between -90 and -10"):
            self.apply(np.zeros((81920,), dtype=np.float32), threshold=-100)

    def test_default_recognizer_keeps_canonical_decoding(self):
        # In production, WAV mode sets squelch_dbfs=None; verify that a
        # vowel-only model still emits its prediction, even for quiet audio.
        def fake_features(_samples, _spec):
            return np.zeros((1, 64, 512), dtype=np.float32), 511, 256

        def infer(_features):
            return vowel_logits(), 1.0

        with patch(
            "speech_asr.quartznet_fixed512_live.fixed512_features",
            side_effect=fake_features,
        ):
            raw = Fixed512StreamingRecognizer(
                spec=self.spec, vocab=self.vocab, infer=infer,
            )
            gated = Fixed512StreamingRecognizer(
                spec=self.spec, vocab=self.vocab, infer=infer,
                squelch_dbfs=-40.0,
            )
            silence = np.zeros((FULL_WINDOW_SAMPLES,), dtype=np.float32)
            first = raw.push_audio(silence)[0]
            second = gated.push_audio(silence)[0]
        self.assertTrue(first.partial_text)
        self.assertEqual(second.partial_text, "")
        self.assertEqual(second.committed_text, "")


if __name__ == "__main__":
    unittest.main()
