import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.ctc_beam import CharacterNgramLM, prefix_beam_decode


VOCAB = {
    "schema": "speech-asr/ctc-vocab",
    "version": 1,
    "blank_index": 0,
    "tokens": ["<blank>", " ", "a", "b", "c"],
}


def frame(**scores):
    values = np.full((len(VOCAB["tokens"]),), -8.0, dtype=np.float32)
    for token, value in scores.items():
        values[VOCAB["tokens"].index(token)] = float(value)
    return values


class CtcBeamTests(unittest.TestCase):
    def test_prefix_beam_decodes_simple_sequence(self):
        logits = np.stack(
            [
                frame(a=8.0),
                frame(**{"<blank>": 8.0}),
                frame(b=8.0),
            ]
        )
        decoded = prefix_beam_decode(
            logits,
            VOCAB,
            beam_width=4,
            token_top_k=3,
        )
        self.assertEqual(decoded["hypothesis"], "ab")

    def test_blank_separates_repeated_ctc_symbol(self):
        logits = np.stack(
            [
                frame(a=8.0),
                frame(**{"<blank>": 8.0}),
                frame(a=8.0),
            ]
        )
        decoded = prefix_beam_decode(
            logits,
            VOCAB,
            beam_width=4,
            token_top_k=3,
        )
        self.assertEqual(decoded["hypothesis"], "aa")

    def test_character_lm_can_resolve_ambiguous_acoustics(self):
        logits = np.stack(
            [
                frame(a=8.0),
                frame(**{"<blank>": 8.0}),
                frame(b=3.0, c=3.0),
            ]
        )
        lm = CharacterNgramLM.train(
            ["ac", "ac", "ac", "ac", "ab"],
            alphabet=VOCAB["tokens"][1:],
            order=3,
            smoothing=0.1,
        )
        without_lm = prefix_beam_decode(
            logits,
            VOCAB,
            beam_width=4,
            token_top_k=3,
        )
        with_lm = prefix_beam_decode(
            logits,
            VOCAB,
            beam_width=4,
            token_top_k=3,
            lm=lm,
            lm_weight=2.0,
        )
        self.assertEqual(without_lm["hypothesis"], "ab")
        self.assertEqual(with_lm["hypothesis"], "ac")

    def test_lm_uses_suffix_backoff(self):
        lm = CharacterNgramLM.train(
            ["ab", "ab", "ac"],
            alphabet=VOCAB["tokens"][1:],
            order=5,
            smoothing=0.1,
        )
        # This long unseen context backs off to a suffix observed in training.
        self.assertGreater(
            lm.log_prob(["c", "c", "a"], "b"),
            lm.log_prob(["c", "c", "a"], "c"),
        )

    def test_decoder_sweep_is_frozen_to_physical_v19_logits(self):
        source = (
            SPEECH / "evaluation" / "tune_cnn_ctc_v19_decoder.py"
        ).read_text(encoding="utf-8")
        self.assertIn('cache_dir = attempt / "edge" / "evaluation"', source)
        self.assertIn('"cached greedy WER', source)
        self.assertIn('"cached greedy CER', source)
        self.assertIn('DEFAULT_TRAIN = ROOT / "work"', source)
        self.assertIn('"beam-widths", default="8,16,32"', source)
        self.assertIn('"lm-weights", default="0.15,0.30,0.45,0.60"', source)
        self.assertIn('"jobs", type=int, default=16', source)
        self.assertIn('"schema": "speech-asr/ctc-decoder-sweep"', source)

    def test_decoder_entrypoint_exists(self):
        path = ROOT / "scripts" / "tune-cnn-ctc-v19-decoder.sh"
        self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
