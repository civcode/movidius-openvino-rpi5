import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "speech-asr" / "python"))

from speech_asr.evaluation import (
    character_error_counts,
    edit_counts,
    latency_summary_ms,
    realtime_factor,
    score_transcript,
    timing_error_ms,
    word_error_counts,
)


class EditMetricTests(unittest.TestCase):
    def test_exact_match_and_normalization(self):
        result = score_transcript("Hello, WORLD!", "hello world")
        self.assertEqual(result["wer"], 0.0)
        self.assertEqual(result["cer"], 0.0)

    def test_substitution(self):
        counts = word_error_counts("one two three", "one four three")
        self.assertEqual(counts.substitutions, 1)
        self.assertEqual(counts.deletions, 0)
        self.assertEqual(counts.insertions, 0)
        self.assertAlmostEqual(counts.rate, 1 / 3)

    def test_deletion(self):
        counts = word_error_counts("one two three", "one three")
        self.assertEqual(counts.deletions, 1)
        self.assertAlmostEqual(counts.rate, 1 / 3)

    def test_insertion(self):
        counts = word_error_counts("one three", "one two three")
        self.assertEqual(counts.insertions, 1)
        self.assertAlmostEqual(counts.rate, 0.5)

    def test_wer_can_exceed_one(self):
        self.assertEqual(word_error_counts("a", "b c").rate, 2.0)

    def test_empty_reference_is_finite(self):
        counts = word_error_counts("", "hello world")
        self.assertEqual(counts.reference_units, 0)
        self.assertEqual(counts.rate, 2.0)

    def test_cer_excludes_spaces(self):
        counts = character_error_counts("a b", "ab")
        self.assertEqual(counts.rate, 0.0)

    def test_generic_edit_counts(self):
        counts = edit_counts(["a", "b"], ["a", "c", "d"])
        self.assertEqual(counts.errors, 2)


class TimingMetricTests(unittest.TestCase):
    def test_realtime_factor(self):
        self.assertEqual(realtime_factor(0.5, 16000), 0.5)

    def test_latency_percentiles(self):
        summary = latency_summary_ms([10, 20, 30, 40, 50])
        self.assertEqual(summary["p50_ms"], 30.0)
        self.assertEqual(summary["p95_ms"], 48.0)

    def test_timing_error(self):
        self.assertEqual(timing_error_ms(16000, 18400), 150.0)

    def test_rejects_empty_latency(self):
        with self.assertRaises(ValueError):
            latency_summary_ms([])


if __name__ == "__main__":
    unittest.main()
