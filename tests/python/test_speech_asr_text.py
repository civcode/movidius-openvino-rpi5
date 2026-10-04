import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "speech-asr" / "python"))

from speech_asr.text import normalize_text_v1


class TextV1Tests(unittest.TestCase):
    def test_case_punctuation_and_unicode_apostrophe(self):
        self.assertEqual(normalize_text_v1("  I\u2019m, HERE!  "), "i'm here")

    def test_numbers_hyphens_and_leading_apostrophe(self):
        self.assertEqual(normalize_text_v1("'Cause 25-Euro"), "'cause 25 euro")

    def test_whitespace_is_collapsed(self):
        self.assertEqual(normalize_text_v1("a\t b\n c"), "a b c")


if __name__ == "__main__":
    unittest.main()
