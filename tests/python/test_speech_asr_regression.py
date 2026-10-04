import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "speech-asr" / "python"))

from speech_asr.regression import parse_speech_sample_output


LOG = """
[ INFO ] Model loading time 2873.59 ms
Utterance 0:
Frames in utterance: 250 frames
Average Infer time per frame: 11.3983 ms
max error: 27.3883
avg error: 2.8932
avg rms error: 3.71711
stdev error: 2.64166
Utterance 1:
Frames in utterance: 100 frames
Average inference time per frame: 10.0 ms
max error: 10.0
avg error: 1.0
avg rms error: 2.0
stdev error: 1.5
"""


class RegressionParserTests(unittest.TestCase):
    def test_parses_and_aggregates_vendor_style_output(self):
        result = parse_speech_sample_output(LOG)
        self.assertEqual(result["model_load_ms"], 2873.59)
        metrics = result["metrics"]
        self.assertEqual(metrics["utterances"], 2)
        self.assertEqual(metrics["total_frames"], 350)
        self.assertAlmostEqual(
            metrics["weighted_mean_infer_ms_per_frame"],
            (250 * 11.3983 + 100 * 10.0) / 350,
        )
        self.assertEqual(metrics["max_error_max"], 27.3883)
        self.assertEqual(len(metrics["per_utterance"]), 2)

    def test_rejects_incomplete_utterance(self):
        with self.assertRaises(ValueError):
            parse_speech_sample_output("Utterance 0:\nFrames in utterance: 3 frames\n")


if __name__ == "__main__":
    unittest.main()
