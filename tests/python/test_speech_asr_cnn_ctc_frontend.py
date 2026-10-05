import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "NumPy is installed in apps/training environments")
class CnnCtcFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from speech_asr.cnn_ctc import load_spec
        from speech_asr.cnn_ctc_frontend import logmel_features

        cls.spec = load_spec(SPEECH / "models" / "cnn_ctc_v1" / "model_spec.json")
        cls.logmel_features = staticmethod(logmel_features)

    def test_fixed_shape_and_valid_length(self):
        samples = [0.0] * 16000
        features, valid = self.logmel_features(samples, self.spec)
        self.assertEqual(tuple(features.shape), (1, 64, 512))
        self.assertEqual(features.dtype.name, "float32")
        self.assertGreater(valid, 0)
        self.assertLess(valid, 512)

    def test_deterministic_features(self):
        samples = [0.1 if index % 2 else -0.1 for index in range(12000)]
        first, first_valid = self.logmel_features(samples, self.spec)
        second, second_valid = self.logmel_features(samples, self.spec)
        self.assertEqual(first_valid, second_valid)
        self.assertTrue((first == second).all())


@unittest.skipUnless(HAS_NUMPY, "NumPy is installed in apps/training environments")
class CnnCtcComparisonTests(unittest.TestCase):
    def test_comparison_reports_argmax_agreement(self):
        import numpy as np
        from speech_asr.cnn_ctc_compare import compare_arrays

        reference = np.zeros((1, 2, 3), dtype=np.float32)
        candidate = np.zeros((1, 2, 3), dtype=np.float32)
        reference[0, 0, 1] = 1.0
        candidate[0, 0, 1] = 0.9
        reference[0, 1, 2] = 1.0
        candidate[0, 1, 0] = 1.1
        result = compare_arrays(reference, candidate)
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["frame_argmax_matches"], 1)
        self.assertEqual(result["frame_argmax_agreement"], 0.5)
        self.assertEqual(result["frame_argmax_mismatches"], 1)
        self.assertGreater(
            result["max_mismatched_reference_top2_margin"],
            0.0,
        )
        self.assertGreater(result["max_abs_error"], 0.0)

    def test_comparison_reports_low_margin_argmax_flip(self):
        import numpy as np
        from speech_asr.cnn_ctc_compare import compare_arrays

        reference = np.array([[[1.0, 0.999, 0.0]]], dtype=np.float32)
        candidate = np.array([[[0.999, 1.0, 0.0]]], dtype=np.float32)
        result = compare_arrays(reference, candidate)
        self.assertEqual(result["frame_argmax_mismatches"], 1)
        self.assertAlmostEqual(
            result["max_mismatched_reference_top2_margin"],
            0.001,
            places=5,
        )
        self.assertAlmostEqual(result["max_abs_error"], 0.001, places=5)


if __name__ == "__main__":
    unittest.main()
