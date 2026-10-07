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
        cls.v4_spec = load_spec(
            SPEECH / "models" / "cnn_ctc_v4" / "model_spec.json"
        )
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


    def test_v4_valid_frame_cmvn_zeroes_padding(self):
        import numpy as np

        samples = [
            0.1 if index % 3 else -0.2
            for index in range(16000)
        ]
        features, valid = self.logmel_features(samples, self.v4_spec)
        self.assertGreater(valid, 1)
        self.assertLess(valid, 512)
        valid_values = features[0, :, :valid]
        padded_values = features[0, :, valid:]
        self.assertTrue(
            np.allclose(
                valid_values.mean(axis=1),
                0.0,
                atol=2e-5,
            )
        )
        self.assertTrue((padded_values == 0.0).all())

    def test_v1_and_v4_differ_only_by_normalization_semantics(self):
        samples = [
            0.15 if index % 5 else -0.1
            for index in range(14000)
        ]
        v1, v1_valid = self.logmel_features(samples, self.spec)
        v4, v4_valid = self.logmel_features(samples, self.v4_spec)
        self.assertEqual(v1_valid, v4_valid)
        self.assertFalse((v1 == v4).all())
        self.assertTrue((v4[0, :, v4_valid:] == 0.0).all())


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
        self.assertGreaterEqual(result["max_softmax_abs_error"], 0.0)
        self.assertGreaterEqual(result["mean_frame_total_variation"], 0.0)
        self.assertGreaterEqual(result["max_frame_total_variation"], 0.0)
        self.assertGreater(result["max_abs_reference"], 0.0)
        self.assertGreaterEqual(
            result["max_abs_error_over_reference_max_abs"],
            0.0,
        )

    def test_comparison_probability_drift_is_zero_for_identical_logits(self):
        import numpy as np
        from speech_asr.cnn_ctc_compare import compare_arrays

        logits = np.array(
            [[[2.0, 1.0, -1.0], [0.1, 0.2, 0.3]]],
            dtype=np.float32,
        )
        result = compare_arrays(logits, logits.copy())
        self.assertEqual(result["frame_argmax_agreement"], 1.0)
        self.assertEqual(result["max_abs_error"], 0.0)
        self.assertEqual(result["max_softmax_abs_error"], 0.0)
        self.assertEqual(result["mean_frame_total_variation"], 0.0)
        self.assertEqual(result["max_frame_total_variation"], 0.0)

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
