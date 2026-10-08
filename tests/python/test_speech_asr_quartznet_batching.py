"""Hardware-free contract tests for exact-shape QuartzNet throughput scheduling."""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "speech-asr" / "python"))

from speech_asr.quartznet_batching import exact_shape_batches, percentile  # noqa: E402
from speech_asr.quartznet_reference_frontend import reference_feature_lengths  # noqa: E402


class QuartzNetBatchingTests(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "frontend": {"stft_center": True, "hop_samples": 160, "pad_to": 16}
        }
        self.records = [
            {"sample_count": 160 * count}
            for count in (20, 40, 21, 20, 50, 40, 20)
        ]

    def test_batch_one_preserves_manifest_order(self):
        batches = exact_shape_batches(self.records, self.spec, 1)
        self.assertEqual([indices[0] for _, indices in batches], list(range(7)))

    def test_grouping_is_exact_and_complete(self):
        for size in (1, 2, 3, 4, 8, 16):
            batches = exact_shape_batches(self.records, self.spec, size)
            visited = [index for _, indices in batches for index in indices]
            self.assertEqual(sorted(visited), list(range(len(self.records))))
            self.assertTrue(all(0 < len(indices) <= size for _, indices in batches))
            for frames, indices in batches:
                self.assertEqual(
                    {
                        reference_feature_lengths(
                            self.records[index]["sample_count"], self.spec
                        )[1] for index in indices
                    },
                    {frames},
                )

    def test_multi_sample_batches_do_not_pad_between_examples(self):
        batches = exact_shape_batches(self.records, self.spec, 8)
        self.assertEqual(sum(len(indices) for _, indices in batches), 7)
        self.assertEqual(len(batches), 3)

    def test_invalid_batch_size_rejected(self):
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                exact_shape_batches(self.records, self.spec, size)

    def test_percentiles(self):
        self.assertEqual(percentile([25.0], 0.95), 25.0)
        self.assertAlmostEqual(percentile([10.0, 20.0, 30.0], 0.95), 29.0)
        for data, fraction in (([], 0.5), ([5.0], -0.1), ([5.0], 1.1)):
            with self.assertRaises(ValueError):
                percentile(data, fraction)


if __name__ == "__main__":
    unittest.main()
