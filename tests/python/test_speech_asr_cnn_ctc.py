import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.cnn_ctc import (
    acoustic_output_length,
    encode_text,
    greedy_decode,
    greedy_decode_logits,
    greedy_decode_logits_diagnostics,
    load_spec,
    load_vocab,
    manifest_record_eligibility,
)


class CnnCtcContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(SPEECH / "models" / "cnn_ctc_v1" / "model_spec.json")
        cls.vocab = load_vocab(SPEECH / "models" / "cnn_ctc_v1" / "vocab.json")

    def test_declared_shapes_and_output_length(self):
        self.assertEqual(self.spec["input_contract"]["shape"], [1, 64, 512])
        self.assertEqual(self.spec["output_contract"]["shape"], [1, 128, 39])
        self.assertEqual(acoustic_output_length(512, self.spec), 128)

    def test_vocab_covers_text_v1_character_set(self):
        encoded = encode_text("hello 123 it's", self.vocab)
        self.assertTrue(encoded)
        self.assertNotIn(0, encoded)

    def test_greedy_ctc_collapses_repeats_and_blanks(self):
        index = {token: i for i, token in enumerate(self.vocab["tokens"])}
        values = [
            index["h"], index["h"], 0,
            index["i"], index["i"], 0,
        ]
        self.assertEqual(greedy_decode(values, self.vocab), "hi")

    def test_greedy_logits(self):
        width = len(self.vocab["tokens"])
        index = {token: i for i, token in enumerate(self.vocab["tokens"])}
        frames = []
        for token in ("a", "a", "<blank>", "b"):
            row = [-10.0] * width
            row[index[token]] = 10.0
            frames.append(row)
        self.assertEqual(greedy_decode_logits(frames, self.vocab), "ab")

    def test_greedy_diagnostics_expose_blank_collapse(self):
        width = len(self.vocab["tokens"])
        index = {token: i for i, token in enumerate(self.vocab["tokens"])}
        frames = []
        for token in ("<blank>", "<blank>", "a", "a", "<blank>"):
            row = [-10.0] * width
            row[index[token]] = 10.0
            frames.append(row)
        result = greedy_decode_logits_diagnostics(frames, self.vocab)
        self.assertEqual(result["hypothesis"], "a")
        self.assertEqual(result["frame_count"], 5)
        self.assertEqual(result["blank_argmax_frames"], 3)
        self.assertEqual(result["nonblank_argmax_frames"], 2)
        self.assertAlmostEqual(result["blank_frame_fraction"], 0.6)
        self.assertEqual(result["collapsed_token_count"], 1)
        self.assertEqual(result["emitted_characters_no_spaces"], 1)

    def test_manifest_eligibility_rejects_fixed_shape_overflow(self):
        record = {
            "audio": {"start_sample": 0, "end_sample": 82161},
            "transcript": {"text": "hi"},
        }
        decision = manifest_record_eligibility(record, self.spec, self.vocab)
        self.assertFalse(decision["eligible"])
        self.assertEqual(decision["reason"], "too_long")

    def test_manifest_eligibility_rejects_target_longer_than_ctc_time(self):
        record = {
            "audio": {"start_sample": 0, "end_sample": 400},
            "transcript": {"text": "ab"},
        }
        decision = manifest_record_eligibility(record, self.spec, self.vocab)
        self.assertFalse(decision["eligible"])
        self.assertEqual(decision["reason"], "target_too_long")

    def test_manifest_eligibility_accepts_normal_record(self):
        record = {
            "audio": {"start_sample": 0, "end_sample": 16000},
            "transcript": {"text": "hello"},
        }
        decision = manifest_record_eligibility(record, self.spec, self.vocab)
        self.assertTrue(decision["eligible"])
        self.assertIsNone(decision["reason"])
        self.assertGreater(decision["valid_output_frames"], 0)

    def test_model_spec_is_json_and_opset_11(self):
        raw = json.loads(
            (SPEECH / "models" / "cnn_ctc_v1" / "model_spec.json").read_text()
        )
        self.assertEqual(raw["export"]["onnx_opset"], 11)


if __name__ == "__main__":
    unittest.main()
