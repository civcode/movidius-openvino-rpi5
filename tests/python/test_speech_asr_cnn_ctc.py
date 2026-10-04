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
    load_spec,
    load_vocab,
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

    def test_model_spec_is_json_and_opset_11(self):
        raw = json.loads(
            (SPEECH / "models" / "cnn_ctc_v1" / "model_spec.json").read_text()
        )
        self.assertEqual(raw["export"]["onnx_opset"], 11)


if __name__ == "__main__":
    unittest.main()
