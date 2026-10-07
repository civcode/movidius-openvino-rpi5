import ast
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.ctc_beam import (
    CharacterNgramLM,
    build_decoder_artifact,
    decode_with_artifact,
    validate_decoder_artifact,
    prefix_beam_decode,
)


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

    def test_lm_artifact_round_trip_and_runtime_decode(self):
        lm = CharacterNgramLM.train(
            ["ac", "ac", "ab"],
            alphabet=VOCAB["tokens"][1:],
            order=5,
            smoothing=0.1,
        )
        artifact = build_decoder_artifact(
            acoustic_model="cnn_ctc_v19",
            vocab=VOCAB,
            config={
                "kind": "prefix-beam",
                "beam_width": 8,
                "token_top_k": 12,
                "lm_weight": 0.3,
                "word_bonus": -0.2,
            },
            lm=lm,
            provenance={"test": True},
        )
        validated = validate_decoder_artifact(artifact, vocab=VOCAB)
        self.assertEqual(validated["decoder"]["beam_width"], 8)
        restored = CharacterNgramLM.from_dict(validated["lm"])
        self.assertEqual(restored.to_dict(), lm.to_dict())

        logits = np.stack(
            [
                frame(a=8.0),
                frame(**{"<blank>": 8.0}),
                frame(b=3.0, c=3.0),
            ]
        )
        decoded = decode_with_artifact(logits, VOCAB, artifact)
        self.assertEqual(decoded["kind"], "ctc-prefix-beam-char-ngram-v1")
        self.assertEqual(decoded["beam_width"], 8)

    def test_decoder_sweep_is_frozen_to_physical_v19_logits(self):
        source = (
            SPEECH / "evaluation" / "tune_cnn_ctc_v19_decoder.py"
        ).read_text(encoding="utf-8")
        self.assertIn('cache_dir = attempt / "edge" / "evaluation"', source)
        self.assertIn('"cached greedy WER', source)
        self.assertIn('"cached greedy CER', source)
        self.assertIn('DEFAULT_TRAIN = ROOT / "work"', source)
        tree = ast.parse(source)
        defaults = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_argument" or not node.args:
                continue
            name = node.args[0]
            if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
                continue
            for keyword in node.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    defaults[name.value] = keyword.value.value
        self.assertEqual(defaults["--beam-widths"], "8,16,32")
        self.assertEqual(defaults["--lm-weights"], "0.15,0.30,0.45,0.60")
        self.assertEqual(defaults["--jobs"], 16)
        self.assertIn('"schema": "speech-asr/ctc-decoder-sweep"', source)

    def test_runtime_decoder_wiring_exists(self):
        freeze = (
            SPEECH / "tools" / "freeze_cnn_ctc_v19_decoder.py"
        ).read_text(encoding="utf-8")
        evaluator = (
            SPEECH / "evaluation" / "evaluate_cnn_ctc_v19.py"
        ).read_text(encoding="utf-8")
        runtime = (
            SPEECH / "runtime" / "decode_cnn_ctc_v19_logits.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"selected_wer": 0.5497732671129346', freeze)
        self.assertIn('"selected_cer": 0.4634567759027818', freeze)
        self.assertIn('"--decoder-artifact"', evaluator)
        self.assertIn("FrozenCtcDecoder.from_artifact(", evaluator)
        self.assertIn("runtime_decoder.decode(", evaluator)
        self.assertIn('"acoustic_plus_decoder_ms"', evaluator)
        self.assertIn('"acoustic_plus_decoder_latency_p50_ms"', evaluator)
        self.assertIn('"acoustic_plus_decoder_latency_p95_ms"', evaluator)
        self.assertIn('"acoustic_plus_decoder_realtime_factor"', evaluator)
        self.assertIn("FrozenCtcDecoder.from_artifact(", runtime)
        self.assertIn("runtime_decoder.decode(", runtime)

    def test_deployed_decoder_edge_runner_is_guarded(self):
        wrapper = (
            ROOT / "scripts" / "evaluate-cnn-ctc-v19-deployed.sh"
        ).read_text(encoding="utf-8")
        controller = (
            SPEECH / "evaluation" / "run_cnn_ctc_v19_deployed_edge.py"
        ).read_text(encoding="utf-8")
        self.assertIn("flock -n 9", wrapper)
        self.assertIn('EXPECTED_SAMPLES=1273', wrapper)
        self.assertIn(
            'EXPECTED_MANIFEST_SHA256="fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"',
            wrapper,
        )
        self.assertIn("--fresh", wrapper)
        self.assertIn("stage_validation_dataset(", controller)
        self.assertIn("os.link(audio, destination)", controller)
        self.assertIn("rsync_push_command(", controller)
        self.assertIn("rsync_pull_command(", controller)
        self.assertIn("edge-speech-preflight.sh", controller)
        self.assertIn('"--preflight-only"', controller)
        self.assertIn('"--backend",\n                    "host"', controller)
        self.assertIn('"--runtime-backend",\n                    "host"', controller)
        self.assertIn('RUNTIME_BACKEND="host"', wrapper)
        self.assertIn('"--runtime-backend"', evaluator)
        self.assertIn("run-myriad-tensor.sh", evaluator)
        self.assertIn('"launcher_backend"', evaluator)
        self.assertIn('if branch != "main":', controller)
        self.assertIn('"acoustic_plus_decoder_latency_p95_ms"', controller)

    def test_decoder_entrypoints_exist(self):
        for name in (
            "tune-cnn-ctc-v19-decoder.sh",
            "freeze-cnn-ctc-v19-decoder.sh",
            "decode-cnn-ctc-v19-logits.sh",
            "benchmark-cnn-ctc-v19-decoder.sh",
            "evaluate-cnn-ctc-v19-deployed.sh",
            "run-cnn-ctc-v19-deployed-edge.sh",
            "run-myriad-tensor.sh",
        ):
            self.assertTrue((ROOT / "scripts" / name).is_file())


if __name__ == "__main__":
    unittest.main()
