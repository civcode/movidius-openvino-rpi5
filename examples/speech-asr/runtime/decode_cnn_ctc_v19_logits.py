#!/usr/bin/env python3
"""Decode cnn_ctc_v19 logits with the frozen CPU beam/LM decoder artifact."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.cnn_ctc import load_spec, load_vocab  # noqa: E402
from speech_asr.ctc_beam import (  # noqa: E402
    FrozenCtcDecoder,
    load_decoder_artifact,
)

DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v19" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v19" / "vocab.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logits", type=pathlib.Path)
    parser.add_argument("--valid-output-frames", type=int, required=True)
    parser.add_argument("--decoder-artifact", type=pathlib.Path, required=True)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    try:
        spec = load_spec(args.spec)
        vocab = load_vocab(args.vocab)
        shape = tuple(int(value) for value in spec["output_contract"]["shape"])
        if not 1 <= args.valid_output_frames <= shape[1]:
            raise ValueError(
                f"valid_output_frames must be in range 1..{shape[1]}"
            )
        values = np.fromfile(args.logits, dtype=np.float32)
        expected = int(np.prod(shape))
        if values.size != expected:
            raise ValueError(
                f"logits contain {values.size} elements, expected {expected}"
            )
        artifact = load_decoder_artifact(args.decoder_artifact, vocab=vocab)
        runtime_decoder = FrozenCtcDecoder.from_artifact(
            artifact,
            vocab=vocab,
        )
        logits = values.reshape(shape)[0, : args.valid_output_frames, :]
        started = time.perf_counter()
        decoded = runtime_decoder.decode(logits, vocab)
        decode_ms = (time.perf_counter() - started) * 1000.0
        result = {
            "schema": "speech-asr/ctc-runtime-decode",
            "version": 1,
            "acoustic_model": "cnn_ctc_v19",
            "hypothesis": decoded["hypothesis"],
            "decode_ms": decode_ms,
            "valid_output_frames": args.valid_output_frames,
            "decoder": {
                key: decoded[key]
                for key in (
                    "kind",
                    "beam_width",
                    "token_top_k",
                    "lm_weight",
                    "word_bonus",
                    "prefix_tokens",
                )
            },
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
