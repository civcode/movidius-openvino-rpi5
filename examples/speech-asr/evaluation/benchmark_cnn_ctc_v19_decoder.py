#!/usr/bin/env python3
"""Benchmark the frozen cnn_ctc_v19 CPU decoder on cached physical logits."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.cnn_ctc import (  # noqa: E402
    load_spec,
    load_vocab,
    manifest_record_eligibility,
)
from speech_asr.contracts import validate_speech_sample  # noqa: E402
from speech_asr.ctc_beam import (  # noqa: E402
    decode_with_artifact,
    load_decoder_artifact,
)
from speech_asr.evaluation import (  # noqa: E402
    character_error_counts,
    latency_summary_ms,
    word_error_counts,
)

DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v19" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v19" / "vocab.json"


def records(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield validate_speech_sample(json.loads(line))


def sum_rate(counts) -> float:
    errors = sum(item.errors for item in counts)
    refs = sum(item.reference_units for item in counts)
    return errors / max(1, refs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--logits-dir", type=pathlib.Path, required=True)
    parser.add_argument("--decoder-artifact", type=pathlib.Path, required=True)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    try:
        spec = load_spec(args.spec)
        vocab = load_vocab(args.vocab)
        artifact = load_decoder_artifact(args.decoder_artifact, vocab=vocab)
        output_shape = tuple(int(v) for v in spec["output_contract"]["shape"])
        output_elements = int(np.prod(output_shape))

        word_counts = []
        char_counts = []
        latencies = []
        samples = 0
        started = time.perf_counter()
        manifest_records = list(records(args.manifest))
        total = len(manifest_records)

        for processed, record in enumerate(manifest_records, start=1):
            decision = manifest_record_eligibility(record, spec, vocab)
            if not decision["eligible"]:
                continue
            logits_path = args.logits_dir / str(record["id"]) / "logits.f32"
            if not logits_path.is_file():
                raise ValueError(f"missing logits cache: {logits_path}")
            values = np.fromfile(logits_path, dtype=np.float32)
            if values.size != output_elements:
                raise ValueError(
                    f"{record['id']}: logits elements {values.size} != "
                    f"{output_elements}"
                )
            logits = values.reshape(output_shape)[
                0, : int(decision["valid_output_frames"]), :
            ]
            decode_started = time.perf_counter()
            decoded = decode_with_artifact(logits, vocab, artifact)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            latencies.append(decode_ms)

            reference = str(record["transcript"]["text"])
            hypothesis = str(decoded["hypothesis"])
            word_counts.append(word_error_counts(reference, hypothesis))
            char_counts.append(character_error_counts(reference, hypothesis))
            samples += 1

            if args.progress_interval > 0 and (
                processed == 1
                or processed % args.progress_interval == 0
                or processed == total
            ):
                elapsed = time.perf_counter() - started
                rate = processed / max(elapsed, 1e-9)
                eta = (total - processed) / max(rate, 1e-9)
                print(
                    "[decoder-bench] "
                    f"processed={processed}/{total} "
                    f"evaluated={samples} "
                    f"rate={rate:.2f}/s eta={eta:.1f}s",
                    flush=True,
                )

        if not latencies:
            raise ValueError("no eligible decoder samples")
        timing = latency_summary_ms(latencies)
        result = {
            "schema": "speech-asr/ctc-decoder-benchmark",
            "version": 1,
            "acoustic_model": "cnn_ctc_v19",
            "decoder": {
                "kind": "ctc-prefix-beam-char-ngram-v1",
                **artifact["decoder"],
            },
            "samples": samples,
            "wer": sum_rate(word_counts),
            "cer": sum_rate(char_counts),
            "decode_latency_p50_ms": timing["p50_ms"],
            "decode_latency_p95_ms": timing["p95_ms"],
            "decode_latency_mean_ms": statistics.fmean(latencies),
            "wall_seconds": time.perf_counter() - started,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
