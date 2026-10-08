#!/usr/bin/env python3
"""Throughput-oriented, zero-training QuartzNet ROCm evaluation.

Distinct from the frozen reference evaluator: batches *only* equal-length
frontend tensors, keeps each waveform independently normalized, retains
manifest-order hypotheses, and reports stage timing plus full WER/CER.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys
import time

import numpy as np
import soundfile as sf
import torch

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))
sys.path.insert(0, str(SPEECH_ROOT / "training"))

from evaluate_quartznet15x5_reference import (  # noqa: E402
    DEFAULT_ARCHIVE,
    DEFAULT_MANIFEST,
    accelerator_runtime,
    choose_device,
    load_manifest,
    sum_rate,
)
from quartznet15x5_reference import (  # noqa: E402
    DEFAULT_SPEC,
    DEFAULT_VOCAB,
    load_reference_model,
    load_reference_spec,
    load_reference_vocab,
    nemo_reference_features,
    reference_frontend_constants,
    reference_output_length,
)
from speech_asr.cnn_ctc import greedy_decode, greedy_decode_logits_diagnostics  # noqa: E402
from speech_asr.evaluation import character_error_counts, word_error_counts  # noqa: E402
from speech_asr.quartznet_batching import exact_shape_batches, percentile  # noqa: E402
from speech_asr.quartznet_reference_frontend import (  # noqa: E402
    nemo_reference_features_numpy,
)
from speech_asr.rocm_diagnostics import inspect_rocm_affinity, parse_cpuset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=pathlib.Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--device", choices=("rocm", "cuda"), default="rocm")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--frontend", choices=("torch", "numpy"), default="torch")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--verify-rocm-affinity", action="store_true")
    parser.add_argument("--verify-decoder-first", type=int, default=20)
    parser.add_argument(
        "--output", type=pathlib.Path,
        default=ROOT / "work/speech-asr/quartznet15x5-reference/rocm-throughput-result.json",
    )
    parser.add_argument(
        "--hypotheses", type=pathlib.Path,
        default=ROOT / "work/speech-asr/quartznet15x5-reference/rocm-throughput-hypotheses.jsonl",
    )
    args = parser.parse_args()

    try:
        if not 1 <= args.batch_size <= 32:
            raise ValueError("--batch-size must be between 1 and 32")
        if args.max_samples is not None and args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        if args.progress_interval < 0 or args.verify_decoder_first < 0:
            raise ValueError("progress interval and decoder verification must be nonnegative")

        spec = load_reference_spec(args.spec)
        vocab = load_reference_vocab(args.vocab)
        records = load_manifest(args.manifest)
        expected = int(spec["dataset"]["expected_utterances"])
        if len(records) != expected:
            raise ValueError(f"manifest has {len(records)} records; expected {expected}")
        if args.max_samples is not None:
            records = records[:args.max_samples]
        batches = exact_shape_batches(records, spec, args.batch_size)

        device, backend = choose_device(args.device, args.device_index)
        if args.verify_rocm_affinity and backend != "rocm":
            raise ValueError("--verify-rocm-affinity requires ROCm")
        model, imported = load_reference_model(args.archive, spec=spec, vocab=vocab)
        model.to(device).eval()
        window, bank = (
            reference_frontend_constants(spec, device=device)
            if args.frontend == "torch" else (None, None)
        )

        expected_cpus = (
            parse_cpuset(os.environ.get("SPEECH_ROCM_CPUSET", "0-15"))
            if args.verify_rocm_affinity else None
        )
        affinity_checks = 0
        max_narrow_threads = 0

        def check_affinity() -> None:
            nonlocal affinity_checks, max_narrow_threads
            snapshot = inspect_rocm_affinity(expected_cpus)
            affinity_checks += 1
            max_narrow_threads = max(max_narrow_threads, snapshot["narrow_thread_count"])

        if expected_cpus is not None:
            check_affinity()

        stage = {
            "audio_read_seconds": 0.0,
            "frontend_seconds": 0.0,
            "model_forward_seconds": 0.0,
            "logit_transfer_seconds": 0.0,
            "decode_scoring_seconds": 0.0,
        }
        batch_latencies_ms = []
        hypotheses = [None] * len(records)
        word_counts = []
        char_counts = []
        audio_seconds = 0.0
        processed = 0
        started = time.perf_counter()

        with torch.inference_mode():
            for padded_frames, indices in batches:
                prepared = []
                preparation_started = time.perf_counter()
                audio_read_before = stage["audio_read_seconds"]
                for index in indices:
                    record = records[index]
                    io_started = time.perf_counter()
                    samples, sample_rate = sf.read(
                        str(args.manifest.parent / record["audio_path"]),
                        dtype="float32", always_2d=False,
                    )
                    stage["audio_read_seconds"] += time.perf_counter() - io_started
                    if sample_rate != int(spec["frontend"]["sample_rate_hz"]):
                        raise ValueError(f"{record['id']}: unexpected sample rate")
                    if np.asarray(samples).ndim != 1:
                        raise ValueError(f"{record['id']}: audio must be mono")
                    if len(samples) != int(record["sample_count"]):
                        raise ValueError(f"{record['id']}: sample count changed")
                    if args.frontend == "torch":
                        features, valid_frames = nemo_reference_features(
                            samples, spec, device=device, window=window, bank=bank
                        )
                    else:
                        numpy_features, valid_frames = nemo_reference_features_numpy(
                            samples, spec
                        )
                        features = torch.from_numpy(numpy_features).to(device)
                    if int(features.shape[-1]) != padded_frames:
                        raise ValueError(
                            f"{record['id']}: feature shape changed "
                            f"{features.shape[-1]} != {padded_frames}"
                        )
                    prepared.append((index, record, len(samples) / sample_rate, valid_frames, features))

                # Explicit boundaries make stage timings interpretable. They
                # also prevent model timings from including queued frontend work.
                joined = torch.cat([item[4] for item in prepared], dim=0)
                torch.cuda.synchronize(device)
                # Include GPU frontend and concatenation time, but exclude
                # audio reads already tracked separately.
                preparation_seconds = time.perf_counter() - preparation_started
                audio_read_seconds = stage["audio_read_seconds"] - audio_read_before
                stage["frontend_seconds"] += max(
                    0.0, preparation_seconds - audio_read_seconds
                )
                begin = time.perf_counter()
                output = model(joined)
                torch.cuda.synchronize(device)
                model_seconds = time.perf_counter() - begin
                stage["model_forward_seconds"] += model_seconds
                batch_latencies_ms.append(1000.0 * model_seconds)

                transfer_started = time.perf_counter()
                logits = output.detach().cpu().numpy()
                stage["logit_transfer_seconds"] += time.perf_counter() - transfer_started
                if logits.shape[0] != len(indices):
                    raise ValueError("acoustic model batch dimension changed")

                if expected_cpus is not None:
                    check_affinity()

                score_started = time.perf_counter()
                for batch_index, (index, record, duration, valid_frames, _) in enumerate(prepared):
                    valid_output = reference_output_length(valid_frames, spec)
                    if not 1 <= valid_output <= logits.shape[1]:
                        raise ValueError(f"{record['id']}: invalid acoustic output length")
                    valid_logits = logits[batch_index, :valid_output]
                    # NumPy argmax retains the same first-index tie policy as
                    # the reference Python decoder's max(range(...), key=...).
                    hypothesis = greedy_decode(np.argmax(valid_logits, axis=1), vocab)
                    if index < args.verify_decoder_first:
                        reference_decoded = greedy_decode_logits_diagnostics(
                            valid_logits, vocab
                        )["hypothesis"]
                        if hypothesis != reference_decoded:
                            raise ValueError(f"{record['id']}: vectorized decoder changed transcript")
                    reference = str(record["text"])
                    words = word_error_counts(reference, hypothesis)
                    chars = character_error_counts(reference, hypothesis)
                    word_counts.append(words)
                    char_counts.append(chars)
                    audio_seconds += duration
                    hypotheses[index] = {
                        "id": record["id"], "reference": reference,
                        "hypothesis": hypothesis,
                        "word_edits": words.to_dict(), "character_edits": chars.to_dict(),
                        "valid_feature_frames": valid_frames,
                        "valid_output_frames": valid_output,
                    }
                stage["decode_scoring_seconds"] += time.perf_counter() - score_started
                previous = processed
                processed += len(indices)
                if args.progress_interval and (
                    previous == 0 or processed == len(records)
                    or processed // args.progress_interval > previous // args.progress_interval
                ):
                    elapsed = time.perf_counter() - started
                    print(
                        f"[quartznet-batch] processed={processed}/{len(records)} "
                        f"batch={len(indices)} frames={padded_frames} "
                        f"wer={sum_rate(word_counts):.6f} "
                        f"rate={processed / max(elapsed, 1e-9):.2f}/s",
                        file=sys.stderr, flush=True,
                    )

        if any(item is None for item in hypotheses):
            raise RuntimeError("batching lost one or more hypotheses")
        if expected_cpus is not None:
            check_affinity()
        wall_seconds = time.perf_counter() - started
        wer = sum_rate(word_counts)
        cer = sum_rate(char_counts)
        full = len(records) == expected and args.max_samples is None
        ceiling = float(spec["qualification"]["reproduction_max_wer"])
        status = ("pass" if wer <= ceiling else "fail") if full else "diagnostic"
        result = {
            "schema": "speech-asr/quartznet-exact-shape-throughput-benchmark",
            "version": 1, "status": status,
            "model_id": spec["id"], "dataset_id": spec["dataset"]["id"],
            "samples": len(records), "full_dev_clean": full,
            "training_performed": False,
            "wer": wer, "cer": cer, "reproduction_max_wer": ceiling,
            "source": imported["source"],
            "batch_policy": {
                "kind": "exact-padded-feature-shape",
                "batch_size_requested": args.batch_size,
                "batches": len(batches),
                "mean_batch_size": len(records) / len(batches),
                "unique_padded_shapes": len(set(frames for frames, _ in batches)),
                "manifest_order_output": True,
                "frontend_individually_normalized": True,
            },
            "runtime": {
                "accelerator": accelerator_runtime(device, backend),
                "frontend": args.frontend,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "wall_seconds": wall_seconds,
                "utterances_per_second": len(records) / wall_seconds,
                "audio_seconds": audio_seconds,
                "end_to_end_rtf_excluding_model_load": wall_seconds / audio_seconds,
                "stages": stage,
                "model_batches": {
                    "p50_ms": percentile(batch_latencies_ms, 0.5),
                    "p95_ms": percentile(batch_latencies_ms, 0.95),
                    "total_seconds": stage["model_forward_seconds"],
                    "model_only_rtf": stage["model_forward_seconds"] / audio_seconds,
                },
                "rocm_affinity": (
                    {
                        "status": "pass", "checks": affinity_checks,
                        "expected_cpus": sorted(expected_cpus),
                        "max_narrow_threads_observed": max_narrow_threads,
                    } if expected_cpus is not None else None
                ),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        args.hypotheses.parent.mkdir(parents=True, exist_ok=True)
        args.hypotheses.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in hypotheses),
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 3 if status == "fail" else 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
