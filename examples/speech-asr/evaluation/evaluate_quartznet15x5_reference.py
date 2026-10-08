#!/usr/bin/env python3
"""Zero-training QuartzNet15x5Base-En reproduction on LibriSpeech dev-clean."""

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
TRAINING = SPEECH_ROOT / "training"
sys.path.insert(0, str(SPEECH_ROOT / "python"))
sys.path.insert(0, str(TRAINING))

from quartznet15x5_reference import (  # noqa: E402
    DEFAULT_SPEC,
    DEFAULT_VOCAB,
    load_reference_model,
    load_reference_spec,
    load_reference_vocab,
    nemo_reference_features,
    reference_output_length,
)
from speech_asr.cnn_ctc import greedy_decode_logits_diagnostics  # noqa: E402
from speech_asr.evaluation import character_error_counts, word_error_counts  # noqa: E402
from speech_asr.quartznet_reference_frontend import (  # noqa: E402
    nemo_reference_features_numpy,
)
from speech_asr.rocm_diagnostics import (  # noqa: E402
    inspect_rocm_affinity,
    parse_cpuset,
    summarize_model_timings,
)


DEFAULT_ARCHIVE = (
    ROOT
    / "work"
    / "speech-asr"
    / "pretrained"
    / "quartznet15x5-en-base-v2"
    / "QuartzNet15x5-En-Base.nemo"
)
DEFAULT_MANIFEST = (
    ROOT / "work" / "speech-asr" / "librispeech" / "dev-clean" / "manifest.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "work"
    / "speech-asr"
    / "quartznet15x5-reference"
    / "dev-clean-result.json"
)
DEFAULT_HYPOTHESES = (
    ROOT
    / "work"
    / "speech-asr"
    / "quartznet15x5-reference"
    / "dev-clean-hypotheses.jsonl"
)


def torch_accelerator_backend() -> str | None:
    if not torch.cuda.is_available():
        return None
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "unknown"


def choose_device(request: str, index: int) -> tuple[torch.device, str]:
    if index < 0:
        raise ValueError("--device-index must be >= 0")
    if request == "cpu":
        return torch.device("cpu"), "cpu"

    backend = torch_accelerator_backend()
    if request in {"cuda", "rocm"}:
        if backend != request:
            raise ValueError(
                f"--device {request} requested but this PyTorch build exposes "
                f"{backend or 'no GPU backend'} "
                f"(torch.version.cuda={getattr(torch.version, 'cuda', None)!r}, "
                f"torch.version.hip={getattr(torch.version, 'hip', None)!r})"
            )
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"--device-index {index} requested but only "
                f"{torch.cuda.device_count()} {request} device(s) are visible"
            )
        return torch.device(f"cuda:{index}"), request

    if request == "auto":
        if backend in {"cuda", "rocm"}:
            if index >= torch.cuda.device_count():
                raise ValueError(
                    f"--device-index {index} requested but only "
                    f"{torch.cuda.device_count()} accelerator device(s) are visible"
                )
            return torch.device(f"cuda:{index}"), backend
        return torch.device("cpu"), "cpu"
    raise ValueError(f"unsupported device: {request}")


def accelerator_runtime(device: torch.device, backend: str) -> dict:
    result = {
        "backend": backend,
        "torch_device": str(device),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_hip_version": getattr(torch.version, "hip", None),
    }
    if backend in {"cuda", "rocm"}:
        index = 0 if device.index is None else int(device.index)
        props = torch.cuda.get_device_properties(index)
        result.update(
            {
                "device_index": index,
                "device_name": torch.cuda.get_device_name(index),
                "device_total_memory_bytes": int(props.total_memory),
            }
        )
    return result


def sum_rate(values) -> float:
    errors = sum(item.errors for item in values)
    references = sum(item.reference_units for item in values)
    return errors / max(1, references)


def load_manifest(path: pathlib.Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("LibriSpeech manifest contains duplicate sample ids")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=pathlib.Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "rocm"),
        default="auto",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--frontend", choices=("torch", "numpy"), default="torch")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="record synchronized acoustic-model forward latency and model-only RTF",
    )
    parser.add_argument(
        "--verify-rocm-affinity",
        action="store_true",
        help="fail if the ROCm bootstrap or any sampled thread loses its CPU mask",
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hypotheses", type=pathlib.Path, default=DEFAULT_HYPOTHESES)
    args = parser.parse_args()

    try:
        spec = load_reference_spec(args.spec)
        vocab = load_reference_vocab(args.vocab)
        records = load_manifest(args.manifest)
        expected = int(spec["dataset"]["expected_utterances"])
        if len(records) != expected:
            raise ValueError(f"dev-clean manifest has {len(records)} records, expected {expected}")
        if args.max_samples is not None:
            if args.max_samples < 1:
                raise ValueError("--max-samples must be positive")
            records = records[: args.max_samples]

        device, accelerator_backend = choose_device(
            args.device,
            args.device_index,
        )
        model, imported = load_reference_model(
            args.archive,
            spec=spec,
            vocab=vocab,
        )
        model.to(device)
        model.eval()

        if args.verify_rocm_affinity and accelerator_backend != "rocm":
            raise ValueError("--verify-rocm-affinity requires --device rocm")
        expected_cpus = (
            parse_cpuset(os.environ.get("SPEECH_ROCM_CPUSET", "0-15"))
            if args.verify_rocm_affinity else None
        )
        affinity_stats = {
            "checks": 0,
            "max_narrow_threads_observed": 0,
            "narrow_thread_observations": 0,
            "first_narrow_thread_examples": [],
        }

        def check_affinity() -> dict:
            snapshot = inspect_rocm_affinity(expected_cpus)
            affinity_stats["checks"] += 1
            count = snapshot["narrow_thread_count"]
            affinity_stats["max_narrow_threads_observed"] = max(
                affinity_stats["max_narrow_threads_observed"], count
            )
            affinity_stats["narrow_thread_observations"] += count
            if count and not affinity_stats["first_narrow_thread_examples"]:
                affinity_stats["first_narrow_thread_examples"] = snapshot[
                    "narrow_thread_examples"
                ]
            return snapshot

        if expected_cpus is not None:
            check_affinity()

        model_latencies_ms = []
        benchmark_audio_seconds = 0.0
        word_counts = []
        char_counts = []
        hypotheses = []
        started = time.monotonic()
        with torch.inference_mode():
            for index, record in enumerate(records, start=1):
                audio_path = args.manifest.parent / record["audio_path"]
                samples, sample_rate = sf.read(
                    str(audio_path),
                    dtype="float32",
                    always_2d=False,
                )
                if sample_rate != int(spec["frontend"]["sample_rate_hz"]):
                    raise ValueError(f"{record['id']}: unexpected sample rate {sample_rate}")
                if np.asarray(samples).ndim != 1:
                    raise ValueError(f"{record['id']}: reference audio must be mono")
                if len(samples) != int(record["sample_count"]):
                    raise ValueError(f"{record['id']}: sample count changed")

                if args.frontend == "torch":
                    features, valid_frames = nemo_reference_features(
                        samples,
                        spec,
                        device=device,
                    )
                else:
                    features_np, valid_frames = nemo_reference_features_numpy(
                        samples,
                        spec,
                    )
                    features = torch.from_numpy(features_np).to(device)
                if args.benchmark and accelerator_backend in {"rocm", "cuda"}:
                    torch.cuda.synchronize(device)
                model_started = time.perf_counter() if args.benchmark else None
                logits = model(features)[0]
                if args.benchmark:
                    if accelerator_backend in {"rocm", "cuda"}:
                        torch.cuda.synchronize(device)
                    model_latencies_ms.append(
                        (time.perf_counter() - model_started) * 1000.0
                    )
                    benchmark_audio_seconds += len(samples) / sample_rate
                if expected_cpus is not None:
                    check_affinity()

                valid_output = reference_output_length(valid_frames, spec)
                if valid_output < 1 or valid_output > logits.shape[0]:
                    raise ValueError(
                        f"{record['id']}: invalid acoustic output length "
                        f"{valid_output} for tensor {tuple(logits.shape)}"
                    )
                decoded = greedy_decode_logits_diagnostics(
                    logits[:valid_output].detach().cpu().numpy(),
                    vocab,
                )
                hypothesis = decoded["hypothesis"]
                reference = str(record["text"])
                words = word_error_counts(reference, hypothesis)
                chars = character_error_counts(reference, hypothesis)
                word_counts.append(words)
                char_counts.append(chars)
                hypotheses.append({
                    "id": record["id"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "word_edits": words.to_dict(),
                    "character_edits": chars.to_dict(),
                    "valid_feature_frames": valid_frames,
                    "valid_output_frames": valid_output,
                })

                if args.progress_interval and (
                    index == 1
                    or index == len(records)
                    or index % args.progress_interval == 0
                ):
                    elapsed = time.monotonic() - started
                    print(
                        f"[quartznet-ref] processed={index}/{len(records)} "
                        f"wer={sum_rate(word_counts):.6f} "
                        f"rate={index / max(elapsed, 1e-9):.2f}/s",
                        file=sys.stderr,
                        flush=True,
                    )

        wer = sum_rate(word_counts)
        cer = sum_rate(char_counts)
        published = float(spec["qualification"]["published_dev_clean_wer"])
        ceiling = float(spec["qualification"]["reproduction_max_wer"])
        full = len(records) == expected and args.max_samples is None
        status = "pass" if full and wer <= ceiling else ("fail" if full else "diagnostic")
        result = {
            "schema": "speech-asr/quartznet-pretrained-reference-evaluation",
            "version": 1,
            "status": status,
            "model_id": spec["id"],
            "dataset_id": spec["dataset"]["id"],
            "samples": len(records),
            "full_dev_clean": full,
            "training_performed": False,
            "wer": wer,
            "cer": cer,
            "published_dev_clean_wer": published,
            "absolute_wer_delta_from_published": wer - published,
            "reproduction_max_wer": ceiling,
            "source": imported["source"],
            "shared_decoder_symbol_count": imported["shared_decoder_symbol_count"],
            "target_only_symbol_count": imported["target_only_symbol_count"],
            "frontend": {
                "implementation": args.frontend,
                "kind": spec["frontend"]["kind"],
                "stft_center": spec["frontend"]["stft_center"],
                "hann_periodic": spec["frontend"]["hann_periodic"],
                "normalization_ddof": spec["frontend"]["normalization_ddof"],
                "evaluation_dither": spec["frontend"]["evaluation_dither"],
            },
            "runtime": {
                "device": str(device),
                "accelerator": accelerator_runtime(device, accelerator_backend),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "soundfile_version": sf.__version__,
                "wall_seconds": time.monotonic() - started,
            },
        }

        if args.benchmark:
            result["runtime"]["model_benchmark"] = summarize_model_timings(
                model_latencies_ms, benchmark_audio_seconds
            )
        if expected_cpus is not None:
            affinity_snapshot = check_affinity()
            result["runtime"]["rocm_affinity"] = {
                "status": "pass",
                "expected_cpus": sorted(expected_cpus),
                **affinity_stats,
                **affinity_snapshot,
            }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        args.hypotheses.parent.mkdir(parents=True, exist_ok=True)
        args.hypotheses.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in hypotheses),
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if status != "fail" else 3
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
