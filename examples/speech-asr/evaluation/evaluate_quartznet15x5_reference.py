#!/usr/bin/env python3
"""Zero-training QuartzNet15x5Base-En reproduction on LibriSpeech dev-clean."""

from __future__ import annotations

import argparse
import json
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


def choose_device(request: str) -> torch.device:
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unsupported device: {request}")


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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--progress-interval", type=int, default=50)
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

        device = choose_device(args.device)
        model, imported = load_reference_model(
            args.archive,
            spec=spec,
            vocab=vocab,
        )
        model.to(device)
        model.eval()

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

                features, valid_frames = nemo_reference_features(
                    samples,
                    spec,
                    device=device,
                )
                logits = model(features)[0]
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
                "kind": spec["frontend"]["kind"],
                "stft_center": spec["frontend"]["stft_center"],
                "hann_periodic": spec["frontend"]["hann_periodic"],
                "normalization_ddof": spec["frontend"]["normalization_ddof"],
                "evaluation_dither": spec["frontend"]["evaluation_dither"],
            },
            "runtime": {
                "device": str(device),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "soundfile_version": sf.__version__,
                "wall_seconds": time.monotonic() - started,
            },
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
