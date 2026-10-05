#!/usr/bin/env python3
"""Evaluate one frozen cnn_ctc_v3 checkpoint and ONNX artifact without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
TRAINING = SPEECH_ROOT / "training"
sys.path.insert(0, str(SPEECH_ROOT / "python"))
sys.path.insert(0, str(TRAINING))

from cnn_ctc_v3 import CnnCtcV3  # noqa: E402
from speech_asr.audio import read_f32le  # noqa: E402
from speech_asr.cnn_ctc import (  # noqa: E402
    acoustic_output_length,
    aggregate_decoder_diagnostics,
    canonical_sha256,
    greedy_decode_logits_diagnostics,
    load_spec,
    load_vocab,
    manifest_record_eligibility,
)
from speech_asr.cnn_ctc_compare import compare_arrays  # noqa: E402
from speech_asr.cnn_ctc_frontend import logmel_features  # noqa: E402
from speech_asr.contracts import validate_speech_sample  # noqa: E402
from speech_asr.evaluation import (  # noqa: E402
    character_error_counts,
    word_error_counts,
)

DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "vocab.json"


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def load_records(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield validate_speech_sample(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def resolve_audio(manifest: pathlib.Path, record: dict[str, Any]) -> pathlib.Path:
    raw = pathlib.Path(record["audio"]["path"])
    return raw if raw.is_absolute() else manifest.parent / raw


def sum_rate(counts) -> float:
    errors = sum(value.errors for value in counts)
    refs = sum(value.reference_units for value in counts)
    return errors / max(1, refs)


def evaluate(
    *,
    checkpoint_path: pathlib.Path,
    onnx_path: pathlib.Path,
    manifest_path: pathlib.Path,
    benchmark_id: str,
    spec_path: pathlib.Path,
    vocab_path: pathlib.Path,
    device_name: str,
    source_experiment_id: str,
    source_attempt_id: str,
) -> dict[str, Any]:
    for path in (
        checkpoint_path,
        onnx_path,
        manifest_path,
        spec_path,
        vocab_path,
    ):
        if not path.is_file():
            raise ValueError(f"required frozen evaluation input missing: {path}")

    spec = load_spec(spec_path)
    vocab = load_vocab(vocab_path)
    if spec.get("id") != "cnn_ctc_v3":
        raise ValueError("frozen reference evaluator is restricted to cnn_ctc_v3")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_id") != "cnn_ctc_v3":
        raise ValueError("checkpoint is not cnn_ctc_v3")
    if checkpoint.get("spec_sha256") != canonical_sha256(spec):
        raise ValueError("checkpoint model spec hash mismatch")
    if checkpoint.get("vocab_sha256") != canonical_sha256(vocab):
        raise ValueError("checkpoint vocabulary hash mismatch")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested for frozen evaluation but unavailable")
        device = torch.device("cuda:0")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("device must be cuda or cpu")

    model = CnnCtcV3(spec, len(vocab["tokens"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    pytorch_words = []
    pytorch_chars = []
    onnx_words = []
    onnx_chars = []
    per_sample = []
    manifest_samples = 0
    skipped = {"too_long": [], "target_too_long": []}
    compared_frames = 0
    matching_frames = 0
    frame_argmax_mismatches = 0
    max_abs_error = 0.0
    min_reference_top2_margin = float("inf")
    max_mismatched_reference_top2_margin = 0.0
    mismatch_margin_weighted_sum = 0.0
    mismatched_samples = 0
    mismatch_sample_examples: list[str] = []
    decoded_hypothesis_mismatches = 0
    decoded_hypothesis_mismatch_examples: list[dict[str, str]] = []

    for record in load_records(manifest_path):
        manifest_samples += 1
        decision = manifest_record_eligibility(record, spec, vocab)
        if not decision["eligible"]:
            reason = str(decision["reason"])
            if reason not in skipped:
                raise ValueError(
                    f"{record['id']}: unknown eligibility reason {reason!r}"
                )
            skipped[reason].append(record["id"])
            continue

        audio = read_f32le(resolve_audio(manifest_path, record))
        features, valid_frames = logmel_features(audio.samples, spec)
        output_frames = acoustic_output_length(valid_frames, spec)
        if output_frames != int(decision["valid_output_frames"]):
            raise ValueError(
                f"{record['id']}: output frame count differs from eligibility"
            )

        torch_features = torch.from_numpy(features).to(device)
        with torch.no_grad():
            pytorch_logits = (
                model(torch_features)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        onnx_logits = session.run(
            [output_name],
            {input_name: features.astype(np.float32, copy=False)},
        )[0].astype(np.float32, copy=False)

        if pytorch_logits.shape != onnx_logits.shape:
            raise ValueError(
                f"{record['id']}: PyTorch/ONNX shape mismatch "
                f"{pytorch_logits.shape} != {onnx_logits.shape}"
            )
        valid_pytorch = pytorch_logits[0, :output_frames, :]
        valid_onnx = onnx_logits[0, :output_frames, :]
        comparison = compare_arrays(
            valid_pytorch[np.newaxis, :, :],
            valid_onnx[np.newaxis, :, :],
        )
        frame_count = int(comparison["frame_count"])
        mismatches = int(comparison["frame_argmax_mismatches"])
        compared_frames += frame_count
        matching_frames += int(comparison["frame_argmax_matches"])
        frame_argmax_mismatches += mismatches
        max_abs_error = max(
            max_abs_error,
            float(comparison["max_abs_error"]),
        )
        min_reference_top2_margin = min(
            min_reference_top2_margin,
            float(comparison["min_reference_top2_margin"]),
        )
        if mismatches:
            mismatched_samples += 1
            if len(mismatch_sample_examples) < 10:
                mismatch_sample_examples.append(str(record["id"]))
            max_mismatched_reference_top2_margin = max(
                max_mismatched_reference_top2_margin,
                float(
                    comparison[
                        "max_mismatched_reference_top2_margin"
                    ]
                ),
            )
            mismatch_margin_weighted_sum += (
                float(
                    comparison[
                        "mean_mismatched_reference_top2_margin"
                    ]
                )
                * mismatches
            )

        pytorch_decoder = greedy_decode_logits_diagnostics(
            valid_pytorch,
            vocab,
        )
        onnx_decoder = greedy_decode_logits_diagnostics(
            valid_onnx,
            vocab,
        )
        if pytorch_decoder["hypothesis"] != onnx_decoder["hypothesis"]:
            decoded_hypothesis_mismatches += 1
            if len(decoded_hypothesis_mismatch_examples) < 10:
                decoded_hypothesis_mismatch_examples.append(
                    {
                        "id": str(record["id"]),
                        "pytorch": str(pytorch_decoder["hypothesis"]),
                        "onnx": str(onnx_decoder["hypothesis"]),
                    }
                )
        reference = record["transcript"]["text"]
        pytorch_word = word_error_counts(
            reference,
            pytorch_decoder["hypothesis"],
        )
        pytorch_char = character_error_counts(
            reference,
            pytorch_decoder["hypothesis"],
        )
        onnx_word = word_error_counts(
            reference,
            onnx_decoder["hypothesis"],
        )
        onnx_char = character_error_counts(
            reference,
            onnx_decoder["hypothesis"],
        )
        pytorch_words.append(pytorch_word)
        pytorch_chars.append(pytorch_char)
        onnx_words.append(onnx_word)
        onnx_chars.append(onnx_char)

        per_sample.append(
            {
                "id": record["id"],
                "reference": reference,
                "pytorch_hypothesis": pytorch_decoder["hypothesis"],
                "onnx_hypothesis": onnx_decoder["hypothesis"],
                "valid_feature_frames": valid_frames,
                "valid_output_frames": output_frames,
                "pytorch_wer": pytorch_word.rate,
                "pytorch_cer": pytorch_char.rate,
                "onnx_wer": onnx_word.rate,
                "onnx_cer": onnx_char.rate,
                "reference_words": len(reference.split()),
                "reference_characters_no_spaces": len(
                    reference.replace(" ", "")
                ),
                "decoder": {
                    key: value
                    for key, value in pytorch_decoder.items()
                    if key != "hypothesis"
                },
            }
        )

    if not per_sample:
        raise ValueError("held-out manifest has no eligible samples")

    frame_argmax_agreement = matching_frames / max(1, compared_frames)
    pytorch_wer = sum_rate(pytorch_words)
    pytorch_cer = sum_rate(pytorch_chars)
    onnx_wer = sum_rate(onnx_words)
    onnx_cer = sum_rate(onnx_chars)
    mean_mismatched_reference_top2_margin = (
        mismatch_margin_weighted_sum / frame_argmax_mismatches
        if frame_argmax_mismatches
        else 0.0
    )
    agreement = {
        "frame_argmax_agreement": frame_argmax_agreement,
        "frame_argmax_matches": matching_frames,
        "frame_argmax_mismatches": frame_argmax_mismatches,
        "compared_frames": compared_frames,
        "max_abs_error": max_abs_error,
        "min_reference_top2_margin": (
            min_reference_top2_margin
            if min_reference_top2_margin != float("inf")
            else 0.0
        ),
        "max_mismatched_reference_top2_margin": (
            max_mismatched_reference_top2_margin
        ),
        "mean_mismatched_reference_top2_margin": (
            mean_mismatched_reference_top2_margin
        ),
        "mismatched_samples": mismatched_samples,
        "mismatch_sample_examples": mismatch_sample_examples,
        "decoded_hypothesis_mismatches": decoded_hypothesis_mismatches,
        "decoded_hypothesis_mismatch_examples": (
            decoded_hypothesis_mismatch_examples
        ),
        "pytorch_wer": pytorch_wer,
        "onnx_wer": onnx_wer,
        "wer_delta": onnx_wer - pytorch_wer,
        "pytorch_cer": pytorch_cer,
        "onnx_cer": onnx_cer,
        "cer_delta": onnx_cer - pytorch_cer,
    }
    if frame_argmax_agreement != 1.0:
        raise ValueError(
            "frozen PyTorch/ONNX frame argmax agreement is not exact: "
            + json.dumps(agreement, sort_keys=True)
        )

    decoder_summary = aggregate_decoder_diagnostics(
        [
            {
                "hypothesis": item["pytorch_hypothesis"],
                "reference_words": item["reference_words"],
                "reference_characters_no_spaces": item[
                    "reference_characters_no_spaces"
                ],
                "decoder": item["decoder"],
            }
            for item in per_sample
        ]
    )
    manifest_sha = sha256_path(manifest_path)
    result = {
        "schema": "speech-asr/frozen-reference-evaluation",
        "version": 1,
        "status": "completed",
        "role": "heldout_test",
        "source": {
            "experiment_id": source_experiment_id,
            "attempt_id": source_attempt_id,
            "checkpoint_sha256": sha256_path(checkpoint_path),
            "onnx_sha256": sha256_path(onnx_path),
        },
        "benchmark": {
            "id": benchmark_id,
            "manifest_sha256": manifest_sha,
            "manifest_samples": manifest_samples,
            "evaluated_samples": len(per_sample),
            "skipped": skipped,
        },
        "model": {
            "id": spec["id"],
            "spec_sha256": canonical_sha256(spec),
            "vocab_sha256": canonical_sha256(vocab),
        },
        "runtime": {
            "repo_commit": git_head(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
            "pytorch_device": str(device),
            "onnx_provider": "CPUExecutionProvider",
        },
        "agreement": {
            "frame_argmax_agreement": frame_argmax_agreement,
            "compared_frames": compared_frames,
            "max_abs_error": max_abs_error,
        },
        "metrics": {
            "pytorch": {
                "wer": pytorch_wer,
                "cer": pytorch_cer,
                "decoder": decoder_summary,
            },
            "onnx": {
                "wer": onnx_wer,
                "cer": onnx_cer,
            },
        },
        "per_sample": per_sample,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--onnx", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--source-experiment-id", required=True)
    parser.add_argument("--source-attempt-id", required=True)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    try:
        result = evaluate(
            checkpoint_path=args.checkpoint,
            onnx_path=args.onnx,
            manifest_path=args.manifest,
            benchmark_id=args.benchmark_id,
            spec_path=args.spec,
            vocab_path=args.vocab,
            device_name=args.device,
            source_experiment_id=args.source_experiment_id,
            source_attempt_id=args.source_attempt_id,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
