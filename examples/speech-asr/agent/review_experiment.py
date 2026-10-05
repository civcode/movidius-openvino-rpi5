#!/usr/bin/env python3
"""Summarize a completed speech-ASR experiment for REVIEW."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def latest_attempt(experiment: pathlib.Path) -> pathlib.Path:
    attempts = sorted((experiment / "attempts").glob("attempt-*"))
    if not attempts:
        raise ValueError(f"no attempts found below {experiment}")
    return attempts[-1]


def attempt_summary(experiment: pathlib.Path, attempt: pathlib.Path) -> dict[str, Any]:
    request = load_json(experiment / "request" / "experiment.json")
    attempt_doc = load_json(attempt / "attempt.json")
    result = load_json(attempt / "results" / "result.json")
    acceptance_path = attempt / "results" / "acceptance-evaluation.json"
    acceptance = load_json(acceptance_path) if acceptance_path.is_file() else None
    training_path = attempt / "results" / "training.json"
    training = load_json(training_path) if training_path.is_file() else None
    compatibility_path = attempt / "results" / "compatibility.json"
    compatibility = load_json(compatibility_path) if compatibility_path.is_file() else None
    hardware_path = attempt / "results" / "hardware.json"
    hardware = load_json(hardware_path) if hardware_path.is_file() else None

    review: dict[str, Any] = {
        "experiment_id": result["experiment_id"],
        "attempt_id": result["attempt_id"],
        "state": attempt_doc["state"],
        "outcome": result["outcome"],
        "failure_class": attempt_doc.get("failure_class"),
        "model": (
            hardware.get("model", {}).get("id")
            if hardware is not None
            else None
        ),
        "benchmark": request.get("benchmark"),
        "acceptance": (
            acceptance.get("status")
            if acceptance is not None
            else None
        ),
        "rejection_reasons": (
            acceptance.get("reasons", [])
            if acceptance is not None
            else []
        ),
        "metrics": result.get("metrics", {}),
        "threshold_checks": (
            acceptance.get("threshold_checks", {})
            if acceptance is not None
            else {}
        ),
    }

    if training is not None:
        history = training.get("history", [])
        review["training"] = {
            "parameter_count": training.get("parameter_count"),
            "epochs": training.get("epochs"),
            "optimizer_steps": training.get("optimizer_steps"),
            "best_epoch": training.get("best_epoch"),
            "best_validation_loss": training.get("best_validation_loss"),
            "gradient_clip_norm": training.get("gradient_clip_norm"),
            "lr_schedule": training.get("lr_schedule"),
            "duration_seconds": training.get("duration_seconds"),
            "train_audio_seconds_per_epoch": training.get(
                "train_audio_seconds_per_epoch"
            ),
            "processed_train_audio_seconds": training.get(
                "processed_train_audio_seconds"
            ),
            "validation_audio_seconds": training.get(
                "validation_audio_seconds"
            ),
            "device_requested": training.get("device_requested"),
            "device_used": training.get("device_used"),
            "model_device": training.get("model_device"),
            "observed_logits_device": training.get("observed_logits_device"),
            "ctc_loss_device": training.get("ctc_loss_device"),
            "cuda_device_name": training.get("cuda_device_name"),
            "cuda_peak_memory_allocated_bytes": training.get(
                "cuda_peak_memory_allocated_bytes"
            ),
            "cuda_peak_memory_reserved_bytes": training.get(
                "cuda_peak_memory_reserved_bytes"
            ),
            "last_epoch": history[-1] if history else None,
        }

    if compatibility is not None:
        comparison = compatibility.get("onnx_comparison", {}).get("comparison", {})
        physical = compatibility.get("pretraining_myriad_probe", {})
        review["compatibility"] = {
            "onnx_frame_argmax_agreement": comparison.get(
                "frame_argmax_agreement"
            ),
            "onnx_max_abs_error": comparison.get("max_abs_error"),
            "ir_status": compatibility.get("ir_validation", {}).get("status"),
            "pretraining_myriad_frame_argmax_agreement": (
                physical.get("comparison", {}).get("frame_argmax_agreement")
            ),
            "pretraining_myriad_frame_argmax_mismatches": (
                physical.get("comparison", {}).get("frame_argmax_mismatches")
            ),
            "pretraining_myriad_max_abs_error": (
                physical.get("comparison", {}).get("max_abs_error")
            ),
            "pretraining_myriad_max_mismatched_top2_margin": (
                physical.get("comparison", {}).get(
                    "max_mismatched_reference_top2_margin"
                )
            ),
            "pretraining_myriad_numerical_gate": physical.get(
                "numerical_gate"
            ),
        }

    if hardware is not None:
        review["decoder"] = hardware.get("metrics", {}).get("decoder")

    return review


def parent_summary(experiment: pathlib.Path) -> dict[str, Any] | None:
    request = load_json(experiment / "request" / "experiment.json")
    parent_id = request.get("parent_experiment_id")
    if not parent_id:
        return None
    parent = experiment.parent / parent_id
    if not parent.is_dir():
        return {"experiment_id": parent_id, "status": "missing-local-parent"}
    attempt = latest_attempt(parent)
    return attempt_summary(parent, attempt)


def delta(candidate: dict[str, Any], parent: dict[str, Any] | None) -> dict[str, Any]:
    if parent is None or parent.get("status") == "missing-local-parent":
        return {}

    candidate_benchmark = candidate.get("benchmark", {})
    parent_benchmark = parent.get("benchmark", {})
    same_benchmark = (
        candidate_benchmark.get("manifest_sha256")
        == parent_benchmark.get("manifest_sha256")
    )
    out: dict[str, Any] = {
        "same_benchmark_manifest": same_benchmark,
        "not_compared": [],
    }
    candidate_metrics = candidate.get("metrics", {})
    parent_metrics = parent.get("metrics", {})

    metric_keys = [
        "inference_latency_p50_ms",
        "inference_latency_p95_ms",
    ]
    if same_benchmark:
        metric_keys.extend(("wer", "cer", "realtime_factor"))
    else:
        out["not_compared"] = ["wer", "cer", "realtime_factor"]

    for key in metric_keys:
        a = candidate_metrics.get(key)
        b = parent_metrics.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[key] = {
                "candidate": a,
                "parent": b,
                "delta": a - b,
                "ratio": (a / b) if b != 0 else None,
            }
    return out

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=pathlib.Path, required=True)
    parser.add_argument("--attempt")
    args = parser.parse_args()

    try:
        experiment = args.experiment.resolve()
        attempt = (
            experiment / "attempts" / args.attempt
            if args.attempt
            else latest_attempt(experiment)
        )
        candidate = attempt_summary(experiment, attempt)
        parent = parent_summary(experiment)
        output = {
            "schema": "speech-asr/review-summary",
            "version": 1,
            "candidate": candidate,
            "parent": parent,
            "delta_vs_parent": delta(candidate, parent),
        }
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(output, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
