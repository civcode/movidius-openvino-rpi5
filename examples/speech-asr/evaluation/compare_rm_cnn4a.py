#!/usr/bin/env python3
"""Compare an rm_cnn4a MYRIAD result with a CPU reference result."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.contracts import (  # noqa: E402
    ContractValidationError,
    validate_acoustic_regression_result,
)


IDENTITY_PATHS = (
    ("model", "id"),
    ("model", "family"),
    ("model", "bin_sha256"),
    ("fixture", "features_sha256"),
    ("fixture", "reference_scores_sha256"),
    ("runtime", "openvino_version"),
)

METRIC_NAMES = (
    "weighted_mean_infer_ms_per_frame",
    "utterance_avg_infer_ms_per_frame_p50",
    "utterance_avg_infer_ms_per_frame_p95",
    "max_error_max",
    "avg_error_mean",
    "rms_error_mean",
)


def load_result(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing result: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc
    try:
        return validate_acoustic_regression_result(value)
    except ContractValidationError as exc:
        raise ValueError(f"invalid acoustic regression result {path}: {exc}") from exc


def metric_delta(reference: float, candidate: float) -> dict[str, float | None]:
    return {
        "reference": reference,
        "candidate": candidate,
        "delta": candidate - reference,
        "ratio": candidate / reference if reference != 0 else None,
    }


def compare_results(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    reference_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    if reference["runtime"]["backend"] != "CPU":
        raise ValueError("reference backend must be CPU")
    if candidate["runtime"]["backend"] != "MYRIAD":
        raise ValueError("candidate backend must be MYRIAD")

    mismatches: list[str] = []
    matches: dict[str, bool] = {}
    for section, key in IDENTITY_PATHS:
        left = reference[section][key]
        right = candidate[section][key]
        path = f"{section}.{key}"
        matches[path] = left == right
        if left != right:
            mismatches.append(f"{path}: reference={left!r}, candidate={right!r}")

    xml_match = reference["model"]["xml_sha256"] == candidate["model"]["xml_sha256"]
    reference_graph = reference["model"].get("graph_sha256")
    candidate_graph = candidate["model"].get("graph_sha256")
    graph_match = None
    if reference_graph is not None and candidate_graph is not None:
        graph_match = reference_graph == candidate_graph

    reference_canonical = reference["model"].get("canonical_graph_sha256")
    candidate_canonical = candidate["model"].get("canonical_graph_sha256")
    canonical_match = None
    if reference_canonical is not None and candidate_canonical is not None:
        canonical_match = reference_canonical == candidate_canonical

    if not xml_match:
        if canonical_match is True:
            # Numeric layer IDs, serialization order, and Model Optimizer
            # metadata may differ while the named executable graph is equal.
            pass
        elif graph_match is True:
            # Older results may only carry the first semantic graph hash.
            pass
        elif canonical_match is False:
            mismatches.append(
                "model.canonical_graph_sha256: named executable IR graph differs "
                f"(reference={reference_canonical!r}, candidate={candidate_canonical!r})"
            )
        elif graph_match is False:
            mismatches.append(
                "model.graph_sha256: executable IR serialization differs "
                f"(reference={reference_graph!r}, candidate={candidate_graph!r}); "
                "canonical graph fingerprints are unavailable"
            )
        else:
            mismatches.append(
                "model.xml_sha256: raw IR XML differs and a semantic graph "
                "fingerprint is unavailable on one or both results"
            )

    if mismatches:
        raise ValueError(
            "candidate does not match frozen artifact/fixture identity:\n- "
            + "\n- ".join(mismatches)
        )

    ref_metrics = reference["metrics"]
    cand_metrics = candidate["metrics"]

    metrics = {
        name: metric_delta(float(ref_metrics[name]), float(cand_metrics[name]))
        for name in METRIC_NAMES
    }

    ref_load = reference["provenance"].get("model_load_ms")
    cand_load = candidate["provenance"].get("model_load_ms")
    if isinstance(ref_load, (int, float)) and isinstance(cand_load, (int, float)):
        metrics["model_load_ms"] = metric_delta(float(ref_load), float(cand_load))

    return {
        "schema": "speech-asr/acoustic-regression-comparison",
        "version": 1,
        "reference": reference_name,
        "candidate": candidate_name,
        "artifact_identity": {
            "xml_sha256_match": xml_match,
            **(
                {"graph_sha256_match": graph_match}
                if graph_match is not None
                else {}
            ),
            **(
                {"canonical_graph_sha256_match": canonical_match}
                if canonical_match is not None
                else {}
            ),
            "bin_sha256_match": matches["model.bin_sha256"],
            "features_sha256_match": matches["fixture.features_sha256"],
            "reference_scores_sha256_match": matches[
                "fixture.reference_scores_sha256"
            ],
            "openvino_version_match": matches["runtime.openvino_version"],
        },
        "counts": {
            "utterances_match": ref_metrics["utterances"] == cand_metrics["utterances"],
            "total_frames_match": ref_metrics["total_frames"] == cand_metrics["total_frames"],
            "utterances": cand_metrics["utterances"],
            "total_frames": cand_metrics["total_frames"],
        },
        "metrics": metrics,
        "acceptance": {
            "thresholds_applied": False,
            "note": "Evidence-only comparison. No MYRIAD pass/fail thresholds are defined.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=pathlib.Path,
        default=SPEECH_ROOT / "models" / "rm_cnn4a" / "cpu-reference-v1.json",
    )
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    reference_path = args.reference if args.reference.is_absolute() else ROOT / args.reference
    candidate_path = args.candidate if args.candidate.is_absolute() else ROOT / args.candidate

    try:
        comparison = compare_results(
            load_result(reference_path),
            load_result(candidate_path),
            reference_name=reference_path.name,
            candidate_name=candidate_path.name,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rendered = json.dumps(comparison, sort_keys=True, indent=2) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
