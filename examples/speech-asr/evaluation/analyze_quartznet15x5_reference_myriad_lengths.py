#!/usr/bin/env python3
"""Summarize QuartzNet MYRIAD accuracy and latency as a function of input length."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_EVIDENCE = (
    ROOT
    / "work"
    / "speech-asr"
    / "deployed-evaluations"
    / "quartznet15x5-reference-libri-dev-clean"
)

# Same-sample exact-static MYRIAD sweep on 2026-10-07:
# T=3088 remained non-catastrophic; T=3104 and every tested larger shape
# collapsed to zero valid-frame argmax agreement with ~630 raw-logit error.
MYRIAD_CATASTROPHIC_ONSET_FRAMES = 3104
MYRIAD_LAST_NONCATASTROPHIC_FRAMES = 3088


def rate(items: list[dict], key: str) -> float:
    errors = sum(int(item[key]["errors"]) for item in items)
    refs = sum(int(item[key]["reference_units"]) for item in items)
    return errors / max(1, refs)


def aggregate(items: list[dict]) -> dict:
    inf = [float(item["inference_ms"]) for item in items]
    word_errors = sum(int(item["word_edits"]["errors"]) for item in items)
    word_refs = sum(int(item["word_edits"]["reference_units"]) for item in items)
    char_errors = sum(int(item["character_edits"]["errors"]) for item in items)
    char_refs = sum(int(item["character_edits"]["reference_units"]) for item in items)
    return {
        "samples": len(items),
        "word_errors": word_errors,
        "reference_words": word_refs,
        "wer": word_errors / max(1, word_refs),
        "character_errors": char_errors,
        "reference_characters": char_refs,
        "cer": char_errors / max(1, char_refs),
        "inference_ms_mean": sum(inf) / max(1, len(inf)),
        "inference_ms_min": min(inf) if inf else None,
        "inference_ms_max": max(inf) if inf else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=pathlib.Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    hypotheses = args.evidence_dir / "hypotheses.jsonl"
    result_path = args.evidence_dir / "result.json"
    if not hypotheses.is_file():
        raise SystemExit(f"missing hypotheses: {hypotheses}")
    if not result_path.is_file():
        raise SystemExit(f"missing result: {result_path}")

    rows = [
        json.loads(line)
        for line in hypotheses.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if len(rows) != int(result.get("samples", -1)):
        raise SystemExit(
            f"sample mismatch: hypotheses={len(rows)} result={result.get('samples')}"
        )

    bins = [
        ("<=512", 0, 512),
        ("528-768", 528, 768),
        ("784-1024", 784, 1024),
        ("1040-1536", 1040, 1536),
        ("1552-2048", 1552, 2048),
        (">2048", 2064, 10**9),
    ]
    by_bin = []
    for name, lo, hi in bins:
        items = [
            item
            for item in rows
            if lo <= int(item["tensor_feature_frames"]) <= hi
        ]
        value = aggregate(items)
        value.update({"bin": name, "min_frames": lo, "max_frames": hi})
        by_bin.append(value)

    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in rows:
        grouped[int(item["tensor_feature_frames"])].append(item)

    by_shape = []
    for frames, items in grouped.items():
        value = aggregate(items)
        value.update({"tensor_feature_frames": frames})
        by_shape.append(value)

    below_catastrophic_onset = [
        item
        for item in rows
        if int(item["tensor_feature_frames"]) < MYRIAD_CATASTROPHIC_ONSET_FRAMES
    ]
    at_or_above_catastrophic_onset = [
        item
        for item in rows
        if int(item["tensor_feature_frames"]) >= MYRIAD_CATASTROPHIC_ONSET_FRAMES
    ]
    long_below_catastrophic_onset = [
        item
        for item in rows
        if 2048 < int(item["tensor_feature_frames"]) < MYRIAD_CATASTROPHIC_ONSET_FRAMES
    ]
    by_error_contribution = sorted(
        by_shape,
        key=lambda value: (value["word_errors"], value["reference_words"]),
        reverse=True,
    )[: args.top]
    by_wer = sorted(
        [value for value in by_shape if value["reference_words"] >= 20],
        key=lambda value: (value["wer"], value["reference_words"]),
        reverse=True,
    )[: args.top]
    worst_utterances = sorted(
        rows,
        key=lambda item: (
            int(item["word_edits"]["errors"]),
            float(item["word_edits"]["rate"]),
            int(item["tensor_feature_frames"]),
        ),
        reverse=True,
    )[: args.top]

    analysis = {
        "schema": "speech-asr/quartznet-reference-myriad-length-analysis",
        "version": 1,
        "source_result_status": result.get("status"),
        "source_wer": result.get("quality", {}).get("wer"),
        "source_cer": result.get("quality", {}).get("cer"),
        "samples": len(rows),
        "overall": aggregate(rows),
        "myriad_shape_boundary": {
            "same_sample_last_noncatastrophic_frames": MYRIAD_LAST_NONCATASTROPHIC_FRAMES,
            "same_sample_catastrophic_onset_frames": MYRIAD_CATASTROPHIC_ONSET_FRAMES,
            "below_catastrophic_onset": aggregate(below_catastrophic_onset),
            "at_or_above_catastrophic_onset": aggregate(
                at_or_above_catastrophic_onset
            ),
            "long_below_catastrophic_onset": aggregate(
                long_below_catastrophic_onset
            ),
        },
        "by_length_bin": by_bin,
        "worst_shapes_by_word_error_contribution": by_error_contribution,
        "worst_shapes_by_wer_min_20_reference_words": by_wer,
        "worst_utterances_by_word_errors": [
            {
                "id": item["id"],
                "tensor_feature_frames": item["tensor_feature_frames"],
                "valid_feature_frames": item["valid_feature_frames"],
                "inference_ms": item["inference_ms"],
                "word_edits": item["word_edits"],
                "character_edits": item["character_edits"],
                "reference": item["reference"],
                "hypothesis": item["hypothesis"],
            }
            for item in worst_utterances
        ],
    }
    out = args.evidence_dir / "length-analysis.json"
    out.write_text(json.dumps(analysis, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print("QuartzNet MYRIAD length analysis")
    print(f"overall: samples={analysis['overall']['samples']} wer={analysis['overall']['wer']:.6f} cer={analysis['overall']['cer']:.6f}")
    print()
    boundary = analysis["myriad_shape_boundary"]
    print("MYRIAD temporal-shape boundary attribution")
    for label in (
        "below_catastrophic_onset",
        "long_below_catastrophic_onset",
        "at_or_above_catastrophic_onset",
    ):
        value = boundary[label]
        print(
            f"{label:>31}  samples={value['samples']:4d}  "
            f"words={value['reference_words']:5d}  "
            f"errors={value['word_errors']:4d}  "
            f"wer={value['wer']:.6f}  cer={value['cer']:.6f}"
        )
    print(
        "same-sample transition: "
        f"T={boundary['same_sample_last_noncatastrophic_frames']} non-catastrophic, "
        f"T={boundary['same_sample_catastrophic_onset_frames']} catastrophic"
    )
    print()
    print("by tensor-feature length")
    for value in by_bin:
        print(
            f"{value['bin']:>10}  samples={value['samples']:4d}  "
            f"words={value['reference_words']:5d}  "
            f"wer={value['wer']:.6f}  cer={value['cer']:.6f}  "
            f"infer_mean_ms={value['inference_ms_mean']:.1f}"
        )
    print()
    print("worst shapes by word-error contribution")
    for value in by_error_contribution:
        print(
            f"T={value['tensor_feature_frames']:4d}  samples={value['samples']:3d}  "
            f"errors={value['word_errors']:4d}/{value['reference_words']:4d}  "
            f"wer={value['wer']:.6f}  infer_mean_ms={value['inference_ms_mean']:.1f}"
        )
    print()
    print("worst utterances by word errors")
    for item in analysis["worst_utterances_by_word_errors"][:10]:
        edits = item["word_edits"]
        print(
            f"{item['id']}  T={item['tensor_feature_frames']:4d}  "
            f"errors={edits['errors']:3d}/{edits['reference_units']:3d}  "
            f"wer={edits['rate']:.6f}  infer_ms={item['inference_ms']:.1f}"
        )
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
