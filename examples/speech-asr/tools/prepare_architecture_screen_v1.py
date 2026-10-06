#!/usr/bin/env python3
"""Derive and verify the deterministic model-quality-v4 architecture screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from qualify_expanded_model_quality_manifests import (  # noqa: E402
    load_records,
    relocate,
    sha256_path,
    stats,
    write_manifest,
)

AMI_ROOT = ROOT / "work" / "speech-asr" / "ami"
FULL_DIR = AMI_ROOT / "model-quality-v4"
SOURCE_TRAIN = FULL_DIR / "train.manifest.jsonl"
VALIDATION = FULL_DIR / "validation.manifest.jsonl"
OUTPUT_DIR = AMI_ROOT / "model-quality-v4-architecture-screen-v1"
OUTPUT_TRAIN = OUTPUT_DIR / "train.manifest.jsonl"
OUTPUT_PROVENANCE = OUTPUT_DIR / "screen.json"

SCREEN_ID = "ami-model-quality-v4-architecture-screen-v1"
SOURCE_TRAIN_SHA256 = "6025d17f08c1815d1710c3ab56cea51a34365f398e9877b4aefec234e64e4096"
VALIDATION_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"
SOURCE_RECORDS = 15738
VALIDATION_RECORDS = 1273
MODULUS = 4
REMAINDER = 0
SCREEN_EPOCHS = 12
RULE_ID = "sha256-id-first-u64-be-mod4-eq0-v1"

TRAIN_GROUPS = (
    "ES2003",
    "ES2005",
    "ES2006",
    "ES2007",
    "ES2008",
    "ES2009",
    "ES2010",
    "ES2012",
    "ES2013",
    "ES2014",
    "ES2015",
    "ES2016",
)
EXPECTED_MEETINGS = {
    f"{group}{suffix}"
    for group in TRAIN_GROUPS
    for suffix in ("a", "b", "c", "d")
}


def repo_relative(path: pathlib.Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def selected_by_rule(sample_id: str) -> bool:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % MODULUS
    return bucket == REMAINDER


def selected_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if selected_by_rule(str(record["id"]))]


def validate_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for path in (SOURCE_TRAIN, VALIDATION):
        if not path.is_file():
            raise ValueError(f"architecture-screen source missing: {path}")
    if sha256_path(SOURCE_TRAIN) != SOURCE_TRAIN_SHA256:
        raise ValueError("model-quality-v4 training manifest changed")
    if sha256_path(VALIDATION) != VALIDATION_SHA256:
        raise ValueError("model-quality-v4 validation manifest changed")

    train = load_records(SOURCE_TRAIN)
    validation = load_records(VALIDATION)
    if len(train) != SOURCE_RECORDS:
        raise ValueError(
            f"model-quality-v4 training record count changed: {len(train)}"
        )
    if len(validation) != VALIDATION_RECORDS:
        raise ValueError(
            f"model-quality-v4 validation record count changed: {len(validation)}"
        )
    meetings = {str(record["metadata"]["meeting"]) for record in train}
    if meetings != EXPECTED_MEETINGS:
        missing = sorted(EXPECTED_MEETINGS - meetings)
        extra = sorted(meetings - EXPECTED_MEETINGS)
        raise ValueError(
            f"model-quality-v4 training meetings changed; missing={missing} extra={extra}"
        )
    return train, validation


def expected_selection(
    train: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = selected_records(train)
    if not selected:
        raise ValueError("architecture-screen selection is empty")
    ratio = len(selected) / len(train)
    if not 0.20 <= ratio <= 0.30:
        raise ValueError(
            f"architecture-screen selection ratio is implausible: {ratio:.6f}"
        )
    meetings = {str(record["metadata"]["meeting"]) for record in selected}
    if meetings != EXPECTED_MEETINGS:
        missing = sorted(EXPECTED_MEETINGS - meetings)
        extra = sorted(meetings - EXPECTED_MEETINGS)
        raise ValueError(
            "architecture-screen must represent every training meeting; "
            f"missing={missing} extra={extra}"
        )
    return selected, stats(selected)


def provenance_document(
    *,
    selected_stats: dict[str, Any],
    manifest_sha256: str,
    validation_stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "speech-asr/architecture-screen",
        "version": 1,
        "status": "valid",
        "id": SCREEN_ID,
        "source": {
            "train_manifest": repo_relative(SOURCE_TRAIN),
            "train_manifest_sha256": SOURCE_TRAIN_SHA256,
            "train_records": SOURCE_RECORDS,
        },
        "selection": {
            "kind": "deterministic_hash_bucket",
            "rule_id": RULE_ID,
            "field": "id",
            "hash": "sha256",
            "integer": "first_8_bytes_unsigned_big_endian",
            "modulus": MODULUS,
            "remainder": REMAINDER,
            "target_fraction": 0.25,
        },
        "train": {
            "manifest": repo_relative(OUTPUT_TRAIN),
            "manifest_sha256": manifest_sha256,
            "stats": selected_stats,
        },
        "validation": {
            "manifest": repo_relative(VALIDATION),
            "manifest_sha256": VALIDATION_SHA256,
            "stats": validation_stats,
            "fraction": 1.0,
        },
        "budget": {
            "epochs": SCREEN_EPOCHS,
            "batch_size": 1,
            "checkpoint_selection": "validation_cer",
        },
        "roles": {
            "purpose": "architecture_ranking_proxy",
            "promotion_requires_fuller_budget_confirmation": True,
            "sealed_heldout_allowed_for_selection": False,
        },
    }


def prepare() -> dict[str, Any]:
    train, validation = validate_sources()
    selected, selected_stats = expected_selection(train)
    relocated = relocate(
        selected,
        source_dir=SOURCE_TRAIN.parent,
        output_dir=OUTPUT_DIR,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_sha = write_manifest(OUTPUT_TRAIN, relocated)
    document = provenance_document(
        selected_stats=selected_stats,
        manifest_sha256=manifest_sha,
        validation_stats=stats(validation),
    )
    OUTPUT_PROVENANCE.write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return document


def verify() -> dict[str, Any]:
    train, validation = validate_sources()
    selected, selected_stats = expected_selection(train)
    if not OUTPUT_TRAIN.is_file() or not OUTPUT_PROVENANCE.is_file():
        raise ValueError(
            "architecture-screen artifacts are missing; run without --verify-only"
        )

    actual = load_records(OUTPUT_TRAIN)
    expected = relocate(
        selected,
        source_dir=SOURCE_TRAIN.parent,
        output_dir=OUTPUT_DIR,
    )
    if actual != expected:
        raise ValueError(
            "architecture-screen manifest differs from deterministic source selection"
        )

    manifest_sha = sha256_path(OUTPUT_TRAIN)
    document = provenance_document(
        selected_stats=selected_stats,
        manifest_sha256=manifest_sha,
        validation_stats=stats(validation),
    )
    recorded = json.loads(OUTPUT_PROVENANCE.read_text(encoding="utf-8"))
    if recorded != document:
        raise ValueError("architecture-screen provenance differs from derived boundary")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        result = verify() if args.verify_only else prepare()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
