#!/usr/bin/env python3
"""Qualify meeting-disjoint expanded AMI training with the frozen v1 validation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.cnn_ctc import (  # noqa: E402
    canonical_sha256,
    load_spec,
    load_vocab,
    manifest_record_eligibility,
)
from speech_asr.contracts import validate_speech_sample  # noqa: E402
from speech_asr.text import normalize_text_v1  # noqa: E402

AMI_ROOT = ROOT / "work" / "speech-asr" / "ami"
DEFAULT_TRAIN = AMI_ROOT / "ami-train-es2005-v1" / "manifest.jsonl"
DEFAULT_VALIDATION = AMI_ROOT / "model-quality-v1" / "validation.manifest.jsonl"
DEFAULT_OUTPUT = AMI_ROOT / "model-quality-v2"
DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "vocab.json"


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(validate_speech_sample(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def eligible_records(records: list[dict[str, Any]], spec: dict, vocab: dict):
    selected = []
    excluded = {"too_long": 0, "target_too_long": 0}
    for record in records:
        decision = manifest_record_eligibility(record, spec, vocab)
        if decision["eligible"]:
            selected.append(record)
        else:
            reason = str(decision["reason"])
            if reason not in excluded:
                raise ValueError(f"{record['id']}: unknown exclusion {reason!r}")
            excluded[reason] += 1
    return selected, excluded


def relocate(records, *, source_dir: pathlib.Path, output_dir: pathlib.Path):
    result = []
    for record in records:
        value = copy.deepcopy(record)
        raw = pathlib.Path(value["audio"]["path"])
        audio = raw if raw.is_absolute() else source_dir / raw
        value["audio"]["path"] = os.path.relpath(
            audio.resolve(), output_dir.resolve()
        )
        result.append(value)
    return result


def write_manifest(path: pathlib.Path, records: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return sha256_path(path)


def stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples = sum(
        int(x["audio"]["end_sample"]) - int(x["audio"]["start_sample"])
        for x in records
    )
    texts = [normalize_text_v1(x["transcript"]["text"]) for x in records]
    return {
        "records": len(records),
        "audio_samples": samples,
        "audio_seconds": samples / 16000.0,
        "words": sum(len(x.split()) for x in texts),
        "characters_no_spaces": sum(len(x.replace(" ", "")) for x in texts),
        "meetings": sorted({str(x["metadata"]["meeting"]) for x in records}),
        "speaker_keys": sorted(
            {
                f"{x['metadata']['meeting']}:{x['metadata']['speaker']}"
                for x in records
            }
        ),
    }


def qualify(
    *,
    train_source: pathlib.Path,
    validation_source: pathlib.Path,
    output_dir: pathlib.Path,
    spec_path: pathlib.Path,
    vocab_path: pathlib.Path,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    vocab = load_vocab(vocab_path)
    train_all = load_records(train_source)
    validation_all = load_records(validation_source)
    train, train_excluded = eligible_records(train_all, spec, vocab)
    validation, validation_excluded = eligible_records(validation_all, spec, vocab)
    if not train or not validation:
        raise ValueError("qualified train and validation sets must be non-empty")

    train_ids = {x["id"] for x in train}
    validation_ids = {x["id"] for x in validation}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise ValueError("train/validation record overlap: " + ", ".join(overlap))
    train_meetings = {str(x["metadata"]["meeting"]) for x in train}
    validation_meetings = {str(x["metadata"]["meeting"]) for x in validation}
    meeting_overlap = sorted(train_meetings & validation_meetings)
    if meeting_overlap:
        raise ValueError("train/validation meeting overlap: " + ", ".join(meeting_overlap))

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.manifest.jsonl"
    validation_path = output_dir / "validation.manifest.jsonl"
    train_sha = write_manifest(
        train_path,
        relocate(train, source_dir=train_source.parent, output_dir=output_dir),
    )
    validation_sha = write_manifest(
        validation_path,
        relocate(
            validation,
            source_dir=validation_source.parent,
            output_dir=output_dir,
        ),
    )
    result = {
        "schema": "speech-asr/model-quality-manifests",
        "version": 2,
        "status": "structurally_valid",
        "sources": {
            "train": {
                "manifest": str(train_source),
                "manifest_sha256": sha256_path(train_source),
                "records": len(train_all),
            },
            "validation": {
                "manifest": str(validation_source),
                "manifest_sha256": sha256_path(validation_source),
                "records": len(validation_all),
            },
        },
        "model": {
            "id": spec["id"],
            "spec_sha256": canonical_sha256(spec),
            "vocab_sha256": canonical_sha256(vocab),
        },
        "partition": {"record_overlap": 0, "meeting_overlap": 0},
        "train": {
            "manifest": str(train_path),
            "manifest_sha256": train_sha,
            "stats": stats(train),
            "excluded": train_excluded,
        },
        "validation": {
            "manifest": str(validation_path),
            "manifest_sha256": validation_sha,
            "stats": stats(validation),
            "excluded": validation_excluded,
        },
    }
    (output_dir / "qualification.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument(
        "--validation-source", type=pathlib.Path, default=DEFAULT_VALIDATION
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    args = parser.parse_args()
    try:
        result = qualify(
            train_source=args.train_source,
            validation_source=args.validation_source,
            output_dir=args.output_dir,
            spec_path=args.spec,
            vocab_path=args.vocab,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
