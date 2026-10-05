#!/usr/bin/env python3
"""Derive deterministic model-quality train/validation manifests from prepared AMI."""

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

DEFAULT_SOURCE = (
    ROOT
    / "work"
    / "speech-asr"
    / "ami"
    / "ami-benchmark-v1"
    / "manifest.jsonl"
)
DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "vocab.json"


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_speakers(value: str) -> tuple[str, ...]:
    speakers = tuple(part.strip() for part in value.split(",") if part.strip())
    if not speakers:
        raise argparse.ArgumentTypeError("speaker list must not be empty")
    if len(set(speakers)) != len(speakers):
        raise argparse.ArgumentTypeError("speaker list contains duplicates")
    return speakers


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            records.append(validate_speech_sample(value))
    if not records:
        raise ValueError(f"source manifest is empty: {path}")
    return records


def canonical_line(record: dict[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_manifest(path: pathlib.Path, records: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_line(record))
            handle.write("\n")
    return sha256_path(path)


def relocate_audio_paths(
    records: list[dict[str, Any]],
    *,
    source_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> list[dict[str, Any]]:
    relocated: list[dict[str, Any]] = []
    for record in records:
        value = copy.deepcopy(record)
        raw = pathlib.Path(value["audio"]["path"])
        source_audio = raw if raw.is_absolute() else source_dir / raw
        value["audio"]["path"] = os.path.relpath(
            source_audio.resolve(),
            output_dir.resolve(),
        )
        relocated.append(value)
    return relocated


def corpus_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    audio_samples = sum(
        int(record["audio"]["end_sample"]) - int(record["audio"]["start_sample"])
        for record in records
    )
    words = 0
    characters = 0
    for record in records:
        text = normalize_text_v1(record["transcript"]["text"])
        words += len(text.split())
        characters += len(text.replace(" ", ""))
    return {
        "records": len(records),
        "audio_samples": audio_samples,
        "audio_seconds": audio_samples / 16000.0,
        "words": words,
        "characters_no_spaces": characters,
        "speakers": sorted(
            {str(record["metadata"]["speaker"]) for record in records}
        ),
        "meetings": sorted(
            {str(record["metadata"]["meeting"]) for record in records}
        ),
    }


def qualify(
    *,
    source_manifest: pathlib.Path,
    spec_path: pathlib.Path,
    vocab_path: pathlib.Path,
    output_dir: pathlib.Path,
    train_speakers: tuple[str, ...],
    validation_speakers: tuple[str, ...],
) -> dict[str, Any]:
    overlap = sorted(set(train_speakers) & set(validation_speakers))
    if overlap:
        raise ValueError(
            "train/validation speaker sets overlap: " + ", ".join(overlap)
        )

    spec = load_spec(spec_path)
    vocab = load_vocab(vocab_path)
    source_records = load_records(source_manifest)

    available = {
        str(record["metadata"].get("speaker"))
        for record in source_records
    }
    missing = sorted(
        (set(train_speakers) | set(validation_speakers)) - available
    )
    if missing:
        raise ValueError(
            "requested speakers absent from source manifest: "
            + ", ".join(missing)
        )

    selected: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
    }
    excluded: dict[str, dict[str, int]] = {
        "train": {"too_long": 0, "target_too_long": 0},
        "validation": {"too_long": 0, "target_too_long": 0},
    }

    train_set = set(train_speakers)
    validation_set = set(validation_speakers)
    for record in source_records:
        speaker = str(record["metadata"]["speaker"])
        role = (
            "train"
            if speaker in train_set
            else "validation"
            if speaker in validation_set
            else None
        )
        if role is None:
            continue

        decision = manifest_record_eligibility(record, spec, vocab)
        if decision["eligible"]:
            selected[role].append(record)
            continue
        reason = str(decision["reason"])
        if reason not in excluded[role]:
            raise ValueError(
                f"{record['id']}: unknown eligibility reason {reason!r}"
            )
        excluded[role][reason] += 1

    if not selected["train"]:
        raise ValueError("no eligible training records")
    if not selected["validation"]:
        raise ValueError("no eligible validation records")

    train_ids = {record["id"] for record in selected["train"]}
    validation_ids = {record["id"] for record in selected["validation"]}
    id_overlap = sorted(train_ids & validation_ids)
    if id_overlap:
        raise ValueError(
            "train/validation record overlap: " + ", ".join(id_overlap)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.manifest.jsonl"
    validation_path = output_dir / "validation.manifest.jsonl"
    train_records = relocate_audio_paths(
        selected["train"],
        source_dir=source_manifest.parent,
        output_dir=output_dir,
    )
    validation_records = relocate_audio_paths(
        selected["validation"],
        source_dir=source_manifest.parent,
        output_dir=output_dir,
    )
    train_sha = write_manifest(train_path, train_records)
    validation_sha = write_manifest(
        validation_path,
        validation_records,
    )

    result = {
        "schema": "speech-asr/model-quality-manifests",
        "version": 1,
        "status": "structurally_valid",
        "source": {
            "manifest": str(source_manifest),
            "manifest_sha256": sha256_path(source_manifest),
            "records": len(source_records),
        },
        "model": {
            "id": spec["id"],
            "spec_sha256": canonical_sha256(spec),
            "vocab_sha256": canonical_sha256(vocab),
        },
        "partition": {
            "train_speakers": list(train_speakers),
            "validation_speakers": list(validation_speakers),
            "record_overlap": 0,
            "speaker_overlap": 0,
        },
        "train": {
            "manifest": str(train_path),
            "manifest_sha256": train_sha,
            "stats": corpus_stats(selected["train"]),
            "excluded": excluded["train"],
        },
        "validation": {
            "manifest": str(validation_path),
            "manifest_sha256": validation_sha,
            "stats": corpus_stats(selected["validation"]),
            "excluded": excluded["validation"],
        },
    }

    qualification_path = output_dir / "qualification.json"
    qualification_path.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=pathlib.Path,
        default=DEFAULT_SOURCE,
    )
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=ROOT
        / "work"
        / "speech-asr"
        / "ami"
        / "model-quality-v1",
    )
    parser.add_argument(
        "--train-speakers",
        type=parse_speakers,
        default=("A", "B", "C"),
    )
    parser.add_argument(
        "--validation-speakers",
        type=parse_speakers,
        default=("D",),
    )
    args = parser.parse_args()

    try:
        result = qualify(
            source_manifest=args.source_manifest,
            spec_path=args.spec,
            vocab_path=args.vocab,
            output_dir=args.output_dir,
            train_speakers=args.train_speakers,
            validation_speakers=args.validation_speakers,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
