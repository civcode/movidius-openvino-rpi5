#!/usr/bin/env python3
"""Qualify the official AMI Full-corpus-ASR unseen evaluation partition."""

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
DEFAULT_SOURCE = (
    AMI_ROOT / "ami-eval-full-corpus-asr-sc-v1" / "manifest.jsonl"
)
DEFAULT_TRAIN = AMI_ROOT / "model-quality-v3" / "train.manifest.jsonl"
DEFAULT_SELECTION = (
    AMI_ROOT / "model-quality-v1" / "validation.manifest.jsonl"
)
DEFAULT_OUTPUT = AMI_ROOT / "heldout-eval-v1"
DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "vocab.json"

OFFICIAL_MEETINGS = (
    "EN2002a", "EN2002b", "EN2002c", "EN2002d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1009a", "IS1009b", "IS1009c", "IS1009d",
    "TS3003a", "TS3003b", "TS3003c", "TS3003d",
)


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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


def eligible_records(
    records: list[dict[str, Any]],
    spec: dict[str, Any],
    vocab: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    excluded = {"too_long": 0, "target_too_long": 0}
    for record in records:
        decision = manifest_record_eligibility(record, spec, vocab)
        if decision["eligible"]:
            selected.append(record)
            continue
        reason = str(decision["reason"])
        if reason not in excluded:
            raise ValueError(
                f"{record['id']}: unknown eligibility exclusion {reason!r}"
            )
        excluded[reason] += 1
    if not selected:
        raise ValueError("held-out evaluation has no eligible records")
    return selected, excluded


def relocate(
    records: list[dict[str, Any]],
    *,
    source_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> list[dict[str, Any]]:
    relocated: list[dict[str, Any]] = []
    for record in records:
        value = copy.deepcopy(record)
        raw = pathlib.Path(value["audio"]["path"])
        audio = raw if raw.is_absolute() else source_dir / raw
        value["audio"]["path"] = os.path.relpath(
            audio.resolve(),
            output_dir.resolve(),
        )
        relocated.append(value)
    return relocated


def write_manifest(
    path: pathlib.Path,
    records: list[dict[str, Any]],
) -> str:
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


def record_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(record["id"]) for record in records}


def meetings(records: list[dict[str, Any]]) -> set[str]:
    return {str(record["metadata"]["meeting"]) for record in records}


def stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples = sum(
        int(record["audio"]["end_sample"])
        - int(record["audio"]["start_sample"])
        for record in records
    )
    texts = [
        normalize_text_v1(record["transcript"]["text"])
        for record in records
    ]
    return {
        "records": len(records),
        "audio_samples": samples,
        "audio_seconds": samples / 16000.0,
        "words": sum(len(text.split()) for text in texts),
        "characters_no_spaces": sum(
            len(text.replace(" ", "")) for text in texts
        ),
        "meetings": sorted(meetings(records)),
        "speaker_keys": sorted(
            {
                f"{record['metadata']['meeting']}:"
                f"{record['metadata']['speaker']}"
                for record in records
            }
        ),
    }


def qualify(
    *,
    source_manifest: pathlib.Path,
    train_manifest: pathlib.Path,
    selection_manifest: pathlib.Path,
    output_dir: pathlib.Path,
    spec_path: pathlib.Path,
    vocab_path: pathlib.Path,
) -> dict[str, Any]:
    for path in (
        source_manifest,
        train_manifest,
        selection_manifest,
        spec_path,
        vocab_path,
    ):
        if not path.is_file():
            raise ValueError(f"required held-out qualification input missing: {path}")

    spec = load_spec(spec_path)
    vocab = load_vocab(vocab_path)
    if spec.get("id") != "cnn_ctc_v3":
        raise ValueError("held-out evaluation is frozen to cnn_ctc_v3")

    source_all = load_records(source_manifest)
    train = load_records(train_manifest)
    selection = load_records(selection_manifest)

    source_meetings = meetings(source_all)
    expected_meetings = set(OFFICIAL_MEETINGS)
    if source_meetings != expected_meetings:
        missing = sorted(expected_meetings - source_meetings)
        extra = sorted(source_meetings - expected_meetings)
        raise ValueError(
            "official Full-corpus-ASR SC meeting identity mismatch: "
            f"missing={missing}, extra={extra}"
        )

    test, excluded = eligible_records(source_all, spec, vocab)

    test_ids = record_ids(test)
    train_ids = record_ids(train)
    selection_ids = record_ids(selection)
    train_record_overlap = sorted(test_ids & train_ids)
    selection_record_overlap = sorted(test_ids & selection_ids)
    train_meeting_overlap = sorted(meetings(test) & meetings(train))
    selection_meeting_overlap = sorted(
        meetings(test) & meetings(selection)
    )
    if (
        train_record_overlap
        or selection_record_overlap
        or train_meeting_overlap
        or selection_meeting_overlap
    ):
        raise ValueError(
            "held-out boundary overlap detected: "
            f"train_records={train_record_overlap}, "
            f"selection_records={selection_record_overlap}, "
            f"train_meetings={train_meeting_overlap}, "
            f"selection_meetings={selection_meeting_overlap}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    relocated = relocate(
        test,
        source_dir=source_manifest.parent,
        output_dir=output_dir,
    )
    manifest_sha = write_manifest(manifest_path, relocated)

    result = {
        "schema": "speech-asr/heldout-evaluation-manifest",
        "version": 1,
        "status": "structurally_valid",
        "role": "test_only",
        "partition": {
            "authority": "AMI Full-corpus-ASR",
            "role": "SC-unseen-evaluation",
            "official_meetings": list(OFFICIAL_MEETINGS),
            "checkpoint_selection_allowed": False,
            "training_allowed": False,
        },
        "sources": {
            "evaluation": {
                "manifest": str(source_manifest),
                "manifest_sha256": sha256_path(source_manifest),
                "records": len(source_all),
            },
            "training_reference": {
                "manifest": str(train_manifest),
                "manifest_sha256": sha256_path(train_manifest),
                "records": len(train),
            },
            "selection_reference": {
                "manifest": str(selection_manifest),
                "manifest_sha256": sha256_path(selection_manifest),
                "records": len(selection),
            },
        },
        "model": {
            "id": spec["id"],
            "spec_sha256": canonical_sha256(spec),
            "vocab_sha256": canonical_sha256(vocab),
        },
        "overlap": {
            "training_records": 0,
            "selection_records": 0,
            "training_meetings": 0,
            "selection_meetings": 0,
        },
        "test": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "excluded": excluded,
            "stats": stats(test),
        },
    }
    (output_dir / "qualification.json").write_text(
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
    parser.add_argument(
        "--train-manifest",
        type=pathlib.Path,
        default=DEFAULT_TRAIN,
    )
    parser.add_argument(
        "--selection-manifest",
        type=pathlib.Path,
        default=DEFAULT_SELECTION,
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    args = parser.parse_args()

    try:
        result = qualify(
            source_manifest=args.source_manifest,
            train_manifest=args.train_manifest,
            selection_manifest=args.selection_manifest,
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
