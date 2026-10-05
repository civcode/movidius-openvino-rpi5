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


def manifest_text(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )


def write_manifest(
    path: pathlib.Path,
    records: list[dict[str, Any]],
) -> str:
    payload = manifest_text(records)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(
                f"held-out manifest is immutable once written: {path}"
            )
    else:
        path.write_text(payload, encoding="utf-8", newline="\n")
    return sha256_path(path)


def write_json_immutable(path: pathlib.Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(
                f"held-out qualification is immutable once written: {path}"
            )
    else:
        path.write_text(payload, encoding="utf-8")


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
    write_json_immutable(output_dir / "qualification.json", result)
    return result


def verify_existing(
    *,
    source_manifest: pathlib.Path,
    train_manifest: pathlib.Path,
    selection_manifest: pathlib.Path,
    output_dir: pathlib.Path,
    spec_path: pathlib.Path,
    vocab_path: pathlib.Path,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.jsonl"
    qualification_path = output_dir / "qualification.json"
    for path in (
        source_manifest,
        train_manifest,
        selection_manifest,
        spec_path,
        vocab_path,
        manifest_path,
        qualification_path,
    ):
        if not path.is_file():
            raise ValueError(
                f"held-out verification input missing: {path}"
            )

    qualification = json.loads(
        qualification_path.read_text(encoding="utf-8")
    )
    if qualification.get("schema") != "speech-asr/heldout-evaluation-manifest":
        raise ValueError("held-out qualification schema mismatch")
    if qualification.get("version") != 1:
        raise ValueError("held-out qualification version mismatch")
    if qualification.get("status") != "structurally_valid":
        raise ValueError("held-out qualification is not structurally_valid")
    if qualification.get("role") != "test_only":
        raise ValueError("held-out qualification role must remain test_only")
    partition = qualification.get("partition", {})
    if partition.get("official_meetings") != list(OFFICIAL_MEETINGS):
        raise ValueError("held-out official meeting identity changed")
    if partition.get("training_allowed") is not False:
        raise ValueError("held-out qualification permits training")
    if partition.get("checkpoint_selection_allowed") is not False:
        raise ValueError("held-out qualification permits checkpoint selection")
    if qualification.get("overlap") != {
        "training_records": 0,
        "selection_records": 0,
        "training_meetings": 0,
        "selection_meetings": 0,
    }:
        raise ValueError("held-out qualification overlap is not zero")

    source_refs = (
        ("evaluation", source_manifest),
        ("training_reference", train_manifest),
        ("selection_reference", selection_manifest),
    )
    for name, path in source_refs:
        ref = qualification.get("sources", {}).get(name, {})
        if ref.get("manifest_sha256") != sha256_path(path):
            raise ValueError(
                f"held-out {name} source hash differs from qualification"
            )

    spec = load_spec(spec_path)
    vocab = load_vocab(vocab_path)
    model = qualification.get("model", {})
    if model.get("id") != "cnn_ctc_v3":
        raise ValueError("held-out qualification model id changed")
    if model.get("spec_sha256") != canonical_sha256(spec):
        raise ValueError("held-out qualification model spec hash changed")
    if model.get("vocab_sha256") != canonical_sha256(vocab):
        raise ValueError("held-out qualification vocabulary hash changed")

    test_records = load_records(manifest_path)
    if meetings(test_records) != set(OFFICIAL_MEETINGS):
        raise ValueError("held-out qualified manifest meetings changed")
    test_ref = qualification.get("test", {})
    if test_ref.get("manifest_sha256") != sha256_path(manifest_path):
        raise ValueError("held-out qualified manifest hash changed")
    if test_ref.get("stats") != stats(test_records):
        raise ValueError("held-out qualified manifest statistics changed")

    train = load_records(train_manifest)
    selection = load_records(selection_manifest)
    ids = record_ids(test_records)
    if ids & record_ids(train) or ids & record_ids(selection):
        raise ValueError("held-out record overlap appeared after qualification")
    if meetings(test_records) & meetings(train):
        raise ValueError("held-out training meeting overlap appeared")
    if meetings(test_records) & meetings(selection):
        raise ValueError("held-out selection meeting overlap appeared")
    return qualification


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
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the sealed held-out manifest without rewriting it",
    )
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    args = parser.parse_args()

    try:
        function = verify_existing if args.verify_only else qualify
        result = function(
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
