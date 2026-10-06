#!/usr/bin/env python3
"""Qualify the fresh model-quality-v4 AMI train/validation development boundary."""

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

from qualify_expanded_model_quality_manifests import (  # noqa: E402
    load_records,
    qualify as qualify_base,
    sha256_path,
)
from speech_asr.cnn_ctc import canonical_sha256  # noqa: E402

AMI_ROOT = ROOT / "work" / "speech-asr" / "ami"
DEFAULT_POLICY = (
    SPEECH_ROOT / "datasets" / "ami" / "splits" / "model-quality-v4-edinburgh-v1.json"
)
DEFAULT_LOCK_DIR = AMI_ROOT / "model-quality-v4-source-lock"
DEFAULT_TRAIN = AMI_ROOT / "ami-model-quality-v4-train-v1" / "manifest.jsonl"
DEFAULT_VALIDATION = (
    AMI_ROOT / "ami-model-quality-v4-validation-v1" / "manifest.jsonl"
)
DEFAULT_OUTPUT = AMI_ROOT / "model-quality-v4"
DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "vocab.json"

EXPECTED_TRAIN_GROUPS = {
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
}
EXPECTED_VALIDATION_GROUPS = {"ES2011"}
FORBIDDEN_GROUPS = {"ES2002", "EN2002", "ES2004", "IS1009", "TS3003"}


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def expand(groups: set[str]) -> set[str]:
    return {
        f"{group}{suffix}"
        for group in groups
        for suffix in ("a", "b", "c", "d")
    }


def meetings(path: pathlib.Path) -> set[str]:
    return {str(record["metadata"]["meeting"]) for record in load_records(path)}


def preparation_exclusions(manifest: pathlib.Path) -> list[dict[str, Any]]:
    provenance_path = manifest.parent / "provenance.json"
    provenance = load_json(provenance_path)
    excluded = provenance.get("excluded_segments", [])
    if not isinstance(excluded, list):
        raise ValueError(f"{provenance_path}: excluded_segments must be an array")
    values: list[dict[str, Any]] = []
    for index, item in enumerate(excluded):
        if not isinstance(item, dict):
            raise ValueError(
                f"{provenance_path}: excluded_segments[{index}] must be an object"
            )
        if item.get("reason") != "non_positive_annotated_segment_interval":
            raise ValueError(
                f"{provenance_path}: unsupported excluded segment reason"
            )
        values.append(dict(item))
    return values


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema") != "speech-asr/model-quality-data-policy":
        raise ValueError("model-quality-v4 policy schema mismatch")
    if policy.get("version") != 1:
        raise ValueError("model-quality-v4 policy version mismatch")
    if policy.get("id") != "ami-model-quality-v4-edinburgh-full-corpus-asr-v1":
        raise ValueError("model-quality-v4 policy id mismatch")
    if policy.get("authority", {}).get("partition_name") != "Full-corpus-ASR":
        raise ValueError("model-quality-v4 must use Full-corpus-ASR")
    if set(policy.get("train_groups", [])) != EXPECTED_TRAIN_GROUPS:
        raise ValueError("model-quality-v4 training groups changed")
    if set(policy.get("validation_groups", [])) != EXPECTED_VALIDATION_GROUPS:
        raise ValueError("model-quality-v4 validation groups changed")
    if set(policy.get("excluded_prior_selection_groups", [])) != {"ES2002"}:
        raise ValueError("model-quality-v4 must exclude prior ES2002 selection data")
    if set(policy.get("sealed_test_groups", [])) != {
        "EN2002",
        "ES2004",
        "IS1009",
        "TS3003",
    }:
        raise ValueError("model-quality-v4 held-out exclusions changed")


def validate_source_lock(
    *,
    policy: dict[str, Any],
    lock_dir: pathlib.Path,
) -> dict[str, Any]:
    lock_path = lock_dir / "source-lock.json"
    train_split_path = lock_dir / "train.split.json"
    validation_split_path = lock_dir / "validation.split.json"
    for path in (lock_path, train_split_path, validation_split_path):
        if not path.is_file():
            raise ValueError(f"model-quality-v4 source lock artifact missing: {path}")

    lock = load_json(lock_path)
    if lock.get("schema") != "speech-asr/ami-source-lock":
        raise ValueError("model-quality-v4 source lock schema mismatch")
    if lock.get("status") != "frozen":
        raise ValueError("model-quality-v4 source lock is not frozen")
    if lock.get("policy_sha256") != canonical_sha256(policy):
        raise ValueError("model-quality-v4 source lock policy hash mismatch")

    train_split = load_json(train_split_path)
    validation_split = load_json(validation_split_path)
    train_split_meetings = {str(x["meeting"]) for x in train_split.get("sources", [])}
    validation_split_meetings = {
        str(x["meeting"]) for x in validation_split.get("sources", [])
    }
    if train_split_meetings != expand(EXPECTED_TRAIN_GROUPS):
        raise ValueError("model-quality-v4 frozen train split meeting set mismatch")
    if validation_split_meetings != expand(EXPECTED_VALIDATION_GROUPS):
        raise ValueError("model-quality-v4 frozen validation split meeting set mismatch")

    audio_sha = lock.get("audio_sha256")
    if not isinstance(audio_sha, dict):
        raise ValueError("model-quality-v4 source lock lacks audio hashes")
    for split in (train_split, validation_split):
        for source in split["sources"]:
            meeting = str(source["meeting"])
            if source["audio"]["sha256"] != audio_sha.get(meeting):
                raise ValueError(
                    f"model-quality-v4 split/source-lock hash mismatch for {meeting}"
                )

    return {
        "source_lock_path": lock_path,
        "source_lock_sha256": sha256_path(lock_path),
        "train_split_path": train_split_path,
        "train_split_sha256": sha256_path(train_split_path),
        "validation_split_path": validation_split_path,
        "validation_split_sha256": sha256_path(validation_split_path),
    }


def validate_partition(
    *,
    train_manifest: pathlib.Path,
    validation_manifest: pathlib.Path,
) -> tuple[set[str], set[str]]:
    train_meetings = meetings(train_manifest)
    validation_meetings = meetings(validation_manifest)
    expected_train = expand(EXPECTED_TRAIN_GROUPS)
    expected_validation = expand(EXPECTED_VALIDATION_GROUPS)
    if train_meetings != expected_train:
        raise ValueError(
            "model-quality-v4 train meeting set mismatch: "
            f"{sorted(train_meetings)} != {sorted(expected_train)}"
        )
    if validation_meetings != expected_validation:
        raise ValueError(
            "model-quality-v4 validation meeting set mismatch: "
            f"{sorted(validation_meetings)} != {sorted(expected_validation)}"
        )
    if train_meetings & validation_meetings:
        raise ValueError("model-quality-v4 train/validation meeting overlap")
    forbidden = expand(FORBIDDEN_GROUPS)
    overlap = sorted((train_meetings | validation_meetings) & forbidden)
    if overlap:
        raise ValueError(
            "model-quality-v4 development boundary touches forbidden meetings: "
            + ", ".join(overlap)
        )
    return train_meetings, validation_meetings


def boundary_document(
    *,
    policy: dict[str, Any],
    lock: dict[str, Any],
    train_meetings: set[str],
    validation_meetings: set[str],
    train_exclusions: list[dict[str, Any]],
    validation_exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": policy["id"],
        "official_partition": "Full-corpus-ASR",
        "policy_sha256": canonical_sha256(policy),
        "source_lock_sha256": lock["source_lock_sha256"],
        "train_split_sha256": lock["train_split_sha256"],
        "validation_split_sha256": lock["validation_split_sha256"],
        "train_meetings": sorted(train_meetings),
        "validation_meetings": sorted(validation_meetings),
        "excluded_prior_selection_groups": ["ES2002"],
        "sealed_test_groups": ["EN2002", "ES2004", "IS1009", "TS3003"],
        "training_allowed_on_train": True,
        "training_allowed_on_validation": False,
        "checkpoint_selection_allowed_on_validation": True,
        "heldout_metrics_allowed_for_model_selection": False,
        "source_exclusions": {
            "train": train_exclusions,
            "validation": validation_exclusions,
        },
    }


def qualify_v4(
    *,
    policy_path: pathlib.Path,
    lock_dir: pathlib.Path,
    train_source: pathlib.Path,
    validation_source: pathlib.Path,
    output_dir: pathlib.Path,
    spec_path: pathlib.Path,
    vocab_path: pathlib.Path,
) -> dict[str, Any]:
    policy = load_json(policy_path)
    validate_policy(policy)
    lock = validate_source_lock(policy=policy, lock_dir=lock_dir)
    train_meetings, validation_meetings = validate_partition(
        train_manifest=train_source,
        validation_manifest=validation_source,
    )
    train_exclusions = preparation_exclusions(train_source)
    validation_exclusions = preparation_exclusions(validation_source)

    try:
        result = qualify_base(
            train_source=train_source,
            validation_source=validation_source,
            output_dir=output_dir,
            spec_path=spec_path,
            vocab_path=vocab_path,
        )
        if set(result["train"]["stats"]["meetings"]) != train_meetings:
            raise ValueError("qualified training set lost an expected meeting")
        if set(result["validation"]["stats"]["meetings"]) != validation_meetings:
            raise ValueError("qualified validation set lost an expected meeting")
        if result["train"]["stats"]["records"] <= 4429:
            raise ValueError("model-quality-v4 training population did not expand beyond v3")
        if result["train"]["stats"]["audio_seconds"] <= 7150.12:
            raise ValueError("model-quality-v4 training duration did not expand beyond v3")

        result["boundary"] = boundary_document(
            policy=policy,
            lock=lock,
            train_meetings=train_meetings,
            validation_meetings=validation_meetings,
            train_exclusions=train_exclusions,
            validation_exclusions=validation_exclusions,
        )
        qualification_path = output_dir / "qualification.json"
        qualification_path.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return result
    except Exception:
        for name in (
            "train.manifest.jsonl",
            "validation.manifest.jsonl",
            "qualification.json",
        ):
            try:
                (output_dir / name).unlink()
            except FileNotFoundError:
                pass
        raise


def verify_existing(
    *,
    policy_path: pathlib.Path,
    lock_dir: pathlib.Path,
    train_source: pathlib.Path,
    validation_source: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    policy = load_json(policy_path)
    validate_policy(policy)
    lock = validate_source_lock(policy=policy, lock_dir=lock_dir)
    train_meetings, validation_meetings = validate_partition(
        train_manifest=train_source,
        validation_manifest=validation_source,
    )
    train_exclusions = preparation_exclusions(train_source)
    validation_exclusions = preparation_exclusions(validation_source)

    qualification_path = output_dir / "qualification.json"
    train_path = output_dir / "train.manifest.jsonl"
    validation_path = output_dir / "validation.manifest.jsonl"
    for path in (qualification_path, train_path, validation_path):
        if not path.is_file():
            raise ValueError(f"model-quality-v4 qualified artifact missing: {path}")
    result = load_json(qualification_path)
    if result.get("schema") != "speech-asr/model-quality-manifests":
        raise ValueError("model-quality-v4 qualification schema mismatch")
    if result.get("version") != 2 or result.get("status") != "structurally_valid":
        raise ValueError("model-quality-v4 qualification is not structurally_valid")
    if result.get("sources", {}).get("train", {}).get("manifest_sha256") != sha256_path(
        train_source
    ):
        raise ValueError("model-quality-v4 training source hash mismatch")
    if result.get("sources", {}).get("validation", {}).get(
        "manifest_sha256"
    ) != sha256_path(validation_source):
        raise ValueError("model-quality-v4 validation source hash mismatch")
    if result.get("train", {}).get("manifest_sha256") != sha256_path(train_path):
        raise ValueError("model-quality-v4 qualified training hash mismatch")
    if result.get("validation", {}).get("manifest_sha256") != sha256_path(
        validation_path
    ):
        raise ValueError("model-quality-v4 qualified validation hash mismatch")
    if meetings(train_path) != train_meetings:
        raise ValueError("model-quality-v4 qualified training meeting set mismatch")
    if meetings(validation_path) != validation_meetings:
        raise ValueError("model-quality-v4 qualified validation meeting set mismatch")
    expected_boundary = boundary_document(
        policy=policy,
        lock=lock,
        train_meetings=train_meetings,
        validation_meetings=validation_meetings,
        train_exclusions=train_exclusions,
        validation_exclusions=validation_exclusions,
    )
    if result.get("boundary") != expected_boundary:
        raise ValueError("model-quality-v4 boundary metadata differs from policy")
    if result.get("train", {}).get("stats", {}).get("records", 0) <= 4429:
        raise ValueError("model-quality-v4 training record count is not expanded")
    if result.get("train", {}).get("stats", {}).get("audio_seconds", 0) <= 7150.12:
        raise ValueError("model-quality-v4 training duration is not expanded")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=pathlib.Path, default=DEFAULT_POLICY)
    parser.add_argument("--lock-dir", type=pathlib.Path, default=DEFAULT_LOCK_DIR)
    parser.add_argument("--train-source", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument(
        "--validation-source",
        type=pathlib.Path,
        default=DEFAULT_VALIDATION,
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    try:
        if args.verify_only:
            result = verify_existing(
                policy_path=args.policy,
                lock_dir=args.lock_dir,
                train_source=args.train_source,
                validation_source=args.validation_source,
                output_dir=args.output_dir,
            )
        else:
            result = qualify_v4(
                policy_path=args.policy,
                lock_dir=args.lock_dir,
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
