#!/usr/bin/env python3
"""Freeze source identities for the model-quality-v4 AMI development boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import urllib.request
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parents[1]
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.ami import (  # noqa: E402
    download_verified,
    sha256_file,
    validate_split_spec,
)
from speech_asr.contracts import canonical_json_sha256  # noqa: E402

POLICY = HERE / "splits" / "model-quality-v4-edinburgh-v1.json"
KNOWN_PINNED_TRAIN_SPLIT = HERE / "splits" / "train-es2005-es2007-v1.json"
DEFAULT_CACHE = ROOT / "work" / "speech-asr" / "ami" / "downloads"
DEFAULT_OUTPUT = (
    ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4-source-lock"
)

OFFICIAL_FULL_CORPUS_ASR_ES_TRAIN = {
    "ES2002",
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
EXPECTED_TRAIN_GROUPS = OFFICIAL_FULL_CORPUS_ASR_ES_TRAIN - {"ES2002"}
EXPECTED_VALIDATION_GROUPS = {"ES2011"}
EXPECTED_SEALED_TEST_GROUPS = {"EN2002", "ES2004", "IS1009", "TS3003"}
EXPECTED_SPEAKERS = ["A", "B", "C", "D"]

TRAIN_SPLIT_ID = "ami-model-quality-v4-train-v1"
VALIDATION_SPLIT_ID = "ami-model-quality-v4-validation-v1"


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_groups(groups: list[str]) -> list[str]:
    return [
        f"{group}{suffix}"
        for group in groups
        for suffix in ("a", "b", "c", "d")
    ]


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema") != "speech-asr/model-quality-data-policy":
        raise ValueError("model-quality-v4 policy schema mismatch")
    if policy.get("version") != 1:
        raise ValueError("model-quality-v4 policy version mismatch")
    if policy.get("id") != "ami-model-quality-v4-edinburgh-full-corpus-asr-v1":
        raise ValueError("model-quality-v4 policy id mismatch")
    authority = policy.get("authority", {})
    if authority.get("partition_name") != "Full-corpus-ASR":
        raise ValueError("model-quality-v4 must use the official Full-corpus-ASR partition")
    if set(policy.get("train_groups", [])) != EXPECTED_TRAIN_GROUPS:
        raise ValueError("model-quality-v4 train groups differ from reviewed policy")
    if set(policy.get("validation_groups", [])) != EXPECTED_VALIDATION_GROUPS:
        raise ValueError("model-quality-v4 validation group must remain ES2011")
    if set(policy.get("excluded_prior_selection_groups", [])) != {"ES2002"}:
        raise ValueError("model-quality-v4 must retire ES2002 from selection")
    if set(policy.get("sealed_test_groups", [])) != EXPECTED_SEALED_TEST_GROUPS:
        raise ValueError("model-quality-v4 sealed-test exclusions changed")
    if policy.get("speakers") != EXPECTED_SPEAKERS:
        raise ValueError("model-quality-v4 speaker selection changed")
    audio = policy.get("audio", {})
    if audio.get("stream") != "Mix-Headset":
        raise ValueError("model-quality-v4 must use Mix-Headset audio")
    if "{meeting}" not in str(audio.get("url_template", "")):
        raise ValueError("model-quality-v4 audio URL template is invalid")
    annotations = policy.get("annotations", {})
    if annotations.get("sha256") != (
        "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"
    ):
        raise ValueError("model-quality-v4 annotation identity changed")
    rules = policy.get("policy", {})
    expected_rules = {
        "training_allowed_on_train": True,
        "training_allowed_on_validation": False,
        "checkpoint_selection_allowed_on_validation": True,
        "heldout_metrics_allowed_for_model_selection": False,
    }
    if rules != expected_rules:
        raise ValueError("model-quality-v4 role policy changed")

    train_meetings = set(expand_groups(policy["train_groups"]))
    validation_meetings = set(expand_groups(policy["validation_groups"]))
    forbidden = {
        f"{group}{suffix}"
        for group in (
            *policy["excluded_prior_selection_groups"],
            *policy["sealed_test_groups"],
        )
        for suffix in ("a", "b", "c", "d")
    }
    if train_meetings & validation_meetings:
        raise ValueError("model-quality-v4 train/validation meetings overlap")
    if (train_meetings | validation_meetings) & forbidden:
        raise ValueError("model-quality-v4 development boundary touches forbidden meetings")


def fetch_official_for_freeze(url: str, destination: pathlib.Path) -> pathlib.Path:
    """Fetch official bytes once and freeze their digest; never trust an unpinned cache."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".official.part",
        dir=destination.parent,
    )
    os.close(fd)
    tmp = pathlib.Path(tmp_name)
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "movidius-openvino-rpi5-speech-asr/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as output:
            shutil.copyfileobj(response, output)
        official_sha = sha256_file(tmp)
        if destination.exists():
            cached_sha = sha256_file(destination)
            if cached_sha != official_sha:
                raise ValueError(
                    f"cached source differs from official first acquisition for "
                    f"{destination.name}: {cached_sha} != {official_sha}"
                )
        else:
            tmp.replace(destination)
    finally:
        if tmp.exists():
            tmp.unlink()
    return destination


def source_entry(policy: dict[str, Any], meeting: str, sha256: str) -> dict[str, Any]:
    filename = f"{meeting}.Mix-Headset.wav"
    url = str(policy["audio"]["url_template"]).format(meeting=meeting)
    return {
        "meeting": meeting,
        "audio": {
            "stream": "Mix-Headset",
            "filename": filename,
            "url": url,
            "sha256": sha256,
        },
        "selections": [
            {"speaker": speaker, "all_segments": True}
            for speaker in EXPECTED_SPEAKERS
        ],
    }


def make_split(
    policy: dict[str, Any],
    *,
    split_id: str,
    role: str,
    meetings: list[str],
    audio_sha256: dict[str, str],
) -> dict[str, Any]:
    value = {
        "schema": "speech-asr/ami-split",
        "version": 1,
        "id": split_id,
        "license": "CC-BY-4.0",
        "annotations": dict(policy["annotations"]),
        "partition": {
            "name": "Full-corpus-ASR",
            "role": role,
            "policy_id": policy["id"],
        },
        "sources": [
            source_entry(policy, meeting, audio_sha256[meeting])
            for meeting in meetings
        ],
    }
    validate_split_spec(value)
    return value


def write_json_once(path: pathlib.Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current != encoded:
            raise ValueError(f"frozen source identity differs from existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def verify_existing(
    *,
    policy: dict[str, Any],
    cache: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, Any]:
    train_path = output / "train.split.json"
    validation_path = output / "validation.split.json"
    lock_path = output / "source-lock.json"
    for path in (train_path, validation_path, lock_path):
        if not path.is_file():
            raise ValueError(f"frozen model-quality-v4 source artifact missing: {path}")

    lock = load_json(lock_path)
    if lock.get("schema") != "speech-asr/ami-source-lock" or lock.get("version") != 1:
        raise ValueError("model-quality-v4 source lock schema mismatch")
    policy_sha = canonical_json_sha256(policy)
    if lock.get("policy_sha256") != policy_sha:
        raise ValueError("model-quality-v4 source lock policy hash mismatch")

    annotations = policy["annotations"]
    annotation_path = cache / annotations["filename"]
    if not annotation_path.is_file():
        raise ValueError(f"cached AMI annotations missing: {annotation_path}")
    if sha256_file(annotation_path) != annotations["sha256"]:
        raise ValueError("cached AMI annotation hash mismatch")

    audio_sha = lock.get("audio_sha256")
    if not isinstance(audio_sha, dict):
        raise ValueError("model-quality-v4 source lock lacks audio hashes")
    train_meetings = expand_groups(policy["train_groups"])
    validation_meetings = expand_groups(policy["validation_groups"])
    all_meetings = train_meetings + validation_meetings
    if set(audio_sha) != set(all_meetings):
        raise ValueError("model-quality-v4 source lock meeting set mismatch")
    for meeting in all_meetings:
        path = cache / f"{meeting}.Mix-Headset.wav"
        if not path.is_file():
            raise ValueError(f"cached AMI source missing: {path}")
        actual = sha256_file(path)
        if actual != audio_sha[meeting]:
            raise ValueError(
                f"cached AMI source hash mismatch for {meeting}: "
                f"{actual} != {audio_sha[meeting]}"
            )

    train = load_json(train_path)
    validation = load_json(validation_path)
    expected_train = make_split(
        policy,
        split_id=TRAIN_SPLIT_ID,
        role="SA-training-development",
        meetings=train_meetings,
        audio_sha256=audio_sha,
    )
    expected_validation = make_split(
        policy,
        split_id=VALIDATION_SPLIT_ID,
        role="SB-development-checkpoint-selection",
        meetings=validation_meetings,
        audio_sha256=audio_sha,
    )
    if train != expected_train:
        raise ValueError("frozen model-quality-v4 training split differs from source lock")
    if validation != expected_validation:
        raise ValueError("frozen model-quality-v4 validation split differs from source lock")

    return {
        "schema": "speech-asr/ami-source-lock-verification",
        "version": 1,
        "status": "valid",
        "policy_sha256": policy_sha,
        "source_lock_sha256": sha256_path(lock_path),
        "train_split_sha256": sha256_path(train_path),
        "validation_split_sha256": sha256_path(validation_path),
        "train_meetings": len(train_meetings),
        "validation_meetings": len(validation_meetings),
    }


def freeze(
    *,
    policy: dict[str, Any],
    cache: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, Any]:
    train_path = output / "train.split.json"
    validation_path = output / "validation.split.json"
    lock_path = output / "source-lock.json"
    existing = [path.exists() for path in (train_path, validation_path, lock_path)]
    if any(existing):
        if not all(existing):
            raise ValueError(
                "model-quality-v4 source lock is partially present; refuse to overwrite"
            )
        return verify_existing(policy=policy, cache=cache, output=output)

    annotations = policy["annotations"]
    download_verified(
        annotations["url"],
        cache / annotations["filename"],
        annotations["sha256"],
    )

    known_split = load_json(KNOWN_PINNED_TRAIN_SPLIT)
    known_audio_sha = {
        str(source["meeting"]): str(source["audio"]["sha256"])
        for source in known_split.get("sources", [])
    }

    train_meetings = expand_groups(policy["train_groups"])
    validation_meetings = expand_groups(policy["validation_groups"])
    audio_sha: dict[str, str] = {}
    for meeting in train_meetings + validation_meetings:
        filename = f"{meeting}.Mix-Headset.wav"
        url = str(policy["audio"]["url_template"]).format(meeting=meeting)
        if meeting in known_audio_sha:
            path = download_verified(
                url,
                cache / filename,
                known_audio_sha[meeting],
            )
        else:
            path = fetch_official_for_freeze(url, cache / filename)
        digest = sha256_file(path)
        audio_sha[meeting] = digest
        print(f"[model-quality-v4] source {meeting} sha256={digest}", flush=True)

    train = make_split(
        policy,
        split_id=TRAIN_SPLIT_ID,
        role="SA-training-development",
        meetings=train_meetings,
        audio_sha256=audio_sha,
    )
    validation = make_split(
        policy,
        split_id=VALIDATION_SPLIT_ID,
        role="SB-development-checkpoint-selection",
        meetings=validation_meetings,
        audio_sha256=audio_sha,
    )
    lock = {
        "schema": "speech-asr/ami-source-lock",
        "version": 1,
        "status": "frozen",
        "acquisition": "official-https-first-acquisition",
        "policy_id": policy["id"],
        "policy_sha256": canonical_json_sha256(policy),
        "annotations_sha256": annotations["sha256"],
        "audio_sha256": audio_sha,
        "train_split_id": TRAIN_SPLIT_ID,
        "validation_split_id": VALIDATION_SPLIT_ID,
    }
    write_json_once(train_path, train)
    write_json_once(validation_path, validation)
    write_json_once(lock_path, lock)
    return verify_existing(policy=policy, cache=cache, output=output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=pathlib.Path, default=POLICY)
    parser.add_argument("--cache", type=pathlib.Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    try:
        policy = load_json(args.policy)
        validate_policy(policy)
        if args.verify_only:
            result = verify_existing(
                policy=policy,
                cache=args.cache,
                output=args.output_dir,
            )
        else:
            result = freeze(
                policy=policy,
                cache=args.cache,
                output=args.output_dir,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
