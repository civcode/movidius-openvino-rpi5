#!/usr/bin/env python3
"""Prepare or verify a deterministic AMI subset."""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parents[1]
REPO_ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.ami import (  # noqa: E402
    acquire_sources,
    prepare_from_spec,
    validate_split_spec,
    verify_prepared_dataset,
)


def resolve_repo_path(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=("smoke", "benchmark"),
        default="smoke",
        help="version-1 split to prepare when --spec is not supplied",
    )
    parser.add_argument(
        "--spec",
        type=pathlib.Path,
        help="explicit split spec; overrides --subset",
    )
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=pathlib.Path("work/speech-asr/ami/downloads"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="output directory; defaults to work/speech-asr/ami/<split-id>",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an already prepared output without network access",
    )
    args = parser.parse_args()

    spec_path = (
        resolve_repo_path(args.spec)
        if args.spec is not None
        else HERE / "splits" / f"{args.subset}-v1.json"
    )
    with spec_path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    validate_split_spec(spec)

    output = (
        resolve_repo_path(args.output)
        if args.output is not None
        else REPO_ROOT / "work" / "speech-asr" / "ami" / spec["id"]
    )
    cache = resolve_repo_path(args.cache)

    if args.verify_only:
        result = verify_prepared_dataset(spec=spec, output_dir=output)
        print(json.dumps(result, sort_keys=True))
        return 0

    annotation, audio = acquire_sources(spec, cache)
    prepare_from_spec(
        spec=spec,
        annotation_zip_path=annotation,
        audio_paths=audio,
        output_dir=output,
    )
    result = verify_prepared_dataset(spec=spec, output_dir=output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
