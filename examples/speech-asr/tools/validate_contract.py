#!/usr/bin/env python3
"""Validate a normalized speech sample or experiment result JSON document."""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from speech_asr.contracts import (  # noqa: E402
    ContractValidationError,
    canonical_json_sha256,
    validate_experiment_result,
    validate_speech_sample,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=pathlib.Path)
    parser.add_argument("--kind", choices=("auto", "sample", "result"), default="auto")
    args = parser.parse_args()

    try:
        with args.document.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    kind = args.kind
    if kind == "auto":
        schema = document.get("schema") if isinstance(document, dict) else None
        if schema == "speech-asr/sample":
            kind = "sample"
        elif schema == "speech-asr/experiment-result":
            kind = "result"
        else:
            print("error: cannot infer contract kind from $.schema", file=sys.stderr)
            return 2

    try:
        if kind == "sample":
            validate_speech_sample(document)
        else:
            validate_experiment_result(document)
    except ContractValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"valid {kind} sha256={canonical_json_sha256(document)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
