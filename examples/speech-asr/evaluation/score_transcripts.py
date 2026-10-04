#!/usr/bin/env python3
"""Score a reference/hypothesis pair with benchmark text-v1 rules."""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from speech_asr.evaluation import score_transcript  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    result = score_transcript(args.reference, args.hypothesis)
    if args.pretty:
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
