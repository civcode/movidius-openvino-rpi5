#!/usr/bin/env python3
"""Prepare a deterministic AMI subset into normalized f32le clips + JSONL."""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.ami import acquire_sources, prepare_from_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=pathlib.Path,
        default=HERE / "splits" / "smoke-v1.json",
    )
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=pathlib.Path("work/speech-asr/ami/downloads"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("work/speech-asr/ami/smoke-v1"),
    )
    args = parser.parse_args()

    with args.spec.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    annotation, audio = acquire_sources(spec, args.cache)
    provenance = prepare_from_spec(
        spec=spec,
        annotation_zip_path=annotation,
        audio_paths=audio,
        output_dir=args.output,
    )
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
