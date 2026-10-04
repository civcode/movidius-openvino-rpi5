#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from speech_asr.openvino_ir import inspect_ir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", type=pathlib.Path)
    parser.add_argument("--bin", dest="bin_path", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    result = inspect_ir(args.xml, args.bin_path)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
