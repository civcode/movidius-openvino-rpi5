#!/usr/bin/env python3
"""Diff two OpenVINO IRs after conservative graph canonicalization."""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import sys
from xml.etree import ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from speech_asr.openvino_ir import canonical_graph_document  # noqa: E402


def render(path: pathlib.Path) -> list[str]:
    root = ET.parse(path).getroot()
    text = json.dumps(
        canonical_graph_document(root),
        sort_keys=True,
        indent=2,
    )
    return text.splitlines()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=pathlib.Path)
    parser.add_argument("right", type=pathlib.Path)
    args = parser.parse_args()

    left = render(args.left)
    right = render(args.right)
    diff = list(difflib.unified_diff(
        left,
        right,
        fromfile=str(args.left),
        tofile=str(args.right),
        lineterm="",
    ))
    if not diff:
        print("canonical executable graphs are identical")
        return 0
    print("\n".join(diff))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
