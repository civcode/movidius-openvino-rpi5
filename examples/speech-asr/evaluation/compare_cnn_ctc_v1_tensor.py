#!/usr/bin/env python3
"""Compare an f32 candidate tensor against cnn_ctc_v1 PyTorch golden output."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.cnn_ctc import load_spec  # noqa: E402
from speech_asr.cnn_ctc_compare import compare_arrays  # noqa: E402

DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v1" / "model_spec.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=pathlib.Path)
    parser.add_argument("candidate", type=pathlib.Path)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--candidate-name", default="candidate")
    args = parser.parse_args()
    try:
        spec = load_spec(args.spec)
        shape = tuple(int(value) for value in spec["output_contract"]["shape"])
        reference = np.fromfile(args.reference, dtype=np.float32)
        candidate = np.fromfile(args.candidate, dtype=np.float32)
        expected = int(np.prod(shape))
        if reference.size != expected or candidate.size != expected:
            raise ValueError(
                f"expected {expected} output elements; "
                f"reference={reference.size}, candidate={candidate.size}"
            )
        result = {
            "schema": "speech-asr/cnn-ctc-tensor-comparison",
            "version": 1,
            "reference": str(args.reference),
            "candidate": args.candidate_name,
            "comparison": compare_arrays(
                reference.reshape(shape),
                candidate.reshape(shape),
            ),
            "acceptance": {
                "thresholds_applied": False,
                "note": "Evidence-only device numerical comparison.",
            },
        }
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
