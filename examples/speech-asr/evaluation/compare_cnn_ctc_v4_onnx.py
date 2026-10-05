#!/usr/bin/env python3
"""Compare exported ONNX output against the PyTorch golden tensor."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.cnn_ctc_compare import compare_arrays  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    try:
        metadata = json.loads((args.export_dir / "export.json").read_text(encoding="utf-8"))
        input_meta = metadata["input"]
        output_meta = metadata["output"]
        input_shape = tuple(input_meta["shape"])
        output_shape = tuple(output_meta["shape"])
        features = np.fromfile(args.export_dir / "golden-input.f32", dtype=np.float32).reshape(input_shape)
        reference = np.fromfile(args.export_dir / "golden-output.f32", dtype=np.float32).reshape(output_shape)

        session = ort.InferenceSession(
            str(args.export_dir / "cnn_ctc_v4.onnx"),
            providers=["CPUExecutionProvider"],
        )
        candidate = session.run(
            [output_meta["name"]],
            {input_meta["name"]: features},
        )[0]
        comparison = compare_arrays(reference, candidate)
        result = {
            "schema": "speech-asr/cnn-ctc-tensor-comparison",
            "version": 1,
            "reference": "pytorch-golden",
            "candidate": "onnxruntime",
            "comparison": comparison,
            "runtime": {
                "onnxruntime_version": ort.__version__,
            },
            "acceptance": {
                "thresholds_applied": False,
                "note": "Evidence-only toolchain comparison; shape and finiteness are mandatory.",
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
