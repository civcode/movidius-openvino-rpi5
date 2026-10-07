#!/usr/bin/env python3
"""Compare zero-training QuartzNet reference ONNX output with PyTorch."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.cnn_ctc_compare import compare_arrays  # noqa: E402

DEFAULT_EXPORT = ROOT / "work" / "speech-asr" / "quartznet15x5-reference" / "export"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=pathlib.Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        metadata = json.loads((args.export_dir / "export.json").read_text(encoding="utf-8"))
        input_shape = tuple(int(value) for value in metadata["input"]["shape"])
        output_shape = tuple(int(value) for value in metadata["output"]["shape"])
        features = np.fromfile(args.export_dir / "golden-input.f32", dtype=np.float32).reshape(input_shape)
        reference = np.fromfile(args.export_dir / "golden-output.f32", dtype=np.float32).reshape(output_shape)
        session = ort.InferenceSession(
            str(args.export_dir / "quartznet15x5_nvidia_ref.onnx"),
            providers=["CPUExecutionProvider"],
        )
        candidate = session.run(["logits"], {"features": features})[0]
        comparison = compare_arrays(reference, candidate)
        result = {
            "schema": "speech-asr/quartznet-pretrained-reference-onnx-parity",
            "version": 1,
            "status": "pass" if comparison["frame_argmax_agreement"] == 1.0 else "fail",
            "training_performed": False,
            "reference": "pytorch-source-reconstruction",
            "candidate": "onnxruntime",
            "comparison": comparison,
            "runtime": {"onnxruntime_version": ort.__version__},
        }
        text = json.dumps(result, sort_keys=True, indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text, end="")
        return 0 if result["status"] == "pass" else 3
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
