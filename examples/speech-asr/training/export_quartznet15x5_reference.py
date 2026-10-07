#!/usr/bin/env python3
"""Export the zero-training 29-class QuartzNet source reconstruction to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import sys

import numpy as np
import onnx
import torch

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))
sys.path.insert(0, str(HERE))

from quartznet15x5_reference import (  # noqa: E402
    DEFAULT_SPEC,
    DEFAULT_VOCAB,
    load_reference_model,
    load_reference_spec,
    load_reference_vocab,
)

DEFAULT_ARCHIVE = (
    ROOT
    / "work"
    / "speech-asr"
    / "pretrained"
    / "quartznet15x5-en-base-v2"
    / "QuartzNet15x5-En-Base.nemo"
)
DEFAULT_OUTPUT = (
    ROOT / "work" / "speech-asr" / "quartznet15x5-reference" / "export"
)


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=pathlib.Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fixed-time-frames",
        type=int,
        help="export a static time axis instead of the default dynamic time axis",
    )
    args = parser.parse_args()
    try:
        spec = load_reference_spec(args.spec)
        vocab = load_reference_vocab(args.vocab)
        model, imported = load_reference_model(args.archive, spec=spec, vocab=vocab)
        model.cpu().eval()

        time_frames = 512 if args.fixed_time_frames is None else int(args.fixed_time_frames)
        if time_frames < 16 or time_frames % int(spec["frontend"]["pad_to"]) != 0:
            raise ValueError(
                "--fixed-time-frames must be a positive multiple of frontend pad_to"
            )
        features = torch.linspace(
            -1.0,
            1.0,
            64 * time_frames,
            dtype=torch.float32,
        ).reshape(1, 64, time_frames)
        with torch.inference_mode():
            golden = model(features).detach().cpu().numpy().astype(np.float32)
        if golden.shape[-1] != 29:
            raise ValueError(f"reference model output width changed: {golden.shape}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = args.output_dir / "quartznet15x5_nvidia_ref.onnx"
        dynamic_axes = (
            {
                "features": {2: "feature_frames"},
                "logits": {1: "output_frames"},
            }
            if args.fixed_time_frames is None
            else None
        )
        torch.onnx.export(
            model,
            features,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )
        onnx.checker.check_model(onnx.load(str(onnx_path)))

        input_path = args.output_dir / "golden-input.f32"
        output_path = args.output_dir / "golden-output.f32"
        features.numpy().tofile(input_path)
        golden.tofile(output_path)
        metadata = {
            "schema": "speech-asr/quartznet-pretrained-reference-export",
            "version": 1,
            "model_id": spec["id"],
            "training_performed": False,
            "source": imported["source"],
            "input": {"name": "features", "shape": list(features.shape), "sha256": sha256_path(input_path)},
            "output": {"name": "logits", "shape": list(golden.shape), "sha256": sha256_path(output_path)},
            "onnx": {
                "path": str(onnx_path),
                "sha256": sha256_path(onnx_path),
                "opset": 11,
                "dynamic_time_axis": args.fixed_time_frames is None,
                "fixed_time_frames": (
                    None
                    if args.fixed_time_frames is None
                    else time_frames
                ),
            },
            "runtime": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "onnx_version": onnx.__version__,
            },
        }
        (args.output_dir / "export.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metadata, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
