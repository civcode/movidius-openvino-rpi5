#!/usr/bin/env python3
"""Export cnn_ctc_v15 checkpoint to fixed-shape ONNX plus golden tensors."""

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

from cnn_ctc_v15 import CnnCtcV15, parameter_count  # noqa: E402
from speech_asr.cnn_ctc import canonical_sha256, load_spec, load_vocab  # noqa: E402

DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v15" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v15" / "vocab.json"


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "cnn_ctc_v15" / "export",
    )
    args = parser.parse_args()

    try:
        spec = load_spec(args.spec)
        vocab = load_vocab(args.vocab)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if checkpoint.get("model_id") != spec["id"]:
            raise ValueError("checkpoint model_id does not match model spec")
        if checkpoint.get("spec_sha256") != canonical_sha256(spec):
            raise ValueError("checkpoint model spec hash does not match current spec")
        if checkpoint.get("vocab_sha256") != canonical_sha256(vocab):
            raise ValueError("checkpoint vocab hash does not match current vocab")

        model = CnnCtcV15(spec, len(vocab["tokens"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        input_shape = tuple(int(value) for value in spec["input_contract"]["shape"])
        total = int(np.prod(input_shape))
        values = np.linspace(-1.0, 1.0, total, dtype=np.float32).reshape(input_shape)
        features = torch.from_numpy(values)
        with torch.no_grad():
            golden = model(features).detach().cpu().numpy().astype(np.float32)

        expected_output = tuple(int(value) for value in spec["output_contract"]["shape"])
        if tuple(golden.shape) != expected_output:
            raise ValueError(
                f"PyTorch output shape {tuple(golden.shape)} != spec {expected_output}"
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = args.output_dir / "cnn_ctc_v15.onnx"
        torch.onnx.export(
            model,
            features,
            onnx_path,
            export_params=True,
            opset_version=int(spec["export"]["onnx_opset"]),
            do_constant_folding=True,
            input_names=[spec["input_contract"]["name"]],
            output_names=[spec["output_contract"]["name"]],
            dynamic_axes=None,
        )
        loaded = onnx.load(str(onnx_path))
        onnx.checker.check_model(loaded)

        golden_input = args.output_dir / "golden-input.f32"
        golden_output = args.output_dir / "golden-output.f32"
        values.tofile(golden_input)
        golden.tofile(golden_output)
        metadata = {
            "schema": "speech-asr/cnn-ctc-export",
            "version": 1,
            "model_id": spec["id"],
            "onnx_opset": int(spec["export"]["onnx_opset"]),
            "parameter_count": parameter_count(model),
            "input": {
                "name": spec["input_contract"]["name"],
                "shape": list(input_shape),
                "dtype": "float32",
                "path": str(golden_input),
                "sha256": sha256_path(golden_input),
            },
            "output": {
                "name": spec["output_contract"]["name"],
                "shape": list(expected_output),
                "dtype": "float32",
                "path": str(golden_output),
                "sha256": sha256_path(golden_output),
            },
            "artifacts": {
                "checkpoint_sha256": sha256_path(args.checkpoint),
                "onnx_sha256": sha256_path(onnx_path),
                "spec_sha256": canonical_sha256(spec),
                "vocab_sha256": canonical_sha256(vocab),
            },
            "runtime": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "onnx_version": onnx.__version__,
            },
        }
        metadata_path = args.output_dir / "export.json"
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
