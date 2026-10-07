#!/usr/bin/env python3
"""Verify the pinned QuartzNet archive can be imported into cnn_ctc_v18."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
sys.path.insert(0, str(SPEECH_ROOT / "python"))
sys.path.insert(0, str(HERE))

from cnn_ctc_v18 import CnnCtcV18  # noqa: E402
from pretrained_cnn_ctc_v18 import load_pretrained_quartznet  # noqa: E402
from speech_asr.cnn_ctc import load_spec, load_vocab  # noqa: E402

DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v18" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v18" / "vocab.json"


def state_sha256(model: torch.nn.Module) -> str:
    """Fingerprint transferred acoustic state, excluding the task-specific head."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("projection."):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=pathlib.Path)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    try:
        spec = load_spec(args.spec)
        vocab = load_vocab(args.vocab)
        torch.manual_seed(int(spec["training"]["seed"]))
        model = CnnCtcV18(spec, len(vocab["tokens"]))
        imported = load_pretrained_quartznet(
            model,
            args.archive,
            list(vocab["tokens"]),
        )

        if imported["shared_decoder_symbol_count"] != 29:
            raise ValueError(
                "pretrained decoder import did not map all 29 source symbols"
            )
        if imported["target_only_symbols_random_init"] != list("0123456789"):
            raise ValueError(
                "cnn_ctc_v18 target-only projection rows must be digits 0-9"
            )
        if imported["encoder_source_tensors_used"] < 500:
            raise ValueError(
                "pretrained encoder import consumed too few source tensors: "
                f"{imported['encoder_source_tensors_used']}"
            )

        model.eval()
        shape = tuple(int(value) for value in spec["input_contract"]["shape"])
        with torch.no_grad():
            logits = model(torch.zeros(shape, dtype=torch.float32))
        expected = tuple(int(value) for value in spec["output_contract"]["shape"])
        if tuple(logits.shape) != expected:
            raise ValueError(
                f"pretrained import output shape {tuple(logits.shape)} != {expected}"
            )
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("pretrained import produced non-finite logits")

        result = {
            "schema": "speech-asr/pretrained-import-verification",
            "version": 1,
            "status": "valid",
            "model_id": spec["id"],
            "source": imported["source"],
            "encoder_source_tensors_used": imported[
                "encoder_source_tensors_used"
            ],
            "decoder_source_tensors_used": imported[
                "decoder_source_tensors_used"
            ],
            "shared_decoder_symbol_count": imported[
                "shared_decoder_symbol_count"
            ],
            "target_only_symbols_random_init": imported[
                "target_only_symbols_random_init"
            ],
            "acoustic_state_sha256": state_sha256(model),
            "output_shape": list(logits.shape),
            "finite_output": True,
        }
        payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
