"""cnn_ctc_v1 model-package utilities that do not depend on PyTorch."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from .text import normalize_text_v1


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_spec(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "speech-asr/custom-model-spec":
        raise ValueError("model spec schema must be speech-asr/custom-model-spec")
    if value.get("version") != 1:
        raise ValueError("model spec version must be 1")
    if value.get("id") != "cnn_ctc_v1":
        raise ValueError("model spec id must be cnn_ctc_v1")

    frontend = value.get("frontend")
    if not isinstance(frontend, dict) or frontend.get("kind") != "logmel-v1":
        raise ValueError("cnn_ctc_v1 frontend must be logmel-v1")
    expected = {
        "sample_rate_hz": 16000,
        "window_samples": 400,
        "hop_samples": 160,
        "fft_size": 512,
        "mel_bins": 64,
        "fixed_frames": 512,
        "fixed_audio_samples": 82160,
    }
    for key, wanted in expected.items():
        if frontend.get(key) != wanted:
            raise ValueError(f"frontend.{key} must be {wanted!r}")

    input_contract = value.get("input_contract")
    if not isinstance(input_contract, dict) or input_contract.get("shape") != [1, 64, 512]:
        raise ValueError("cnn_ctc_v1 input shape must be [1,64,512]")
    output_contract = value.get("output_contract")
    if not isinstance(output_contract, dict) or output_contract.get("shape") != [1, 128, 39]:
        raise ValueError("cnn_ctc_v1 output shape must be [1,128,39]")
    export = value.get("export")
    if not isinstance(export, dict) or export.get("onnx_opset") != 11:
        raise ValueError("cnn_ctc_v1 ONNX opset must be 11")
    return value


def load_vocab(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "speech-asr/ctc-vocab" or value.get("version") != 1:
        raise ValueError("invalid CTC vocabulary contract")
    tokens = value.get("tokens")
    if not isinstance(tokens, list) or not tokens or tokens[0] != "<blank>":
        raise ValueError("CTC vocabulary must start with <blank>")
    if len(tokens) != len(set(tokens)):
        raise ValueError("CTC vocabulary tokens must be unique")
    if value.get("blank_index") != 0:
        raise ValueError("cnn_ctc_v1 blank index must be 0")
    return value


def encode_text(text: str, vocab: dict) -> tuple[int, ...]:
    normalized = normalize_text_v1(text)
    index = {token: i for i, token in enumerate(vocab["tokens"])}
    encoded = []
    for char in normalized:
        if char not in index:
            raise ValueError(f"character is not in cnn_ctc_v1 vocabulary: {char!r}")
        encoded.append(index[char])
    return tuple(encoded)


def greedy_decode(indices: Iterable[int], vocab: dict) -> str:
    tokens = vocab["tokens"]
    blank = vocab["blank_index"]
    output: list[str] = []
    previous = None
    for raw in indices:
        value = int(raw)
        if value < 0 or value >= len(tokens):
            raise ValueError(f"CTC index out of range: {value}")
        if value != blank and value != previous:
            output.append(tokens[value])
        previous = value
    return normalize_text_v1("".join(output))


def greedy_decode_logits(logits: Sequence[Sequence[float]], vocab: dict) -> str:
    indices = []
    width = len(vocab["tokens"])
    for frame_index, frame in enumerate(logits):
        if len(frame) != width:
            raise ValueError(
                f"logit frame {frame_index} has width {len(frame)}, expected {width}"
            )
        best = max(range(width), key=lambda index: float(frame[index]))
        indices.append(best)
    return greedy_decode(indices, vocab)


def conv1d_output_length(
    length: int,
    *,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> int:
    if length < 1:
        return 0
    return math.floor(
        (length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    )


def acoustic_output_length(feature_frames: int, spec: dict) -> int:
    length = int(feature_frames)
    network = spec["network"]
    for kernel, stride, padding in zip(
        network["kernels"],
        network["strides"],
        network["paddings"],
    ):
        length = conv1d_output_length(
            length,
            kernel=int(kernel),
            stride=int(stride),
            padding=int(padding),
        )
    return length
