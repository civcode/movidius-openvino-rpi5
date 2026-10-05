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
    model_id = value.get("id")
    if model_id not in {"cnn_ctc_v1", "cnn_ctc_v2", "cnn_ctc_v3"}:
        raise ValueError("unsupported cnn_ctc model spec id")

    frontend = value.get("frontend")
    if not isinstance(frontend, dict) or frontend.get("kind") != "logmel-v1":
        raise ValueError(f"{model_id} frontend must be logmel-v1")
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
        raise ValueError(f"{model_id} input shape must be [1,64,512]")
    output_contract = value.get("output_contract")
    if not isinstance(output_contract, dict) or output_contract.get("shape") != [1, 128, 39]:
        raise ValueError(f"{model_id} output shape must be [1,128,39]")
    export = value.get("export")
    if not isinstance(export, dict) or export.get("onnx_opset") != 11:
        raise ValueError(f"{model_id} ONNX opset must be 11")

    network = value.get("network")
    if not isinstance(network, dict):
        raise ValueError(f"{model_id} network must be an object")
    if model_id == "cnn_ctc_v1":
        for key in ("channels", "kernels", "strides", "paddings"):
            values = network.get(key)
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError(f"cnn_ctc_v1 network.{key} must contain three values")
    else:
        expected_kind = (
            "residual-temporal-v1"
            if model_id == "cnn_ctc_v2"
            else "residual-temporal-v2"
        )
        if network.get("kind") != expected_kind:
            raise ValueError(f"{model_id} network.kind must be {expected_kind}")
        stem = network.get("stem")
        blocks = network.get("residual_blocks")
        if not isinstance(stem, list) or len(stem) != 2:
            raise ValueError(f"{model_id} network.stem must contain two layers")
        if not isinstance(blocks, list) or len(blocks) != 5:
            raise ValueError(f"{model_id} residual_blocks must contain five blocks")
        expected_norm = "batchnorm" if model_id == "cnn_ctc_v2" else "none"
        if network.get("normalization") != expected_norm:
            raise ValueError(f"{model_id} normalization must be {expected_norm}")
        if network.get("activation") != "relu":
            raise ValueError(f"{model_id} activation must be relu")
        if [int(block.get("kernel", 0)) for block in blocks] != [11, 19, 27, 35, 43]:
            raise ValueError(f"{model_id} residual kernel schedule is frozen")
        if any(int(block.get("channels", 0)) != 96 for block in blocks):
            raise ValueError(f"{model_id} residual channels must remain 96")
        if (
            model_id == "cnn_ctc_v3"
            and network.get("residual_projection_init") != "kaiming_scaled_0.01"
        ):
            raise ValueError(
                "cnn_ctc_v3 residual projection init must be kaiming_scaled_0.01"
            )
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
        raise ValueError("cnn_ctc blank index must be 0")
    return value


def encode_text(text: str, vocab: dict) -> tuple[int, ...]:
    normalized = normalize_text_v1(text)
    index = {token: i for i, token in enumerate(vocab["tokens"])}
    encoded = []
    for char in normalized:
        if char not in index:
            raise ValueError(f"character is not in cnn_ctc vocabulary: {char!r}")
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
    if spec.get("id") in {"cnn_ctc_v2", "cnn_ctc_v3"}:
        layers = network["stem"]
        for layer in layers:
            length = conv1d_output_length(
                length,
                kernel=int(layer["kernel"]),
                stride=int(layer["stride"]),
                padding=int(layer["padding"]),
            )
        return length

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


def model_resource_estimate(spec: dict, vocab_size: int = 39) -> dict:
    """Return deterministic parameter/MAC/receptive-field estimates for fixed input."""
    if spec.get("id") not in {"cnn_ctc_v2", "cnn_ctc_v3"}:
        raise ValueError("resource estimator targets cnn_ctc_v2/v3")

    network = spec["network"]
    input_frames = int(spec["input_contract"]["shape"][2])
    in_channels = int(spec["input_contract"]["shape"][1])
    length = input_frames
    receptive_field = 1
    jump = 1
    parameters = 0
    macs = 0

    for layer in network["stem"]:
        out_channels = int(layer["channels"])
        kernel = int(layer["kernel"])
        stride = int(layer["stride"])
        padding = int(layer["padding"])
        out_length = conv1d_output_length(
            length,
            kernel=kernel,
            stride=stride,
            padding=padding,
        )
        use_batchnorm = spec["network"]["normalization"] == "batchnorm"
        parameters += in_channels * out_channels * kernel
        if use_batchnorm:
            parameters += 2 * out_channels  # BatchNorm gamma/beta
        else:
            parameters += out_channels  # Conv bias
        macs += in_channels * out_channels * kernel * out_length
        receptive_field += (kernel - 1) * jump
        jump *= stride
        length = out_length
        in_channels = out_channels

    for block in network["residual_blocks"]:
        channels = int(block["channels"])
        kernel = int(block["kernel"])
        projection_kernel = int(block["projection_kernel"])
        if channels != in_channels:
            raise ValueError("generation-1 residual block changes channel width")
        parameters += channels * channels * kernel
        parameters += 2 * channels if use_batchnorm else channels
        parameters += channels * channels * projection_kernel
        parameters += 2 * channels if use_batchnorm else channels
        macs += channels * channels * kernel * length
        macs += channels * channels * projection_kernel * length
        receptive_field += (kernel - 1) * jump

    projection_kernel = int(network["projection_kernel"])
    parameters += in_channels * vocab_size * projection_kernel + vocab_size
    macs += in_channels * vocab_size * projection_kernel * length

    return {
        "parameters": parameters,
        "macs_fixed_input": macs,
        "receptive_field_feature_frames": receptive_field,
        "output_frames": length,
        "fp16_weight_bytes": parameters * 2,
    }


def manifest_record_eligibility(record: dict, spec: dict, vocab: dict) -> dict:
    """Return the fixed-shape CTC eligibility decision for one manifest record."""
    sample_count = int(record["audio"]["end_sample"]) - int(record["audio"]["start_sample"])
    fixed_audio = int(spec["frontend"]["fixed_audio_samples"])
    window = int(spec["frontend"]["window_samples"])
    hop = int(spec["frontend"]["hop_samples"])
    fixed_frames = int(spec["frontend"]["fixed_frames"])

    if sample_count > fixed_audio:
        return {
            "eligible": False,
            "reason": "too_long",
            "sample_count": sample_count,
            "valid_feature_frames": None,
            "valid_output_frames": None,
            "target_length": None,
        }

    valid_frames = (
        1
        if sample_count < window
        else min(fixed_frames, 1 + (sample_count - window) // hop)
    )
    output_frames = acoustic_output_length(valid_frames, spec)
    target_length = len(encode_text(record["transcript"]["text"], vocab))
    if target_length > output_frames:
        return {
            "eligible": False,
            "reason": "target_too_long",
            "sample_count": sample_count,
            "valid_feature_frames": valid_frames,
            "valid_output_frames": output_frames,
            "target_length": target_length,
        }

    return {
        "eligible": True,
        "reason": None,
        "sample_count": sample_count,
        "valid_feature_frames": valid_frames,
        "valid_output_frames": output_frames,
        "target_length": target_length,
    }
