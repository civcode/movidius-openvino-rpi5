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
    if model_id not in {
        "cnn_ctc_v1",
        "cnn_ctc_v2",
        "cnn_ctc_v3",
        "cnn_ctc_v4",
        "cnn_ctc_v5",
        "cnn_ctc_v6",
        "cnn_ctc_v7",
        "cnn_ctc_v8",
        "cnn_ctc_v9",
        "cnn_ctc_v10",
    }:
        raise ValueError("unsupported cnn_ctc model spec id")

    frontend = value.get("frontend")
    expected_frontend_kind = "logmel-v2" if model_id == "cnn_ctc_v4" else "logmel-v1"
    if (
        not isinstance(frontend, dict)
        or frontend.get("kind") != expected_frontend_kind
    ):
        raise ValueError(
            f"{model_id} frontend must be {expected_frontend_kind}"
        )
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
    expected_normalization = (
        "per_mel_bin_mean_valid_zero_pad"
        if model_id == "cnn_ctc_v4"
        else "per_mel_bin_mean"
    )
    if frontend.get("normalization") != expected_normalization:
        raise ValueError(
            f"{model_id} frontend.normalization must be "
            f"{expected_normalization}"
        )

    input_contract = value.get("input_contract")
    if not isinstance(input_contract, dict) or input_contract.get("shape") != [1, 64, 512]:
        raise ValueError(f"{model_id} input shape must be [1,64,512]")
    output_contract = value.get("output_contract")
    expected_output_shape = [1, 256, 39] if model_id == "cnn_ctc_v8" else [1, 128, 39]
    if not isinstance(output_contract, dict) or output_contract.get("shape") != expected_output_shape:
        raise ValueError(f"{model_id} output shape must be {expected_output_shape}")
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
        expected_kind = {
            "cnn_ctc_v2": "residual-temporal-v1",
            "cnn_ctc_v3": "residual-temporal-v2",
            "cnn_ctc_v4": "residual-temporal-v2",
            "cnn_ctc_v5": "residual-temporal-v3",
            "cnn_ctc_v6": "residual-temporal-v4",
            "cnn_ctc_v7": "residual-temporal-v2",
            "cnn_ctc_v8": "residual-temporal-v5",
            "cnn_ctc_v9": "residual-temporal-v6",
            "cnn_ctc_v10": "residual-temporal-v7",
        }[model_id]
        if network.get("kind") != expected_kind:
            raise ValueError(f"{model_id} network.kind must be {expected_kind}")
        stem = network.get("stem")
        blocks = network.get("residual_blocks")
        if not isinstance(stem, list) or len(stem) != 2:
            raise ValueError(f"{model_id} network.stem must contain two layers")
        expected_block_count = 8 if model_id in {"cnn_ctc_v9", "cnn_ctc_v10"} else 5
        if not isinstance(blocks, list) or len(blocks) != expected_block_count:
            raise ValueError(
                f"{model_id} residual_blocks must contain {expected_block_count} blocks"
            )
        expected_stem_channels = (
            [48, 64]
            if model_id == "cnn_ctc_v5"
            else ([64, 112] if model_id in {"cnn_ctc_v7", "cnn_ctc_v10"} else ([64, 72] if model_id == "cnn_ctc_v8" else [64, 96]))
        )
        if [int(layer.get("channels", 0)) for layer in stem] != expected_stem_channels:
            raise ValueError(
                f"{model_id} stem channels must remain {expected_stem_channels}"
            )
        expected_norm = "batchnorm" if model_id == "cnn_ctc_v2" else "none"
        if network.get("normalization") != expected_norm:
            raise ValueError(f"{model_id} normalization must be {expected_norm}")
        if network.get("activation") != "relu":
            raise ValueError(f"{model_id} activation must be relu")
        expected_kernels = (
            [9, 9, 13, 13, 17]
            if model_id == "cnn_ctc_v5"
            else ([7] * 8 if model_id in {"cnn_ctc_v9", "cnn_ctc_v10"} else [11, 19, 27, 35, 43])
        )
        if [int(block.get("kernel", 0)) for block in blocks] != expected_kernels:
            raise ValueError(f"{model_id} residual kernel schedule is frozen")
        expected_channels = 64 if model_id == "cnn_ctc_v5" else (112 if model_id in {"cnn_ctc_v7", "cnn_ctc_v10"} else (72 if model_id == "cnn_ctc_v8" else 96))
        if any(
            int(block.get("channels", 0)) != expected_channels for block in blocks
        ):
            raise ValueError(
                f"{model_id} residual channels must remain {expected_channels}"
            )
        if (
            model_id in {
                "cnn_ctc_v3",
                "cnn_ctc_v4",
                "cnn_ctc_v5",
                "cnn_ctc_v6",
                "cnn_ctc_v7",
                "cnn_ctc_v8",
        "cnn_ctc_v9",
        "cnn_ctc_v10",
            }
            and network.get("residual_projection_init") != "kaiming_scaled_0.01"
        ):
            raise ValueError(
                f"{model_id} residual projection init must be kaiming_scaled_0.01"
            )
        if model_id == "cnn_ctc_v8":
            expected_strides = [1, 2]
            if [int(layer.get("stride", 0)) for layer in stem] != expected_strides:
                raise ValueError("cnn_ctc_v8 stem strides must remain [1,2]")
            expected_dilations = [1, 2, 2, 2, 2]
            if [int(block.get("dilation", 0)) for block in blocks] != expected_dilations:
                raise ValueError("cnn_ctc_v8 residual dilation schedule is frozen")
            expected_paddings = [5, 18, 26, 34, 42]
            if [int(block.get("padding", -1)) for block in blocks] != expected_paddings:
                raise ValueError("cnn_ctc_v8 residual padding schedule is frozen")
        if model_id == "cnn_ctc_v9":
            expected_strides = [2, 2]
            if [int(layer.get("stride", 0)) for layer in stem] != expected_strides:
                raise ValueError("cnn_ctc_v9 stem strides must remain [2,2]")
            expected_dilations = [1, 2, 3, 4, 4, 3, 2, 1]
            if [int(block.get("dilation", 0)) for block in blocks] != expected_dilations:
                raise ValueError("cnn_ctc_v9 residual dilation schedule is frozen")
            expected_paddings = [3, 6, 9, 12, 12, 9, 6, 3]
            if [int(block.get("padding", -1)) for block in blocks] != expected_paddings:
                raise ValueError("cnn_ctc_v9 residual padding schedule is frozen")
        if model_id == "cnn_ctc_v10":
            expected_strides = [2, 2]
            if [int(layer.get("stride", 0)) for layer in stem] != expected_strides:
                raise ValueError("cnn_ctc_v10 stem strides must remain [2,2]")
            expected_dilations = [1, 2, 3, 4, 4, 3, 2, 1]
            if [int(block.get("dilation", 0)) for block in blocks] != expected_dilations:
                raise ValueError("cnn_ctc_v10 residual dilation schedule is frozen")
            expected_paddings = [3, 6, 9, 12, 12, 9, 6, 3]
            if [int(block.get("padding", -1)) for block in blocks] != expected_paddings:
                raise ValueError("cnn_ctc_v10 residual padding schedule is frozen")
        if (
            model_id in {"cnn_ctc_v5", "cnn_ctc_v6"}
            and network.get("intermediate_ctc_after_block") != 3
        ):
            raise ValueError(f"{model_id} intermediate CTC block must remain 3")
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
    return greedy_decode_logits_diagnostics(logits, vocab)["hypothesis"]


def greedy_decode_logits_diagnostics(
    logits: Sequence[Sequence[float]],
    vocab: dict,
) -> dict:
    indices: list[int] = []
    width = len(vocab["tokens"])
    for frame_index, frame in enumerate(logits):
        if len(frame) != width:
            raise ValueError(
                f"logit frame {frame_index} has width {len(frame)}, expected {width}"
            )
        best = max(range(width), key=lambda index: float(frame[index]))
        indices.append(best)

    blank = int(vocab["blank_index"])
    hypothesis = greedy_decode(indices, vocab)
    blank_frames = sum(1 for value in indices if value == blank)
    collapsed_tokens = 0
    previous = None
    argmax_runs = 0
    for value in indices:
        if previous is None or value != previous:
            argmax_runs += 1
        if value != blank and value != previous:
            collapsed_tokens += 1
        previous = value

    frame_count = len(indices)
    return {
        "hypothesis": hypothesis,
        "frame_count": frame_count,
        "blank_argmax_frames": blank_frames,
        "nonblank_argmax_frames": frame_count - blank_frames,
        "blank_frame_fraction": (
            blank_frames / frame_count if frame_count else 0.0
        ),
        "argmax_runs": argmax_runs,
        "collapsed_token_count": collapsed_tokens,
        "emitted_words": len(hypothesis.split()),
        "emitted_characters_no_spaces": len(hypothesis.replace(" ", "")),
    }


def aggregate_decoder_diagnostics(per_sample: Sequence[dict]) -> dict:
    if not per_sample:
        raise ValueError("decoder diagnostics require at least one evaluated sample")

    frame_count = sum(int(item["decoder"]["frame_count"]) for item in per_sample)
    blank_frames = sum(
        int(item["decoder"]["blank_argmax_frames"]) for item in per_sample
    )
    nonblank_frames = sum(
        int(item["decoder"]["nonblank_argmax_frames"]) for item in per_sample
    )
    collapsed_tokens = sum(
        int(item["decoder"]["collapsed_token_count"]) for item in per_sample
    )
    emitted_words = sum(
        int(item["decoder"]["emitted_words"]) for item in per_sample
    )
    emitted_characters = sum(
        int(item["decoder"]["emitted_characters_no_spaces"])
        for item in per_sample
    )
    reference_words = sum(int(item["reference_words"]) for item in per_sample)
    reference_characters = sum(
        int(item["reference_characters_no_spaces"]) for item in per_sample
    )
    empty_hypotheses = sum(
        1 for item in per_sample if not str(item["hypothesis"]).strip()
    )

    return {
        "frame_count": frame_count,
        "blank_argmax_frames": blank_frames,
        "nonblank_argmax_frames": nonblank_frames,
        "blank_frame_fraction": (
            blank_frames / frame_count if frame_count else 0.0
        ),
        "collapsed_token_count": collapsed_tokens,
        "emitted_words": emitted_words,
        "reference_words": reference_words,
        "emitted_characters_no_spaces": emitted_characters,
        "reference_characters_no_spaces": reference_characters,
        "emitted_to_reference_character_ratio": (
            emitted_characters / max(1, reference_characters)
        ),
        "empty_hypotheses": empty_hypotheses,
        "empty_hypothesis_fraction": empty_hypotheses / len(per_sample),
    }


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
    if spec.get("id") in {
        "cnn_ctc_v2",
        "cnn_ctc_v3",
        "cnn_ctc_v4",
        "cnn_ctc_v5",
        "cnn_ctc_v6",
        "cnn_ctc_v7",
        "cnn_ctc_v8",
        "cnn_ctc_v9",
        "cnn_ctc_v10",
    }:
        layers = network["stem"]
        for layer in layers:
            length = conv1d_output_length(
                length,
                kernel=int(layer["kernel"]),
                stride=int(layer["stride"]),
                padding=int(layer["padding"]),
                dilation=int(layer.get("dilation", 1)),
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
    if spec.get("id") not in {
        "cnn_ctc_v2",
        "cnn_ctc_v3",
        "cnn_ctc_v4",
        "cnn_ctc_v5",
        "cnn_ctc_v6",
        "cnn_ctc_v7",
        "cnn_ctc_v8",
        "cnn_ctc_v9",
        "cnn_ctc_v10",
    }:
        raise ValueError("resource estimator targets cnn_ctc_v2/v3/v4/v5/v6/v7/v8/v9/v10")

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
        dilation = int(layer.get("dilation", 1))
        out_length = conv1d_output_length(
            length,
            kernel=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        use_batchnorm = spec["network"]["normalization"] == "batchnorm"
        parameters += in_channels * out_channels * kernel
        if use_batchnorm:
            parameters += 2 * out_channels  # BatchNorm gamma/beta
        else:
            parameters += out_channels  # Conv bias
        macs += in_channels * out_channels * kernel * out_length
        receptive_field += (kernel - 1) * dilation * jump
        jump *= stride
        length = out_length
        in_channels = out_channels

    for block in network["residual_blocks"]:
        channels = int(block["channels"])
        kernel = int(block["kernel"])
        projection_kernel = int(block["projection_kernel"])
        dilation = int(block.get("dilation", 1))
        if channels != in_channels:
            raise ValueError("generation-1 residual block changes channel width")
        parameters += channels * channels * kernel
        parameters += 2 * channels if use_batchnorm else channels
        parameters += channels * channels * projection_kernel
        parameters += 2 * channels if use_batchnorm else channels
        macs += channels * channels * kernel * length
        macs += channels * channels * projection_kernel * length
        receptive_field += (kernel - 1) * dilation * jump

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
