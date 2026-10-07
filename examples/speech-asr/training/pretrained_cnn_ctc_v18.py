"""Import the pinned NVIDIA QuartzNet15x5Base-En NeMo checkpoint into cnn_ctc_v18."""

from __future__ import annotations

import hashlib
import io
import pathlib
import tarfile
from typing import Iterable

import torch
from torch import nn

SOURCE_SIZE_BYTES = 71083664
SOURCE_SHA512 = "74e8284e77098906afb7a15a861ef60ec14db1a4acb206fa719492fa43050ad69a91c245652c05c5f0ded38b5903ed55"
ENCODER_MEMBER = ".nemo_tmp/JasperEncoder.pt"
DECODER_MEMBER = ".nemo_tmp/JasperDecoderForCTC.pt"
SOURCE_TOKENS = [" "] + list("abcdefghijklmnopqrstuvwxyz") + ["'", "<blank>"]


def sha512_path(path: pathlib.Path) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(path: pathlib.Path) -> dict:
    size = path.stat().st_size
    if size != SOURCE_SIZE_BYTES:
        raise ValueError(f"pretrained archive size mismatch: {size} != {SOURCE_SIZE_BYTES}")
    digest = sha512_path(path)
    if digest != SOURCE_SHA512:
        raise ValueError(f"pretrained archive SHA-512 mismatch: {digest} != {SOURCE_SHA512}")
    return {"path": str(path), "size_bytes": size, "sha512": digest}


def _load_torch_member(archive: pathlib.Path, member: str) -> dict[str, torch.Tensor]:
    with tarfile.open(archive, mode="r:gz") as tar:
        try:
            extracted = tar.extractfile(member)
        except KeyError as exc:
            raise ValueError(f"pretrained archive member missing: {member}") from exc
        if extracted is None:
            raise ValueError(f"pretrained archive member is not a file: {member}")
        payload = extracted.read()
    try:
        value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(io.BytesIO(payload), map_location="cpu")
    if isinstance(value, dict) and isinstance(value.get("state_dict"), dict):
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise ValueError(f"pretrained member did not contain a state dict: {member}")
    state = {}
    for raw_key, tensor in value.items():
        if not isinstance(raw_key, str) or not torch.is_tensor(tensor):
            continue
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        state[key] = tensor.detach().cpu()
    return state


def _source_tensor(state: dict[str, torch.Tensor], candidates: Iterable[str]) -> tuple[str, torch.Tensor]:
    for key in candidates:
        if key in state:
            return key, state[key]
    raise ValueError("pretrained tensor missing; tried: " + ", ".join(candidates))


def _copy_tensor(
    target_state: dict[str, torch.Tensor],
    target_key: str,
    source_state: dict[str, torch.Tensor],
    *source_candidates: str,
) -> str:
    source_key, value = _source_tensor(source_state, source_candidates)
    wanted = target_state[target_key]
    if tuple(value.shape) != tuple(wanted.shape):
        raise ValueError(
            f"pretrained shape mismatch for {target_key}: "
            f"{tuple(value.shape)} from {source_key} != {tuple(wanted.shape)}"
        )
    target_state[target_key] = value.to(dtype=wanted.dtype).clone()
    return source_key


def _conv_candidates(block: int, index: int, suffix: str = "weight") -> tuple[str, ...]:
    base = f"encoder.{block}.mconv.{index}"
    return (
        f"{base}.{suffix}",
        f"{base}.conv.{suffix}",
        f"{base}.conv.weight" if suffix == "weight" else f"{base}.conv.{suffix}",
    )


def _bn_candidates(block: int, index: int, suffix: str) -> tuple[str, ...]:
    base = f"encoder.{block}.mconv.{index}"
    return (f"{base}.{suffix}", f"{base}.bn.{suffix}")


def _res_conv_candidates(block: int) -> tuple[str, ...]:
    return (
        f"encoder.{block}.res.0.0.weight",
        f"encoder.{block}.res.0.0.conv.weight",
        f"encoder.{block}.res.0.weight",
        f"encoder.{block}.res.0.conv.weight",
    )


def _res_bn_candidates(block: int, suffix: str) -> tuple[str, ...]:
    return (
        f"encoder.{block}.res.0.1.{suffix}",
        f"encoder.{block}.res.1.{suffix}",
    )


def _copy_bn(target_state, prefix, source_state, block, mconv_index, used):
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        used.add(_copy_tensor(
            target_state,
            f"{prefix}.{suffix}",
            source_state,
            *_bn_candidates(block, mconv_index, suffix),
        ))
    target_key = f"{prefix}.num_batches_tracked"
    if target_key in target_state:
        for candidate in _bn_candidates(block, mconv_index, "num_batches_tracked"):
            if candidate in source_state:
                value = source_state[candidate]
                if tuple(value.shape) == tuple(target_state[target_key].shape):
                    target_state[target_key] = value.to(dtype=target_state[target_key].dtype).clone()
                    used.add(candidate)
                    break


def _copy_tcs(target_state, prefix, source_state, block, mconv_base, used):
    used.add(_copy_tensor(
        target_state,
        f"{prefix}.depthwise.weight",
        source_state,
        *_conv_candidates(block, mconv_base),
    ))
    used.add(_copy_tensor(
        target_state,
        f"{prefix}.pointwise.weight",
        source_state,
        *_conv_candidates(block, mconv_base + 1),
    ))
    _copy_bn(target_state, f"{prefix}.batchnorm", source_state, block, mconv_base + 2, used)


def load_pretrained_quartznet(
    model: nn.Module,
    archive: pathlib.Path,
    target_tokens: list[str],
) -> dict:
    provenance = verify_source(archive)
    encoder = _load_torch_member(archive, ENCODER_MEMBER)
    decoder = _load_torch_member(archive, DECODER_MEMBER)
    target = model.state_dict()
    used_encoder: set[str] = set()
    used_decoder: set[str] = set()

    _copy_tcs(target, "prologue.0", encoder, 0, 0, used_encoder)

    for target_block in range(15):
        source_block = target_block + 1
        for repeat, target_index in enumerate((0, 3, 6, 9, 12)):
            _copy_tcs(
                target,
                f"blocks.{target_block}.main.{target_index}",
                encoder,
                source_block,
                repeat * 5,
                used_encoder,
            )
        used_encoder.add(_copy_tensor(
            target,
            f"blocks.{target_block}.residual.0.weight",
            encoder,
            *_res_conv_candidates(source_block),
        ))
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            used_encoder.add(_copy_tensor(
                target,
                f"blocks.{target_block}.residual.1.{suffix}",
                encoder,
                *_res_bn_candidates(source_block, suffix),
            ))
        tracked = f"blocks.{target_block}.residual.1.num_batches_tracked"
        if tracked in target:
            for candidate in _res_bn_candidates(source_block, "num_batches_tracked"):
                if candidate in encoder:
                    target[tracked] = encoder[candidate].to(dtype=target[tracked].dtype).clone()
                    used_encoder.add(candidate)
                    break

    _copy_tcs(target, "c2.0", encoder, 16, 0, used_encoder)
    used_encoder.add(_copy_tensor(
        target,
        "c3.0.weight",
        encoder,
        *_conv_candidates(17, 0),
    ))
    _copy_bn(target, "c3.1", encoder, 17, 1, used_encoder)

    decoder_weight_key, decoder_weight = _source_tensor(
        decoder,
        (
            "decoder_layers.0.weight",
            "decoder_layers.0.conv.weight",
            "decoder.0.weight",
            "decoder.0.conv.weight",
        ),
    )
    decoder_bias_key, decoder_bias = _source_tensor(
        decoder,
        (
            "decoder_layers.0.bias",
            "decoder_layers.0.conv.bias",
            "decoder.0.bias",
            "decoder.0.conv.bias",
        ),
    )
    if decoder_weight.ndim == 2:
        decoder_weight = decoder_weight.unsqueeze(-1)
    if tuple(decoder_weight.shape[1:]) != tuple(target["projection.weight"].shape[1:]):
        raise ValueError(
            f"pretrained decoder feature shape mismatch: {tuple(decoder_weight.shape)}"
        )
    if decoder_weight.shape[0] != len(SOURCE_TOKENS) or decoder_bias.shape[0] != len(SOURCE_TOKENS):
        raise ValueError("pretrained decoder vocabulary size is not 29")

    target_index = {token: index for index, token in enumerate(target_tokens)}
    shared = []
    for source_index, token in enumerate(SOURCE_TOKENS):
        mapped = "<blank>" if token == "<blank>" else token
        if mapped not in target_index:
            raise ValueError(f"target vocabulary lacks pretrained token {mapped!r}")
        dest = target_index[mapped]
        target["projection.weight"][dest] = decoder_weight[source_index].to(
            dtype=target["projection.weight"].dtype
        )
        target["projection.bias"][dest] = decoder_bias[source_index].to(
            dtype=target["projection.bias"].dtype
        )
        shared.append({"token": mapped, "source_index": source_index, "target_index": dest})
    used_decoder.update({decoder_weight_key, decoder_bias_key})

    missing, unexpected = model.load_state_dict(target, strict=True)
    if missing or unexpected:
        raise ValueError(f"internal pretrained load mismatch: missing={missing} unexpected={unexpected}")

    target_only = [token for token in target_tokens if token not in set(SOURCE_TOKENS)]
    return {
        "source": provenance,
        "encoder_member": ENCODER_MEMBER,
        "decoder_member": DECODER_MEMBER,
        "encoder_source_tensors_used": len(used_encoder),
        "decoder_source_tensors_used": len(used_decoder),
        "shared_decoder_symbols": shared,
        "shared_decoder_symbol_count": len(shared),
        "target_only_symbols_random_init": target_only,
        "target_only_symbol_count": len(target_only),
        "policy": "full encoder plus shared CTC decoder rows; target-only rows retain seeded initialization",
    }
