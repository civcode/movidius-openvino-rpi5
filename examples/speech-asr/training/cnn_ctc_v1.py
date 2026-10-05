"""PyTorch implementation and dataset helpers for cnn_ctc_v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import Dataset

SPEECH_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.audio import read_f32le  # noqa: E402
from speech_asr.cnn_ctc import (  # noqa: E402
    acoustic_output_length,
    encode_text,
    manifest_record_eligibility,
)
from speech_asr.cnn_ctc_frontend import logmel_features  # noqa: E402
from speech_asr.contracts import validate_speech_sample  # noqa: E402


class CnnCtcV1(nn.Module):
    def __init__(self, spec: dict, vocab_size: int):
        super().__init__()
        frontend = spec["frontend"]
        network = spec["network"]
        channels = [int(value) for value in network["channels"]]
        kernels = [int(value) for value in network["kernels"]]
        strides = [int(value) for value in network["strides"]]
        paddings = [int(value) for value in network["paddings"]]
        if not (len(channels) == len(kernels) == len(strides) == len(paddings) == 3):
            raise ValueError("cnn_ctc_v1 expects exactly three temporal conv blocks")

        in_channels = int(frontend["mel_bins"])
        blocks = []
        for out_channels, kernel, stride, padding in zip(
            channels, kernels, strides, paddings
        ):
            blocks.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    bias=True,
                )
            )
            blocks.append(nn.ReLU())
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.projection = nn.Conv1d(
            in_channels,
            vocab_size,
            kernel_size=int(network["projection_kernel"]),
            bias=True,
        )

    def forward(self, features):
        encoded = self.encoder(features)
        logits = self.projection(encoded)
        return logits.transpose(1, 2)


def read_manifest(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            records.append(validate_speech_sample(value))
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def resolve_audio_path(manifest: Path, record: dict) -> Path:
    path = Path(record["audio"]["path"])
    if not path.is_absolute():
        path = manifest.parent / path
    return path


class ManifestCtcDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        spec: dict,
        vocab: dict,
        *,
        max_samples: int | None = None,
    ):
        self.manifest = manifest
        self.spec = spec
        self.vocab = vocab
        eligible = []
        skipped_long = []
        skipped_target = []
        for record in read_manifest(manifest):
            decision = manifest_record_eligibility(record, spec, vocab)
            if not decision["eligible"]:
                if decision["reason"] == "too_long":
                    skipped_long.append(record["id"])
                elif decision["reason"] == "target_too_long":
                    skipped_target.append(record["id"])
                else:
                    raise ValueError(
                        f"{record['id']}: unknown {spec['id']} eligibility reason "
                        f"{decision['reason']!r}"
                    )
                continue
            eligible.append(record)

        if max_samples is not None:
            eligible = eligible[: max(0, int(max_samples))]
        if not eligible:
            raise ValueError(
                f"no eligible manifest samples for {spec['id']}; "
                f"too_long={skipped_long}, target_too_long={skipped_target}"
            )
        self.records = eligible
        self.skipped_long = tuple(skipped_long)
        self.skipped_target = tuple(skipped_target)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        audio_path = resolve_audio_path(self.manifest, record)
        audio = read_f32le(audio_path)
        features, valid_frames = logmel_features(audio.samples, self.spec)
        target = encode_text(record["transcript"]["text"], self.vocab)
        output_frames = acoustic_output_length(valid_frames, self.spec)
        return {
            "id": record["id"],
            "features": torch.from_numpy(features[0]),
            "input_length": output_frames,
            "target": torch.tensor(target, dtype=torch.long),
            "target_length": len(target),
            "text": record["transcript"]["text"],
        }


def collate_ctc(batch: Iterable[dict]) -> dict:
    values = list(batch)
    if not values:
        raise ValueError("cannot collate empty batch")
    return {
        "ids": [item["id"] for item in values],
        "texts": [item["text"] for item in values],
        "features": torch.stack([item["features"] for item in values], dim=0),
        "input_lengths": torch.tensor(
            [item["input_length"] for item in values],
            dtype=torch.long,
        ),
        "targets": torch.cat([item["target"] for item in values], dim=0),
        "target_lengths": torch.tensor(
            [item["target_length"] for item in values],
            dtype=torch.long,
        ),
    }
