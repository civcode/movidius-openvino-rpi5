"""Model-neutral frontend and feature tensor contracts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .audio import CanonicalAudio


@dataclass(frozen=True)
class FeatureTensor:
    """Portable float32 feature tensor with explicit layout and shape."""

    layout: str
    shape: tuple[int, ...]
    values: tuple[float, ...]
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.dtype != "float32":
            raise ValueError("only float32 feature tensors are supported")
        if not self.layout:
            raise ValueError("feature tensor layout must be non-empty")
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError("feature tensor dimensions must be positive")
        expected = math.prod(self.shape)
        if expected != len(self.values):
            raise ValueError(
                f"feature tensor shape requires {expected} values, got {len(self.values)}"
            )

    def to_f32le(self) -> bytes:
        payload = bytearray()
        for value in self.values:
            payload += struct.pack("<f", float(value))
        return bytes(payload)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_f32le()).hexdigest()


class FrontendProfile(Protocol):
    """A model package's deterministic canonical-audio-to-tensor transform."""

    profile_id: str

    def extract(self, audio: CanonicalAudio) -> FeatureTensor:
        ...


@dataclass(frozen=True)
class RawAudioFrontend:
    """Identity frontend used for plumbing tests, never as an ASR default."""

    profile_id: str = "raw-audio-test-v1"

    def extract(self, audio: CanonicalAudio) -> FeatureTensor:
        return FeatureTensor(
            layout="BT",
            shape=(1, audio.sample_count),
            values=audio.samples,
        )


def dump_feature_tensor(base_path: Path, tensor: FeatureTensor, profile_id: str) -> dict:
    """Write deterministic .f32 data and canonical JSON metadata."""

    data_path = base_path.with_suffix(".f32")
    metadata_path = base_path.with_suffix(".json")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    payload = tensor.to_f32le()
    data_path.write_bytes(payload)

    metadata = {
        "schema": "speech-asr/feature-tensor",
        "version": 1,
        "profile_id": profile_id,
        "dtype": tensor.dtype,
        "encoding": "f32le",
        "layout": tensor.layout,
        "shape": list(tensor.shape),
        "elements": len(tensor.values),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "data_file": data_path.name,
    }
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return metadata
