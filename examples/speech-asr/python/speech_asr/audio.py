"""Canonical audio I/O and deterministic resampling for speech-ASR."""

from __future__ import annotations

import hashlib
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

CANONICAL_SAMPLE_RATE = 16000


class AudioFormatError(ValueError):
    """Raised when an input audio representation is unsupported or malformed."""


@dataclass(frozen=True)
class CanonicalAudio:
    samples: tuple[float, ...]
    sample_rate_hz: int = CANONICAL_SAMPLE_RATE
    channels: int = 1
    encoding: str = "f32le"

    def __post_init__(self) -> None:
        if self.sample_rate_hz != CANONICAL_SAMPLE_RATE:
            raise AudioFormatError("canonical audio must be 16000 Hz")
        if self.channels != 1:
            raise AudioFormatError("canonical audio must be mono")
        if self.encoding != "f32le":
            raise AudioFormatError("canonical audio must use f32le")

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / CANONICAL_SAMPLE_RATE

    def slice(self, start_sample: int, end_sample: int) -> "CanonicalAudio":
        if start_sample < 0 or end_sample <= start_sample or end_sample > len(self.samples):
            raise ValueError("invalid canonical audio slice")
        return CanonicalAudio(self.samples[start_sample:end_sample])


def _decode_pcm_sample(data: bytes, offset: int, width: int) -> int:
    if width == 1:
        return data[offset] - 128
    if width == 2:
        return struct.unpack_from("<h", data, offset)[0]
    if width == 3:
        raw = int.from_bytes(data[offset : offset + 3], "little", signed=False)
        if raw & 0x800000:
            raw -= 1 << 24
        return raw
    if width == 4:
        return struct.unpack_from("<i", data, offset)[0]
    raise AudioFormatError(f"unsupported PCM sample width: {width} bytes")


def _pcm_scale(width: int) -> float:
    if width == 1:
        return 128.0
    if width == 2:
        return 32768.0
    if width == 3:
        return 8388608.0
    if width == 4:
        return 2147483648.0
    raise AudioFormatError(f"unsupported PCM sample width: {width} bytes")


def decode_pcm_to_mono_float32(
    data: bytes, *, sample_width: int, channels: int
) -> tuple[float, ...]:
    """Decode interleaved integer PCM bytes and average channels to mono."""

    if channels <= 0:
        raise AudioFormatError("channel count must be positive")
    frame_size = sample_width * channels
    if frame_size <= 0 or len(data) % frame_size:
        raise AudioFormatError("PCM payload is not an integer number of frames")

    scale = _pcm_scale(sample_width)
    output = []
    for frame_offset in range(0, len(data), frame_size):
        total = 0.0
        for channel in range(channels):
            offset = frame_offset + channel * sample_width
            total += _decode_pcm_sample(data, offset, sample_width) / scale
        output.append(total / channels)
    return tuple(output)


def resample_linear(
    samples: Sequence[float], source_rate_hz: int, target_rate_hz: int = CANONICAL_SAMPLE_RATE
) -> tuple[float, ...]:
    """Deterministically resample mono samples using linear interpolation.

    Output sample positions are calculated with integer arithmetic so endpoint
    selection does not depend on accumulated floating-point phase.
    """

    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("sample rates must be positive")
    if not samples:
        return ()
    if source_rate_hz == target_rate_hz:
        return tuple(float(value) for value in samples)

    output_count = (len(samples) * target_rate_hz + source_rate_hz // 2) // source_rate_hz
    output = []
    last = len(samples) - 1
    for output_index in range(output_count):
        position_num = output_index * source_rate_hz
        left = position_num // target_rate_hz
        fraction_num = position_num % target_rate_hz
        if left >= last:
            output.append(float(samples[last]))
            continue
        fraction = fraction_num / target_rate_hz
        left_value = float(samples[left])
        right_value = float(samples[left + 1])
        output.append(left_value + (right_value - left_value) * fraction)
    return tuple(output)


def read_wav_canonical(path: Path) -> CanonicalAudio:
    """Read uncompressed integer PCM WAV and normalize it to canonical audio."""

    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise AudioFormatError(f"{path}: compressed WAV is not supported")
        sample_width = wav.getsampwidth()
        channels = wav.getnchannels()
        source_rate = wav.getframerate()
        payload = wav.readframes(wav.getnframes())

    mono = decode_pcm_to_mono_float32(
        payload, sample_width=sample_width, channels=channels
    )
    return CanonicalAudio(resample_linear(mono, source_rate))


def encode_f32le(samples: Iterable[float]) -> bytes:
    payload = bytearray()
    for value in samples:
        value = float(value)
        if value < -1.0 or value > 1.0:
            raise AudioFormatError("canonical samples must be in range -1..1")
        payload += struct.pack("<f", value)
    return bytes(payload)


def decode_f32le(payload: bytes) -> tuple[float, ...]:
    if len(payload) % 4:
        raise AudioFormatError("f32le payload length must be divisible by four")
    return tuple(value[0] for value in struct.iter_unpack("<f", payload))


def write_f32le(path: Path, audio: CanonicalAudio) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encode_f32le(audio.samples)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_f32le(path: Path) -> CanonicalAudio:
    return CanonicalAudio(decode_f32le(path.read_bytes()))
