"""NumPy implementation of the cnn_ctc_v1 log-mel frontend.

NumPy is imported lazily so dependency-free contract/tooling tests remain usable
in work/venv-tools. Runtime/training callers use work/venv-apps or
work/venv-training.
"""

from __future__ import annotations

import math
from typing import Sequence


def _np():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError(
            "cnn_ctc_v1 frontend requires NumPy; use the apps or training uv environment"
        ) from exc
    return numpy


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(spec: dict):
    np = _np()
    frontend = spec["frontend"]
    sample_rate = int(frontend["sample_rate_hz"])
    fft_size = int(frontend["fft_size"])
    mel_bins = int(frontend["mel_bins"])
    fmin = float(frontend["fmin_hz"])
    fmax = float(frontend["fmax_hz"])
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, fft_size // 2 + 1, dtype=np.float64)
    mel_points = np.linspace(
        hz_to_mel(fmin),
        hz_to_mel(fmax),
        mel_bins + 2,
        dtype=np.float64,
    )
    hz_points = np.array([mel_to_hz(float(value)) for value in mel_points])
    bank = np.zeros((mel_bins, len(fft_freqs)), dtype=np.float64)
    for index in range(mel_bins):
        left, center, right = hz_points[index : index + 3]
        rising = (fft_freqs - left) / max(center - left, 1e-12)
        falling = (right - fft_freqs) / max(right - center, 1e-12)
        bank[index] = np.maximum(0.0, np.minimum(rising, falling))
    return bank.astype(np.float32)


def fixed_audio(samples: Sequence[float], spec: dict):
    np = _np()
    count = int(spec["frontend"]["fixed_audio_samples"])
    values = np.asarray(samples, dtype=np.float32)
    original = min(len(values), count)
    output = np.zeros((count,), dtype=np.float32)
    if original:
        output[:original] = values[:original]
    return output, original


def logmel_features(samples: Sequence[float], spec: dict):
    np = _np()
    frontend = spec["frontend"]
    window_samples = int(frontend["window_samples"])
    hop_samples = int(frontend["hop_samples"])
    fft_size = int(frontend["fft_size"])
    fixed_frames = int(frontend["fixed_frames"])
    log_floor = float(frontend["log_floor"])

    audio, original_samples = fixed_audio(samples, spec)
    expected = window_samples + (fixed_frames - 1) * hop_samples
    if len(audio) != expected:
        raise ValueError(
            f"fixed audio length {len(audio)} does not match frontend geometry {expected}"
        )

    window = np.hanning(window_samples).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(audio, window_samples)[::hop_samples]
    frames = frames[:fixed_frames]
    if frames.shape != (fixed_frames, window_samples):
        raise ValueError(f"unexpected frame shape: {frames.shape}")

    spectrum = np.fft.rfft(frames * window[None, :], n=fft_size, axis=1)
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(np.float32)
    mel = mel_filterbank(spec) @ power.T
    logged = np.log(np.maximum(mel, log_floor)).astype(np.float32)
    if frontend.get("normalization") == "per_mel_bin_mean":
        logged = logged - logged.mean(axis=1, keepdims=True)

    if original_samples < window_samples:
        valid_frames = 1
    else:
        valid_frames = min(
            fixed_frames,
            1 + (original_samples - window_samples) // hop_samples,
        )
    return logged[None, :, :], int(valid_frames)
