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


def _slaney_hz_to_mel(hz):
    np = _np()
    values = np.asarray(hz, dtype=np.float64)
    f_sp = 200.0 / 3.0
    mels = values / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_region = values >= min_log_hz
    mels = np.where(
        log_region,
        min_log_mel + np.log(np.maximum(values, min_log_hz) / min_log_hz) / logstep,
        mels,
    )
    return mels


def _slaney_mel_to_hz(mels):
    np = _np()
    values = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3.0
    hz = values * f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_region = values >= min_log_mel
    hz = np.where(
        log_region,
        min_log_hz * np.exp(logstep * (values - min_log_mel)),
        hz,
    )
    return hz


def mel_filterbank_slaney(spec: dict):
    """Librosa/NeMo-style Slaney mel triangles with area normalization."""
    np = _np()
    frontend = spec["frontend"]
    sample_rate = int(frontend["sample_rate_hz"])
    fft_size = int(frontend["fft_size"])
    mel_bins = int(frontend["mel_bins"])
    fmin = float(frontend["fmin_hz"])
    fmax = float(frontend["fmax_hz"])
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, fft_size // 2 + 1, dtype=np.float64)
    mel_points = np.linspace(
        float(_slaney_hz_to_mel(fmin)),
        float(_slaney_hz_to_mel(fmax)),
        mel_bins + 2,
        dtype=np.float64,
    )
    hz_points = _slaney_mel_to_hz(mel_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]
    fdiff = np.diff(hz_points)
    lower = -ramps[:-2] / np.maximum(fdiff[:-1, None], 1e-12)
    upper = ramps[2:] / np.maximum(fdiff[1:, None], 1e-12)
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / np.maximum(hz_points[2:] - hz_points[:-2], 1e-12)
    weights *= enorm[:, None]
    return weights.astype(np.float32)


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
    log_floor = (
        None
        if frontend.get("kind") == "logmel-v3"
        else float(frontend["log_floor"])
    )

    audio, original_samples = fixed_audio(samples, spec)
    expected = window_samples + (fixed_frames - 1) * hop_samples
    if len(audio) < expected:
        raise ValueError(
            f"fixed audio length {len(audio)} is shorter than frontend geometry {expected}"
        )

    frontend_kind = frontend.get("kind")
    if frontend_kind == "logmel-v3":
        preemphasis = float(frontend["preemphasis"])
        processed = np.zeros_like(audio)
        if original_samples:
            processed[0] = audio[0]
        if original_samples > 1:
            processed[1:original_samples] = (
                audio[1:original_samples]
                - preemphasis * audio[: original_samples - 1]
            )
        audio = processed
        if frontend.get("window") != "hann-periodic":
            raise ValueError("logmel-v3 requires hann-periodic window")
        window = np.hanning(window_samples + 1).astype(np.float32)[:-1]
    else:
        window = np.hanning(window_samples).astype(np.float32)

    frames = np.lib.stride_tricks.sliding_window_view(audio, window_samples)[::hop_samples]
    frames = frames[:fixed_frames]
    if frames.shape != (fixed_frames, window_samples):
        raise ValueError(f"unexpected frame shape: {frames.shape}")

    if original_samples < window_samples:
        valid_frames = 1
    else:
        valid_frames = min(
            fixed_frames,
            1 + (original_samples - window_samples) // hop_samples,
        )

    spectrum = np.fft.rfft(frames * window[None, :], n=fft_size, axis=1)
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(np.float32)
    if frontend_kind == "logmel-v3":
        if int(frontend.get("power", 0)) != 2:
            raise ValueError("logmel-v3 requires power=2")
        if frontend.get("mel_scale") != "slaney" or frontend.get("mel_norm") != "slaney":
            raise ValueError("logmel-v3 requires Slaney mel scale and normalization")
        mel = mel_filterbank_slaney(spec) @ power.T
        guard = float(frontend["log_guard"])
        logged = np.log(mel + guard).astype(np.float32)
    else:
        mel = mel_filterbank(spec) @ power.T
        assert log_floor is not None
        logged = np.log(np.maximum(mel, log_floor)).astype(np.float32)

    normalization = frontend.get("normalization")
    if normalization == "per_mel_bin_mean":
        # logmel-v1 historical behavior: the mean includes fixed-shape padding.
        logged = logged - logged.mean(axis=1, keepdims=True)
    elif normalization == "per_mel_bin_mean_std_valid_zero_pad":
        # logmel-v3: approximate the NeMo per_feature normalization contract
        # deterministically over real frames only; execution padding stays neutral.
        valid = logged[:, :valid_frames]
        valid_mean = valid.mean(axis=1, keepdims=True)
        centered = valid - valid_mean
        variance = (centered * centered).mean(axis=1, keepdims=True)
        denom = np.sqrt(variance + float(frontend["normalization_eps"]))
        logged[:, :valid_frames] = centered / denom
        if valid_frames < fixed_frames:
            logged[:, valid_frames:] = 0.0
    elif normalization == "per_mel_bin_mean_valid_zero_pad":
        # logmel-v2: padding is an execution-shape artifact and must not affect
        # utterance CMVN. Normalize only real frames, then keep padded feature
        # frames neutral in normalized space.
        valid_mean = logged[:, :valid_frames].mean(axis=1, keepdims=True)
        logged = logged - valid_mean
        if valid_frames < fixed_frames:
            logged[:, valid_frames:] = 0.0
    else:
        raise ValueError(
            f"unsupported cnn_ctc frontend normalization: {normalization!r}"
        )

    return logged[None, :, :], int(valid_frames)
