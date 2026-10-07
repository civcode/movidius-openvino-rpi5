"""NumPy reproduction of the historical NeMo QuartzNet evaluation frontend.

This module is dependency-light so the same source-domain feature contract can
run on Raspberry Pi without installing PyTorch.
"""

from __future__ import annotations

from typing import Sequence

from .cnn_ctc_frontend import mel_filterbank_slaney


def reference_feature_lengths(sample_count: int, spec: dict) -> tuple[int, int]:
    """Return (valid_frames, padded_tensor_frames) for the frozen source frontend."""

    if sample_count < 1:
        raise ValueError("reference audio must contain at least one sample")
    frontend = spec["frontend"]
    if not bool(frontend["stft_center"]):
        raise ValueError("reference frontend requires centered STFT")
    hop = int(frontend["hop_samples"])
    pad_to = int(frontend["pad_to"])
    if hop < 1 or pad_to < 1:
        raise ValueError("invalid reference frontend framing")
    valid_frames = sample_count // hop
    # torch.stft(center=True) pads n_fft/2 on both sides, yielding
    # 1 + floor(sample_count / hop) frames for the frozen geometry.
    stft_frames = valid_frames + 1
    padded_frames = ((stft_frames + pad_to - 1) // pad_to) * pad_to
    return valid_frames, padded_frames


def nemo_reference_features_numpy(
    samples: Sequence[float],
    spec: dict,
):
    """Reproduce the qualified NeMo FilterbankFeatures eval path with NumPy."""

    import numpy as np

    frontend = spec["frontend"]
    if int(frontend["sample_rate_hz"]) != 16000:
        raise ValueError("reference frontend requires 16 kHz audio")
    if frontend["evaluation_dither"] != 0:
        raise ValueError("reference evaluation dither must remain disabled")
    if frontend["window"] != "hann" or bool(frontend["hann_periodic"]):
        raise ValueError("reference frontend requires a non-periodic Hann window")
    if not bool(frontend["stft_center"]) or frontend["stft_pad_mode"] != "constant":
        raise ValueError("reference frontend requires centered constant-pad STFT")
    if int(frontend["magnitude_power"]) != 2:
        raise ValueError("reference frontend requires a power spectrum")
    if frontend["mel_scale"] != "slaney" or frontend["mel_norm"] != "slaney":
        raise ValueError("reference frontend requires Slaney mel filters")
    if frontend["normalization"] != "per_feature":
        raise ValueError("reference frontend requires per-feature normalization")
    if int(frontend["normalization_ddof"]) != 1:
        raise ValueError("reference frontend requires ddof=1 normalization")

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    sample_count = int(audio.size)
    valid_frames, padded_frames = reference_feature_lengths(sample_count, spec)
    if valid_frames < 2:
        raise ValueError("reference audio is too short for stable normalization")

    preemphasis = np.float32(frontend["preemphasis"])
    processed = np.empty_like(audio)
    processed[0] = audio[0]
    processed[1:] = audio[1:] - preemphasis * audio[:-1]

    n_fft = int(frontend["fft_size"])
    win_length = int(frontend["window_samples"])
    hop = int(frontend["hop_samples"])
    if n_fft < win_length or (n_fft - win_length) % 2:
        raise ValueError("reference FFT/window centering geometry changed")

    centered = np.pad(
        processed,
        (n_fft // 2, n_fft // 2),
        mode="constant",
        constant_values=0.0,
    )
    frames = np.lib.stride_tricks.sliding_window_view(centered, n_fft)[::hop]
    expected_stft_frames = valid_frames + 1
    if frames.shape[0] != expected_stft_frames:
        raise ValueError(
            f"reference STFT frame count changed: {frames.shape[0]} "
            f"!= {expected_stft_frames}"
        )

    window = np.zeros((n_fft,), dtype=np.float32)
    offset = (n_fft - win_length) // 2
    window[offset : offset + win_length] = np.hanning(win_length).astype(np.float32)
    windowed = np.asarray(frames * window[None, :], dtype=np.float32)

    # NumPy FFT promotes float32 to complex128. Round back to complex64 before
    # the power calculation to match torch.stft(float32) arithmetic closely.
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1).astype(np.complex64)
    power = np.asarray(
        spectrum.real * spectrum.real + spectrum.imag * spectrum.imag,
        dtype=np.float32,
    )

    mel_spec = {"frontend": {
        "sample_rate_hz": int(frontend["sample_rate_hz"]),
        "fft_size": n_fft,
        "mel_bins": int(frontend["mel_bins"]),
        "fmin_hz": float(frontend["fmin_hz"]),
        "fmax_hz": float(frontend["fmax_hz"]),
    }}
    bank = np.asarray(mel_filterbank_slaney(mel_spec), dtype=np.float32)
    mel = np.asarray(bank @ power.T, dtype=np.float32)
    features = np.log(
        mel + np.float32(frontend["log_guard_value"])
    ).astype(np.float32)

    valid = features[:, :valid_frames]
    mean = valid.mean(axis=1, keepdims=True, dtype=np.float32)
    centered_valid = np.asarray(valid - mean, dtype=np.float32)
    variance = np.asarray(
        (centered_valid * centered_valid).sum(
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        / np.float32(valid_frames - 1),
        dtype=np.float32,
    )
    std = np.asarray(
        np.sqrt(variance).astype(np.float32)
        + np.float32(frontend["normalization_epsilon"]),
        dtype=np.float32,
    )
    features = np.asarray((features - mean) / std, dtype=np.float32)
    features[:, valid_frames:] = np.float32(frontend["pad_value"])

    if features.shape[1] > padded_frames:
        raise ValueError("reference padded feature target is shorter than STFT output")
    if features.shape[1] < padded_frames:
        features = np.pad(
            features,
            ((0, 0), (0, padded_frames - features.shape[1])),
            mode="constant",
            constant_values=float(frontend["pad_value"]),
        ).astype(np.float32)
    if features.shape != (int(frontend["mel_bins"]), padded_frames):
        raise ValueError(f"unexpected reference feature shape: {features.shape}")
    return features[None, :, :], valid_frames
