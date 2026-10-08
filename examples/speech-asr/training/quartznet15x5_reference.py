"""Exact-source QuartzNet15x5Base-En reference model and NeMo-style frontend."""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Sequence

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent

from cnn_ctc_v18 import CnnCtcV18
from pretrained_cnn_ctc_v18 import SOURCE_TOKENS, load_pretrained_quartznet
from speech_asr.cnn_ctc import conv1d_output_length
from speech_asr.cnn_ctc_frontend import mel_filterbank_slaney

DEFAULT_SPEC = SPEECH_ROOT / "models" / "quartznet15x5_nvidia_ref" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "quartznet15x5_nvidia_ref" / "vocab.json"


def load_reference_spec(path: pathlib.Path = DEFAULT_SPEC) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "speech-asr/pretrained-reference-spec" or value.get("version") != 1:
        raise ValueError("invalid pretrained-reference spec")
    if value.get("id") != "quartznet15x5_nvidia_ref":
        raise ValueError("unexpected pretrained-reference model id")
    if value["output_contract"] != {
        "name": "logits",
        "layout": "NTV",
        "dtype": "float32",
        "classes": 29,
        "blank_index": 28,
    }:
        raise ValueError("reference output contract changed")
    if value["qualification"].get("training_allowed") is not False:
        raise ValueError("reference qualification must remain zero-training")
    if value["qualification"].get("openvino_allowed") is not False:
        raise ValueError("reference qualification must not use OpenVINO")
    if value["qualification"].get("myriad_allowed") is not False:
        raise ValueError("reference qualification must not use MYRIAD")
    return value


def load_reference_vocab(path: pathlib.Path = DEFAULT_VOCAB) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = list(SOURCE_TOKENS)
    if value.get("schema") != "speech-asr/ctc-vocab" or value.get("version") != 1:
        raise ValueError("invalid reference CTC vocabulary")
    if value.get("tokens") != expected:
        raise ValueError("reference vocabulary no longer matches NVIDIA source order")
    if value.get("blank_index") != len(expected) - 1:
        raise ValueError("reference blank must remain the final NVIDIA source class")
    return value


class QuartzNet15x5Reference(CnnCtcV18):
    """The v18 QuartzNet topology with the untouched 29-class NVIDIA CTC head."""

    def __init__(self, spec: dict):
        if spec.get("id") != "quartznet15x5_nvidia_ref":
            raise ValueError("QuartzNet15x5Reference requires the reference spec")
        bridge = copy.deepcopy(spec)
        bridge["id"] = "cnn_ctc_v18"
        super().__init__(bridge, 29)


def load_reference_model(
    archive: pathlib.Path,
    *,
    spec: dict,
    vocab: dict,
) -> tuple[QuartzNet15x5Reference, dict]:
    torch.manual_seed(1337)
    model = QuartzNet15x5Reference(spec)
    imported = load_pretrained_quartznet(
        model,
        archive,
        list(vocab["tokens"]),
    )
    if imported["shared_decoder_symbol_count"] != 29:
        raise ValueError("reference import did not load all 29 source decoder rows")
    if imported["target_only_symbol_count"] != 0:
        raise ValueError("reference import must not retain randomly initialized target-only rows")
    if imported["target_only_symbols_random_init"]:
        raise ValueError("reference import contains non-source decoder symbols")
    model.eval()
    return model, imported


def nemo_reference_features(
    samples: Sequence[float],
    spec: dict,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Reproduce the historical NeMo FilterbankFeatures eval path.

    The source preprocessor uses centered STFT, a non-periodic Hann window,
    Slaney-normalized mel weights, per-feature sample standard deviation
    (ddof=1), masks the final centered STFT frame via floor(N/hop), and pads
    the feature time axis to a multiple of 16. Dither is disabled in eval.
    """

    frontend = spec["frontend"]
    if frontend["evaluation_dither"] != 0:
        raise ValueError("reference evaluation dither must remain disabled")
    audio = torch.as_tensor(np.asarray(samples, dtype=np.float32), device=device).reshape(1, -1)
    sample_count = int(audio.shape[1])
    hop = int(frontend["hop_samples"])
    valid_frames = sample_count // hop
    if valid_frames < 2:
        raise ValueError("reference audio is too short for stable per-feature normalization")

    preemphasis = float(frontend["preemphasis"])
    audio = torch.cat(
        (
            audio[:, :1],
            audio[:, 1:] - preemphasis * audio[:, :-1],
        ),
        dim=1,
    )
    window = torch.hann_window(
        int(frontend["window_samples"]),
        periodic=bool(frontend["hann_periodic"]),
        dtype=torch.float32,
        device=device,
    )
    spectrum = torch.stft(
        audio,
        n_fft=int(frontend["fft_size"]),
        hop_length=hop,
        win_length=int(frontend["window_samples"]),
        center=bool(frontend["stft_center"]),
        window=window,
        return_complex=True,
        pad_mode=str(frontend["stft_pad_mode"]),
    ).abs()
    if int(frontend["magnitude_power"]) != 2:
        raise ValueError("reference frontend requires power spectrogram")
    spectrum = spectrum.pow(2.0)

    mel_spec = {"frontend": {
        "sample_rate_hz": int(frontend["sample_rate_hz"]),
        "fft_size": int(frontend["fft_size"]),
        "mel_bins": int(frontend["mel_bins"]),
        "fmin_hz": float(frontend["fmin_hz"]),
        "fmax_hz": float(frontend["fmax_hz"]),
    }}
    bank = torch.from_numpy(mel_filterbank_slaney(mel_spec)).to(device=device, dtype=torch.float32)
    features = torch.matmul(bank.unsqueeze(0), spectrum)
    features = torch.log(features + float(frontend["log_guard_value"]))

    valid_frames = min(valid_frames, int(features.shape[-1]))
    valid = features[:, :, :valid_frames]
    mean = valid.mean(dim=2, keepdim=True)
    centered = valid - mean
    variance = centered.pow(2).sum(dim=2, keepdim=True) / float(valid_frames - 1)
    std = torch.sqrt(variance) + float(frontend["normalization_epsilon"])
    features = (features - mean) / std
    if valid_frames < features.shape[-1]:
        features[:, :, valid_frames:] = float(frontend["pad_value"])

    pad_to = int(frontend["pad_to"])
    remainder = int(features.shape[-1]) % pad_to
    if remainder:
        features = torch.nn.functional.pad(
            features,
            (0, pad_to - remainder),
            value=float(frontend["pad_value"]),
        )
    return features.contiguous(), valid_frames


def reference_output_length(feature_frames: int, spec: dict) -> int:
    layer = spec["network"]["prologue"]
    return conv1d_output_length(
        int(feature_frames),
        kernel=int(layer["kernel"]),
        stride=int(layer["stride"]),
        padding=int(layer["padding"]),
        dilation=int(layer["dilation"]),
    )
