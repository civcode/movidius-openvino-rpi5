"""Fixed-T=512 streaming geometry for the qualified QuartzNet reference model.

The MYRIAD graph remains [1,64,512] for every request. Source audio is divided
on the model output lattice so overlapping window logits can be stitched before
a single global CTC collapse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .quartznet_reference_frontend import (
    nemo_reference_features_numpy,
    reference_feature_lengths,
)

FIXED_TENSOR_FRAMES = 512
FULL_VALID_FEATURE_FRAMES = 511
FEATURE_HOP_SAMPLES = 160
OUTPUT_STRIDE_FEATURE_FRAMES = 2
OUTPUT_STRIDE_SAMPLES = FEATURE_HOP_SAMPLES * OUTPUT_STRIDE_FEATURE_FRAMES
FULL_WINDOW_SAMPLES = FULL_VALID_FEATURE_FRAMES * FEATURE_HOP_SAMPLES
FULL_OUTPUT_FRAMES = 256
DEFAULT_HOP_OUTPUT_FRAMES = 128


@dataclass(frozen=True)
class Fixed512Window:
    index: int
    start_output_frame: int
    end_output_frame: int
    start_sample: int
    end_sample: int
    valid_feature_frames: int
    valid_output_frames: int

    @property
    def sample_count(self) -> int:
        return self.end_sample - self.start_sample


def quartznet_output_frames(valid_feature_frames: int, spec: dict) -> int:
    if valid_feature_frames < 1:
        raise ValueError("valid_feature_frames must be positive")
    layer = spec["network"]["prologue"]
    kernel = int(layer["kernel"])
    stride = int(layer["stride"])
    padding = int(layer["padding"])
    dilation = int(layer["dilation"])
    return (
        valid_feature_frames
        + 2 * padding
        - dilation * (kernel - 1)
        - 1
    ) // stride + 1


def validate_fixed512_geometry(spec: dict) -> None:
    frontend = spec["frontend"]
    if int(frontend["hop_samples"]) != FEATURE_HOP_SAMPLES:
        raise ValueError("fixed512 policy requires a 160-sample frontend hop")
    if int(frontend["pad_to"]) != 16:
        raise ValueError("fixed512 policy requires pad_to=16")

    valid, tensor = reference_feature_lengths(FULL_WINDOW_SAMPLES, spec)
    if valid != FULL_VALID_FEATURE_FRAMES or tensor != FIXED_TENSOR_FRAMES:
        raise ValueError(
            "fixed512 source window no longer maps to 511 valid / 512 tensor frames"
        )
    if quartznet_output_frames(FULL_VALID_FEATURE_FRAMES, spec) != FULL_OUTPUT_FRAMES:
        raise ValueError("fixed512 QuartzNet output geometry changed")
    if quartznet_output_frames(FIXED_TENSOR_FRAMES, spec) != FULL_OUTPUT_FRAMES:
        raise ValueError("fixed512 carrier output geometry changed")


def plan_fixed512_windows(
    sample_count: int,
    spec: dict,
    *,
    hop_output_frames: int = DEFAULT_HOP_OUTPUT_FRAMES,
) -> tuple[Fixed512Window, ...]:
    """Cover one utterance with fixed-T=512 inference windows.

    Window starts are aligned to the QuartzNet output lattice. With the default
    hop of 128 output frames, full windows overlap by approximately 50%.
    """

    validate_fixed512_geometry(spec)
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 1 <= hop_output_frames <= FULL_OUTPUT_FRAMES:
        raise ValueError(
            f"hop_output_frames must be in 1..{FULL_OUTPUT_FRAMES}"
        )

    global_valid_features, _ = reference_feature_lengths(sample_count, spec)
    global_output_frames = quartznet_output_frames(global_valid_features, spec)

    starts = [0]
    while starts[-1] + FULL_OUTPUT_FRAMES < global_output_frames:
        starts.append(starts[-1] + hop_output_frames)

    windows: list[Fixed512Window] = []
    for index, start_output in enumerate(starts):
        start_sample = start_output * OUTPUT_STRIDE_SAMPLES
        end_sample = min(sample_count, start_sample + FULL_WINDOW_SAMPLES)
        segment_samples = end_sample - start_sample
        if segment_samples < 2 * FEATURE_HOP_SAMPLES:
            raise ValueError(
                "fixed512 planner produced a segment too short for frontend normalization"
            )
        valid_features, tensor_frames = reference_feature_lengths(
            segment_samples,
            spec,
        )
        if tensor_frames > FIXED_TENSOR_FRAMES:
            raise ValueError(
                f"fixed512 planner overflow: tensor_frames={tensor_frames}"
            )
        valid_outputs = quartznet_output_frames(valid_features, spec)
        end_output = min(global_output_frames, start_output + valid_outputs)
        if end_output <= start_output:
            raise ValueError("fixed512 planner produced an empty output interval")
        windows.append(
            Fixed512Window(
                index=index,
                start_output_frame=start_output,
                end_output_frame=end_output,
                start_sample=start_sample,
                end_sample=end_sample,
                valid_feature_frames=valid_features,
                valid_output_frames=valid_outputs,
            )
        )

    if windows[0].start_output_frame != 0:
        raise AssertionError("fixed512 window coverage does not begin at frame zero")
    coverage = np.zeros((global_output_frames,), dtype=np.int16)
    for window in windows:
        coverage[window.start_output_frame : window.end_output_frame] += 1
    if np.any(coverage == 0):
        missing = np.flatnonzero(coverage == 0)
        raise AssertionError(
            f"fixed512 window coverage has gaps; first missing frame={int(missing[0])}"
        )
    return tuple(windows)


def fixed512_features(
    samples: Sequence[float],
    spec: dict,
) -> tuple[np.ndarray, int, int]:
    """Create one fixed [1,64,512] tensor from an inference-window waveform."""

    features, valid_feature_frames = nemo_reference_features_numpy(samples, spec)
    tensor_frames = int(features.shape[2])
    if tensor_frames > FIXED_TENSOR_FRAMES:
        raise ValueError(
            f"fixed512 feature tensor overflow: {tensor_frames} > {FIXED_TENSOR_FRAMES}"
        )
    if tensor_frames < FIXED_TENSOR_FRAMES:
        features = np.pad(
            features,
            ((0, 0), (0, 0), (0, FIXED_TENSOR_FRAMES - tensor_frames)),
            mode="constant",
            constant_values=float(spec["frontend"]["pad_value"]),
        ).astype(np.float32)
    features = np.ascontiguousarray(features, dtype=np.float32)
    valid_output_frames = quartznet_output_frames(valid_feature_frames, spec)
    return features, valid_feature_frames, valid_output_frames


class Fixed512LogitStitcher:
    """Choose each global CTC frame from the window closest to its center."""

    def __init__(self, output_frames: int, classes: int):
        if output_frames < 1 or classes < 1:
            raise ValueError("output_frames and classes must be positive")
        self.output_frames = output_frames
        self.classes = classes
        self._logits = np.empty((output_frames, classes), dtype=np.float32)
        self._distance = np.full((output_frames,), np.inf, dtype=np.float64)
        self._owner = np.full((output_frames,), -1, dtype=np.int32)

    def add(
        self,
        window: Fixed512Window,
        logits: np.ndarray,
    ) -> None:
        values = np.asarray(logits, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.classes:
            raise ValueError(
                f"window logits must have shape [T,{self.classes}], got {values.shape}"
            )
        if values.shape[0] < window.valid_output_frames:
            raise ValueError("window logits are shorter than valid_output_frames")

        center = (FULL_OUTPUT_FRAMES - 1) / 2.0
        usable = min(
            window.valid_output_frames,
            self.output_frames - window.start_output_frame,
        )
        for local_index in range(usable):
            global_index = window.start_output_frame + local_index
            distance = abs(local_index - center)
            if distance < self._distance[global_index]:
                self._logits[global_index, :] = values[local_index, :]
                self._distance[global_index] = distance
                self._owner[global_index] = window.index

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        missing = np.flatnonzero(self._owner < 0)
        if missing.size:
            raise ValueError(
                f"stitched logits missing global output frame {int(missing[0])}"
            )
        return self._logits.copy(), self._owner.copy()


def fixed512_policy_dict(hop_output_frames: int = DEFAULT_HOP_OUTPUT_FRAMES) -> dict:
    if not 1 <= hop_output_frames <= FULL_OUTPUT_FRAMES:
        raise ValueError(
            f"hop_output_frames must be in 1..{FULL_OUTPUT_FRAMES}"
        )
    hop_samples = hop_output_frames * OUTPUT_STRIDE_SAMPLES
    return {
        "tensor_feature_frames": FIXED_TENSOR_FRAMES,
        "full_window_valid_feature_frames": FULL_VALID_FEATURE_FRAMES,
        "full_window_samples": FULL_WINDOW_SAMPLES,
        "full_window_ms": FULL_WINDOW_SAMPLES * 1000.0 / 16000.0,
        "full_window_output_frames": FULL_OUTPUT_FRAMES,
        "hop_output_frames": hop_output_frames,
        "hop_samples": hop_samples,
        "hop_ms": hop_samples * 1000.0 / 16000.0,
        "overlap_samples": FULL_WINDOW_SAMPLES - hop_samples,
        "overlap_ms": (
            (FULL_WINDOW_SAMPLES - hop_samples) * 1000.0 / 16000.0
        ),
        "ownership": "closest_fixed_window_center",
        "decode": "single_global_ctc_collapse_after_logit_stitch",
    }
