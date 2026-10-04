"""Parse and summarize OpenVINO 2020.3 speech_sample regression output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .evaluation import percentile

_UTTERANCE_RE = re.compile(r"^Utterance\s+(\d+)\s*:")
_FRAMES_RE = re.compile(r"Frames in utterance:\s*([0-9]+)", re.I)
_INFER_RE = re.compile(r"Average\s+(?:Infer|inference)\s+time per frame:\s*([0-9.eE+-]+)\s*ms", re.I)
_MAX_ERROR_RE = re.compile(r"max error:\s*([0-9.eE+-]+)", re.I)
_AVG_ERROR_RE = re.compile(r"avg error:\s*([0-9.eE+-]+)", re.I)
_RMS_ERROR_RE = re.compile(r"avg rms error:\s*([0-9.eE+-]+)", re.I)
_STDEV_ERROR_RE = re.compile(r"stdev error:\s*([0-9.eE+-]+)", re.I)
_LOAD_RE = re.compile(r"Model loading time\s*([0-9.eE+-]+)\s*ms", re.I)


@dataclass(frozen=True)
class UtteranceRegression:
    index: int
    frames: int
    infer_ms_per_frame: float
    max_error: float
    avg_error: float
    rms_error: float
    stdev_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "frames": self.frames,
            "infer_ms_per_frame": self.infer_ms_per_frame,
            "max_error": self.max_error,
            "avg_error": self.avg_error,
            "rms_error": self.rms_error,
            "stdev_error": self.stdev_error,
        }


def parse_speech_sample_output(text: str) -> dict[str, Any]:
    current: dict[str, Any] | None = None
    utterances: list[UtteranceRegression] = []
    model_load_ms = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        required = ("frames", "infer_ms_per_frame", "max_error", "avg_error", "rms_error", "stdev_error")
        missing = [key for key in required if key not in current]
        if missing:
            raise ValueError(f"utterance {current['index']} missing metrics: {', '.join(missing)}")
        utterances.append(UtteranceRegression(**current))
        current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _LOAD_RE.search(line)
        if match:
            model_load_ms = float(match.group(1))

        match = _UTTERANCE_RE.match(line)
        if match:
            finish()
            current = {"index": int(match.group(1))}
            continue
        if current is None:
            continue

        patterns = (
            ("frames", _FRAMES_RE, int),
            ("infer_ms_per_frame", _INFER_RE, float),
            ("max_error", _MAX_ERROR_RE, float),
            ("avg_error", _AVG_ERROR_RE, float),
            ("rms_error", _RMS_ERROR_RE, float),
            ("stdev_error", _STDEV_ERROR_RE, float),
        )
        for key, regex, convert in patterns:
            match = regex.search(line)
            if match:
                current[key] = convert(match.group(1))
                break
    finish()

    if not utterances:
        raise ValueError("no complete speech_sample utterance metrics found")

    total_frames = sum(item.frames for item in utterances)
    weighted_infer = (
        sum(item.frames * item.infer_ms_per_frame for item in utterances) / total_frames
        if total_frames else 0.0
    )
    infer_values = [item.infer_ms_per_frame for item in utterances]

    return {
        "model_load_ms": model_load_ms,
        "metrics": {
            "utterances": len(utterances),
            "total_frames": total_frames,
            "weighted_mean_infer_ms_per_frame": weighted_infer,
            "utterance_avg_infer_ms_per_frame_p50": percentile(infer_values, 0.50),
            "utterance_avg_infer_ms_per_frame_p95": percentile(infer_values, 0.95),
            "max_error_max": max(item.max_error for item in utterances),
            "avg_error_mean": sum(item.avg_error for item in utterances) / len(utterances),
            "rms_error_mean": sum(item.rms_error for item in utterances) / len(utterances),
            "failures": 0,
            "per_utterance": [item.to_dict() for item in utterances],
        },
    }
