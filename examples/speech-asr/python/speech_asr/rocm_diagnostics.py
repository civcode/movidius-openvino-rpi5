"""Optional, hardware-light diagnostics for ROCm QuartzNet inference."""

from __future__ import annotations

import os
import statistics


def parse_cpuset(value: str) -> set[int]:
    """Parse taskset-style lists such as 0-3,8,10-11."""
    result: set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            raise ValueError("empty CPU set component")
        parts = chunk.split("-")
        if len(parts) == 1:
            if not parts[0].isdigit():
                raise ValueError(f"invalid CPU id: {chunk}")
            first = last = int(parts[0])
        elif len(parts) == 2 and all(part.isdigit() for part in parts):
            first, last = (int(part) for part in parts)
        else:
            raise ValueError(f"invalid CPU range: {chunk}")
        if last < first:
            raise ValueError(f"reversed CPU range: {chunk}")
        result.update(range(first, last + 1))
    if not result:
        raise ValueError("CPU set must not be empty")
    return result


def inspect_rocm_affinity(expected_cpus: set[int]) -> dict:
    """Fail if the ROCm bootstrap or any observed Linux thread loses CPUs."""
    if os.environ.get("SPEECH_ROCM_AFFINITY_RESTORED") != "1":
        raise RuntimeError("ROCm affinity bootstrap marker is missing")
    expected = tuple(sorted(expected_cpus))
    if not expected:
        raise ValueError("expected CPU set is empty")
    main_mask = tuple(sorted(os.sched_getaffinity(0)))
    if main_mask != expected:
        raise RuntimeError(
            f"ROCm main-thread affinity drift: expected={expected}, actual={main_mask}"
        )
    counts: dict[str, int] = {}
    task_ids = [int(tid) for tid in os.listdir("/proc/self/task") if tid.isdigit()]
    for tid in task_ids:
        try:
            mask = tuple(sorted(os.sched_getaffinity(tid)))
        except ProcessLookupError:
            continue  # A thread may finish between listing and inspection.
        if mask != expected:
            raise RuntimeError(
                f"ROCm thread affinity drift: tid={tid}, "
                f"expected={expected}, actual={mask}"
            )
        key = ",".join(str(cpu) for cpu in mask)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        raise RuntimeError("no ROCm threads could be inspected")
    return {
        "main_thread_affinity": list(main_mask),
        "thread_affinity_mask_counts": counts,
        "thread_count_observed": sum(counts.values()),
    }


def summarize_model_timings(latencies_ms: list[float], audio_seconds: float) -> dict:
    """Summarize synchronized acoustic-model timings, excluding the frontend."""
    if not latencies_ms or audio_seconds <= 0:
        raise ValueError("model timings and audio duration must be positive")
    ordered = sorted(latencies_ms)
    position = 0.95 * (len(ordered) - 1)
    left = int(position)
    p95 = ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (position - left)
    total_seconds = sum(ordered) / 1000.0
    return {
        "utterances": len(ordered),
        "audio_seconds": audio_seconds,
        "model_inference_ms_p50": statistics.median(ordered),
        "model_inference_ms_p95": p95,
        "model_inference_seconds_total": total_seconds,
        "model_inference_rtf": total_seconds / audio_seconds,
        "scope": "acoustic-model-forward-only; excludes frontend, audio IO, decoding and model load",
    }
