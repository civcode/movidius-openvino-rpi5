#!/usr/bin/env python3
"""ROCm Python bootstrap that restores the requested CPU affinity after HIP init."""

from __future__ import annotations

import os
import pathlib
import runpy
import sys
import time


def parse_cpuset(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("empty CPU-set component")
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            stop = int(right)
            if start < 0 or stop < start:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(start, stop + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"invalid CPU id: {part}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU set must not be empty")
    return cpus


def restore_all_threads(cpus: set[int]) -> dict:
    """Apply one affinity mask to every currently existing Linux task."""

    task_dir = pathlib.Path("/proc/self/task")
    applied = 0
    disappeared = 0
    failures: list[str] = []

    tids = []
    if task_dir.is_dir():
        tids = [int(item.name) for item in task_dir.iterdir() if item.name.isdigit()]
    if not tids:
        tids = [0]

    for tid in tids:
        try:
            os.sched_setaffinity(tid, cpus)
            applied += 1
        except ProcessLookupError:
            disappeared += 1
        except OSError as exc:
            failures.append(f"{tid}:{exc}")

    if failures:
        raise RuntimeError(
            "failed to restore ROCm thread affinity: " + ", ".join(failures)
        )
    return {
        "applied": applied,
        "disappeared": disappeared,
    }


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python-rocm-affinity.py TARGET.py [args...]")

    cpuset_text = os.environ.get("SPEECH_ROCM_CPUSET", "0-15")
    threads_text = os.environ.get("SPEECH_ROCM_THREADS", "16")
    cpus = parse_cpuset(cpuset_text)
    threads = int(threads_text)
    if threads < 1:
        raise ValueError("SPEECH_ROCM_THREADS must be positive")

    import torch

    if not getattr(torch.version, "hip", None):
        raise RuntimeError(
            "ROCm bootstrap requires a HIP PyTorch build; "
            f"torch.version.hip={getattr(torch.version, 'hip', None)!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm bootstrap cannot see a usable HIP GPU")

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(min(threads, len(cpus)))
    except RuntimeError:
        pass

    # HIP/ROCr initialization can rewrite Linux task affinities. Restore every
    # task after initialization, then repeat once to catch helper threads that
    # were still being created while the first pass ran.
    first = restore_all_threads(cpus)
    time.sleep(0.02)
    second = restore_all_threads(cpus)
    os.sched_setaffinity(0, cpus)

    os.environ["SPEECH_ROCM_AFFINITY_RESTORED"] = "1"
    os.environ["SPEECH_ROCM_AFFINITY_CPUS"] = cpuset_text
    os.environ["SPEECH_ROCM_AFFINITY_THREADS"] = str(threads)
    os.environ["SPEECH_ROCM_AFFINITY_TASKS"] = str(second["applied"])

    target = pathlib.Path(sys.argv[1]).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)

    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
