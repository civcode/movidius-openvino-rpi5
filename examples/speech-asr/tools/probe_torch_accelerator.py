#!/usr/bin/env python3
"""Probe a CUDA or ROCm PyTorch accelerator and run a short GEMM timing."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def observed_backend() -> str | None:
    if not torch.cuda.is_available():
        return None
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "rocm"), required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if args.device_index < 0:
        raise SystemExit("--device-index must be >= 0")
    backend = observed_backend()
    if backend != args.device:
        raise SystemExit(
            f"requested {args.device}, but PyTorch backend is {backend or 'unavailable'}; "
            f"torch={torch.__version__} cuda={getattr(torch.version, 'cuda', None)!r} "
            f"hip={getattr(torch.version, 'hip', None)!r}"
        )
    if args.device_index >= torch.cuda.device_count():
        raise SystemExit(
            f"device index {args.device_index} out of range; "
            f"visible={torch.cuda.device_count()}"
        )
    if args.matrix_size < 256 or args.iterations < 1:
        raise SystemExit("invalid benchmark dimensions")

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(args.device_index)

    a = torch.randn(
        (args.matrix_size, args.matrix_size),
        device=device,
        dtype=torch.float32,
    )
    b = torch.randn_like(a)
    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize(device)

    times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        _ = a @ b
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000.0)

    n = args.matrix_size
    tflops = [
        (2.0 * n * n * n) / (ms / 1000.0) / 1.0e12
        for ms in times
    ]
    result = {
        "status": "pass",
        "requested_backend": args.device,
        "observed_backend": backend,
        "device_index": args.device_index,
        "device_name": torch.cuda.get_device_name(args.device_index),
        "device_total_memory_bytes": int(props.total_memory),
        "visible_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_hip_version": getattr(torch.version, "hip", None),
        "matrix_size": n,
        "iterations": args.iterations,
        "gemm_ms_median": statistics.median(times),
        "gemm_ms_min": min(times),
        "gemm_tflops_median": statistics.median(tflops),
        "gemm_tflops_max": max(tflops),
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
