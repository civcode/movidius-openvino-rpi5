"""Numerical comparison helpers for cnn_ctc_v1 tensors."""

from __future__ import annotations

import math


def compare_arrays(reference, candidate) -> dict:
    import numpy as np

    left = np.asarray(reference, dtype=np.float32)
    right = np.asarray(candidate, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: reference {left.shape}, candidate {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("tensor comparison requires finite values")

    diff = right.astype(np.float64) - left.astype(np.float64)
    abs_diff = np.abs(diff)
    rms = math.sqrt(float(np.mean(diff * diff))) if diff.size else 0.0

    result = {
        "shape": list(left.shape),
        "elements": int(left.size),
        "signed_mean_error": float(np.mean(diff)) if diff.size else 0.0,
        "mean_abs_error": float(np.mean(abs_diff)) if diff.size else 0.0,
        "rms_error": rms,
        "max_abs_error": float(np.max(abs_diff)) if diff.size else 0.0,
    }
    if left.ndim == 3 and left.shape[0] == 1:
        ref_argmax = np.argmax(left[0], axis=-1)
        cand_argmax = np.argmax(right[0], axis=-1)
        result["frame_argmax_agreement"] = float(np.mean(ref_argmax == cand_argmax))
        result["frame_argmax_matches"] = int(np.sum(ref_argmax == cand_argmax))
        result["frame_count"] = int(ref_argmax.size)
    return result
