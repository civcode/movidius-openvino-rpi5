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
        ref_frames = left[0]
        cand_frames = right[0]
        ref_argmax = np.argmax(ref_frames, axis=-1)
        cand_argmax = np.argmax(cand_frames, axis=-1)
        matches = ref_argmax == cand_argmax
        frame_abs_error = np.max(
            np.abs(cand_frames.astype(np.float64) - ref_frames.astype(np.float64)),
            axis=-1,
        )
        top_two = np.partition(
            ref_frames.astype(np.float64),
            kth=-2,
            axis=-1,
        )[:, -2:]
        top1 = np.max(top_two, axis=-1)
        top2 = np.min(top_two, axis=-1)
        reference_margin = top1 - top2

        # If every candidate logit differs from reference by at most e, the
        # top two can exchange order only when the reference margin is <= 2e.
        # This distinguishes harmless FP16 tie-breaking from a disagreement
        # too large to be explained by the measured numerical error.
        explainable = reference_margin <= (2.0 * frame_abs_error + 1e-12)
        mismatches = ~matches
        unexplained = mismatches & ~explainable

        result["frame_argmax_agreement"] = float(np.mean(matches))
        result["frame_argmax_matches"] = int(np.sum(matches))
        result["frame_count"] = int(ref_argmax.size)
        result["frame_argmax_mismatches"] = int(np.sum(mismatches))
        result["frame_argmax_explainable_mismatches"] = int(
            np.sum(mismatches & explainable)
        )
        result["frame_argmax_unexplained_mismatches"] = int(
            np.sum(unexplained)
        )
        result["min_reference_top2_margin"] = float(
            np.min(reference_margin)
        )
        result["max_frame_abs_error"] = float(np.max(frame_abs_error))
    return result
