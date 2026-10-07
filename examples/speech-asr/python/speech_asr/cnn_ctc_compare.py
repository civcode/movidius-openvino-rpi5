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

    max_abs_reference = float(np.max(np.abs(left))) if left.size else 0.0
    max_abs_candidate = float(np.max(np.abs(right))) if right.size else 0.0
    max_abs_error = float(np.max(abs_diff)) if diff.size else 0.0
    result = {
        "shape": list(left.shape),
        "elements": int(left.size),
        "signed_mean_error": float(np.mean(diff)) if diff.size else 0.0,
        "mean_abs_error": float(np.mean(abs_diff)) if diff.size else 0.0,
        "rms_error": rms,
        "max_abs_error": max_abs_error,
        "reference_min": float(np.min(left)) if left.size else 0.0,
        "reference_max": float(np.max(left)) if left.size else 0.0,
        "candidate_min": float(np.min(right)) if right.size else 0.0,
        "candidate_max": float(np.max(right)) if right.size else 0.0,
        "max_abs_reference": max_abs_reference,
        "max_abs_candidate": max_abs_candidate,
        "max_abs_error_over_reference_max_abs": (
            max_abs_error / max(max_abs_reference, 1e-12)
        ),
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
            kth=ref_frames.shape[-1] - 2,
            axis=-1,
        )[:, -2:]
        top1 = np.max(top_two, axis=-1)
        top2 = np.min(top_two, axis=-1)
        reference_margin = top1 - top2
        mismatches = ~matches
        mismatch_margins = reference_margin[mismatches]

        ref_shifted = ref_frames.astype(np.float64) - np.max(
            ref_frames.astype(np.float64),
            axis=-1,
            keepdims=True,
        )
        cand_shifted = cand_frames.astype(np.float64) - np.max(
            cand_frames.astype(np.float64),
            axis=-1,
            keepdims=True,
        )
        ref_exp = np.exp(ref_shifted)
        cand_exp = np.exp(cand_shifted)
        ref_prob = ref_exp / np.sum(ref_exp, axis=-1, keepdims=True)
        cand_prob = cand_exp / np.sum(cand_exp, axis=-1, keepdims=True)
        prob_abs = np.abs(cand_prob - ref_prob)
        frame_total_variation = 0.5 * np.sum(prob_abs, axis=-1)

        result["frame_argmax_agreement"] = float(np.mean(matches))
        result["frame_argmax_matches"] = int(np.sum(matches))
        result["frame_count"] = int(ref_argmax.size)
        result["frame_argmax_mismatches"] = int(np.sum(mismatches))
        result["min_reference_top2_margin"] = float(
            np.min(reference_margin)
        )
        result["max_frame_abs_error"] = float(np.max(frame_abs_error))
        result["mean_softmax_abs_error"] = float(np.mean(prob_abs))
        result["max_softmax_abs_error"] = float(np.max(prob_abs))
        result["mean_frame_total_variation"] = float(
            np.mean(frame_total_variation)
        )
        result["max_frame_total_variation"] = float(
            np.max(frame_total_variation)
        )
        result["max_mismatched_reference_top2_margin"] = (
            float(np.max(mismatch_margins))
            if mismatch_margins.size
            else 0.0
        )
        result["mean_mismatched_reference_top2_margin"] = (
            float(np.mean(mismatch_margins))
            if mismatch_margins.size
            else 0.0
        )
    return result
