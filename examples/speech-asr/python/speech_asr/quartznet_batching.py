"""Deterministic exact-shape batching for zero-training QuartzNet evaluation.

Only utterances whose independently normalized frontend tensors have the
same padded time dimension may share a batch. No temporal padding is added
across utterances, preserving the original temporal boundary contract.
"""

from __future__ import annotations

from collections import defaultdict

from .quartznet_reference_frontend import reference_feature_lengths


def exact_shape_batches(records: list[dict], spec: dict, batch_size: int) -> list[tuple[int, list[int]]]:
    """Return (padded time frames, manifest indices) in deterministic groups."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        _, padded_frames = reference_feature_lengths(int(record["sample_count"]), spec)
        groups[padded_frames].append(index)

    if batch_size == 1:
        # The reference evaluator's original manifest order is useful for
        # apples-to-apples timing and hypothesis parity comparisons.
        by_index = {
            index: padded for padded, indices in groups.items() for index in indices
        }
        return [(by_index[index], [index]) for index in range(len(records))]

    batches = []
    for frames, indices in groups.items():
        for start in range(0, len(indices), batch_size):
            batches.append((frames, indices[start : start + batch_size]))
    if sorted(index for _, batch in batches for index in batch) != list(range(len(records))):
        raise AssertionError("exact-shape batches must visit every sample exactly once")
    return batches


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("percentile requires data and a fraction in [0,1]")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    idx = int(pos)
    return ordered[idx] + (ordered[min(idx + 1, len(ordered) - 1)] - ordered[idx]) * (pos - idx)
