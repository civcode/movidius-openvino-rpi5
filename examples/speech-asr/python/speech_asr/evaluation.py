"""Deterministic speech recognition scoring and timing metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from .text import normalize_text_v1

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        # A finite convention is more useful for machine-readable benchmark
        # results than NaN/Inf when the normalized reference is empty.
        return self.errors / max(1, self.reference_units)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["errors"] = self.errors
        result["rate"] = self.rate
        return result


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Return Levenshtein S/D/I counts with deterministic tie breaking.

    Ties are resolved substitution, then deletion, then insertion. The total
    edit distance is invariant; the fixed tie order keeps detailed counts
    reproducible across implementations.
    """

    # Each cell stores (distance, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref_unit in enumerate(reference, start=1):
        current = [(i, 0, i, 0)]
        for j, hyp_unit in enumerate(hypothesis, start=1):
            if ref_unit == hyp_unit:
                current.append(previous[j - 1])
                continue

            diagonal = previous[j - 1]
            deletion = previous[j]
            insertion = current[j - 1]
            candidates = (
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3], 0),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3], 1),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1, 2),
            )
            best = min(candidates, key=lambda value: (value[0], value[4]))
            current.append(best[:4])
        previous = current

    distance, substitutions, deletions, insertions = previous[-1]
    assert distance == substitutions + deletions + insertions
    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_units=len(reference),
    )


def word_error_counts(reference: str, hypothesis: str) -> EditCounts:
    ref = normalize_text_v1(reference).split()
    hyp = normalize_text_v1(hypothesis).split()
    return edit_counts(ref, hyp)


def character_error_counts(reference: str, hypothesis: str) -> EditCounts:
    # Spaces are formatting rather than lexical characters for this benchmark.
    ref = list(normalize_text_v1(reference).replace(" ", ""))
    hyp = list(normalize_text_v1(hypothesis).replace(" ", ""))
    return edit_counts(ref, hyp)


def score_transcript(reference: str, hypothesis: str) -> dict[str, Any]:
    words = word_error_counts(reference, hypothesis)
    chars = character_error_counts(reference, hypothesis)
    return {
        "reference": normalize_text_v1(reference),
        "hypothesis": normalize_text_v1(hypothesis),
        "wer": words.rate,
        "cer": chars.rate,
        "word_edits": words.to_dict(),
        "character_edits": chars.to_dict(),
    }


def percentile(values: Iterable[float], quantile: float) -> float:
    """Linear-interpolated percentile for quantile in [0, 1]."""

    data = sorted(float(value) for value in values)
    if not data:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in range 0..1")
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(data) - 1)
    fraction = position - lower
    return data[lower] + (data[upper] - data[lower]) * fraction


def latency_summary_ms(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if any(value < 0 for value in data):
        raise ValueError("latency values must be non-negative")
    return {
        "p50_ms": percentile(data, 0.50),
        "p95_ms": percentile(data, 0.95),
    }


def realtime_factor(processing_seconds: float, audio_samples: int) -> float:
    if processing_seconds < 0:
        raise ValueError("processing_seconds must be non-negative")
    if audio_samples <= 0:
        raise ValueError("audio_samples must be positive")
    return processing_seconds / (audio_samples / SAMPLE_RATE)


def timing_error_ms(reference_sample: int, observed_sample: int) -> float:
    return (observed_sample - reference_sample) * 1000.0 / SAMPLE_RATE
