"""Deterministic recorded-audio streaming primitives for speech ASR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from .audio import CANONICAL_SAMPLE_RATE, CanonicalAudio
from .evaluation import latency_summary_ms, score_transcript
from .text import normalize_text_v1

SAMPLES_PER_MS = CANONICAL_SAMPLE_RATE // 1000


@dataclass(frozen=True)
class StreamingConfig:
    """Explicit runtime streaming parameters.

    Milliseconds are authoritative. Sample counts are derived at 16 kHz.
    """

    chunk_ms: int = 320
    overlap_ms: int = 80
    left_context_ms: int = 160
    right_context_ms: int = 80
    stabilization_repeats: int = 2

    def __post_init__(self) -> None:
        for name in ("chunk_ms", "overlap_ms", "left_context_ms", "right_context_ms"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        if self.overlap_ms >= self.chunk_ms:
            raise ValueError("overlap_ms must be smaller than chunk_ms")
        if not isinstance(self.stabilization_repeats, int) or self.stabilization_repeats < 1:
            raise ValueError("stabilization_repeats must be a positive integer")

    @property
    def chunk_samples(self) -> int:
        return self.chunk_ms * SAMPLES_PER_MS

    @property
    def overlap_samples(self) -> int:
        return self.overlap_ms * SAMPLES_PER_MS

    @property
    def hop_samples(self) -> int:
        return self.chunk_samples - self.overlap_samples

    @property
    def left_context_samples(self) -> int:
        return self.left_context_ms * SAMPLES_PER_MS

    @property
    def right_context_samples(self) -> int:
        return self.right_context_ms * SAMPLES_PER_MS

    def to_dict(self) -> dict:
        return {
            "sample_rate_hz": CANONICAL_SAMPLE_RATE,
            "chunk_ms": self.chunk_ms,
            "overlap_ms": self.overlap_ms,
            "left_context_ms": self.left_context_ms,
            "right_context_ms": self.right_context_ms,
            "stabilization_repeats": self.stabilization_repeats,
            "derived": {
                "chunk_samples": self.chunk_samples,
                "overlap_samples": self.overlap_samples,
                "hop_samples": self.hop_samples,
                "left_context_samples": self.left_context_samples,
                "right_context_samples": self.right_context_samples,
            },
        }


@dataclass(frozen=True)
class StreamChunk:
    index: int
    nominal_start_sample: int
    nominal_end_sample: int
    inference_start_sample: int
    inference_end_sample: int
    emit_start_sample: int
    emit_end_sample: int
    available_sample: int
    is_final: bool
    audio: CanonicalAudio

    @property
    def emitted_sample_count(self) -> int:
        return self.emit_end_sample - self.emit_start_sample


class SampleRingBuffer:
    """Bounded float-sample buffer addressed by absolute sample index."""

    def __init__(self, capacity_samples: int):
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self.capacity_samples = capacity_samples
        self._base_sample = 0
        self._next_sample = 0
        self._samples: list[float] = []

    @property
    def start_sample(self) -> int:
        return self._base_sample

    @property
    def end_sample(self) -> int:
        return self._next_sample

    def append(self, samples: Iterable[float]) -> None:
        values = [float(value) for value in samples]
        self._samples.extend(values)
        self._next_sample += len(values)
        overflow = len(self._samples) - self.capacity_samples
        if overflow > 0:
            del self._samples[:overflow]
            self._base_sample += overflow

    def slice(self, start_sample: int, end_sample: int) -> tuple[float, ...]:
        if start_sample < self._base_sample or end_sample > self._next_sample:
            raise ValueError("requested samples are outside the retained ring-buffer range")
        if end_sample < start_sample:
            raise ValueError("end_sample must be >= start_sample")
        left = start_sample - self._base_sample
        right = end_sample - self._base_sample
        return tuple(self._samples[left:right])


def plan_chunks(audio: CanonicalAudio, config: StreamingConfig) -> tuple[StreamChunk, ...]:
    """Plan deterministic overlapping chunks with midpoint emission ownership."""

    total = audio.sample_count
    if total == 0:
        return ()

    starts = list(range(0, total, config.hop_samples))
    ends = [min(start + config.chunk_samples, total) for start in starts]

    chunks: list[StreamChunk] = []
    for index, (start, end) in enumerate(zip(starts, ends)):
        if index == 0:
            emit_start = 0
        else:
            overlap_start = start
            overlap_end = ends[index - 1]
            actual_overlap = max(0, overlap_end - overlap_start)
            emit_start = overlap_start + actual_overlap // 2

        if index + 1 == len(starts):
            emit_end = total
        else:
            overlap_start = starts[index + 1]
            overlap_end = end
            actual_overlap = max(0, overlap_end - overlap_start)
            emit_end = overlap_start + actual_overlap // 2

        inference_start = max(0, start - config.left_context_samples)
        inference_end = min(total, end + config.right_context_samples)
        chunk_audio = CanonicalAudio(audio.samples[inference_start:inference_end])
        chunks.append(
            StreamChunk(
                index=index,
                nominal_start_sample=start,
                nominal_end_sample=end,
                inference_start_sample=inference_start,
                inference_end_sample=inference_end,
                emit_start_sample=emit_start,
                emit_end_sample=emit_end,
                available_sample=inference_end,
                is_final=index + 1 == len(starts),
                audio=chunk_audio,
            )
        )

    if chunks[0].emit_start_sample != 0 or chunks[-1].emit_end_sample != total:
        raise AssertionError("chunk emission ownership does not cover source boundaries")
    for left, right in zip(chunks, chunks[1:]):
        if left.emit_end_sample != right.emit_start_sample:
            raise AssertionError("chunk emission ownership has a gap or overlap")
    return tuple(chunks)


class VoiceActivityDetector(Protocol):
    def is_speech(self, samples: Sequence[float]) -> bool:
        ...


@dataclass(frozen=True)
class EnergyVad:
    rms_threshold: float = 0.01

    def __post_init__(self) -> None:
        if not math.isfinite(self.rms_threshold) or self.rms_threshold < 0:
            raise ValueError("rms_threshold must be finite and non-negative")

    def rms(self, samples: Sequence[float]) -> float:
        if not samples:
            return 0.0
        return math.sqrt(sum(float(value) ** 2 for value in samples) / len(samples))

    def is_speech(self, samples: Sequence[float]) -> bool:
        return self.rms(samples) >= self.rms_threshold


@dataclass(frozen=True)
class DecodedHypothesis:
    text: str
    source_end_sample: int


class ChunkDecoder(Protocol):
    def decode(self, chunk: StreamChunk) -> DecodedHypothesis | None:
        ...


@dataclass(frozen=True)
class ScriptedUpdate:
    available_sample: int
    source_end_sample: int
    text: str

    def __post_init__(self) -> None:
        if self.available_sample < 0 or self.source_end_sample < 0:
            raise ValueError("scripted update sample indices must be non-negative")
        if self.source_end_sample > self.available_sample:
            raise ValueError("source_end_sample cannot be after available_sample")


class ScriptedCumulativeDecoder:
    """Deterministic decoder fixture for recorded replay and tests.

    It represents cumulative decoder hypotheses becoming available at declared
    sample positions. It is not a speech model.
    """

    def __init__(self, updates: Sequence[ScriptedUpdate]):
        self._updates = tuple(sorted(updates, key=lambda item: item.available_sample))
        for left, right in zip(self._updates, self._updates[1:]):
            if left.available_sample == right.available_sample:
                raise ValueError("scripted updates must have unique available_sample values")
        self._cursor = 0

    def decode(self, chunk: StreamChunk) -> DecodedHypothesis | None:
        latest: ScriptedUpdate | None = None
        while (
            self._cursor < len(self._updates)
            and self._updates[self._cursor].available_sample <= chunk.available_sample
        ):
            latest = self._updates[self._cursor]
            self._cursor += 1
        if latest is None:
            return None
        return DecodedHypothesis(
            text=latest.text,
            source_end_sample=latest.source_end_sample,
        )


@dataclass(frozen=True)
class TokenStability:
    index: int
    token: str
    first_seen_sample: int
    stable_sample: int

    @property
    def delay_ms(self) -> float:
        return (self.stable_sample - self.first_seen_sample) * 1000.0 / CANONICAL_SAMPLE_RATE


class TranscriptStabilizer:
    """Stabilize cumulative hypotheses after N identical consecutive prefixes."""

    def __init__(self, repeats: int = 2):
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        self.repeats = repeats
        self._history: list[tuple[str, ...]] = []
        self._first_seen: dict[int, tuple[str, int]] = {}
        self._stable: list[str] = []

    @property
    def stable_tokens(self) -> tuple[str, ...]:
        return tuple(self._stable)

    @staticmethod
    def _common_prefix(hypotheses: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
        if not hypotheses:
            return ()
        limit = min(len(value) for value in hypotheses)
        count = 0
        for index in range(limit):
            token = hypotheses[0][index]
            if all(value[index] == token for value in hypotheses[1:]):
                count += 1
            else:
                break
        return hypotheses[0][:count]

    def update(self, text: str, observed_sample: int) -> tuple[TokenStability, ...]:
        tokens = tuple(normalize_text_v1(text).split())

        for index, token in enumerate(tokens):
            previous = self._first_seen.get(index)
            if previous is None or previous[0] != token:
                self._first_seen[index] = (token, observed_sample)
        for index in tuple(self._first_seen):
            if index >= len(tokens):
                del self._first_seen[index]

        if tuple(tokens[: len(self._stable)]) != tuple(self._stable):
            raise ValueError("decoder revised a token after it was stabilized")

        self._history.append(tokens)
        if len(self._history) > self.repeats:
            self._history.pop(0)
        if len(self._history) < self.repeats:
            return ()

        prefix = self._common_prefix(self._history)
        newly_stable: list[TokenStability] = []
        for index in range(len(self._stable), len(prefix)):
            token = prefix[index]
            first_token, first_sample = self._first_seen[index]
            if first_token != token:
                raise AssertionError("stability first-seen token mismatch")
            self._stable.append(token)
            newly_stable.append(
                TokenStability(
                    index=index,
                    token=token,
                    first_seen_sample=first_sample,
                    stable_sample=observed_sample,
                )
            )
        return tuple(newly_stable)

    def finalize(self, text: str, observed_sample: int) -> tuple[TokenStability, ...]:
        tokens = tuple(normalize_text_v1(text).split())
        if tuple(tokens[: len(self._stable)]) != tuple(self._stable):
            raise ValueError("final hypothesis revises a stabilized token")
        newly_stable: list[TokenStability] = []
        for index in range(len(self._stable), len(tokens)):
            token = tokens[index]
            previous = self._first_seen.get(index)
            first_sample = observed_sample
            if previous is not None and previous[0] == token:
                first_sample = previous[1]
            self._stable.append(token)
            newly_stable.append(
                TokenStability(
                    index=index,
                    token=token,
                    first_seen_sample=first_sample,
                    stable_sample=observed_sample,
                )
            )
        return tuple(newly_stable)


@dataclass(frozen=True)
class TranscriptEvent:
    kind: str
    text: str
    observed_sample: int
    source_end_sample: int
    stable_token_count: int
    newly_stable: tuple[TokenStability, ...]

    @property
    def latency_ms(self) -> float:
        return (self.observed_sample - self.source_end_sample) * 1000.0 / CANONICAL_SAMPLE_RATE

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "text": self.text,
            "observed_sample": self.observed_sample,
            "source_end_sample": self.source_end_sample,
            "latency_ms": self.latency_ms,
            "stable_token_count": self.stable_token_count,
            "newly_stable": [
                {
                    "index": item.index,
                    "token": item.token,
                    "first_seen_sample": item.first_seen_sample,
                    "stable_sample": item.stable_sample,
                    "delay_ms": item.delay_ms,
                }
                for item in self.newly_stable
            ],
        }


def compare_offline_streaming(offline_text: str, streaming_text: str) -> dict:
    scored = score_transcript(offline_text, streaming_text)
    return {
        "offline_text": scored["reference"],
        "streaming_text": scored["hypothesis"],
        "exact_match": scored["reference"] == scored["hypothesis"],
        "wer": scored["wer"],
        "cer": scored["cer"],
        "word_edits": scored["word_edits"],
        "character_edits": scored["character_edits"],
    }


def replay_recorded_audio(
    audio: CanonicalAudio,
    decoder: ChunkDecoder,
    *,
    config: StreamingConfig,
    offline_text: str | None = None,
    vad: VoiceActivityDetector | None = None,
    vad_gate: bool = False,
) -> dict:
    """Replay canonical recorded audio through a cumulative streaming decoder."""

    chunks = plan_chunks(audio, config)
    stabilizer = TranscriptStabilizer(config.stabilization_repeats)
    events: list[TranscriptEvent] = []
    stability: list[TokenStability] = []
    last_hypothesis: DecodedHypothesis | None = None
    speech_chunks = 0

    for chunk in chunks:
        emit_left = chunk.emit_start_sample - chunk.inference_start_sample
        emit_right = chunk.emit_end_sample - chunk.inference_start_sample
        emitted_samples = chunk.audio.samples[emit_left:emit_right]
        is_speech = True
        if vad is not None:
            is_speech = vad.is_speech(emitted_samples)
            speech_chunks += int(is_speech)

        hypothesis = None
        if not vad_gate or is_speech:
            hypothesis = decoder.decode(chunk)
        if hypothesis is None:
            continue

        normalized = normalize_text_v1(hypothesis.text)
        if (
            last_hypothesis is not None
            and normalized == normalize_text_v1(last_hypothesis.text)
            and hypothesis.source_end_sample == last_hypothesis.source_end_sample
        ):
            continue

        newly_stable = stabilizer.update(normalized, chunk.available_sample)
        stability.extend(newly_stable)
        last_hypothesis = DecodedHypothesis(
            text=normalized,
            source_end_sample=hypothesis.source_end_sample,
        )
        events.append(
            TranscriptEvent(
                kind="partial",
                text=normalized,
                observed_sample=chunk.available_sample,
                source_end_sample=hypothesis.source_end_sample,
                stable_token_count=len(stabilizer.stable_tokens),
                newly_stable=newly_stable,
            )
        )

    final_text = normalize_text_v1(last_hypothesis.text if last_hypothesis else "")
    final_source_end = (
        last_hypothesis.source_end_sample if last_hypothesis is not None else audio.sample_count
    )
    final_observed = chunks[-1].available_sample if chunks else 0
    final_new_stable = stabilizer.finalize(final_text, final_observed)
    stability.extend(final_new_stable)
    events.append(
        TranscriptEvent(
            kind="final",
            text=final_text,
            observed_sample=final_observed,
            source_end_sample=final_source_end,
            stable_token_count=len(stabilizer.stable_tokens),
            newly_stable=final_new_stable,
        )
    )

    stability_delays = [item.delay_ms for item in stability]
    metrics = {
        "audio_samples": audio.sample_count,
        "chunk_count": len(chunks),
        "speech_chunk_count": speech_chunks if vad is not None else None,
        "final_event_latency_ms": events[-1].latency_ms,
        "stabilized_token_count": len(stability),
        "word_stabilization_delay_ms": (
            latency_summary_ms(stability_delays) if stability_delays else None
        ),
    }
    comparison = (
        compare_offline_streaming(offline_text, final_text)
        if offline_text is not None
        else None
    )

    return {
        "config": config.to_dict(),
        "chunks": [
            {
                "index": chunk.index,
                "nominal_start_sample": chunk.nominal_start_sample,
                "nominal_end_sample": chunk.nominal_end_sample,
                "inference_start_sample": chunk.inference_start_sample,
                "inference_end_sample": chunk.inference_end_sample,
                "emit_start_sample": chunk.emit_start_sample,
                "emit_end_sample": chunk.emit_end_sample,
                "available_sample": chunk.available_sample,
                "is_final": chunk.is_final,
            }
            for chunk in chunks
        ],
        "events": [event.to_dict() for event in events],
        "metrics": metrics,
        "offline_comparison": comparison,
    }
