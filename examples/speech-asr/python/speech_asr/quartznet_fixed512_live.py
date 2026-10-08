"""Online fixed-T=512 QuartzNet stitching and incremental CTC decoding.

This module preserves the accepted fixed512 policy while allowing audio to
arrive incrementally. A frame is committed only after no future overlapping
window can replace its center-owned logit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .quartznet_fixed512 import (
    DEFAULT_HOP_OUTPUT_FRAMES,
    FULL_OUTPUT_FRAMES,
    FULL_WINDOW_SAMPLES,
    OUTPUT_STRIDE_SAMPLES,
    Fixed512Window,
    fixed512_features,
    quartznet_output_frames,
    validate_fixed512_geometry,
)
from .quartznet_reference_frontend import reference_feature_lengths
from .text import normalize_text_v1


InferenceFunction = Callable[[np.ndarray], tuple[np.ndarray, float]]


class IncrementalCtcDecoder:
    """Greedy CTC collapse, with optional pause-spaced *display* text.

    The canonical CTC transcript is not modified. Long blank-only stretches
    can optionally separate words for the interactive microphone display.
    """

    def __init__(self, vocab: dict, *, pause_space_frames: int = 0):
        self.tokens = tuple(vocab["tokens"])
        self.blank = int(vocab["blank_index"])
        if not (0 <= self.blank < len(self.tokens)):
            raise ValueError("invalid CTC blank index")
        if pause_space_frames < 0:
            raise ValueError("pause_space_frames must not be negative")
        self.pause_space_frames = pause_space_frames
        self.previous: int | None = None
        self.emitted: list[str] = []
        self._display_emitted: list[str] = []
        self._blank_run = 0

    @staticmethod
    def _advance(
        values: Sequence[int],
        *,
        tokens: Sequence[str],
        blank: int,
        previous: int | None,
        emitted: list[str],
        display_emitted: list[str],
        blank_run: int,
        pause_space_frames: int,
    ) -> tuple[int | None, int]:
        for raw in values:
            value = int(raw)
            if not 0 <= value < len(tokens):
                raise ValueError(f"CTC index out of range: {value}")
            if value == blank:
                blank_run += 1
            else:
                if value != previous:
                    token = tokens[value]
                    emitted.append(token)
                    if (
                        pause_space_frames
                        and blank_run >= pause_space_frames
                        and display_emitted
                        and display_emitted[-1] != " "
                        and token != " "
                    ):
                        display_emitted.append(" ")
                    display_emitted.append(token)
                blank_run = 0
            previous = value
        return previous, blank_run

    def push(self, values: Sequence[int]) -> str:
        self.previous, self._blank_run = self._advance(
            values,
            tokens=self.tokens,
            blank=self.blank,
            previous=self.previous,
            emitted=self.emitted,
            display_emitted=self._display_emitted,
            blank_run=self._blank_run,
            pause_space_frames=self.pause_space_frames,
        )
        return self.text

    def preview(self, values: Sequence[int], *, display: bool = False) -> str:
        emitted = list(self.emitted)
        display_emitted = list(self._display_emitted)
        self._advance(
            values,
            tokens=self.tokens,
            blank=self.blank,
            previous=self.previous,
            emitted=emitted,
            display_emitted=display_emitted,
            blank_run=self._blank_run,
            pause_space_frames=self.pause_space_frames,
        )
        return normalize_text_v1("".join(display_emitted if display else emitted))

    @property
    def text(self) -> str:
        return normalize_text_v1("".join(self.emitted))

    @property
    def display_text(self) -> str:
        return normalize_text_v1("".join(self._display_emitted))



class OnlineFixed512LogitStitcher:
    """Streaming form of the accepted center-owned fixed512 stitcher."""

    def __init__(self, vocab: dict, *, pause_space_frames: int = 0):
        self.classes = len(vocab["tokens"])
        self.decoder = IncrementalCtcDecoder(
            vocab, pause_space_frames=pause_space_frames
        )
        self.next_commit_frame = 0
        self._pending: dict[int, tuple[float, int, np.ndarray]] = {}

    def add_window(self, window: Fixed512Window, logits: np.ndarray) -> None:
        values = np.asarray(logits, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.classes:
            raise ValueError(
                f"window logits must have shape [T,{self.classes}], got {values.shape}"
            )
        if values.shape[0] < window.valid_output_frames:
            raise ValueError("window logits are shorter than valid_output_frames")

        center = (FULL_OUTPUT_FRAMES - 1) / 2.0
        for local_index in range(window.valid_output_frames):
            global_index = window.start_output_frame + local_index
            if global_index < self.next_commit_frame:
                continue
            distance = abs(local_index - center)
            previous = self._pending.get(global_index)
            if previous is None or distance < previous[0]:
                self._pending[global_index] = (
                    distance,
                    window.index,
                    values[local_index].copy(),
                )

    def _contiguous_argmax(self, start: int, stop: int | None = None) -> list[int]:
        values: list[int] = []
        index = start
        while stop is None or index < stop:
            item = self._pending.get(index)
            if item is None:
                break
            values.append(int(np.argmax(item[2])))
            index += 1
        return values

    def commit_before(self, frame_exclusive: int) -> str:
        if frame_exclusive < self.next_commit_frame:
            raise ValueError("cannot move the committed CTC boundary backwards")
        indices = self._contiguous_argmax(
            self.next_commit_frame,
            frame_exclusive,
        )
        expected = frame_exclusive - self.next_commit_frame
        if len(indices) != expected:
            missing = self.next_commit_frame + len(indices)
            raise ValueError(f"missing stitched logit at global frame {missing}")
        start = self.next_commit_frame
        self.decoder.push(indices)
        self.next_commit_frame = frame_exclusive
        for index in range(start, frame_exclusive):
            self._pending.pop(index, None)
        return self.decoder.text

    @property
    def committed_text(self) -> str:
        return self.decoder.display_text

    @property
    def partial_text(self) -> str:
        pending = self._contiguous_argmax(self.next_commit_frame)
        return self.decoder.preview(pending, display=True)


@dataclass(frozen=True)
class Fixed512LiveUpdate:
    window_index: int
    source_samples: int
    committed_output_frames: int
    inference_ms: float
    committed_text: str
    partial_text: str
    final: bool = False

    @property
    def source_seconds(self) -> float:
        return self.source_samples / 16000.0


class Fixed512StreamingRecognizer:
    """Incremental audio -> fixed512 inference -> globally stitched CTC text."""

    def __init__(
        self,
        *,
        spec: dict,
        vocab: dict,
        infer: InferenceFunction,
        hop_output_frames: int = DEFAULT_HOP_OUTPUT_FRAMES,
        pause_space_frames: int = 0,
    ) -> None:
        validate_fixed512_geometry(spec)
        if not 1 <= hop_output_frames <= FULL_OUTPUT_FRAMES:
            raise ValueError(
                f"hop_output_frames must be in 1..{FULL_OUTPUT_FRAMES}"
            )
        if int(spec["frontend"]["sample_rate_hz"]) != 16000:
            raise ValueError("fixed512 live runtime requires 16 kHz source frontend")
        if len(vocab["tokens"]) != 29 or int(vocab["blank_index"]) != 28:
            raise ValueError("fixed512 live runtime requires the 29-class source vocab")

        self.spec = spec
        self.vocab = vocab
        self.infer = infer
        self.hop_output_frames = hop_output_frames
        self.hop_samples = hop_output_frames * OUTPUT_STRIDE_SAMPLES
        self.stitcher = OnlineFixed512LogitStitcher(
            vocab, pause_space_frames=pause_space_frames
        )

        self.total_samples = 0
        self._buffer_start = 0
        self._buffer = np.empty((0,), dtype=np.float32)
        self._next_start_sample = 0
        self._next_start_output = 0
        self._window_index = 0
        self._finished = False

    @property
    def window_count(self) -> int:
        return self._window_index

    def _append_buffer(self, samples: np.ndarray) -> None:
        if samples.size:
            self._buffer = np.concatenate((self._buffer, samples))
            self.total_samples += int(samples.size)

    def _slice_buffer(self, start: int, end: int) -> np.ndarray:
        if start < self._buffer_start:
            raise ValueError("requested live audio was already discarded")
        buffer_end = self._buffer_start + int(self._buffer.size)
        if end > buffer_end:
            raise ValueError("requested live audio has not arrived yet")
        left = start - self._buffer_start
        right = end - self._buffer_start
        return np.ascontiguousarray(self._buffer[left:right], dtype=np.float32)

    def _trim_buffer(self) -> None:
        keep_from = self._next_start_sample
        if keep_from <= self._buffer_start:
            return
        drop = min(keep_from - self._buffer_start, int(self._buffer.size))
        self._buffer = self._buffer[drop:].copy()
        self._buffer_start += drop

    def _process_window(
        self,
        *,
        start_sample: int,
        start_output: int,
        end_sample: int,
        commit_live_prefix: bool,
    ) -> Fixed512LiveUpdate:
        segment = self._slice_buffer(start_sample, end_sample)
        features, valid_features, valid_outputs = fixed512_features(
            segment,
            self.spec,
        )
        logits, inference_ms = self.infer(features)
        values = np.asarray(logits, dtype=np.float32)
        if values.shape == (1, FULL_OUTPUT_FRAMES, 29):
            values = values[0]
        if values.shape != (FULL_OUTPUT_FRAMES, 29):
            raise ValueError(
                "fixed512 inference must return [256,29] or [1,256,29], "
                f"got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError("fixed512 inference returned non-finite logits")

        window = Fixed512Window(
            index=self._window_index,
            start_output_frame=start_output,
            end_output_frame=start_output + valid_outputs,
            start_sample=start_sample,
            end_sample=end_sample,
            valid_feature_frames=valid_features,
            valid_output_frames=valid_outputs,
        )
        self.stitcher.add_window(window, values)
        self._window_index += 1

        if commit_live_prefix:
            commit_limit = min(
                start_output + self.hop_output_frames,
                window.end_output_frame,
            )
            if commit_limit > self.stitcher.next_commit_frame:
                self.stitcher.commit_before(commit_limit)

        return Fixed512LiveUpdate(
            window_index=window.index,
            source_samples=self.total_samples,
            committed_output_frames=self.stitcher.next_commit_frame,
            inference_ms=float(inference_ms),
            committed_text=self.stitcher.committed_text,
            partial_text=self.stitcher.partial_text,
        )

    def push_audio(
        self,
        samples: Sequence[float] | np.ndarray,
    ) -> tuple[Fixed512LiveUpdate, ...]:
        if self._finished:
            raise RuntimeError("cannot push audio after finalize")
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not np.isfinite(array).all():
            raise ValueError("live audio contains non-finite samples")
        self._append_buffer(array)

        updates: list[Fixed512LiveUpdate] = []
        while self.total_samples - self._next_start_sample >= FULL_WINDOW_SAMPLES:
            end = self._next_start_sample + FULL_WINDOW_SAMPLES
            updates.append(
                self._process_window(
                    start_sample=self._next_start_sample,
                    start_output=self._next_start_output,
                    end_sample=end,
                    commit_live_prefix=True,
                )
            )
            self._next_start_sample += self.hop_samples
            self._next_start_output += self.hop_output_frames
            self._trim_buffer()
        return tuple(updates)

    def finalize(self) -> Fixed512LiveUpdate:
        if self._finished:
            raise RuntimeError("fixed512 live recognizer already finalized")
        self._finished = True

        minimum = 2 * int(self.spec["frontend"]["hop_samples"])
        if self.total_samples < minimum:
            return Fixed512LiveUpdate(
                window_index=max(0, self._window_index - 1),
                source_samples=self.total_samples,
                committed_output_frames=self.stitcher.next_commit_frame,
                inference_ms=0.0,
                committed_text=self.stitcher.committed_text,
                partial_text=self.stitcher.committed_text,
                final=True,
            )

        valid_features, _ = reference_feature_lengths(
            self.total_samples,
            self.spec,
        )
        global_outputs = quartznet_output_frames(valid_features, self.spec)

        last_start_output: int | None = None
        if self._window_index:
            last_start_output = self._next_start_output - self.hop_output_frames

        final_inference_ms = 0.0
        final_window_index = max(0, self._window_index - 1)
        if last_start_output is None:
            update = self._process_window(
                start_sample=0,
                start_output=0,
                end_sample=self.total_samples,
                commit_live_prefix=False,
            )
            final_inference_ms += update.inference_ms
            final_window_index = update.window_index
            last_start_output = 0

        while last_start_output + FULL_OUTPUT_FRAMES < global_outputs:
            start_output = last_start_output + self.hop_output_frames
            start_sample = start_output * OUTPUT_STRIDE_SAMPLES
            update = self._process_window(
                start_sample=start_sample,
                start_output=start_output,
                end_sample=min(
                    self.total_samples,
                    start_sample + FULL_WINDOW_SAMPLES,
                ),
                commit_live_prefix=False,
            )
            final_inference_ms += update.inference_ms
            final_window_index = update.window_index
            last_start_output = start_output

        self.stitcher.commit_before(global_outputs)
        return Fixed512LiveUpdate(
            window_index=final_window_index,
            source_samples=self.total_samples,
            committed_output_frames=self.stitcher.next_commit_frame,
            inference_ms=final_inference_ms,
            committed_text=self.stitcher.committed_text,
            partial_text=self.stitcher.committed_text,
            final=True,
        )
