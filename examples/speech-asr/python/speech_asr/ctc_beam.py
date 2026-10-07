"""CPU CTC prefix-beam decoding and a compact character n-gram LM."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .text import normalize_text_v1

NEG_INF = float("-inf")
START = "<s>"


def log_add(*values: float) -> float:
    finite = [value for value in values if value != NEG_INF]
    if not finite:
        return NEG_INF
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def log_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


@dataclass
class CharacterNgramLM:
    order: int
    smoothing: float
    alphabet: tuple[str, ...]
    counts: dict[tuple[str, ...], Counter]
    totals: dict[tuple[str, ...], int]
    training_utterances: int
    training_characters: int

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        alphabet: Sequence[str],
        order: int = 5,
        smoothing: float = 0.1,
    ) -> "CharacterNgramLM":
        if order < 1:
            raise ValueError("LM order must be >= 1")
        if smoothing <= 0:
            raise ValueError("LM smoothing must be > 0")
        alphabet_tuple = tuple(alphabet)
        if not alphabet_tuple or len(set(alphabet_tuple)) != len(alphabet_tuple):
            raise ValueError("LM alphabet must contain unique symbols")

        counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        totals: dict[tuple[str, ...], int] = defaultdict(int)
        utterances = 0
        characters = 0
        history_width = order - 1
        allowed = set(alphabet_tuple)

        for raw in texts:
            text = normalize_text_v1(raw)
            if any(char not in allowed for char in text):
                unknown = sorted({char for char in text if char not in allowed})
                raise ValueError(f"LM training text contains unsupported symbols: {unknown!r}")
            utterances += 1
            characters += len(text)
            history = [START] * history_width
            for char in text:
                context = tuple(history[-history_width:]) if history_width else ()
                counts[context][char] += 1
                totals[context] += 1
                if history_width:
                    history.append(char)

        return cls(
            order=order,
            smoothing=float(smoothing),
            alphabet=alphabet_tuple,
            counts=dict(counts),
            totals=dict(totals),
            training_utterances=utterances,
            training_characters=characters,
        )

    def log_prob(self, prefix_tokens: Sequence[str], token: str) -> float:
        if token not in self.alphabet:
            raise ValueError(f"LM token is outside alphabet: {token!r}")
        width = self.order - 1
        if width:
            context_values = [START] * max(0, width - len(prefix_tokens))
            context_values.extend(prefix_tokens[-width:])
            context = tuple(context_values)
        else:
            context = ()
        token_counts = self.counts.get(context)
        count = 0 if token_counts is None else int(token_counts.get(token, 0))
        total = int(self.totals.get(context, 0))
        denominator = total + self.smoothing * len(self.alphabet)
        return math.log((count + self.smoothing) / denominator)

    def summary(self) -> dict:
        return {
            "kind": "character-ngram-additive-v1",
            "order": self.order,
            "smoothing": self.smoothing,
            "alphabet_size": len(self.alphabet),
            "contexts": len(self.counts),
            "training_utterances": self.training_utterances,
            "training_characters": self.training_characters,
        }


def prefix_beam_decode(
    logits: Sequence[Sequence[float]] | np.ndarray,
    vocab: dict,
    *,
    beam_width: int = 16,
    token_top_k: int = 12,
    lm: CharacterNgramLM | None = None,
    lm_weight: float = 0.0,
    word_bonus: float = 0.0,
) -> dict:
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if token_top_k < 1:
        raise ValueError("token_top_k must be >= 1")
    if lm_weight < 0:
        raise ValueError("lm_weight must be >= 0")

    tokens = tuple(vocab["tokens"])
    blank = int(vocab["blank_index"])
    if blank != 0:
        raise ValueError("prefix beam decoder requires blank index 0")

    matrix = np.asarray(logits, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(tokens):
        raise ValueError(
            f"logits must have shape [frames,{len(tokens)}], got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("logits contain non-finite values")
    log_probs = log_softmax(matrix)
    nonblank_top_k = min(token_top_k, len(tokens) - 1)

    # prefix -> (probability ending blank, probability ending nonblank)
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}

    def lm_increment(prefix: tuple[int, ...], token_index: int) -> float:
        if lm is None or lm_weight == 0.0:
            lm_part = 0.0
        else:
            prefix_symbols = [tokens[index] for index in prefix]
            lm_part = lm_weight * lm.log_prob(
                prefix_symbols,
                tokens[token_index],
            )
        word_part = (
            word_bonus
            if tokens[token_index] == " "
            and prefix
            and tokens[prefix[-1]] != " "
            else 0.0
        )
        return lm_part + word_part

    for frame in log_probs:
        if nonblank_top_k >= len(tokens) - 1:
            candidates = list(range(1, len(tokens)))
        else:
            nonblank = frame[1:]
            selected = np.argpartition(nonblank, -nonblank_top_k)[-nonblank_top_k:]
            candidates = [int(index) + 1 for index in selected]
        candidates.sort(key=lambda index: (-float(frame[index]), index))

        next_beams: dict[tuple[int, ...], list[float]] = {}

        def slot(prefix: tuple[int, ...]) -> list[float]:
            return next_beams.setdefault(prefix, [NEG_INF, NEG_INF])

        blank_lp = float(frame[blank])
        for prefix, (p_blank, p_nonblank) in beams.items():
            total = log_add(p_blank, p_nonblank)
            current = slot(prefix)
            current[0] = log_add(current[0], total + blank_lp)

            last = prefix[-1] if prefix else None
            for token_index in candidates:
                token_lp = float(frame[token_index])
                if token_index == last:
                    # Same symbol without an intervening blank is the same CTC
                    # prefix; only a blank-ending path may append it anew.
                    current = slot(prefix)
                    current[1] = log_add(
                        current[1],
                        p_nonblank + token_lp,
                    )
                    if p_blank != NEG_INF:
                        extended = prefix + (token_index,)
                        target = slot(extended)
                        target[1] = log_add(
                            target[1],
                            p_blank
                            + token_lp
                            + lm_increment(prefix, token_index),
                        )
                else:
                    extended = prefix + (token_index,)
                    target = slot(extended)
                    target[1] = log_add(
                        target[1],
                        total
                        + token_lp
                        + lm_increment(prefix, token_index),
                    )

        ranked = sorted(
            next_beams.items(),
            key=lambda item: (
                -log_add(item[1][0], item[1][1]),
                item[0],
            ),
        )
        beams = {
            prefix: (values[0], values[1])
            for prefix, values in ranked[:beam_width]
        }

    best_prefix, (best_blank, best_nonblank) = min(
        beams.items(),
        key=lambda item: (
            -log_add(item[1][0], item[1][1]),
            item[0],
        ),
    )
    hypothesis = normalize_text_v1("".join(tokens[index] for index in best_prefix))
    return {
        "hypothesis": hypothesis,
        "score": log_add(best_blank, best_nonblank),
        "beam_width": beam_width,
        "token_top_k": token_top_k,
        "lm_weight": float(lm_weight),
        "word_bonus": float(word_bonus),
        "prefix_tokens": len(best_prefix),
    }
