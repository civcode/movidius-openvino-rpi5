"""CPU CTC prefix-beam decoding and a compact character n-gram LM."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import pathlib
from typing import Any, Iterable, Mapping, Sequence

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
                full_context = (
                    tuple(history[-history_width:]) if history_width else ()
                )
                for context_length in range(len(full_context) + 1):
                    context = (
                        full_context[-context_length:]
                        if context_length
                        else ()
                    )
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
        while context not in self.counts and context:
            context = context[1:]
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

    def to_dict(self) -> dict[str, Any]:
        contexts = []
        for context in sorted(self.counts):
            token_counts = self.counts[context]
            contexts.append(
                {
                    "context": list(context),
                    "counts": [
                        [token, int(token_counts[token])]
                        for token in sorted(token_counts)
                    ],
                }
            )
        return {
            "schema": "speech-asr/character-ngram-lm",
            "version": 1,
            "kind": "character-ngram-additive-v1",
            "order": self.order,
            "smoothing": self.smoothing,
            "alphabet": list(self.alphabet),
            "training_utterances": self.training_utterances,
            "training_characters": self.training_characters,
            "contexts": contexts,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CharacterNgramLM":
        if value.get("schema") != "speech-asr/character-ngram-lm":
            raise ValueError("invalid character n-gram LM schema")
        if value.get("version") != 1:
            raise ValueError("unsupported character n-gram LM version")
        if value.get("kind") != "character-ngram-additive-v1":
            raise ValueError("unsupported character n-gram LM kind")

        order = value.get("order")
        smoothing = value.get("smoothing")
        alphabet_raw = value.get("alphabet")
        contexts_raw = value.get("contexts")
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            raise ValueError("LM order must be a positive integer")
        if (
            not isinstance(smoothing, (int, float))
            or isinstance(smoothing, bool)
            or not math.isfinite(float(smoothing))
            or float(smoothing) <= 0
        ):
            raise ValueError("LM smoothing must be a positive finite number")
        if (
            not isinstance(alphabet_raw, list)
            or not alphabet_raw
            or any(not isinstance(token, str) for token in alphabet_raw)
        ):
            raise ValueError("LM alphabet must be a non-empty string list")
        alphabet = tuple(alphabet_raw)
        if len(alphabet) != len(set(alphabet)):
            raise ValueError("LM alphabet contains duplicate symbols")
        if not isinstance(contexts_raw, list):
            raise ValueError("LM contexts must be a list")

        allowed = set(alphabet)
        counts: dict[tuple[str, ...], Counter] = {}
        totals: dict[tuple[str, ...], int] = {}
        for entry in contexts_raw:
            if not isinstance(entry, Mapping):
                raise ValueError("LM context entry must be an object")
            context_raw = entry.get("context")
            token_counts_raw = entry.get("counts")
            if (
                not isinstance(context_raw, list)
                or any(not isinstance(token, str) for token in context_raw)
                or len(context_raw) > order - 1
            ):
                raise ValueError("invalid LM context")
            context = tuple(context_raw)
            if context in counts:
                raise ValueError("duplicate LM context")
            if not isinstance(token_counts_raw, list):
                raise ValueError("LM counts must be a list")
            token_counts: Counter = Counter()
            for pair in token_counts_raw:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or not isinstance(pair[0], str)
                    or pair[0] not in allowed
                    or not isinstance(pair[1], int)
                    or isinstance(pair[1], bool)
                    or pair[1] <= 0
                ):
                    raise ValueError("invalid LM token count")
                if pair[0] in token_counts:
                    raise ValueError("duplicate LM token count")
                token_counts[pair[0]] = pair[1]
            counts[context] = token_counts
            totals[context] = sum(token_counts.values())

        training_utterances = value.get("training_utterances")
        training_characters = value.get("training_characters")
        for label, raw in (
            ("training_utterances", training_utterances),
            ("training_characters", training_characters),
        ):
            if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
                raise ValueError(f"LM {label} must be a non-negative integer")

        return cls(
            order=order,
            smoothing=float(smoothing),
            alphabet=alphabet,
            counts=counts,
            totals=totals,
            training_utterances=training_utterances,
            training_characters=training_characters,
        )


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_decoder_artifact(
    *,
    acoustic_model: str,
    vocab: Mapping[str, Any],
    config: Mapping[str, Any],
    lm: CharacterNgramLM,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if acoustic_model != "cnn_ctc_v19":
        raise ValueError("decoder artifact is frozen to cnn_ctc_v19")
    expected_config = {
        "kind": "prefix-beam",
        "beam_width": 8,
        "token_top_k": 12,
        "lm_weight": 0.3,
        "word_bonus": -0.2,
    }
    normalized_config = {
        "kind": config.get("kind"),
        "beam_width": config.get("beam_width"),
        "token_top_k": config.get("token_top_k"),
        "lm_weight": float(config.get("lm_weight", 0.0)),
        "word_bonus": float(config.get("word_bonus", 0.0)),
    }
    if normalized_config != expected_config:
        raise ValueError(
            f"cnn_ctc_v19 decoder config must remain {expected_config!r}"
        )
    tokens = vocab.get("tokens")
    if (
        vocab.get("blank_index") != 0
        or not isinstance(tokens, list)
        or tuple(tokens[1:]) != lm.alphabet
    ):
        raise ValueError("decoder LM alphabet does not match v19 vocabulary")
    return {
        "schema": "speech-asr/ctc-decoder-artifact",
        "version": 1,
        "acoustic_model": acoustic_model,
        "decoder": normalized_config,
        "vocab_sha256": canonical_json_sha256(vocab),
        "lm": lm.to_dict(),
        "provenance": dict(provenance),
    }


def validate_decoder_artifact(
    value: Mapping[str, Any],
    *,
    vocab: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if value.get("schema") != "speech-asr/ctc-decoder-artifact":
        raise ValueError("invalid CTC decoder artifact schema")
    if value.get("version") != 1:
        raise ValueError("unsupported CTC decoder artifact version")
    if value.get("acoustic_model") != "cnn_ctc_v19":
        raise ValueError("decoder artifact acoustic model must be cnn_ctc_v19")
    decoder = value.get("decoder")
    if not isinstance(decoder, Mapping):
        raise ValueError("decoder artifact decoder config is missing")
    lm_raw = value.get("lm")
    if not isinstance(lm_raw, Mapping):
        raise ValueError("decoder artifact LM is missing")
    lm = CharacterNgramLM.from_dict(lm_raw)
    if vocab is not None:
        if value.get("vocab_sha256") != canonical_json_sha256(vocab):
            raise ValueError("decoder artifact vocabulary hash mismatch")
        tokens = vocab.get("tokens")
        if not isinstance(tokens, list) or tuple(tokens[1:]) != lm.alphabet:
            raise ValueError("decoder artifact LM alphabet mismatch")
    build_decoder_artifact(
        acoustic_model="cnn_ctc_v19",
        vocab=(
            vocab
            if vocab is not None
            else {"blank_index": 0, "tokens": ["<blank>", *lm.alphabet]}
        ),
        config=decoder,
        lm=lm,
        provenance=(
            value.get("provenance")
            if isinstance(value.get("provenance"), Mapping)
            else {}
        ),
    )
    return dict(value)


def load_decoder_artifact(
    path: pathlib.Path,
    *,
    vocab: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("decoder artifact must be a JSON object")
    return validate_decoder_artifact(value, vocab=vocab)


@dataclass
class FrozenCtcDecoder:
    acoustic_model: str
    config: dict[str, Any]
    lm: CharacterNgramLM

    @classmethod
    def from_artifact(
        cls,
        artifact: Mapping[str, Any],
        *,
        vocab: Mapping[str, Any],
    ) -> "FrozenCtcDecoder":
        validated = validate_decoder_artifact(artifact, vocab=vocab)
        return cls(
            acoustic_model=str(validated["acoustic_model"]),
            config=dict(validated["decoder"]),
            lm=CharacterNgramLM.from_dict(validated["lm"]),
        )

    def decode(
        self,
        logits: Sequence[Sequence[float]] | np.ndarray,
        vocab: dict,
    ) -> dict:
        result = prefix_beam_decode(
            logits,
            vocab,
            beam_width=int(self.config["beam_width"]),
            token_top_k=int(self.config["token_top_k"]),
            lm=self.lm,
            lm_weight=float(self.config["lm_weight"]),
            word_bonus=float(self.config["word_bonus"]),
        )
        return {
            **result,
            "kind": "ctc-prefix-beam-char-ngram-v1",
            "acoustic_model": self.acoustic_model,
        }


def decode_with_artifact(
    logits: Sequence[Sequence[float]] | np.ndarray,
    vocab: dict,
    artifact: Mapping[str, Any],
) -> dict:
    """One-shot convenience wrapper; repeated runtime decoding should reuse FrozenCtcDecoder."""
    return FrozenCtcDecoder.from_artifact(
        artifact,
        vocab=vocab,
    ).decode(logits, vocab)


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
