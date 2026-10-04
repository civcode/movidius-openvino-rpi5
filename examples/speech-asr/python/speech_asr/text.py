"""Versioned transcript normalization used by speech-ASR manifests and scoring."""

from __future__ import annotations

import unicodedata

_APOSTROPHES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u02bc": "'",
    "\uff07": "'",
}


def normalize_text_v1(text: str) -> str:
    """Normalize English ASR reference/hypothesis text according to text-v1."""

    if not isinstance(text, str):
        raise TypeError("text-v1 input must be a string")
    text = unicodedata.normalize("NFKC", text)
    text = "".join(_APOSTROPHES.get(ch, ch) for ch in text).lower()

    out = []
    for index, ch in enumerate(text):
        if ch.isalnum():
            out.append(ch)
            continue
        if ch == "'":
            left = index > 0 and text[index - 1].isalnum()
            right = index + 1 < len(text) and text[index + 1].isalnum()
            if left or right:
                out.append(ch)
                continue
        out.append(" ")
    return " ".join("".join(out).split())
