#!/usr/bin/env python3
"""Build a scripted streaming fixture from a normalized speech manifest."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.contracts import validate_speech_sample  # noqa: E402
from speech_asr.text import normalize_text_v1  # noqa: E402

SAMPLES_PER_MS = 16


def load_record(manifest: pathlib.Path, sample_id: str | None) -> dict:
    selected = None
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = validate_speech_sample(json.loads(line))
            if sample_id is None:
                return value
            if value["id"] == sample_id:
                selected = value
                break
    if selected is None:
        wanted = sample_id if sample_id is not None else "<first record>"
        raise ValueError(f"sample not found in manifest: {wanted}")
    return selected


def build_script(record: dict, decoder_delay_ms: int) -> dict:
    if decoder_delay_ms < 0:
        raise ValueError("decoder_delay_ms must be non-negative")
    audio = record["audio"]
    start = audio["start_sample"]
    end = audio["end_sample"]
    sample_count = end - start
    words = record["transcript"].get("words", [])
    if not words:
        raise ValueError("sample has no timed words")

    delay_samples = decoder_delay_ms * SAMPLES_PER_MS
    updates_by_available: dict[int, dict] = {}
    cumulative: list[str] = []
    for word in words:
        cumulative.append(word["text"])
        source_end = word["end_sample"] - start
        if source_end < 0 or source_end > sample_count:
            raise ValueError("word timing is outside clip-relative audio range")
        available = min(sample_count, source_end + delay_samples)
        updates_by_available[available] = {
            "available_sample": available,
            "source_end_sample": source_end,
            "text": normalize_text_v1(" ".join(cumulative)),
        }

    updates = [updates_by_available[key] for key in sorted(updates_by_available)]
    if not updates:
        raise ValueError("no streaming updates were generated")

    return {
        "schema": "speech-asr/scripted-streaming-updates",
        "version": 1,
        "sample_id": record["id"],
        "offline_text": normalize_text_v1(record["transcript"]["text"]),
        "decoder_delay_ms": decoder_delay_ms,
        "updates": updates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--decoder-delay-ms", type=int, default=80)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output = args.output if args.output.is_absolute() else ROOT / args.output
    try:
        record = load_record(manifest, args.sample_id)
        script = build_script(record, args.decoder_delay_ms)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(script, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    audio_path = pathlib.Path(record["audio"]["path"])
    if not audio_path.is_absolute():
        audio_path = manifest.parent / audio_path
    print(f"sample: {record['id']}")
    print(f"audio: {audio_path}")
    print(f"updates: {output}")
    print(
        "replay: ./scripts/replay-speech.sh "
        f"{audio_path} --audio-format f32le --updates {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
