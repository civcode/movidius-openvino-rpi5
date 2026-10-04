#!/usr/bin/env python3
"""Replay recorded canonical audio through deterministic streaming mechanics."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.audio import encode_f32le, read_f32le, read_wav_canonical  # noqa: E402
from speech_asr.contracts import (  # noqa: E402
    ContractValidationError,
    canonical_json_sha256,
    validate_streaming_contract,
    validate_streaming_replay_result,
)
from speech_asr.streaming import (  # noqa: E402
    EnergyVad,
    ScriptedCumulativeDecoder,
    ScriptedUpdate,
    StreamingConfig,
    replay_recorded_audio,
)

STREAMING_CONTRACT = SPEECH_ROOT / "contracts" / "streaming-v1.json"
IMPLEMENTATION = SPEECH_ROOT / "python" / "speech_asr" / "streaming.py"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.strip()


def read_audio(path: pathlib.Path, kind: str):
    selected = kind
    if selected == "auto":
        selected = "wav" if path.suffix.lower() == ".wav" else "f32le"
    if selected == "wav":
        return read_wav_canonical(path)
    if selected == "f32le":
        return read_f32le(path)
    raise ValueError(f"unsupported audio format: {kind}")


def load_script(path: pathlib.Path, sample_count: int):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "speech-asr/scripted-streaming-updates":
        raise ValueError("script $.schema must be speech-asr/scripted-streaming-updates")
    if document.get("version") != 1:
        raise ValueError("script $.version must be 1")
    offline_text = document.get("offline_text")
    if offline_text is not None and not isinstance(offline_text, str):
        raise ValueError("script $.offline_text must be a string when present")
    values = document.get("updates")
    if not isinstance(values, list) or not values:
        raise ValueError("script $.updates must be a non-empty array")

    updates = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"script $.updates[{index}] must be an object")
        try:
            available = value["available_sample"]
            source_end = value["source_end_sample"]
            text = value["text"]
        except KeyError as exc:
            raise ValueError(f"script $.updates[{index}] missing {exc.args[0]}") from exc
        if not isinstance(available, int) or isinstance(available, bool):
            raise ValueError(f"script $.updates[{index}].available_sample must be integer")
        if not isinstance(source_end, int) or isinstance(source_end, bool):
            raise ValueError(f"script $.updates[{index}].source_end_sample must be integer")
        if not isinstance(text, str):
            raise ValueError(f"script $.updates[{index}].text must be string")
        if available > sample_count or source_end > sample_count:
            raise ValueError(f"script $.updates[{index}] exceeds recorded audio range")
        updates.append(
            ScriptedUpdate(
                available_sample=available,
                source_end_sample=source_end,
                text=text,
            )
        )
    return document, offline_text, updates


def render_summary(result: dict) -> str:
    comparison = result.get("offline_comparison")
    lines = [
        f"status: {result['status']}",
        f"audio samples: {result['source']['sample_count']}",
        f"chunks: {result['metrics']['chunk_count']}",
        f"events: {len(result['events'])}",
        f"final latency ms: {result['metrics']['final_event_latency_ms']:.3f}",
        f"stabilized tokens: {result['metrics']['stabilized_token_count']}",
    ]
    delay = result["metrics"].get("word_stabilization_delay_ms")
    if delay is not None:
        lines.append(
            f"word stabilization delay ms: p50={delay['p50_ms']:.3f} "
            f"p95={delay['p95_ms']:.3f}"
        )
    if comparison is not None:
        lines.append(
            f"offline comparison: exact={comparison['exact_match']} "
            f"wer={comparison['wer']:.6f} cer={comparison['cer']:.6f}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=pathlib.Path)
    parser.add_argument("--updates", type=pathlib.Path, required=True)
    parser.add_argument("--audio-format", choices=("auto", "wav", "f32le"), default="auto")
    parser.add_argument("--chunk-ms", type=int, default=320)
    parser.add_argument("--overlap-ms", type=int, default=80)
    parser.add_argument("--left-context-ms", type=int, default=160)
    parser.add_argument("--right-context-ms", type=int, default=80)
    parser.add_argument("--stabilization-repeats", type=int, default=2)
    parser.add_argument("--vad-rms-threshold", type=float)
    parser.add_argument("--vad-gate", action="store_true")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "streaming" / "replay-result.json",
    )
    args = parser.parse_args()

    try:
        streaming_contract = validate_streaming_contract(
            json.loads(STREAMING_CONTRACT.read_text(encoding="utf-8"))
        )
        audio = read_audio(args.audio, args.audio_format)
        script, offline_text, updates = load_script(args.updates, audio.sample_count)
        config = StreamingConfig(
            chunk_ms=args.chunk_ms,
            overlap_ms=args.overlap_ms,
            left_context_ms=args.left_context_ms,
            right_context_ms=args.right_context_ms,
            stabilization_repeats=args.stabilization_repeats,
        )
        vad = (
            EnergyVad(args.vad_rms_threshold)
            if args.vad_rms_threshold is not None
            else None
        )
        replay = replay_recorded_audio(
            audio,
            ScriptedCumulativeDecoder(updates),
            config=config,
            offline_text=offline_text,
            vad=vad,
            vad_gate=args.vad_gate,
        )
        result = {
            "schema": "speech-asr/streaming-replay-result",
            "version": 1,
            "status": "completed",
            "source": {
                "path": str(args.audio),
                "source_sha256": sha256_path(args.audio),
                "canonical_audio_sha256": sha256_bytes(encode_f32le(audio.samples)),
                "sample_count": audio.sample_count,
            },
            "config": {
                **replay["config"],
                "vad": {
                    "kind": "energy-v1" if vad is not None else "disabled",
                    "rms_threshold": args.vad_rms_threshold,
                    "gate_decoder": args.vad_gate,
                },
            },
            "decoder": {
                "kind": "scripted-cumulative-v1",
                "script_path": str(args.updates),
                "script_sha256": canonical_json_sha256(script),
                "purpose": "streaming-mechanics fixture; not an acoustic model",
            },
            "events": replay["events"],
            "metrics": replay["metrics"],
            "offline_comparison": replay["offline_comparison"],
            "provenance": {
                "repo_commit": git_head(),
                "streaming_contract_sha256": canonical_json_sha256(streaming_contract),
                "implementation_sha256": sha256_path(IMPLEMENTATION),
                "python_version": platform.python_version(),
            },
        }
        validate_streaming_replay_result(result)
    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
        ContractValidationError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(render_summary(result), end="")
    print(f"result: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
