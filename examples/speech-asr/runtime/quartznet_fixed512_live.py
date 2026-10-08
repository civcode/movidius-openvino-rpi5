#!/usr/bin/env python3
"""Live fixed-T=512 QuartzNet ASR for WAV files or an ALSA microphone."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.audio import read_wav_canonical  # noqa: E402
from speech_asr.myriad_tensor_client import PersistentTensorServer  # noqa: E402
from speech_asr.quartznet_fixed512 import (  # noqa: E402
    FIXED_TENSOR_FRAMES,
    FULL_OUTPUT_FRAMES,
)
from speech_asr.quartznet_fixed512_live import (  # noqa: E402
    Fixed512LiveUpdate,
    Fixed512StreamingRecognizer,
)

DEFAULT_SPEC = SPEECH_ROOT / "models" / "quartznet15x5_nvidia_ref" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "quartznet15x5_nvidia_ref" / "vocab.json"
DEFAULT_IR_DIR = (
    ROOT
    / "work"
    / "speech-asr"
    / "quartznet15x5-reference"
    / "myriad"
    / "openvino"
    / "fp16"
)
DEFAULT_LOG = (
    ROOT
    / "work"
    / "speech-asr"
    / "quartznet15x5-reference"
    / "fixed512-live"
    / "myriad-runtime.log"
)


def work_alias(path: pathlib.Path) -> str:
    try:
        relative = path.resolve().relative_to((ROOT / "work").resolve())
    except ValueError as exc:
        raise ValueError(f"path must be below repository work/: {path}") from exc
    return "/work/" + relative.as_posix()


def load_contracts(spec_path: pathlib.Path, vocab_path: pathlib.Path) -> tuple[dict, dict]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if spec.get("id") != "quartznet15x5_nvidia_ref":
        raise ValueError("unexpected QuartzNet model spec")
    expected = [" "] + list("abcdefghijklmnopqrstuvwxyz") + ["'", "<blank>"]
    if vocab.get("tokens") != expected or vocab.get("blank_index") != 28:
        raise ValueError("QuartzNet 29-class vocabulary changed")
    return spec, vocab


def validate_ir(ir_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    xml = ir_dir / "quartznet15x5_nvidia_ref.xml"
    binary = ir_dir / "quartznet15x5_nvidia_ref.bin"
    artifacts = ir_dir / "artifacts.json"
    for path in (xml, binary, artifacts):
        if not path.is_file():
            raise ValueError(f"required fixed512 artifact missing: {path}")

    value = json.loads(artifacts.read_text(encoding="utf-8"))
    if value.get("model_id") != "quartznet15x5_nvidia_ref":
        raise ValueError("MYRIAD artifact model identity changed")
    if int(value.get("carrier_time_frames", -1)) != FIXED_TENSOR_FRAMES:
        raise ValueError("MYRIAD artifact is not the fixed T=512 carrier")
    if value.get("input_shape") != [1, 64, 512]:
        raise ValueError("MYRIAD fixed512 input shape changed")
    if value.get("output_shape") != [1, 256, 29]:
        raise ValueError("MYRIAD fixed512 output shape changed")
    return xml, binary


def runtime_preflight(platform_name: str, backend: str) -> None:
    proc = subprocess.run(
        [
            str(ROOT / "scripts" / "run-myriad-tensor.sh"),
            "--platform",
            platform_name,
            "--backend",
            backend,
            "check",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "MYRIAD tensor runtime preflight failed:\n"
            + "\n".join(proc.stdout.splitlines()[-40:])
        )
    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", file=sys.stderr)


class ConsoleTranscript:
    def __init__(self, *, plain: bool, show_timing: bool):
        self.tty = sys.stdout.isatty() and not plain
        self.show_timing = show_timing
        self.last_partial = ""
        self._line_active = False

    def update(self, value: Fixed512LiveUpdate) -> None:
        text = value.partial_text
        if value.final:
            if self.tty:
                sys.stdout.write("\r\033[2K")
            print(text, flush=True)
            self._line_active = False
        elif text != self.last_partial:
            if self.tty:
                sys.stdout.write("\r\033[2K" + text)
                sys.stdout.flush()
                self._line_active = True
            else:
                print(f"[partial] {text}", flush=True)
        self.last_partial = text

        if self.show_timing and not value.final:
            print(
                "[fixed512-live] "
                f"window={value.window_index} "
                f"audio_s={value.source_seconds:.2f} "
                f"infer_ms={value.inference_ms:.3f} "
                f"committed_frames={value.committed_output_frames}",
                file=sys.stderr,
                flush=True,
            )

    def status(self, message: str) -> None:
        if self.tty and self._line_active:
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()
            self._line_active = False
        print(message, file=sys.stderr, flush=True)


def feed_wav(
    recognizer: Fixed512StreamingRecognizer,
    console: ConsoleTranscript,
    path: pathlib.Path,
    *,
    realtime: bool,
    feed_samples: int,
) -> None:
    audio = read_wav_canonical(path)
    console.status(
        f"[fixed512-live] WAV {path} duration={audio.duration_seconds:.2f}s "
        f"realtime={'yes' if realtime else 'no'}"
    )
    values = np.asarray(audio.samples, dtype=np.float32)
    started = time.monotonic()

    for start in range(0, values.size, feed_samples):
        end = min(values.size, start + feed_samples)
        if realtime:
            target = end / 16000.0
            delay = target - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
        for update in recognizer.push_audio(values[start:end]):
            console.update(update)

    console.update(recognizer.finalize())


def microphone_command(device: str) -> list[str]:
    return [
        "arecord",
        "-q",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        "16000",
        "-t",
        "raw",
    ]


def feed_microphone(
    recognizer: Fixed512StreamingRecognizer,
    console: ConsoleTranscript,
    *,
    device: str,
    read_frames: int,
) -> None:
    if shutil.which("arecord") is None:
        raise RuntimeError("arecord is required for --microphone")
    command = microphone_command(device)
    console.status(
        f"[fixed512-live] microphone={device} 16kHz mono; Ctrl+C to stop"
    )
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    if proc.stdout is None or proc.stderr is None:
        proc.kill()
        raise RuntimeError("unable to open ALSA capture pipes")

    stop_requested = False
    previous_sigint = signal.getsignal(signal.SIGINT)

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        if not stop_requested:
            stop_requested = True
            console.status(
                "[fixed512-live] stop requested; finishing current inference"
            )

    signal.signal(signal.SIGINT, request_stop)
    try:
        bytes_per_read = read_frames * 2
        while not stop_requested:
            payload = proc.stdout.read(bytes_per_read)
            if not payload:
                break
            usable = len(payload) - (len(payload) % 2)
            if usable == 0:
                continue
            pcm = np.frombuffer(payload[:usable], dtype="<i2")
            samples = pcm.astype(np.float32) / 32768.0
            for update in recognizer.push_audio(samples):
                console.update(update)
            if stop_requested:
                break
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)

    stderr = proc.stderr.read().decode("utf-8", errors="replace").strip()
    if proc.returncode not in {0, -15} and stderr:
        console.status("[fixed512-live] arecord: " + stderr.splitlines()[-1])
    console.update(recognizer.finalize())


def list_microphones() -> int:
    if shutil.which("arecord") is None:
        print("arecord is not installed", file=sys.stderr)
        return 2
    return subprocess.run(["arecord", "-L"], check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--wav", type=pathlib.Path)
    source.add_argument("--microphone", action="store_true")
    parser.add_argument("--list-microphones", action="store_true")
    parser.add_argument("--alsa-device", default="default")
    parser.add_argument("--wav-realtime", action="store_true")
    parser.add_argument("--feed-samples", type=int, default=4096)
    parser.add_argument("--read-frames", type=int, default=4096)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--ir-dir", type=pathlib.Path, default=DEFAULT_IR_DIR)
    parser.add_argument("--platform", default="arm64")
    parser.add_argument(
        "--runtime-backend",
        choices=("auto", "host", "docker"),
        default="host",
    )
    parser.add_argument("--hop-output-frames", type=int, default=128)
    parser.add_argument("--log", type=pathlib.Path, default=DEFAULT_LOG)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--show-timing", action="store_true")
    args = parser.parse_args()

    try:
        if args.list_microphones:
            return list_microphones()
        if args.wav is None and not args.microphone:
            parser.error("choose --wav PATH or --microphone")
        if args.feed_samples < 1 or args.read_frames < 1:
            raise ValueError("audio feed/read sizes must be positive")

        spec, vocab = load_contracts(args.spec, args.vocab)
        xml, binary = validate_ir(args.ir_dir)
        runtime_preflight(args.platform, args.runtime_backend)

        command = [
            str(ROOT / "scripts" / "run-myriad-tensor.sh"),
            "--platform",
            args.platform,
            "--backend",
            args.runtime_backend,
            "custom-server",
            "--model",
            work_alias(xml),
            "--weights",
            work_alias(binary),
            "--warmup",
            "1",
        ]

        console = ConsoleTranscript(
            plain=args.plain,
            show_timing=args.show_timing,
        )
        with PersistentTensorServer(
            command=command,
            cwd=ROOT,
            input_elements=64 * FIXED_TENSOR_FRAMES,
            output_elements=FULL_OUTPUT_FRAMES * 29,
            log_path=args.log,
        ) as server:
            if server.warmup != 1:
                raise RuntimeError("fixed512 live warmup policy changed")
            console.status(
                "[fixed512-live] MYRIAD ready "
                f"load_ms={server.load_ms:.3f} T={FIXED_TENSOR_FRAMES}"
            )

            def infer(features: np.ndarray) -> tuple[np.ndarray, float]:
                flat, infer_ms = server.infer(features)
                return flat.reshape(FULL_OUTPUT_FRAMES, 29), infer_ms

            recognizer = Fixed512StreamingRecognizer(
                spec=spec,
                vocab=vocab,
                infer=infer,
                hop_output_frames=args.hop_output_frames,
            )
            if args.wav is not None:
                if not args.wav.is_file():
                    raise ValueError(f"WAV file missing: {args.wav}")
                feed_wav(
                    recognizer,
                    console,
                    args.wav,
                    realtime=args.wav_realtime,
                    feed_samples=args.feed_samples,
                )
            else:
                feed_microphone(
                    recognizer,
                    console,
                    device=args.alsa_device,
                    read_frames=args.read_frames,
                )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
