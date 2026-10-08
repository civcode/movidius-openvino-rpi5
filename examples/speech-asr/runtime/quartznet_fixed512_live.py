#!/usr/bin/env python3
"""Live fixed-T=512 QuartzNet ASR for WAV files or an ALSA microphone."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata

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
    OUTPUT_STRIDE_SAMPLES,
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


def locate_fixed512_ir(
    requested: pathlib.Path,
    *,
    staged_root: pathlib.Path | None = None,
) -> pathlib.Path:
    """Reuse the model previously staged by the fixed512 edge qualification.

    The controller stores IR files below a run-specific fixed512-eval directory,
    whereas the interactive app defaults to the build output directory.
    Explicit --ir-dir arguments always take precedence.
    """
    if requested != DEFAULT_IR_DIR:
        return requested
    try:
        validate_ir(requested)
        return requested
    except (ValueError, OSError):
        pass

    if staged_root is None:
        staged_root = ROOT / "work/speech-asr/fixed512-eval"

    candidates: list[pathlib.Path] = []
    for candidate in staged_root.glob("*/input/openvino/fp16"):
        try:
            validate_ir(candidate)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        candidates.append(candidate)

    if candidates:
        # Prefer the most recently staged complete fixed512 artifact.
        chosen = max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))
        print(
            f"[fixed512-live] using previously staged fixed512 model: {chosen}",
            file=sys.stderr,
            flush=True,
        )
        return chosen

    raise ValueError(
        "fixed512 model not found in the default build directory or any "
        f"edge qualification staging directory under {staged_root}. "
        "Copy quartznet15x5_nvidia_ref.xml, .bin, and artifacts.json from "
        "Oberon's prepared fixed512 IR, or supply --ir-dir PATH."
    )


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
    """Print stable text once and, on a TTY, revise only its temporary suffix.

    Long transcripts must never be repainted: they wrap over many rows. The
    optional preview is rendered after the committed text and erased in place
    before another update. Non-TTY stdout always remains committed-only text.
    """

    def __init__(self, *, plain: bool, show_timing: bool, show_preview: bool = False):
        self.tty = sys.stdout.isatty() and not plain
        self.show_timing = show_timing
        self.show_preview = show_preview
        self.printed_text = ""
        self.last_preview = ""
        self._output_open = False
        self._column = 0
        self._pending_wrap = False
        self._preview_rows = 0
        self._preview_column = 0
        self._preview_started_after_wrap = False

    @staticmethod
    def _cell_width(char: str) -> int:
        # The recognized vocabulary is ASCII, but pause delimiters may contain
        # other printable characters. Handle combining and wide glyphs.
        if unicodedata.combining(char) or unicodedata.category(char) == "Cf":
            return 0
        return 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1

    @classmethod
    def _layout(
        cls, text: str, column: int, pending_wrap: bool, columns: int
    ) -> tuple[int, bool, int, int | None, int | None]:
        """Track physical terminal rows, including the right-edge wrap state."""
        row = 0
        first_row = None
        first_column = None
        for char in text:
            if char == "\n":
                row += 1
                column = 0
                pending_wrap = False
                continue
            width = cls._cell_width(char)
            if not width:
                continue
            if pending_wrap or column + width > columns:
                row += 1
                column = 0
                pending_wrap = False
            if first_row is None:
                first_row, first_column = row, column
            if column + width == columns:
                column = columns - 1
                pending_wrap = True
            else:
                column += width
        return column, pending_wrap, row, first_row, first_column

    def _erase_preview(self) -> None:
        if not self._preview_rows:
            return
        # Only clear the screen cells containing the temporary suffix.
        # The confirmed transcript above and to the left is untouched.
        back = self._preview_rows - 1
        sys.stdout.write("\r")
        if back:
            sys.stdout.write(f"\033[{back}A")
        sys.stdout.write(f"\033[{self._preview_column + 1}G\033[K")
        for _ in range(back):
            sys.stdout.write("\033[1B\r\033[2K")
        if back:
            sys.stdout.write(f"\033[{back}A")
        sys.stdout.write(f"\033[{self._preview_column + 1}G")
        sys.stdout.flush()
        # If the first preview glyph forced a wrap, that wrap has already
        # happened physically. Keep tracking the new row, not the old margin.
        if self._preview_started_after_wrap:
            self._column = self._preview_column
            self._pending_wrap = False
        self._preview_rows = 0
        self._preview_started_after_wrap = False
        self.last_preview = ""

    def _write_stable(self, text: str) -> None:
        if not text:
            return
        if self.tty:
            columns = max(2, shutil.get_terminal_size((80, 24)).columns)
            self._column, self._pending_wrap, _, _, _ = self._layout(
                text, self._column, self._pending_wrap, columns
            )
        sys.stdout.write(text)
        sys.stdout.flush()
        self._output_open = True

    def _write_preview(self, suffix: str) -> None:
        if not suffix:
            return
        columns = max(2, shutil.get_terminal_size((80, 24)).columns)
        _, _, last_row, first_row, first_column = self._layout(
            suffix, self._column, self._pending_wrap, columns
        )
        if first_row is None or first_column is None:
            return
        self._preview_rows = last_row - first_row + 1
        self._preview_column = first_column
        self._preview_started_after_wrap = first_row > 0
        sys.stdout.write(suffix)
        sys.stdout.flush()

    def _end_output_line(self) -> None:
        if self.tty:
            self._erase_preview()
        if self._output_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._output_open = False
            self._column = 0
            self._pending_wrap = False

    def update(self, value: Fixed512LiveUpdate) -> None:
        if self.tty:
            self._erase_preview()

        # Only committed frames may be appended permanently: partial preview
        # tokens can change after center-owned overlap stitching.
        text = value.partial_text if value.final else value.committed_text
        if text.startswith(self.printed_text):
            new_text = text[len(self.printed_text):]
        else:
            self._end_output_line()
            new_text = text
        self._write_stable(new_text)
        self.printed_text = text

        if value.final:
            if self._output_open:
                self._end_output_line()
            else:
                sys.stdout.write("\n")
                sys.stdout.flush()
            return

        # Diagnostics are separate from transcript output. On a TTY the
        # preview comes after timing so diagnostics cannot interrupt it.
        if self.show_timing:
            if self.tty:
                self._end_output_line()
            print(
                "[fixed512-live] "
                f"window={value.window_index} "
                f"audio_s={value.source_seconds:.2f} "
                f"infer_ms={value.inference_ms:.3f} "
                f"committed_frames={value.committed_output_frames}",
                file=sys.stderr,
                flush=True,
            )

        if self.show_preview:
            suffix = (
                value.partial_text[len(value.committed_text):]
                if value.partial_text.startswith(value.committed_text)
                else ""
            )
            if self.tty:
                self._write_preview(suffix)
            else:
                # A pipe must never receive revisions or ANSI cursor codes.
                # Continue supporting the original stderr preview for pipes.
                suffix = suffix[-80:]
                if suffix and suffix != self.last_preview:
                    print(f"[preview] {suffix}", file=sys.stderr, flush=True)
            self.last_preview = suffix

    def status(self, message: str) -> None:
        if self.tty:
            self._end_output_line()
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


class Pcm16ChunkDecoder:
    """Preserve sample boundaries across arbitrary ALSA pipe read sizes."""

    def __init__(self) -> None:
        self.pending = b""
        self.total_samples = 0

    def decode(self, payload: bytes) -> np.ndarray:
        data = self.pending + payload
        complete = len(data) & ~1
        self.pending = data[complete:]
        values = np.frombuffer(data[:complete], dtype="<i2").astype(np.float32)
        self.total_samples += int(values.size)
        return values / 32768.0

    def finish(self) -> None:
        if self.pending:
            raise RuntimeError("microphone capture ended with a truncated PCM16 sample")


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
    decoder = Pcm16ChunkDecoder()
    # A file-backed stderr stream avoids deadlocking if ALSA emits more
    # diagnostics than an undrained subprocess.PIPE can hold.
    with tempfile.TemporaryFile(mode="w+b") as capture_errors:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=capture_errors,
            bufsize=0,
            # Let the Python SIGINT handler drain/finalize audio before
            # explicitly terminating ALSA capture in the finally block.
            start_new_session=True,
        )
        if proc.stdout is None:
            proc.kill()
            proc.wait()
            raise RuntimeError("unable to open ALSA capture pipe")

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
            while not stop_requested:
                # A bounded wait keeps Ctrl+C responsive even if arecord
                # stops producing PCM while its stdout remains open.
                readable, _, _ = select.select([proc.stdout], [], [], 0.2)
                if not readable:
                    continue
                payload = os.read(proc.stdout.fileno(), read_frames * 2)
                if not payload:
                    break
                samples = decoder.decode(payload)
                if samples.size:
                    for update in recognizer.push_audio(samples):
                        console.update(update)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
            proc.stdout.close()

        capture_errors.seek(0)
        errors = capture_errors.read().decode("utf-8", errors="replace").strip()
        if not stop_requested and proc.returncode != 0:
            detail = errors.splitlines()[-1] if errors else "no ALSA details available"
            raise RuntimeError(
                f"arecord exited with status {proc.returncode}: {detail}"
            )
        if not stop_requested and decoder.total_samples == 0:
            raise RuntimeError("microphone capture ended without recording audio")
        decoder.finish()

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
    parser.add_argument(
        "--pause-space-ms",
        type=int,
        default=None,
        help=(
            "insert display-only separators after long CTC blank runs; "
            "default 600 ms for microphone, off for WAV; 0 disables"
        ),
    )
    parser.add_argument(
        "--pause-delimiter",
        default=" ",
        metavar="TEXT",
        help="printable string inserted at a speech pause (default: space)",
    )
    squelch = parser.add_mutually_exclusive_group()
    squelch.add_argument(
        "--squelch-dbfs",
        type=float,
        help=(
            "suppress quiet microphone CTC output below this RMS level; "
            "default -40 dBFS for microphone, disabled for WAV"
        ),
    )
    squelch.add_argument(
        "--no-squelch",
        action="store_true",
        help="disable automatic microphone silence suppression",
    )
    parser.add_argument("--log", type=pathlib.Path, default=DEFAULT_LOG)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="show revisable tentative text inline on TTY (stderr for pipes)",
    )
    parser.add_argument("--show-timing", action="store_true")
    args = parser.parse_args()

    try:
        if args.list_microphones:
            return list_microphones()
        if args.wav is None and not args.microphone:
            parser.error("choose --wav PATH or --microphone")
        if args.feed_samples < 1 or args.read_frames < 1:
            raise ValueError("audio feed/read sizes must be positive")
        if args.pause_space_ms is not None and args.pause_space_ms < 0:
            raise ValueError("--pause-space-ms must be >= 0")
        if not args.pause_delimiter or not args.pause_delimiter.isprintable():
            raise ValueError("--pause-delimiter must be a non-empty printable string")
        pause_ms = (
            args.pause_space_ms
            if args.pause_space_ms is not None
            else (600 if args.microphone else 0)
        )
        pause_space_frames = (
            (pause_ms * 16 + OUTPUT_STRIDE_SAMPLES - 1) // OUTPUT_STRIDE_SAMPLES
        )

        squelch_dbfs = (
            None if args.no_squelch else (
                args.squelch_dbfs if args.squelch_dbfs is not None
                else (-40.0 if args.microphone else None)
            )
        )
        if squelch_dbfs is not None and (
            not np.isfinite(squelch_dbfs) or not -90 <= squelch_dbfs <= -10
        ):
            raise ValueError("--squelch-dbfs must be between -90 and -10")

        if args.wav is not None and not args.wav.is_file():
            raise ValueError(f"WAV file missing: {args.wav}")
        spec, vocab = load_contracts(args.spec, args.vocab)
        ir_dir = locate_fixed512_ir(args.ir_dir)
        xml, binary = validate_ir(ir_dir)
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
            show_preview=args.show_preview,
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
            if args.microphone:
                console.status(
                    "[fixed512-live] squelch="
                    + ("off" if squelch_dbfs is None else f"{squelch_dbfs:g} dBFS")
                )

            def infer(features: np.ndarray) -> tuple[np.ndarray, float]:
                flat, infer_ms = server.infer(features)
                return flat.reshape(FULL_OUTPUT_FRAMES, 29), infer_ms

            recognizer = Fixed512StreamingRecognizer(
                spec=spec,
                vocab=vocab,
                infer=infer,
                hop_output_frames=args.hop_output_frames,
                pause_space_frames=pause_space_frames,
                pause_delimiter=args.pause_delimiter,
                squelch_dbfs=squelch_dbfs,
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
