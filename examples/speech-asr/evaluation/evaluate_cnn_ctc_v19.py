#!/usr/bin/env python3
"""Evaluate cnn_ctc_v19 FP16 IR on MYRIAD against a normalized AMI manifest."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import pathlib
import queue
import re
import select
import subprocess
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.audio import read_f32le  # noqa: E402
from speech_asr.cnn_ctc import (  # noqa: E402
    acoustic_output_length,
    aggregate_decoder_diagnostics,
    canonical_sha256,
    greedy_decode_logits_diagnostics,
    load_spec,
    load_vocab,
    manifest_record_eligibility,
)
from speech_asr.cnn_ctc_frontend import logmel_features  # noqa: E402
from speech_asr.contracts import validate_experiment_result, validate_speech_sample  # noqa: E402
from speech_asr.evaluation import (  # noqa: E402
    character_error_counts,
    latency_summary_ms,
    word_error_counts,
)

DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v19" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v19" / "vocab.json"
DEFAULT_MANIFEST = ROOT / "work" / "speech-asr" / "ami" / "ami-smoke-v1" / "manifest.jsonl"
DEFAULT_IR = ROOT / "work" / "speech-asr" / "cnn_ctc_v19" / "openvino" / "fp16"

EXECUTION_MODE = "persistent-tensor-stream-v2"
MAX_SERVER_RESTARTS = 4
CACHE_VERSION = 3
READY_RE = re.compile(
    r"READY protocol=tensor-stream-v2 input_elements=(\d+) "
    r"output_elements=(\d+) load_ms=([0-9.]+) warmup=(\d+)"
)
TIMING_RE = re.compile(
    r"TIMING request=(\d+) inference_ms=([0-9.]+)"
)


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def records(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield validate_speech_sample(json.loads(line))


def resolve_audio(manifest: pathlib.Path, record: dict) -> pathlib.Path:
    path = pathlib.Path(record["audio"]["path"])
    return path if path.is_absolute() else manifest.parent / path


def container_work(path: pathlib.Path) -> str:
    try:
        relative = path.resolve().relative_to((ROOT / "work").resolve())
    except ValueError as exc:
        raise ValueError(f"path must be below repository work/: {path}") from exc
    return "/work/" + relative.as_posix()


def sum_rate(counts) -> float:
    errors = sum(value.errors for value in counts)
    refs = sum(value.reference_units for value in counts)
    return errors / max(1, refs)


def write_f32_file_exact(path: pathlib.Path, values: np.ndarray) -> None:
    """Write a complete little-endian float32 tensor despite short OS writes."""
    array = np.ascontiguousarray(values, dtype="<f4")
    payload = memoryview(array).cast("B")
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb", buffering=0) as handle:
            written = 0
            while written < len(payload):
                count = handle.write(payload[written:])
                if count is None:
                    count = 0
                if count <= 0:
                    raise OSError(
                        f"short tensor-file write: {len(payload)} bytes requested, "
                        f"{written} written"
                    )
                written += count
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def require_persistent_runtime(platform: str) -> None:
    proc = subprocess.run(
        [
            str(ROOT / "run.sh"),
            "--platform",
            platform,
            "bench",
            "--help",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0 or "--stdin" not in proc.stdout:
        raise ValueError(
            "runtime image lacks persistent tensor-stream support; "
            f"rebuild it from this checkout with: ./build.sh --platform {platform}"
        )


def run_single_shot_parity(
    *,
    platform: str,
    xml: pathlib.Path,
    binary: pathlib.Path,
    feature_path: pathlib.Path,
    output_path: pathlib.Path,
    output_elements: int,
) -> np.ndarray:
    proc = subprocess.run(
        [
            str(ROOT / "run.sh"),
            "--platform",
            platform,
            "custom",
            "--model",
            container_work(xml),
            "--weights",
            container_work(binary),
            "--iterations",
            "1",
            "--tensor",
            container_work(feature_path),
            "--output",
            container_work(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=180.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "single-shot MYRIAD parity probe failed:\n"
            + "\n".join(proc.stdout.splitlines()[-40:])
        )
    if not output_path.is_file():
        raise RuntimeError("single-shot MYRIAD parity output is missing")
    values = np.fromfile(output_path, dtype=np.float32)
    if values.size != output_elements:
        raise RuntimeError(
            f"single-shot MYRIAD parity output has {values.size} elements, "
            f"expected {output_elements}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError("single-shot MYRIAD parity output contains non-finite values")
    return values


class PersistentMyriadServer:
    def __init__(
        self,
        *,
        command: list[str],
        input_elements: int,
        output_elements: int,
        log_path: pathlib.Path,
        startup_timeout: float = 180.0,
        request_timeout: float = 120.0,
    ) -> None:
        self.command = command
        self.input_elements = input_elements
        self.output_elements = output_elements
        self.output_bytes = output_elements * np.dtype("<f4").itemsize
        self.request_timeout = request_timeout
        self.log_path = log_path
        self.load_ms = 0.0
        self.warmup = 0
        self._ready = threading.Event()
        self._timings: queue.Queue[tuple[int, float]] = queue.Queue()
        self._stderr_tail: list[str] = []
        self._closed = False

        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self.proc.stdin is None or self.proc.stdout is None or self.proc.stderr is None:
            self.proc.kill()
            raise RuntimeError("persistent MYRIAD server pipes are unavailable")

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="myriad-server-stderr",
            daemon=True,
        )
        self._stderr_thread.start()
        atexit.register(self.close)

        deadline = time.monotonic() + startup_timeout
        while not self._ready.wait(0.1):
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "persistent MYRIAD server exited during startup:\n"
                    + "\n".join(self._stderr_tail[-30:])
                )
            if time.monotonic() >= deadline:
                self.close()
                raise RuntimeError(
                    "persistent MYRIAD server startup timed out:\n"
                    + "\n".join(self._stderr_tail[-30:])
                )

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for raw in iter(self.proc.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            self._log_handle.write(line + "\n")
            self._log_handle.flush()
            self._stderr_tail.append(line)
            if len(self._stderr_tail) > 60:
                del self._stderr_tail[:-60]

            ready = READY_RE.fullmatch(line)
            if ready is not None:
                input_elements = int(ready.group(1))
                output_elements = int(ready.group(2))
                if input_elements != self.input_elements:
                    self._stderr_tail.append(
                        "client/server input element mismatch: "
                        f"{input_elements} != {self.input_elements}"
                    )
                    return
                if output_elements != self.output_elements:
                    self._stderr_tail.append(
                        "client/server output element mismatch: "
                        f"{output_elements} != {self.output_elements}"
                    )
                    return
                self.load_ms = float(ready.group(3))
                self.warmup = int(ready.group(4))
                self._ready.set()
                print(f"[myriad-server] {line}", file=sys.stderr, flush=True)
                continue

            timing = TIMING_RE.fullmatch(line)
            if timing is not None:
                self._timings.put((int(timing.group(1)), float(timing.group(2))))
                continue

            print(f"[myriad-server] {line}", file=sys.stderr, flush=True)

    def _read_exact(self, count: int) -> bytes:
        assert self.proc.stdout is not None
        chunks: list[bytes] = []
        remaining = count
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + self.request_timeout
        while remaining:
            timeout = max(0.0, deadline - time.monotonic())
            if timeout == 0.0:
                raise RuntimeError(
                    "persistent MYRIAD server response timed out:\n"
                    + "\n".join(self._stderr_tail[-30:])
                )
            try:
                ready, _, _ = select.select([fd], [], [], timeout)
                if not ready:
                    raise RuntimeError(
                        "persistent MYRIAD server response timed out:\n"
                        + "\n".join(self._stderr_tail[-30:])
                    )
                chunk = os.read(fd, remaining)
            except RuntimeError:
                raise
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "persistent MYRIAD server output pipe read failed: "
                    f"{exc}\n"
                    + "\n".join(self._stderr_tail[-30:])
                ) from exc
            if not chunk:
                raise RuntimeError(
                    "persistent MYRIAD server closed its output pipe:\n"
                    + "\n".join(self._stderr_tail[-30:])
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def infer(self, features: np.ndarray, request_index: int) -> tuple[np.ndarray, float]:
        if self.proc.poll() is not None:
            raise RuntimeError(
                "persistent MYRIAD server is not running:\n"
                + "\n".join(self._stderr_tail[-30:])
            )
        values = np.ascontiguousarray(features, dtype="<f4")
        if values.size != self.input_elements:
            raise ValueError(
                f"persistent server input has {values.size} elements, "
                f"expected {self.input_elements}"
            )

        assert self.proc.stdin is not None
        payload = memoryview(values).cast("B")
        sent = 0
        try:
            while sent < len(payload):
                count = self.proc.stdin.write(payload[sent:])
                if count is None:
                    count = 0
                if count <= 0:
                    raise RuntimeError(
                        "persistent MYRIAD server input pipe closed:\n"
                        + "\n".join(self._stderr_tail[-30:])
                    )
                sent += count
            self.proc.stdin.flush()
        except RuntimeError:
            raise
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise RuntimeError(
                "persistent MYRIAD server input pipe write failed: "
                f"{exc}\n"
                + "\n".join(self._stderr_tail[-30:])
            ) from exc

        raw = self._read_exact(self.output_bytes)
        try:
            timing_index, infer_ms = self._timings.get(timeout=self.request_timeout)
        except queue.Empty as exc:
            raise RuntimeError(
                "persistent MYRIAD server did not report request timing:\n"
                + "\n".join(self._stderr_tail[-30:])
            ) from exc
        if timing_index != request_index:
            raise RuntimeError(
                f"persistent MYRIAD timing sequence mismatch: "
                f"{timing_index} != {request_index}"
            )

        logits = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
        return logits, infer_ms

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.proc.stdin is not None and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5.0)
        try:
            self._stderr_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._log_handle.close()
        except Exception:
            pass


def next_server_session_id(work: pathlib.Path) -> str:
    highest = 0
    for path in work.glob("persistent-server-*.log"):
        match = re.fullmatch(r"persistent-server-(\\d{4})\\.log", path.name)
        if match is not None:
            highest = max(highest, int(match.group(1)))
    return f"persistent-server-{highest + 1:04d}"


def write_sample_cache(
    *,
    cache_path: pathlib.Path,
    record_id: str,
    platform: str,
    feature_path: pathlib.Path,
    logits_path: pathlib.Path,
    feature_sha256: str,
    xml_sha256: str,
    bin_sha256: str,
    evaluator_sha256: str,
    output_elements: int,
    infer_ms: float,
    session_id: str,
    server_load_ms: float,
) -> None:
    value = {
        "schema": "speech-asr/myriad-sample-cache",
        "version": CACHE_VERSION,
        "execution_mode": EXECUTION_MODE,
        "sample_id": record_id,
        "platform": platform,
        "feature_sha256": feature_sha256,
        "xml_sha256": xml_sha256,
        "bin_sha256": bin_sha256,
        "evaluator_sha256": evaluator_sha256,
        "output_elements": output_elements,
        "logits_sha256": sha256_path(logits_path),
        "inference_ms": infer_ms,
        "server_session_id": session_id,
        "server_load_ms": server_load_ms,
    }
    cache_path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def load_sample_cache(
    *,
    cache_path: pathlib.Path,
    record_id: str,
    platform: str,
    feature_path: pathlib.Path,
    logits_path: pathlib.Path,
    feature_sha256: str,
    xml_sha256: str,
    bin_sha256: str,
    evaluator_sha256: str,
    output_elements: int,
) -> tuple[float, str, float] | None:
    required = (cache_path, feature_path, logits_path)
    if any(not path.is_file() for path in required):
        return None
    if sha256_path(feature_path) != feature_sha256:
        return None
    if logits_path.stat().st_size != output_elements * np.dtype(np.float32).itemsize:
        return None

    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    expected = {
        "schema": "speech-asr/myriad-sample-cache",
        "version": CACHE_VERSION,
        "execution_mode": EXECUTION_MODE,
        "sample_id": record_id,
        "platform": platform,
        "feature_sha256": feature_sha256,
        "xml_sha256": xml_sha256,
        "bin_sha256": bin_sha256,
        "evaluator_sha256": evaluator_sha256,
        "output_elements": output_elements,
        "logits_sha256": sha256_path(logits_path),
    }
    for key, value in expected.items():
        if cache.get(key) != value:
            return None

    infer_ms = cache.get("inference_ms")
    load_ms = cache.get("server_load_ms")
    session_id = cache.get("server_session_id")
    if (
        not isinstance(infer_ms, (int, float))
        or infer_ms < 0
        or not isinstance(load_ms, (int, float))
        or load_ms < 0
        or not isinstance(session_id, str)
        or re.fullmatch(r"persistent-server-\\d{4}", session_id) is None
    ):
        return None
    return float(infer_ms), session_id, float(load_ms)



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark-id", default="ami-smoke-v1")
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--ir-dir", type=pathlib.Path, default=DEFAULT_IR)
    parser.add_argument("--experiment-id", default="cnn_ctc_v19-ami-smoke-myriad")
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "cnn_ctc_v19" / "evaluation",
    )
    parser.add_argument("--platform", required=True, choices=("armv7", "arm64", "amd64"))
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="emit corpus-evaluation progress every N manifest records; 0 disables",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "cnn_ctc_v19" / "ami-smoke-result.json",
    )
    args = parser.parse_args()

    server: PersistentMyriadServer | None = None
    try:
        spec = load_spec(args.spec)
        vocab = load_vocab(args.vocab)
        if args.progress_interval < 0:
            raise ValueError("progress_interval must be >= 0")
        xml = args.ir_dir / "cnn_ctc_v19.xml"
        binary = args.ir_dir / "cnn_ctc_v19.bin"
        for path in (args.manifest, xml, binary):
            if not path.is_file():
                raise ValueError(f"required file is missing: {path}")

        work = args.work_dir
        work.mkdir(parents=True, exist_ok=True)
        require_persistent_runtime(args.platform)
        word_counts = []
        char_counts = []
        latencies = []
        per_sample = []
        skipped = {"too_long": [], "target_too_long": []}
        manifest_samples = 0
        total_audio_samples = 0
        total_infer_seconds = 0.0

        input_shape = tuple(int(value) for value in spec["input_contract"]["shape"])
        output_shape = tuple(int(value) for value in spec["output_contract"]["shape"])
        input_elements = int(np.prod(input_shape))
        output_elements = int(np.prod(output_shape))
        xml_sha = sha256_path(xml)
        bin_sha = sha256_path(binary)
        evaluator_sha = sha256_path(pathlib.Path(__file__))

        reused_samples = 0
        new_samples = 0
        request_index = 0
        server_restarts = 0
        parity_checked = False
        parity_report = None
        session_loads: dict[str, float] = {}
        manifest_total = sum(
            1
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        evaluation_started = time.monotonic()
        print(
            f"[myriad-eval] manifest_records={manifest_total} "
            f"execution_mode={EXECUTION_MODE} resumable=true",
            flush=True,
        )

        def register_session(session_id: str, load_ms: float) -> None:
            previous = session_loads.get(session_id)
            if previous is not None and abs(previous - load_ms) > 1e-9:
                raise ValueError(
                    f"inconsistent load time for cached server session {session_id}"
                )
            session_loads[session_id] = load_ms

        def ensure_server() -> PersistentMyriadServer:
            nonlocal server, request_index
            if server is not None:
                return server
            session_id = next_server_session_id(work)
            log_path = work / f"{session_id}.log"
            command = [
                str(ROOT / "run.sh"),
                "--platform",
                args.platform,
                "custom-server",
                "--model",
                container_work(xml),
                "--weights",
                container_work(binary),
                "--warmup",
                "1",
            ]
            server = PersistentMyriadServer(
                command=command,
                input_elements=input_elements,
                output_elements=output_elements,
                log_path=log_path,
            )
            if server.warmup != 1:
                raise RuntimeError(
                    f"persistent MYRIAD server warmup mismatch: {server.warmup}"
                )
            register_session(session_id, server.load_ms)
            server.session_id = session_id
            request_index = 0
            return server

        def infer_with_restart(
            feature_values: np.ndarray,
            *,
            record_id: str,
        ) -> tuple[PersistentMyriadServer, np.ndarray, float]:
            nonlocal server, request_index, server_restarts
            last_error: RuntimeError | None = None
            for restart_attempt in range(MAX_SERVER_RESTARTS + 1):
                active = ensure_server()
                request_index += 1
                try:
                    logits_flat, infer_ms = active.infer(
                        feature_values,
                        request_index=request_index,
                    )
                    return active, logits_flat, infer_ms
                except RuntimeError as exc:
                    last_error = exc
                    failed_session = active.session_id
                    if restart_attempt >= MAX_SERVER_RESTARTS:
                        raise
                    print(
                        "[myriad-eval] persistent server request failed; "
                        f"record={record_id} session={failed_session} "
                        f"restart={server_restarts + 1}/{MAX_SERVER_RESTARTS}: "
                        f"{exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    active.close()
                    server = None
                    server_restarts += 1
            assert last_error is not None
            raise last_error

        for record in records(args.manifest):
            manifest_samples += 1
            decision = manifest_record_eligibility(record, spec, vocab)
            if not decision["eligible"]:
                reason = decision["reason"]
                if reason not in skipped:
                    raise ValueError(
                        f"{record['id']}: unknown cnn_ctc_v19 eligibility reason {reason!r}"
                    )
                skipped[reason].append(record["id"])
                continue

            audio_path = resolve_audio(args.manifest, record)
            audio = read_f32le(audio_path)
            if audio.sample_count != decision["sample_count"]:
                raise ValueError(
                    f"{record['id']}: manifest sample count {decision['sample_count']} "
                    f"does not match audio sample count {audio.sample_count}"
                )
            features, valid_frames = logmel_features(audio.samples, spec)
            if tuple(features.shape) != input_shape:
                raise ValueError(
                    f"{record['id']}: frontend shape {tuple(features.shape)} "
                    f"does not match input contract {input_shape}"
                )
            if valid_frames != decision["valid_feature_frames"]:
                raise ValueError(
                    f"{record['id']}: frontend frame count {valid_frames} does not match "
                    f"eligibility frame count {decision['valid_feature_frames']}"
                )

            sample_dir = work / record["id"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            feature_path = sample_dir / "features.f32"
            logits_path = sample_dir / "logits.f32"
            cache_path = sample_dir / "sample-cache.json"
            feature_values = np.ascontiguousarray(features, dtype=np.float32)
            feature_sha = hashlib.sha256(
                feature_values.tobytes(order="C")
            ).hexdigest()

            cached = load_sample_cache(
                cache_path=cache_path,
                record_id=str(record["id"]),
                platform=args.platform,
                feature_path=feature_path,
                logits_path=logits_path,
                feature_sha256=feature_sha,
                xml_sha256=xml_sha,
                bin_sha256=bin_sha,
                evaluator_sha256=evaluator_sha,
                output_elements=output_elements,
            )
            # Every evaluator invocation must physically re-prove stream parity
            # before any cached corpus logits can be accepted. This also covers
            # a retry after all v3 sample caches were written but sealing failed.
            if not parity_checked:
                cached = None
            if cached is not None:
                infer_ms, session_id, load_ms = cached
                register_session(session_id, load_ms)
                reused_samples += 1
                logits_flat = np.fromfile(logits_path, dtype=np.float32)
            else:
                write_f32_file_exact(feature_path, feature_values)

                single_shot = None
                parity_output_path = sample_dir / "parity-single-shot.f32"
                if not parity_checked:
                    single_shot = run_single_shot_parity(
                        platform=args.platform,
                        xml=xml,
                        binary=binary,
                        feature_path=feature_path,
                        output_path=parity_output_path,
                        output_elements=output_elements,
                    )

                active, logits_flat, infer_ms = infer_with_restart(
                    feature_values,
                    record_id=str(record["id"]),
                )

                if single_shot is not None:
                    if not np.isfinite(logits_flat).all():
                        raise RuntimeError(
                            "persistent MYRIAD parity output contains non-finite values"
                        )
                    full_argmax = float(
                        np.mean(
                            single_shot.reshape(output_shape).argmax(axis=-1)
                            == logits_flat.reshape(output_shape).argmax(axis=-1)
                        )
                    )
                    valid_output_frames = int(decision["valid_output_frames"])
                    single_valid = single_shot.reshape(output_shape)[
                        0, :valid_output_frames, :
                    ]
                    stream_valid = logits_flat.reshape(output_shape)[
                        0, :valid_output_frames, :
                    ]
                    valid_argmax = float(
                        np.mean(
                            single_valid.argmax(axis=-1)
                            == stream_valid.argmax(axis=-1)
                        )
                    )
                    max_abs_error = float(
                        np.max(np.abs(single_shot - logits_flat))
                    )
                    if valid_argmax != 1.0 or max_abs_error > 0.01:
                        raise RuntimeError(
                            "persistent MYRIAD parity gate failed: "
                            f"sample={record['id']} "
                            f"valid_argmax_agreement={valid_argmax:.9f} "
                            f"full_argmax_agreement={full_argmax:.9f} "
                            f"max_abs_error={max_abs_error:.9f}"
                        )
                    parity_report = {
                        "sample_id": record["id"],
                        "single_shot_logits_sha256": sha256_path(
                            parity_output_path
                        ),
                        "persistent_logits_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                logits_flat, dtype=np.float32
                            ).tobytes(order="C")
                        ).hexdigest(),
                        "valid_argmax_agreement": valid_argmax,
                        "full_argmax_agreement": full_argmax,
                        "max_abs_error": max_abs_error,
                        "max_abs_error_threshold": 0.01,
                    }
                    parity_checked = True
                    print(
                        "[myriad-eval] persistent parity gate: PASS "
                        f"sample={record['id']} "
                        f"valid_argmax_agreement={valid_argmax:.9f} "
                        f"max_abs_error={max_abs_error:.9f}",
                        flush=True,
                    )

                write_f32_file_exact(logits_path, logits_flat)
                write_sample_cache(
                    cache_path=cache_path,
                    record_id=str(record["id"]),
                    platform=args.platform,
                    feature_path=feature_path,
                    logits_path=logits_path,
                    feature_sha256=feature_sha,
                    xml_sha256=xml_sha,
                    bin_sha256=bin_sha,
                    evaluator_sha256=evaluator_sha,
                    output_elements=output_elements,
                    infer_ms=infer_ms,
                    session_id=active.session_id,
                    server_load_ms=active.load_ms,
                )
                session_id = active.session_id
                load_ms = active.load_ms
                new_samples += 1

            if logits_flat.size != output_elements:
                raise RuntimeError(
                    f"{record['id']}: output elements {logits_flat.size} != "
                    f"{output_elements}"
                )
            logits = logits_flat.reshape(output_shape)
            latencies.append(infer_ms)
            total_infer_seconds += infer_ms / 1000.0
            total_audio_samples += audio.sample_count

            output_frames = acoustic_output_length(valid_frames, spec)
            if output_frames != decision["valid_output_frames"]:
                raise ValueError(
                    f"{record['id']}: output frame count {output_frames} does not match "
                    f"eligibility output count {decision['valid_output_frames']}"
                )
            decoder = greedy_decode_logits_diagnostics(
                logits[0, :output_frames, :],
                vocab,
            )
            hypothesis = decoder["hypothesis"]
            reference = record["transcript"]["text"]
            words = word_error_counts(reference, hypothesis)
            chars = character_error_counts(reference, hypothesis)
            word_counts.append(words)
            char_counts.append(chars)
            per_sample.append(
                {
                    "id": record["id"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "audio_samples": audio.sample_count,
                    "valid_feature_frames": valid_frames,
                    "valid_output_frames": output_frames,
                    "load_ms": load_ms,
                    "inference_ms": infer_ms,
                    "wer": words.rate,
                    "cer": chars.rate,
                    "reference_words": len(reference.split()),
                    "reference_characters_no_spaces": len(reference.replace(" ", "")),
                    "word_edits": words.to_dict(),
                    "character_edits": chars.to_dict(),
                    "decoder": {
                        key: value
                        for key, value in decoder.items()
                        if key != "hypothesis"
                    },
                    "hardware_execution": {
                        "mode": EXECUTION_MODE,
                        "server_session_id": session_id,
                    },
                }
            )
            if (
                args.progress_interval > 0
                and (
                    manifest_samples == 1
                    or manifest_samples % args.progress_interval == 0
                    or manifest_samples == manifest_total
                )
            ):
                elapsed = time.monotonic() - evaluation_started
                records_per_second = manifest_samples / max(elapsed, 1e-9)
                remaining = manifest_total - manifest_samples
                eta_seconds = remaining / max(records_per_second, 1e-9)
                print(
                    "[myriad-eval] "
                    f"processed={manifest_samples}/{manifest_total} "
                    f"percent={100.0 * manifest_samples / max(1, manifest_total):.1f} "
                    f"evaluated={len(per_sample)} reused={reused_samples} "
                    f"new={new_samples} sessions={len(session_loads)} "
                    f"restarts={server_restarts} "
                    f"rate={records_per_second:.2f}/s "
                    f"elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s",
                    flush=True,
                )

        if server is not None:
            server.close()
            server = None

        print(
            "[myriad-eval] "
            f"processed={manifest_samples}/{manifest_total} "
            f"evaluated={len(per_sample)} reused={reused_samples} "
            f"new={new_samples} sessions={len(session_loads)} "
            f"restarts={server_restarts} "
            "hardware_pass_complete=true",
            flush=True,
        )

        if not per_sample:
            raise ValueError(
                "no eligible cnn_ctc_v19 samples in manifest; "
                f"skipped={skipped}"
            )
        if not session_loads:
            raise ValueError("no persistent MYRIAD server session provenance")

        latency = latency_summary_ms(latencies)
        decoder_summary = aggregate_decoder_diagnostics(per_sample)
        manifest_sha = sha256_path(args.manifest)
        spec_sha = canonical_sha256(spec)
        load_values = list(session_loads.values())
        result = {
            "schema": "speech-asr/experiment-result",
            "version": 1,
            "experiment_id": args.experiment_id,
            "status": "completed",
            "benchmark": {
                "id": args.benchmark_id,
                "contract_version": 1,
                "manifest_sha256": manifest_sha,
                "text_normalization_version": "text-v1",
            },
            "model": {
                "id": spec["id"],
                "family": spec["family"],
                "spec_sha256": spec_sha,
                "artifact_sha256": {
                    "cnn_ctc_v19.xml": xml_sha,
                    "cnn_ctc_v19.bin": bin_sha,
                },
            },
            "runtime": {
                "repo_commit": git_head(),
                "backend": "MYRIAD",
                "openvino_version": "2020.3.2",
                "target": args.platform,
            },
            "metrics": {
                "wer": sum_rate(word_counts),
                "cer": sum_rate(char_counts),
                "realtime_factor": total_infer_seconds / (total_audio_samples / 16000.0),
                "inference_latency_p50_ms": latency["p50_ms"],
                "inference_latency_p95_ms": latency["p95_ms"],
                "failures": 0,
                "manifest_samples": manifest_samples,
                "evaluated_samples": len(per_sample),
                "skipped": skipped,
                "measurement_scope": (
                    "MYRIAD steady-state Infer() only; one unmeasured warmup per "
                    "persistent server session; frontend/decoder/IPC excluded"
                ),
                "model_load_ms": {
                    "sessions": len(load_values),
                    "min": min(load_values),
                    "max": max(load_values),
                    "mean": sum(load_values) / len(load_values),
                },
                "hardware_execution": {
                    "mode": EXECUTION_MODE,
                    "server_sessions": len(load_values),
                    "server_restarts": server_restarts,
                    "max_server_restarts": MAX_SERVER_RESTARTS,
                    "warmup_per_session": 1,
                    "reused_samples": reused_samples,
                    "new_samples": new_samples,
                    "persistent_parity_gate": parity_report,
                },
                "decoder": decoder_summary,
                "per_sample": per_sample,
            },
            "provenance": {
                "dataset_manifest_sha256": manifest_sha,
                "model_spec_sha256": spec_sha,
                "vocab_sha256": canonical_sha256(vocab),
                "frontend_kind": spec["frontend"]["kind"],
                "hardware_execution_mode": EXECUTION_MODE,
                "persistent_parity_gate": parity_report,
                "evaluator_sha256": evaluator_sha,
            },
        }
        validate_experiment_result(result)
    except Exception as exc:
        if server is not None:
            server.close()
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"status: completed\n"
        f"samples: {len(per_sample)} evaluated / {manifest_samples} manifest\n"
        f"skipped: too_long={len(skipped['too_long'])}, "
        f"target_too_long={len(skipped['target_too_long'])}\n"
        f"WER: {result['metrics']['wer']:.6f}\n"
        f"CER: {result['metrics']['cer']:.6f}\n"
        f"inference-only RTF: {result['metrics']['realtime_factor']:.6f}\n"
        f"latency p50/p95 ms: {latency['p50_ms']:.3f}/{latency['p95_ms']:.3f}\n"
        f"persistent server sessions: {len(session_loads)}\n"
        f"result: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
