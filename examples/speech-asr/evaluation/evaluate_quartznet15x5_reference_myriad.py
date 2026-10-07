#!/usr/bin/env python3
"""Evaluate the qualified 29-class QuartzNet source model on MYRIAD/LibriSpeech.

The source frontend and CTC head remain unchanged. OpenVINO runtime reshape
specializes the carrier IR to each exact source-frontend padded time length.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import pathlib
import platform
import queue
import re
import select
import subprocess
import sys
import threading
import time
from collections import defaultdict

import numpy as np
import onnxruntime as ort
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.cnn_ctc import (
    aggregate_decoder_diagnostics,
    conv1d_output_length,
    greedy_decode_logits_diagnostics,
)
from speech_asr.cnn_ctc_compare import compare_arrays
from speech_asr.evaluation import (
    character_error_counts,
    latency_summary_ms,
    word_error_counts,
)
from speech_asr.quartznet_reference_frontend import (
    nemo_reference_features_numpy,
    reference_feature_lengths,
)

EXPECTED_SAMPLES = 2703
QUALIFIED_WER = 0.037939781625675524
QUALIFIED_CER = 0.012506494000177396
MAX_WER = 0.05
MAX_ABS_WER_DELTA = 0.005
DIAGNOSTIC_MAX_FRAME_TOTAL_VARIATION = 0.002
SOURCE_SHA384 = "74e8284e77098906afb7a15a861ef60ec14db1a4acb206fa719492fa43050ad69a91c245652c05c5f0ded38b5903ed55"
EXECUTION_MODE = "exact-time-runtime-reshape-v1"

DEFAULT_SPEC = SPEECH_ROOT / "models" / "quartznet15x5_nvidia_ref" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "quartznet15x5_nvidia_ref" / "vocab.json"
DEFAULT_MANIFEST = ROOT / "work" / "speech-asr" / "librispeech" / "dev-clean" / "manifest.jsonl"
DEFAULT_IR = ROOT / "work" / "speech-asr" / "quartznet15x5-reference" / "myriad" / "openvino" / "fp16"
DEFAULT_ONNX = ROOT / "work" / "speech-asr" / "quartznet15x5-reference" / "myriad" / "dynamic" / "quartznet15x5_nvidia_ref.onnx"
DEFAULT_OUTPUT = ROOT / "work" / "speech-asr" / "quartznet15x5-reference" / "myriad-evaluation" / "result.json"

READY_RE = re.compile(
    r"READY protocol=tensor-stream-v2 input_elements=(\d+) "
    r"output_elements=(\d+) load_ms=([0-9.]+) warmup=(\d+)"
)
TIMING_RE = re.compile(r"TIMING request=(\d+) inference_ms=([0-9.]+)")


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


def load_spec_vocab(spec_path: pathlib.Path, vocab_path: pathlib.Path) -> tuple[dict, dict]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if spec.get("schema") != "speech-asr/pretrained-reference-spec" or spec.get("version") != 1:
        raise ValueError("invalid QuartzNet source-reference spec")
    if spec.get("id") != "quartznet15x5_nvidia_ref":
        raise ValueError("unexpected source-reference model id")
    expected_tokens = [" "] + list("abcdefghijklmnopqrstuvwxyz") + ["'", "<blank>"]
    if vocab.get("tokens") != expected_tokens or vocab.get("blank_index") != 28:
        raise ValueError("source-reference 29-class vocabulary changed")
    if spec.get("source", {}).get("sha384") != SOURCE_SHA384:
        raise ValueError("source-reference pretrained identity changed")
    return spec, vocab


def read_manifest(path: pathlib.Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != EXPECTED_SAMPLES:
        raise ValueError(
            f"LibriSpeech dev-clean manifest has {len(records)} records, "
            f"expected {EXPECTED_SAMPLES}"
        )
    ids = [str(record.get("id")) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("LibriSpeech dev-clean manifest contains duplicate ids")
    for record in records:
        for key in ("id", "audio_path", "sample_rate_hz", "sample_count", "text"):
            if key not in record:
                raise ValueError(f"manifest record lacks {key}: {record!r}")
        if int(record["sample_rate_hz"]) != 16000:
            raise ValueError(f"{record['id']}: expected 16 kHz audio")
    return records


def reference_output_length(feature_frames: int, spec: dict) -> int:
    layer = spec["network"]["prologue"]
    return conv1d_output_length(
        int(feature_frames),
        kernel=int(layer["kernel"]),
        stride=int(layer["stride"]),
        padding=int(layer["padding"]),
        dilation=int(layer["dilation"]),
    )


def container_work(path: pathlib.Path) -> str:
    try:
        relative = path.resolve().relative_to((ROOT / "work").resolve())
    except ValueError as exc:
        raise ValueError(f"path must be below repository work/: {path}") from exc
    return "/work/" + relative.as_posix()


def require_runtime(platform_name: str, backend: str) -> str:
    proc = subprocess.run(
        [
            str(ROOT / "scripts" / "run-myriad-tensor.sh"),
            "--platform",
            platform_name,
            "--backend",
            backend,
            "check-reshape",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            "MYRIAD runtime reshape preflight failed:\n"
            + "\n".join(proc.stdout.splitlines()[-40:])
        )
    resolved = None
    for line in proc.stdout.splitlines():
        if line.startswith("runtime_backend="):
            resolved = line.split("=", 1)[1].strip()
    if resolved not in {"host", "docker"}:
        raise ValueError("MYRIAD runtime did not report a usable backend")
    return resolved


class PersistentMyriadServer:
    def __init__(
        self,
        *,
        command: list[str],
        input_elements: int,
        output_elements: int,
        log_path: pathlib.Path,
        startup_timeout: float = 240.0,
        request_timeout: float = 180.0,
    ) -> None:
        self.command = command
        self.input_elements = input_elements
        self.output_elements = output_elements
        self.output_bytes = output_elements * 4
        self.request_timeout = request_timeout
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
            raise RuntimeError("persistent MYRIAD server pipes unavailable")
        self._thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._thread.start()
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
            if len(self._stderr_tail) > 80:
                del self._stderr_tail[:-80]
            ready = READY_RE.fullmatch(line)
            if ready is not None:
                got_in = int(ready.group(1))
                got_out = int(ready.group(2))
                if got_in != self.input_elements or got_out != self.output_elements:
                    self._stderr_tail.append(
                        f"client/server tensor mismatch: {got_in}/{got_out} != "
                        f"{self.input_elements}/{self.output_elements}"
                    )
                    return
                self.load_ms = float(ready.group(3))
                self.warmup = int(ready.group(4))
                self._ready.set()
                continue
            timing = TIMING_RE.fullmatch(line)
            if timing is not None:
                self._timings.put((int(timing.group(1)), float(timing.group(2))))

    def _read_exact(self, count: int) -> bytes:
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        remaining = count
        chunks: list[bytes] = []
        deadline = time.monotonic() + self.request_timeout
        while remaining:
            timeout = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                raise RuntimeError(
                    "persistent MYRIAD response timed out:\n"
                    + "\n".join(self._stderr_tail[-30:])
                )
            chunk = os.read(fd, remaining)
            if not chunk:
                raise RuntimeError(
                    "persistent MYRIAD output pipe closed:\n"
                    + "\n".join(self._stderr_tail[-30:])
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def infer(self, values: np.ndarray, request_index: int) -> tuple[np.ndarray, float]:
        array = np.ascontiguousarray(values, dtype="<f4")
        if array.size != self.input_elements:
            raise ValueError(
                f"server input has {array.size} elements, expected {self.input_elements}"
            )
        assert self.proc.stdin is not None
        payload = memoryview(array).cast("B")
        sent = 0
        while sent < len(payload):
            count = self.proc.stdin.write(payload[sent:])
            if count is None or count <= 0:
                raise RuntimeError("persistent MYRIAD input pipe closed")
            sent += count
        self.proc.stdin.flush()

        raw = self._read_exact(self.output_bytes)
        try:
            timing_index, infer_ms = self._timings.get(timeout=self.request_timeout)
        except queue.Empty as exc:
            raise RuntimeError("MYRIAD server did not report request timing") from exc
        if timing_index != request_index:
            raise RuntimeError(
                f"MYRIAD timing sequence mismatch: {timing_index} != {request_index}"
            )
        return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True), infer_ms

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
            self.proc.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5.0)
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._log_handle.close()
        except Exception:
            pass


def sum_rate(counts) -> float:
    errors = sum(item.errors for item in counts)
    refs = sum(item.reference_units for item in counts)
    return errors / max(1, refs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--ir-dir", type=pathlib.Path, default=DEFAULT_IR)
    parser.add_argument("--dynamic-onnx", type=pathlib.Path, default=DEFAULT_ONNX)
    parser.add_argument("--platform", choices=("arm64", "armv7", "amd64"), required=True)
    parser.add_argument(
        "--runtime-backend",
        choices=("auto", "host", "docker"),
        default="host",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "quartznet15x5-reference" / "myriad-evaluation",
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    server: PersistentMyriadServer | None = None
    try:
        if args.progress_interval < 0:
            raise ValueError("--progress-interval must be >= 0")
        if args.max_samples is not None and args.max_samples < 1:
            raise ValueError("--max-samples must be positive")

        spec, vocab = load_spec_vocab(args.spec, args.vocab)
        records = read_manifest(args.manifest)
        full = args.max_samples is None
        if args.max_samples is not None:
            records = records[: args.max_samples]

        xml = args.ir_dir / "quartznet15x5_nvidia_ref.xml"
        binary = args.ir_dir / "quartznet15x5_nvidia_ref.bin"
        artifacts_path = args.ir_dir / "artifacts.json"
        for path in (args.manifest, xml, binary, artifacts_path, args.dynamic_onnx):
            if not path.is_file():
                raise ValueError(f"required file missing: {path}")

        artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
        if artifacts.get("model_id") != "quartznet15x5_nvidia_ref":
            raise ValueError("MYRIAD artifact model identity changed")
        if artifacts.get("precision") != "FP16":
            raise ValueError("MYRIAD artifact precision changed")
        hashes = artifacts.get("artifacts", {})
        if hashes.get("xml_sha256") != sha256_path(xml):
            raise ValueError("MYRIAD XML hash mismatch")
        if hashes.get("bin_sha256") != sha256_path(binary):
            raise ValueError("MYRIAD BIN hash mismatch")
        if hashes.get("dynamic_onnx_sha256") != sha256_path(args.dynamic_onnx):
            raise ValueError("dynamic ONNX hash mismatch")

        resolved_backend = require_runtime(args.platform, args.runtime_backend)
        ort_session = ort.InferenceSession(
            str(args.dynamic_onnx),
            providers=["CPUExecutionProvider"],
        )

        prepared: list[dict] = []
        grouped: dict[int, list[dict]] = defaultdict(list)
        frontend_latencies: list[float] = []
        for index, record in enumerate(records):
            audio_path = args.manifest.parent / str(record["audio_path"])
            if not audio_path.is_file():
                raise ValueError(f"{record['id']}: audio missing: {audio_path}")
            valid_frames, tensor_frames = reference_feature_lengths(
                int(record["sample_count"]),
                spec,
            )
            valid_output_frames = reference_output_length(valid_frames, spec)
            full_output_frames = reference_output_length(tensor_frames, spec)
            item = {
                "index": index,
                "record": record,
                "audio_path": audio_path,
                "valid_frames": valid_frames,
                "tensor_frames": tensor_frames,
                "valid_output_frames": valid_output_frames,
                "full_output_frames": full_output_frames,
            }
            prepared.append(item)
            grouped[tensor_frames].append(item)

        shape_order = sorted(grouped, key=lambda value: grouped[value][0]["index"])
        args.work_dir.mkdir(parents=True, exist_ok=True)
        word_counts = []
        char_counts = []
        decoder_samples = []
        per_sample: list[dict | None] = [None] * len(prepared)
        infer_latencies: list[float] = []
        load_ms: list[float] = []
        total_infer_seconds = 0.0
        total_audio_samples = 0
        parity = None
        processed = 0
        evaluation_started = time.monotonic()

        for session_index, tensor_frames in enumerate(shape_order, start=1):
            group = grouped[tensor_frames]
            output_frames = int(group[0]["full_output_frames"])
            input_elements = 64 * tensor_frames
            output_elements = output_frames * 29
            command = [
                str(ROOT / "scripts" / "run-myriad-tensor.sh"),
                "--platform",
                args.platform,
                "--backend",
                resolved_backend,
                "custom-server",
                "--model",
                container_work(xml),
                "--weights",
                container_work(binary),
                "--reshape-time",
                str(tensor_frames),
                "--warmup",
                "1",
            ]
            log_path = args.work_dir / f"shape-{tensor_frames:05d}.log"
            server = PersistentMyriadServer(
                command=command,
                input_elements=input_elements,
                output_elements=output_elements,
                log_path=log_path,
            )
            if server.warmup != 1:
                raise RuntimeError("MYRIAD session warmup count changed")
            load_ms.append(server.load_ms)

            for request_index, item in enumerate(group, start=1):
                record = item["record"]
                samples, sample_rate = sf.read(
                    str(item["audio_path"]),
                    dtype="float32",
                    always_2d=False,
                )
                if sample_rate != 16000 or np.asarray(samples).ndim != 1:
                    raise ValueError(f"{record['id']}: expected 16 kHz mono FLAC")
                if len(samples) != int(record["sample_count"]):
                    raise ValueError(f"{record['id']}: sample count changed")

                frontend_started = time.perf_counter()
                features, valid_frames = nemo_reference_features_numpy(samples, spec)
                frontend_ms = (time.perf_counter() - frontend_started) * 1000.0
                frontend_latencies.append(frontend_ms)
                if valid_frames != item["valid_frames"]:
                    raise ValueError(f"{record['id']}: valid feature length changed")
                if int(features.shape[2]) != tensor_frames:
                    raise ValueError(f"{record['id']}: padded feature length changed")
                features = np.ascontiguousarray(features, dtype=np.float32)

                logits_flat, infer_ms = server.infer(
                    features,
                    request_index=request_index,
                )
                if logits_flat.size != output_elements:
                    raise RuntimeError(
                        f"{record['id']}: MYRIAD output has {logits_flat.size} "
                        f"elements, expected {output_elements}"
                    )
                logits = logits_flat.reshape(1, output_frames, 29)
                if not np.isfinite(logits).all():
                    raise RuntimeError(f"{record['id']}: non-finite MYRIAD logits")

                if parity is None:
                    reference_logits = ort_session.run(
                        ["logits"],
                        {"features": features},
                    )[0]
                    full_comparison = compare_arrays(reference_logits, logits)
                    valid_output_frames = int(item["valid_output_frames"])
                    valid_comparison = compare_arrays(
                        reference_logits[:, :valid_output_frames, :],
                        logits[:, :valid_output_frames, :],
                    )
                    if valid_comparison.get("frame_argmax_agreement") != 1.0:
                        raise RuntimeError(
                            "ONNX/MYRIAD valid-frame semantic parity failed: "
                            f"argmax={valid_comparison.get('frame_argmax_agreement')}"
                        )
                    valid_max_tv = float(
                        valid_comparison.get("max_frame_total_variation", 1.0)
                    )
                    parity = {
                        "sample_id": record["id"],
                        "tensor_frames": tensor_frames,
                        "valid_output_frames": valid_output_frames,
                        "hard_gate": {
                            "valid_frame_argmax_agreement": 1.0,
                            "pass": True,
                        },
                        "probability_drift_diagnostic": {
                            "reference_max_frame_total_variation": (
                                DIAGNOSTIC_MAX_FRAME_TOTAL_VARIATION
                            ),
                            "within_reference_tolerance": (
                                valid_max_tv
                                <= DIAGNOSTIC_MAX_FRAME_TOTAL_VARIATION
                            ),
                            "gating": False,
                        },
                        "valid_comparison": valid_comparison,
                        "full_tensor_comparison": full_comparison,
                    }
                    print(
                        "[quartznet-myriad] semantic probe: PASS "
                        f"sample={record['id']} "
                        f"valid_argmax="
                        f"{valid_comparison['frame_argmax_agreement']:.9f} "
                        f"valid_max_tv={valid_max_tv:.9f} "
                        f"full_max_tv="
                        f"{full_comparison['max_frame_total_variation']:.9f} "
                        f"tv_reference_pass="
                        f"{valid_max_tv <= DIAGNOSTIC_MAX_FRAME_TOTAL_VARIATION}",
                        flush=True,
                    )

                valid_logits = logits[0, : item["valid_output_frames"], :]
                decoder = greedy_decode_logits_diagnostics(valid_logits, vocab)
                hypothesis = decoder["hypothesis"]
                reference = str(record["text"])
                words = word_error_counts(reference, hypothesis)
                chars = character_error_counts(reference, hypothesis)
                word_counts.append(words)
                char_counts.append(chars)
                decoder_samples.append(
                    {
                        "reference_words": len(reference.split()),
                        "reference_characters_no_spaces": len(reference.replace(" ", "")),
                        "decoder": {
                            key: value for key, value in decoder.items()
                            if key != "hypothesis"
                        },
                        "hypothesis": hypothesis,
                    }
                )
                infer_latencies.append(infer_ms)
                total_infer_seconds += infer_ms / 1000.0
                total_audio_samples += int(record["sample_count"])
                per_sample[item["index"]] = {
                    "id": record["id"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "word_edits": words.to_dict(),
                    "character_edits": chars.to_dict(),
                    "frontend_ms": frontend_ms,
                    "inference_ms": infer_ms,
                    "valid_feature_frames": item["valid_frames"],
                    "tensor_feature_frames": tensor_frames,
                    "valid_output_frames": item["valid_output_frames"],
                    "tensor_output_frames": output_frames,
                    "shape_session_index": session_index,
                }
                processed += 1
                if args.progress_interval and (
                    processed == 1
                    or processed == len(prepared)
                    or processed % args.progress_interval == 0
                ):
                    elapsed = time.monotonic() - evaluation_started
                    rate = processed / max(elapsed, 1e-9)
                    print(
                        "[quartznet-myriad] "
                        f"processed={processed}/{len(prepared)} "
                        f"wer={sum_rate(word_counts):.6f} "
                        f"shapes={session_index}/{len(shape_order)} "
                        f"rate={rate:.2f}/s",
                        flush=True,
                    )

            server.close()
            server = None

        if any(item is None for item in per_sample):
            raise RuntimeError("internal error: missing per-sample MYRIAD result")
        if parity is None:
            raise RuntimeError("semantic parity gate was not executed")

        wer = sum_rate(word_counts)
        cer = sum_rate(char_counts)
        abs_wer_delta = abs(wer - QUALIFIED_WER)
        quality_pass = wer <= MAX_WER and abs_wer_delta <= MAX_ABS_WER_DELTA
        status = "diagnostic" if not full else ("pass" if quality_pass else "fail")
        inference_latency = latency_summary_ms(infer_latencies)
        frontend_latency = latency_summary_ms(frontend_latencies)
        audio_seconds = total_audio_samples / 16000.0

        hypotheses_path = args.output.with_name("hypotheses.jsonl")
        hypotheses_path.parent.mkdir(parents=True, exist_ok=True)
        hypotheses_path.write_text(
            "".join(
                json.dumps(item, sort_keys=True) + "\n"
                for item in per_sample
                if item is not None
            ),
            encoding="utf-8",
        )
        decoder_summary = aggregate_decoder_diagnostics(decoder_samples)
        result = {
            "schema": "speech-asr/quartznet-reference-myriad-evaluation",
            "version": 1,
            "status": status,
            "model_id": "quartznet15x5_nvidia_ref",
            "training_performed": False,
            "full_dev_clean": full,
            "samples": len(prepared),
            "dataset_id": "librispeech-dev-clean",
            "manifest": {
                "path": str(args.manifest),
                "sha256": sha256_path(args.manifest),
            },
            "quality": {
                "wer": wer,
                "cer": cer,
                "qualified_pytorch_wer": QUALIFIED_WER,
                "qualified_pytorch_cer": QUALIFIED_CER,
                "absolute_wer_delta": abs_wer_delta,
                "max_wer": MAX_WER,
                "max_absolute_wer_delta": MAX_ABS_WER_DELTA,
                "pass": quality_pass if full else None,
            },
            "decoder": decoder_summary,
            "semantic_parity": parity,
            "frontend": {
                "implementation": "numpy-nemo-reference-v1",
                "latency_p50_ms": frontend_latency["p50_ms"],
                "latency_p95_ms": frontend_latency["p95_ms"],
                "latency_mean_ms": sum(frontend_latencies) / len(frontend_latencies),
                "measurement_scope": "Pi CPU NumPy frontend only",
            },
            "myriad": {
                "execution_mode": EXECUTION_MODE,
                "runtime_backend": resolved_backend,
                "runtime_target": args.platform,
                "openvino_version": "2020.3.2",
                "precision": "FP16",
                "exact_time_shapes": shape_order,
                "shape_sessions": len(shape_order),
                "model_load_ms": {
                    "min": min(load_ms),
                    "max": max(load_ms),
                    "mean": sum(load_ms) / len(load_ms),
                    "total": sum(load_ms),
                },
                "inference_latency_p50_ms": inference_latency["p50_ms"],
                "inference_latency_p95_ms": inference_latency["p95_ms"],
                "inference_latency_mean_ms": sum(infer_latencies) / len(infer_latencies),
                "inference_realtime_factor": total_infer_seconds / audio_seconds,
                "measurement_scope": (
                    "MYRIAD steady-state Infer() only; exact runtime reshape per "
                    "source padded time length; one unmeasured warmup per shape; "
                    "frontend/IPC/model-load excluded"
                ),
            },
            "artifacts": {
                "xml_sha256": sha256_path(xml),
                "bin_sha256": sha256_path(binary),
                "dynamic_onnx_sha256": sha256_path(args.dynamic_onnx),
                "artifacts_json_sha256": sha256_path(artifacts_path),
            },
            "runtime": {
                "repo_commit": git_head(),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "onnxruntime_version": ort.__version__,
                "soundfile_version": sf.__version__,
            },
            "hypotheses": str(hypotheses_path),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if status != "fail" else 3
    except Exception as exc:
        if server is not None:
            server.close()
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
