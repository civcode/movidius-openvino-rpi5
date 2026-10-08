"""Client for the persistent OpenVINO 2020.3 tensor-stream server."""

from __future__ import annotations

import atexit
import os
import pathlib
import queue
import re
import select
import subprocess
import threading
import time

import numpy as np


READY_RE = re.compile(
    r"READY protocol=tensor-stream-v2 input_elements=(\d+) "
    r"output_elements=(\d+) load_ms=([0-9.]+) warmup=(\d+)"
)
TIMING_RE = re.compile(r"TIMING request=(\d+) inference_ms=([0-9.]+)")


class PersistentTensorServer:
    def __init__(
        self,
        *,
        command: list[str],
        cwd: pathlib.Path,
        input_elements: int,
        output_elements: int,
        log_path: pathlib.Path,
        startup_timeout: float = 240.0,
        request_timeout: float = 180.0,
    ) -> None:
        self.command = list(command)
        self.cwd = pathlib.Path(cwd)
        self.input_elements = int(input_elements)
        self.output_elements = int(output_elements)
        self.output_bytes = self.output_elements * 4
        self.request_timeout = float(request_timeout)
        self.load_ms = 0.0
        self.warmup = 0
        self._request_index = 0
        self._ready = threading.Event()
        self._timings: queue.Queue[tuple[int, float]] = queue.Queue()
        self._stderr_tail: list[str] = []
        self._closed = False

        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            # Ctrl+C belongs to the interactive client. Keep the persistent
            # MYRIAD model alive until it finishes the final audio window.
            start_new_session=True,
        )
        if self.proc.stdin is None or self.proc.stdout is None or self.proc.stderr is None:
            self.proc.kill()
            raise RuntimeError("persistent tensor-server pipes unavailable")

        self._thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._thread.start()
        atexit.register(self.close)

        deadline = time.monotonic() + startup_timeout
        while not self._ready.wait(0.1):
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "persistent tensor server exited during startup:\n"
                    + "\n".join(self._stderr_tail[-30:])
                )
            if time.monotonic() >= deadline:
                self.close()
                raise RuntimeError(
                    "persistent tensor server startup timed out:\n"
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
                self._timings.put(
                    (int(timing.group(1)), float(timing.group(2)))
                )

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
                    "persistent tensor response timed out:\n"
                    + "\n".join(self._stderr_tail[-30:])
                )
            chunk = os.read(fd, remaining)
            if not chunk:
                raise RuntimeError(
                    "persistent tensor output pipe closed:\n"
                    + "\n".join(self._stderr_tail[-30:])
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def infer(self, values: np.ndarray) -> tuple[np.ndarray, float]:
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
                raise RuntimeError("persistent tensor input pipe closed")
            sent += count
        self.proc.stdin.flush()

        self._request_index += 1
        raw = self._read_exact(self.output_bytes)
        try:
            timing_index, infer_ms = self._timings.get(timeout=self.request_timeout)
        except queue.Empty as exc:
            raise RuntimeError("tensor server did not report request timing") from exc
        if timing_index != self._request_index:
            raise RuntimeError(
                f"tensor timing sequence mismatch: "
                f"{timing_index} != {self._request_index}"
            )
        return (
            np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True),
            infer_ms,
        )

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

    def __enter__(self) -> "PersistentTensorServer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
