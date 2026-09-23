"""
mobilenet_client.py - shared helpers for the examples that drive the
long-running mobilenet_server backend (the C++ MYRIAD inference server
started by examples/webcam/infer-server.sh over stdin/stdout).

Protocol (fixed-size binary frames):
  stdin : 1x3x224x224 float32 NCHW tensors (602112 bytes each)
  stdout: 1000 float32 logits (4000 bytes) per request, until EOF

Also provides the preprocessing pinned by the ONNX model-zoo entry and
mirrored from mobilenet-test/main.cpp (resize to 224x224, BGR->RGB, /255,
mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225], NCHW float32).
"""

import os
import select
import signal
import subprocess
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
INFER_SERVER = os.path.join(HERE, "webcam", "infer-server.sh")
DEFAULT_LABELS = os.path.join(REPO_ROOT, "vendor/models/labels/synset.txt")

INPUT_SHAPE = (1, 3, 224, 224)          # what the IR expects
INPUT_BYTES = 224 * 224 * 3 * 4         # float32 tensor, 602112 bytes
OUTPUT_BYTES = 1000 * 4                 # 1000 ImageNet classes, float32

# Preprocessing pinned by the ONNX model-zoo entry and mirrored from
# mobilenet-test/main.cpp (RGB channels, /255, then (v - mean) / std).
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def note(msg):
    print(msg, file=sys.stderr, flush=True)


def die(msg, code=1):
    print("mobilenet_client: " + msg, file=sys.stderr, flush=True)
    sys.exit(code)


# The C++ servers' exit codes (shared by all three inference servers):
#   0 clean   1 runtime failure   2 device missing / bad command line
#   3 model or IO failure   4 bad command line (mobilenet_server)
def server_exit_meaning(code):
    return {
        1: "runtime failure (see its diagnostics)",
        2: "device not available or bad command line",
        3: "model or IO failure",
        4: "bad command line",
    }.get(code, "unknown")


class LinePipe:
    """Line/binary reader over a server's stdout file descriptor.

    Keeps a per-instance remainder buffer so a chunk read can end in the
    middle of a line or a binary payload.  `readline(timeout)` returns text
    lines (raising TimeoutError if the server is silent past `timeout`),
    and `read_exact(n)` returns exactly n bytes (for binary payloads such
    as the seg MASK body).  When a subprocess is attached, an exited server
    is detected and the pipe drained before the failure is reported.

    Shared by the ssd/seg stream clients so the line/binary buffering has
    one tested implementation.
    """

    def __init__(self, fd, proc=None):
        self.fd = fd
        self.proc = proc
        self.buf = bytearray()

    def _fill(self):
        try:
            chunk = os.read(self.fd, 65536)
        except OSError:
            chunk = b""
        self.buf += chunk
        return chunk

    def readline(self, timeout=None):
        """Return the next line (no newline) as text, or None on clean EOF."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            pos = self.buf.find(b"\n")
            if pos != -1:
                line, self.buf = self.buf[:pos], self.buf[pos + 1:]
                return line.decode("utf-8", "replace")
            if self.proc is not None and self.proc.poll() is not None:
                # server exited: drain whatever is left in the pipe
                self._fill()
                pos = self.buf.find(b"\n")
                if pos == -1:
                    if self.buf:
                        line, self.buf = bytes(self.buf), bytearray()
                        return line.decode("utf-8", "replace")
                    raise RuntimeError(
                        "inference server exited with code %s before the "
                        "response was complete (see its diagnostics above)"
                        % self.proc.returncode)
                line, self.buf = self.buf[:pos], self.buf[pos + 1:]
                return line.decode("utf-8", "replace")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("no server response within %.0f s" % timeout)
            if deadline is not None:
                ready, _, _ = select.select([self.fd], [], [],
                                            max(0.001, deadline - time.monotonic()))
                if not ready:
                    continue
            if not self._fill():
                # EOF: return the trailing partial line, else None
                line, self.buf = bytes(self.buf), bytearray()
                return line.decode("utf-8", "replace") if line else None

    def read_exact(self, n):
        """Return exactly n bytes, drawing on any buffered remainder first."""
        out = bytes(self.buf[:n])
        self.buf = self.buf[n:]
        while len(out) < n:
            chunk = os.read(self.fd, max(n - len(out), 4096))
            if not chunk:
                raise RuntimeError("server closed while reading a binary payload")
            out += chunk
        return bytes(out)


def windowed_fps(times, n=10):
    """Average fps over the last n frame intervals (the `times` list holds
    per-frame elapsed seconds, most recent last).  Honest for short runs:
    excludes the one-time server compile and settles at the steady rate."""
    tail = list(times)[-n:]
    if not tail:
        return 0.0
    total = sum(tail)
    return len(tail) / total if total > 0 else 0.0


def load_labels(path):
    """(class_id, display_name) per line, like the 'n0xxxxx name' synset lines."""
    if not path or not os.path.isfile(path):
        note("note: no label file at %s - showing class ids only" % (path or "<unset>"))
        return []
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            cid, name = "-", line
            if len(line) > 9 and line[0] == "n" and line[1:6].isdigit():
                space = line.find(" ")
                if space != -1 and space + 1 < len(line):
                    cid, name = line[:space], line[space + 1:]
            pairs.append((cid, name))
    return pairs


def preprocess(frame_bgr):
    """BGR uint8 image -> model input tensor (float32, NCHW, 1x3x224x224)."""
    if cv2 is None:
        die("OpenCV is required: pip install opencv-python numpy")
    small = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis, ...],
                                  dtype=np.float32)
    assert tensor.shape == INPUT_SHAPE
    return tensor


def topk_probs(logits, labels, k):
    """softmax + top-k; returns [(prob, class_id, name), ...]"""
    m = float(logits.max())
    e = np.exp(logits.astype(np.float64) - m)
    probs = e / e.sum()
    order = np.argsort(probs)[::-1][:k]
    out = []
    for idx in order:
        idx = int(idx)
        cid, name = ("?", "-")
        if idx < len(labels):
            cid, name = labels[idx]
        out.append((float(probs[idx]), cid, name))
    return out


class MyriadClient:
    """Drives the mobilenet_server child process over its stdin/stdout pipes."""

    def __init__(self, backend, ir, device, request_timeout):
        self.request_timeout = request_timeout
        self.proc = subprocess.Popen(
            [INFER_SERVER, backend, ir, device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,            # server diagnostics stream to our terminal
            start_new_session=True, # so we can kill the whole group (docker too)
        )
        note("inference backend: %s ir=%s device=%s (server pid %d)"
             % (backend, ir, device, self.proc.pid))

    def _read_exact(self, n):
        buf = bytearray()
        deadline = time.monotonic() + self.request_timeout
        while len(buf) < n:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "inference server exited with code %s before the response "
                    "was complete (see its diagnostics above)" % self.proc.returncode)
            ready, _, _ = select.select([self.proc.stdout], [], [],
                                        max(0.001, deadline - time.monotonic()))
            if not ready:
                raise RuntimeError("no inference response within %.0f s"
                                   % self.request_timeout)
            chunk = os.read(self.proc.stdout.fileno(), n - len(buf))
            if not chunk:
                raise RuntimeError("inference server closed the response pipe")
            buf += chunk
        return bytes(buf)

    def infer(self, tensor):
        data = tensor.tobytes()
        if len(data) != INPUT_BYTES:
            raise AssertionError("tensor is %d bytes, expected %d"
                                 % (len(data), INPUT_BYTES))
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            code = self.proc.poll()
            raise RuntimeError(
                "server closed its stdin pipe before the first response"
                " (exit code %s: %s)" % (code, server_exit_meaning(code))) from e
        raw = self._read_exact(OUTPUT_BYTES)
        return np.frombuffer(raw, dtype="<f4")

    def close(self):
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except Exception:
                pass
