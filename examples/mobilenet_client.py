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

import argparse
import fcntl
import os
import queue
import select
import signal
import subprocess
import sys
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PYTHON_ROOT = os.path.join(REPO_ROOT, "python")
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)
from ov203.camera import (
    FakeCamera, LatestFrame, add_camera_capture_args, add_fake_camera_args,
    fourcc_from_tag, load_static_image, note_camera_settings, open_camera,
)
from ov203.device import resolve_servers
from ov203.display import DisplayPump, WindowWatcher
from ov203.process import (
    ProcCpu, server_exit_meaning, spawn_servers, stop_server, wait_alive,
)
from ov203.protocol import MobileNetPipeClient
from ov203.request_pool import RequestPool
from ov203.util import windowed_rate
from ov203.runtime import models_root

INFER_SERVER = os.path.join(HERE, "webcam", "infer-server.sh")
DEFAULT_LABELS = str(models_root(__file__) / "labels" / "synset.txt")

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
def windowed_fps(times, n=10):
    """Average fps over the last n frame intervals (the `times` list holds
    per-frame elapsed seconds, most recent last).  Callers must NOT append
    the first (warm-up) round-trip here: it contains the one-time device
    compile and would drag the average for n frames.  The stream clients
    report that sample separately as `warm` and window only steady-state
    frames, so the printed fps is the steady rate from frame 2 onward.

    Note this is a *duration-derived* rate: it says how fast the measured work
    would run if iterations touched end to end, and it ignores everything
    outside the interval that was timed - waiting for a camera frame included
    or excluded changes the number completely.  Two stream clients historically
    timed different windows, which is why their fps columns disagreed with each
    other and with their own t= column.  Prefer windowed_rate(), which counts
    real completions against real elapsed time.
    """
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


class MyriadClient(MobileNetPipeClient):
    """Compatibility wrapper for the historical MobileNet client API."""

    def __init__(self, backend, ir, device, request_timeout, server_script=None):
        super().__init__(
            server_script or INFER_SERVER,
            backend,
            ir,
            device,
            request_timeout,
            input_bytes=INPUT_BYTES,
            output_elements=OUTPUT_BYTES // 4,
        )

def bench_processes(client_factory, payload, workers, iters, progress_every=1.0):
    """Measure N servers, each driven by its own client process.

    Returns (aggregate_fps, seconds, cores_busy, per_worker_iters).

    Why processes and not the --servers thread pool: a single Python client
    cannot feed more than about one server's worth of requests, because the
    per-request queue, numpy and syscall work is GIL-serialised.  Measured here
    with MobileNet v2 on CPU, one client process plateaued at ~150-180
    inferences/s whether it had 1, 2, 4 or 8 servers attached, while four
    separate client+server pairs reached ~550/s.  So to measure the ceiling of
    N servers the client has to leave the GIL as well.

    Each worker builds its own server (client_factory runs in the child), does
    one warm round trip that carries the device boot/compile, waits until every
    worker is ready, then pushes `payload` `iters` times.  The parent samples a
    shared completion counter, so the reported rate covers only the steady
    window with all workers running at once.

    Fork is requested explicitly: the closure and payload are inherited rather
    than pickled, and the children start before this process has any pool
    threads, so nothing is inherited in a locked state.
    """
    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    done = ctx.Value("L", 0)
    ready = ctx.Value("L", 0)
    go = ctx.Event()

    def child():
        client = client_factory()
        try:
            client.send(payload)
            client.recv()                       # warm: boot + compile, untimed
            with ready.get_lock():
                ready.value += 1
            go.wait()
            for _ in range(iters):
                client.send(payload)
                client.recv()
                with done.get_lock():
                    done.value += 1
        finally:
            client.close()

    workers = max(1, int(workers))
    procs = [ctx.Process(target=child, daemon=True) for _ in range(workers)]
    for proc in procs:
        proc.start()
    timeout = 240.0                             # model load per server
    deadline = time.monotonic() + timeout
    while ready.value < workers and time.monotonic() < deadline:
        time.sleep(0.05)
    if ready.value < workers:
        go.set()
        for proc in procs:
            proc.terminate()
        raise RuntimeError("only %d/%d benchmark servers became ready in %.0f s"
                           % (ready.value, workers, timeout))
    cpu = ProcCpu([p.pid for p in procs], include_children=True)
    go.set()
    t0 = time.monotonic()
    last = t0
    cores = None
    while any(p.is_alive() for p in procs):
        time.sleep(0.2)
        now = time.monotonic()
        # Sample while they live.  Once a worker is reaped its /proc counters are
        # gone, and a later reading subtracts from a baseline that included them,
        # which reports a negative core count.  cores() is cached, so calling it
        # every pass is cheap.
        live = cpu.cores()
        # Keep the peak: workers finish at slightly different times, and every
        # sample taken after one of them is reaped loses that process's counters
        # from the total and reads low (even negative).  The run is meant to be
        # saturated, so the highest reading is the honest one.
        if live is not None and (cores is None or live > cores):
            cores = live
        if progress_every and now - last >= progress_every:
            with done.get_lock():
                n = done.value
            print("bench: %7d / %d inferences  %6.1f fps so far  cpu=%s cores"
                  % (n, workers * iters, n / (now - t0),
                     "%.2f" % cores if cores is not None else "?"),
                  file=sys.stderr, flush=True)
            last = now
    span = time.monotonic() - t0
    for proc in procs:
        proc.join(timeout=5.0)
    with done.get_lock():
        total = done.value
    return (total / span if span > 0 else 0.0, span, cores, total)
