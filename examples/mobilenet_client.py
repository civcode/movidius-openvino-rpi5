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

from collections import deque
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
    and `read_exact(n, timeout)` returns exactly n bytes (for binary
    payloads such as the seg MASK body); any bytes that arrive past the
    payload stay in the internal buffer for the next readline/read_exact.
    When a subprocess is attached, an exited server is detected and the
    pipe drained before the failure is reported (one drain fill per
    readline is enough here because each server flushes a complete response
    per frame).

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
        """Return the next line (no newline) as text, or None on clean EOF.

        Note: with an attached `proc` (all real uses), a server that has
        exited raises RuntimeError instead of returning None; the None-EOF
        return applies to a plain fd with no process to report."""
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
                # EOF.  With an attached process, give it a moment to be
                # reaped so the exit code (a far better diagnostic than a
                # bare EOF) can be reported deterministically.
                if self.proc is not None:
                    if self.proc.poll() is None:
                        try:
                            self.proc.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                    if self.proc.returncode is not None:
                        raise RuntimeError(
                            "inference server exited with code %s before the "
                            "response was complete (see its diagnostics above)"
                            % self.proc.returncode)
                line, self.buf = bytes(self.buf), bytearray()
                return line.decode("utf-8", "replace") if line else None

    def read_exact(self, n, timeout=None):
        """Return exactly n bytes, drawing on any buffered remainder first.

        Never reads more than the remaining payload count, so protocol
        lines that follow the payload in the same pipe write stay in the
        buffer (or in the pipe) for the next readline.  Optional timeout
        like readline's, so a server that announces a MASK and then hangs
        fails with TimeoutError instead of blocking forever."""
        out = bytearray(self.buf[:n])
        self.buf = self.buf[n:]
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(out) < n:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("no server payload within %.0f s" % timeout)
                ready, _, _ = select.select([self.fd], [], [],
                                            max(0.001, remaining))
                if not ready:
                    continue
            chunk = os.read(self.fd, n - len(out))
            if not chunk:
                raise RuntimeError("server closed while reading a binary payload")
            out += chunk
        return bytes(out)


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


def windowed_rate(stamps, n=10):
    """Completions per second over the last n wall-clock completion stamps.

    Counts actual finished iterations against the elapsed span between them, so
    the rate cannot disagree with the run's own t= column no matter where the
    loop waits (camera, inference, display).  `stamps` holds time.monotonic()
    values taken as each iteration finished, oldest first.
    """
    tail = list(stamps)[-(n + 1):]
    if len(tail) < 2:
        return 0.0
    span = tail[-1] - tail[0]
    return (len(tail) - 1) / span if span > 0 else 0.0


class ProcCpu:
    """Average CPU cores busy across a set of processes over a sampled window.

    /proc/<pid>/stat totals utime+stime over every thread of the process, so
    two readings bracketed by a wall-clock span give "cores busy" - the number
    that shows whether an inference path is compute-bound or mostly idling on
    round trips.  Start the window after warm-up (reset()) and read cores()
    from the steady loop; dead or unreadable pids are skipped rather than
    raising, so a server restart cannot take the reporter down.
    """

    def __init__(self, pids=()):
        self.hz = os.sysconf("SC_CLK_TCK") or 100
        self.logical = os.cpu_count() or 1
        self._pids = []
        self._ticks = 0
        self._when = 0.0
        self.add(pids)
        self.reset()

    def add(self, pids):
        for pid in pids:
            if pid is not None and int(pid) not in self._pids:
                self._pids.append(int(pid))
        return self

    def reset(self):
        """Begin a fresh measurement window."""
        self._ticks = self._total_ticks()
        self._when = time.monotonic()
        return self

    def _total_ticks(self):
        total = 0
        for pid in self._pids:
            try:
                with open("/proc/%d/stat" % pid, "rb") as fh:
                    fields = fh.read().decode("ascii", "replace")
                fields = fields.rsplit(")", 1)[1].split()
                total += int(fields[11]) + int(fields[12])      # utime + stime
            except (OSError, IndexError, ValueError):
                continue
        return total

    def cores(self):
        """Mean cores busy since reset(); None if nothing is tracked yet."""
        if not self._pids:
            return None
        span = time.monotonic() - self._when
        if span <= 0:
            return None
        return (self._total_ticks() - self._ticks) / self.hz / span

    def percent(self):
        """Cores busy as a share of all logical CPUs (None if unmeasurable)."""
        cores = self.cores()
        return None if cores is None else 100.0 * cores / self.logical


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


def pipe_capacity(stream, want=0):
    """Return the byte capacity of the pipe behind *stream*, growing it to `want`.

    A fresh Linux pipe is only 64 KiB (16 pages), which is smaller than one
    inference request, so a writer blocks until the reader drains it.  Raising
    the capacity is what lets a client hand the server its next whole request
    while the server is still computing the current one.  Best effort by
    design: F_SETPIPE_SZ is refused above /proc/sys/fs/pipe-max-size, needs
    Linux, and an unenlarged pipe is merely slower, not wrong - so 0 (unknown)
    or the previous size is always a safe answer.
    """
    try:
        fd = stream.fileno()
        if want:
            fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, int(want))
        return fcntl.fcntl(fd, fcntl.F_GETPIPE_SZ)
    except (OSError, AttributeError, ValueError):
        return 0


def wait_alive(proc, timeout=2.5):
    """Fail fast if the inference server exits during start-up.

    Start-up errors (missing model XML, missing host runtime, missing python
    dependency, unavailable device) make the server exit within a couple of
    seconds, while a healthy server stays alive as long as its warm-up takes
    (a MYRIAD stick needs ~15-20 s to boot, and that is normal).  This
    reacts to early death only, never to a slow start.
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        code = proc.poll()
        if code is not None:
            die("inference server exited during startup (exit code %s: %s); "
                "see its diagnostics above" % (code, server_exit_meaning(code)))
        time.sleep(0.05)


def add_camera_capture_args(ap, width_opt="--camera-width",
                           height_opt="--camera-height",
                           width_default=0, height_default=0,
                           fourcc_default="MJPG"):
    """Register the shared webcam-capture options on an ArgumentParser.

    All three stream examples (webcam/ssd/seg) negotiate the capture the same
    way, so the flags and their help text live here and a new capture knob
    lands in every example at once.  The geometry flag names and defaults are
    parameters because seg_stream historically spells them --width/--height
    with 640x480 defaults; renaming them would break existing invocations.
    width_opt/height_opt may be a single option string or a sequence of
    aliases - the first one names the dest, so a caller adding
    ('--width', '--camera-width') keeps args.width while both spellings work.
    """
    width_opt = ([width_opt] if isinstance(width_opt, str) else list(width_opt))
    height_opt = ([height_opt] if isinstance(height_opt, str) else list(height_opt))
    ap.add_argument(*width_opt, type=int, default=width_default,
                    help="request capture width (0 = device default)")
    ap.add_argument(*height_opt, type=int, default=height_default,
                    help="request capture height (0 = device default)")
    ap.add_argument("--camera-fps", type=float, default=0.0,
                    help="request capture frame rate (0 = device default).  Sets how "
                         "often a *new* image arrives: the loop re-classifies the "
                         "newest capture above that rate, and --fresh-frame paces "
                         "itself to it")
    ap.add_argument("--camera-fourcc", default=fourcc_default,
                    help="request this pixel format (MJPG / YUYV / none).  MJPG is "
                         "compressed and reaches far higher rates than the "
                         "bandwidth-bound YUYV default; cameras with no "
                         "compressed mode (a PS3 Eye / ov534, for instance) keep "
                         "their native format automatically; 'none' leaves the "
                         "device alone")
    ap.add_argument("--fresh-frame", action="store_true",
                    help="pace the loop to the camera: wait for a capture this run "
                         "has not seen yet.  Off by default, where the newest frame "
                         "is handed out repeatedly, so inference faster than the "
                         "device shows up as throughput instead of being capped by "
                         "the camera rate")


def fourcc_from_tag(tag):
    """'MJPG' -> OpenCV fourcc int; 'none'/'' -> None (leave the device alone)."""
    tag = (tag or "").strip().upper()
    if tag in ("", "NONE", "DEFAULT", "0"):
        return None
    if len(tag) != 4:
        die("--camera-fourcc needs a 4-char tag like MJPG or YUYV (got %r)" % tag)
    if cv2 is None:
        die("OpenCV is required for --camera-fourcc: pip install opencv-python")
    return cv2.VideoWriter_fourcc(*tag)


def fourcc_tag(value):
    """OpenCV fourcc int -> readable 4-char tag ('MJPG'); 0 -> 'none'."""
    if not value:
        return "none"
    tag = "".join(chr((int(value) >> (8 * i)) & 0xFF) for i in range(4))
    return tag.strip() or "none"


def configure_camera(cap, fourcc=None, width=0, height=0, fps=0.0):
    """Negotiate format -> geometry -> rate on an already-opened capture.

    Order matters: V4L2 re-derives the frame interval when the pixel format or
    the geometry changes, so asking for the format first and the rate last is
    what actually sticks on most drivers.  Every request is read back, because
    a driver is free to accept only some of them.

    Returns the accepted (width, height, fourcc_tag, fps).
    """
    if fourcc is not None:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fourcc_tag(cap.get(cv2.CAP_PROP_FOURCC)),
            cap.get(cv2.CAP_PROP_FPS))


def open_camera(index, fourcc=None, width=0, height=0, fps=0.0):
    """Open /dev/video<index> through the native V4L2 backend and negotiate it.

    CAP_V4L2 is named explicitly because FOURCC/FPS negotiation only goes
    through the native backend; picking it up front stops the format request
    from being silently dropped by backend auto-selection.

    A rejected format request is dropped and negotiation is retried from a
    clean open: plenty of cameras simply have no compressed mode (a PS3 Eye /
    ov534 is YUV-only yet reaches 640x480@60 on its own), and leaving a
    request the driver refused in place can perturb the geometry/rate step.

    Returns (cap, accepted) where accepted is configure_camera's tuple, or
    (None, None) when the device will not open.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, None
    accepted = configure_camera(cap, fourcc, width, height, fps)
    if fourcc is not None and accepted[2] != fourcc_tag(fourcc):
        note("camera %d: no %s mode, driver kept %s - renegotiating without a "
             "format request" % (index, fourcc_tag(fourcc), accepted[2]))
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None, None
        accepted = configure_camera(cap, None, width, height, fps)
    return cap, accepted


def note_camera_settings(index, accepted, requested_fourcc=None,
                         requested_fps=0.0, requested_size=None):
    """note() what the driver accepted against what was asked for.

    Also spells out the frame-freshness default: read() hands the newest
    capture out without consuming it, so a loop rate above the device rate is
    re-classified frames rather than new ones, and --fresh-frame is what paces
    the loop to the camera.
    """
    w, h, tag, fps = accepted
    note("camera %d: %dx%d fourcc=%s nominal=%.0f fps"
         " (requested fourcc=%s fps=%.0f%s)"
         % (index, w, h, tag, fps,
            fourcc_tag(requested_fourcc) if requested_fourcc is not None else "none",
            requested_fps or 0.0,
            "" if not requested_size else " size=%dx%d" % requested_size))
    note("note: read() hands out the newest capture without consuming it, so this")
    note("      loop is not paced by the device - a rate above %.1f fps means" % (fps or 0.0))
    note("      captures are being re-classified (--fresh-frame paces it to the")
    note("      camera).  'v4l2-ctl --list-formats-ext -d /dev/video%d' lists the real"
         % index)
    note("      rate per format, and dim light cuts it (auto-exposure lengthens frames).")


class LatestFrame:
    """Keeps the newest captured frame readable by any number of requests.

    A dedicated thread drains the device as fast as it produces frames and
    overwrites a single slot, and read() hands out whatever is newest WITHOUT
    consuming it.  That is what lets an inference-throughput measurement stand
    on its own: the loop may classify one captured frame a thousand times
    instead of being paced by the camera, and read() blocks only until the very
    first frame exists.

    read(fresh=True) restores consuming behaviour - wait for a capture this
    caller has not been served yet - which is what you want when measuring a
    live camera's real end-to-end rate rather than raw inference speed.

    reuse() reports how many reads were served from the same capture, so a
    caller can show plainly that an fps above the camera rate comes from
    repeated frames, while camera_fps() gives the device rate from the capture
    stamps.  Used for the camera path only; --video and --file keep their
    one-shot semantics.
    """

    def __init__(self, cap, history=32):
        self._cap = cap
        self._cv = threading.Condition()
        self._frame = None
        self._eof = False
        self._stop = False
        # Capture bookkeeping.  _captures counts frames the device delivered
        # (the stamps behind camera_fps()); _served counts read() calls and
        # _distinct the captures actually handed out, so reuse() can say how
        # much of a high fps is duplicate work rather than new images.
        self._arrivals = []
        self._captures = 0
        self._served = 0
        self._distinct = 0
        self._served_seq = 0
        self._history = max(2, history)
        self._last_wait = 0.0
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self):
        while not self._stop:
            ok, frame = self._cap.read()
            if not ok:
                with self._cv:
                    self._eof = True
                    self._cv.notify_all()
                return
            now = time.monotonic()
            with self._cv:
                self._frame = frame
                self._captures += 1
                self._arrivals.append(now)
                if len(self._arrivals) > self._history:
                    del self._arrivals[:len(self._arrivals) - self._history]
                self._cv.notify_all()

    def read(self, fresh=False):
        """Return the newest captured frame (BGR); None once the stream ended.

        The slot is not consumed, so back-to-back calls return the same capture
        until the drain thread replaces it.  The array is that shared slot -
        copy it before drawing into it.  With fresh=True the call instead waits
        for a capture that has not been served yet, which paces the caller to
        the device rate.
        """
        t0 = time.monotonic()
        with self._cv:
            while self._frame is None and not self._eof:
                self._cv.wait()
            if self._frame is None:
                return None
            if fresh:
                while not self._eof and self._captures == self._served_seq:
                    self._cv.wait()
                if self._frame is None:
                    return None
            self._served += 1
            if self._captures != self._served_seq:
                self._distinct += 1
                self._served_seq = self._captures
            self._last_wait = time.monotonic() - t0
            return self._frame

    @property
    def last_wait_s(self):
        """Seconds the most recent read() blocked waiting for the device."""
        return self._last_wait

    def reuse(self):
        """(reads served, distinct captures served); reads/distinct is the
        factor by which the loop re-classified frames instead of getting new
        ones - 1.0 when paced to the camera, larger when inference is faster."""
        with self._cv:
            return self._served, self._distinct

    def camera_fps(self, n=16):
        """Measured device delivery rate over the last n captured frames.

        With the default non-consuming read() this is information, not a
        ceiling: inference may run faster than the device delivers.  Only
        read(fresh=True) paces the loop to this rate.
        """
        with self._cv:
            stamps = list(self._arrivals)
        stamps = stamps[-(n + 1):]
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return (len(stamps) - 1) / span if span > 0 else 0.0

    def close(self):
        self._stop = True
        self._thread.join(timeout=2.0)
        try:
            self._cap.release()
        except Exception:
            pass


class WindowWatcher:
    """Detects that the user closed our GUI window via the window manager.

    OpenCV's getWindowProperty(WND_PROP_VISIBLE) keeps returning 1.0 after
    the X close button has destroyed the window (and waitKey(0) keeps
    blocking), so the client would otherwise run on forever.  We ask the X
    server instead whether a window with our title still exists, polling at
    most twice a second; if xwininfo is unavailable we fall back to "still
    open" and let the existing checks apply.
    """

    def __init__(self, title):
        self._title = title
        self._last_poll = 0.0
        self._closed = False

    def closed(self):
        now = time.monotonic()
        if now - self._last_poll < 0.5:
            return self._closed
        self._last_poll = now
        try:
            out = subprocess.run(
                ["xwininfo", "-root", "-tree"],
                capture_output=True, text=True, timeout=2).stdout
            self._closed = self._title not in out
        except (OSError, subprocess.SubprocessError):
            pass
        return self._closed


def _docker_container_for(proc):
    """If proc is a 'docker run' client, find its container's name.

    The launchers name their containers ov203-<sample>[-cpu]-<pid> where
    <pid> is the launcher's pid, and the launcher execs docker run, so the
    pid is proc.pid.  Matching the pid suffix makes the lookup exact.
    """
    try:
        with open("/proc/%d/cmdline" % proc.pid, "rb") as f:
            cmd = f.read().decode()
    except OSError:
        return None
    if "docker" not in cmd or "run" not in cmd:
        return None
    try:
        out = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=ov203-",
             "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        name = line.strip()
        if name.endswith("-%d" % proc.pid):
            return name
    return None


def stop_server(proc):
    """Terminate an inference server started with Popen(start_new_session=True).

    Every server treats end-of-stdin as the end of the protocol and exits 0;
    the MYRIAD servers additionally release the VPU with a clean mvnc/XLink
    deinit.  A hard kill orphans the stick's XLink session and can leave the
    USB device slow to re-enumerate, so close stdin first and give the
    server a few seconds to exit cleanly before escalating.

    The docker backend still needs special treatment in the escalation path:
    a SIGTERM to the `docker run` client does not reliably terminate it (we
    measured >170 s), and on this host the kernel also drops unhandled
    SIGTERMs to the container's PID 1 (both 'python3' and 'sh' PID 1s
    survived docker stop's grace period and needed SIGKILL), so ask the
    daemon to stop the container with a zero grace period - that SIGKILLs
    the server and the docker run client exits - then SIGTERM the process
    group, and only SIGKILL as a last resort.
    """
    if proc.poll() is not None:
        return
    # Clean shutdown: EOF on stdin is the protocol terminator for every
    # server (C++ and Python alike).
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except (OSError, ValueError):
        pass
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    container = _docker_container_for(proc)
    if container:
        try:
            subprocess.run(["docker", "stop", "-t", "0", container],
                           capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


class MyriadClient:
    """Drives the mobilenet_server child process over its stdin/stdout pipes."""

    def __init__(self, backend, ir, device, request_timeout, server_script=None):
        self.request_timeout = request_timeout
        self.proc = subprocess.Popen(
            [server_script or INFER_SERVER, backend, ir, device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,            # server diagnostics stream to our terminal
            start_new_session=True, # so we can kill the whole group (docker too)
        )
        # The request payload is 602112 B but a fresh Linux pipe holds only
        # 64 KiB, so a client write blocks until the server has drained most of
        # it - and a serial server does that only after answering the previous
        # request.  The transfer then sits on the critical path however deep the
        # client queues, so throughput stops at 1/(compute+transfer).  Growing
        # the pipe to fit a whole request is what makes depth > 1 pay off.
        self.pipe_bytes = pipe_capacity(self.proc.stdin, INPUT_BYTES + 4096)
        note("inference backend: %s ir=%s device=%s (server pid %d, request pipe "
             "%.0f KiB)" % (backend, ir, device, self.proc.pid,
                            self.pipe_bytes / 1024.0))

    def _stdin_closed_message(self):
        code = self.proc.poll()
        if code is None:
            return ("server closed its stdin pipe but has not exited yet "
                    "(see its diagnostics above)")
        return ("server closed its stdin pipe (exit code %s: %s)"
                % (code, server_exit_meaning(code)))

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

    def send(self, tensor):
        """Write one request tensor to the server without waiting for a reply.

        Split out of infer() so a pipeline thread can queue the next payload
        while an earlier response is still outstanding.  The server loop is
        strictly serial (read a whole request, compute it, write the logits),
        so a client that writes and then reads inline leaves the server idle
        for exactly as long as it takes the client to capture, preprocess and
        hand over the following frame.

        The payload goes out with os.write() over a memoryview rather than
        stdin.write(tensor.tobytes()).  A 602112 B request copied through the
        GIL costs about as much as the inference itself, and every server
        shares this process's GIL, so that copy - not the devices - caps
        throughput once there is more than one server.  os.write() releases the
        GIL while the kernel does the copy, and skipping tobytes() drops an
        allocation and a memcpy per request.
        """
        view = memoryview(tensor).cast("B")
        if view.nbytes != INPUT_BYTES:
            raise AssertionError("tensor is %d bytes, expected %d"
                                 % (view.nbytes, INPUT_BYTES))
        if not view.c_contiguous:
            raise AssertionError("tensor must be C-contiguous to send")
        fd = self.proc.stdin.fileno()
        off = 0
        try:
            while off < view.nbytes:
                off += os.write(fd, view[off:])
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(self._stdin_closed_message()) from e

    def recv(self):
        """Read one response (1000 float32 logits) and return it as an array."""
        raw = self._read_exact(OUTPUT_BYTES)
        return np.frombuffer(raw, dtype="<f4")

    def infer(self, tensor):
        self.send(tensor)
        return self.recv()

    def close(self):
        stop_server(self.proc)
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except Exception:
                pass


class InferPipeline:
    """Keep requests queued at the server so it never waits on the client.

    A synchronous client spends the whole round trip idle and the server
    spends the whole client-side prep idle.  Two daemon threads break that
    ping-pong: a writer pushes queued tensors to stdin in submission order and
    a reader collects logits as they come back, so up to `depth` requests are
    parked in the pipe.  That removes the round-trip bubble but adds no CPU
    parallelism - the server still computes one request at a time, so scaling
    past one request's worth of cores means more server processes, not a deeper
    queue.

    Usage::

        pipe = InferPipeline(client, depth=2)
        pipe.submit(tensor)               # returns as soon as it is queued
        logits, infer_ms = pipe.get()     # oldest result, blocks until ready
    """

    def __init__(self, client, depth=2):
        self._client = client
        self._depth = max(1, int(depth))
        self._queued = queue.Queue(maxsize=self._depth)
        self._done = queue.Queue()
        self._sent = deque()
        self._lock = threading.Lock()
        self._error = None
        self._stop = threading.Event()
        self._in_flight = 0
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._writer.start()
        self._reader.start()

    # ------------------------------------------------------------- threads
    def _fail(self, exc):
        if self._stop.is_set():
            return
        with self._lock:
            if self._error is None:
                self._error = exc

    def _raise(self):
        with self._lock:
            err = self._error
        if err is not None:
            raise RuntimeError("inference pipeline failed: %s" % err) from err

    def _write_loop(self):
        while not self._stop.is_set():
            try:
                tensor = self._queued.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._client.send(tensor)
            except Exception as exc:
                self._fail(exc)
                return
            with self._lock:
                self._sent.append(time.monotonic())

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                logits = self._client.recv()
            except Exception as exc:
                self._fail(exc)
                return
            with self._lock:
                t_sent = self._sent.popleft() if self._sent else None
            infer_s = (time.monotonic() - t_sent) if t_sent else 0.0
            self._done.put((logits, infer_s * 1000.0))

    # ------------------------------------------------------------ public
    def submit(self, tensor):
        """Queue *tensor* for inference; blocks only when `depth` are parked."""
        self._raise()
        with self._lock:
            self._in_flight += 1
        self._queued.put(tensor)          # back-pressure: keeps depth honoured

    def poll(self, timeout=None):
        """Like get(), but return None when nothing finished within `timeout`.

        Lets a supervising thread wait without confusing "still busy" with
        "server died": a real failure raises here rather than looking like an
        empty result.
        """
        try:
            result = self._done.get(timeout=timeout)
        except queue.Empty:
            self._raise()               # surface a writer/reader failure
            return None
        with self._lock:
            self._in_flight -= 1
        return result

    def get(self, timeout=None):
        """Return the oldest finished (logits, infer_ms), blocking until ready.

        infer_ms spans from the moment this request was written to the server
        until its logits come back.  The server computes one request at a time,
        so with depth > 1 that includes the queue wait behind whichever request
        it caught - a deeper queue raises throughput and per-request latency
        together.  Use depth=1 when the number has to be a pure round trip.
        """
        if timeout is None:
            timeout = self._client.request_timeout
        result = self.poll(timeout)
        if result is None:
            raise RuntimeError("no inference response within %.0f s" % timeout)
        return result

    @property
    def in_flight(self):
        """Submitted but not yet collected requests."""
        with self._lock:
            return self._in_flight

    @property
    def depth(self):
        return self._depth

    @property
    def request_timeout(self):
        """The per-request budget of the client behind this pipeline."""
        return self._client.request_timeout

    def close(self):
        self._stop.set()
        for thread in (self._writer, self._reader):
            thread.join(timeout=1.0)

class ServerPool:
    """Spread requests over several server processes, each with its own pipeline.

    One server computes one request at a time, so a deeper queue in front of it
    cannot raise throughput past 1/compute - the only way to use more of a many
    core CPU is more servers.  Requests go to whichever server has the least
    outstanding, and a collector thread per pipeline funnels results into one
    queue, so get() returns whatever finished first regardless of which server
    did it (results are not tied to a specific frame in a live stream, exactly
    as with a single overlapped pipeline).

    Presents the same submit()/get()/in_flight/close() face as InferPipeline,
    so callers use one or the other interchangeably.

    On a single MYRIAD stick more servers contend for the same device rather
    than adding throughput - this is for CPU backends.
    """

    def __init__(self, clients, depth=1, name=None):
        self._pipes = [InferPipeline(client, depth=depth) for client in clients]
        self._out = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._in_flight = 0
        self._error = None
        self._collectors = []
        for index, pipe in enumerate(self._pipes):
            thread = threading.Thread(target=self._collect, args=(index, pipe),
                                      daemon=True)
            thread.start()
            self._collectors.append(thread)

    def _fail(self, exc):
        with self._lock:
            if self._error is None:
                self._error = exc

    def _raise(self):
        with self._lock:
            err = self._error
        if err is not None:
            raise RuntimeError("inference pool failed: %s" % err) from err

    def _collect(self, index, pipe):
        while not self._stop.is_set():
            try:
                result = pipe.poll(0.5)
            except Exception as exc:          # server died / protocol failure
                self._fail(exc)
                return
            if result is not None:
                self._out.put(result)

    def submit(self, tensor):
        """Queue *tensor* on the least-loaded server (blocks when all are full)."""
        self._raise()
        pipe = min(self._pipes, key=lambda p: p.in_flight)
        pipe.submit(tensor)
        with self._lock:
            self._in_flight += 1

    def get(self, timeout=None):
        """Oldest finished (logits, infer_ms) across every server."""
        if timeout is None:
            timeout = self._pipes[0].request_timeout
        try:
            result = self._out.get(timeout=timeout)
        except queue.Empty:
            self._raise()
            raise RuntimeError("no inference response within %.0f s" % timeout)
        with self._lock:
            self._in_flight -= 1
        self._raise()
        return result

    @property
    def in_flight(self):
        with self._lock:
            return self._in_flight

    @property
    def servers(self):
        return len(self._pipes)

    def close(self):
        self._stop.set()
        for thread in self._collectors:
            thread.join(timeout=1.5)
        for pipe in self._pipes:
            pipe.close()
