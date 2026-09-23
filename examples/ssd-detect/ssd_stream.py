#!/usr/bin/env python3
"""
ssd_stream.py - live webcam object detection on the Intel Movidius MA2450
(Milestone 2 of the SSDLite integration).

Captures frames with OpenCV, sends each frame (full resolution, RGB) to the
long-running `ssd_detect --stdin` backend started by infer-ssd-server.sh
(host-native or Docker), and prints/draws the detected bounding boxes.  The
server keeps the compiled network warm, so steady-state latency is the
device rate (~90 ms/frame, ~11 fps) instead of the ~1.7 s stick boot.

Protocol (binary frames over the server's stdin):
  client -> server: uint32 width + uint32 height (little endian) + width*height*3 RGB bytes
  server -> client: "FRAME <w> <h> <infer_ms>"
                    "DET <label> <score> <x1> <y1> <x2> <y2>"  (0..N lines)
                    "END"

Modes:
  GUI (default): OpenCV window with the live frame and labelled boxes
                 (q/Esc to quit).
  --headless    : no window; one block of detections printed per frame.
  --file PATH   : read frames from an image file (.ppm/.jpg/.png) instead of
                 the webcam, looping it (deterministic test without a camera).

Usage:
  python3 examples/ssd-detect/ssd_stream.py                # GUI, webcam 0
  python3 examples/ssd-detect/ssd_stream.py --headless     # CLI output only
  python3 examples/ssd-detect/ssd_stream.py --file vendor/models/images/dog_ssd.ppm --frames 5
  python3 examples/ssd-detect/ssd_stream.py --backend docker --min-conf 0.4
"""

import argparse
import os
import select
import signal
import struct
import subprocess
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
INFER_SERVER = os.path.join(HERE, "infer-ssd-server.sh")


def note(msg):
    print(msg, file=sys.stderr, flush=True)


def die(msg, code=1):
    print("ssd_stream: " + msg, file=sys.stderr, flush=True)
    sys.exit(code)


class SsdClient:
    """Drives the ssd_detect --stdin child process over its stdin/stdout pipes."""

    def __init__(self, backend, device, min_conf, request_timeout):
        self.request_timeout = request_timeout
        self._line_buf = bytearray()
        self.proc = subprocess.Popen(
            [INFER_SERVER, backend, device, str(min_conf)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,            # server diagnostics stream to our terminal
            start_new_session=True, # so we can kill the whole group (docker too)
        )
        note("inference backend: %s device=%s min-conf=%s (server pid %d)"
             % (backend, device, min_conf, self.proc.pid))

    # The frame response is read with raw os.read on the pipe fd (never the
    # buffered reader), because mixing select() with a buffered reader can
    # leave whole responses sitting in the reader's internal buffer while
    # select() reports the fd as empty.  A chunk can contain several lines;
    # they are split at the first newline and the rest is kept in _line_buf.
    def _readline(self):
        deadline = time.monotonic() + self.request_timeout
        while True:
            pos = self._line_buf.find(b"\n")
            if pos != -1:
                line = bytes(self._line_buf[:pos])
                del self._line_buf[:pos + 1]
                return line.decode("ascii", "replace")
            if self.proc.poll() is not None:
                # server exited: drain what is left in the pipe, then decide
                try:
                    self._line_buf += os.read(self.proc.stdout.fileno(), 1 << 20)
                except OSError:
                    pass
                pos = self._line_buf.find(b"\n")
                if pos == -1:
                    raise RuntimeError(
                        "inference server exited with code %s before the frame "
                        "response was complete (see its diagnostics above)" % self.proc.returncode)
                line = bytes(self._line_buf[:pos])
                del self._line_buf[:pos + 1]
                return line.decode("ascii", "replace")
            if time.monotonic() >= deadline:
                raise RuntimeError("no frame response within %.0f s" % self.request_timeout)
            ready, _, _ = select.select([self.proc.stdout], [], [],
                                        max(0.001, deadline - time.monotonic()))
            if not ready:
                continue
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("inference server closed the response pipe")
            self._line_buf += chunk

    def detect(self, rgb):
        """rgb: HxWx3 uint8 (RGB order); returns (w, h, infer_ms, detections).
        detections: [(label, score, x1, y1, x2, y2), ...] in image pixels."""
        h, w = rgb.shape[:2]
        try:
            self.proc.stdin.write(struct.pack("<II", w, h))
            self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("inference server closed its stdin pipe (see its "
                               "diagnostics above)")
        line = self._readline()
        parts = line.split()
        if parts[0] != "FRAME":
            raise RuntimeError("unexpected server line: %r" % line)
        infer_ms = float(parts[3])
        dets = []
        while True:
            line = self._readline()
            if line == "END":
                break
            p = line.split()
            if p[0] != "DET" or len(p) != 7:
                raise RuntimeError("unexpected server line: %r" % line)
            dets.append((p[1], float(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6])))
        return w, h, infer_ms, dets

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
        for pipe in (self.proc.stdin, self.proc.stdout):
            try:
                pipe.close()
            except (BrokenPipeError, OSError):
                pass


def read_ppm(path):
    """Minimal P6 PPM reader (Pillow-independent); returns HxWx3 RGB uint8."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"P6":
        raise ValueError("not a P6 PPM: " + path)
    idx = 2

    def next_int():
        nonlocal idx
        while data[idx:idx + 1] in (b" ", b"\n", b"\r", b"\t"):
            idx += 1
        s = idx
        while 48 <= data[idx] <= 57:
            idx += 1
        return int(data[s:idx])

    w, h, mv = next_int(), next_int(), next_int()
    idx += 1
    body = data[idx:idx + w * h * 3]
    if len(body) != w * h * 3:
        raise ValueError("truncated PPM body: " + path)
    return np.frombuffer(body, dtype=np.uint8).reshape(h, w, 3)


def frame_source(args):
    """Yields BGR uint8 frames from the webcam or a looping image file."""
    if args.file:
        if args.file.lower().endswith(".ppm"):
            frame = read_ppm(args.file)
        else:
            frame = cv2.imread(args.file)
            if frame is None:
                die("cannot read image %s (OpenCV %s)" % (args.file, cv2.__version__))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        n = 0
        while args.frames == 0 or n < args.frames:
            yield rgb
            n += 1
        return
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        die("cannot open camera %d (%s); try --camera <N> or v4l2-ctl --list-devices"
            % (args.camera, cv2.__version__))
    if args.camera_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    note("camera %d: %dx%d" % (args.camera, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                die("camera frame grab failed")
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(
        description="live webcam SSDLite object detection on the Movidius "
                    "MA2450 (OpenVINO 2020.3 MYRIAD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (/dev/videoN)")
    ap.add_argument("--camera-width", type=int, default=0, help="request capture width (0 = device default)")
    ap.add_argument("--camera-height", type=int, default=0, help="request capture height (0 = device default)")
    ap.add_argument("--file", default=None,
                    help="read frames from this image file (.ppm/.jpg/.png) instead of the webcam")
    ap.add_argument("--frames", type=int, default=0,
                    help="in --file mode, stop after N frames (0 = loop forever)")
    ap.add_argument("--backend", choices=["auto", "host", "docker"], default="auto",
                    help="where ssd_detect runs")
    ap.add_argument("--device", default="MYRIAD", help="OpenVINO device name")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="detections are kept at score >= this")
    ap.add_argument("--request-timeout", type=float, default=60.0,
                    help="max seconds to wait for one inference (first frame "
                         "includes the ~1.7 s stick boot)")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI window; results are printed to stdout")
    args = ap.parse_args()

    # GUI (and webcam capture) need OpenCV; headless --file mode only needs numpy
    if cv2 is None and not (args.headless and args.file):
        die("OpenCV is required: pip install opencv-python numpy")

    client = SsdClient(args.backend, args.device, args.min_conf, args.request_timeout)

    stopping = {"flag": False}

    def on_sigint(_signum, _frame):
        note("\ninterrupt - shutting down")
        stopping["flag"] = True
    signal.signal(signal.SIGINT, on_sigint)

    window = None
    if not args.headless:
        window = "ssdlite @ MYRIAD (q to quit)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    t_start = time.monotonic()
    frame_no = 0
    try:
        for rgb in frame_source(args):
            if stopping["flag"]:
                break
            t0 = time.monotonic()
            w, h, infer_ms, dets = client.detect(rgb)
            dt = time.monotonic() - t0
            frame_no += 1
            t = time.monotonic() - t_start
            fps = frame_no / t if t > 0 else 0.0

            # results always go to the command line
            if dets:
                listing = ", ".join("%s %.2f (%d,%d,%d,%d)" % d for d in dets)
            else:
                listing = "(no detections at confidence >= %.2f)" % args.min_conf
            print("[t=%7.2fs fps=%5.1f infer=%6.1fms] %s" % (t, fps, infer_ms, listing),
                  flush=True)

            if not args.headless:
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for label, score, x1, y1, x2, y2 in dets:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = "%s %.2f" % (label, score)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 255, 0), -1)
                    cv2.putText(img, text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 1, cv2.LINE_AA)
                status = "t=%.1fs  fps=%.1f  infer=%.1f ms" % (t, fps, infer_ms)
                cv2.putText(img, status, (8, img.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(window, img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except RuntimeError as ex:
        die("inference failure: %s" % ex)
    finally:
        if window:
            cv2.destroyAllWindows()
        client.close()
        note("bye (%d frame%s read)" % (frame_no, "s" if frame_no != 1 else ""))


if __name__ == "__main__":
    main()
