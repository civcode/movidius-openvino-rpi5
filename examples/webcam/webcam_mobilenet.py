#!/usr/bin/env python3
"""
webcam_mobilenet.py - live webcam classification on the Intel Movidius MA2450.

Captures frames with OpenCV, preprocesses them exactly like
mobilenet-test/main.cpp (resize to 224x224, BGR->RGB, x/255, mean
[0.485, 0.456, 0.406], std [0.229, 0.224, 0.225], NCHW float32) and classifies
them on the stick through the long-running mobilenet_server backend, started
by infer-server.sh (host-native or Docker).  The server keeps the compiled
network warm, so steady-state latency is the device rate (~44 ms/frame)
instead of the ~1.6 s stick boot per frame.

In both modes the results are printed to stdout, one line per classified
frame; diagnostics always go to stderr.

Modes:
  GUI (default): additionally opens an OpenCV window with the live frame and
                 the top-k overlay (q/Esc to quit).
  --headless    : no window is opened; command-line output only.

Usage:
  python3 examples/webcam/webcam_mobilenet.py                  # GUI, fp16
  python3 examples/webcam/webcam_mobilenet.py --headless       # CLI output only
  python3 examples/webcam/webcam_mobilenet.py --camera 1 --topk 3
  python3 examples/webcam/webcam_mobilenet.py --ir fp32 --backend docker
"""

import argparse
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
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

INPUT_SHAPE = (1, 3, 224, 224)          # what the IR expects
INPUT_BYTES = 224 * 224 * 3 * 4         # float32 tensor, 602112 bytes
OUTPUT_BYTES = 1000 * 4                 # 1000 ImageNet classes, float32

# Preprocessing pinned by the ONNX model-zoo entry and mirrored from
# mobilenet-test/main.cpp (RGB channels, /255, then (v - mean) / std).
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_LABELS = os.path.join(REPO_ROOT, "vendor/models/labels/synset.txt")


def note(msg):
    print(msg, file=sys.stderr, flush=True)


def die(msg, code=1):
    print("webcam_mobilenet: " + msg, file=sys.stderr, flush=True)
    sys.exit(code)


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
    """BGR uint8 capture -> model input tensor (float32, NCHW, 1x3x224x224)."""
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
            [os.path.join(HERE, "infer-server.sh"), backend, ir, device],
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
        self.proc.stdin.write(data)
        self.proc.stdin.flush()
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
        self.proc.stdin.close()
        self.proc.stdout.close()


def main():
    ap = argparse.ArgumentParser(
        description="live webcam MobileNet v2 classification on the Movidius "
                    "MA2450 (OpenVINO 2020.3 MYRIAD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (/dev/videoN)")
    ap.add_argument("--camera-width", type=int, default=0, help="request capture width (0 = device default)")
    ap.add_argument("--camera-height", type=int, default=0, help="request capture height (0 = device default)")
    ap.add_argument("--backend", choices=["auto", "host", "docker"], default="auto",
                    help="where mobilenet_server runs")
    ap.add_argument("--ir", choices=["fp16", "fp32"], default="fp16",
                    help="model precision (vendor/models/mobilenet-v2-ov203/<ir>)")
    ap.add_argument("--device", default="MYRIAD", help="OpenVINO device name")
    ap.add_argument("--labels", default=DEFAULT_LABELS, help="synset label file")
    ap.add_argument("--topk", type=int, default=5, help="classes to report")
    ap.add_argument("--every", type=int, default=1,
                    help="classify every Nth frame (others reuse the last result)")
    ap.add_argument("--max-fps", type=float, default=0.0,
                    help="throttle the capture/classify loop to N fps (0 = unthrottled)")
    ap.add_argument("--request-timeout", type=float, default=60.0,
                    help="max seconds to wait for one inference (first frame "
                         "includes the ~1.6 s stick boot)")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI window; results are printed to stdout")
    args = ap.parse_args()

    if cv2 is None:
        die("OpenCV is required: pip install opencv-python numpy "
            "(headless mode still needs it for the webcam capture)")

    labels = load_labels(args.labels)
    if labels and len(labels) != OUTPUT_BYTES // 4:
        note("note: label file has %d lines but the model outputs %d classes"
             % (len(labels), OUTPUT_BYTES // 4))

    # ------------------------------------------------------------------ camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        die("cannot open camera %d (%s); try --camera <N> or v4l2-ctl --list-devices"
            % (args.camera, cv2.__version__))
    if args.camera_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    note("camera %d: %dx%d" % (args.camera, w, h))

    # ------------------------------------------------------------- inference
    client = MyriadClient(args.backend, args.ir, args.device, args.request_timeout)

    stopping = {"flag": False}

    def on_sigint(_signum, _frame):
        note("\ninterrupt - shutting down")
        stopping["flag"] = True
    signal.signal(signal.SIGINT, on_sigint)

    window = None
    if not args.headless:
        window = "webcam mobilenet @ MYRIAD (q to quit)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last_probs = []
    last_infer_ms = 0.0
    frame_no = 0
    classified_no = 0
    t_start = time.monotonic()

    try:
        while not stopping["flag"]:
            t_frame0 = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                die("camera frame grab failed")
            frame_no += 1

            classified = frame_no % args.every == 0
            if classified:
                classified_no += 1
                tensor = preprocess(frame)
                t0 = time.monotonic()
                logits = client.infer(tensor)
                last_infer_ms = (time.monotonic() - t0) * 1000.0
                last_probs = topk_probs(logits, labels, args.topk)
            probs = last_probs

            # ------------------------------------------------------- reporting
            t = time.monotonic() - t_start
            fps = frame_no / t if t > 0 else 0.0
            top = "  |  ".join(
                "%d. %7.4f %s %s" % (i + 1, p, cid, name)
                for i, (p, cid, name) in enumerate(probs))

            # results always go to the command line, on every classified frame
            if classified:
                print("[t=%7.2fs fps=%5.1f infer=%6.1fms] %s" % (t, fps, last_infer_ms, top),
                      flush=True)

            if not args.headless:
                img = frame.copy()
                for i, (p, cid, name) in enumerate(probs):
                    y = 24 + 24 * i
                    text = "%d. %.4f  %s %s" % (i + 1, p, cid, name)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(img, (4, y - th - 2), (8 + tw, y + 2), (0, 0, 0), -1)
                    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 255, 0), 1, cv2.LINE_AA)
                status = "t=%.1fs  fps=%.1f  infer=%.1f ms" % (t, fps, last_infer_ms)
                cv2.putText(img, status, (8, img.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(window, img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if args.max_fps > 0:
                # sleep whatever of the per-frame budget is still unspent
                remainder = 1.0 / args.max_fps - (time.monotonic() - t_frame0)
                if remainder > 0:
                    time.sleep(remainder)
    except RuntimeError as ex:
        die("inference failure: %s" % ex)
    finally:
        cap.release()
        if window:
            cv2.destroyAllWindows()
        client.close()
        note("bye (%d frame%s read, %d classified)" % (frame_no, "s" if frame_no != 1 else "", classified_no))


if __name__ == "__main__":
    main()
