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
  --video PATH  : read frames from a video file (.mp4/.avi/.mkv/.mov) instead
                 of the webcam; single pass, the run ends at the end of the
                 video (e.g. vendor/models/images/sample_640x360.mp4).

Usage:
  python3 examples/ssd-detect/ssd_stream.py                # GUI, webcam 0
  python3 examples/ssd-detect/ssd_stream.py --headless     # CLI output only
  python3 examples/ssd-detect/ssd_stream.py --file vendor/models/images/dog_ssd.ppm --frames 5
  python3 examples/ssd-detect/ssd_stream.py --video vendor/models/images/sample_640x360.mp4
  python3 examples/ssd-detect/ssd_stream.py --backend docker --min-conf 0.4
"""

import argparse
import os
import signal
import struct
import subprocess
import sys
import time
# The aarch64 OpenCV wheel bundles no fonts, so Qt prints a QFontDatabase
# warning on every window operation.  The overlay uses OpenCV's own text
# renderer, so the warning is pure noise; silence Qt warnings (override by
# setting QT_LOGGING_RULES yourself).
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
INFER_SERVER = os.path.join(HERE, "infer-ssd-server.sh")

sys.path.insert(0, os.path.dirname(HERE))  # examples/ (shared client helpers)
from mobilenet_client import (            # noqa: E402
    LinePipe,
    LatestFrame,
    server_exit_meaning,
    stop_server,
    wait_alive,
    windowed_fps,
    WindowWatcher,
)


def note(msg):
    print(msg, file=sys.stderr, flush=True)


def die(msg, code=1):
    print("ssd_stream: " + msg, file=sys.stderr, flush=True)
    sys.exit(code)


class SsdClient:
    """Drives the ssd_detect --stdin child process over its stdin/stdout pipes.

    The frame response is read with raw os.read on the pipe fd (never the
    buffered reader), via the shared LinePipe, because mixing select() with
    a buffered reader can leave whole responses sitting in the reader's
    internal buffer while select() reports the fd as empty.
    """

    def __init__(self, server_cmd, backend, device, min_conf, request_timeout):
        self.request_timeout = request_timeout
        self.proc = subprocess.Popen(
            [server_cmd, backend, device, str(min_conf)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,            # server diagnostics stream to our terminal
            start_new_session=True, # so we can kill the whole group (docker too)
        )
        self.pipe = LinePipe(self.proc.stdout.fileno(), self.proc)
        note("inference backend: %s device=%s min-conf=%s (server pid %d)"
             % (backend, device, min_conf, self.proc.pid))

    def _readline(self):
        try:
            return self.pipe.readline(timeout=self.request_timeout)
        except TimeoutError as ex:
            raise RuntimeError(str(ex)) from ex

    def detect(self, rgb):
        """rgb: HxWx3 uint8 (RGB order); returns (w, h, infer_ms, detections).
        detections: [(label, score, x1, y1, x2, y2), ...] in image pixels."""
        h, w = rgb.shape[:2]
        try:
            self.proc.stdin.write(struct.pack("<II", w, h))
            self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            code = self.proc.poll()
            if code is None:
                raise RuntimeError(
                    "server closed its stdin pipe but has not exited yet "
                    "(see its diagnostics above)") from e
            raise RuntimeError(
                "server closed its stdin pipe (exit code %s: %s)"
                % (code, server_exit_meaning(code))) from e
        line = self._readline()
        if line is None:
            raise RuntimeError("server closed the response pipe")
        parts = line.split()
        if parts[0] == "ERROR":
            raise RuntimeError("server reported: " + line[5:].strip())
        if parts[0] != "FRAME" or len(parts) != 4:
            raise RuntimeError("unexpected server line: %r" % line)
        try:
            infer_ms = float(parts[3])
        except ValueError:
            raise RuntimeError("unexpected server line: %r" % line) from None
        dets = []
        while True:
            line = self._readline()
            if line is None:
                raise RuntimeError("server closed the response before END")
            if line == "END":
                break
            if line.startswith("ERROR"):
                raise RuntimeError("server reported: " + line[5:].strip())
            # The label may contain spaces (COCO has multi-word classes, e.g.
            # 'fire hydrant'), so split off the five trailing numeric fields
            # and treat everything between DET and them as the label.  The
            # leading-token check keeps malformed lines a clean error instead
            # of an IndexError.
            if line.split(" ", 1)[0] != "DET":
                raise RuntimeError("unexpected server line: %r" % line)
            p = line.split("DET", 1)[1].rsplit(" ", 5)
            if len(p) != 6:
                raise RuntimeError("unexpected server line: %r" % line)
            try:
                det = (p[0].strip(), float(p[1]), int(p[2]), int(p[3]),
                       int(p[4]), int(p[5]))
            except ValueError:
                raise RuntimeError("unexpected server line: %r" % line) from None
            dets.append(det)
        return w, h, infer_ms, dets

    def close(self):
        stop_server(self.proc)
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
    """Yields RGB uint8 frames from a video file (--video), a looping image
    file (--file), or the webcam (priority: --video, --file, webcam)."""
    if args.video:
        if cv2 is None:
            die("OpenCV is required for --video: pip install opencv-python")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            die("cannot open video %s (OpenCV %s)" % (args.video, cv2.__version__))
        note("video %s: %dx%d %.0f fps, %d frames (single pass)"
             % (args.video,
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                cap.get(cv2.CAP_PROP_FPS),
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        n = 0
        try:
            while not (args.frames and n >= args.frames):
                ok, frame = cap.read()
                if not ok:
                    break  # end of video
                n += 1
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        return
    if args.file:
        if args.file.lower().endswith(".ppm"):
            # read_ppm returns RGB directly (PIL writes PPM in RGB order);
            # the model expects RGB - no channel conversion at all.
            rgb = read_ppm(args.file)
        else:
            if cv2 is None:
                die("OpenCV is required for %s (pip install opencv-python); "
                    "or convert the file to .ppm" % args.file)
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
    # Inference is slower than the camera rate: drain the capture in a
    # thread and classify only the freshest available frame.
    src = LatestFrame(cap)
    try:
        while True:
            frame = src.read()
            if frame is None:
                die("camera stream ended")
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        src.close()


def main():
    ap = argparse.ArgumentParser(
        description="live webcam SSDLite object detection on the Movidius "
                    "MA2450 (OpenVINO 2020.3 MYRIAD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (/dev/videoN)")
    ap.add_argument("--camera-width", type=int, default=0, help="request capture width (0 = device default)")
    ap.add_argument("--camera-height", type=int, default=0, help="request capture height (0 = device default)")
    ap.add_argument("infer_server", nargs="?", default=INFER_SERVER,
                    help="server script to run for inference (see examples/ssd-detect/README.md)")
    ap.add_argument("--file", default=None,
                    help="read frames from this image file (.ppm/.jpg/.png) instead of the webcam")
    ap.add_argument("--video", default=None,
                    help="process this video file (.mp4/.avi/.mkv/.mov) instead of the webcam; "
                         "single pass, ends at the end of the video (OpenCV must be able "
                         "to decode the codec)")
    ap.add_argument("--frames", type=int, default=0,
                    help="in --file/--video mode, stop after N frames "
                         "(0 = loop the image forever / whole video)")
    ap.add_argument("--backend", choices=["auto", "host", "docker"], default="auto",
                    help="where ssd_detect runs")
    ap.add_argument("--device", default="MYRIAD",
                   help="MYRIAD, HETERO:MYRIAD, or CPU (amd64: OpenVINO CPU, FP32 IR; "
                        "arm64: Python full-TensorFlow server; see docs/CPU-BACKENDS.md)")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="detections are kept at score >= this")
    ap.add_argument("--request-timeout", type=float, default=60.0,
                    help="max seconds to wait for one inference (first frame "
                         "includes the ~1.7 s stick boot)")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI window; results are printed to stdout")
    args = ap.parse_args()

    # GUI, webcam capture and --video need OpenCV; headless --file mode only
    # needs numpy (for .ppm)
    if cv2 is None and (args.video or not (args.headless and args.file)):
        die("OpenCV is required: pip install opencv-python numpy")

    client = SsdClient(args.infer_server, args.backend, args.device,
                       args.min_conf, args.request_timeout)
    wait_alive(client.proc)

    stopping = {"flag": False}

    def on_sigint(_signum, _frame):
        note("\ninterrupt - shutting down")
        stopping["flag"] = True
    signal.signal(signal.SIGINT, on_sigint)

    window = None
    watcher = None
    if not args.headless:
        window = "ssdlite @ %s (q to quit)" % args.device.upper()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        watcher = WindowWatcher(window)

    t_start = time.monotonic()
    frame_no = 0
    frame_times = []
    warmup_s = None
    try:
        for rgb in frame_source(args):
            if stopping["flag"]:
                break
            t0 = time.monotonic()
            w, h, infer_ms, dets = client.detect(rgb)
            frame_no += 1
            elapsed = time.monotonic() - t0
            if warmup_s is None:
                # first round-trip carries the one-time device compile:
                # report it as warm-up, keep it out of the fps window
                warmup_s = elapsed
                note("warmup: first inference %.0f ms (includes device "
                     "compile); excluded from fps" % (warmup_s * 1000))
            else:
                frame_times.append(elapsed)
            t = time.monotonic() - t_start
            fps_str = "%5.1f" % windowed_fps(frame_times) if frame_times else " warm"

            # results always go to the command line
            if dets:
                listing = ", ".join("%s %.2f (%d,%d,%d,%d)" % d for d in dets)
            else:
                listing = "(no detections at confidence >= %.2f)" % args.min_conf
            print("[t=%7.2fs fps=%5s infer=%6.1fms] %s" % (t, fps_str, infer_ms, listing),
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
                status = "t=%.1fs  fps=%s  infer=%.1f ms" % (t, fps_str.strip(), infer_ms)
                cv2.putText(img, status, (8, img.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(window, img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27 \
                        or (watcher.closed() if watcher else False) \
                        or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
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
