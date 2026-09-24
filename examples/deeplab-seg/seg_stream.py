#!/usr/bin/env python3
"""
seg_stream.py -- DeepLabV3 (Pascal VOC) segmentation client.

Sends camera frames (or a single image file) to the C++ seg_detect
server in --stdin mode and renders a per-pixel class map over the frame.

Usage:
  python3 examples/deeplab-seg/seg_stream.py                 # GUI, webcam 0
  python3 examples/deeplab-seg/seg_stream.py --headless       # CLI output only
  python3 examples/deeplab-seg/seg_stream.py --file dog_ssd.ppm --frames 1
  python3 examples/deeplab-seg/seg_stream.py --video vendor/models/images/sample_640x360.mp4
  python3 examples/deeplab-seg/seg_stream.py --backend docker --mask-out mask.ppm
  python3 examples/deeplab-seg/seg_stream.py --window-size 960x540   # initial GUI window size
  python3 examples/deeplab-seg/seg_stream.py path/to/my-launcher.sh --backend host

The first (optional) positional argument replaces the default launcher
(examples/deeplab-seg/infer-seg-server.sh); a custom launcher is invoked as
`<launcher> <backend> <device>` and must speak the protocol below.

The server protocol (see seg_detect.cpp):

  in : uint32 w  uint32 h  (little-endian) + w*h*3 RGB bytes
  out: FRAME  <w> <h> <total_ms> <infer_ms>
       CLASSES <n>
       CLASS  <id> <name> <pixels>
       MASK   <w> <h>  + w*h uint16 LE class ids
       END

--mask-out writes the class map as a P6 PPM (1 byte/pixel, value = class
id 0..20); map it to colours with the PALETTE in this file or the
seg_detect README.

Requires numpy + opencv:  pip3 install numpy opencv-python(-headless)
"""

import argparse
import os
import signal
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# single executable path, invoked as [DEFAULT_SERVER_CMD, backend, device]
# (the script must be mode 755, like the other launchers)
DEFAULT_SERVER_CMD = os.path.join(HERE, "infer-seg-server.sh")

sys.path.insert(0, os.path.dirname(HERE))  # examples/ (shared client helpers)
from mobilenet_client import LinePipe, server_exit_meaning, windowed_fps

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# Pascal VOC palette (index = class id)
PALETTE = [
    (50, 50, 50),        # 0  background
    (135, 206, 250),     # 1  aeroplane
    (255, 140, 0),       # 2  bicycle
    (255, 99, 71),       # 3  bird
    (100, 149, 237),     # 4  boat
    (153, 50, 204),      # 5  bottle
    (255, 215, 0),       # 6  bus
    (220, 20, 120),      # 7  car
    (255, 184, 100),     # 8  cat
    (144, 238, 144),     # 9  chair
    (101, 67, 33),       # 10 cow
    (210, 180, 140),     # 11 diningtable
    (160, 120, 80),      # 12 dog
    (139, 69, 19),       # 13 horse
    (127, 127, 127),     # 14 motorbike
    (255, 0, 0),         # 15 person
    (0, 150, 60),        # 16 pottedplant
    (255, 255, 240),     # 17 sheep
    (200, 120, 100),     # 18 sofa
    (80, 80, 160),       # 19 train
    (0, 191, 255),       # 20 tvmonitor
]


def note(msg):
    print(msg, file=sys.stderr, flush=True)


def die(msg, code=1):
    print("seg_stream: " + msg, file=sys.stderr, flush=True)
    sys.exit(code)


def read_ppm(path):
    """Minimal P6 PPM reader (Pillow-independent); returns HxWx3 RGB uint8.
    PIL writes PPM in RGB order, and the server expects RGB - no conversion."""
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


class SegClient:
    """Drives the seg_detect --stdin child process over its stdin/stdout pipes."""

    def __init__(self, server_cmd, backend, device, request_timeout):
        self.request_timeout = request_timeout
        # single executable path convention (same as ssd_stream.py): the
        # launcher must be an executable script, invoked as <path> <backend> <device>
        self.proc = subprocess.Popen(
            [server_cmd, backend, device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,            # server diagnostics stream to our terminal
            start_new_session=True, # so we can kill the whole group (docker too)
        )
        self.pipe = LinePipe(self.proc.stdout.fileno(), self.proc)
        note("inference backend: %s device=%s (server pid %d)"
             % (backend, device, self.proc.pid))

    def segment(self, rgb, w, h):
        """Send one RGB frame; returns (classes, mask_bytes, total_ms,
        infer_ms) where classes is a list of (id, name, pixels)."""
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

        classes = []
        mask_bytes = None
        total_ms = infer_ms = 0.0
        while True:
            line = self._readline()
            if line is None:
                raise RuntimeError("server closed stdout mid-frame")
            if line.startswith("ERROR"):
                raise RuntimeError("server reported: " + line[5:].strip())
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "FRAME":
                if len(tok) != 5:
                    raise RuntimeError("unexpected server line: %r" % line)
                try:
                    total_ms = float(tok[3])
                    infer_ms = float(tok[4])
                except ValueError:
                    raise RuntimeError("unexpected server line: %r" % line) from None
            elif tok[0] == "CLASSES":
                for _ in range(int(tok[1])):
                    l = self._readline()
                    if l is None:
                        raise RuntimeError("server closed mid-CLASS")
                    t = l.split()
                    # CLASS <id> <name ...> <pixels>; the name may contain
                    # spaces, so take the id first and the pixel count last
                    if len(t) < 4 or t[0] != "CLASS":
                        raise RuntimeError("unexpected server line: %r" % l)
                    classes.append((int(t[1]), " ".join(t[2:-1]), int(t[-1])))
            elif tok[0] == "MASK":
                if len(tok) != 3:
                    raise RuntimeError("unexpected server line: %r" % line)
                fw, fh = int(tok[1]), int(tok[2])
                if fw <= 0 or fh <= 0:
                    raise RuntimeError("bad MASK dimensions: %r" % line)
                try:
                    mask_bytes = self.pipe.read_exact(fw * fh * 2,
                                                      timeout=self.request_timeout)
                except TimeoutError as ex:
                    raise RuntimeError(str(ex)) from ex
            elif tok[0] == "END":
                return classes, mask_bytes, total_ms, infer_ms
        # unreachable

    def _readline(self):
        try:
            return self.pipe.readline(timeout=self.request_timeout)
        except TimeoutError as ex:
            raise RuntimeError(str(ex)) from ex

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


def overlay(frame_bgr, mask_u16, w, h, alpha=0.4):
    m = np.frombuffer(mask_u16, dtype="uint16").reshape(h, w)
    pal = np.array(PALETTE, dtype="uint8")
    colored = pal[m]              # (h, w, 3) RGB from palette
    rgb_frame = frame_bgr[:, :, ::-1]
    out = (alpha * colored + (1 - alpha) * rgb_frame).astype("uint8")
    # Back to BGR for cv2.  The channel reversal is a negative-stride view;
    # cv2's in-place functions (putText, imshow) reject non-contiguous
    # arrays ("Layout of the output array img is incompatible with cv::Mat"),
    # so materialise a contiguous copy here.
    return np.ascontiguousarray(out[:, :, ::-1])


def write_class_map(path, mask_u16, w, h):
    """Write the class map as a 1-byte/pixel P6 PPM (value = class id 0..20)."""
    m = np.frombuffer(mask_u16, dtype="uint16").reshape(h, w).astype("uint8")
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(m.tobytes())


def parse_window_size(spec):
    """'WxH' -> (W, H) or None for the default (frame size)."""
    if not spec:
        return None
    try:
        w, h = spec.lower().split("x")
        w, h = int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "window size must be WxH, e.g. 960x540 (got %r)" % spec)
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError("window size W and H must be >= 1")
    return w, h


_WIN_STATE = {}  # window name -> {"sized": bool}


def show_window(name, image, size_spec):
    """Show <image> in a user-resizable <name> window.

    The window is created with WINDOW_NORMAL (not the default WINDOW_AUTOSIZE)
    so the user can drag it to any size; cv2 scales the image to fit.  The
    initial size is set once - from <size_spec> ((W, H)) or, if unset, from the
    first image's dimensions - and is never touched again, so a user resize
    is never overwritten by the per-frame updates.
    """
    if name not in _WIN_STATE:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        _WIN_STATE[name] = {"sized": False}
    if image is not None:
        cv2.imshow(name, image)
    st = _WIN_STATE[name]
    if st["sized"]:
        return
    if size_spec is not None:
        w, h = size_spec
    elif image is not None:
        h, w = image.shape[:2]
    else:
        return  # no spec and no image yet: size comes with the first frame
    cv2.resizeWindow(name, w, h)  # works on WINDOW_NORMAL (not WINDOW_AUTOSIZE)
    st["sized"] = True


def main():
    ap = argparse.ArgumentParser(
        description="live webcam DeepLabV3 (Pascal VOC) segmentation on the "
                    "Movidius MA2450 (OpenVINO 2020.3 MYRIAD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("server_cmd", nargs="?", default=DEFAULT_SERVER_CMD,
                    help="server launcher script, invoked as '<script> <backend> "
                         "<device>' (default: examples/deeplab-seg/infer-seg-server.sh)")
    ap.add_argument("--backend", choices=["auto", "host", "docker"], default="auto",
                    help="where seg_detect runs (passed to the server script)")
    ap.add_argument("--device", default="MYRIAD",
                   help="OpenVINO device: MYRIAD, HETERO:MYRIAD, or CPU (CPU in amd64 images)")
    ap.add_argument("--request-timeout", type=float, default=60.0,
                    help="max seconds to wait for one frame response (first frame "
                         "includes the ~1.7 s stick boot + 513x513 inference)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (/dev/videoN)")
    ap.add_argument("--width", type=int, default=640, help="request capture width (0 = device default)")
    ap.add_argument("--height", type=int, default=480, help="request capture height (0 = device default)")
    ap.add_argument("--file", default=None,
                    help="read frames from this image file (.ppm/.jpg/.png) instead of the webcam "
                         "(image kept at its native resolution)")
    ap.add_argument("--video", default=None,
                    help="process this video file (.mp4/.avi/.mkv/.mov) instead of the webcam "
                         "(single pass, ends at the end of the video; frames kept at native size)")
    ap.add_argument("--frames", type=int, default=0,
                    help="in --file/--video mode, stop after N frames "
                         "(0 = loop the image forever / whole video)")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI window; results are printed to stdout")
    ap.add_argument("--window-size", default=None, metavar="WxH",
                    help="initial GUI window size in pixels, e.g. 960x540 "
                         "(the window is resizable either way; with this unset it "
                         "opens at the frame size)")
    ap.add_argument("--mask-out", default=None,
                    help="write the class map of the last frame as a 1-byte/pixel P6 PPM")
    args = ap.parse_args()
    win = "DeepLabV3 segmentation"
    try:
        win_size = parse_window_size(args.window_size)
    except argparse.ArgumentTypeError as ex:
        ap.error(str(ex))
    args.window_size = win_size

    # GUI, webcam capture and --video need OpenCV; headless --file .ppm only
    # needs numpy
    if cv2 is None and (args.video or not (args.headless and args.file
                                           and args.file.lower().endswith(".ppm"))):
        die("OpenCV is required: pip install opencv-python numpy")

    if not os.path.exists(args.server_cmd):
        die("server launcher not found: %s" % args.server_cmd)
    client = SegClient(args.server_cmd, args.backend, args.device, args.request_timeout)

    stopping = {"flag": False}
    t_start = time.monotonic()
    def on_sigint(_signum, _frame):
        note("\ninterrupt - shutting down")
        stopping["flag"] = True
    signal.signal(signal.SIGINT, on_sigint)

    frame_times = []
    warmup_s = None
    try:
        if args.file:
            if args.file.lower().endswith(".ppm"):
                rgb = read_ppm(args.file)
            else:
                frame = cv2.imread(args.file)
                if frame is None:
                    die("cannot read image %s (OpenCV %s)" % (args.file, cv2.__version__))
                rgb = frame[:, :, ::-1]
            h, w = rgb.shape[:2]
            n = 0
            while args.frames == 0 or n < args.frames:
                if stopping["flag"]:
                    break
                t0 = time.monotonic()
                classes, mask, total_ms, infer_ms = client.segment(rgb, w, h)
                elapsed = time.monotonic() - t0
                if warmup_s is None:
                    # first round-trip carries the one-time device compile:
                    # report it as warm-up, keep it out of the fps window
                    warmup_s = elapsed
                    note("warmup: first segmentation %.0f ms (includes device "
                         "compile); excluded from fps" % (warmup_s * 1000))
                else:
                    frame_times.append(elapsed)
                n += 1
                fps_str = "%5.1f" % windowed_fps(frame_times) if frame_times else " warm"
                print("[frame %d] %dx%d %5s fps total %6.0f ms (infer %5.0f ms): "
                      % (n, w, h, fps_str, total_ms, infer_ms)
                      + ", ".join("%s %.1f%%" % (name, 100.0 * px / (w * h))
                                  for cid, name, px in classes),
                      flush=True)
                if args.mask_out:
                    write_class_map(args.mask_out, mask, w, h)
                    note("wrote class map to %s" % args.mask_out)
            if not args.headless and n:
                overlaid = overlay(rgb[:, :, ::-1], mask, w, h)
                show_window(win, overlaid, args.window_size)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            is_video = bool(args.video)
            cap = cv2.VideoCapture(args.video if is_video else args.camera)
            if not cap.isOpened():
                if is_video:
                    die("cannot open video %s (OpenCV %s)" % (args.video, cv2.__version__))
                die("cannot open camera %d (%s); try --camera <N> or v4l2-ctl --list-devices"
                    % (args.camera, cv2.__version__))
            if is_video:
                note("video %s: %dx%d %.0f fps, %d frames (single pass)"
                     % (args.video,
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        cap.get(cv2.CAP_PROP_FPS),
                        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
            else:
                if args.width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                if args.height:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                note("camera %d: %dx%d" % (args.camera, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            last_mask = None
            last_dims = None
            video_frames = 0
            if not args.headless:
                show_window(win, None, args.window_size)  # resizable from frame one
            try:
                while True:
                    if stopping["flag"]:
                        break
                    if is_video and args.frames and video_frames >= args.frames:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        if is_video:
                            break  # end of video - clean stop
                        die("camera frame grab failed")
                    video_frames += 1
                    h, w = frame.shape[:2]
                    rgb = frame[:, :, ::-1]
                    t0 = time.monotonic()
                    classes, mask, total_ms, infer_ms = client.segment(rgb, w, h)
                    elapsed = time.monotonic() - t0
                    if warmup_s is None:
                        warmup_s = elapsed
                        note("warmup: first segmentation %.0f ms (includes "
                             "device compile); excluded from fps"
                             % (warmup_s * 1000))
                    else:
                        frame_times.append(elapsed)
                    fps_str = ("%5.1f" % windowed_fps(frame_times)
                               if frame_times else " warm")
                    last_mask = mask
                    last_dims = (w, h)
                    if args.headless:
                        print("[t=%7.2fs fps=%5s total=%6.0f ms (infer %5.0f ms)] %s"
                              % (time.monotonic() - t_start, fps_str, total_ms, infer_ms,
                                 ", ".join("%s %.1f%%" % (name, 100.0 * px / (w * h))
                                           for cid, name, px in classes)),
                              flush=True)
                    else:
                        overlaid = overlay(frame, mask, w, h)
                        txt = " ".join("%s %.0f%%" % (name, 100.0 * px / (w * h))
                                       for cid, name, px in classes[:5])
                        cv2.putText(overlaid, "%.0f ms  fps %s  %s" % (total_ms, fps_str.strip(), txt),
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1, cv2.LINE_AA)
                        show_window(win, overlaid, args.window_size)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:
                            break
                        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                            break
            finally:
                cap.release()
            if args.mask_out and last_mask is not None:
                last_w, last_h = last_dims
                write_class_map(args.mask_out, last_mask, last_w, last_h)
                note("wrote class map to %s" % args.mask_out)
    except RuntimeError as ex:
        die("inference failure: %s" % ex)
    finally:
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        client.close()
        frames_done = len(frame_times) + (1 if warmup_s is not None else 0)
        note("bye (%d frame%s read)"
             % (frames_done, "s" if frames_done != 1 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
