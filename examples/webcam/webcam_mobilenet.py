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

In both modes the results are printed to stdout, one block per classified
frame (a status line, then one line per top-k class); diagnostics always go
to stderr.

Modes:
  GUI (default): additionally opens an OpenCV window with the live frame and
                 the top-k overlay (q/Esc to quit).
  --headless    : no window is opened; command-line output only.

Usage:
  python3 examples/webcam/webcam_mobilenet.py                  # GUI, fp16
  python3 examples/webcam/webcam_mobilenet.py --headless       # CLI output only
  python3 examples/webcam/webcam_mobilenet.py --camera 1 --topk 3
  python3 examples/webcam/webcam_mobilenet.py --ir fp32 --backend docker
  python3 examples/webcam/webcam_mobilenet.py --headless \
      --video vendor/models/images/sample_640x360.mp4 --every 10

Frame rate: the loop can never run faster than the camera delivers frames,
because LatestFrame.read() consumes its slot and blocks for the next
capture.  A webcam left at its OpenCV default is frequently 1280x720 YUYV,
which is USB-bandwidth bound at ~9 fps - so the fps you see can be the
capture rate rather than anything to do with inference, and the CPU looks
idle because both processes spend their time blocked waiting for frames.
The status line therefore reports the measured device rate as 'cam=...',
and a one-time note says which of the two is the ceiling.  To raise it:
  --camera-fourcc MJPG --camera-fps 30   (compressed format, far higher rates)
  --camera-width 640 --camera-height 480 (YUYV reaches 30 fps here)
  more light                             (auto-exposure lengthens each frame)
'v4l2-ctl --list-formats-ext -d /dev/videoN' lists the real rates.
"""

import argparse
import os
import signal
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mobilenet_client import (            # noqa: E402
    DEFAULT_LABELS,
    OUTPUT_BYTES,
    LatestFrame,
    MyriadClient,
    add_camera_capture_args,
    fourcc_from_tag,
    load_labels,
    note,
    note_camera_settings,
    open_camera,
    preprocess,
    topk_probs,
    wait_alive,
    windowed_fps,
    WindowWatcher,
)
# The aarch64 OpenCV wheel bundles no fonts, so Qt prints a QFontDatabase
# warning on every window operation.  The overlay uses OpenCV's own text
# renderer, so the warning is pure noise; silence Qt warnings (override by
# setting QT_LOGGING_RULES yourself).
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

try:
    import cv2
except ImportError:
    cv2 = None

import threading


class PipelinedInfer:
    """Run client.infer() in a background thread so preprocessing of the
    next frame can overlap with inference of the current frame.

    Usage::

        pipeline = PipelinedInfer(client)
        pipeline.submit(tensor_1)          # starts infer_1 in background
        tensor_2 = preprocess(frame_2)     # runs while infer_1 is in-flight
        logits_1 = pipeline.wait()         # blocks until infer_1 finishes
        pipeline.submit(tensor_2)          # starts infer_2
    """

    def __init__(self, client):
        self._client = client
        self._thread = None
        self._result = None
        self._error = None
        self._t_submit = 0.0
        self._t_done = 0.0

    def submit(self, tensor):
        """Begin async inference on *tensor*; call wait() before next submit()."""
        if self._thread is not None:
            raise RuntimeError("previous inference still in flight")
        self._result = None
        self._error = None
        self._t_submit = time.monotonic()

        def _run():
            try:
                self._result = self._client.infer(tensor)
            except Exception as exc:
                self._error = exc

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def wait(self):
        """Block until the background inference finishes; return logits."""
        if self._thread is None:
            raise RuntimeError("no inference in progress")
        self._thread.join()
        self._t_done = time.monotonic()
        self._thread = None
        if self._error:
            raise RuntimeError("inference failed: %s" % self._error) from self._error
        return self._result

    @property
    def infer_ms(self):
        """Wall-clock ms of the most recent completed infer()."""
        return (self._t_done - self._t_submit) * 1000.0

    @property
    def running(self):
        return self._thread is not None


def die(msg, code=1):
    print("webcam_mobilenet: " + msg, file=sys.stderr, flush=True)
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser(
        description="live webcam MobileNet v2 classification on the Movidius "
                    "MA2450 (OpenVINO 2020.3 MYRIAD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("server_script", nargs="?", default=None,
                    help="launcher script that starts mobilenet_server, invoked as "
                         "'<script> <backend> <ir> <device>' (default: "
                         "examples/webcam/infer-server.sh)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (/dev/videoN)")
    ap.add_argument("--video", default=None,
                    help="process this video file (.mp4/.avi/.mkv/.mov) instead of the webcam; "
                         "single pass, ends at the end of the video (OpenCV must be able "
                         "to decode the codec)")
    ap.add_argument("--frames", type=int, default=0,
                    help="in --video mode, stop after N frames (0 = whole video)")
    add_camera_capture_args(ap)
    ap.add_argument("--backend", choices=["auto", "host", "docker"], default="auto",
                    help="where mobilenet_server runs")
    ap.add_argument("--ir", choices=["fp16", "fp32"], default=None,
                    help="model precision (vendor/models/mobilenet-v2-ov203/<ir>); "
                         "default: fp32 for --device CPU (the 2020.3 CPU plugin "
                         "cannot take FP16 inputs), fp16 otherwise")
    ap.add_argument("--device", default="MYRIAD",
                   help="MYRIAD (default) or CPU (amd64: OpenVINO CPU, FP32 IR; "
                        "arm64: Python ONNX Runtime server; see docs/CPU-BACKENDS.md)")
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

    # ------------------------------------------------------------- source: video or camera
    is_video = bool(args.video)
    if is_video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            die("cannot open video %s (OpenCV %s)" % (args.video, cv2.__version__))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        note("video %s: %dx%d %.0f fps, %d frames (single pass)"
             % (args.video, w, h, cap.get(cv2.CAP_PROP_FPS),
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    else:
        # Uncompressed YUYV is USB-bandwidth bound: a typical webcam offers
        # 1920x1080@6 and 1280x720@9 in YUYV but 640x480@30, while MJPG reaches
        # 720p@60 / 480p@120.  OpenCV leaves the device at whatever it defaults
        # to - very often 720p YUYV, i.e. ~9 fps - unless told otherwise, and
        # that rate (not inference) then caps the whole loop.  The shared
        # open_camera() asks for format, then geometry, then rate, and reads
        # them all back because a driver may accept only some of them.
        want_fourcc = fourcc_from_tag(args.camera_fourcc)
        cap, accepted = open_camera(args.camera, want_fourcc, args.camera_width,
                                    args.camera_height, args.camera_fps)
        if cap is None:
            die("cannot open camera %d (%s); try --camera <N> or v4l2-ctl "
                "--list-devices" % (args.camera, cv2.__version__))
        w, h = accepted[0], accepted[1]
        note_camera_settings(args.camera, accepted, want_fourcc, args.camera_fps,
                             (args.camera_width, args.camera_height))

    # Camera mode: drain the capture in a thread and classify only the
    # freshest available frame, so the ring cannot fill up behind a slow
    # request.  Note this does NOT decouple the loop rate from the device:
    # read() consumes the slot, so each iteration still needs a brand-new
    # capture and a slow camera caps the fps whatever inference does.
    source = LatestFrame(cap) if not is_video else None

    # ------------------------------------------------------------- inference
    if args.server_script and not os.path.exists(args.server_script):
        die("server launcher not found: %s" % args.server_script)
    ir = args.ir or ("fp32" if args.device.upper() == "CPU" else "fp16")
    client = MyriadClient(args.backend, ir, args.device, args.request_timeout,
                          server_script=args.server_script)
    wait_alive(client.proc)

    stopping = {"flag": False}

    def on_sigint(_signum, _frame):
        note("\ninterrupt - shutting down")
        stopping["flag"] = True
    signal.signal(signal.SIGINT, on_sigint)

    window = None
    watcher = None
    if not args.headless:
        window = "webcam mobilenet @ %s (q to quit)" % args.device.upper()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        watcher = WindowWatcher(window)

    # --- pipelined inference: overlap preprocessing of frame N+1 with
    #     inference of frame N to hide the preprocessing latency ---------
    pipeline = PipelinedInfer(client)

    last_probs = []
    last_infer_ms = 0.0
    frame_times = []
    warmup_s = None
    cam_warned = None
    fps_str = " warm"
    frame_no = 0
    classified_no = 0
    t_start = time.monotonic()

    def _read_frame():
        """Read one frame from camera/video; return frame or None on EOF."""
        nonlocal frame_no
        if is_video:
            ok, f = cap.read()
            if not ok:
                return None
        else:
            f = source.read()
        frame_no += 1
        return f

    def _show(img, probs, t_val, fps_s, infer_ms_val):
        """Draw classification overlay, show the frame, return True to quit."""
        for i, (p, cid, name) in enumerate(probs):
            y = 24 + 24 * i
            text = "%d. %.4f  %s %s" % (i + 1, p, cid, name)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (4, y - th - 2), (8 + tw, y + 2), (0, 0, 0), -1)
            cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)
        cam = ("  cam=%.1f fps" % source.camera_fps()) if source is not None else ""
        status = "t=%.1fs  fps=%s  infer=%.1f ms%s" % (
            t_val, fps_s.strip(), infer_ms_val, cam)
        cv2.putText(img, status, (8, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, img)
        key = cv2.waitKey(1) & 0xFF
        return (key == ord("q") or key == 27
                or (watcher.closed() if watcher else False)
                or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1)

    try:
        # --- warm-up: find first classified frame, preprocess, submit infer
        frame = _read_frame()
        while frame is not None and frame_no % args.every != 0:
            if not args.headless:
                if _show(frame.copy(), last_probs, 0, " warm", 0):
                    break
            frame = _read_frame()

        if frame is None:
            die("camera stream ended")

        tensor = preprocess(frame)
        t_frame0 = time.monotonic()
        pipeline.submit(tensor)
        classified_no = 1

        while not stopping["flag"]:
            if is_video and args.frames and frame_no >= args.frames:
                break

            # --- read next frame(s) while infer runs in background ----
            frame_next = _read_frame()
            next_tensor = None

            # skip non-classified frames (display them with last results)
            while frame_next is not None and frame_no % args.every != 0:
                if not args.headless:
                    t = time.monotonic() - t_start
                    if _show(frame_next.copy(), last_probs, t, fps_str, last_infer_ms):
                        stopping["flag"] = True
                        break
                frame_next = _read_frame()

            if not stopping["flag"] and frame_next is not None:
                classified_no += 1
                next_tensor = preprocess(frame_next)

            # --- wait for previous inference to complete ----------------
            logits = pipeline.wait()
            last_infer_ms = pipeline.infer_ms
            last_probs = topk_probs(logits, labels, args.topk)
            elapsed = time.monotonic() - t_frame0
            if warmup_s is None:
                warmup_s = elapsed
                note("warmup: first classification %.0f ms (includes "
                     "device compile); excluded from fps"
                     % (warmup_s * 1000))
            else:
                frame_times.append(elapsed)
            fps_str = ("%5.1f" % windowed_fps(frame_times)
                       if frame_times else " warm")

            # One-time verdict once a few steady frames are in: say plainly
            # whether the capture rate or inference is what caps the loop.
            if source is not None and cam_warned is None and len(frame_times) >= 5:
                cam_fps = source.camera_fps()
                infer_fps = 1000.0 / last_infer_ms if last_infer_ms > 0 else 0.0
                if cam_fps > 0 and infer_fps > 0 and cam_fps < infer_fps * 0.9:
                    note("\nnote: CAMERA-BOUND - the device delivers %.1f fps while "
                         "inference could sustain %.1f fps (%.1f ms), so the capture "
                         "rate is the ceiling and the CPU is mostly idle waiting for "
                         "frames.  To go faster: --camera-fourcc MJPG --camera-fps 30, "
                         "a smaller --camera-width/--camera-height, or more light "
                         "(auto-exposure lengthens each frame in dim rooms).\n"
                         % (cam_fps, infer_fps, last_infer_ms))
                elif cam_fps > 0:
                    note("note: inference-bound - camera %.1f fps, inference %.1f fps "
                         "(%.1f ms)\n" % (cam_fps, infer_fps, last_infer_ms))
                cam_warned = True

            # --- reporting ----------------------------------------------
            t = time.monotonic() - t_start
            top = "\n".join(
                "  %d. %7.4f %s %s" % (i + 1, p, cid, name)
                for i, (p, cid, name) in enumerate(last_probs))
            cam_s = (" cam=%5.1f" % source.camera_fps()) if source is not None else ""
            print("[t=%7.2fs fps=%5s infer=%6.1fms%s]\n%s"
                  % (t, fps_str, last_infer_ms, cam_s, top),
                  flush=True)

            # --- display current classified frame -----------------------
            if not args.headless:
                if _show(frame.copy(), last_probs, t, fps_str, last_infer_ms):
                    break

            if stopping["flag"]:
                break

            # --- throttle -----------------------------------------------
            if args.max_fps > 0:
                remainder = 1.0 / args.max_fps - (time.monotonic() - t_frame0)
                if remainder > 0:
                    time.sleep(remainder)

            # --- submit next inference ----------------------------------
            if next_tensor is not None:
                t_frame0 = time.monotonic()
                pipeline.submit(next_tensor)
                frame = frame_next
            else:
                break   # stream ended while waiting

    except RuntimeError as ex:
        die("inference failure: %s" % ex)
    finally:
        if source is not None:
            source.close()
        else:
            cap.release()
        if window:
            cv2.destroyAllWindows()
        client.close()
        note("bye (%d frame%s read, %d classified)" % (frame_no, "s" if frame_no != 1 else "", classified_no))


if __name__ == "__main__":
    main()
