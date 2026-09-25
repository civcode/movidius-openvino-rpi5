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
    load_labels,
    note,
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
    ap.add_argument("--camera-width", type=int, default=0, help="request capture width (0 = device default)")
    ap.add_argument("--camera-height", type=int, default=0, help="request capture height (0 = device default)")
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
    cap = cv2.VideoCapture(args.video if is_video else args.camera)
    if not cap.isOpened():
        if is_video:
            die("cannot open video %s (OpenCV %s)" % (args.video, cv2.__version__))
        die("cannot open camera %d (%s); try --camera <N> or v4l2-ctl --list-devices"
            % (args.camera, cv2.__version__))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if is_video:
        note("video %s: %dx%d %.0f fps, %d frames (single pass)"
             % (args.video, w, h, cap.get(cv2.CAP_PROP_FPS),
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    else:
        if args.camera_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        note("camera %d: %dx%d" % (args.camera, w, h))

    # Camera mode: inference is slower than the camera rate, so drain the
    # capture in a thread and classify only the freshest available frame.
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

    last_probs = []
    last_infer_ms = 0.0
    frame_times = []
    warmup_s = None
    fps_str = " warm"
    frame_no = 0
    classified_no = 0
    t_start = time.monotonic()

    try:
        while not stopping["flag"]:
            if is_video and args.frames and frame_no >= args.frames:
                break
            t_frame0 = time.monotonic()
            if is_video:
                ok, frame = cap.read()
                if not ok:
                    break  # end of video - clean stop
            else:
                frame = source.read()
                if frame is None:
                    die("camera stream ended")
            frame_no += 1

            classified = frame_no % args.every == 0
            if classified:
                classified_no += 1
                tensor = preprocess(frame)
                t0 = time.monotonic()
                logits = client.infer(tensor)
                last_infer_ms = (time.monotonic() - t0) * 1000.0
                last_probs = topk_probs(logits, labels, args.topk)
                # steady-state fps over the last 10 classified frames; the
                # first (warm-up) round-trip carries the one-time device
                # compile, so it is reported separately and excluded.
                # windowed_fps expects per-frame DURATIONS, not stamps
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
            probs = last_probs

            # ------------------------------------------------------- reporting
            t = time.monotonic() - t_start
            top = "\n".join(
                "  %d. %7.4f %s %s" % (i + 1, p, cid, name)
                for i, (p, cid, name) in enumerate(probs))

            # results always go to the command line, on every classified frame
            if classified:
                print("[t=%7.2fs fps=%5s infer=%6.1fms]\n%s" % (t, fps_str, last_infer_ms, top),
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
                status = "t=%.1fs  fps=%s  infer=%.1f ms" % (t, fps_str.strip(), last_infer_ms)
                cv2.putText(img, status, (8, img.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(window, img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27 \
                        or (watcher.closed() if watcher else False) \
                        or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if args.max_fps > 0:
                # sleep whatever of the per-frame budget is still unspent
                remainder = 1.0 / args.max_fps - (time.monotonic() - t_frame0)
                if remainder > 0:
                    time.sleep(remainder)
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
