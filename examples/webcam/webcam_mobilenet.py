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
"""

import argparse
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mobilenet_client import (            # noqa: E402
    DEFAULT_LABELS,
    OUTPUT_BYTES,
    MyriadClient,
    load_labels,
    note,
    preprocess,
    topk_probs,
    windowed_fps,
)

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
    frame_times = []
    fps = 0.0
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
                # steady-state fps over the last 10 classified frames (excludes
                # the one-time server compile, like the other clients)
                frame_times.append(time.monotonic())
                if len(frame_times) >= 2:
                    fps = windowed_fps(frame_times)
            probs = last_probs

            # ------------------------------------------------------- reporting
            t = time.monotonic() - t_start
            top = "\n".join(
                "  %d. %7.4f %s %s" % (i + 1, p, cid, name)
                for i, (p, cid, name) in enumerate(probs))

            # results always go to the command line, on every classified frame
            if classified:
                print("[t=%7.2fs fps=%5.1f infer=%6.1fms]\n%s" % (t, fps, last_infer_ms, top),
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
