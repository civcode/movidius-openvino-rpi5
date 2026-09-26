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

Frame rate: every frame the camera delivers is classified exactly once, never
twice.  The capture thread keeps only the newest frame and read() blocks until a
new one arrives, so the fps column counts distinct images per second and the
device rate ('cam=...') is its ceiling: faster inference drops frames (the
startup note reports how many) rather than repeating them.  That is what makes
--servers N meaningful - the servers split up *different* frames.

'cpu=' reports cores busy across the client and every server over the steady
loop.

A webcam is still worth configuring well, because it caps the distinct images
you get: one left at OpenCV's default is frequently 1280x720 YUYV, which is
USB-bandwidth bound at ~9 fps.
  --camera-fourcc MJPG --camera-fps 30   (compressed format, far higher rates)
  --camera-width 640 --camera-height 480 (YUYV reaches 30 fps here)
  more light                             (auto-exposure lengthens each frame)
'v4l2-ctl --list-formats-ext -d /dev/videoN' lists the real rates.

Measure the inference ceiling with a source that never runs dry (a webcam caps
demand at 30-60 fps no matter how many servers run):
  --fake-camera --fake-camera-fps 0 --servers 4   static image, unbounded rate
With --fake-camera the image is preprocessed once up front, so the number is the
servers rather than this thread's resize and normalise.

Bench the inference path (no webcam needed, works off --video too):
  python3 examples/webcam/webcam_mobilenet.py --headless --device CPU \
      --video vendor/models/images/sample_640x360.mp4 --bench 500
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
    REPO_ROOT,
    FakeCamera,
    LatestFrame,
    MyriadClient,
    ProcCpu,
    RequestPool,
    resolve_servers,
    spawn_servers,
    bench_processes,
    add_camera_capture_args,
    add_fake_camera_args,
    fourcc_from_tag,
    load_labels,
    load_static_image,
    note,
    note_camera_settings,
    open_camera,
    preprocess,
    topk_probs,
    wait_alive,
    windowed_rate,
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
                    help="stop after N frames read (0 = whole video, or run until "
                         "the window closes / Ctrl-C for a live or fake camera)")
    add_camera_capture_args(ap)
    add_fake_camera_args(ap, default_image=os.path.join(
        REPO_ROOT, "vendor", "models", "images", "banana.ppm"))
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
                    help="classify every Nth frame (others reuse the last result); "
                         "each capture is served once, so N>1 skips real frames")
    ap.add_argument("--servers", type=int, default=1,
                    help="run N inference server processes in parallel, one request "
                         "in flight each.  A server computes one request at a time, "
                         "so this is the way to use more than one core; it costs a "
                         "model load per server and only scales a CPU backend")
    ap.add_argument("--bench", type=int, default=0,
                    help="run N inferences on one prepared frame with no capture "
                         "and no display in the loop, then report max inference "
                         "throughput, latency spread and CPU cores busy (0 = off)")
    ap.add_argument("--report-every", type=int, default=1,
                    help="print the status/top-k block every Nth classified frame. "
                         "Rendering and writing one line costs milliseconds per "
                         "frame, so raise this (e.g. 100) when measuring a maximum "
                         "rate; the fps/cpu counters still track every frame")
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

    # ------------------------------------------------------------- source: video, fake or camera
    is_video = bool(args.video)
    if args.fake_camera and is_video:
        die("--fake-camera and --video are alternatives; pick one")
    is_fake = bool(args.fake_camera)
    if is_video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            die("cannot open video %s (OpenCV %s)" % (args.video, cv2.__version__))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        note("video %s: %dx%d %.0f fps, %d frames (single pass)"
             % (args.video, w, h, cap.get(cv2.CAP_PROP_FPS),
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    elif is_fake:
        # A webcam tops out at 30-60 fps, so it can never tell you how fast a
        # pool of servers could really go.  FakeCamera delivers frame ticks at
        # any rate (0 = as fast as asked) from one still image, and every
        # delivery is a distinct frame, so each server gets its own work.
        image = load_static_image(args.fake_camera_image)
        cap = FakeCamera(image, args.fake_camera_fps)
        w, h = image.shape[1], image.shape[0]
        note("fake camera: %s (%dx%d) delivering %s fps"
             % (os.path.basename(args.fake_camera_image), w, h,
                "%.0f" % args.fake_camera_fps if args.fake_camera_fps else "unbounded"))
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

    # Real camera: a thread drains the capture into one slot so a slow consumer
    # always gets the newest frame rather than a queue of stale ones, and each
    # capture is served exactly once.  A video file or the fake camera is pulled
    # directly instead - neither has a buffer to overflow, and giving the fake
    # camera a free-running thread made it spin millions of deliveries a second
    # and starve the worker threads.
    source = LatestFrame(cap) if not (is_video or is_fake) else None

    # The fake camera's pixels never change, so preprocessing happens once up
    # front and the request bytes are built once too: the worker threads are
    # then left with a single os.write() per request, so a measured rate is the
    # servers' and not this process's resize, normalise or GIL handover.
    fake_payload = memoryview(preprocess(image)).cast("B") if is_fake else None

    # ------------------------------------------------------------- inference
    if args.server_script and not os.path.exists(args.server_script):
        die("server launcher not found: %s" % args.server_script)
    ir = args.ir or ("fp32" if args.device.upper() == "CPU" else "fp16")
    # A server computes one request at a time, so throughput beyond ~1/compute
    # needs several processes - that is what --servers buys.  A single MYRIAD
    # stick stays one device however many processes hold it, so extra servers
    # contend there instead of scaling.
    # A CPU backend scales with processes - that is what --servers buys.  One
    # MYRIAD stick is a single device however many processes hold it, so
    # resolve_servers() clamps non-CPU devices to one instead of contending.
    nservers = resolve_servers(args.servers, args.device)
    clients = spawn_servers(
        lambda: MyriadClient(args.backend, ir, args.device, args.request_timeout,
                             server_script=args.server_script),
        nservers)
    client = clients[0]
    if len(clients) > 1:
        note("inference servers: %d x %s in parallel"
             % (len(clients), args.device.upper()))
        note("note: the pool keeps one request in flight per server.  Against a "
             "live camera the source caps distinct frames, so extra servers only "
             "help when inference is the slow side; --fake-camera "
             "--fake-camera-fps 0 removes the camera to measure the servers alone")

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

    # --- parallel inference servers ------------------------------------------
    # One worker thread per server, each with one request in flight: a server is
    # strictly serial, so depth beyond that buys nothing (measured: depth
    # 1/2/4/8 all within noise of ~180 fps) while --servers scales.  Queuing the
    # payload here keeps capture and preprocessing overlapping compute.
    pipeline = RequestPool([srv.infer for srv in clients],
                           default_timeout=args.request_timeout)

    last_probs = []
    last_infer_ms = 0.0
    stamps = []          # completion stamps: fps is a real wall-clock rate
    cam_note = None
    fps_str = " warm"
    frame_no = 0
    classified_no = 0
    t_start = time.monotonic()
    # The inference path spans two processes, so the utilisation worth watching
    # is their total; the window is reset after warm-up below.
    cpu = ProcCpu([os.getpid()] + [c.proc.pid for c in clients])

    def _budget_left():
        """False once --frames has been reached in --video mode."""
        return not (is_video and args.frames and frame_no >= args.frames)

    def _budget_left():
        """False once --frames has been reached (any source)."""
        return not (args.frames and frame_no >= args.frames)

    def _read_frame():
        """Return the next frame (BGR) from video, fake camera or webcam.

        Camera frames come from LatestFrame, which serves each capture exactly
        once: the call blocks until the device delivers a new one, so an
        iteration is always a distinct image and never a repeat.  The fake
        camera is pulled directly, which makes it paced by demand - at
        --fake-camera-fps 0 the loop asks for a frame as often as the servers
        finish one, and that rate is the inference ceiling by construction.
        None means the stream ended.
        """
        nonlocal frame_no
        if is_video or is_fake:
            ok, f = cap.read()
            if not ok:
                return None
        else:
            f = source.read()
            if f is None:
                return None
        frame_no += 1
        return f

    def _payload(frame):
        """Preprocess a frame into a request payload (bytes for the fake camera)."""
        return fake_payload if is_fake else preprocess(frame)

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
        cores = cpu.cores()
        status = "t=%.1fs  fps=%s  infer=%.1f ms  cpu=%s%s" % (
            t_val, fps_s.strip(), infer_ms_val,
            "%.2f cores" % cores if cores is not None else "?", cam)
        cv2.putText(img, status, (8, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, img)
        key = cv2.waitKey(1) & 0xFF
        return (key == ord("q") or key == 27
                or (watcher.closed() if watcher else False)
                or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1)

    # ---------------------------------------------------------- bench mode
    # Max inference throughput measured on the inference path alone: one
    # prepared tensor is replayed through the pipeline with no capture and no
    # drawing in the loop, so the number does not depend on a camera at all.
    if args.bench:
        probe = None
        for _ in range(30):
            probe = _read_frame()
            if probe is not None:
                break
        if probe is None:
            die("bench: no frame available to build a tensor from")
        tensor = fake_payload if is_fake else preprocess(probe)
        note("bench: %d inferences per server on one %dx%d frame "
             "(device=%s ir=%s servers=%d)"
             % (args.bench, probe.shape[1], probe.shape[0],
                args.device, ir, len(clients)))
        if len(clients) > 1:
            # Several servers cannot be measured from this one process: its GIL
            # serialises the per-request work and the pool plateaus at about one
            # server's rate however many servers are attached.  Hand each server
            # its own client process and measure the real ceiling.
            pipeline.close()
            for srv in clients:
                srv.close()
            script, backend, device, tmo = (args.server_script, args.backend,
                                            args.device, args.request_timeout)

            def make_client():
                return MyriadClient(backend, ir, device, tmo, server_script=script)

            try:
                agg, span, cores, total = bench_processes(make_client, tensor,
                                                          nservers, args.bench)
            except RuntimeError as ex:
                die("bench: %s" % ex)
            print("bench: %6.1f fps aggregate from %d servers x %d inferences in "
                  "%.2f s | %5.1f fps per server | cpu=%s of %d cores"
                  % (agg, nservers, args.bench, span, agg / nservers,
                     "%.2f" % cores if cores is not None else "?",
                     os.cpu_count() or 1),
                  flush=True)
            if source is not None:
                source.close()
            else:
                cap.release()
            return
        # One server: keep the in-process path, which also reports latency spread.
        # One round trip first: it carries the device compile and would
        # otherwise dominate the average.
        pipeline.submit(tensor)
        _, warm_ms = pipeline.get()
        note("bench: warm inference %.1f ms" % warm_ms)
        pipeline.submit(tensor)             # keep the server's queue fed
        cpu.reset()
        t0 = time.monotonic()
        lat = []
        for i in range(args.bench):
            _, infer_ms = pipeline.get()
            lat.append(infer_ms)
            if i + 1 < args.bench:
                pipeline.submit(tensor)      # keep one request in flight
        span = time.monotonic() - t0
        lat.sort()
        cores = cpu.cores() or 0.0
        print("bench: %6.1f fps over %d inferences in %.2f s | "
              "infer_ms p50=%.1f min=%.1f max=%.1f | cpu=%.2f of %d cores (%.0f%%)"
              % (args.bench / span, args.bench, span,
                 lat[len(lat) // 2], lat[0], lat[-1],
                 cores, cpu.logical, 100.0 * cores / cpu.logical), flush=True)
        pipeline.close()
        if source is not None:
            source.close()
        else:
            cap.release()
        for srv in clients:
            srv.close()
        return

    try:
        # --- warm-up: one full round trip, kept out of the statistics -----
        frame = _read_frame()
        while frame is not None and frame_no % args.every != 0:
            if not args.headless:
                if _show(frame.copy(), last_probs, 0, " warm", 0):
                    break
            frame = _read_frame()
        if frame is None:
            die("camera stream ended")

        pipeline.submit(_payload(frame))
        logits, last_infer_ms = pipeline.get()
        last_probs = topk_probs(logits, labels, args.topk)
        classified_no = 1
        note("warmup: first classification %.0f ms (includes device compile); "
             "excluded from fps" % last_infer_ms)
        # Utilisation would be meaningless if it covered the model load and
        # compile, so the sampling window starts here - steady loop only.
        cpu.reset()
        stamps.append(time.monotonic())

        while not stopping["flag"]:
            # --- feed: keep one request in flight per server ---------------
            # Submitting exactly one frame per collected result would leave the
            # pool with a single outstanding request no matter how many servers
            # --servers started, so top the pool up instead of trading one for
            # one.  submit() blocks only when every server is already busy.
            fed = 0
            while pipeline.in_flight < len(clients):
                nxt = _read_frame() if _budget_left() else None
                if nxt is not None and args.every > 1:
                    # --every only means something against real captures, so
                    # _read_frame() paces itself to the device in that case.
                    while nxt is not None and frame_no % args.every != 0:
                        if not args.headless and _show(
                                nxt.copy(), last_probs,
                                time.monotonic() - t_start, fps_str, last_infer_ms):
                            stopping["flag"] = True
                            break
                        nxt = _read_frame() if _budget_left() else None
                    if stopping["flag"]:
                        break
                if nxt is None:
                    break                          # stream ended / budget spent
                pipeline.submit(_payload(nxt))
                frame = nxt
                classified_no += 1
                fed += 1
            if stopping["flag"]:
                break
            if fed == 0 and pipeline.in_flight == 0:
                break                              # nothing queued and none left

            # --- collect the oldest finished result -----------------------
            logits, last_infer_ms = pipeline.get()
            # top-k is only needed for the line or the overlay about to be
            # drawn; sorting 1000 logits every frame is real work at a few
            # hundred fps, so skip it on frames that report nothing.
            if not args.headless or classified_no % args.report_every == 0:
                last_probs = topk_probs(logits, labels, args.topk)
            stamps.append(time.monotonic())
            fps = windowed_rate(stamps)
            fps_str = "%5.1f" % fps
            cores = cpu.cores()
            cpu_str = "%.2f" % cores if cores is not None else "?"
            t = time.monotonic() - t_start

            # Say once what is limiting the run: each frame is served exactly
            # once, so either the source is the ceiling or the servers are.
            if cam_note is None and len(stamps) > 6:
                if is_fake and not args.fake_camera_fps:
                    note("\n\nnote: FAKE CAMERA unbounded - frames were demanded as fast as "
                         "the %d server%s could answer, so %.1f fps is the inference "
                         "ceiling of this configuration (%s cores busy).  Add --servers to "
                         "raise it; with a real camera the device rate would cap it.\n"
                         % (len(clients), "s" if len(clients) > 1 else "", fps,
                            cpu_str))
                else:
                    src_fps = (source.camera_fps() if source is not None
                               else args.fake_camera_fps)
                    if src_fps > 0 and fps >= src_fps * 0.9:
                        note("\n\nnote: SOURCE-LIMITED - the source delivers %.1f fps and the "
                             "loop keeps up (%d frames dropped), so this rate is the source, "
                             "not inference.  Raise the capture rate, or drive the servers "
                             "harder with --fake-camera --fake-camera-fps 0.\n"
                             % (src_fps, source.dropped() if source else 0))
                    elif src_fps > 0:
                        note("\n\nnote: INFERENCE-LIMITED - the source delivers %.1f fps but the "
                             "loop sustains only %.1f fps (%d server%s, %d frames dropped).  "
                             "%.1f fps is this configuration's ceiling for distinct images; "
                             "--servers N is what raises it.\n"
                             % (src_fps, fps, len(clients),
                                "s" if len(clients) > 1 else "",
                                source.dropped() if source else 0, fps))
                cam_note = True

            # --- reporting ------------------------------------------------
            # Rendering the top-k block and writing a line costs milliseconds per
            # frame, which caps a measured rate all by itself - so --report-every
            # exists to keep the loop honest while still printing progress.
            if classified_no % args.report_every == 0:
                top = "\n".join(
                    "  %d. %7.4f %s %s" % (i + 1, p, cid, name)
                    for i, (p, cid, name) in enumerate(last_probs))
                if is_fake:
                    cam_s = ((" cam=%5.1f" % args.fake_camera_fps)
                             if args.fake_camera_fps else " cam=  unbdd")
                elif source is not None:
                    cam_s = " cam=%5.1f" % source.camera_fps()
                else:
                    cam_s = ""
                print("[t=%7.2fs fps=%5s infer=%6.1fms cpu=%s cores%s]\n%s"
                      % (t, fps_str, last_infer_ms, cpu_str, cam_s, top),
                  flush=True)

            # --- display current classified frame -----------------------
            if not args.headless:
                if _show(frame.copy(), last_probs, t, fps_str, last_infer_ms):
                    break

            # --- throttle -------------------------------------------------
            if args.max_fps > 0:
                due = stamps[-1] + 1.0 / args.max_fps - time.monotonic()
                if due > 0:
                    time.sleep(due)

    except RuntimeError as ex:
        die("inference failure: %s" % ex)
    finally:
        pipeline.close()
        if source is not None:
            source.close()
        else:
            cap.release()
        if window:
            cv2.destroyAllWindows()
        for srv in clients:
            srv.close()
        note("bye (%d frame%s read, %d classified)" % (frame_no, "s" if frame_no != 1 else "", classified_no))


if __name__ == "__main__":
    main()
