# Live webcam classification on the Movidius MA2450

A Python app that captures webcam frames, preprocesses them for MobileNet v2
and classifies them **on the stick** (OpenVINO 2020.3 `MYRIAD` device) in
real time.  In camera mode the capture is drained in a background thread and
only the freshest available frame is classified, so the window never lags
behind a queue of frames accumulated while inference was in flight.

```
webcam_mobilenet.py (host Python, OpenCV)
  |  1x3x224x224 float32 over stdin   <-  1000 float32 logits over stdout
  v
infer-server.sh  ->  mobilenet_server  ->  libmyriadPlugin.so  ->  MA2450
```

The inference side is a small C++ program (`mobilenet_server.cpp`, built by
the Dockerfile like `mobilenet_classify`) that loads and compiles the network
**once** and then serves frames until stdin closes.  That is what keeps the
steady-state latency at the device rate (~44 ms/frame on FP16) instead of
paying the ~1.6 s stick boot on every frame.  The Python process never
touches the OpenVINO runtime; it only speaks the fixed-size byte protocol:

| direction | payload |
|---|---|
| stdin  | `1 * 3 * 224 * 224 * 4` = 602112 bytes, little-endian float32, NCHW |
| stdout | `1000 * 4` = 4000 bytes, little-endian float32 logits |
| stderr | startup diagnostics (device list, `ready` line, errors) |

Preprocessing is exactly the one pinned by the ONNX model-zoo entry and
used by `mobilenet-test/main.cpp`: resize to 224x224 (area), BGR -> RGB,
`x / 255`, subtract mean `[0.485, 0.456, 0.406]`, divide by std
`[0.229, 0.224, 0.225]`, NCHW float32.

## Requirements

* The model corpus: `./scripts/prepare-mobilenet.sh`
  (IR at `vendor/models/mobilenet-v2-ov203/{fp16,fp32}/`, labels at
  `vendor/models/labels/synset.txt`).
* Host Python 3 with `numpy` and `opencv-python`
  (`pip install -r examples/webcam/requirements.txt`).  Headless mode still
  needs OpenCV for the capture.
* For `--device CPU` on arm64 only: `pip install onnxruntime` (the CPU
  backend is then `mobilenet_cpu_server.py`, see `docs/CPU-BACKENDS.md`).
* One of the inference backends, selected automatically by
  `infer-server.sh` (override with `--backend`):
  * **host** (preferred, no Docker): `./scripts/pull-runtime.sh --platform <t>`
    so that `work/host-runtime/<target>/openvino/bin/mobilenet_server` exists.
    If the host runtime is missing but `/opt/openvino/bin/mobilenet_server`
    exists (i.e. you are already inside the runtime image), the server is
    launched from there automatically.
  * **docker**: the runtime image from `./build.sh --platform <t>`
    (it contains `mobilenet_server` in `/opt/openvino/bin`).
* A webcam visible to the host (`/dev/videoN`, your user in the `video`
  group) and the MA2450 stick attached with the usual udev/permissions
  setup (see `README.md` / `logs/HOST-RUN.md`).

## Usage

```bash
# GUI mode: OpenCV window with the live frame + top-k overlay, q/Esc to quit
python3 examples/webcam/webcam_mobilenet.py

# Headless mode: no window, one result line per classified frame on stdout
python3 examples/webcam/webcam_mobilenet.py --headless

# second camera, three classes, throttled to 10 fps, classifying every 2nd frame
python3 examples/webcam/webcam_mobilenet.py --camera 1 --topk 3 --max-fps 10 --every 2

# FP32 model, force the container backend
python3 examples/webcam/webcam_mobilenet.py --ir fp32 --backend docker

# no camera? process a video file instead (single pass, ends at video end):
python3 examples/webcam/webcam_mobilenet.py --headless \
    --video vendor/models/images/sample_640x360.mp4 --every 10 --frames 60
```

`--video` takes any file OpenCV can decode (.mp4/.avi/.mkv/.mov, h264 etc.); frames
are classified in order and the run stops at the end of the video
(`--frames N` cuts the pass short).  A short sample clip is vendored at
`vendor/models/images/sample_640x360.mp4` (Big Buck Bunny, 640×360, ~13 s, 0.6 MB).

The `fps` column is the steady-state rate over the last 10 completed
iterations; the first classified frame prints `fps= warm` because its
round-trip includes the one-time MYRIAD compile (reported on a separate
`warmup:` line and excluded from the average).  It is counted as completed
iterations per wall-clock second, so it cannot drift away from the `t=` column.

**Every frame is classified exactly once.** `LatestFrame.read()` keeps only the
newest capture in one slot and *consumes* it, so it blocks until the device
delivers a new frame: the same image is never classified twice.  `fps` therefore
counts distinct images, and `cam=` beside it is the device rate that ceilings
them.  A loop faster than the camera drops frames rather than repeating them, and
after a few steady frames a one-time note says which limit you are at -
`SOURCE-LIMITED` (the camera is) or `INFERENCE-LIMITED` (the servers are), with
how many frames were dropped.

That once-per-frame rule is what makes `--servers N` worth having: the servers
split up *different* frames instead of duplicating one image N times.

The capture rate still sets how many distinct images exist - a webcam left at
OpenCV's default is frequently 1280x720 **YUYV**, USB-bandwidth bound at ~9 fps
(the same device typically does 640x480@30 in YUYV and 720p@60 in MJPG).  To
raise it:

```bash
# compressed format: far higher rates than bandwidth-bound YUYV
python3 examples/webcam/webcam_mobilenet.py --camera-fourcc MJPG --camera-fps 30

# or drop the resolution (YUYV reaches 30 fps at 640x480)
python3 examples/webcam/webcam_mobilenet.py --camera-width 640 --camera-height 480
```

`v4l2-ctl --list-formats-ext -d /dev/video0` lists the real rate per format.
Cameras with no compressed mode - a PS3 Eye (`ov534`) is YUV-only, yet does
640x480@60 natively - are detected automatically: the rejected `MJPG` request
is dropped and the capture is renegotiated without touching the pixel format.
Dim light makes things worse: auto-exposure lengthens each frame's exposure
time, which on a sensor like the PS3 Eye's can drop 60 fps far below it.

Main options (see `--help`); the optional first positional argument replaces
the launcher script (invoked as `<script> <backend> <ir> <device>`;
default `examples/webcam/infer-server.sh`):

| option | default | meaning |
|---|---|---|
| `--camera N` | 0 | webcam index | `--video PATH` | off | classify frames of a video file instead of the webcam (single pass) |
| `--camera-width N` | 0 | request capture width (0 = device default) |
| `--camera-height N` | 0 | request capture height (0 = device default) |
| `--camera-fps F` | 0 | request capture frame rate (0 = device default); a slow camera caps the loop whatever the backend does |
| `--camera-fourcc TAG` | MJPG | request pixel format (`MJPG` / `YUYV` / `none`); MJPG is compressed and reaches far higher rates than the bandwidth-bound YUYV default |
| `--backend auto\|host\|docker` | auto | where `mobilenet_server` runs |
| `--ir fp16\|fp32` | auto (fp32 with `--device CPU`, else fp16) | model precision; auto because the 2020.3 CPU plugin cannot take FP16 inputs |
| `--device NAME` | MYRIAD | `MYRIAD` (default) or `CPU`.  CPU per target: amd64 = OpenVINO CPU (FP32 IR); arm64 = Python ONNX Runtime server `mobilenet_cpu_server.py` (host needs `pip install onnxruntime`; docker uses the in-image copy); armv7 = not supported (error).  See `docs/CPU-BACKENDS.md` |
| `--topk N` | 5 | classes to report / draw |
| `--every N` | 1 | classify every Nth frame; each capture is served once, so N>1 skips real frames |
| `--servers N` | 1 | run N server processes in parallel, one request in flight each. This is what uses more than one core; each costs a model load |
| `--report-every N` | 1 | print the status/top-k block every Nth frame. Rendering and writing a line costs milliseconds, so raise it when measuring a maximum rate |
| `--fake-camera` | off | serve frames from a static image instead of the webcam, to load the servers past what a camera can feed |
| `--fake-camera-fps F` | 30 | deliveries per second from the fake camera (0 = as fast as the loop asks) |
| `--fake-camera-image PATH` | `vendor/models/images/banana.ppm` | the still image to serve |
| `--bench N` | 0 (off) | replay one prepared tensor N times per server with no capture and no drawing, then report max inference throughput and CPU cores busy |
| `--max-fps F` | 0 (off) | throttle the loop |
| `--request-timeout S` | 60 | wait budget per inference (first frame includes the stick boot) |
| `--headless` | off | command-line output only, no window |

## Measuring throughput

`--bench` measures the inference path on its own - no camera, no window, one
frame replayed - so the number is comparable between backends and machines, and
no webcam can supply enough frames to exhaust the servers:

```bash
python3 examples/webcam/webcam_mobilenet.py --headless --device CPU \
    --fake-camera --fake-camera-fps 0 --bench 500 --servers 4
# bench:  520.7 fps aggregate from 4 servers x 600 inferences in 4.61 s |
#         130.2 fps per server | cpu=3.98 of 32 cores
```

Measured on a 32-core amd64 host, FP32 IR, 500-600 inferences per server:

| `--servers` | aggregate fps | per server | cores busy |
|---|---|---|---|
| 1 | 146 | 146 | 1.00 |
| 2 | 277 | 139 | 1.99 |
| 4 | 521 | 130 | 3.98 |
| 8 | 950 | 119 | ~8 |
| 12 | 1235 | 103 | 11.67 |

Two things to know about reading these numbers:

* **The pool has to be kept full.** Submitting one request and immediately
  collecting it leaves a single request outstanding no matter how many servers
  `--servers` starts - the extra servers idle, and a live run shows no gain at
  all.  The examples now prime one request per server and refill after each
  collect.  Measured with `--fake-camera --fake-camera-fps 0` from **one** client
  process: 148 / 310 / 617 / 1001 / 1355 fps at 1 / 2 / 4 / 8 / 12 servers
  (cpu 1.0 / 2.0 / 4.0 / 7.9 / 11.6 cores).  `--bench` with `--servers` still
  uses one client process per server (`bench_processes`), which is a useful
  cross-check but no longer the only way to scale.
* The older flat column in `docs/CPU-BACKENDS.md` (1-12 servers all ~190 fps)
  was OpenVINO's `CPU_BIND_THREAD=YES` pinning every server's inference thread
  to the same core - see `cpu_threading.hpp`.  The small drop at one server
  (~181 to ~146 fps) is that thread losing core affinity.

Queue depth is not a lever: measured at depth 1/2/4/8 a single server stayed
within noise of ~180 fps, because the server computes one request at a time - so
`--depth` was removed in favour of `--servers`.

A **live** run measures something different: every frame is classified once, so
its rate is `min(camera fps, what the client can feed)`.  With a 30 fps webcam
that is 30 fps whatever the servers can do, and `--report-every` plus the
`SOURCE-LIMITED` / `INFERENCE-LIMITED` note make clear which of the two you are
looking at.

The first frame is slow (~1.7 s: USB stick boot + VPU compile), then the
loop runs at the device rate.

## Troubleshooting

* `device MYRIAD not available` - the server retries the device probe
  (6 attempts, 500 ms apart) to ride out transient USB/mvnc states; if it
  still fails, run `./scripts/verify.sh --no-device` first, then check
  `lsusb`, udev permissions and (container backend) the device-cgroup rule.
* `host runtime missing .../mobilenet_server` - rebuild the image and re-run
  `./scripts/pull-runtime.sh --platform <target>` (older pulls predate this
  example).
* `mvnc ... global mutex initialization failed` - see `logs/HOST-RUN.md`
  (`/tmp/mvnc.mutex` ownership).
* No camera: `v4l2-ctl --list-devices`, or run the user as a member of
  `video`.
