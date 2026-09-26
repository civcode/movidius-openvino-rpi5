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

**By default the camera does not set that rate.** `LatestFrame.read()` keeps the
newest capture in one slot and does **not** consume it, so when inference
outruns the device the same frame is classified again instead of the loop
blocking for the next capture; the `fps` you see is then the inference path's
throughput, and `cam=` beside it is the real device rate.  After a few steady
frames a one-time note says which of the two the run is doing, and how many
times on average each capture was classified.  Pass `--fresh-frame` to go back
to one iteration per capture - that is the honest end-to-end camera rate, and
it is the number to compare against a `cam=` reading.

The device still matters for what you *see*: a webcam left at OpenCV's default
is frequently 1280x720 **YUYV**, which is USB-bandwidth bound at ~9 fps (the
same device typically does 640x480@30 in YUYV and 720p@60 in MJPG), so without
`--fresh-frame` a stale frame just gets re-classified many times.  To raise the
capture rate itself:

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
| `--every N` | 1 | classify every Nth frame (implies `--fresh-frame`: skipping only counts against real captures) |
| `--fresh-frame` | off | pace the loop to the camera: wait for a capture this run has not seen. Off by default, where the newest frame is served repeatedly - see "Frame rate" above and "Measuring throughput" below |
| `--depth N` | 2 | inference requests kept outstanding, so frame N+1's capture/preprocessing overlaps frame N's compute. The server still runs one request at a time, so this hides latency, it does not add cores |
| `--servers N` | 1 | run N server processes in parallel. This is what actually uses more cores for a small model; each costs a model load |
| `--bench N` | 0 (off) | replay one prepared tensor through the pipeline N times with no capture and no drawing, then report max inference throughput, latency spread and CPU cores busy |
| `--max-fps F` | 0 (off) | throttle the loop |
| `--request-timeout S` | 60 | wait budget per inference (first frame includes the stick boot) |
| `--headless` | off | command-line output only, no window |

## Measuring throughput

`--bench` measures the inference path on its own - no camera, no window, one
frame replayed - so the number is comparable between backends and machines:

```bash
python3 examples/webcam/webcam_mobilenet.py --headless --device CPU \
    --video vendor/models/images/sample_640x360.mp4 --bench 800
# bench:  151.5 fps over 800 inferences in 5.28 s | infer_ms p50=13.0 ... | cpu=1.02 of 32 cores
```

Scale it with `--servers` (one server computes one request at a time, so more
servers is how you use more cores). Measured on a 32-core amd64 host, FP32 IR,
`--depth 2`, 800-1200 inferences per run:

| `--servers` | throughput before the thread-placement fix | after |
|---|---|---|
| 1 | ~181 fps | ~151 fps |
| 2 | ~190 fps | ~296 fps |
| 4 | ~190 fps | ~565 fps |
| 8 | ~190 fps | ~1027 fps |
| 12 | ~190 fps | ~1324 fps |

The flat "before" column was OpenVINO's `CPU_BIND_THREAD=YES` pinning every
server's inference thread to the same core - see `docs/CPU-BACKENDS.md` and
`cpu_threading.hpp`; the small dip at one server is that thread losing core
affinity. `cpu=` in the output is client+server cores over the steady window.

One caveat: a **live** run is not bench-limited. On the video path this client
caps around ~118 fps no matter how many servers run, because decoding and
preprocessing happen in the client's single Python thread - that is the next
thing to parallelise if a live rate matters more than an inference number.

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
