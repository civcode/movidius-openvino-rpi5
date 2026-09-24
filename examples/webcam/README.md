# Live webcam classification on the Movidius MA2450

A Python app that captures webcam frames, preprocesses them for MobileNet v2
and classifies them **on the stick** (OpenVINO 2020.3 `MYRIAD` device) in
real time.

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

The `fps` column is the steady-state rate over the last 10 classified
frames; the first classified frame prints `fps= warm` because its
round-trip includes the one-time MYRIAD compile (reported on a separate
`warmup:` line and excluded from the average).

Main options (see `--help`); the optional first positional argument replaces
the launcher script (invoked as `<script> <backend> <ir> <device>`;
default `examples/webcam/infer-server.sh`):

| option | default | meaning |
|---|---|---|
| `--camera N` | 0 | webcam index | `--video PATH` | off | classify frames of a video file instead of the webcam (single pass) |
| `--backend auto\|host\|docker` | auto | where `mobilenet_server` runs |
| `--ir fp16\|fp32` | fp16 | model precision |
| `--device NAME` | MYRIAD | OpenVINO device |
| `--topk N` | 5 | classes to report / draw |
| `--every N` | 1 | classify every Nth frame |
| `--max-fps F` | 0 (off) | throttle the loop |
| `--request-timeout S` | 60 | wait budget per inference (first frame includes the stick boot) |
| `--headless` | off | command-line output only, no window |

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
