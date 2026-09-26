# SSDLite-MobileNetV2 object detection on the MA2450

Single-image object detection with
[`ssdlite_mobilenet_v2_coco_2018_05_09`](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)
(TensorFlow 1.x Object Detection API, COCO) running on the Intel Movidius NCS1
through this project's self-built **OpenVINO 2020.3.2 MYRIAD** runtime.
Milestone 1: one photo in, a list of detections out, on the command line.

## Model and conversion

`scripts/prepare-ssdlite.sh` downloads the model-zoo release
(`frozen_inference_graph.pb` + `pipeline.config`, sha256-pinned), the pinned
COCO label map, and converts the frozen graph with the vendored 2020.3
Model Optimizer:

```
mo_tf.py --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json \
         --tensorflow_object_detection_api_pipeline_config pipeline.config \
         --data_type FP16
```

in a `python:3.7-slim` container with TensorFlow 1.15 installed (MO 2020.3's
TF front-end requires `tensorflow<2`; 1.x wheels on PyPI exist only up to
python 3.7). The converter runs natively — MO is pure Python and the IR is
architecture-independent.

Outputs:

```
vendor/models/ssdlite_mobilenet_v2/
├── source/            frozen_inference_graph.pb, pipeline.config, tarball
└── openvino/          ssdlite_mobilenet_v2.{xml,bin}   (FP16)
vendor/models/labels/coco.txt          "id<TAB>name" (0=background, sparse COCO ids 1..90)
vendor/models/images/dog_ssd.ppm        known-content test photo (person + dog), P6 PPM
vendor/models/images/sample_640x360.mp4 known-content test clip (Big Buck Bunny, 640x360, ~13 s)
```

## The IR's input/output contract (verified from the XML)

| | layer | shape | precision |
|---|---|---|---|
| input | `image_tensor` | `[1,3,300,300]` NCHW | **FP16** |
| output | `detection_boxes/sink_port_0` | `[1,1,100,7]` | FP16 on MYRIAD |

* **Preprocessing is inside the graph.** The first two ops are
  `Preprocessor/mul` (const `2/255 ≈ 0.007843018`) then `Preprocessor/sub`
  (const `-1.0`), i.e. `x' = x·2/255 − 1` mapping raw 0..255 to [-1, 1].
  The client therefore feeds **raw 0..255 RGB** pixel values; it does *not*
  normalise.
* **Postprocessing is inside the graph too.** The single `DetectionOutput`
  op (opset 1) runs decode + NMS itself and yields `keep_top_k = 100` rows,
  each `[image_id, class, score, xmin, ymin, xmax, ymax]` with
  **normalized 0..1** coordinates (`normalized=1`, `code_type=CENTER_SIZE`,
  `nms_threshold=0.6`, `confidence_threshold=0.3`, `num_classes=91`).
  Unused slots carry `image_id = -1`.

So the client only has to: resize (bilinear, half-pixel sampling — the
reference for the TF resize), convert pixels to FP16, run inference, convert
the output rows back to original-image pixels (the blob's element width is
read from its real byte size, because the MYRIAD TensorDesc can claim FP32
while handing back FP16 — the same trap as in `mobilenet-test`), filter, and
print. All of that post-processing logic lives in `ssd_postprocess.hpp` and
is exercised by `ssd_test` without hardware.

## Files

| file | role |
|---|---|
| `ssd_detect.cpp` | the detector app; single-image mode and `--stdin` frame-stream mode |
| `ssd_postprocess.hpp` | parse/filter/convert of the `DetectionOutput` rows |
| `ssd_test.cpp` | device-free unit tests (run in the Docker build stage) |
| `infer-ssd-server.sh` | starts `ssd_detect --stdin` (host-native or Docker backend) |
| `ssd_stream.py` | webcam/video-file/image client (OpenCV -> stdio -> server), GUI + headless |
| `CMakeLists.txt` | builds both binaries for the selected target |

## Building

`ssd_detect` is built by the same `./build.sh` pipeline as the other demos
(`ssd_test` runs in the build stage and gates the image build). Nothing
extra to run.

## Running

```
# through the wrapper (mounts vendor/models, selects MYRIAD):
./run.sh ssd --image /models/images/dog_ssd.ppm
./run.sh ssd --image /models/images/dog_ssd.ppm --min-conf 0.3 --max-detections 20 --iterations 100

# directly inside the runtime container:
/opt/openvino/bin/ssd_detect --model /models/ssdlite_mobilenet_v2/openvino/ssdlite_mobilenet_v2.xml \
    --weights /models/ssdlite_mobilenet_v2/openvino/ssdlite_mobilenet_v2.bin \
    --labels /models/labels/coco.txt --image /models/images/dog_ssd.ppm \
    --device MYRIAD
```

Options: `--min-conf` (default 0.5, detections kept at `score >= min-conf`),
`--max-detections` (default 10, highest confidence first), `--iterations N`
(latency loop over the same frame), `--debug`.

## Live webcam / video stream (Milestone 2)

`ssd_detect --stdin` keeps the compiled network warm and serves frames over
stdio, driven by the OpenCV client `ssd_stream.py` (same hybrid pattern as
the MobileNet webcam example; the server is started by
`infer-ssd-server.sh`, host-native or Docker).  A different launcher can be
passed as the optional first positional argument; it is invoked as
`<launcher> <backend> <device> <min-conf>` (all clients share this
positional-launcher convention).  In camera mode the capture is drained in a
background thread and only the freshest available frame is classified, so the
window never lags behind a queue of frames accumulated while inference was in
flight:

```
# GUI (default): window with labelled boxes, q/Esc to quit
python3 examples/ssd-detect/ssd_stream.py

# headless: one detection block per frame on stdout
python3 examples/ssd-detect/ssd_stream.py --headless

# no camera? loop an image file (works without OpenCV for .ppm):
python3 examples/ssd-detect/ssd_stream.py --headless \
    --file vendor/models/images/dog_ssd.ppm --frames 5

# or process a video file (single pass, ends at the end of the video):
python3 examples/ssd-detect/ssd_stream.py --headless \
    --video vendor/models/images/sample_640x360.mp4 --frames 30
```

Same pipeline on the host CPU instead of the MA2450 (amd64/arm64 images,
see below):

```
python3 examples/ssd-detect/ssd_stream.py --headless --device CPU \
    --video vendor/models/images/sample_640x360.mp4 --frames 30
```

> `--device` is passed straight to the inference server.  CPU per target
> (see `docs/CPU-BACKENDS.md`):
> * **amd64** - the OV C++ server with the FP32 IRs (`openvino_fp32/`, the
>   2020.3 CPU plugin cannot take FP16 inputs); fast (JIT kernels).
> * **arm64** - the Python **full-TensorFlow** server
>   (`ssd_cpu_server.py`, runs the frozen graph directly; the detection NMS
>   is inlined TF `While`/`TensorArray` control flow that cannot be ONNX-
>   converted, so no model conversion is needed).  Host backend; needs
>   `pip install tensorflow`.
> * **armv7** - not available (no 32-bit CPU runtime); MYRIAD only.
>
> `ssd_detect --list-devices` (no IR) prints the devices actually visible in
> the selected runtime.

`--video` takes any file OpenCV can decode (.mp4/.avi/.mkv/.mov, h264 etc.); frames are
read in order at their native size and the run stops at the end of the video
(`--frames N` cuts the pass short).  A short sample clip is vendored at
`vendor/models/images/sample_640x360.mp4` (Big Buck Bunny, 640×360, ~13 s, 0.6 MB).

Protocol (binary frames, the server's stdout is otherwise protocol-only):

```
client -> server:  uint32 width + uint32 height (little endian) + width*height*3 RGB bytes
server -> client:  "FRAME <w> <h> <infer_ms>"
                   "DET <label> <score> <x1> <y1> <x2> <y2>"   (0..N lines)
                   "END"
```

The client needs `numpy` + `opencv-python(-headless)` (pip; the runtime
image deliberately ships no Python libraries).  Options mirror the single-
image mode (`--min-conf`, `--backend`, `--device`) plus `--camera`,
`--camera-width/-height`, `--camera-fps`, `--camera-fourcc`, `--file`,
`--video`, `--frames`, `--request-timeout`.

As in the webcam example, the capture rate - not inference - often sets the
loop rate: `LatestFrame.read()` consumes its slot, so each iteration blocks
for a brand-new frame, and a webcam left at OpenCV's default is frequently
1280x720 YUYV, which is USB-bandwidth bound at ~9 fps.  Use
`--camera-fourcc MJPG --camera-fps 30` (or a smaller `--camera-width/-height`)
to raise it; `v4l2-ctl --list-formats-ext -d /dev/videoN` lists the real rates.
Note that the `fps` column here times only `client.detect()`, so it excludes
the wait for the camera and can read well above the true end-to-end rate.

## Verified results (MA2450, amd64 image)

Output, one line per detection:

```
person         0.87 [25, 18, 745, 570]
dog            0.71 [300, 120, 700, 575]
```

coordinates are original-image pixels, clamped.

## Tests

`ssd_test` (no hardware, Docker build stage):

* normalized→pixel box conversion with clamping
* confidence threshold keeps exactly-`min-conf` rows (`>=` convention)
* padded rows (`image_id < 0`) and degenerate boxes (`x2<=x1`, `y2<=y1`) dropped
* `max_detections` keeps the highest-confidence tail
* COCO label lookup incl. sparse ids and the `class_<id>` fallback
* shared `half.hpp` sign-bit round trip (regression guard for the 2024 bug)

## Verified results (MA2450, amd64 image)

```
load+read IR    : 15.16 ms
model input     : image_tensor [1,3,300,300] FP32 (raw 0-255 RGB; the 2/255 scale and -1 offset are inside the graph)
model output    : DetectionOutput [1,1,100,7] ...
input image     : /models/images/dog_ssd.ppm 768x576 -> 300x300 in 0.64 ms
compile to MYRIAD : 1673.87 ms
inference       : 92.23 ms mean over 1 run(s) (10.8 fps)
bicycle        0.96 [139, 118, 568, 430]
dog            0.88 [131, 218, 314, 539]
car            0.87 [460, 81, 691, 172]
cat            0.63 [128, 219, 311, 545]
postprocess     : 0.10 ms, 4 detection(s)
```

Cross-check on CPU (same IR, same half-pixel bilinear preprocessing, OpenVINO
2026.4): same four detections, boxes within a couple of pixels, scores
within fp16 noise - the C++ pre/post-processing matches the reference
engine: `bicycle 0.96 (141,119,568,430)`, `car 0.88 (460,81,690,172)`,
`dog 0.84` / `cat 0.70 (132,218,315,539)`.

Unit tests: `ssd_test` 17/17 pass (locally and in the Docker build stage).

Webcam stream (Milestone 2), run inside the runtime container with the
camera and stick passed through (`-v /dev:/dev`), 640x480 capture:

```
warmup: first inference 1520 ms (includes device compile); excluded from fps
[t=   1.77s fps= warm infer=  91.2ms] (no detections at confidence >= 0.50)
[t=   1.87s fps= 10.8 infer=  91.1ms] (no detections at confidence >= 0.50)
...
[t=  44.90s fps=  9.2 infer=  91.3ms] bed 0.87 (230,171,631,431)
45 s run ended at t=44.9 s: steady-state ~9.2 fps, inference 91 ms/frame,
detection stable (score 0.86-0.88, box jitter < 10 px over the run)
```

The first frame is labelled `fps= warm`: the first request carries the
one-time MYRIAD compile, so its wall time is reported on the `warmup:` line
and kept out of the fps average.  From the second frame on, `fps` is the
mean rate over the last 10 request round-trips, so it reads the steady rate
immediately instead of ramping up out of a compile-dominated first sample.

File-loop mode (no camera) on `dog_ssd.ppm`: same detections as the
single-image run, 92 ms/frame, clean EOF shutdown.
