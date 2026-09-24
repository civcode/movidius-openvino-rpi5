# DeepLabV3 (Pascal VOC) segmentation on the MA2450

Semantic segmentation of camera frames with the OpenVINO Model Zoo
**DeepLabV3 (MobileNetV2, Pascal VOC 21 classes, 513×513)** on the
Movidius MYRIAD (MA2450) using the project's self-built OpenVINO 2020.3.2
runtime.

Two pieces:

* `seg_detect` — C++ inference server (2020.3 `InferenceEngine` API).
  Single-image mode prints a per-class pixel histogram; `--stdin` mode
  runs a binary frame protocol for streaming.
* `seg_stream.py` — Python/OpenCV client (GUI or `--headless`), plus
  `--file` single-image and `--video` video-file modes for deterministic tests.

## IR contract (verified, `scripts/prepare-deeplabv3.sh` step 5)

The converted IR is the runtime contract — do not assume it:

| | |
|---|---|
| input | `ImageTensor` **NCHW [1,3,513,513] FP16**, **BGR**, raw **0..255** |
| baked in | BGR→RGB channel swap, in-graph bilinear resize to 513×513 (identity at the fixed input size), normalization **(x/127.5) − 1** |
| output | `ArgMax/sink_port_0` **[1,513,513]** per-pixel class ids 0..20 |

The source TF graph does `x*(1/127.5) - 1` (`mul_1/x = 0.0078431377...`,
`sub_7/y = 1.0`); MO bakes that into the IR together with the channel
reversal (`--reverse_input_channels`) and the resize. The client therefore
feeds raw 0..255 BGR and does its own 513×513 resize.

Verified empirically (OpenVINO 2026.4 CPU probe on the same IR): raw
0..255 BGR gives a sane multi-class histogram, while any pre-normalised
input (x/127.5−1, x/128−1, x/255−1) collapses to 100 % background.

> Note: the `.bin` stores f16 constants **little-endian** (the XML's
> inline `value=` hex is big-endian).  When inspecting constants, decode
> accordingly — a wrong endianness read made the scale look like 6.3e-5
> instead of 1/127.5.

## Build / prepare

```
./scripts/prepare-deeplabv3.sh      # OMZ TF graph -> FP16 IR via vendored MO
./build.sh                          # Docker image (seg_test runs in the build stage)
```

## Run

Single image (console histogram, `--mask-out` writes a class-map PPM):

```
./run.sh seg --image /models/images/dog_ssd.ppm
# or inside the image / host runtime:
seg_detect --model ... --weights ... --labels ... --image img.ppm [--device MYRIAD] [--mask-out mask.ppm]
```

Streaming (webcam or video):

```
python3 examples/deeplab-seg/seg_stream.py --headless          # terminal stats
python3 examples/deeplab-seg/seg_stream.py                     # GUI overlay
python3 examples/deeplab-seg/seg_stream.py --file img.ppm      # single image, no camera
python3 examples/deeplab-seg/seg_stream.py --video vendor/models/images/sample_640x360.mp4
python3 examples/deeplab-seg/seg_stream.py --mask-out mask.ppm # also save the class map

# same pipeline on the host CPU (amd64/arm64 images; arm64 = generic C++ path):
python3 examples/deeplab-seg/seg_stream.py --headless --device CPU \
    --video vendor/models/images/sample_640x360.mp4 --frames 10
# drive the CPU server directly (per-target runtime, see above):
examples/deeplab-seg/infer-seg-server.sh host CPU < /tmp/frame.bin
```

`--video` processes a video file (any container/codec OpenCV can decode) instead
of the webcam: frames are read in order at their native size, in a single pass,
ending at the end of the video (`--frames N` cuts the pass short).  A short
sample clip is vendored at `vendor/models/images/sample_640x360.mp4`
(Big Buck Bunny, 640×360, ~13 s, 0.6 MB).

`--file` keeps the image at its native resolution (the C++ server resizes
internally).  `seg_stream.py` starts the server via `infer-seg-server.sh`
(host/docker/auto backend, same pattern as the SSDLite launcher);
pass a different launcher as the first positional argument.

`--device CPU` picks the best CPU path per target (see
`docs/CPU-BACKENDS.md`):

* **amd64**: the C++ OpenVINO server with the FP32 IRs
  (`openvino_fp32/`, produced by `scripts/prepare-deeplabv3.sh`) - the 2020.3
  CPU plugin rejects FP16 inputs.
* **arm64**: `seg_cpu_server.py` - the TF 1.x frozen graph
  (`source/frozen_inference_graph.pb`) run by full TensorFlow 2.x.  No model
  conversion; the same `FRAME/CLASSES/CLASS/MASK/END` protocol, same
  preprocessing (513×513 bilinear resize, raw 0..255, in-graph normalization)
  and same post-processing (nearest-neighbour mask resize, per-class
  histogram) as the C++ server.  `pip install tensorflow`
  (+ `opencv-python-headless` for the resize) is the only dependency.
  The docker backend runs the in-image copy of the server (the image must
  have been rebuilt with the cpu-servers step - see the Dockerfile).
* **armv7**: not available (no 32-bit CPU runtime) - MYRIAD only.

CPU parity against the C++ server is a tracked regression test:
`./scripts/cpu-parity-seg.sh` (same frame through both servers; on the
dog_ssd.ppm reference frame it passes with 99.75 % mask agreement - the
residual flips are ArgMax ties between the two FP32 backends).

## Streaming protocol (`seg_detect --stdin`)

```
in : repeated frames, each = uint32 width + uint32 height (little endian)
     + width*height*3 RGB bytes
out: per frame:
     FRAME  <w> <h> <total_ms> <infer_ms>
     CLASSES <n>
     CLASS  <id> <name> <pixels>         (n lines, pixel count desc)
     MASK   <w> <h>  + width*height uint16 LE class ids
     END
```

Startup diagnostics go to stderr; stdout carries only the protocol.

## Postprocessing rules

* The IR declares the output as **I32** class ids, but a backend may hand
  back I16/I32 or an FP16/FP32 copy of the same values.  The blob is
  decoded byte-wise and the first interpretation that yields plausible
  ids (integer values 0..20) wins — this matters because an int32 id 12
  re-read as a float is an 8.8e-44 denormal that silently rounds to class
  0 (the "100 % background" bug, caught by `seg_test`).
* The output is interpreted from its shape: `H*W` elements → direct class
  ids; `C*H*W` with `C == numClasses` → logits, argmax over classes.
  Anything else is a hard error (the layout is never guessed).
* The 513×513 mask is resized back to the frame size with
  **nearest-neighbour** sampling only — bilinear averaging would blend
  class boundaries.
* `seg_test` (built in the Docker `seg` stage, no hardware needed) covers:
  half roundtrip, direct ids, logits argmax, raw blob decoding
  (I32/FP32/I16/FP16), nearest resize, labels, out-of-range ids,
  unsupported shapes.

## Visualization

Class-map PPM (`--mask-out`): single-channel P6 (maxval 255), 1 byte/pixel,
value = class id 0..20.
The GUI client draws a fixed 21-colour Pascal VOC palette over the frame
(alpha 0.4); class statistics go to stderr/overlay text.  The window is
created as `WINDOW_NORMAL`, so it can be dragged to any size (the overlay is
scaled to fit); `--window-size WxH` (e.g. `960x540`) sets the initial size
- without it the window opens at the frame's dimensions.  `q`/Esc or
closing the window stops the run.  The mask file
and the overlay are separate artifacts — the mask is never drawn through
the palette.

## Timing

Single-image mode prints preprocess / inference / postprocess / total
milliseconds; stream mode reports them per frame on the `FRAME` line.

## Verified results (MA2450)

Single image, `dog_ssd.ppm` (768×576, contains a dog, cat, bicycle and car):

```
load+read IR    :  7.0 ms
compile to MYRIAD: 1726 ms
preprocess      :  4.0 ms     inference: 674 ms     postprocess: 0.7 ms

classes present (768x576 mask):
     0  background  344027 (77.8%)
     2  bicycle      49629 (11.2%)
    12  dog          33342 ( 7.5%)
     7  car          14173 ( 3.2%)
     8  cat           1197 ( 0.3%)
```

CPU cross-check (OpenVINO 2026.4, same IR, same raw-0..255 BGR input):
background 78.7%, bicycle 10.5%, dog 5.6%, car 3.0%, cat 2.2% — same
classes, same ordering, consistent with the MYRIAD run.

Streaming (webcam 640×480, `seg_stream.py --headless`):
~1.5 fps, server 676-679 ms/frame (infer ~674 ms); live classes
(background/dog/cat) track the camera scene sensibly.  As in `ssd_stream.py`,
the first frame prints `fps= warm` and the compile-inflated round-trip goes
to the `warmup:` line instead of the fps average.

The 513×513 DeepLabV3 mask is ~5× the SSDLite 300×300 cost, so ~1.5 fps
on the MA2450 is the expected regime (SSDLite: ~10 fps).
