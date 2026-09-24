# CPU inference backends across targets (concept)

Status: **Phase 1a (webcam, ORT) and 1b (ssd, full TF) implemented and
committed (2026-09-24); 1c (seg) in progress; 1d (packaging) pending.**
Scope: run every example (ssd-detect, deeplab-seg, webcam) on the host **CPU**
on `amd64`, `arm64`, and (if feasible) `armv7`, in addition to the existing
MYRIAD path.

## 1. Problem

OpenVINO 2020.3's CPU plugin is built on mkl-dnn 0.21.3:

| target | mkl-dnn 0.21.3 state | verdict |
|---|---|---|
| amd64 | full SSE4.2/AVX2/AVX-512 kernels | good CPU engine — **keep** |
| arm64 | no AArch64/NEON kernels; generic C++ only | slow — **replace** |
| armv7 | refuses to build (64-bit only) | no OV CPU at all — **replace** |

Decision (2026-09-24): the **amd64** CPU path stays OpenVINO (`--device CPU`
already works there). The **arm64** CPU path switches to an alternative
inference runtime that has real ARM/NEON kernels.

## 2. Enabler: the CPU backend is a pluggable server

Every example is a Python client + a C++ server joined by a plain
stdin/stdout protocol (no sockets, no shared memory). The client only knows
the *launcher* (`examples/*/infer-*.sh`), which maps
`(backend, device, target)` → server command. **Therefore "a different CPU
runtime" = a different server that speaks the same protocol.** The Python
clients need no changes (only help-text updates).

**Verified per example** (the three protocols are per-example, not shared):

| example | stdin (per frame) | stdout (per frame) |
|---------|-------------------|--------------------|
| ssd-detect | `uint32 w` + `uint32 h` (LE) + `w*h*3` RGB | `FRAME <w> <h> <infer_ms>`; `DET <label> <score> <x1> <y1> <x2> <y2>` ×N; `END` |
| deeplab-seg | `uint32 w` + `uint32 h` (LE) + `w*h*3` RGB | `FRAME <w> <h> <total_ms> <infer_ms>`; `CLASSES <n>`; `CLASS <id> <name> <pixels>` ×n; `MASK <w> <h>` + `w*h` uint16 LE class ids; `END` |
| webcam | `1x3x224x224` float32 tensor (602112 B; client preprocesses) | raw `1000 * 4` fp32 bytes logits; client does top-k itself |

All three are trivially implementable in Python (struct + numpy).

## 3. Candidate runtimes for the arm64 CPU path

| # | runtime | form on the Pi | ARM perf | model format | notes |
|---|---------|----------------|----------|--------------|-------|
| A | **ONNX Runtime** | `pip install onnxruntime` (official manylinux **aarch64** wheels) | XNNPACK + oneDNN EPs, NEON | `.onnx` (export from the TF 1.x frozen `.pb`) | **recommended**; FP32 parity with our FP32 IRs; no build step |
| B | TensorFlow Lite (+ XNNPACK delegate) | `pip install tensorflow` (aarch64 wheel); `.pb → .tflite` via TF1's `TFLiteConverter` | XNNPACK NEON; strong INT8 story | `.tflite` | good alternative; same subgraph caveat as A for the detector |
| C | full TensorFlow (aarch64 wheel, AWS-maintained since 2.10) | `pip install tensorflow` | slowest (no XNNPACK) | runs the frozen `.pb` **as-is** | zero conversion (the `DetectionOutput` op runs natively); useful as fallback |
| D | ncnn | build in image | fastest raw ARM (hand-rolled NEON) | `.param/.bin` via ONNX | no NMS op → hand-rolled post-processing; heaviest integration cost |
| E | OpenVINO generic CPU (already wired for arm64) | in the runtime image | slowest of all here (generic C++) | IR | keep as *secondary* same-stack reference, not the default |

**Recommendation: A (ONNX Runtime).** Reasons:
- official aarch64 PyPI wheels → zero build steps, works on the Pi's 64-bit
  OS *and* inside the arm64 Docker image (`pip install` one line);
- XNNPACK/oneDNN are well-optimized for Cortex-A76;
- FP32 graphs match the semantics of the FP32 IRs we already validate;
- clean escape hatch: C (full TF) can serve as a dev fallback while ONNX
  export is being ironed out.

## 4. Resulting backend matrix — **v1 scope (frozen)**

| target | `--device MYRIAD` | `--device CPU` |
|--------|-------------------|----------------|
| amd64 | OV C++ server, FP16 IR | OV C++ server, FP32 IR (**unchanged**) |
| arm64 | OV C++ server, FP16 IR | **Python CPU server, per example** (below) |
| armv7 | OV C++ server, FP16 IR | **deferred — design only, no v1 code** — see §9 |

The arm64 CPU backend is **per-example, chosen by the launcher** (the model
graphs force this split, see spike log §12):

| example | arm64 CPU runtime | model artifact |
|---------|-------------------|----------------|
| webcam (mobilenet) | **ONNX Runtime** (python) | existing `mobilenetv2-7.onnx` (no conversion) |
| ssd-detect | **full TensorFlow** (python) | existing frozen `frozen_inference_graph.pb` (no conversion) |
| deeplab-seg | **full TensorFlow** (python) | existing frozen `frozen_inference_graph.pb` (no conversion; 2026-09-24 spike: 99.75 % mask agreement vs C++) |

Frozen for v1 (advisor-reviewed 2026-09-24): amd64 keeps the existing
OpenVINO CPU path; arm64 gets python CPU servers (ORT for the classifier,
full TF for the detectors); armv7 gets nothing but documentation.
Everything else (armv7 CPU, INT8, ONNX-backbone detector optimization) is
post-v1.

**Spike evidence** (see §12):
- classification (mobilenet): ORT on the existing `mobilenetv2-7.onnx` vs
  the OV CPU C++ server on the identical 224×224 tensor → logits match to
  the byte (max |Δlogit| = 0.0000), identical top-5; 7.1 ms/infer on amd64.
- detection (ssdlite): the frozen graph's NMS post-processing is inlined as
  TF `While`+`TensorArray` control flow → **not ONNX-convertible as a whole**.
  Running the full frozen graph in TF gives exact parity with the OV C++
  server (same 6 detections, boxes within 2 px, scores within ~0.02).
  TF CPU is slow (≈0.8 s/frame on amd64 TF2) — honest cost of the v1 path;
  ONNX-backbone + Python decode/NMS is the post-v1 speed optimization.

`--device CPU` keeps its meaning "run on the host CPU, best available path
for this target" — the launcher picks the implementation per target; the CLI
does not gain a new device name.

## 5. Model conversion — **none needed for v1**

Spike finding (2026-09-24): v1 reuses the artifacts the prepare scripts
already produce — no new conversion step:

- **mobilenet-v2**: `vendor/models/onnx/mobilenetv2-7.onnx` (ONNX, from the
  ONNX zoo; already downloaded by `prepare-mobilenet.sh`) → ORT directly.
- **ssdlite_mobilenet_v2**: `source/frozen_inference_graph.pb` (TF 1.x frozen
  graph) → full TF directly. Its detection NMS is inlined as TF
  `While`+`TensorArray` control flow, which is **not ONNX-convertible as a
  whole** (verified: no `raw_box_predictions` node; the postprocessor is a
  `BatchMultiClassNonMaxSuppression` while loop with 90 unrolled
  `NonMaxSuppressionV2` ops).
- **deeplabv3**: `source/frozen_inference_graph.pb` → full TF directly
  (graph ends in `ArgMax`, no control flow).

FP32 IRs are kept untouched (still the amd64 OV-CPU path).

**Post-v1 optimization** (performance, not correctness): export only the
SSD *backbone* (up to the six box/class predictor conv outputs) to ONNX via
tf2onnx (TF2 container + `tf.compat.v1.import_graph_def`), and implement
anchor decode + NMS in the Python server — that path is 5–10× faster than
carrying NMS in TF, but adds decode parity risk, so it is explicitly not
v1.

## 6. New Python CPU servers (live in the repo, not in the image build)

Each mirrors its C++ sibling's protocol exactly:

| file | runtime | model input | extra work in server |
|------|---------|-------------|----------------------|
| `examples/webcam/mobilenet_cpu_server.py` | ONNX Runtime | client already sends fp32 224 tensor | none (raw logits out, as `mobilenet_server` does) |
| `examples/ssd-detect/ssd_cpu_server.py` | full TF (frozen `.pb`) | RGB frame → bilinear resize 300×300, uint8 (the TF input is `uint8` `image_tensor`) | none — NMS is in the graph; scale normalized boxes to frame px |
| `examples/deeplab-seg/seg_cpu_server.py` | full TF (frozen `.pb`) | RGB frame → bilinear resize 513×513, uint8 (the TF `ImageTensor` is NHWC **RGB** — verified: feeding BGR mis-segments ~5 % of pixels; the C++ IR path is BGR and its swap is in the IR) | downsample 513×513 label map to frame size (nearest neighbour) |

Dependencies on the Pi: `pip install onnxruntime tensorflow numpy
opencv-python` (aarch64 wheels exist for all; `tensorflow` is the big one).
Correctness check per example: run C++ server (OV CPU, amd64) and Python
server on the same frames and diff the `DET`/top-k output (labels + box
tolerance). **Done for all three** — tracked as reproducible scripts:
`scripts/cpu-parity-webcam.sh`, `scripts/cpu-parity-ssd.sh`,
`scripts/cpu-parity-seg.sh`. Results: webcam exact (top-5 identical,
max |Δlogit| = 2.2e-5); ssd boxes within 2 px, scores within ~0.02;
seg 99.75 % mask agreement (residual = ArgMax tie flips between the two
FP32 backends).

## 7. Backend contract (the seam that must not drift)

The contract any CPU backend server must honor, so backends stay
interchangeable behind the launcher:

1. **Process contract**: long-running child of the launcher; frames on
   stdin, results on stdout, diagnostics on stderr; exits 0 on stdin EOF.
   The per-example protocol is exactly as in §2 (byte-for-byte, including
   endianness and the webcam raw-logits format).
2. **Launcher selection rules** (`examples/*/infer-*.sh`):
   ```
   device=MYRIAD → existing OV server (all targets)
   device=CPU:
     target=amd64 → existing OV server + FP32 IR   (unchanged)
     target=arm64 → python3 <example>_cpu_server.py (host backend);
                    same script inside the arm64 image (docker backend)
     target=armv7 → hard error: "CPU inference is not available on armv7; use MYRIAD"
   ```
3. **Required model artifact per backend**:
   - OV server (MYRIAD/CPU): `openvino/` (FP16) or `openvino_fp32/` (FP32) IR, as today.
   - ORT server (arm64 CPU, webcam): `onnx/mobilenetv2-7.onnx` (already produced).
   - TF server (arm64 CPU, ssd/seg): `source/frozen_inference_graph.pb`
     (already produced) + existing label files.
4. **Where detection post-processing lives**:
   - OV MYRIAD/FP32 paths: NMS is **inside the graph** (MO inlined the TF
     postprocessor; the IR exposes a `DetectionOutput`-style op).
   - TF arm64 path: NMS is **inside the frozen graph** (the inlined
     `BatchMultiClassNonMaxSuppression` while loop); the server is thin.
     Parity requirement: for the same frame, the same `DET` lines must come
     out of both servers (labels + box tolerance) — enforced by the diff
     test in §11 (already proven for ssd: 6/6 detections, ±2 px).
   - seg: no NMS anywhere; the server downsamples the 513×513 class map to
     the frame size (today in `seg_postprocess.hpp`, mirrored in numpy).
   - webcam: no post-processing in the server (raw logits); top-k stays in
     the client.

## 8. Packaging

- **Pi host backend (primary on the Pi):** no image needed — the launcher
  runs the Python server directly with the system `python3` + `pip`
  (one-time `pip install onnxruntime tensorflow numpy opencv-python`; the
  tensorflow aarch64 wheel is the heavy one, ~200 MB).
- **arm64 Docker image (for `--backend docker`):** add
  `pip install onnxruntime tensorflow` to the arm64 stage and `COPY` the
  three server scripts (small; they're pure Python).
- **amd64 / armv7 images:** unchanged (aside from the already-landed
  amd64 OV-CPU work).
- **Already in the working tree:** arm64 `ENABLE_MKL_DNN=ON` + OpenBLAS
  (OV generic CPU on arm64). Proposal: **keep it** as the secondary
  same-stack backend (matrix row E); it costs nothing at runtime and gives
  the apples-to-apples OpenVINO number on the Pi.

## 9. armv7 ("if possible")

- No 32-bit Linux wheels for `onnxruntime`; current `tensorflow` wheels are
  aarch64-only. Older `tflite_runtime` releases had 32-bit wheels (no longer
  published). Building ONNX Runtime or TFLite from source on armv7 is
  possible (both support 32-bit ARM) but adds a real build burden to a
  legacy target.
- **Decision: armv7 CPU is out of scope for the initial implementation.**
  armv7 keeps MYRIAD only; revisit later if demand arises (TFLite from
  source is the likeliest candidate).

## 10. Open decisions

1. ~~Keep arm64 OV generic CPU as the secondary backend?~~ **Decided: yes**
   (already in the tree; it is the benchmark baseline on the Pi and costs
   nothing at runtime).
2. ~~Confirm the per-example arm64 CPU split~~ **Decided (2026-09-24,
   advisor-reviewed): ORT for webcam, full TF for ssd/seg.**
3. Still open: INT8 / ONNX-backbone detector optimization (good Pi5 fit;
   separate work - see Phase 2).

## 11. Implementation phases

**Phase 1a — webcam (DONE, commit `7e5ddbc`)**
1. `examples/webcam/mobilenet_cpu_server.py` (ORT, raw-logits protocol)
2. `infer-server.sh`: `CPU` + arm64 → python server (host backend; the
docker image predates it until 1d); armv7 → hard error
3. README/help updates
4. `scripts/cpu-parity-webcam.sh` (PASS: top-5 identical, max |Δlogit| 2.2e-5)

**Phase 1b — ssd-detect (DONE, commit `845fdb6`)**
1. `examples/ssd-detect/ssd_cpu_server.py` (full TF, `DET` protocol)
2. `infer-ssd-server.sh` branch as in 1a
3. `scripts/cpu-parity-ssd.sh` (PASS: 4/4 detections, boxes ≤1 px, scores ≤0.02)

**Phase 1c — deeplab-seg (spike done; implementing)**
1. spike: full TF on `deeplabv3/source/frozen_inference_graph.pb` vs C++
   `seg_detect` (OV CPU) on dog_ssd.ppm — **done 2026-09-24, 99.75 % mask
   agreement** (see §12)
2. `examples/deeplab-seg/seg_cpu_server.py` (full TF, `MASK` protocol) +
   launcher branch + `scripts/cpu-parity-seg.sh`

**Phase 1d — packaging & validation (pending)**
1. arm64 Dockerfile: `pip install onnxruntime tensorflow` + COPY scripts
   (docker backend only; host backend is the primary Pi path)
2. On-Pi validation: accuracy diff vs C++ server; benchmark MYRIAD vs CPU

**Phase 2 (optional)** — ONNX-backbone SSD detector (5–10× faster than
full-TF NMS), INT8, armv7 CPU from source (TFLite), HETERO:MYRIAD,CPU.

## 12. Spike log / risks / open items

**Done (2026-09-24):**
- Verified all three per-example protocols (ssd: `FRAME`/`DET`/`END` text;
  seg: `FRAME`/`CLASSES`/`CLASS`/`MASK`/`END`; webcam: raw 4000-byte fp32
  logits; clients launch only via `infer-*.sh`).
- **Classifier parity spike**: host tensor (banana.ppm, exact client
  preprocessing) → C++ `mobilenet_server` (OV CPU, FP32 IR) and ORT
  (`mobilenetv2-7.onnx`, CPUExecutionProvider) in a `python:3.11-slim`
  container. Result: **identical logits (max |Δ| = 0.0000), identical top-5**,
  7.1 ms/infer (amd64 host). Note: the ONNX input name is `data`, not `x`.

**Done (2026-09-24, ssd detector spike):**
- Graph probe: input `image_tensor` is **uint8** [?, ?, ?, 3]; outputs
  `detection_boxes/scores/classes/num_detections`; NMS is an inlined
  `BatchMultiClassNonMaxSuppression` **while loop** (TensorArrays + 90
  unrolled `NonMaxSuppressionV2`) — **tf2onnx cannot convert it** (no
  `While`/`TensorArray` support; no `raw_box_predictions` node exists).
- **Full-TF parity**: ran the frozen graph in TF2 (python:3.10 container)
  on dog_ssd.ppm (768×576) → same 6 detections as the C++ OV CPU server,
  same classes, boxes within 2 px, scores within ~0.02. TF CPU: ~0.8 s/frame
  (amd64, TF2 eager — a real cost; the ONNX-backbone v2 path fixes this).

**Done (2026-09-24, deeplab-seg detector spike):**
- Graph probe: `deeplabv3/source/frozen_inference_graph.pb` has **970 nodes,
  no control flow**; input `ImageTensor` is **uint8** NHWC [1,?, ?,3] (the
  graph does the in-graph normalization); output `SemanticPredictions`
  (argmax) [1,513,513].
- **Full-TF parity** (dog_ssd.ppm 768×576, downsample to 513): 5 classes
  present on both sides (background/bicycle/dog/car/cat); per-class pixel
  deltas all ≤0.19 % of the frame; **mask agreement 99.75 %** (the flips are
  ArgMax ties at class boundaries between TF-oneDNN and OV-mkldnn FP32).
  TF CPU: ~172 ms/frame inference (amd64 docker; the 0.8 s SSDLite figure
  above includes the Python NMS decode path).
- **Channel-order gotcha**: the TF `ImageTensor` takes **RGB** — feeding BGR
  (what the C++ IR path sends) mis-segments ~5 % of pixels (dog/cat
  swap). The BGR expectation lives in the IR, where the swap op is baked in;
  the TF server therefore feeds RGB to the graph and mirrors all other
  preprocessing (bilinear 513 resize, raw 0..255, in-graph normalization).

- **glibc**: manylinux wheels need glibc ≥ 2.17 — fine on bullseye (2.31)
  and 64-bit Raspberry Pi OS.
- **Performance is estimated until measured on the Pi**: expect ORT/XNNPACK
  MobileNetV2@224 single-digit ms; full-TF SSDLite@300 and
  DeepLabV3-mnv2@513 are slow (sub-1-fps to ~1 fps class) on Pi5 (honest slow-CPU
  baseline; ONNX-backbone v2 will bring ssd to tens of ms).

## 13. Benchmark plan (the original goal)

On the RPi5, per example:

```
MYRIAD  : infer-*-server.sh docker|host MYRIAD
CPU     : infer-*-server.sh host CPU        (arm64: ORT for webcam,
                                              full TF for ssd/seg)
CPU-OV  : (optional) OV generic CPU inside the arm64 image (same-stack)
```

Report fps + ms/frame for each; the ratio MYRIAD/CPU answers "is the stick
worth it".
