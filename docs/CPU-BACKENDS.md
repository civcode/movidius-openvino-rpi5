# CPU inference backends across targets (concept)

Status: **Phases 1a-1d implemented (2026-09-24); on-Pi validation +
MYRIAD-vs-CPU benchmark pending (needs the RPi5).**
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
| E | ~~OpenVINO generic CPU~~ | ~~in the runtime image~~ | ~~generic C++~~ | IR | **removed 2026-09-25**: cannot run inference on arm64 (no AArch64 kernels in mkl-dnn 0.21.3; see §12) |

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
- **arm64 OV generic CPU (matrix row E): removed 2026-09-25.** The arm64
  image no longer sets `ENABLE_MKL_DNN` (and drops OpenBLAS). The first
  live arm64 test proved the C++ OV CPU path cannot run inference there
  (mkl-dnn 0.21.3 has no AArch64 kernels; full root cause in §12). The
  Python servers remain the arm64 CPU path.

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

1. ~~Keep arm64 OV generic CPU as the secondary backend?~~ **Decided yes
   (2026-09-24), reversed 2026-09-25**: the first live arm64 run showed the
   C++ OV CPU path cannot run inference there (mkl-dnn 0.21.3 has no
   AArch64 kernels; see §12). The arm64 image drops the CPU plugin; the
   Python servers are the arm64 CPU path and MYRIAD the fast path.
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

**Phase 1c — deeplab-seg (DONE, commit `9219df7`)**
1. spike: full TF on `deeplabv3/source/frozen_inference_graph.pb` vs C++
   `seg_detect` (OV CPU) on dog_ssd.ppm — **done 2026-09-24, 99.75 % mask
   agreement** (see §12)
2. `examples/deeplab-seg/seg_cpu_server.py` (full TF, `MASK` protocol) +
   launcher branch + `scripts/cpu-parity-seg.sh`

**Phase 1d — packaging (DONE) & validation (on-Pi, pending)**
1. arm64 Dockerfile: the runtime stage installs `python3-pip` +
   `tensorflow onnxruntime opencv-python-headless` (arm64 only) and copies
   the three Python servers to `/opt/openvino-demo/cpu-servers/`; the three
   launchers gained a docker branch for `CPU` + arm64 that runs the
   in-image copy. QEMU-verified 2026-09-24: the pinned bullseye snapshot
   supplies the deps (tf 2.20, ort 1.19.2, cv2 5.0, numpy 2.0) and all three
   servers produce valid protocol output under aarch64 emulation.
2. On-Pi validation (**done 2026-09-25**, arm64 image rebuilt without the
   OV CPU plugin - see §12): in-image ORT server on the reference 224x224
   tensor: 43.2 ms/infer, logits byte-identical to the zoo reference
   (max |diff| = 0.0000), top-5 556/818/811/827/918 - identical to the
   MYRIAD baseline; in-image TF servers on dog.ppm: SSD 2775 ms/frame (3
   detections: cat/car/bicycle), seg 1073 ms/frame + 1071 ms postprocess
   (5 classes - same set as the amd64 spike). MYRIAD self-test on the same
   image: RESULT PASS (max |diff| 0.0525, top-5 identical). The C++ OV CPU
   server comparison is amd64-only now (the plugin no longer ships on
   arm64).

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

**Done (2026-09-25, first live arm64 OV CPU test - failure root cause):**
- First ever live C++ OV CPU inference on the Pi 5 (arm64 image, 4-core
  build): `mobilenet_server --device CPU` on the MobileNet v2 FP32 IR
  fails with `Supported primitive descriptors list is empty for node:
  ...linearbottleneck0_batchnorm0_fwd/variance/Fused_Add_` (throw site
  `mkldnn_node.cpp:306`).
- Root cause (GDB object dumps + source + bisection): the MKLDNN graph
  optimizer fuses `1x1 conv + bias Add + ReLU` and, in this model, merges
  the following depthwise-conv block into the same `MKLDNNConvolutionNode`
  (`fusedWith = [ReLU, next Convolution]`). That merge appends a custom
  mkl-dnn `dw_conv` post-op to the convolution; the `dw_conv` post-op is
  implemented only in the x86 JIT conv kernels (SSE4.2/AVX2). On AArch64
  only the generic GEMM/reference convs exist, and conv primitive-desc
  creation with that post-op combo yields zero descriptors.
- Bisection: the identical conv block works (4 descriptors) when the
  dwconv block is not merged (sub-model layers 0-13) and fails (0
  descriptors) when it is (layers 0-14) - so every MobileNet-family graph
  hits this, not just this IR.
- **Consequence:** the arm64 image no longer ships the OV CPU plugin
  (`ENABLE_MKL_DNN=OFF`, no OpenBLAS); `--device CPU` on arm64 routes to
  the Python servers (§4), and `list` shows MYRIAD only.

- **glibc**: manylinux wheels need glibc ≥ 2.17 — fine on bullseye (2.31)
  and 64-bit Raspberry Pi OS.
- **Performance is estimated until measured on the Pi**: expect ORT/XNNPACK
  MobileNetV2@224 single-digit ms; full-TF SSDLite@300 and
  DeepLabV3-mnv2@513 are slow (sub-1-fps to ~1 fps class) on Pi5 (honest slow-CPU
  baseline; ONNX-backbone v2 will bring ssd to tens of ms).

- **2026-09-25 — GUI examples validated on CPU (arm64, in-image servers):**
  all three examples ran on the Pi 5 with `--device CPU --backend docker`,
  OpenCV windows on the local X display: webcam (live camera + sample video)
  at 11.7 fps (camera) / 20.9 fps (video) with ~35-42 ms infer; ssd-detect on
  dog.ppm at 12.4 fps, ~77-85 ms infer, cat 0.81 / car 0.77 / bicycle 0.72
  (identical to the direct-server run); deeplab-seg on dog.ppm at ~700 ms
  infer, 1.4 fps, 5 classes (identical).

- **Bug found and fixed in that run - launcher protocol pollution:** the CPU
  docker branches of all three `infer-*.sh` launchers were missing
  `-e OV_QUIET=1`, so the `container-entry.sh` banner ("OV target : arm64")
  leaked onto the protocol stdout.  The text-protocol clients (ssd, seg)
  aborted on the first unexpected line; the binary webcam client silently
  consumed the banner bytes and returned shifted (garbage) logits.  Added
  `-e OV_QUIET=1` to all three CPU `docker run` calls.

**Done (2026-09-25, SSD/Deeplab IR unblocked natively on arm64):**
- The historical `tensorflow==1.15` pin in the prepare scripts was never
  actually required: MO 2020.3's TF front end loads frozen graphs through
  `tensorflow.compat.v1`, which TF 2.x still provides. `prepare-ssdlite.sh`
  and `prepare-deeplabv3.sh` now run Model Optimizer 2020.3 **natively on
  arm64** (no Docker, no emulation) from `work/venv-cpu` (TF 2.21.0,
  numpy 2.5.3) via `scripts/mo_compat_run.py`, which restores the removed
  `np.float`/`np.int`/`np.bool`/`np.str`/`np.object`/`np.unicode` aliases and
  self-patches the two staged-tree call sites Python 3.13 / ElementTree no
  longer accept (`exec("import ...")` in `versions_checker.py`,
  `Element.getchildren()` in `ie_ir_ver_2/emitter.py`).
- Conversion on the Pi: SSDLite FP16/FP32 ~28 s each, DeepLabV3 ~13 s each.
  All four IRs land in `vendor/models/*/openvino/` and `./run.sh ssd|seg`
  (docker and host MYRIAD) work unchanged - the IR is
  architecture-independent.

**Done (2026-09-25, host CPU backends + full 24-cell matrix):**
- The three launchers now fall back to `work/venv-cpu/bin/python` (TF 2.21 +
  ORT 1.30 + cv2 5.0.0) when the system python3 lacks onnxruntime/tensorflow,
  so the host CPU cells work without any package-manager changes.
  Measured arm64 host: webcam ORT ~40-43 ms/frame (24 fps); SSD full-TF
  ~2.9 s/frame (3 detections, same as docker: cat 0.81 / car 0.77 / bicycle
  0.72); seg full-TF ~1.07 s/frame + ~1.07 s post (5 classes). One TF 2.21 /
  numpy 2 compatibility fix in `ssd_cpu_server.py`: `num_detections` is a
  rank-1 tensor, so `int(num_dets)` needed a `[0]` fallback.
- `device_probe.hpp`: the MYRIAD probe window grew from 3 s to 12 s
  (11 x 1 s) so a cell can start while the stick is still re-enumerating
  after the previous cell's shutdown.
- Client shutdown (`mobilenet_client.py`): `stop_server` now closes the
  server's stdin first (the protocol's EOF terminator) and waits 5 s for a
  clean exit before escalating to `docker stop -t 0` / SIGTERM / SIGKILL.
  The MYRIAD servers exit 0 on EOF and release the VPU with a clean
  mvnc/XLink deinit - a hard kill orphaned the stick's XLink session and
  left it slow to re-enumerate, which is why back-to-back MYRIAD cells in
  the matrix intermittently saw "MYRIAD not available".
- `scripts/test-examples.sh` 24-cell matrix (3 examples x docker|host x
  MYRIAD|CPU x headless|gui) after all of the above: **24 PASS / 0 SKIP /
  0 FAIL** (`logs/rpi5-arm64/matrix-2026-09-25.log`).

## 13. Benchmark plan (the original goal)

On the RPi5, per example:

```
MYRIAD  : infer-*-server.sh docker|host MYRIAD
CPU     : infer-*-server.sh docker CPU      (arm64: ORT for webcam,
                                              full TF for ssd/seg)
```

Report fps + ms/frame for each; the ratio MYRIAD/CPU answers "is the stick
worth it".

## 14. CPU thread placement: why the CPU backend would not scale past ~190 fps

Investigated 2026-09-26 on the amd64 host (32 logical cores), MobileNet v2 FP32
IR, `mobilenet_server --device CPU`.

**Symptom.** `--bench` throughput sat at ~180-190 inferences/s no matter how the
client was arranged: queue depth 1/2/4/8 made no difference, and neither did
running 4 separate server processes - latency grew in proportion to the server
count while the aggregate rate stayed flat.  Whole-tree CPU was ~1.05 cores, so
the machine looked 97 % idle and the inference path looked under-loaded.

**What it was not.**  Each of these was measured and rejected: the capture rate
(bench mode uses no camera), the client (0.03 cores, and `tobytes()`-free
`os.write()` changed nothing), the GIL (`sys.setswitchinterval()` 5 ms → 50 µs:
187.9 / 188.4 / 188.3 fps), a cgroup CPU limit (`cpu.max` = `max 100000`, i.e.
none), disk (server `read_bytes` = 0, `majflt` = 0), and process affinity
(`taskset` reported 0-31).  Control: four plain CPU-burn processes did reach
4.00 cores, so the box was not the limit.

**Cause.**  The OpenVINO 2020.3 CPU plugin binds its inference worker thread per
`CPU_BIND_THREAD`, default `YES` (`inference-engine/include/ie_plugin_config.hpp`:
`YES` pins threads to cores, "best for static benchmarks"; `NO` disables it).
The binding is **per thread**, invisible in the process mask, and every process
picks the *same* core - `/proc/<pid>/task/<tid>/status` showed the plugin's
worker with `Cpus_allowed_list: 0` in all four servers.  So N servers queue on
CPU 0 and the workload is capped at one core's worth of inference.

**Fix.**  `cpu_threading.hpp` (shared, like `device_probe.hpp`) sets
`CPU_BIND_THREAD=NO` for CPU devices in `mobilenet_server`, `ssd_detect` and
`seg_detect`; each logs what it applied (`mobilenet_server: CPU_BIND_THREAD=NO`).
`mobilenet_server` also takes `--bind-thread yes|no|numa`, `--streams N|auto`
and `--threads N`.  Verified with `scripts/cpu-parity-webcam.sh`: PASS, max
|Δlogit| 0.000022, identical top-5.

**Measured - and then partly corrected.**  Lifting the pin was necessary but not
sufficient: with the pin gone, *one* client process still cannot push more than
about one server's worth of requests, whatever `--servers` says.

| `--servers` | one client process, pooled threads | one client process per server |
|---|---|---|
| 1 | ~146 fps | ~146 fps |
| 2 | ~155 fps | ~277 fps (139 each) |
| 4 | ~152 fps | ~521 fps (130 each) |
| 8 | ~137 fps | ~950 fps (119 each) |
| 12 | - | ~1235 fps (103 each) |

The flat second column is the GIL again: each request costs this interpreter
queue handoffs, response parsing and syscall setup, so those costs serialise no
matter which worker thread pays them.  The measurement path therefore runs one
client process per server (`bench_processes()` in `examples/mobilenet_client.py`)
- the third column.  A live `--servers N` pool still helps up to roughly one
server's rate, and the tool says so when it starts.

So the "~190 fps regardless of servers" reading had two independent causes, the
first hiding the second: `CPU_BIND_THREAD=YES` put every server on one core, and
the client process caps the request rate at about one server's throughput.  Only
the first is fixed in the server; the second is a property of a
single-Python-client design, and an earlier version of the table above claimed
~565 fps from one client process with four servers, which a repeat run did not
reproduce.

**Caveats worth remembering.**

* One lone server is *slower* unpinned (~181 → ~146 fps): its thread loses core
  affinity and cache locality.  The fix pays off only when several inferences
  run at once, which is the point of `--servers`.
* Unpinning does not make a single inference use several cores, and neither does
  queue depth: measured at depth 1/2/4/8, one server stayed within noise of
  ~180 fps.  `--depth` was removed in favour of `--servers`; intra-request
  parallelism would need `CPU_THROUGHPUT_STREAMS` plus several `InferRequest`s
  per server.
* A **live** run is client-bound: on the video path it caps near ~118 fps
  whatever `--servers` says, because decode plus preprocessing happens in the
  client's single Python thread.  `--fake-camera` takes the camera out of that
  path and `--bench` takes the drawing out; `--bench` with `--servers` uses one
  client process per server.
* The arm64 CPU servers (`mobilenet_cpu_server.py` ONNX Runtime, the TF servers)
  were not touched - their threading knobs are different and they were not
  measurable here.
* Numbers recorded earlier in this document were taken with the pin in place, so
  treat any CPU-throughput figure predating this section as measured under a
  one-core ceiling.
