# OpenVINO 2020.3.2 + Movidius MA2450 on Linux (ARMv7, ARM64, x86_64)

This project builds the OpenVINO 2020.3.2 MYRIAD stack from source and runs an
Intel Movidius MA2450 / Neural Compute Stick through OpenVINO's own
`libmyriadPlugin.so`, MVNC/XLink transport, libusb and `usb-ma2450.mvcmd`
firmware.

The same repository and Dockerfile support three host targets:

| Project target | Docker platform | Intended host | OpenVINO host binaries |
|---|---|---|---|
| `armv7` | `linux/arm/v7` | Raspberry Pi / Pi 5 compatibility fallback | ARMHF / ELF32 ARM |
| `arm64` | `linux/arm64` | Raspberry Pi 5 / other AArch64 Linux | native ELF64 AArch64 |
| `amd64` | `linux/amd64` | Ubuntu or other x86_64 Linux | native ELF64 x86-64 |

The preferred Pi 5 path is now native ARM64. The original ARMv7 userspace path is retained as a known-good compatibility fallback. ARM64 and amd64 are native builds and do **not** use the ARMv7 toolchain.

OpenVINO is pinned to `2020.3.2`, commit
`0b3773b7405d955d48642667ac5113289b9baab2`. This is the known-good revision for
this project and MA2450 firmware. The port does not introduce MA2Host into the
normal runtime path.

## Runtime architecture

```text
hello_myriad / mobilenet_classify
        |
        v
OpenVINO Inference Engine 2020.3.2
        |
        v
libmyriadPlugin.so            <- built from source for armv7, arm64 or amd64
        |
        v
MVNC / XLink / libusb
        |
        +---- uploads usb-ma2450.mvcmd
        |
        v
Movidius MA2450
```

The `.mvcmd` file is device firmware, not ARM host code. All host targets use
the same pinned firmware payload sourced from Intel's 2020.3.2 Raspbian runtime
package, while `libmyriadPlugin.so` and the applications are built natively for
the selected host target.

Every target also ships the Hetero and Multi-Device plugins, plus the CPU
plugin on **amd64 and arm64** (the pinned mkl-dnn 0.21.3 has full x86
kernels; on AArch64 it predates NEON kernels, so arm64 uses the slow generic
C++ path - fine as a CPU baseline, e.g. to benchmark against the MYRIAD
stick, but not a fast inference engine; armv7 cannot build it at all,
mkl-dnn 0.21.3 refuses 32-bit targets).  amd64/arm64 images therefore
support `--device CPU` in the examples.

## Quick start

Platform selection is centralized in `scripts/platform.sh`. If `--platform` is omitted, `x86_64` maps to `amd64`, `aarch64`/`arm64` maps to native `arm64`, and `armv7l` maps to `armv7`.

### Raspberry Pi 5 — preferred native ARM64

```bash
./build.sh --platform arm64
./run.sh --platform arm64 list
./run.sh --platform arm64
```

Legacy ARMv7 fallback:

```bash
./build.sh --platform armv7
./run.sh --platform armv7 list
```

### Ubuntu / x86_64

```bash
./build.sh --platform amd64
./run.sh --platform amd64 list
./run.sh --platform amd64
```

You can inspect the resolved configuration without cloning OpenVINO or invoking
Docker:

```bash
./build.sh --platform armv7 --print-platform
./build.sh --platform arm64 --print-platform
./build.sh --platform amd64 --print-platform
```

Default image tags are:

```text
openvino-2020.3-movidius-armv7:latest
openvino-2020.3-movidius-arm64:latest
openvino-2020.3-movidius-amd64:latest
```

Override them with `IMAGE=...` or `--image ...`.

For environment-based platform selection, prefer `OV_PLATFORM=armv7|arm64|amd64`.
A legacy `TARGET=<valid-project-target>` value is still accepted, but unrelated
`TARGET` values from CI/container environments are ignored so auto-detection remains reliable.

## Build behavior

`build.sh` automatically prepares the pinned source/firmware dependencies when
needed, validates the patch set and invokes one architecture-neutral Dockerfile.
Important properties of the multi-platform build are:

- `armv7` uses `toolchain/armv7-native.toolchain.cmake`.
- `arm64` does not pass a toolchain and configures natively as AArch64.
- `amd64` does not pass a toolchain and configures natively as x86_64.
- MYRIAD options are kept the same on all targets (`ENABLE_VPU`,
  `ENABLE_MYRIAD`, firmware boot enabled).
- BuildKit caches are target-qualified, so ARMv7, ARM64 and x86 CMake trees cannot contaminate each other.
- The installed OpenVINO `lib/<arch>` directory is discovered rather than
  hard-coded as `armv7l`, `aarch64`, or `intel64`.
- The Docker build checks the ELF class/machine of `hello_myriad` and
  `mobilenet_classify` against the selected target.
- `libmyriadPlugin.so`, `plugins.xml` and `.mvcmd` firmware are validated in the
  same installed library directory.
- The final image records `/opt/openvino/runtime-manifest.env` and
  `/opt/openvino/firmware.sha256`.
- ARM64 validation expects ELF64 with ELF machine ID 183 (`AArch64`) for both the demo binary and `libmyriadPlugin.so`.

Useful build variants:

```bash
./build.sh --platform arm64 --target configure
./build.sh --platform arm64 --target builder
./build.sh --platform amd64 --target configure
./build.sh --platform amd64 --target builder
BUILD_JOBS=4 ./build.sh --platform amd64
./build.sh --platform armv7 --no-cache
./build.sh --platform armv7 --no-patches
```

`--no-patches` is diagnostic only. See `patches/README.md`.

## USB access and firmware re-enumeration

The default runtime uses the same constrained Docker USB access strategy on all platforms:

```text
--network=host
-v /dev:/dev
--device-cgroup-rule='c 189:* rwm'
```

The stick changes USB state when OpenVINO uploads firmware, so the container
must see device nodes created after startup. The project intentionally does not
use `--privileged` by default.

The original Pi measurements showed that host networking was required for the
libusb/XLink re-enumeration path on that host. The native ARM64 and amd64 paths conservatively
keep the same option until hardware validation proves it can be narrowed on a
specific system.

Check the stick with:

```bash
lsusb | grep -i -E '03e7|myriad'
```

### udev permissions

The rule in `scripts/99-movidius.rules` is CPU-architecture independent.
On a Linux host:

```bash
sudo cp scripts/99-movidius.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo usermod -aG plugdev "$USER"
```

Log out/in after changing group membership. Docker daemon access and USB udev
permissions are separate concerns.

## Runtime modes

The same command shapes are available for all targets:

```bash
./run.sh --platform amd64               # tiny-model demo
./run.sh --platform amd64 list          # MYRIAD enumeration only
./run.sh --platform amd64 bench --iterations 20
./run.sh --platform amd64 shell
./run.sh --platform amd64 custom --model /work/model.xml --weights /work/model.bin
./run.sh --platform amd64 mobilenet
```

Use `arm64` on a Pi 5 by default; use `armv7` only for the compatibility fallback. Global options (`--platform`, `--image`,
`--verbose`) go before the mode.

## MobileNet v2

Prepare the shared model corpus once:

```bash
./scripts/prepare-mobilenet.sh
```

Model Optimizer is Python-only and runs in a native container matching the host
CPU (`linux/arm64` on a Pi 5, `linux/amd64` on x86_64). That conversion step is
independent of the inference target.

Then run:

```bash
./run.sh --platform amd64 mobilenet
./run.sh --platform amd64 mobilenet --image /models/images/cat.ppm
IR=fp32 ./run.sh --platform amd64 mobilenet
```

See `HOWTO-MOBILENET.md` for details.

### Live webcam classification

`examples/webcam/` runs the same model from a webcam in real time: a Python
frontend (OpenCV capture, GUI or `--headless` CLI output) drives a small
long-running C++ inference server that keeps the stick's compiled network
warm.  See `examples/webcam/README.md`.

### Object detection (SSDLite-MobileNetV2, COCO)

`examples/ssd-detect/` runs the COCO-pretrained SSDLite detector
(`ssdlite_mobilenet_v2_coco_2018_05_09`, FP16 IR from
`scripts/prepare-ssdlite.sh`) on the stick: one photo in, a list of labelled
bounding boxes out.

```bash
./run.sh ssd --image /models/images/dog_ssd.ppm
```

Verified on MA2450: 4 detections on `dog_ssd.ppm` (bicycle 0.96, dog 0.88,
car 0.87, cat 0.63), inference 92 ms (~11 fps), matching a CPU reference run
of the same IR (boxes within a couple of pixels, scores within fp16 noise).
`ssd_test` (17 device-free checks) passes in the Docker build stage.

Live streaming is also supported: `ssd_detect --stdin` serves binary RGB
frames over stdio, and `examples/ssd-detect/ssd_stream.py` (OpenCV, GUI or
headless) drives it with labelled boxes - ~9 fps on a real webcam on the
stick.  See `examples/ssd-detect/README.md`.

### Semantic segmentation (DeepLabV3, Pascal VOC 21 classes)

`examples/deeplab-seg/` runs the DeepLabV3 (MobileNetV2, 513×513) semantic
segmenter (`deeplabv3_mnv2_pascal_train_aug_2018_01_29`, FP16 IR from
`scripts/prepare-deeplabv3.sh`) on the stick: one photo in, a per-pixel
class map out, resized back to the original size with nearest-neighbour
sampling.

```bash
./run.sh seg --image /models/images/dog_ssd.ppm
```

Verified on the MA2450: compile 1726 ms, inference 674 ms (total 679 ms
for a 768×576 photo); the dog_ssd.ppm mask comes out as background 77.8%,
bicycle 11.2%, dog 7.5%, car 3.2%, cat 0.3% - matching a CPU cross-check
on the same IR.  Webcam streaming runs at ~1.5 fps (the 513×513 mask is
~5× the SSDLite cost).

The IR contract (inspected, not assumed): input `ImageTensor` NCHW
[1,3,513,513] FP16 BGR raw 0..255 - the BGR→RGB swap, the resize and the
(x/127.5)−1 normalization are all baked into the graph; output `ArgMax`
[1,513,513] per-pixel class ids 0..20, declared **I32** - the blob is
decoded byte-wise (I32/FP32/I16/FP16) and values verified, because an
int32 id re-read as a float is a denormal that silently rounds to class
0.  `seg_test` (34 device-free checks) passes in the Docker build stage.  Streaming works the same way as the
detector: `seg_detect --stdin` + `examples/deeplab-seg/seg_stream.py`
(GUI overlay / `--headless` / `--file`).  See
`examples/deeplab-seg/README.md`.

## Host-native execution without Docker at inference time

Docker remains the primary and most deterministic runtime, but the built tree
can be extracted for direct host execution.

```bash
./scripts/pull-runtime.sh --platform amd64
./scripts/host-run.sh --platform amd64 list
./scripts/host-run.sh --platform amd64 mobilenet
```

On `arm64` and `amd64`, binaries execute directly with an extracted OpenVINO `LD_LIBRARY_PATH`. On `armv7`, the existing compatibility design is preserved: an ARMHF
loader/sysroot is extracted from the image and the aarch64 kernel executes the
32-bit ARM binaries through compatibility support. No QEMU is required on that
Pi path.

Extracted trees coexist at:

```text
work/host-runtime/armv7/
work/host-runtime/arm64/
work/host-runtime/amd64/
```

Each contains `host-runtime.env`; `host-run.sh` refuses to execute a runtime for
the wrong host architecture. See `HOWTO-HOST-RUN.md`.

## Firmware and dependency preparation

`fetch-runtime.sh` downloads Intel's pinned Raspbian runtime package only as a
source of the known-compatible VPU firmware and ARM reference artifacts. The
self-built container does not copy its host-side `libmyriadPlugin.so`.

```bash
./scripts/clone-openvino.sh
./scripts/fetch-runtime.sh
./scripts/prepare-deps.sh --platform amd64
```

`prepare-deps.sh` creates the flat firmware ZIP names expected by OpenVINO's old
dependency downloader. Those ZIPs are intentionally shared by all targets
because they contain only device firmware.

## Verification

Hardware-free static checks:

```bash
./ci/verify-static.sh
```

Platform-aware runtime report:

```bash
./scripts/verify.sh --platform armv7 --no-device
./scripts/verify.sh --platform arm64 --no-device
./scripts/verify.sh --platform amd64 --no-device
```

With an MA2450 attached, omit `--no-device` to add enumeration, tiny-model and
MobileNet checks.

`scripts/validate-reference.sh` is intentionally ARMv7-only. It uses Intel's
official Raspbian runtime as a historical/reference sanity check. For the ARM64 and amd64 self-built images use `scripts/verify.sh --platform arm64` or `--platform amd64`.

The files under `logs/` are retained as historical evidence from the original
Pi implementation. They should not be interpreted as native ARM64 or amd64 validation results.
Record new hardware results separately by platform.

## Project layout

```text
Dockerfile                         one Dockerfile for armv7 + arm64 + amd64
build.sh                           source/deps + target-aware Docker build
run.sh                             target-aware Docker runtime wrapper
container-entry.sh                 architecture-neutral image entrypoint
scripts/platform.sh                canonical platform mapping + lib discovery
scripts/fetch-runtime.sh           pinned Intel package -> VPU firmware
scripts/prepare-deps.sh            local firmware dependency mirror
scripts/prepare-mobilenet.sh       ONNX -> OpenVINO IR using native host MO
scripts/pull-runtime.sh            extract runtime per target
scripts/host-run.sh                ARMHF fallback or native ARM64/amd64 execution
scripts/verify.sh                  platform-aware verification report
scripts/validate-reference.sh      ARMv7 Intel-reference check only
patches/README.md                  patch applicability notes
ci/verify-static.sh                no-hardware architecture regressions
.github/workflows/static.yml       armv7/arm64/amd64 static matrix
smoke-test/                        tiny OpenVINO example
mobilenet-test/                    MobileNet classifier example
examples/webcam/                   live webcam classifier (Python + inference server)
examples/ssd-detect/               SSDLite-MobileNetV2 COCO detector
examples/deeplab-seg/              DeepLabV3 Pascal VOC segmenter
toolchain/armv7-native.toolchain.cmake
logs/                              historical Pi build/runtime evidence
```

## Troubleshooting

**Image architecture mismatch:** rebuild with the same target you pass to
`run.sh`:

```bash
./build.sh --platform amd64
./run.sh --platform amd64 list
```

**`MYRIAD` is not listed:** first separate plugin problems from USB problems.
Run `scripts/verify.sh --platform <target> --no-device`. It checks the plugin,
firmware, `plugins.xml`, ELF architecture and `ldd` resolution without requiring
the stick.

**Firmware missing:** the runtime expects `usb-ma2450.mvcmd` beside
`libmyriadPlugin.so`. Re-run `scripts/fetch-runtime.sh`, `scripts/prepare-deps.sh`
and rebuild.

**Firmware boot/re-enumeration times out:** verify `/dev/bus/usb` permissions,
the udev rule, Docker's device cgroup rule and the host-network setting. Compare
`lsusb` before/after a run.

**Host-native ARMv7 fails:** the compatibility path requires an ARM host with ARM32 support. Do not try to run the extracted `armv7` tree on x86.

**Host-native ARM64 fails:** verify `uname -m` reports `aarch64`/`arm64` and use `scripts/verify.sh --platform arm64 --no-device` to confirm the extracted application/plugin are ELF64 AArch64. If the host libc is incompatible with the Bullseye-built binaries, use the Docker path.

**Host-native amd64 fails with a libc error:** use the Docker path as the primary
runtime. The extracted binaries were built against the Bullseye userspace and
native execution depends on host ABI compatibility.

## Scope and validation status

The source tree now contains all three target paths and static checks. Physical
hardware validation still has to be run on the actual systems after the port:
full source build on each host, MA2450 boot/re-enumeration, tiny inference,
MobileNet reference comparison and repeated-run stability. See
`IMPLEMENTATION_NOTES.md` and `MULTI_PLATFORM_TODO.md` for the remaining
hardware gates.
