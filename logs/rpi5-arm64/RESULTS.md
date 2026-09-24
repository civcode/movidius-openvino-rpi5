# OpenVINO 2020.3.2 + Movidius MA2450 - native arm64 (Raspberry Pi 5) build and example results

Date: 2026-09-24. All timings measured on this host. Raw logs in `logs/rpi5-arm64/`.

## 1. Host

| item | value |
|---|---|
| machine | Raspberry Pi 5, 4 cores, 8 GiB RAM, 29 GB SD card |
| kernel | `Linux edge 6.18.50+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.18.50-1+rpt1 aarch64` |
| Docker | 29.8.1 (BuildKit, default `docker buildx` builder) |
| qemu-user (binfmt) | 1:10.0.13+ds-0+deb13u1 (`/usr/libexec/qemu-binfmt/x86_64-binfmt-P`, flags `POF`) |
| stick | Intel Myriad VPU, USB ROM id `03e7:2150` before boot |
| disk free at start | ~6.3 GB (build cache pruned before the cold `-j4` build) |

Repo: branch `multiplatform-arm64`, HEAD `62be395` ("examples: fix seg_stream GUI crash").
`vendor/openvino-2020.3.2` = tag 2020.3.2 @ `0b3773b7405d955d48642667ac5113289b9baab2`,
ngraph `1797d7fb`, ade `cbe2db61` (fetched by `scripts/fetch-openvino.sh`, 222 MB, 26 s).

## 2. Build

`build.sh` was changed (uncommitted, `git status` shows `M build.sh`) so the arm64 target
uses all cores instead of the hard-coded 2: the native-target branch now sets
`BUILD_JOBS` from `getconf _NPROCESSORS_ONLN` (capped at 8), armv7 stays at 2 jobs, and the
existing `BUILD_JOBS=` env override still wins. This was requested because the first build
left two of the four cores idle.

Two full builds were run from an empty build cache (builder caches were pruned with
`docker builder prune -f` before the `-j4` run, so both are cold: each re-synced the source
and started compiling at `[  0%]`):

| run | command | total | builder stage (`make -C /work/build`) |
|---|---|---|---|
| `-j2` (old default) | `./build.sh --platform arm64` | **1576 s** (26.3 min) | 1401.3 s |
| `-j4` (patched) | `BUILD_JOBS=4 ./build.sh --platform arm64` | **1279 s** (21.3 min) | 1138.8 s |

Speedup: 1.23x on the compile stage, 1.23x overall. `make -C /work/build -j4` is visible in
`build-arm64-j4.log` line 911. Both runs: rc=0, no compile or link errors anywhere in the
OpenVINO build, configure stage used the *native* toolchain (`use_cmake_toolchain=0`,
`dpkg_arch=arm64`), i.e. no cross-compiler and no QEMU for the OpenVINO build itself.

Image: `openvino-2020.3-movidius-arm64:latest` = `db57a09f9438`, 299 MB on disk.
Stages built: `build-deps -> configure -> builder -> smoke -> mobilenet -> webcam -> ssd ->
seg -> runtime`. The example stages run their unit tests during the build (`test_half`,
`ssd_test`, `seg_test`, Dockerfile lines 307/342/377) - all passed.

Binaries in the image, all ELF64 / AArch64 (validated by the stage checks):
`hello_myriad`, `mobilenet_classify`, `mobilenet_server`, `ssd_detect`, `seg_detect`.

## 3. Model conversion (arm64 native)

`./scripts/prepare-mobilenet.sh` - rc=0 in **114 s**, fully native
(`MO_PLATFORM=linux/arm64`, `python:3.8-slim`, ONNX front end is pure Python):
IR v10 FP16 `xml 137973 B / bin 6975776 B`, FP32 `bin 13951520 B`, MO self-report
"Total execution time: 14.26 seconds". Byte sizes identical to the armv7 baseline.

## 4. Examples that ran on the stick (arm64)

| test | command | result |
|---|---|---|
| device enumeration | `./run.sh --platform arm64 list` | `available devices : MYRIAD`, plugin api 2.1, firmware `pcie-ma248x` + `usb-ma2x8x` + `usb-ma2450` |
| tiny demo IR | `./run.sh --platform arm64` | compile 1488.05 ms, inference **1.80 ms (555.7 fps)**, RESULT: PASS |
| MobileNet self-test + photos | `./run.sh --platform arm64 mobilenet` | fp16: compile 1600-1627 ms, inference **44.56-44.62 ms (22.4 fps)**, `max abs diff 0.0525`, top-5 556/818/811/827/918 = reference -> PASS; fp32 IR also PASS (`0.0671`); photo top-1 identical to the armv7 baseline (banana 0.7170, cup 0.9600, cat->tabby 0.2463, eagle->kite 0.8785, dog->tandem bicycle 0.5825) |
| host-native runtime (no Docker for inference) | `./scripts/pull-runtime.sh --platform arm64` then `./scripts/host-run.sh --platform arm64 list|demo|mobilenet` | arm64 needs no sysroot/loader (unlike armv7): binaries run directly. demo 1489 ms compile / 1.84 ms inference; mobilenet self-test PASS (`0.0525`, same top-5) |
| webcam inference backend, 1000 known-content images | `examples/accuracy-test/fetch-sample-images.sh` (14 s, git clone) then `python examples/accuracy-test/accuracy_test.py --backend host --ir fp16` | **top-1 814/1000 = 81.40 %**, **top-5 959/1000 = 95.90 %**, 18.73 img/s, server start 1.70 s, total 55.1 s |
| same, fp32 IR | `accuracy_test.py --backend host --ir fp32` | 81.40 % / 95.90 %, 18.66 img/s, 55.3 s |
| Python client unit tests | `./scripts/test-python-clients.sh` | all checks PASS (LinePipe/windowed_fps, resize bounds, launcher scripts, ssd_stream fake-server incl. multi-word labels + malformed lines + server-exit diagnostics, seg_stream fake-server MASK/END) - needs a host venv with numpy+opencv (`work/venv-examples`, arm64 wheels) |
| static multi-platform checks | `./ci/verify-static.sh` | `static multi-platform checks: PASS` (arm64 -> ELF64/AArch64/183, `lib/aarch64`, native host runtime) |

The accuracy test drives `mobilenet_server` through `examples/webcam/infer-server.sh`, so the
webcam example's inference backend and its 602112 B in / 4000 B out protocol are exercised
end to end. The live-camera GUI part (`webcam_mobilenet.py` + OpenCV window) is **not**
tested: this host has no `/dev/video*` camera.

## 5. Blocked: SSD and Deeplabv3 IR conversion on arm64

`./run.sh --platform arm64 ssd` and `... seg` stop with
`no IR in vendor/models/... - run ./scripts/prepare-ssdlite.sh first`.
The example *binaries* are fine (`ssd_detect --help` and `seg_detect --help` work in the
arm64 image, and their post-processing unit tests pass); only the Model Optimizer step fails.

`prepare-ssdlite.sh` (default native path) fails in 35 s at the pip step:

```
ERROR: Could not find a version that satisfies the requirement tensorflow==1.15.0
       (from versions: 2.10.0rc0, ... 2.13.1)
```

`prepare-deeplabv3.sh` fails the same way in 8 s. Root causes, verified:

1. **No TensorFlow 1.x wheel for aarch64 on PyPI.** TF 1.x was never published for
   aarch64 (tensorflow/tensorflow#62695); official aarch64 wheels start at TF 2.9. The
   aarch64 entries that pip *does* list for TF 2.10-2.13 are 1941-byte stub wheels
   (`tensorflow-2.10.0-cp37-...-manylinux_2_17_aarch64.whl`, 1941 bytes) - not real builds.
   piwheels has no TensorFlow 1.15 either. `MO` 2020.3's TF front end needs `tensorflow<2`.
2. **x86_64 emulation fallback (`MO_PLATFORM=linux/amd64`) does not work with this host's
   qemu-user.** `prepare-ssdlite.sh` with `MO_PLATFORM=linux/amd64` got through the pip
   install (python:3.7-slim + TF 1.15.0 + numpy 1.18.5, rc=0) and died inside MO with

   ```
   ImportError: .../numpy/linalg/_umath_linalg.cpython-37m-x86_64-linux-gnu.so:
                failed to map segment from shared object
   ```

   Attempts that did **not** fix it: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
   `OPENBLAS_MAIN_FREE=1`; replacing numpy's bundled
   `numpy.libs/libopenblasp-r0-34a18dc3.3.7.so` with Debian's `libopenblas.so.0` (this fixes
   the OpenBLAS mmap error but the next numpy extension fails the same way); a
   `numpy.core._multiarray_tests` stub; `QEMU_GUEST_BASE=0x100000 / 0x1000000000`;
   `QEMU_RESERVED_VA=0x10000000|0x40000000|0x100000000` (qemu then refuses with
   `Cannot allocate vsyscall page`); re-registering binfmt with
   `docker run --privileged --rm tonistiigi/binfmt --install x86_64` (handler unchanged);
   building numpy from source under emulation (`pip install --no-binary :all:
   numpy==1.18.5`, 1267 s, build-dependency step failed).
   Reading the wheels with `readelf -lW`: numpy's `cp37` `manylinux1`/`manylinux2010`
   extension modules (1.18.5, 1.19.5, 1.21.6 all identical) have four `PT_LOAD`s with
   `p_align 0x200000`, `p_memsz > p_filesz` in the second segment, and file offsets that
   are not monotone with the virtual addresses - qemu-user 10.0.13 cannot map them, while
   TF's own `1.1 GB _pywrap_tensorflow_internal.so` and `libtensorflow_framework.so.1` map
   fine. So this is a qemu-user / manylinux-wheel ELF-layout problem, not a numpy bug and
   not a USB/OpenVINO problem.
3. **What does work on arm64 for these two models:** the pinned downloads.
   `vendor/models/ssdlite_mobilenet_v2/source/` contains
   `ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz` (51025348 B),
   `frozen_inference_graph.pb` (19911343 B), `pipeline.config`, `mscoco_label_map.pbtxt`, and
   `vendor/models/labels/coco.txt` / `pascal_voc.txt` were produced. Only the MO invocation
   is missing, so converting on any x86_64 machine and copying
   `vendor/models/ssdlite_mobilenet_v2/openvino/` + `vendor/models/deeplabv3/openvino/` back
   to this Pi makes `./run.sh --platform arm64 ssd|seg` work unchanged (the IR is
   architecture-independent, the runtime here is already built and verified).

Options, in the order I would try them:

* convert on an x86_64 host (`./scripts/prepare-ssdlite.sh` / `prepare-deeplabv3.sh` with the
  default `MO_PLATFORM=linux/amd64`) and copy the IR back - no recipe change, no hacks;
* publish/consume a community aarch64 TF 1.15 wheel (e.g. `noahzhy/tf-aarch64`,
  `KumaTea/tensorflow-aarch64`) - unpinned third-party binaries, so deliberately **not**
  done here;
* keep the arm64 MO container on the ONNX front end only (works, section 3) and pick
  ONNX-exported models for the SSD/segmentation examples - that changes the pinned
  conversion semantics (`ssd_v2_support.json`, `--output detection_classes,...`), so it is a
  recipe change, not a fix.

## 6. State left on the host

* persistent container `mo203-tf` (`python:3.7-slim`, `--platform linux/amd64`, 2.0 GB layer)
  left running with TF 1.15 installed, to keep the emulation experiment reproducible:
  remove it with `docker rm -f mo203-tf` when you do not need it.
* `work/tmp-x86/` (extracted `.so` files from the qemu analysis) has been deleted.
* `work/venv-examples/` = host venv with numpy 2.5.3 + OpenCV 5.0.0 (arm64 wheels) for the
  Python clients and the accuracy test.
* `examples/accuracy-test/images/` = 1000 ImageNet sample JPEGs (research-licensed, do not
  redistribute).
* disk after everything: ~3.0 GB free on `/dev/mmcblk0p2` (90 % used) - a full re-build needs
  `docker builder prune -f` first.
* the stick is left in ROM mode (`03e7:2150`) via `sudo python3 scripts/reset-stick.py`.

## 7. Reproduce

```bash
./build.sh --platform arm64                      # 1279 s with the patched -j4 default
./scripts/prepare-mobilenet.sh                   # 114 s, native arm64
./run.sh --platform arm64 list                   # MYRIAD enumeration
./run.sh --platform arm64                        # tiny demo IR, 1.80 ms
./run.sh --platform arm64 mobilenet              # self-test + photos
./scripts/pull-runtime.sh --platform arm64
./scripts/host-run.sh --platform arm64 list|demo|mobilenet
python3 -m venv work/venv-examples && work/venv-examples/bin/pip install numpy opencv-python
./examples/accuracy-test/fetch-sample-images.sh
work/venv-examples/bin/python examples/accuracy-test/accuracy_test.py --backend host --ir fp16
./scripts/test-python-clients.sh
./ci/verify-static.sh
# blocked on arm64 (see section 5):
./scripts/prepare-ssdlite.sh                     # rc=1, no aarch64 TF 1.15 wheel
./scripts/prepare-deeplabv3.sh                   # rc=1, same
./run.sh --platform arm64 ssd                    # "no IR ... run prepare-ssdlite.sh first"
./run.sh --platform arm64 seg                    # "no IR ... run prepare-deeplabv3.sh first"
```
