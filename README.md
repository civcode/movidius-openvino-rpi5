# OpenVINO 2020.3.2 with Movidius (MA2450) on a Raspberry Pi 5

OpenVINO **2020.3.2** - the last release line that still supports Intel Movidius
Myriad devices - built from source for **armv7l (armhf)** and run in a 32-bit
container on a **Raspberry Pi 5** (Debian 13, aarch64) with a USB2 Compute
Module 2 stick (`03e7:2150` in ROM mode, `03e7:f63b` when booted).

Everything needed to reproduce the build lives in this directory: `Dockerfile`,
`build.sh`, `run.sh`, the toolchain file, the patches, a tiny test model, and
scripts that fetch and verify the pinned vendor payload.

---

## Table of contents

| Section | What it covers |
|---|---|
| [Why 2020.3.2 and armv7l](#why-202032-and-armv7l) | the version/device constraints |
| [Host facts](#host-facts) | what the Pi 5 actually does with an arm32v7 container |
| [Quick start](#quick-start) | the three commands |
| [Beginner how-to](#how-to-mobilenetmd) | step-by-step MobileNet-on-the-stick walkthrough: [`HOWTO-MOBILENET.md`](HOWTO-MOBILENET.md) |
| [Project layout](#project-layout) | every file in here |
| [Vendor payload and pinning](#vendor-payload-and-pinning) | what is downloaded, from where, and how it is pinned |
| [How the armv7l target is forced](#how-the-armv7l-target-is-forced) | the toolchain file and what it changes |
| [Build stages](#build-stages) | build-deps -> configure -> builder -> smoke -> runtime |
| [Patches](#patches) | the two patches and why each is needed |
| [USB access on a Pi 5](#usb-access-on-a-pi-5) | the real root cause of `ncDeviceOpen:1009` and the flag set that fixes it |
| [Reference-runtime validation](#reference-runtime-validation-first) | proving the premise with Intel's own binaries before building |
| [Smoke test](#the-smoke-test) | the app and the hand-written tiny IR |
| [Measured results](#measured-results) | numbers |
| [MobileNet v2 demo](#mobilenet-v2-end-to-end-demo) | a real network, converted with the vendored Model Optimizer, checked against the ONNX reference |
| [Inference without Docker](#running-the-artifacts-directly-on-the-host) | copy the runtime + IR out of the image and run them on the host: beginner guide [`HOWTO-HOST-RUN.md`](HOWTO-HOST-RUN.md), details [`logs/HOST-RUN.md`](logs/HOST-RUN.md) |
| [Verification report](#verification-report) | `scripts/verify.sh` and what `logs/VERIFICATION.md` proves |
| [Troubleshooting](#troubleshooting) | failure signatures and fixes |
| [Limitations](#limitations-and-notes) | what this build does *not* do |

---

## Why 2020.3.2 and armv7l

* OpenVINO dropped Myriad support in 2021. The 2020.3 release is the last one
  with a MYRIAD plugin, `myriad_compile` and the mvnc/XLink transport, so the
  source tree is pinned to tag **`2020.3.2`** (commit
  `0b3773b7405d955d48642667ac5113289b9baab2`).
* The Myriad VPU toolchain in that release (fathom / `myriad_compile`, the
  `mvnc`/XLink transport) is only wired up for the architectures Intel shipped.
  A native aarch64 build of 2020.3 does not build the VPU path at all: the
  CMake code selects `TARGET_vpu:BOOL=OFF` and `myriad_compile` is not built.
* Intel's own answer for this device class was the **Raspbian** package
  (`l_openvino_toolkit_runtime_raspbian_p_2020.3.355`), which ships armv7l
  binaries.  So this project builds the same **armv7l** target, in an armhf
  container, on the Pi 5.

## Host facts

Measured on this machine (`edge`, Raspberry Pi 5):

```
$ uname -a
Linux edge 6.18.50+rpt-rpi-2712 #1 SMP PREEMPT ... aarch64
$ docker --version ; docker buildx version
Docker version 29.8.1, build ...
github.com/docker/buildx v0.37.0
$ grep CONFIG_COMPAT /boot/config-$(uname -r)
CONFIG_COMPAT=y
```

* arm32v7 (`armhf`) containers run **natively** - the 32-bit userland executes
  on the aarch64 kernel through `CONFIG_COMPAT`, no QEMU binfmt is involved.
  Inside such a container `uname -m` reports the *kernel* arch (`aarch64`)
  while `dpkg --print-architecture` reports `armhf`; that mismatch is exactly
  why the toolchain file is needed (see below).
* Memory: 8 GB RAM + 2 GB swap.  `BUILD_JOBS` is capped at 2 by default - the
  VPU model-introspection/`fathom` translation units are memory hungry.
* Two xHCI controllers: the stick enumerates on the USB2 bus as `03e7:2150`
  (ROM mode) and re-enumerates on the USB3 bus as `03e7:f63b` after firmware
  boot.  This re-enumeration is the heart of the USB problem solved below.

---

## Quick start

```bash
cd openvino-2020.3-rpi5

# 1. fetch + pin the vendor payload (source tree, official runtime, firmware,
#    local dependency mirror).  Skipped automatically by build.sh if present.
./scripts/clone-openvino.sh
./scripts/fetch-runtime.sh
./scripts/prepare-deps.sh

# 2. build the runtime image (hours on a Pi 5; the CMake configure stage alone
#    is fast and is the thing to iterate on)
./build.sh                      # full image
./build.sh --target configure   # CMake configure only - a few minutes

# 3. run it against the stick
./run.sh                        # device enumeration + tiny model on MYRIAD
./run.sh list                   # plugin/device enumeration only
./run.sh bench --iterations 20  # repeat inferences on the tiny model
./run.sh shell                  # bash inside the runtime image
```

A real network (converted with the vendored Model Optimizer, see
[MobileNet v2 demo](#mobilenet-v2-end-to-end-demo) and the beginner walkthrough
[`HOWTO-MOBILENET.md`](HOWTO-MOBILENET.md)):

```bash
./scripts/prepare-mobilenet.sh                   # download + MO convert + photos (~2 min)
./run.sh mobilenet                               # numerical check vs the ONNX reference
./run.sh mobilenet --image /models/images/cat.ppm
```

Then run the very same binaries with **no Docker at inference time** (beginner guide:
[`HOWTO-HOST-RUN.md`](HOWTO-HOST-RUN.md)):

```bash
./scripts/pull-runtime.sh                         # image -> work/host-runtime/, once
./scripts/host-run.sh mobilenet                   # host process: RESULT: PASS, 44.5 ms
./scripts/host-run.sh mobilenet --image vendor/models/images/banana.ppm --topk 3
```

Optional, but recommended before the first real build:

```bash
./scripts/validate-reference.sh        # Intel's own armv7l binaries, same stick
./scripts/validate-reference.sh --no-device
```

`run.sh` uses `--network=host -v /dev:/dev --device-cgroup-rule='c 189:* rwm'`
instead of `--privileged`.  That is not a style preference - it is the minimal
set that works on this host, and the reason is documented in
[USB access on a Pi 5](#usb-access-on-a-pi-5).

## Project layout

```
Dockerfile                         5 stages: build-deps, configure, builder, smoke, runtime
build.sh                           vendor payload + pristine source + docker build
run.sh                             host-side docker run wrapper (USB flags, modes)
container-entry.sh                 entrypoint inside the runtime image
HOWTO-MOBILENET.md                 beginner walkthrough: prepare, convert, classify on the stick
HOWTO-HOST-RUN.md                  beginner walkthrough: run the same binaries with no Docker
requirements-build.txt             pinned pure-python build tools (installed in a venv)
.dockerignore                      build-context hygiene
toolchain/armv7-native.toolchain.cmake   forces the armv7l target on a native compiler
patches/0001-cross-compile-skip-host-protoc.patch
patches/0002-ade-gcc10-no-error-warnings.patch
smoke-test/main.cpp                the inference smoke test (OpenVINO 2020.3 C++ API)
smoke-test/CMakeLists.txt          builds it against the installed tree
smoke-test/model/make_tiny_ir.py   regenerates model.xml/model.bin (IR v10, stdlib only)
mobilenet-test/main.cpp            MobileNet v2 classifier demo (PPM reader, softmax, top-k)
mobilenet-test/CMakeLists.txt      builds it against the installed tree
scripts/clone-openvino.sh          pinned git clone (tag 2020.3.2 + submodules)
scripts/fetch-runtime.sh           official 2020.3.2 raspbian runtime (headers + .mvcmd)
scripts/prepare-deps.sh            local dependency mirror (firmware zips etc.)
scripts/validate-reference.sh      prove the premise with Intel's armv7l binaries
scripts/reference-check.Dockerfile throwaway image used by validate-reference.sh
scripts/reset-stick.py             put the stick back into ROM mode (sudo, on the host)
scripts/verify.sh                  writes the verification report (logs/VERIFICATION.md)
scripts/prepare-mobilenet.sh       ONNX zoo model + Model Optimizer 2020.3.2 + photos
scripts/pull-runtime.sh            copy the runtime + armhf libs OUT of the image (host runs)
scripts/host-run.sh                run those artifacts directly on the host (no Docker)
scripts/99-movidius-ncs2.rules     udev rule so a normal user may open the stick
docs/diagnostics/                  throwaway mvnc/XLink/libusb probes kept as evidence
logs/                              build logs, RESULTS.md, VERIFICATION.md, MOBILENET.md,
                                   HOST-RUN.md, host-run-evidence.txt, mobilenet-run.log,
                                   mo-fp16-transcript.txt
work/                              scratch: reference-check artifacts, your own blobs/logs
  host-runtime/                    created by scripts/pull-runtime.sh: openvino/ (34 MB) +
                                   sysroot/ (3 MB armhf loader/libs) + bin/ wrappers
vendor/                            pinned third-party payload (see below)
vendor/models/                     MobileNet corpus: onnx, IR fp32/fp16, labels, photos
```

---

## Vendor payload and pinning

`vendor/` is created by the scripts, not committed:

| Path | Source | Pinned by |
|---|---|---|
| `vendor/openvino-2020.3.2/` | `git clone --recursive --branch 2020.3.2` | tag `2020.3.2`, commit `0b3773b7405d955d48642667ac5113289b9baab2`; submodules `ngraph@1797d7fb712d2ffa6048ce5f5b3b560b84ab5ae4`, `inference-engine/thirdparty/ade@cbe2db61a659c2cc304c3837406f95c39dfa938e` |
| `vendor/reference-runtime/l_openvino_toolkit_runtime_raspbian_p_2020.3.355/` | Intel's 2020.3.2 Raspbian runtime package | the official package name/version |
| `vendor/firmware/*.mvcmd` | copied out of that package (`usb-ma2450.mvcmd`, `usb-ma2x8x.mvcmd`, `pcie-ma248x.mvcmd`) | package version + `FIRMWARE_PACKAGE_VERSION 1656` |
| `vendor/deps/*.zip` | `scripts/prepare-deps.sh` builds the firmware archives in the layout `vpu_dependencies.cmake` expects (`mvnc/<device>.mvcmd`) | filename `firmware_<dev>_1656.zip` |
| `vendor/models/` | `scripts/prepare-mobilenet.sh`: ONNX model zoo tarball, synset labels, four test photos | sha256 of the tarball, of the `.onnx` and of every photo |

Two reproducibility mechanisms are used deliberately:

1. **`IE_PATH_TO_DEPS=/work/deps`.**  In this tree
   `cmake/download/download_and_check.cmake` treats every dependency URL as a
   local path when `IE_PATH_TO_DEPS` is set, and `file(COPY ...)`s it instead of
   downloading.  With the mirror prepared, the build performs **no network
   fetches at all**, which also removes the x86-only dependency downloads (the
   `protoc-3.7.1-linux-x86_64.tar.gz` trap, see [patches](#patches)).
2. **`snapshot.debian.org`.**  `arm32v7/debian:bullseye` is an old, partially
   archived suite; the live mirrors return 404s for many of its packages.  Both
   image stages therefore rewrite `/etc/apt/sources.list` to a fixed snapshot
   (`SNAPSHOT_DATE=20260824T000000Z`) and disable `Check-Valid-Until` /
   `Check-Date`, so the package set is frozen and reproducible.

**Python policy:** no Python *library* is installed through APT.  `python3` and
`python3-venv` come from APT (the venv enabler only), and every Python library
goes into `/opt/venv` from `requirements-build.txt` (pinned, pure-python).  No
`--break-system-packages`, no global pip installs.

`build.sh` recomputes `SYNC_STAMP = <source commit> + <sha256 of all patches>`
and passes it to the Dockerfile; the patched source lives in a BuildKit cache
mount and is re-created whenever that stamp changes.  The vendor tree itself is
reset to pristine (including submodule working trees) before every build, and
patches are verified with `--dry-run` first.

## How the armv7l target is forced

`toolchain/armv7-native.toolchain.cmake` is the core trick: it is a toolchain
file that uses the **native** compiler inside the armhf container but reports

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)     # the kernel would say aarch64
set(CMAKE_CROSSCOMPILING TRUE)
```

With `CMAKE_SYSTEM_PROCESSOR=armv7l` the 2020.3 CMake logic takes the armv7l
branch everywhere, which is what the MYRIAD path requires.  Observed effects in
the generated `CMakeCache.txt` / configure log:

```
The target system is: Linux -  - armv7l
ENABLE_VPU:BOOL=ON                ENABLE_MYRIAD:BOOL=ON
ENABLE_MYRIAD_NO_BOOT:BOOL=OFF    ENABLE_MKL_DNN:BOOL=OFF   (no CPU plugin)
ENABLE_GNA / ENABLE_CLDNN:OFF     THREADING:STRING=SEQ
NGRAPH_BUILD_DIR:STRING=/work/src/bin/armv7l/Release/lib
PLUGIN_FILES:INTERNAL=MYRIAD:myriadPlugin;HETERO:HeteroPlugin;MULTI:MultiDevicePlugin;TEMPLATE:templatePlugin
```

`CMAKE_FIND_ROOT_PATH` stays `/` so the container's own `libusb-1.0-0-dev`,
zlib and headers are found, and `-DTHREADS_PTHREAD_ARG=-pthread` skips CMake's
Threads-module probe (which would otherwise run a host-side compile test in the
cross-compiling configuration).

## Build stages

```
build-deps   arm32v7/debian:bullseye + build-essential, cmake, patch, libusb-1.0-0-dev,
             zlib1g-dev, python3, python3-venv  +  /opt/venv from requirements-build.txt
configure    copy vendor/openvino-2020.3.2 -> cache mount /work/src, apply patches/,
             cmake configure, print the cache summary
builder      make -j$BUILD_JOBS && make install  ->  /work/stage, then ngraph is moved
             from the staging prefix top level into deployment_tools/ngraph/{lib,include,
             cmake} so the tree matches Intel's raspbian package layout
smoke        regenerate the tiny IR, build smoke-test/main.cpp against the install tree,
             readelf + ldd sanity
mobilenet    build mobilenet-test/main.cpp against the same install tree, readelf + ldd
runtime      clean arm32v7 debian + libusb-1.0-0 + python3, copy /work/stage/deployment_tools
             to /opt/openvino, install hello_myriad + model + mobilenet_classify, entrypoint,
             and run both binaries with --help as a link sanity check
```

Iterate with `./build.sh --target configure` (seconds) before committing to the
compile.  `/work/src` and `/work/build` are BuildKit **cache** mounts, so a
re-run continues where it stopped; `vendor/deps` is a read-only **bind** mount
(cache mounts would let the build write into the mirror).

One BuildKit subtlety: a layer is keyed on the resolved instruction text, not on
what a cache mount contains, so changing `SYNC_STAMP` alone would *not* invalidate
the `make` layer.  Both the `configure` and `builder` RUNs therefore reference
`${SYNC_STAMP}` and assert that `/work/src/.sync_stamp` matches it - a changed patch
set forces a real recompile, and a cache that does not match the expected source
aborts instead of shipping a stale tree.

---

## Patches

`patches/NNNN-*.patch` are applied in sorted order inside the build (against a
copy of the vendor tree), and are verified by `build.sh` with `--dry-run`
against the pristine tree first.  `vendor/openvino-2020.3.2/` is never modified:
`build.sh` resets the superproject **and every submodule working tree** before
each build (the `.gitmodules` entries use `ignore = dirty`, so a dirty submodule
would otherwise be invisible in `git status` and a patch could be applied twice).

`./build.sh --no-patches` builds the vendor tree as upstream wrote it, to show that
these two patches are what make the build possible here (it also switches
`SYNC_STAMP` so the source cache is re-synced).

### `0001-cross-compile-skip-host-protoc.patch`

`cmake/dependencies.cmake` resolves the `protoc` dependency with
`DownloadAndExtractPlatformSpecific(...)`, i.e. it asks for
**`protoc-3.7.1-linux-x86_64.tar.gz`** whatever the target is.  On an armv7l
cross-compile configuration that file is never provided (and with
`IE_PATH_TO_DEPS` set, nothing is downloaded at all), and configure dies:

```
CMake Error at cmake/download/download_and_check.cmake:51 (file):
CMake Error at cmake/download/extract.cmake:16 (message):
  error: file to extract does not exist:
  '/work/stage/download/protoc-3.7.1-linux-x86_64.tar.gz'
```

The patch restricts that host-protoc block to amd64/x86_64 hosts.  Protobuf is
only needed for the ONNX importer (`NGRAPH_ONNX_IMPORT_ENABLE=FALSE` here) and
for the demo model converters, neither of which is part of a MYRIAD-only build.

### `0002-ade-gcc10-no-error-warnings.patch`

The `ade` graph library (submodule `inference-engine/thirdparty/ade`) sets

```cmake
if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wconversion -Wshadow ...")
```

unconditionally.  gcc 10 (the compiler in `arm32v7/debian:bullseye`) warns about
`redundant-move` in `ade/source/execution_engine.cpp:141`, and `-Werror` turns
that into a build failure:

```
/work/src/inference-engine/thirdparty/ade/sources/ade/source/execution_engine.cpp:141:21:
  error: redundant move in return statement [-Werror=redundant-move]
```

The patch keeps `-Werror` and downgrades the few diagnostics that newer
compilers add (`redundant-move`, `maybe-uninitialized`, `unused-variable`,
`unknown-pragmas`) back to warnings.

Because `ade` is a submodule, the patch file is generated inside it with full
prefixes (`a/inference-engine/thirdparty/ade/sources/ade/CMakeLists.txt`) so that
`patch -p1 -d /work/src` finds it from the superproject root.

## USB access on a Pi 5

This section is the reason the project exists.  The failure everyone hits with a
Movidius stick in a container on a Pi 5 is:

```
I: ncDeviceOpen:920 XLinkBootRemote is running for 1-ma2450
I: ncDeviceOpen:939 XLinkBootRemote ... returned success X_LINK_SUCCESS
   (15 seconds of nothing)
E: ncDeviceOpen:1009 Failed to find booted device after boot
W: ncDeviceOpen:1011 Device (1-ma2450) doesn't disappear after firmware loading
```

### What is actually happening

1. The stick boots into ROM mode: `03e7:2150` on the Pi 5's USB2 bus (bus 003).
2. `ncDeviceOpen()` finds it, `XLinkBootRemote()` uploads `usb-ma2450.mvcmd`
   successfully (~160 ms).
3. The device **re-enumerates** as `03e7:f63b` on the USB3/SuperSpeed bus
   (bus 004, it shows up as "VSC Loopback Device").
4. mvnc then looks for the booted device.  In `mvnc_api.c` that loop is
   `XLinkFindAllSuitableDevices(X_LINK_ANY_STATE, ...)` before/after boot - and
   after the boot it still sees only the *stale* unbooted entry
   (`name='1-ma2450'`, `XLinkFindAllSuitableDevices(X_LINK_BOOTED)` returns 0),
   so `countAfter != countBefore` never becomes true, the 15 s
   `g_deviceConnectTimeoutSec` expires, and you get the message above.

### The root cause is libusb's device cache, not permissions

OpenVINO 2020.3.2's vendored XLink (`inference-engine/thirdparty/movidius/XLink`)
enumerates USB with **libusb 1.0.24**, which

* builds a device list **once per context**, and `libusb_get_device_list()`
  returns that cached list, and
* refreshes it only from **`NETLINK_KOBJECT_UEVENT`** hotplug events.

A container in its own (bridge) network namespace receives **no kernel uevents**.
So the re-enumerated stick is invisible to libusb for the whole run, even
though the kernel, sysfs and `/dev/bus/usb` inside the container are perfectly
correct and live:

```
/sys/bus/usb/devices/4-1/idVendor = 03e7      /sys/bus/usb/devices/4-1/idProduct = f63b
/dev/bus/usb/004/0NN  crw-rw-r-- 1 root root 189,3NN
```

This was measured directly with a probe program built from the vendored XLink
sources (`docs/diagnostics/xlink_probe.c`, it calls `XLinkFindAllSuitableDevices` before and
after `XLinkBoot`):

| container network | after boot, what XLink reports | result |
|---|---|---|
| bridge (private netns) | `n=1 name='1-ma2450'`, `X_LINK_BOOTED` count `0` | `ncDeviceOpen:1009` |
| `--network=host` | `n=1 name='1-'`, `X_LINK_BOOTED` count `1` | works |

(For reference: without `XLINK_USE_BUS` - which this build does not define - the
XLink device names are `1-ma2450` before boot and `1-` after boot, because
`supportedDevices[]` in `XLink/pc/protocols/usb_boot.c` maps `0x2150 -> "ma2450"`
and the boot pid `0xf63b` to an empty name.)

### The flag set that works

```
docker run --rm --platform linux/arm/v7 \
    --network=host \
    -v /dev:/dev \
    --device-cgroup-rule='c 189:* rwm' \
    IMAGE
```

Measured matrix (same tiny model, stick reset to ROM mode before every run,
official OpenVINO 2020.3.2 `compile_tool`):

| Flags | Result |
|---|---|
| `--privileged -v /dev:/dev` (private netns) | **fail** `ncDeviceOpen:1009` - uevents still missing |
| `--privileged --network=host` (no `-v /dev:/dev`) | **fail** - privileged device nodes are static, the new bus-4 node is missing |
| `--device=/dev/bus/usb --device-cgroup-rule=... --network=host` | **fail** - `--device` publishes only nodes that exist at container start |
| `-v /dev/bus/usb:/dev/bus/usb --device-cgroup-rule=... --network=host` | **fail** (node visible after boot, XLink view still stale) |
| `-v /dev:/dev --network=host` (no cgroup rule) | **fail** (permission denied on the new node) |
| **`-v /dev:/dev --device-cgroup-rule='c 189:* rwm' --network=host`** | **works** (used by `run.sh`) |
| `--privileged --network=host -v /dev:/dev` | works |

`--privileged` is therefore *not* required: the minimum is a live `/dev` bind, a
cgroup rule for char major **189** (`/dev/bus/usb`), and the host network
namespace so libusb sees the uevents.  If `--network=host` is unacceptable for
your deployment, see [Alternatives](#alternatives-if-host-networking-is-not-allowed).

### Resetting the stick

A run that fails after boot leaves the stick in the booted (`03e7:f63b`) state,
which then looks like "the stick is not detected" on the next attempt.  On the
host:

```bash
sudo python3 scripts/reset-stick.py        # f63b -> 2150 (ROM mode)
```

(`scripts/reset-stick.py` issues `USBDEVFS_RESET`, falling back to
deauthorize + port rebind; run it from the host, not from the container.)

---

## Reference-runtime validation first

Before building anything, `scripts/validate-reference.sh` checks the premise with
Intel's **own** armv7l binaries from the official 2020.3.2 raspbian runtime
(`vendor/reference-runtime/.../deployment_tools/inference_engine/lib/armv7l/`)
inside an arm32v7 container built by `scripts/reference-check.Dockerfile`:

1. regenerate the tiny IR,
2. `compile_tool -m model.xml -d MYRIAD` with the official compiler - this boots
   the stick and compiles for the VPU,
3. compile `smoke-test/main.cpp` against the official headers,
4. run `hello_myriad_ref` against the stick.

Why this order matters: it separates *host/stick/flags/model* problems from
*our build* problems.  Every stage below is known-good against the official
runtime first, so if the image built here misbehaves, the difference is in our
build and not in the environment.  It also produced the USB findings above -
the same failure with the official binaries, the same fix, no custom code
involved.

## The smoke test

`smoke-test/main.cpp` uses the OpenVINO 2020.3 C++ API exactly as documented in
that release's headers:

* `Core::ReadNetwork(xml, bin)`, `CNNNetwork::getInputsInfo()` /
  `getOutputsInfo()`, `InputInfo::setPrecision(FP16)` + `setLayout(NCHW)`,
  `Core::LoadNetwork(network, "MYRIAD")`,
* `ExecutableNetwork::CreateInferRequest()`, `InferRequest::GetBlob(name)`,
  `InferRequest::Infer()`.  Note that `GetInputBlob`/`GetOutputBlob` do **not**
  exist in 2020.3 - the blob accessor is `GetBlob(name)`.
* shapes come from `blob->getTensorDesc().getDims()` and
  `InputInfo::getInputData()->getDims()`; `Layout` is `enum : uint8_t` in
  `ie_common.h`, so it has no `name()`/`toString()`.
* linking needs `-linference_engine -linference_engine_legacy
  -linference_engine_transformations -linference_engine_lp_transformations
  -lngraph` (`Data::setPrecision` and the legacy `CNNNetwork` live in
  `libinference_engine_legacy`; there is no `libngraph_backend.so` in the
  ngraph part of the package).  `libinference_engine.so` alone does **not**
  link - it references those libraries and they reference back into it, so
  `smoke-test/CMakeLists.txt` passes them inside `-Wl,--start-group ... -Wl,--end-group`
  and adds both library directories to the rpath.  Skipping this gives errors like
  `undefined reference to typeinfo for ngraph::op::v0::Interpolate` or
  `vtable for InferenceEngine::GatherLayer`.
* it prints the plugin version (`InferenceEngine::GetVersion`), input/output
  shapes, load+compile time, per-inference times, and FP16 min/max/mean of the
  output blob, and returns a non-zero exit code on any failure.

`smoke-test/model/make_tiny_ir.py` regenerates the test model with the Python
standard library only: a hand-written **IR v10** (`version="opset1"`) network
`Parameter -> Constant -> Convolution -> Result`, 1x3x32x32 input, 8 output
channels, everything FP16 (432-byte `.bin`).  Hand-written IR is deliberate: the
MYRIAD-only build has no ONNX importer, and a stdlib generator keeps the whole
project free of framework dependencies.  The `.bin` file is derived from the
`.xml` name by the 2020.3 parser, and `compile_tool` in 2020.3 has **no `-w`**
option.

## Measured results

Reference runtime (`scripts/validate-reference.sh`, official 2020.3.2 raspbian
binaries, same Pi 5 and stick):

```
plugin MYRIAD   api 2.1 build 2020.3.2-3506-c35b42b1d89-releases/2020/3 (myriadPlugin)
network         : tiny_conv_fp16
  input  input dims 1,3,32,32 -> FP16 NCHW
  output conv dims 1,8,32,32 FP32
load+compile    : 1492.97 ms
  inference 1/2/3 : 1.83 / 1.82 / 1.78 ms      mean 1.81 ms (551.78 fps)
RESULT: PASS
compile_tool    : Done. LoadNetwork time elapsed: 1487 ms   blob 1472 bytes
after teardown  : bus 003 03e7:2150  (stick back in ROM mode)
```

Self-built image (`./build.sh` then `sudo ./scripts/reset-stick.py && ./run.sh`):

```
available devices : MYRIAD
  plugin MYRIAD   api 2.1 build custom__ (myriadPlugin)
network         : tiny_conv_fp16
  input  input dims 1,3,32,32 -> FP16 NCHW
  output conv dims 1,8,32,32 FP32
load+compile    : 1479.92 ms
  inference 1/2/3 : 1.89 / 1.78 / 1.76 ms      mean 1.81 ms (551.94 fps)
  output conv (1,8,32,32, desc FP32, 4 B/elem) min -1.05859 max 2.64844 mean 1.02026
RESULT: PASS
20-iteration run: mean 1.78 ms (562.67 fps)
compile_tool    : Done. LoadNetwork time elapsed: 1485 ms   blob 1472 bytes
build time      : 748 s of compiling with -j2, image 251 MB (75.6 MB content)
```

The compiled blobs are not merely the same size, they are the same bytes:

| compiler | blob | md5 |
|---|---|---|
| Intel's `compile_tool` (official raspbian runtime) | 1472 B | `2488440e8766d7053e84864058057e78` |
| our `compile_tool` (this build) | 1472 B | `2488440e8766d7053e84864058057e78` |

Byte-identical blob and the same timing as Intel's runtime, which is the practical
proof that this build reproduces the official 2020.3.2 behaviour.  Build timings,
patch reproductions, the probe transcripts and the Docker flag matrix are in
[`logs/RESULTS.md`](logs/RESULTS.md).

---

## MobileNet v2 end-to-end demo

The tiny hand-written IR proves the toolchain; a real network proves that this build can
actually convert and run an off-the-shelf model.  `mobilenetv2-7` (ONNX model zoo, opset 7,
Top-1 70.94 %) is converted with the **Model Optimizer from the same pinned 2020.3.2 source
tree** and classified on the stick:

```bash
./scripts/prepare-mobilenet.sh        # zoo download -> MO 2020.3.2 -> IR + labels + photos
./run.sh mobilenet                    # 1x3x224x224 tensor, compared with the zoo's own output
./run.sh mobilenet --image /models/images/banana.ppm --topk 3
IR=fp32 ./run.sh mobilenet            # the FP32-weight IR instead of FP16
```

| step | result |
|---|---|
| Model Optimizer 2020.3.2 (`--data_type FP16`) | `IR version 10`, 266 layers, 14.35 s, 148 MB peak |
| IR size | 138 KB xml + 13.95 MB bin (FP32) / 6.98 MB bin (FP16) |
| load + compile on MYRIAD | ~1.65-1.70 s |
| inference | **44.5 ms** (22.5 fps), identical for FP32 and FP16 weights |
| vs the zoo's `test_data_set_0/output_0.pb` | max abs diff **0.0525** (logits up to 26.27), mean 0.0122, **top-5 identical** |
| photos | tabby cat, banana, cup, bald eagle correct; see the table in `logs/MOBILENET.md` |

Every command with its full output is saved in [`logs/mobilenet-run.log`](logs/mobilenet-run.log),
the Model Optimizer transcript in `logs/mo-fp16-transcript.txt`, and the summary evidence is
section 10 of `logs/VERIFICATION.md` (regenerated by `./scripts/verify.sh`).

Four things were worth learning here, all written up in
[`logs/MOBILENET.md`](logs/MOBILENET.md):

* 2020.3 Model Optimizer has **no `--compress_to_fp16`** - FP16 weights come from
  `--data_type FP16`.
* `mobilenetv2-10.onnx` (opset 10) fails in this MO version because its batch dimension is
  a `dim_param`: partial inference stops at the final `Gemm`.  The opset 7 export has
  static shapes and converts in 14 s.
* MO's own unit tests have to be excluded from the staged tree: `mo/utils/import_extensions.py`
  imports every `.py` file in the package, and the vendored `*_test.py` files import a
  test-only helper that is not part of the source tree.
* Conversion runs in a **native arm64** `python:3.8-slim` container (MO is pure Python, the
  Pi 5 kernel runs arm64 natively), while inference stays in the arm32v7 runtime image -
  that is why no armhf `onnx`/`numpy` wheel is needed and no emulation is involved.

## Running the artifacts directly on the host

Beginner step-by-step version of this section: [`HOWTO-HOST-RUN.md`](HOWTO-HOST-RUN.md).

The containers are only needed for the **build** and the **Model Optimizer conversion**.  The
runtime, the demo binaries, `compile_tool` and the IR can all be copied out of the image and
executed as ordinary host processes - the aarch64 kernel runs the ELF32 hard-float binaries
in compat mode (`CONFIG_COMPAT=y`), no QEMU and no `apt install libc6:armhf`:

```bash
./scripts/pull-runtime.sh                 # image -> work/host-runtime/{openvino,sysroot} (37 MB)
./scripts/host-run.sh list                # device enumeration, no container
./scripts/host-run.sh mobilenet           # numerical self-test on the stick
./scripts/host-run.sh mobilenet --image vendor/models/images/banana.ppm --topk 3
./scripts/host-run.sh compile --model vendor/models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml \
                              --blob  vendor/models/blobs/mobilenet-v2.blob
./scripts/host-run.sh shell               # ov binaries on PATH via loader wrappers
```

Measured on this host: enumeration `MYRIAD`; inference **44.54 ms** and `RESULT: PASS` with
max `|diff|` 0.0525 / mean 0.0122 and the identical top-5 (`556, 818, 811, 827, 918`) - the
same numbers as `./run.sh mobilenet` inside the container; `compile_tool -d MYRIAD` produced
a 6 999 040 B device blob in 1.66 s.  `--network=host` and the uevent problem disappear
here, because the host *is* the uevent namespace.

Raw evidence - `CONFIG_COMPAT=y`, zero `qemu-arm` binfmt handlers, `readelf` of the ELF32 ARM
binary, the executed binary itself, the 10-library sysroot, the blob header - is in
[`logs/host-run-evidence.txt`](logs/host-run-evidence.txt).

Two host-side requirements (details, transcripts and the reproduction of the mutex quirk in
[`logs/HOST-RUN.md`](logs/HOST-RUN.md)):

1. **USB node permissions.** `/dev/bus/usb/NNN/NNN` is `root:root 0664`, so an unprivileged
   run fails in the boot path (`ncDeviceOpen:1009`).  Install `scripts/99-movidius-ncs2.rules`
   (`SUBSYSTEM=="usb", ATTR{idVendor}=="03e7", MODE="0660", GROUP="plugdev"`) + `udevadm
   trigger`, or run under `sudo`.
2. **`/tmp/mvnc.mutex`.** `mvnc` opens that fixed path with `O_CREAT` and `exit(1)`s if it
   cannot; with Debian's `fs.protected_regular = 2` a file created by *another* user in
   world-writable `/tmp` is not openable - even by root - which shows up as
   `ncDeviceOpen:754 global mutex initialization failed`.  Run as one user, or
   `rm`/`chmod 0666` the file.

The compiled blob is architecture-independent (it is VPU code) but compiler/firmware-pinned,
and it is ELF-formatted (`7f 65 6c 66`) unlike the NCSDK-era `mvNCCompile` `.fby` graphs - so
loading it with a raw-XLink host loader needs the matching 2020.3 `.mvcmd` and is **not**
verified here.  Do not mix the two boot paths: pushing an OpenVINO `.mvcmd` with the MA2Host
boot tool wedged the stick's USB interface during this experiment (see
[HOST-RUN.md §5](logs/HOST-RUN.md); recovery then needs a physical replug).

## Verification report

`scripts/verify.sh` regenerates a single artifact that shows the whole stack works,
top to bottom:

```bash
./scripts/verify.sh > logs/VERIFICATION.md      # add --no-device to skip the stick steps
```

| Section of the report | Evidence it prints |
|---|---|
| 1. Host | `uname -a` (aarch64 kernel), `CONFIG_COMPAT=y`, and that **no `qemu-arm` binfmt handler exists** - so 32-bit armv7 code is not being emulated |
| 2. Docker | Docker 29.8.1 / buildx, image id and `linux/arm` platform |
| 3. arm32 userspace | `getconf LONG_BIT` = 32, `hello_myriad` ELF header `EI_CLASS=01` (ELF32) + `e_machine=28 00` (EM_ARM), `ldd` resolving `ld-linux-armhf.so.3` and `/usr/lib/arm-linux-gnueabihf` |
| 4. Image contents | `plugins.xml` = HETERO, MULTI, MYRIAD only; the three `.mvcmd` next to `libmyriadPlugin.so`; `ngraph/lib` present |
| 5. Device enumeration | `available devices : MYRIAD`, plugin api 2.1 |
| 6. Inference | compile + 10 inferences on the stick, `RESULT: PASS`, stick back in ROM mode afterwards |
| 7. `compile_tool` | blob md5 compared with the blob from Intel's official runtime |
| 8. Python policy | every APT package line from the Dockerfile (`python3`, `python3-venv` only) and the pinned pip libraries |
| 9. Vendor pin | commit `0b3773b740…`, tag `2020.3.2`, submodules `ngraph@1797d7f`, `ade@cbe2db6` |
| 10. MobileNet demo | the corpus, the numerical check against the ONNX reference output and the top-1 class of each photo (skipped when `vendor/models/` is absent) |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ncDeviceOpen:1009 Failed to find booted device after boot` + `Device (1-ma2450) doesn't disappear after firmware loading` | libusb's device list is never refreshed because the container gets no kernel uevents | run with `--network=host` (plus `-v /dev:/dev` and the char-189 cgroup rule) |
| same failure, but with `--privileged` and no `-v /dev:/dev` | privileged containers get a **static** `/dev/bus/usb` snapshot from container start | add `-v /dev:/dev` |
| stick never appears in the container at all | `--device=/dev/bus/usb` copies nodes that existed at start only | bind-mount `/dev` instead |
| `unable to find image locally` / `pull access denied` on `docker run --platform linux/arm/v7 <image>` | Docker 29's default provenance attestation makes the build output an OCI *index* | build with `--provenance=false` (already the default in `build.sh`) |
| apt fails with 404 / `Release file ... not valid yet` | live mirrors dropped part of the old bullseye armhf pool | keep the `snapshot.debian.org` pin (`SNAPSHOT_DATE` build-arg) |
| `error: file to extract does not exist: .../protoc-3.7.1-linux-x86_64.tar.gz` (usually preceded by `file COPY cannot find "/work/deps/protoc-3.7.1-linux-x86_64.tar.gz"`) | x86-only host protoc dependency in `cmake/dependencies.cmake` | `patches/0001-...` (already applied by `build.sh`) |
| `failed to fetch anonymous token: Get "https://auth.docker.io/token...` before any stage runs | the `# syntax=docker/dockerfile:1.4` front end is resolved over the network and DNS was flaky | `docker pull docker.io/docker/dockerfile:1.4` once |
| a rebuild finishes in seconds after you edited a patch | BuildKit reused the `make` layer; cache mounts are not part of the layer key | the Dockerfile asserts `/work/src/.sync_stamp == SYNC_STAMP`; if it ever trips, run `./build.sh --no-cache` |
| `error: redundant move in return statement [-Werror=redundant-move]` in ade | gcc 10 vs ade's unconditional `-Werror` | `patches/0002-...` |
| `make: *** [Makefile:1301: cmake_check_build_system] Error 1` + `CMakeSystem.cmake:6 (include): ... toolchain ...` | the toolchain file was not visible in the stage that re-ran CMake | the `builder` stage mounts `toolchain/` too |
| the stick is not detected although it is plugged in | previous run left it booted (`03e7:f63b`) | `sudo python3 scripts/reset-stick.py` (add `--power` if needed) |
| compile is killed / swap thrash | 8 GB box, heavy VPU translation units | `BUILD_JOBS=1 ./build.sh` |
| `VPU_LOG_LEVEL` config key rejected by `compile_tool` | that key is not supported for MYRIAD in 2020.3.2 (`[NOT_FOUND] config key is not supported for VPU`) | use `CONFIG_KEY(LOG_LEVEL)` / `VPU_CONFIG_KEY(LOG_LEVEL)`; for full `mvLog` output in a custom program call `ncGlobalSetOption(NC_RW_LOG_LEVEL, {NC_LOG_DEBUG})` |
| no mvnc debug output with `IE_VPU_LOG_LEVEL=LOG_DEBUG` | the shipped 2020.3.2 build does not forward it to `mvnc` | build the standalone probe in `docs/diagnostics/` (it sets `NC_RW_LOG_LEVEL` itself) |

### Diagnostics in `docs/diagnostics/`

The throwaway probes that produced the findings above live in
[`docs/diagnostics/`](docs/diagnostics/README.md) (they are not part of the image,
and `build.sh`/`run.sh` never touch them):

* `xlink_probe.c` - links the vendored `XLink` sources directly and prints
  `XLinkFindAllSuitableDevices` counts and device names before/after `XLinkBoot`.
  This is what proved the stale-libusb theory.
* `mvnc_probe.c` - standalone `ncAvailableDevices()` + `ncDeviceOpen()` with
  `NC_LOG_DEBUG`.
* `usb_list.c` - minimal libusb enumerator (prints `libusb_get_version()`).
* `docker-flags-test.sh` - the flag matrix from the table above.
* `xlink-probe-run.sh`, `probe-run.sh`, `xlink-probe-flags.sh` - build/run helpers
  for the probes (the standalone mvnc/XLink compile recipe is in those scripts:
  `-D__PC__ -DHAVE_STRUCT_TIMESPEC -DUSE_USB_VSC`, include `mvnc/include`,
  `mvnc/include/watchdog`, `XLink/shared/include`, `XLink/pc`, `XLink/pc/protocols`,
  link with `g++ ... -lusb-1.0 -lpthread -ldl`; `ncDeviceOpen` needs a non-NULL
  `watchdogHndl` from `watchdog_create()`).
* `diag-usb.sh` - host-side USB topology dump (buses, controllers, `4-1` sysfs
  attributes).

Their outputs (probe binaries, compiled blobs, `mvnc-debug.log`) are in
`work/reference-check/`, which is scratch and can be deleted at any time.

### Alternatives, if host networking is not allowed

`--network=host` is the *no-code-change* fix.  Two other routes exist and were
evaluated:

1. **`ENABLE_MYRIAD_NO_BOOT=ON` + boot the stick beforehand.**  With `NO_BOOT`
   defined, `mvnc_api.c` takes the `#ifdef NO_BOOT` branch that reports
   `X_LINK_BOOTED` immediately ("Connect to already boot device"), so libusb's
   stale list never matters.  You boot the stick once from the host (or from a
   privileged one-shot container using OpenVINO's own `usb-ma2450.mvcmd`) and
   then run inference against the already-booted device.  Cost: the stick must be
   re-flashed whenever it falls back to ROM mode, and `myriad_compile` runs
   against a device that must stay booted across runs.
2. **Patch the vendored XLink** so that its device search re-initialises the
   libusb context (`libusb_exit(NULL)` + `libusb_init(NULL)` on the first failed
   attempt) instead of trusting the cached list.  This makes the private-netns
   configuration work as well.  Not done here: keeping Intel's libraries
   unmodified makes the image comparable to the official runtime, and the
   `--network=host` requirement is cheap to document.

## Limitations and notes

* **No CPU plugin.**  2020.3.2's raspbian package does not ship one either, and
  this build disables `MKL_DNN`, `GNA`, `CLDNN`, GPU/OpenCL.  Available devices
  are `MYRIAD`, `HETERO`, `MULTI`, `TEMPLATE`.  `THREADING=SEQ` follows from the
  armv7l cross-compile configuration (no TBB/OMP in this configuration).
* `ENABLE_MYRIAD_NO_BOOT=OFF`, i.e. the plugin boots the stick itself - that is
  the configuration the USB section above is about.
* The `.mvcmd` firmware files must sit next to `libmyriadPlugin.so`: `mvnc`'s
  `getFirmwarePath()` resolves the directory with `dladdr()` and overwrites
  `customFirmwareDirectory`.  The install tree produced here keeps them together,
  exactly as the official package does.
* Blob/network precision: the smoke test asks for FP16 activations
  (`setPrecision(Precision::FP16)`); asking for FP32 input makes the MYRIAD
  compiler insert a conversion node or fail.
* `--platform linux/arm/v7` is required on both build and run on this host: the
  kernel reports `aarch64`, so without it Docker would use an arm64 image.
* Disk: the build directory plus the runtime image needs several GB.  Check
  `docker system df` and `df -h /` before starting; `logs/RESULTS.md` records the
  sizes actually used.
* The MobileNet corpus (`vendor/models/`, ~30 MB) is **not** part of the image: it is
  excluded by `.dockerignore` and mounted read-only at `/models` by `./run.sh mobilenet`.
  Generating it needs Docker and internet on the host; the photo preprocessing uses the
  host's `python3` + Pillow and is skipped (with a message) if Pillow is missing - the
  demo itself takes binary PPMs and needs no Python at all.
* `./run.sh mobilenet` needs the corpus; without it the mode exits with a hint to run
  `./scripts/prepare-mobilenet.sh`, and section 10 of `logs/VERIFICATION.md` says the same.
* FP32 and FP16 IR variants run at the same speed on the VPU (~44.5 ms) because the MYRIAD
  compiler works in FP16 internally; only the `.bin` size differs.
