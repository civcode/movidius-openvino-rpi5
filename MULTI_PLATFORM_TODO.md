# TODO: Convert `movidius-openvino-rpi5` into a Multi-Platform Linux Project

> **Implementation note (2026-09-20):** the repository now contains the
> multi-platform structural changes described here. Hardware-dependent acceptance
> items remain intentionally unchecked until they are run on a Pi 5 and an x86_64
> Linux host with an MA2450 attached. See `IMPLEMENTATION_NOTES.md`.

## Goal

Refactor the project so the same repository can build and run the Intel Movidius MA2450 / MYRIAD OpenVINO 2020.3.2 stack on both:

- **Raspberry Pi 5 host** running an **ARMv7/armhf userspace container** (`linux/arm/v7`), preserving the existing known-good behavior.
- **x86_64 Ubuntu/Linux host** running a **native amd64 container** (`linux/amd64`).

The project should retain the same high-level runtime path on both platforms:

```text
application (hello_myriad / mobilenet_classify)
    -> OpenVINO Inference Engine 2020.3.2
    -> libmyriadPlugin.so
    -> mvnc / XLink / libusb
    -> usb-ma2450.mvcmd firmware boot
    -> Movidius MA2450 device
```

The x86 target should **not** use MA2Host as the normal runtime. It should build and use OpenVINO's own MYRIAD plugin, MVNC/XLink transport, and firmware-loading mechanism, just as the current Raspberry Pi build does.

---

# 1. Define the Supported Platform Matrix

- [ ] Document the canonical target names used by project scripts:
  - `armv7` = Raspberry Pi-compatible OpenVINO ARMHF build.
  - `amd64` = native x86_64 Linux build.
- [ ] Define the corresponding Docker platform strings:
  - `armv7` -> `linux/arm/v7`
  - `amd64` -> `linux/amd64`
- [ ] Define default base images for each target:
  - `armv7` -> `arm32v7/debian:bullseye`
  - `amd64` -> `debian:bullseye` or `amd64/debian:bullseye`
- [ ] Decide whether the public target name should be `amd64` or `x86_64`; use one consistently in CLI options and documentation.
- [ ] Preserve the current OpenVINO source pin:
  - tag/release: OpenVINO `2020.3.2`
  - pinned commit: `0b3773b7405d955d48642667ac5113289b9baab2`
- [ ] Preserve MA2450 firmware compatibility as a hard requirement.
- [ ] Explicitly state that Pi 5 remains an **ARMv7 userspace build on an aarch64 kernel**, rather than changing the Pi build to aarch64.
- [ ] Define minimum tested environments, for example:
  - Raspberry Pi 5 + 64-bit Raspberry Pi OS/Debian kernel with ARM32 compat enabled.
  - Ubuntu x86_64 with Docker Engine and USB access.
- [ ] Add a table to the README showing which combinations are supported, tested, experimental, or unsupported.

### Acceptance criteria

- [ ] A reader can determine the target architecture, Docker platform, and runtime mode without reading the Dockerfile.
- [ ] Existing Raspberry Pi behavior is explicitly retained rather than implicitly replaced.

---

# 2. Add a Central Platform Configuration Layer

Create one source of truth for platform-specific values instead of scattering architecture checks throughout shell scripts.

Suggested options:

```text
scripts/platform.sh
```

or:

```text
platform/
  armv7.env
  amd64.env
```

- [ ] Add a helper that accepts `--platform armv7|amd64`.
- [ ] Add optional automatic detection from `uname -m`:
  - `x86_64` -> `amd64`
  - `aarch64`/`arm64` on Pi -> `armv7` by default for this project.
- [ ] Allow an explicit CLI option to override automatic detection.
- [ ] Export normalized variables such as:

```bash
TARGET=armv7|amd64
DOCKER_PLATFORM=linux/arm/v7|linux/amd64
BASE_IMAGE=...
IMAGE=...
USE_CMAKE_TOOLCHAIN=0|1
CMAKE_TOOLCHAIN_FILE=...
EXPECTED_ELF_MACHINE=...
```

- [ ] Keep platform selection independent from host detection where possible so CI/buildx can build another target explicitly.
- [ ] Fail clearly on unsupported architectures rather than silently selecting ARM.
- [ ] Add `--print-platform` or equivalent diagnostics to simplify troubleshooting.

### Acceptance criteria

- [ ] `./build.sh --platform armv7` selects the current ARMHF behavior.
- [ ] `./build.sh --platform amd64` selects native x86_64 behavior.
- [ ] All other top-level scripts consume the same normalized platform configuration.

---

# 3. Refactor `build.sh`

Current Pi-specific items in `build.sh` include:

```text
BASE_IMAGE=arm32v7/debian:bullseye
IMAGE=openvino-2020.3-rpi5:latest
--platform linux/arm/v7
```

- [ ] Add `--platform armv7|amd64` argument parsing.
- [ ] Load the centralized platform configuration.
- [ ] Replace the hard-coded default image name with a target-specific name, e.g.:

```text
openvino-2020.3-movidius-armv7:latest
openvino-2020.3-movidius-amd64:latest
```

- [ ] Pass `DOCKER_PLATFORM` into `docker build` instead of hard-coding `linux/arm/v7`.
- [ ] Pass target information as Docker build arguments:

```text
TARGET
TARGETPLATFORM / BUILDPLATFORM (where useful)
BASE_IMAGE
USE_CMAKE_TOOLCHAIN
CMAKE_TOOLCHAIN_FILE
```

- [ ] Keep `BUILD_JOBS` configurable per target.
- [ ] Consider a higher default job count on x86 while retaining a conservative default on Pi.
- [ ] Verify `--provenance=false` is still required on current Docker for both targets; retain it if it avoids local image/OCI-index issues.
- [ ] Preserve existing `--target`, `--no-cache`, `--no-patches`, and `--image` behavior.
- [ ] Include the selected platform in the source/cache stamp if target-specific configure artifacts could collide.
- [ ] Ensure BuildKit cache directories do not accidentally reuse an ARM-configured CMake tree for amd64 or vice versa.
  - Prefer target-qualified cache paths or target-qualified cache IDs.
- [ ] Print a build summary before invoking Docker, including target, Docker platform, base image, image tag, OpenVINO commit, and patch mode.
- [ ] Update the post-build instructions to use the same platform selection for `run.sh`.

### Acceptance criteria

- [ ] ARM and amd64 builds can exist locally at the same time without cache contamination.
- [ ] Building one target after the other does not require manual deletion of `/work/build` caches.
- [ ] `build.sh` no longer contains an unconditional `linux/arm/v7` runtime/build selection.

---

# 4. Refactor the Dockerfile to Be Architecture-Neutral

## 4.1 Base image selection

Current Dockerfile uses:

```dockerfile
FROM --platform=linux/arm/v7 ${BASE_IMAGE}
```

- [ ] Change stages to consume a build argument instead of hard-coding ARMv7.
- [ ] Ensure Dockerfile syntax allows the platform argument in all relevant `FROM` lines.
- [ ] Apply the selected platform consistently to:
  - build dependency stage
  - configure stage
  - builder stage
  - runtime stage
- [ ] Avoid architecture-specific image references outside centralized defaults.

## 4.2 Conditional CMake toolchain

The current ARM build forces:

```text
-DCMAKE_TOOLCHAIN_FILE=/work/toolchain-ro/armv7-native.toolchain.cmake
```

- [ ] Keep the ARMv7 toolchain file for `armv7`.
- [ ] Do **not** pass the ARMv7 toolchain file for `amd64`.
- [ ] Construct optional CMake arguments inside the Docker build, e.g.:

```bash
CMAKE_PLATFORM_ARGS=""
if [ "$TARGET" = "armv7" ]; then
    CMAKE_PLATFORM_ARGS="-DCMAKE_TOOLCHAIN_FILE=/work/toolchain-ro/armv7-native.toolchain.cmake"
fi
```

- [ ] Verify native amd64 CMake detects:
  - x86_64 host/target
  - `CMAKE_CROSSCOMPILING=FALSE`
- [ ] Verify ARM continues to report the expected `armv7l` target despite the Pi's aarch64 kernel.

## 4.3 Keep the MYRIAD feature configuration identical

- [ ] Keep these options enabled for both targets unless a verified x86-specific source issue requires otherwise:

```text
-DENABLE_VPU=ON
-DENABLE_MYRIAD=ON
-DENABLE_MYRIAD_NO_BOOT=OFF
-DENABLE_INTEL_MYRIAD_COMMON_ENABLE=ON
-DENABLE_IR_READER=ON
```

- [ ] Keep CPU/GPU/GNA plugins disabled initially to minimize differences between platforms.
- [ ] Keep `THREADING=SEQ` initially unless x86 requires a change.
- [ ] Do not introduce an OpenVINO version upgrade as part of this port.

## 4.4 Remove hard-coded `lib/armv7l` paths

The Dockerfile currently uses explicit paths such as:

```text
/work/stage/deployment_tools/inference_engine/lib/armv7l
```

- [ ] Replace every hard-coded architecture directory with discovery.
- [ ] Add a reusable shell expression/helper, for example:

```bash
OV_IE_LIB="$(find /work/stage/deployment_tools/inference_engine/lib \
  -mindepth 1 -maxdepth 1 -type d | head -n1)"
```

- [ ] Validate the discovered directory contains at minimum:
  - `libinference_engine.so`
  - `libmyriadPlugin.so`
  - `plugins.xml`
  - `usb-ma2450.mvcmd`
- [ ] Fail if zero or more than one unexpected architecture directory is discovered, rather than silently using the wrong one.
- [ ] Use the discovered path for:
  - `LD_LIBRARY_PATH`
  - `myriad_compile --help`
  - smoke-test `ldd`
  - MobileNet-test `ldd`
  - runtime self-tests
- [ ] Keep nGraph path architecture-neutral (`deployment_tools/ngraph/lib`) unless the amd64 build proves otherwise.

## 4.5 Build smoke and MobileNet examples natively per target

- [ ] Pass the ARM toolchain to `smoke-test` only for the `armv7` target.
- [ ] Do not pass a toolchain for amd64.
- [ ] Do the same for `mobilenet-test`.
- [ ] Retain `OV_ROOT` and `OV_RUNTIME_PREFIX` behavior.
- [ ] Add architecture verification after building:
  - ARM should produce ELF32 ARM hard-float binaries.
  - amd64 should produce ELF64 x86-64 binaries.
- [ ] Make architecture checks target-aware rather than merely printing `readelf` output.

### Acceptance criteria

- [ ] `libmyriadPlugin.so` is compiled from the same pinned OpenVINO source for both ARMv7 and x86_64.
- [ ] The amd64 image contains a native x86_64 `libmyriadPlugin.so` and native x86_64 demo binaries.
- [ ] The ARM image continues to contain ARMv7 versions.
- [ ] Firmware remains next to `libmyriadPlugin.so`, preserving OpenVINO's firmware lookup behavior.

---

# 5. Audit the OpenVINO Patch Set Per Architecture

Current patches:

```text
patches/0001-cross-compile-skip-host-protoc.patch
patches/0002-ade-gcc10-no-error-warnings.patch
```

- [ ] Test each patch against an amd64 configure/build.
- [ ] Determine whether patch `0001` is:
  - harmless on amd64,
  - unnecessary on amd64 but safe to apply globally,
  - or needs a platform-specific condition.
- [ ] Preserve the ARM behavior that avoids using an inappropriate x86 host `protoc` during the ARM-target build.
- [ ] Verify patch `0002` remains appropriate for the GCC version in the selected amd64 base image.
- [ ] Add comments to each patch or a `patches/README.md` documenting:
  - original failure
  - affected platform(s)
  - whether it is still required on amd64
- [ ] If platform-specific patches become necessary, implement explicit selection instead of relying on failed patch application.
- [ ] Keep `--no-patches` useful for reproducing upstream failures on each platform.

### Acceptance criteria

- [ ] Patch application is deterministic on both targets.
- [ ] A patch is not silently masking an amd64-specific issue.

---

# 6. Make Dependency Preparation Multi-Platform

Inspect and refactor:

```text
scripts/prepare-deps.sh
requirements-build.txt
vendor/deps/
```

- [ ] Identify which cached dependency archives are host architecture-specific.
- [ ] Ensure amd64 does not reuse ARM-only binaries from the dependency mirror.
- [ ] Confirm source-only dependencies can be shared safely.
- [ ] Separate platform-specific dependency payloads where necessary, e.g.:

```text
vendor/deps/common/
vendor/deps/armv7/
vendor/deps/amd64/
```

- [ ] Update `IE_PATH_TO_DEPS` handling if the upstream OpenVINO downloader expects a flat directory.
- [ ] Review `requirements-build.txt` comments that are specifically about absent armv7 PyPI wheels.
- [ ] Make Python build requirements target-aware if amd64 can use wheels while ARM uses source/system packages.
- [ ] Do not introduce architecture-specific Python packages unless actually required by the OpenVINO build.
- [ ] Verify all builds can still operate reproducibly/offline after dependency preparation.

### Acceptance criteria

- [ ] `prepare-deps.sh` produces the correct payload for each target.
- [ ] No ARM executable is accidentally invoked during an amd64 build or vice versa.

---

# 7. Separate Firmware Acquisition from Host Runtime Architecture

The MA2450 firmware runs on the VPU, not on the Linux CPU, so it should be treated as device firmware rather than an ARM host binary.

- [ ] Refactor firmware acquisition so `usb-ma2450.mvcmd` is clearly platform-independent from the project's point of view.
- [ ] Keep a single known-good firmware source/version pinned to OpenVINO 2020.3 compatibility.
- [ ] Do not replace the firmware merely because the host becomes x86_64.
- [ ] Ensure the final installation places firmware beside `libmyriadPlugin.so` for both targets.
- [ ] Validate presence of:
  - `usb-ma2450.mvcmd`
  - any other firmware retained by the current package (`usb-ma2x8x.mvcmd`, `pcie-ma248x.mvcmd`)
- [ ] Add checksum validation for firmware if not already present.
- [ ] Document why firmware can be sourced from the existing pinned Intel/Raspbian runtime even when the host plugin is x86_64.
- [ ] Avoid invoking MA2Host/ma2boot during normal setup or testing.

### Acceptance criteria

- [ ] The x86 image boots an MA2450 through OpenVINO's MYRIAD plugin using the same compatible firmware flow as ARM.

---

# 8. Refactor `run.sh`

Current Pi-specific runtime setting:

```text
--platform linux/arm/v7
```

- [ ] Add/consume `--platform armv7|amd64` using the same platform helper as `build.sh`.
- [ ] Select the matching default image tag.
- [ ] Replace hard-coded Docker platform with `DOCKER_PLATFORM`.
- [ ] Keep all current modes:
  - demo
  - list
  - shell
  - bench
  - custom
  - mobilenet
- [ ] Keep the same OpenVINO device name: `MYRIAD`.
- [ ] Ensure model mounts and paths are platform-independent.
- [ ] Print the selected target/image before running when verbose mode is enabled.

### Acceptance criteria

- [ ] The same command shape works on both hosts, e.g.:

```bash
./run.sh --platform armv7 mobilenet
./run.sh --platform amd64 mobilenet
```

- [ ] If automatic detection is enabled, `./run.sh mobilenet` selects the correct local target by default.

---

# 9. Validate USB Passthrough on x86 Linux

The current Pi run command uses:

```text
--network=host
-v /dev:/dev
--device-cgroup-rule='c 189:* rwm'
```

This is designed to survive device re-enumeration when OpenVINO uploads firmware.

- [ ] Start by using the exact same USB access strategy on amd64.
- [ ] Verify the unbooted and booted MA2450 USB device can both be observed inside the container.
- [ ] Verify the post-firmware device node is visible without restarting the container.
- [ ] Verify cgroup permissions permit access to newly created USB char devices.
- [ ] Test whether `--network=host` is actually required on the x86 host for the same libusb/XLink re-enumeration reason.
- [ ] Do **not** remove `--network=host` from the common path until tested; a difference in host networking/udev behavior could reintroduce the known timeout.
- [ ] Compare behavior with and without `--network=host` and record the result.
- [ ] Avoid falling back to `--privileged` as the default if the narrower current permissions work.
- [ ] Test rootful Docker first; document rootless Docker separately if desired.
- [ ] Verify USB bus/device numbering differences do not leak into scripts.
- [ ] Ensure no script assumes the Pi-specific bus transition documented in the current comments.

### Acceptance criteria

- [ ] `Core::GetAvailableDevices()` reports `MYRIAD` in the amd64 container with the stick attached.
- [ ] Device boot/re-enumeration succeeds repeatedly without physically replugging the stick between healthy runs.

---

# 10. Generalize the Udev Rule and Host Setup

Current rule:

```text
scripts/99-movidius-ncs2.rules
```

- [ ] Verify the vendor/product matching is architecture-independent.
- [ ] Rename the rule or comments if they unnecessarily imply Raspberry Pi only.
- [ ] Document installation on Ubuntu x86_64.
- [ ] Document required group membership (`plugdev` or selected group).
- [ ] Add commands to reload rules and retrigger devices.
- [ ] Document Docker daemon permissions separately from USB udev permissions.
- [ ] Include troubleshooting commands using `lsusb` before and after firmware boot.

### Acceptance criteria

- [ ] A clean Ubuntu x86 host can be prepared using documented steps without changing source code.

---

# 11. Keep `container-entry.sh` Architecture-Neutral

This script is already comparatively portable because it discovers the architecture directory.

- [ ] Retain dynamic discovery of `inference_engine/lib/<arch>`.
- [ ] Tighten discovery to reject ambiguous results.
- [ ] Remove comments that imply `armv7l` is the only expected directory.
- [ ] Print the detected architecture/lib directory in diagnostics.
- [ ] Verify firmware discovery/output works for `armv7l` and the amd64 install directory (likely `intel64` or whatever the build actually emits).
- [ ] Do not hard-code the expected amd64 directory name unless upstream requires it.
- [ ] Ensure `LD_LIBRARY_PATH` includes both:
  - discovered Inference Engine lib directory
  - `${OV_ROOT}/ngraph/lib`

### Acceptance criteria

- [ ] No changes to `container-entry.sh` are required when switching between the two images at runtime.

---

# 12. Refactor `scripts/host-run.sh`

The current host-native mode is highly ARM-specific because it runs ARMHF binaries directly on the Pi's aarch64 kernel using an extracted loader/sysroot.

- [ ] Split platform-specific host execution logic.
- [ ] Suggested structure:

```text
scripts/host-run.sh
scripts/host-run-armv7.sh
scripts/host-run-amd64.sh
```

or functions selected by the centralized target helper.

## ARM path

- [ ] Preserve the current ARMHF sysroot approach.
- [ ] Preserve explicit use of `ld-linux-armhf.so.3` where required.
- [ ] Preserve the no-QEMU design.

## amd64 path

- [ ] Run the extracted native x86_64 executables directly where host glibc compatibility permits.
- [ ] Set `LD_LIBRARY_PATH` to the extracted OpenVINO and nGraph directories.
- [ ] Do not use `ld-linux-armhf.so.3`.
- [ ] Do not inspect for `arm-linux-gnueabihf`.
- [ ] Do not require a foreign-architecture sysroot.
- [ ] Optionally use an extracted x86 runtime/sysroot only if needed for deterministic host-native execution.
- [ ] Treat Docker execution as the primary supported mode if native-host ABI compatibility becomes fragile.

## Common host-runtime extraction

- [ ] Refactor runtime extraction so architecture-independent files and architecture-specific libs are copied predictably.
- [ ] Include firmware next to `libmyriadPlugin.so`.
- [ ] Add a manifest describing the target architecture of an extracted runtime.
- [ ] Refuse to run an ARM runtime on x86 or an amd64 runtime on ARM with a clear message.

### Acceptance criteria

- [ ] `host-run.sh` contains no unconditional ARM loader invocation.
- [ ] Native amd64 host execution, if advertised as supported, passes device enumeration and MobileNet inference.

---

# 13. Refactor `scripts/pull-runtime.sh`

Current script assumptions include ARMHF loader/sysroot extraction.

- [ ] Add target selection.
- [ ] For `armv7`, retain the existing extracted ARMHF loader/sysroot behavior.
- [ ] For `amd64`, copy the native OpenVINO runtime and demo binaries without ARMHF-specific files.
- [ ] Store extracted runtimes in target-specific directories, e.g.:

```text
work/host-runtime/armv7/
work/host-runtime/amd64/
```

- [ ] Include architecture metadata/manifest.
- [ ] Validate ELF architecture during extraction.
- [ ] Prevent one target from overwriting the other's extracted runtime.

### Acceptance criteria

- [ ] Both runtime trees can coexist.

---

# 14. Refactor `scripts/prepare-mobilenet.sh`

Current default contains an ARM-specific Model Optimizer platform selection.

- [ ] Replace the fixed `MO_PLATFORM=linux/arm64` default with host/target-aware logic.
- [ ] Use `linux/amd64` when running the converter on x86_64.
- [ ] Keep model output architecture-neutral where possible; OpenVINO IR XML/BIN should be reusable across host architectures.
- [ ] Do not generate separate ARM and x86 copies of identical IR unless testing shows a genuine difference.
- [ ] Preserve FP16 and FP32 preparation modes.
- [ ] Verify generated OpenVINO 2020.3-compatible IR produces identical expected inference results on both target builds.
- [ ] Keep source/reference test tensors common to both platforms.

### Acceptance criteria

- [ ] One prepared MobileNet model corpus can be mounted into either runtime image.

---

# 15. Audit `smoke-test/CMakeLists.txt`

The current file already discovers `lib/*`, which is a good multi-platform foundation.

- [ ] Remove ARM-only wording from comments.
- [ ] Verify the glob resolves exactly one intended architecture lib directory on amd64.
- [ ] Keep runtime RPATH/install assumptions architecture-neutral.
- [ ] Confirm linked libraries are found correctly on both targets.
- [ ] Add compile-time or test-time architecture logging if useful.
- [ ] Do not duplicate source code per architecture.

### Acceptance criteria

- [ ] The exact same `smoke-test/main.cpp` builds and runs on ARMv7 and amd64.

---

# 16. Audit `mobilenet-test/CMakeLists.txt` and `main.cpp`

- [ ] Keep `main.cpp` unchanged unless an actual architecture bug appears.
- [ ] Verify no type-size, alignment, or buffer assumptions differ on x86_64.
- [ ] Preserve the OpenVINO API path:

```text
Core -> ReadNetwork -> LoadNetwork("MYRIAD") -> InferRequest
```

- [ ] Keep architecture directory discovery dynamic in CMake.
- [ ] Verify FP16/FP32 handling and output element-size logic on amd64.
- [ ] Compare numerical output with the existing reference tensor.

### Acceptance criteria

- [ ] The same MobileNet source and same IR run successfully on both platforms.
- [ ] No MA2Host-specific code is added to the example.

---

# 17. Refactor Diagnostic Scripts

Current diagnostic files contain many explicit ARM references, including:

```text
docs/diagnostics/xlink-probe-run.sh
docs/diagnostics/diag-usb.sh
docs/diagnostics/xlink-probe-flags.sh
docs/diagnostics/docker-flags-test.sh
docs/diagnostics/probe-run.sh
```

- [ ] Replace hard-coded `--platform linux/arm/v7` with the platform helper.
- [ ] Replace hard-coded `/ov/lib/armv7l` with dynamic lib-directory discovery.
- [ ] Ensure standalone `mvnc_probe` builds natively for each target.
- [ ] Ensure standalone XLink probe builds natively for each target.
- [ ] Keep the probes useful for differentiating:
  - USB enumeration problem
  - firmware boot problem
  - device re-enumeration problem
  - MVNC open problem
  - OpenVINO plugin problem
- [ ] Update diagnostic README wording from “inside the armv7 container” to platform-neutral explanations where appropriate.
- [ ] Preserve Pi-specific findings as historical/platform-specific notes rather than deleting them.

### Acceptance criteria

- [ ] A failed amd64 device-open can be diagnosed with the same low-level tools used on Pi.

---

# 18. Refactor `scripts/verify.sh`

- [ ] Add platform selection/detection.
- [ ] Replace hard-coded expected ARM architecture with target-specific assertions.
- [ ] Verify image metadata matches the requested Docker platform.
- [ ] Verify executable ELF architecture:
  - ARM: ELF32 / ARM
  - amd64: ELF64 / x86-64
- [ ] Verify `libmyriadPlugin.so` architecture too.
- [ ] Verify `plugins.xml` includes MYRIAD.
- [ ] Verify `usb-ma2450.mvcmd` is present next to the MYRIAD plugin.
- [ ] Run `hello_myriad --list-only` / equivalent enumeration.
- [ ] Run tiny-model inference.
- [ ] Optionally run MobileNet if the model corpus exists.
- [ ] Emit a machine-readable summary or clear PASS/FAIL lines for CI.

### Acceptance criteria

- [ ] `verify.sh` proves both host architecture correctness and actual MA2450 inference.

---

# 19. Refactor `scripts/validate-reference.sh`

The current reference validation is based on Intel's ARM/Raspbian runtime.

- [ ] Decide whether reference validation remains ARM-only or gains an x86 reference package.
- [ ] If retained as ARM-only, explicitly label it as such and skip it gracefully on amd64.
- [ ] If an official matching x86 OpenVINO 2020.3 package is used, add a separate amd64 reference validation path.
- [ ] Do not mix the reference runtime with the self-built runtime during normal tests.
- [ ] Keep the self-built `libmyriadPlugin.so` as the main project artifact.

### Acceptance criteria

- [ ] The presence of an ARM-only historical validation tool does not block amd64 builds.

---

# 20. Rename Raspberry-Pi-Specific Project Identifiers

Optional but recommended after functional portability is proven.

- [ ] Consider renaming the repository/project title from `movidius-openvino-rpi5` to something platform-neutral, for example:
  - `movidius-openvino-linux`
  - `openvino-ma2450-linux`
  - `openvino-2020.3-movidius-linux`
- [ ] Replace image name `openvino-2020.3-rpi5` with target-qualified neutral names.
- [ ] Keep backward-compatible aliases/environment variables for one release if others depend on current names.
- [ ] Update comments that state the project itself is Raspberry-Pi-only.
- [ ] Keep a dedicated Raspberry Pi section documenting its unusual ARMHF-on-aarch64 arrangement.

### Acceptance criteria

- [ ] Naming reflects the new scope without losing the Pi-specific instructions that remain necessary.

---

# 21. Update README Architecture Explanation

Rewrite the README around a common architecture plus target-specific notes.

Suggested structure:

```text
Overview
Supported platforms
How it works
Quick start
  Raspberry Pi 5
  Ubuntu x86_64
Build architecture
MYRIAD firmware boot path
USB passthrough
MobileNet example
Host-native runtime
Diagnostics
Known limitations
Reproducibility
```

- [ ] Explain that `libmyriadPlugin.so` is built from source inside the Docker build for each host target.
- [ ] Explain that `usb-ma2450.mvcmd` is VPU firmware and not an ARM host executable.
- [ ] Explain that MobileNet uses OpenVINO's MYRIAD plugin, not MA2Host.
- [ ] Explain why Raspberry Pi requires ARMv7 while x86 uses a native build.
- [ ] Remove statements that incorrectly generalize ARM restrictions to x86.
- [ ] Preserve historically verified Pi behavior as its own subsection.
- [ ] Clarify the pinned OpenVINO version/device-support rationale.
- [ ] Include exact build/run examples for each platform.

Example desired UX:

```bash
# Pi 5
./build.sh --platform armv7
./run.sh --platform armv7

# Ubuntu x86_64
./build.sh --platform amd64
./run.sh --platform amd64
```

### Acceptance criteria

- [ ] A new user can follow only the section for their architecture and get a working smoke test.

---

# 22. Update `HOWTO-HOST-RUN.md`

- [ ] Split into common concepts, Pi ARMHF instructions, and x86_64 instructions.
- [ ] Keep `ld-linux-armhf.so.3` instructions only in the Pi section.
- [ ] Document amd64 dynamic-loader/library requirements separately.
- [ ] Explain why Docker is the most reproducible execution route.
- [ ] Clearly mark host-native execution as optional if it is more sensitive to distro/glibc differences.

---

# 23. Update `HOWTO-MOBILENET.md`

- [ ] Remove assumptions that the classifier executable is ARM32.
- [ ] Keep model preparation common.
- [ ] Add platform-specific build/run examples only where needed.
- [ ] Verify expected/reference output instructions are identical.
- [ ] Add a result table with measured ARM and x86 host-side load/inference timing, making clear that VPU inference is still performed by the same MA2450 device.

---

# 24. Preserve Existing Logs as Historical Evidence

Current files under `logs/` contain Pi-specific evidence and should not be mechanically rewritten as if they were x86 results.

- [ ] Leave existing Pi verification logs intact.
- [ ] Label them clearly as Raspberry Pi 5 / ARMv7 results.
- [ ] Add new amd64 logs rather than replacing Pi logs.
- [ ] Suggested layout:

```text
logs/
  rpi5-armv7/
  ubuntu-amd64/
```

- [ ] Record build/configure outputs for amd64.
- [ ] Record ELF architecture evidence.
- [ ] Record device enumeration.
- [ ] Record tiny-model inference.
- [ ] Record MobileNet result comparison.
- [ ] Record USB re-enumeration behavior.

### Acceptance criteria

- [ ] Claims in the README can be traced to platform-specific evidence.

---

# 25. Add Automated Static Tests That Do Not Require Hardware

Hardware-free checks should run for both targets wherever possible.

- [ ] Build OpenVINO builder/runtime image for ARMv7.
- [ ] Build OpenVINO builder/runtime image for amd64.
- [ ] Verify `libmyriadPlugin.so` exists.
- [ ] Verify firmware files exist.
- [ ] Verify `plugins.xml` contains MYRIAD.
- [ ] Verify demo binaries launch with `--help` without loading a device.
- [ ] Verify `myriad_compile --help` works.
- [ ] Compile the tiny model/IR tooling.
- [ ] Validate ELF machine type.
- [ ] Ensure no target contains libraries for the other target accidentally.
- [ ] Run shellcheck on shell scripts if feasible.
- [ ] Run simple grep-based regression checks preventing new hard-coded `/armv7l` paths in common scripts.

### Acceptance criteria

- [ ] Most portability regressions are detectable without an attached compute stick.

---

# 26. Add Hardware-in-the-Loop Test Procedure

CI may not have a USB Movidius device, so define a reproducible manual/HIL test suite.

For each platform:

- [ ] Confirm device appears in `lsusb` before runtime start.
- [ ] Run device enumeration.
- [ ] Confirm OpenVINO reports `MYRIAD`.
- [ ] Run tiny model for at least 3 iterations.
- [ ] Run MobileNet FP16.
- [ ] Compare MobileNet output to the stored reference.
- [ ] Run repeated open/close cycles to detect USB re-enumeration instability.
- [ ] Run at least 20 inference iterations.
- [ ] Stop/start the container without physically reconnecting the stick.
- [ ] Replug device and rerun.
- [ ] Capture failures from both MVNC and XLink probes if any instability occurs.

Suggested result matrix:

| Test | Pi 5 / ARMv7 | Ubuntu / amd64 |
|---|---|---|
| Image builds | [ ] | [ ] |
| MYRIAD enumerates | [ ] | [ ] |
| Firmware boot succeeds | [ ] | [ ] |
| Tiny IR inference | [ ] | [ ] |
| MobileNet FP16 | [ ] | [ ] |
| Reference output matches | [ ] | [ ] |
| Repeated boot cycles | [ ] | [ ] |
| Host-native runtime | [ ] | [ ] |

---

# 27. Add CI Matrix

Even without attached hardware, CI can validate build portability.

- [ ] Add a matrix with at least:

```text
armv7
amd64
```

- [ ] Use native amd64 CI for amd64.
- [ ] Decide whether ARM CI uses:
  - a self-hosted Pi runner, or
  - Docker buildx/QEMU for build-only checks.
- [ ] Do not claim inference was hardware-tested if CI only built the image under QEMU.
- [ ] Cache dependency mirror/downloads carefully by target.
- [ ] Cache Docker build layers per target.
- [ ] Upload build metadata/logs as CI artifacts.
- [ ] Run static scripts/tests for every pull request.

### Acceptance criteria

- [ ] A change that hard-codes ARM into common code fails the amd64 CI job.
- [ ] A change that breaks ARM configuration fails the ARM build job.

---

# 28. Introduce a Platform-Aware Runtime Manifest

To make extracted runtimes and debugging less error-prone:

- [ ] Generate a manifest during build containing:

```text
project version/commit
OpenVINO commit
TARGET
Docker platform
base image
compiler version
CMake architecture
OpenVINO lib directory
firmware checksum
build timestamp (optional/reproducible policy permitting)
```

- [ ] Install it into the image, e.g. `/opt/openvino/BUILD-INFO.txt`.
- [ ] Copy it with host-runtime extraction.
- [ ] Print it in verbose diagnostics.

### Acceptance criteria

- [ ] A user can determine whether a runtime is ARM or amd64 without inspecting ELF headers manually.

---

# 29. Normalize Architecture Discovery in One Shell Helper

There are multiple places that currently discover or assume architecture directories.

- [ ] Add a shared function, e.g.:

```bash
find_openvino_ie_libdir() {
    ...
}
```

- [ ] Validate required contents before returning the path.
- [ ] Use the same helper in:
  - container entrypoint
  - verification scripts
  - host-run scripts
  - diagnostics where practical
- [ ] For Docker build stages where the helper file is unavailable, use the exact same validation logic.
- [ ] Do not depend on `uname -m` to infer OpenVINO's install directory name.

### Acceptance criteria

- [ ] The project never needs to know whether amd64 upstream calls its directory `intel64`, `x86_64`, or another name unless an upstream tool explicitly requires it.

---

# 30. Make Cache Keys Architecture-Safe

The current Docker build uses persistent BuildKit cache mounts for source and build trees.

- [ ] Ensure `/work/build` cache identity differs by target.
- [ ] Ensure patched source cache can be shared only if it contains no generated target-specific files.
- [ ] Prefer named/ID'd caches such as:

```text
ov203-build-armv7
ov203-build-amd64
```

- [ ] Include target and relevant compiler/configuration values in cache invalidation where necessary.
- [ ] Test alternating builds:

```text
armv7 -> amd64 -> armv7 -> amd64
```

without `--no-cache`.

### Acceptance criteria

- [ ] No CMake cache from one target leaks into another target.

---

# 31. Check Base-Distro and Compiler Compatibility

- [ ] Verify OpenVINO 2020.3.2 builds cleanly with the compiler supplied by the chosen amd64 Bullseye image.
- [ ] If compiler issues appear, prefer a pinned compiler package/version over broad source modifications.
- [ ] Keep ARM and amd64 on the same distro release where practical to reduce unrelated differences.
- [ ] Verify runtime dependencies on amd64 using `ldd`.
- [ ] Record required packages such as `libusb-1.0-0`.
- [ ] Check whether `libudev` is dynamically required by the built XLink/MVNC stack on amd64.
- [ ] Confirm the final runtime image includes every dynamically linked library required by:
  - `hello_myriad`
  - `mobilenet_classify`
  - `libmyriadPlugin.so`
  - MVNC/XLink libraries

### Acceptance criteria

- [ ] `ldd` reports no `not found` entries in either runtime image.

---

# 32. Verify MYRIAD Plugin Registration on amd64

- [ ] Inspect generated/installed `plugins.xml` on amd64.
- [ ] Confirm MYRIAD plugin path resolves correctly.
- [ ] Confirm `InferenceEngine::Core` can instantiate without missing shared libraries.
- [ ] Run list-only mode with no device attached and distinguish:
  - plugin load succeeds but no hardware is found
  - plugin itself is missing/broken
- [ ] Add this distinction to diagnostics/output where practical.

### Acceptance criteria

- [ ] An amd64 build can load the MYRIAD plugin even before hardware testing.

---

# 33. Verify MA2450 Firmware Lookup on amd64

The plugin expects firmware near the plugin library.

- [ ] Use `strace`, debug logging, or MVNC/XLink logs if necessary to confirm the amd64 plugin searches the intended path.
- [ ] Verify firmware is not accidentally searched under a Pi-only path.
- [ ] Verify `dladdr()`-relative lookup works with the amd64 plugin install tree.
- [ ] Confirm firmware file permissions permit reading by the runtime user.

### Acceptance criteria

- [ ] Firmware upload starts without manually specifying an `.mvcmd` path in the application.

---

# 34. Avoid Introducing MA2Host into the Main Runtime

- [ ] Keep MA2Host/ma2boot experimentation out of the standard `run.sh` path.
- [ ] Do not use MA2Host to pre-boot the device before OpenVINO unless a separately documented diagnostic experiment explicitly requires it.
- [ ] Keep OpenVINO responsible for:
  - discovering the unbooted stick
  - uploading `usb-ma2450.mvcmd`
  - rediscovering the booted device
  - transferring the compiled network
  - inference I/O
- [ ] Retain any warning from existing project evidence that incompatible manual boot methods can leave the stick in a bad state requiring a replug.

### Acceptance criteria

- [ ] Normal MobileNet execution consists only of OpenVINO application + MYRIAD plugin stack.

---

# 35. Confirm Network Compilation Behavior on x86

- [ ] Run `myriad_compile` / `compile_tool` on amd64 without a stick if supported by the build.
- [ ] Compile the tiny IR.
- [ ] Compile MobileNet FP16.
- [ ] Compare generated blob metadata/size against ARM-generated output for informational purposes.
- [ ] Do not require byte-identical blobs unless there is evidence that OpenVINO should produce them identically.
- [ ] Verify both host architectures ultimately load a compatible graph onto the same MA2450 firmware/device.

### Acceptance criteria

- [ ] `LoadNetwork(..., "MYRIAD")` succeeds on amd64 using the self-built plugin and compiler stack.

---

# 36. Add User-Friendly Error Messages

- [ ] If Docker architecture does not match selected target, fail with the expected platform.
- [ ] If no USB device is present, say so without implying plugin build failure.
- [ ] If MYRIAD plugin cannot load, print the discovered OpenVINO lib path and `ldd` hints.
- [ ] If firmware is missing, name the expected `usb-ma2450.mvcmd` path.
- [ ] If USB re-enumeration times out, point to the Docker USB/network troubleshooting section.
- [ ] If the wrong host-runtime architecture is extracted, fail before executing it.

### Acceptance criteria

- [ ] Common errors are actionable without reading build logs.

---

# 37. Suggested Implementation Order

Use this order to minimize simultaneous variables and preserve the known-good Pi path.

## Phase A — Refactor without changing behavior

- [ ] Add centralized platform configuration.
- [ ] Make `build.sh` select `armv7` explicitly.
- [ ] Replace hard-coded `armv7l` library paths with discovery.
- [ ] Make Dockerfile toolchain argument conditional, but test only ARM first.
- [ ] Refactor `run.sh` to use `DOCKER_PLATFORM`.
- [ ] Run the complete existing Pi verification suite.

**Gate:** Pi results must still match the current known-good behavior before adding x86.

## Phase B — Add amd64 build support

- [ ] Add amd64 base image/platform mapping.
- [ ] Configure OpenVINO natively without the ARM toolchain.
- [ ] Resolve any x86-specific dependency/patch issues.
- [ ] Build `libmyriadPlugin.so`, `hello_myriad`, and `mobilenet_classify` for x86_64.
- [ ] Pass all hardware-free checks.

**Gate:** Native x86 runtime image is internally complete and MYRIAD plugin loads.

## Phase C — Hardware validation on x86

- [ ] Attach MA2450.
- [ ] Verify USB visibility.
- [ ] Run MYRIAD enumeration.
- [ ] Run tiny model.
- [ ] Run MobileNet.
- [ ] Compare reference output.
- [ ] Stress repeated firmware boot/re-enumeration.

**Gate:** x86 passes the same functional inference tests as Pi.

## Phase D — Host-native support

- [ ] Refactor `pull-runtime.sh`.
- [ ] Add native amd64 `host-run.sh` path.
- [ ] Validate library/ABI requirements.

**Gate:** only document this as supported after actual device inference passes outside Docker.

## Phase E — Documentation and CI

- [ ] Rewrite README around both platforms.
- [ ] Split historical logs by platform.
- [ ] Add CI matrix.
- [ ] Add architecture regression checks.

---

# 38. Definition of Done

The project can be considered genuinely multi-platform when all of the following are true:

- [ ] One repository supports both `armv7` and `amd64` targets.
- [ ] One common Dockerfile builds both targets.
- [ ] Raspberry Pi 5 still passes its existing smoke and MobileNet tests.
- [ ] Ubuntu/x86_64 builds OpenVINO 2020.3.2 from source.
- [ ] Ubuntu/x86_64 produces a native `libmyriadPlugin.so`.
- [ ] The x86 plugin uses OpenVINO's built-in MVNC/XLink stack rather than MA2Host.
- [ ] Both platforms package `usb-ma2450.mvcmd` beside the MYRIAD plugin.
- [ ] Both platforms enumerate the same physical MA2450 as `MYRIAD`.
- [ ] Both platforms successfully boot/re-enumerate the stick.
- [ ] Both platforms run the tiny inference model.
- [ ] Both platforms run the MobileNet example using the same IR/model corpus.
- [ ] MobileNet output matches the stored reference within the chosen tolerance.
- [ ] Common scripts contain no unconditional ARM-only runtime assumptions.
- [ ] ARM-specific loader/sysroot handling is isolated to the ARM host-native path.
- [ ] Architecture-specific Docker caches cannot contaminate each other.
- [ ] Documentation clearly distinguishes Pi-specific requirements from common Movidius/OpenVINO behavior.
- [ ] Build-only CI covers both architectures.
- [ ] Hardware verification results are recorded separately for Pi/ARM and Ubuntu/amd64.

---

# 39. Files Most Likely to Change

## Must change

```text
Dockerfile
build.sh
run.sh
container-entry.sh
scripts/host-run.sh
scripts/pull-runtime.sh
scripts/prepare-mobilenet.sh
scripts/verify.sh
README.md
HOWTO-HOST-RUN.md
HOWTO-MOBILENET.md
```

## Likely to change

```text
scripts/prepare-deps.sh
scripts/fetch-runtime.sh
scripts/validate-reference.sh
requirements-build.txt
smoke-test/CMakeLists.txt
mobilenet-test/CMakeLists.txt
docs/diagnostics/*.sh
docs/diagnostics/README.md
```

## Probably reusable unchanged or nearly unchanged

```text
smoke-test/main.cpp
mobilenet-test/main.cpp
smoke-test/model/make_tiny_ir.py
scripts/99-movidius-ncs2.rules
usb-ma2450.mvcmd firmware payload
OpenVINO source pin
```

## New files worth adding

```text
scripts/platform.sh
patches/README.md
platform/                 # optional alternative to scripts/platform.sh
  armv7.env
  amd64.env
ci/                       # optional
  verify-static.sh
```

---

# 40. First Concrete Refactoring Milestone

Before attempting an x86 build, make the current Pi build go through the new abstractions with **zero functional change**:

- [ ] Add `TARGET=armv7` platform configuration.
- [ ] Replace all common-script `linux/arm/v7` literals with `${DOCKER_PLATFORM}`.
- [ ] Replace Dockerfile `lib/armv7l` runtime references with discovered lib directory.
- [ ] Make CMake toolchain use conditional but select it for ARM.
- [ ] Target-qualify BuildKit build caches.
- [ ] Run current Pi verification.
- [ ] Compare resulting binaries, plugin discovery, firmware files, device enumeration, tiny inference, and MobileNet result against existing logs.

Only after this passes should `TARGET=amd64` be enabled. This isolates refactoring regressions from actual x86-porting problems and gives a clean baseline for the multi-platform work.

---

# ARM64 extension addendum

The initial multi-platform plan covered the established `armv7` path and a new `amd64` path. The project has since been extended with a native `arm64` target for Raspberry Pi 5/AArch64 Linux.

Implemented ARM64 integration items:

- [x] Add canonical `arm64` / `aarch64` target aliases in `scripts/platform.sh`.
- [x] Map `arm64` to Docker platform `linux/arm64` and an ARM64 Debian Bullseye base image.
- [x] Make `arm64` a native CMake build with no `armv7-native.toolchain.cmake`.
- [x] Validate ARM64 output as ELF64 / `AArch64` (`e_machine=183`).
- [x] Give ARM64 its own target-qualified BuildKit source/build caches and image tag.
- [x] Make Raspberry Pi 5 auto-detection select `arm64`; retain explicit `--platform armv7` as the compatibility fallback.
- [x] Support native ARM64 host-runtime extraction without an ARMHF sysroot.
- [x] Accept only an AArch64 host for `host-run.sh --platform arm64`.
- [x] Extend runtime verification and CI matrices to `armv7`, `arm64`, and `amd64`.
- [x] Extend low-level diagnostics and documentation to the ARM64 target.
- [x] Keep the MA2450 `.mvcmd` firmware shared across all host architectures.
- [ ] Validate the complete OpenVINO 2020.3.2 ARM64 build on physical Raspberry Pi 5 hardware.
- [ ] Validate MA2450 firmware boot, re-enumeration, tiny inference, and MobileNet on native ARM64.
- [ ] Retain ARM64 hardware logs under `logs/rpi5-arm64/` and compare them with the historical ARMv7 baseline.
