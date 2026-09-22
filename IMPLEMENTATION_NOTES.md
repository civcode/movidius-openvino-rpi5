# Multi-platform implementation notes

This tree implements the structural changes from `MULTI_PLATFORM_TODO.md` and extends the project to three host targets: `armv7`, `arm64`, and `amd64`.

## Implemented in source

- Central target mapping in `scripts/platform.sh`.
- `build.sh`, `run.sh`, runtime extraction and host-native execution accept `--platform armv7|arm64|amd64`.
- Auto-detection maps `aarch64`/`arm64` hosts to the native `arm64` target, `armv7l` to `armv7`, and `x86_64` to `amd64`.
- A single Dockerfile builds all three targets.
- Only `armv7` uses `toolchain/armv7-native.toolchain.cmake`; `arm64` and `amd64` configure natively.
- `arm64` uses `linux/arm64`, an ARM64 Debian Bullseye userspace, ELF64 AArch64 binaries, and ELF machine ID 183 validation.
- BuildKit source/build caches are target-qualified.
- OpenVINO library directories are discovered from the installed tree rather than assuming `armv7l`, `aarch64`, or `intel64`.
- ELF class/machine assertions are target-aware for both example applications and runtime verification.
- The same OpenVINO MYRIAD/MVNC/XLink path and pinned MA2450 firmware are kept on all targets; MA2Host is not used by the runtime.
- The runtime image contains a target manifest and firmware checksums.
- Host-native extraction is isolated: `armv7` uses the extracted ARMHF loader/sysroot; `arm64` and `amd64` execute native binaries directly with `LD_LIBRARY_PATH`.
- Model Optimizer uses a native container matching the host architecture.
- Platform-aware verification/static CI checks include all three target mappings.
- Manual build CI includes `arm`, `arm64`, and `amd64` targets via Buildx/QEMU where needed.

## ARM64 design

The ARM64 path is not an ARMv7 build renamed to `arm64`. It intentionally does not pass the ARMv7 toolchain. In an ARM64 container, Debian's native AArch64 GCC/G++ builds OpenVINO, `libmyriadPlugin.so`, XLink/MVNC, `hello_myriad`, and `mobilenet_classify` as AArch64 binaries. The MA2450 `.mvcmd` payload remains shared because it is device-side firmware.

On a Raspberry Pi 5, the intended preference order is:

1. `arm64` — native 64-bit userspace and host runtime.
2. `armv7` — retained compatibility fallback with the previously established ARMHF approach.

## Requires real hardware/environment validation

The repository can be refactored and statically checked here, but the following gates require the respective systems and an MA2450 stick and therefore are not claimed as completed by this change:

- Full OpenVINO 2020.3.2 source build for native ARM64.
- Native ARM64 MYRIAD enumeration and firmware upload on Raspberry Pi 5.
- Tiny-model and MobileNet inference on the physical stick using the ARM64 plugin.
- Full OpenVINO source build and physical-stick validation on ARMv7 and amd64 after this three-target extension.
- Host-native ARM64 inference outside Docker.
- Comparison of generated MYRIAD blobs and runtime timings across platforms.

Run `./scripts/verify.sh --platform <target>` on each hardware platform and keep new results separate from the historical Pi ARMv7 files already under `logs/`.

## Validation performed in this implementation environment

The hardware-free validation for this change includes:

- `ci/verify-static.sh` for `armv7`, `arm64`, and `amd64` mappings.
- `bash -n` over every active shell script.
- YAML parsing for both GitHub Actions workflows.
- CLI platform-selection/help smoke tests.
- Checks that ARM64 is native (`USE_CMAKE_TOOLCHAIN=0`, ELF64, e_machine 183, native host-runtime kind).
- CMake examples remain architecture-directory agnostic (`lib/*`).

Docker is not installed in the implementation environment, so the full OpenVINO Docker builds, runtime `ldd` checks, and physical MA2450 tests cannot be run here. Those are delegated to the manual build workflow and `docs/HARDWARE-TEST.md`; no ARM64 hardware result is fabricated.
