# Measured results — OpenVINO 2020.3.2 built on a Raspberry Pi 5 for the Movidius NCS2

All measurements were taken on host `edge` (Raspberry Pi 5, Raspberry Pi OS 13/trixie,
kernel 6.18.50+rpt-rpi-2712, 8 GB RAM + 2 GB swap), with the Intel NCS2
(`03e7:2150`) in the same USB port (`3-1`) every time.

## 1. What was built

| item | value |
|---|---|
| source | `openvino` tag `2020.3.2`, commit `0b3773b7405d955d48642667ac5113289b9baab2` |
| submodules | `ngraph` @ `1797d7fb…`, `inference-engine/thirdparty/ade` @ `cbe2db61…` |
| container | `arm32v7/debian:bullseye` (gcc 10.2.1, cmake 3.18.4, python 3.9.2), executed natively on the aarch64 kernel |
| target as seen by CMake | `Linux-192.168.0.129-armv7l`, `ARCH=armv7l`, `ARM=ON`, `THREADING=SEQ` |
| plugins built | MYRIAD, HETERO, MULTI, TEMPLATE (no CPU plugin, no GNA/GPU) |
| image | `openvino-2020.3-rpi5:latest` — 251 MB unpacked / 75.6 MB compressed content |
| installed runtime | 41 MB (`deployment_tools/inference_engine/lib/armv7l` + `deployment_tools/ngraph`) |

## 2. Build timings (Pi 5, `BUILD_JOBS=2`, ccache-less)

Measured on a cold cache (empty `/work/build`), which is what `./build.sh` does the
first time or after `--no-cache`:

| stage | wall time |
|---|---|
| `build-deps`: apt (snapshot.debian.org) + venv + conversion toolchain | 77 s |
| `configure`: copy + patch the tree, then CMake configure | 28 s |
| `builder`: compile + install with `-j2` | **748 s (12.5 min)** |
| `smoke`: tiny IR + `hello_myriad` build | 4 s |
| `runtime`: final image assembly | 2 s |

Re-running `./build.sh` with everything already built: **16 s** - `make` is invoked
again but finds the tree current (incremental), and only the image layers are rebuilt.

`BUILD_JOBS=2` never came close to the memory limit (peak RSS observed well under 2 GB;
swap untouched). `-j4` is safe on this 8 GB board.

## 3. Patches that were required

| patch | why |
|---|---|
| `patches/0001-cross-compile-skip-host-protoc.patch` | `inference_engine/CMakeLists.txt` downloads `protoc-3.7.1-linux-x86_64.tar.gz` on any Linux host; configure aborts with `add_custom command depends on file ... protoc which does not exist`. The gate is narrowed to amd64 hosts, so the arm64/aarch64 host builds `protoc` from the vendored sources instead. |
| `patches/0002-ade-gcc10-no-error-warnings.patch` | the vendored `ade` turns `-Werror` on unconditionally; gcc 10 reports `error: redundant move in return statement [-Werror=redundant-move]` in `execution_engine.cpp:141`, so ade never compiles. The patch keeps `-Werror` and downgrades the gcc-10-specific diagnostics. |

Reproduced on purpose with `./build.sh --no-patches --target configure` (it builds the
vendor tree without `patches/`, and uses a different `SYNC_STAMP` so the source cache is
re-synced).  Configure then aborts exactly as upstream would on this host:

```
-- Copying from local folder /work/deps/protoc-3.7.1-linux-x86_64.tar.gz to
   /work/stage/download/protoc-3.7.1-linux-x86_64.tar.gz ...
CMake Error at cmake/download/download_and_check.cmake:51 (file):
  file COPY cannot find "/work/deps/protoc-3.7.1-linux-x86_64.tar.gz": No such file or directory
CMake Error at cmake/download/extract.cmake:16 (message):
  error: file to extract does not exist: '/work/stage/download/protoc-3.7.1-linux-x86_64.tar.gz'
```

(log in `logs/no-patches-configure.log`; the host-protoc download is only attempted
because the host is not amd64 - on an x86_64 machine upstream works, which is why this
failure never appears in Intel's own docs).

## 4. Key CMake cache values after configure

```
CMAKE_SYSTEM_NAME:STRING   = Linux
CMAKE_SYSTEM_PROCESSOR:STRING = armv7l
ARCH:STRING                = armv7l
ARM:BOOL                   = ON
ENABLE_VPU:BOOL            = ON
ENABLE_MYRIAD:BOOL         = ON
ENABLE_MYRIAD_NO_BOOT:BOOL = OFF
ENABLE_MKL_DNN:BOOL        = OFF     (no CPU plugin)
ENABLE_GNA / ENABLE_CLDNN  = OFF
THREADING:STRING           = SEQ
NGRAPH_BUILD_DIR:STRING    = /work/src/bin/armv7l/Release/lib
PLUGIN_FILES               = MYRIAD:myriadPlugin;HETERO:HeteroPlugin;MULTI:MultiDevicePlugin;TEMPLATE:templatePlugin
CMAKE_INSTALL_PREFIX       = /work/stage
IE_PATH_TO_DEPS            = /work/deps   (zero downloads during the build)
```

## 5. Reference-runtime validation first (before our own build)

`./scripts/validate-reference.sh` runs the *official* Intel
`l_openvino_toolkit_runtime_raspbian_p_2020.3.355` inside the same arm32v7 container
against the stick. Once the container USB issue was fixed (see §7):

```
compile_tool -m model.xml -d MYRIAD -ip FP16 -o /out/blob/blob.bin
  Build ......... 2020.3.2-3506-c35b42b1d89-releases/2020/3
  Done. LoadNetwork time elapsed: 1484 ms
  blob size: 1472 bytes
```

So the host, the stick, the firmware and the IR were all proven good before a single line of
our own build was trusted.

## 6. Standalone mvnc/XLink probe (root-cause evidence)

`docs/diagnostics/xlink_probe.c` boots the stick through `XLinkBootRemote()` and then
enumerates devices again. With a private container network namespace:

```
BEFORE any_state(ANY)  n=1  name='1-ma2450'  protocol=X_LINK_USB_VSC  state=BOOTED
  XLinkBootRemote(1-ma2450) -> X_LINK_SUCCESS   (161 ms)
AFTER  any_state(ANY)  n=1  name='1-ma2450'  protocol=X_LINK_USB_VSC  state=BOOTED
AFTER  booted(ANY)     n=0
```

`countAfter == countBefore` and the name still contains `ma2450` — the exact condition
`mvnc_api.c:1011` reports as `Device (1-ma2450) doesn't disappear`.

With `--network=host`:

```
BEFORE any_state(ANY)  n=1  name='1-ma2450'
  XLinkBootRemote(1-ma2450) -> X_LINK_SUCCESS
AFTER  any_state(ANY)  n=1  name='1-'       protocol=X_LINK_USB_VSC  state=BOOTED
AFTER  booted(ANY)     n=1
```

The stick re-enumerates as `1-` (the booted VSC name) and `mvnc` accepts it. This is a
libusb property, not an OpenVINO property: libusb 1.0.24 keeps a cached device list that is
maintained from netlink `KO_UEVENT` hotplug broadcasts; a container in its own network
namespace never receives them, and `libusb_get_device_list()` has no rescan entry point.

## 7. Container flag matrix (same probe, same port, stick reset before each row)

| flags | result |
|---|---|
| `--privileged --network=host` | **FAIL** — `/dev/bus/usb` holds the static nodes created at container start; no re-enumeration at all |
| `--device=/dev/bus/usb:/dev/bus/usb --device-cgroup-rule='c 189:* rwm' --network=host` | **FAIL** — same static-node problem |
| `-v /dev:/dev --device-cgroup-rule='c 189:* rwm' --network=host` | **OK** ← the flags `run.sh` uses |
| `-v /dev:/dev --device-cgroup-rule='c 189:* rwm'` (no `--network=host`) | **FAIL** — libusb never learns the device changed |
| `-v /dev:/dev --network=host` (no cgroup rule) | **FAIL** — device node present but `/dev/bus/usb/003/022: open failed` |
| `--privileged --network=host -v /dev:/dev` | OK (works, but grants far more than needed) |

`run.sh` therefore runs:

```
docker run --rm --platform linux/arm/v7 --network=host \
    -v /dev:/dev --device-cgroup-rule='c 189:* rwm' ...
```

## 8. Final run with the self-built runtime

`sudo ./scripts/reset-stick.py` then `./run.sh`:

```
OV install    : /opt/openvino (libs: /opt/openvino/inference_engine/lib/armv7l:/opt/openvino/ngraph/lib)
stick firmware: pcie-ma248x.mvcmd usb-ma2450.mvcmd usb-ma2x8x.mvcmd

available devices : MYRIAD
  plugin MYRIAD   api 2.1 build custom__ (myriadPlugin)

network         : tiny_conv_fp16
  input  input dims 1,3,32,32 -> FP16 NCHW
  output conv dims 1,8,32,32 FP32
load+compile    : 1479.92 ms      (1485 ms with compile_tool)
  inference 1 : 1.89 ms
  inference 2 : 1.78 ms
  inference 3 : 1.76 ms
mean inference  : 1.81 ms (551.9 fps)
  output conv (1,8,32,32, desc precision FP32, 4 B/elem) min -1.05859 max 2.64844 mean 1.02026
RESULT: PASS
```

A 20-iteration run: **mean 1.78 ms (562.7 fps)**.
`compile_tool` from our own build produces a **1472-byte** HCL1 blob — the same size the
official 2020.3.2 runtime produces for the same IR, which is a good sign that the compiler
path is identical.

After the run the stick returns to ROM mode (`03e7:2150`) on its own, exactly like the
official runtime.

## 9. Re-verification with the final image

After the last Dockerfile change (`./build.sh` end to end, then
`sudo ./scripts/reset-stick.py && ./run.sh`):

```
load+compile    : 1493.89 ms
  inference 1/2/3 : 1.84 / 1.76 / 1.76 ms      mean 1.79 ms (559.45 fps)
RESULT: PASS
after teardown  : bus 003 03e7:2150   (stick back in ROM mode by itself)
```

Run-to-run spread over four runs was 1.78-1.81 ms mean inference and 1480-1494 ms load+compile.

## 10. Two cache behaviours worth knowing

* BuildKit keys a layer on the **resolved instruction text**, not on the contents of a
  `type=cache` mount.  A changed `SYNC_STAMP` therefore re-syncs `/work/src` but does
  **not** by itself invalidate the `make` layer.  Both the configure and the builder RUN
  now reference `${SYNC_STAMP}` and assert
  `test "$(cat /work/src/.sync_stamp)" = "${SYNC_STAMP}"`, so the compile layer is
  invalidated whenever the source/patch stamp changes, and a mismatched cache aborts
  instead of silently reusing a stale tree.
* `# syntax=docker/dockerfile:1.4` is resolved through the registry.  On a flaky DNS the
  build can fail before any stage runs:
  `failed to fetch anonymous token: Get "https://auth.docker.io/token...`: pre-pull it once
  (`docker pull docker.io/docker/dockerfile:1.4`) and the problem disappears.

## 11. Comparison with the official reference runtime

| | official 2020.3.2 raspbian runtime | this build |
|---|---|---|
| plugin version string | `2020.3.2-3506-c35b42b1d89-releases/2020/3` | `custom__` (untracked git describe) |
| load + compile | 1493 ms | 1480–1485 ms |
| mean inference (3 iterations) | 1.81 ms | 1.81 ms |
| mean inference (20 iterations) | — | 1.78 ms |
| blob for the tiny IR | 1472 B, md5 `2488440e8766d7053e84864058057e78` | 1472 B, md5 `2488440e8766d7053e84864058057e78` (byte-identical) |
| firmware files | 3 × `.mvcmd` next to `libmyriadPlugin.so` | same (copied by `vpu_copy_firmware`) |

The self-built runtime performs the same as Intel's raspbian package on this board.

## 12. One-shot verification report

`./scripts/verify.sh > logs/VERIFICATION.md` regenerates the whole evidence chain in a
single file (host facts, no `qemu-arm` binfmt handler, ELF32/EM_ARM + `ld-linux-armhf.so.3`
inside the container, plugin list, MYRIAD enumeration, compile + inference on the stick,
`compile_tool` blob md5 versus Intel's blob, the APT/pip Python policy, the vendor pin).
Latest run: load+compile 1490.31 ms, mean inference 1.77 ms (564.06 fps), `RESULT: PASS`,
stick back in ROM mode (`03e7:2150`) after teardown.
