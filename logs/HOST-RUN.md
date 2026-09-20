# Running the artifacts directly on the host (no Docker at inference time)

**Answer: yes.** The Model Optimizer conversion and the MYRIAD compilation stay inside
containers, the arm32v7 runtime plus the IR are copied out once, and inference then runs as
an ordinary host process.  Measured on `edge` (Pi 5, kernel
`6.18.50+rpt-rpi-2712`, `CONFIG_COMPAT=y`) on 2026-09-20.

## 1. What was extracted

`./scripts/pull-runtime.sh` (image → `work/host-runtime/`, 37 MB total):

| Path | Contents |
| --- | --- |
| `sysroot/lib/ld-linux-armhf.so.3` | the arm32v7 dynamic loader, taken from the image |
| `sysroot/lib/arm-linux-gnueabihf/`, `sysroot/usr/lib/arm-linux-gnueabihf/` | the only 10 non-OpenVINO libraries the binaries need: glibc 2.31 (`libc/libm/libdl/libpthread/librt/libgcc_s`), `libstdc++.so.6.0.28`, `libusb-1.0.so.0.3.0`, `libudev.so.1.7.0` (3.0 MB) |
| `openvino/` | `== image /opt/openvino`: `bin/hello_myriad`, `bin/mobilenet_classify`, `inference_engine/lib/armv7l/{libmyriadPlugin.so, mvnc, XLink, plugins.xml, compile_tool, usb-ma2450.mvcmd, usb-ma2x8x.mvcmd, pcie-ma248x.mvcmd}`, `ngraph/` (34 MB) |
| `openvino-demo/model/` | the tiny hand-written IR (`model.xml` + `model.bin`) |

No host packages are installed: no `apt install libc6:armhf`, no dpkg multiarch, no QEMU
binfmt handler.  (The multiarch alternative - `sudo apt install libc6:armhf
libstdc++6:armhf libusb-1.0-0:armhf libudev1:armhf` and then running the binaries directly -
should also work, since Debian 13 arm64 and Bullseye armhf share glibc symbol versions; the
extracted sysroot was chosen to leave the host untouched.  Untested here.)  The aarch64 kernel runs the ELF32 hard-float binaries in compat mode, and
because `PT_INTERP` (`/lib/ld-linux-armhf.so.3`) does not exist on the host, the loader is
invoked explicitly:

```bash
work/host-runtime/sysroot/lib/ld-linux-armhf.so.3 \
    --library-path <sysroot libs>:<openvino lib dirs> \
    work/host-runtime/openvino/bin/mobilenet_classify ...
```

`./scripts/host-run.sh` wraps that (and generates `work/host-runtime/bin/` wrappers so the
binaries can also be called by name inside `./scripts/host-run.sh shell`).

Raw evidence for all of the above (kernel config, `binfmt_misc` check, `readelf`, the executed
arm32 binary, the sysroot file list, the blob header): [`host-run-evidence.txt`](host-run-evidence.txt).

## 2. Measured on the stick, host-native (no container involved)

```bash
./scripts/host-run.sh list
./scripts/host-run.sh demo                            # tiny model
./scripts/host-run.sh mobilenet                       # numerical self-test
./scripts/host-run.sh mobilenet --image vendor/models/images/banana.ppm --topk 3
./scripts/host-run.sh compile --model vendor/models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml \
                              --blob  vendor/models/blobs/mobilenet-v2-ov203.blob
```

| Item | Container (`./run.sh`) | Host (`./scripts/host-run.sh`) |
| --- | --- | --- |
| device enumeration | `MYRIAD` | `MYRIAD` |
| tiny model (`demo`) | 1.78 ms | 1.83 ms / 547 fps |
| load + compile | 1646-1696 ms | 1657 ms (`compile_tool -d MYRIAD`, host-native) |
| inference (FP16 IR) | 44.40-44.57 ms | **44.54 ms** (banana photo, 22.5 fps) |
| numerical check vs ONNX zoo reference | `RESULT: PASS`, max `abs|diff|` 0.0525, mean 0.0122, top-5 `556,818,811,827,918` | **`RESULT: PASS`, max 0.0525, mean 0.0122, identical top-5** |
| MYRIAD blob (`compile_tool`) | - | 6 999 040 B written in 2.1 s, host-native |

`--network=host` and the uevent problem disappear on the host: the host *is* the uevent
namespace, so the plugin sees the post-boot device (`03e7:f63b`) immediately.

Every command above was executed twice - once before and once after the stick recovered from
the wedge in §5 - with full transcripts in [`host-run-evidence.txt`](host-run-evidence.txt)
(see its `RE-VERIFIED` block).  Beginner walkthrough of the same path:
[`../HOWTO-HOST-RUN.md`](../HOWTO-HOST-RUN.md).

## 3. Two host-side requirements

### 3.1 Write access to the USB device nodes

`/dev/bus/usb/NNN/NNN` is `crw-rw-r-- root:root` by default, so an unprivileged run gets
past enumeration and then fails in the boot path:

```text
E: [ncAPI] ncDeviceOpen:1009  Failed to find booted device after boot
ERROR: Can not init Myriad device: NC_ERROR
```

Fix (this is what Intel's own NCSDK installer used to do) - install the shipped rule and
re-trigger (the running user must be in `plugdev`):

```bash
sudo cp scripts/99-movidius-ncs2.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=usb --action=change
stat -c '%A %U:%G %n' /dev/bus/usb/003/*     # -> crw-rw---- root:plugdev
```

After that `./scripts/host-run.sh mobilenet` works as a normal user.  Without the rule, run
it under `sudo`.

### 3.2 `/tmp/mvnc.mutex` must belong to the user who runs it

`mvnc` (OpenVINO's bundled NCSDK) creates a fixed lock file with
`open("/tmp/mvnc.mutex", O_CREAT, 0660)` in `ncDeviceOpen`
(`inference-engine/thirdparty/movidius/mvnc/src/mvnc_api.c:745-756`) and `exit(1)`s if that
`open` fails.  Debian sets `fs.protected_regular = 2`, and on this host that restriction
applies **even to root**: a regular file in world-writable `/tmp` that another user created
and that is not readable by "other" cannot be opened with `O_CREAT`.  Reproduced natively:

```text
$ umask 017; : > /tmp/armtest2          # -rw-rw---- chris chris
$ sudo /tmp/ctest                        # open("/tmp/armtest2", O_RDONLY|O_CREAT, 0660)
open O_RDONLY|O_CREAT -> fd=-1 Permission denied
$ /tmp/ctest                             # same user, same file
open O_RDONLY|O_CREAT -> fd=3 ok
```

Symptom when it happens:

```text
E: [ncAPI] ncDeviceOpen:754  global mutex initialization failed
```

Rule: run all host-native OpenVINO invocations as one user, or clear/relax the file
(`rm /tmp/mvnc.mutex` / `chmod 0666 /tmp/mvnc.mutex`).  Container runs are unaffected - the
container has its own `/tmp`.  `scripts/host-run.sh` prints a warning when it detects the
mismatch.

## 4. Copying only the compiled blob out (device-side artifacts)

`compile_tool -d MYRIAD` works without a stick attached and also runs host-native, so the
VPU artifact can be produced anywhere:

```bash
./scripts/host-run.sh compile --model .../mobilenet-v2-ov203.xml --blob vendor/models/blobs/mobilenet-v2-ov203.blob
# Done. LoadNetwork time elapsed: 1657 ms   ->  6999040 bytes
```

The blob is **not** host-architecture dependent (it is VPU code), but it is *compiler and
firmware* dependent: it starts with `7f 65 6c 66` (ELF) - the Myriad X graph image - while
the NCSDK-era `mvNCCompile` `.fby` graphs used in the MA2Host experiments start with
`00 00 01 00 02 00 01 00`.  So a host-side loader that talks raw XLink/USB (see the
`flash-movidius` MA2Host tool) would need the matching firmware (`usb-ma2x8x.mvcmd` from
this 2020.3 package) and the ELF-graph format; the existing `ma2host blob/run` commands were
validated against `mvNCCompile` graphs only.  **Not verified end-to-end** - and see §5.

## 5. Hazard found while testing this (stick left wedged)

Attempting to boot the stick with OpenVINO's own `.mvcmd` through the MA2Host boot path
(`flash-movidius/tools/ma2boot .../usb-ma2450.mvcmd`, and `ma2host connect <fw>`) pushed the
firmware but wedged the stick's USB interface:

```text
usb 3-1: device descriptor read/64, error -71
usb 3-1: Device not responding to setup address.
usb usb3-port1: unable to enumerate USB device
```

After that the device disappears from `lsusb` and no software recovery on this board worked:
`scripts/reset-stick.py` (`USBDEVFS_RESET`, `--power`), unbind/rebind of `xhci-hcd.1`
(`/sys/bus/platform/drivers/xhci-hcd/`), and xhci runtime-PM toggling.  It needs a physical
replug (or a host reboot).  Conclusion: do not mix the two boot paths - let OpenVINO boot
the stick, and use MA2Host's own paired firmware for MA2Host experiments.

Recovery outcome: the stick re-appeared on its own about 20 minutes later (`03e7:2150` again,
no reboot, no replug on this host), and every host-native command in this document was then
re-run successfully - `list`, `demo` (1494 ms compile, 1.83 ms / 547 fps),
`mobilenet` (44.60 ms, `RESULT: PASS`, max |diff| 0.0525, identical top-5),
`mobilenet --image …/banana.ppm` (44.51 ms, banana 0.7170), `mobilenet --iterations 20`
(44.46 ms mean over 20 runs) - see the *RE-VERIFIED* block of
[`host-run-evidence.txt`](host-run-evidence.txt).  Do not count on that self-recovery.

## 6. What this does *not* avoid

* **Model Optimizer still needs a Python 3.8 container.** The host runs Python 3.13, and
  MO 2020.3 targets 3.6-3.8 (`onnx==1.12.0`/`numpy==1.21.6` wheels do not exist for 3.13).
  Conversion (`scripts/prepare-mobilenet.sh`) stays in `python:3.8-slim`.
* **The build still needs the arm32v7 container** unless OpenVINO 2020.3 is rebuilt
  natively for aarch64 (nothing in `mvnc`/`XLink` CMake gates on CPU arch -
  `inference-engine/thirdparty/movidius/{mvnc,XLink,CMakeLists.txt}` contain no
  `CMAKE_SYSTEM_PROCESSOR` conditions at all - so it is plausible, but untested: it would
  need its own run of the patch set (`patches/0001`, `patches/0002`) and its own
  verification, and the CPU plugin would still have to be switched off).
* The runtime is still 32-bit ARM code; the host just executes it natively.
