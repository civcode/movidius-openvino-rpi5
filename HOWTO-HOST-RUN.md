# How to run OpenVINO 2020.3 + MobileNet on the Movidius stick **without Docker**

Beginner walkthrough for the host-only path: build once inside the container, copy the
runtime out, then run inference as a normal program on the Raspberry Pi 5.

Everything here is run from the project directory:

```bash
cd /home/chris/workspace/flash-movidius-openvino/openvino-2020.3-rpi5
```

Related guides: [`HOWTO-MOBILENET.md`](HOWTO-MOBILENET.md) (build + convert + classify in
Docker), [`logs/HOST-RUN.md`](logs/HOST-RUN.md) (the technical write-up),
[`logs/host-run-evidence.txt`](logs/host-run-evidence.txt) (raw evidence).

---

## 0. The idea in one picture

```
  ONCE, in Docker                                   EVERY TIME, on the host
┌───────────────────────────────┐    ┌──────────────────────────────────────┐
│ build the arm32v7 OpenVINO    │    │ no Docker, no container, no QEMU     │
│ 2020.3.2 runtime image  ──────┼──┐ │                                      │
│                               │  │ │  work/host-runtime/openvino/  <──────┼── the copied
│ convert ONNX ──▶ OpenVINO IR  │  │ │  work/host-runtime/sysroot/   <──────┼── runtime +
│ (python 3.8 container)        │  │ │  vendor/models/... (your IR)         │   its 10 libs
└───────────────────────────────┘  │ └──────────────────┬───────────────────┘
                                   │                    ▼
                                   └────────────▶  NCS2 stick on USB  (03e7)
```

* The **runtime** (plugins, `mvnc`, `XLink`, firmware `.mvcmd`, demo binaries) can leave the
  image and run directly on the host: the Pi 5 kernel is aarch64 with `CONFIG_COMPAT=y`, so
  it executes the 32-bit ARM binaries natively.
* The **Model Optimizer** stays in a container (OpenVINO 2020.3's MO needs Python 3.8; the
  host has Python 3.13).
* Nothing is installed on the host: no `libc6:armhf`, no multiarch, no QEMU.

---

## 1. Prerequisites

You already have the project working inside Docker (steps 1-5 of
[`HOWTO-MOBILENET.md`](HOWTO-MOBILENET.md)).  This guide only moves the result out.

| You need | Check |
| --- | --- |
| the built image | `docker images \| grep openvino-2020.3-rpi5` |
| the MobileNet IR (optional, for the demo) | `ls vendor/models/mobilenet-v2-ov203/fp16/` |
| a Pi 5 kernel with 32-bit compat | `zcat /proc/config.gz \| grep '^CONFIG_COMPAT='` |
| `sudo` | `sudo true` |
| membership of `plugdev` (for the USB rule) | `id -nG \| tr ' ' '\n' \| grep plugdev` |
| the stick plugged in | `lsusb \| grep 03e7` |

If the image or the IR is missing, do steps 1-5 of
[`HOWTO-MOBILENET.md`](HOWTO-MOBILENET.md) first (`./build.sh`, then
`./scripts/prepare-mobilenet.sh`); both need Docker.

Two things this guide deliberately does **not** do on the host:

* it installs no Python package (`pip install openvino`, `--break-system-packages`, APT
  `python3-*` libraries) - the host path is the extracted C++ runtime only;
* it installs no foreign-architecture Debian packages (see the FAQ).

The only system change is the one udev rule in step 3a.

> Troubleshooting, not a normal step: if `lsusb` shows **no** `03e7` device at all, the stick is in the wedged state described in
> [HOST-RUN.md §5](logs/HOST-RUN.md): unplug it, wait 10 s, plug it back in (on this host it
> also recovered by itself after about 20 minutes, but a replug is the reliable fix).

---

## 2. Step 1 - copy the runtime out of the image (once, ~30 s)

```bash
./scripts/pull-runtime.sh
```

Expected output:

```text
host runtime ready in /home/chris/workspace/flash-movidius-openvino/openvino-2020.3-rpi5/work/host-runtime
  sysroot  3.0M,  10 libraries
  openvino 34M
run it with:  ./scripts/host-run.sh list
```

What you now have (37 MB total):

```text
work/host-runtime/
├── sysroot/lib/ld-linux-armhf.so.3              the arm32v7 dynamic loader, from the image
├── sysroot/lib/arm-linux-gnueabihf/*.so*        glibc 2.31: libc libm libdl libpthread
│                                                librt libgcc_s
├── sysroot/usr/lib/arm-linux-gnueabihf/         libstdc++.so.6.0.28, libusb-1.0.so.0.3.0,
│                                                libudev.so.1.7.0
├── openvino/bin/hello_myriad                    the smoke-test binary
├── openvino/bin/mobilenet_classify              the MobileNet demo binary
├── openvino/inference_engine/lib/armv7l/        libmyriadPlugin.so, mvnc, XLink,
│                                                plugins.xml, the 3 .mvcmd firmwares,
│                                                compile_tool
├── openvino/ngraph/                             ngraph libs/headers/cmake
└── openvino-demo/model/                         the tiny hand-written IR
```

Re-run this script whenever you rebuild the image; it replaces `work/host-runtime/` from
scratch.

---

## 3. Step 2 - prepare the host (once, 2 minutes)

### 3a. Let a normal user open the stick

By default the USB device nodes are `crw-rw-r-- root root`, so an ordinary user can *see* the
stick but cannot drive the boot handshake (`ncDeviceOpen:1009 Failed to find booted device
after boot`).  Install the shipped udev rule:

```bash
sudo cp scripts/99-movidius-ncs2.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb --action=change
```

Verify (the stick must be plugged in):

```bash
ls -l /dev/bus/usb/*/* | grep 03e7   # after the stick has been opened once
# crw-rw---- 1 root plugdev 189, 267 /dev/bus/usb/003/012
```

The rule is `SUBSYSTEM=="usb", ATTR{idVendor}=="03e7", MODE="0660", GROUP="plugdev"`.
If you skip this step, prefix every command below with `sudo` - that works too.

### 3b. Pick one user for the `/tmp/mvnc.mutex` lock

`mvnc` opens the fixed file `/tmp/mvnc.mutex` with `O_CREAT` and `exit(1)`s if that fails.
Debian sets `fs.protected_regular = 2`, which means a file in world-writable `/tmp` created
by **another** user cannot be opened with `O_CREAT` - not even by root.  Practical rule:
**always run the host OpenVINO commands as the same user** (or run them all with `sudo`).

```bash
ls -l /tmp/mvnc.mutex 2>/dev/null     # does it exist, and who owns it?
rm -f /tmp/mvnc.mutex                 # if it belongs to someone else: delete it
```

`./scripts/host-run.sh` detects the bad case and warns instead of failing mysteriously.

---

## 4. Step 3 - first host run (1 minute)

```bash
./scripts/host-run.sh list
```

Expected (a container-free `hello_myriad`):

```text
OpenVINO InferenceEngine smoke test
available devices : MYRIAD
  plugin MYRIAD   api 2.1 build custom__ (myriadPlugin)
RESULT: PASS (no --model given: plugin enumeration only)
```

Then the tiny model and the numerical self-test:

```bash
./scripts/host-run.sh demo
./scripts/host-run.sh mobilenet
```

`demo` ends with

```text
load+compile    : 1490.74 ms
  inference 1 : 1.85 ms
mean inference  : 1.85 ms (540.67 fps)
RESULT: PASS
```

and the tail of `mobilenet` is

```text
  max |diff|    : 0.0525  (reference logits up to 26.2702)
  mean |diff|   : 0.0122
  top-1         : device 556 vs reference 556
  top-1        : device 556  reference 556  = match
  top-2        : device 818  reference 818  = match
  top-3        : device 811  reference 811  = match
  top-4        : device 827  reference 827  = match
  top-5        : device 918  reference 918  = match

RESULT: PASS (top-1 matches, max |diff| 0.0525 vs tolerance 1.0000)
```

Those are the same numbers you get from `./run.sh mobilenet` inside the container - the
container is not part of the result.

---

## 5. Step 4 - classify a photo on the host

```bash
./scripts/host-run.sh mobilenet --image vendor/models/images/banana.ppm --topk 3
```

```text
load+compile    : 1647.01 ms on MYRIAD
input source    : vendor/models/images/banana.ppm 224x224, mean 0.49,0.46,0.41 std 0.23,0.22,0.22
inference       : 44.55 ms mean over 1 run(s) (22.4 fps)
logits          : 1000 values, 4 B/elem, desc precision FP32

top 3:
   1.    954   0.7170  n07753592  banana  logit 13.211
   2.    923   0.0417  n07579787  plate  logit 10.367
   3.    684   0.0317  n03840681  ocarina, sweet potato  logit 10.094
```

Benchmark it:

```bash
./scripts/host-run.sh mobilenet --image vendor/models/images/cat.ppm --iterations 20
# inference       : 44.47 ms mean over 20 run(s) (22.5 fps)
IR=fp32 ./scripts/host-run.sh mobilenet          # use the FP32 weights instead
```

Notes: `--image` needs a 224×224 binary PPM (the corpus in `vendor/models/images/` already
is; how to make one is section 7 of [`HOWTO-MOBILENET.md`](HOWTO-MOBILENET.md)).  Unlike the
container mode, **all paths are host paths here** - there is no `/models` mount to remember.

---

## 6. Step 5 - compile the graph to a device blob (also host-native)

```bash
./scripts/host-run.sh compile \
    --model vendor/models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml \
    --blob    vendor/models/blobs/mobilenet-v2.blob
```

```text
blob generated successful
LoadNetwork time elapsed: 1657 ms
-rw-rw-r-- 1 chris chris 6999040 vendor/models/blobs/mobilenet-v2.blob
```

The blob is VPU code, not ARM code, so the host architecture is irrelevant.  But it is
pinned to the compiler **and** to the firmware that loads it, and OpenVINO's blob is
ELF-formatted (`7f 65 6c 66`) while NCSDK-era `.fby` graphs are not - so do not expect a
raw-XLink/MA2Host loader to accept it (untested here, see [HOST-RUN.md §4](logs/HOST-RUN.md)).

---

## 7. Optional: a shell with the binaries on `PATH`

```bash
./scripts/host-run.sh shell
```

```text
host-native OpenVINO 2020.3 shell: ov wrappers on PATH (hello_myriad,
mobilenet_classify, compile_tool).  LD_LIBRARY_PATH is exported too.
hostov# mobilenet_classify --model vendor/models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml \
         --weights vendor/models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.bin \
         --labels vendor/models/labels/synset.txt --image vendor/models/images/cup.ppm --topk 2
```

The wrappers in `work/host-runtime/bin/` just call the extracted loader for you, e.g.

```bash
work/host-runtime/bin/mobilenet_classify --help
```

or the long form:

```bash
work/host-runtime/sysroot/lib/ld-linux-armhf.so.3 --library-path \
  work/host-runtime/sysroot/lib/arm-linux-gnueabihf:\
work/host-runtime/sysroot/usr/lib/arm-linux-gnueabihf:\
work/host-runtime/openvino/inference_engine/lib/armv7l:\
work/host-runtime/openvino/ngraph/lib \
  work/host-runtime/openvino/bin/mobilenet_classify --help
```

---

## 8. Why this works without QEMU (and how to prove it)

The binaries are ARM 32-bit hard-float:

```bash
readelf -hl work/host-runtime/openvino/bin/mobilenet_classify | grep -E 'Class:|Machine:|interpreter'
#   Class:  ELF32
#   Machine: ARM
#   [Requesting program interpreter: /lib/ld-linux-armhf.so.3]
```

That interpreter does not exist on the host, which is exactly why the loader is invoked
explicitly.  The kernel runs the ELF32 binary in its ARM compat mode (`CONFIG_COMPAT=y`), and
no translator is involved - you can check that no `qemu-arm` binfmt handler is registered:

```bash
ls /proc/sys/fs/binfmt_misc/ | grep -c '^qemu-arm$'      # -> 0
zcat /proc/config.gz | grep '^CONFIG_COMPAT='            # -> CONFIG_COMPAT=y
dpkg-query -W -f='${Status}\n' libc6:armhf               # -> not installed
```

And the binary really is the host executing it:

```bash
work/host-runtime/sysroot/lib/ld-linux-armhf.so.3 --library-path <the four dirs> \
    work/host-runtime/openvino/bin/mobilenet_classify
# --model is required   <- printed by main(), so it ran
```

Performance confirms it too: 44.5 ms for MobileNet v2 on the stick is the same as the native
container run; a TCG translator would be far slower.

---

## 9. Cheat sheet

| Goal | Command |
| --- | --- |
| copy the runtime out (once) | `./scripts/pull-runtime.sh` |
| list devices, host process | `./scripts/host-run.sh list` |
| tiny model on the stick | `./scripts/host-run.sh demo` |
| MobileNet numerical self-test | `./scripts/host-run.sh mobilenet` |
| classify a photo | `./scripts/host-run.sh mobilenet --image vendor/models/images/banana.ppm --topk 3` |
| benchmark | `./scripts/host-run.sh mobilenet --image vendor/models/images/cat.ppm --iterations 20` |
| use FP32 weights | `IR=fp32 ./scripts/host-run.sh mobilenet` |
| compile a device blob | `./scripts/host-run.sh compile --model …/fp16/….xml --blob vendor/models/blobs/mobilenet-v2.blob` |
| shell with binaries on PATH | `./scripts/host-run.sh shell` |
| help | `./scripts/host-run.sh --help` |
| USB permission rule (once) | `sudo cp scripts/99-movidius-ncs2.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=usb --action=change` |
| convert a new ONNX model | still `./scripts/prepare-mobilenet.sh` (needs Docker) |
| reset the stick to ROM | `sudo python3 scripts/reset-stick.py` |

---

## 10. Troubleshooting

| What you see | Cause / fix |
| --- | --- |
| `work/host-runtime is not populated yet` | first call of `host-run.sh` runs `./scripts/pull-runtime.sh` for you; if that fails, the image `openvino-2020.3-rpi5:latest` is missing → `./build.sh` |
| `available devices : ` (empty) | the MYRIAD plugin found no stick → plug it in, check `lsusb \| grep 03e7` |
| `ncDeviceOpen:1009 Failed to find booted device after boot` as a normal user | USB nodes are `root:root` → install the udev rule (step 3a) or use `sudo` |
| `ncDeviceOpen:754 global mutex initialization failed` (even under `sudo`) | `/tmp/mvnc.mutex` belongs to a different user and `fs.protected_regular=2` blocks `O_CREAT` → `rm /tmp/mvnc.mutex` or use one user consistently |
| `error while loading shared libraries: libm.so.6` | you called the loader with the wrong `--library-path`; use `./scripts/host-run.sh` (or its `bin/` wrappers) instead of typing the loader line by hand |
| `hint: host-side stick state -> no 03e7 device visible` | the stick is not enumerated at all (wedged) → physical replug, then `sudo python3 scripts/reset-stick.py` if it comes back as `03e7:f63b` |
| `RESULT: FAIL - no MYRIAD plugin (…does the container have --device=/dev/bus/usb?)` | ignore the "container" wording: host-side, it means the plugin could not open the stick - see the two rows above |
| Model Optimizer fails when you try to convert on the host | host Python is 3.13; MO 2020.3 needs 3.8 → use `./scripts/prepare-mobilenet.sh` (Docker) |
| `RESULT: PASS` is missing / `max|diff|` is huge | wrong IR directory or `--reference` file: check `vendor/models/mobilenet-v2-ov203/<fp16\|fp32>/` |

---

## 11. FAQ

**Do I still need Docker at all?**
For two things only: building the runtime image and running the Model Optimizer. After that,
`work/host-runtime/` + `vendor/models/` are ordinary files you can run from any shell, cron
job, or systemd service.

**Do I have to keep the running container alive?**
No. Inference involves no container: `host-run.sh` forks the arm32 binary directly.

**Should I just `apt install libc6:armhf` instead?**
You could (`libc6:armhf libstdc++6:armhf libusb-1.0-0:armhf libudev1:armhf`), and the binaries
would then run without the loader wrapper - but that puts foreign-architecture packages on the
host. The extracted sysroot leaves the host untouched, which is why this project uses it.

**Is the host run faster than the container run?**
The same: 44.54 ms vs 44.4-44.6 ms, byte-identical results. The stick, not the host CPU, is
the bottleneck.

**Do I need `--network=host` here?**
No. That flag fixes a *container* problem (libusb never sees the uevent that announces the
re-enumerated stick). On the host you are already in the uevent namespace.

**Can I copy `work/host-runtime/` to another Pi 5?**
Yes - it is self-contained, as long as the target kernel has `CONFIG_COMPAT=y` and you apply
the udev rule there. A pure aarch64-only kernel without compat support cannot run these
ELF32 binaries.

**Can the host run the stick and the container run the stick at the same time?**
No. One process owns the USB device, and the `/tmp/mvnc.mutex` lock is shared between both
paths.

---

## 12. Evidence

* [`logs/HOST-RUN.md`](logs/HOST-RUN.md) - the write-up: extraction, measurements, the two
  host-side requirements (with the `fs.protected_regular` reproduction), the blob format
  finding, and the hazard section.
* [`logs/host-run-evidence.txt`](logs/host-run-evidence.txt) - kernel config, binfmt check,
  `readelf`, the executed binary, the sysroot file list, the blob header, the run numbers.
* Memory of this project: `README.md` section *Running the artifacts directly on the host*.
