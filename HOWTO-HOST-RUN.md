# Run the built OpenVINO/Movidius runtime directly on the Linux host

Docker is the primary runtime, but this project can extract the built artifacts and run them without Docker at inference time on all three supported targets.

## 1. Build the matching image

Raspberry Pi 5, preferred native ARM64:

```bash
./build.sh --platform arm64
```

Raspberry Pi compatibility fallback:

```bash
./build.sh --platform armv7
```

x86_64 Ubuntu/Linux:

```bash
./build.sh --platform amd64
```

## 2. Extract the runtime

```bash
./scripts/pull-runtime.sh --platform arm64
# or
./scripts/pull-runtime.sh --platform armv7
# or
./scripts/pull-runtime.sh --platform amd64
```

The targets coexist:

```text
work/host-runtime/armv7/
  openvino/
  openvino-demo/
  sysroot/                 # ARMHF loader/libs only
  host-runtime.env

work/host-runtime/arm64/
  openvino/
  openvino-demo/
  host-runtime.env

work/host-runtime/amd64/
  openvino/
  openvino-demo/
  host-runtime.env
```

The extracted OpenVINO library directory is discovered from `openvino/inference_engine/lib/<arch>`; scripts do not assume the upstream directory name.

## 3. Run it

Pi 5 native ARM64:

```bash
./scripts/host-run.sh --platform arm64 list
./scripts/host-run.sh --platform arm64 demo
./scripts/host-run.sh --platform arm64 mobilenet
```

Pi ARMv7 fallback:

```bash
./scripts/host-run.sh --platform armv7 list
./scripts/host-run.sh --platform armv7 demo
```

x86_64:

```bash
./scripts/host-run.sh --platform amd64 list
./scripts/host-run.sh --platform amd64 demo
./scripts/host-run.sh --platform amd64 mobilenet
```

Compile an IR to a MYRIAD blob, for example on ARM64:

```bash
./scripts/host-run.sh --platform arm64 compile \
  --model vendor/models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml \
  --blob work/mobilenet.blob
```

## How the host paths differ

### arm64 on Raspberry Pi 5

The image and host CPU are both AArch64. No foreign loader or ARMHF sysroot is used. `host-run.sh` executes the extracted ELF64 AArch64 binaries directly and sets `LD_LIBRARY_PATH` to the extracted Inference Engine and nGraph libraries.

### armv7 compatibility path

The image contains ARMHF binaries. On a 64-bit Pi kernel, `pull-runtime.sh` extracts `/lib/ld-linux-armhf.so.3` and the required ARMHF system libraries. `host-run.sh` invokes binaries through that loader. This preserves the original known-good no-QEMU design.

### amd64 on x86_64

The image and host CPU are both x86_64. No foreign loader or sysroot is used. `host-run.sh` executes the extracted binaries directly with the matching OpenVINO library path.

If host glibc compatibility is a problem for either native target, use `run.sh`/Docker instead; it is the more deterministic path.

## Safety checks

`host-run.sh` reads `host-runtime.env` and refuses obvious target/host mismatches. It also checks that the discovered OpenVINO lib directory contains both `libmyriadPlugin.so` and `usb-ma2450.mvcmd`.

For ARM64, `pull-runtime.sh` verifies the extracted application as ELF64/AArch64 when host `readelf` is available. `scripts/verify.sh` additionally validates both the application and plugin by parsing their ELF headers inside the runtime image.

The MVNC stack uses `/tmp/mvnc.mutex`. If that file was created by another user with restrictive permissions, `host-run.sh` warns before attempting inference.

## USB permissions

Host-native execution needs normal-user USB permissions. Install the supplied rule:

```bash
sudo cp scripts/99-movidius.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo usermod -aG plugdev "$USER"
```

After logging in again, inspect:

```bash
lsusb | grep -i -E '03e7|myriad'
stat -c '%A %U:%G %n' /dev/bus/usb/*/*
```

## Remove extracted runtimes

They are derived artifacts and can be recreated at any time:

```bash
rm -rf work/host-runtime/armv7
rm -rf work/host-runtime/arm64
rm -rf work/host-runtime/amd64
```
