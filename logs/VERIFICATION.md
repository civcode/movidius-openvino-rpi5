# Verification report - OpenVINO 2020.3.2 / Movidius MYRIAD on Raspberry Pi 5

Generated: 2026-09-20T09:57:46+02:00   host: edge   image: openvino-2020.3-rpi5:latest

## 1. Host

```text
Linux edge 6.18.50+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.18.50-1+rpt1 (2026-09-11) aarch64 GNU/Linux
aarch64
arm64
               total        used        free      shared  buff/cache   available
Mem:           7.9Gi       1.9Gi       711Mi       1.2Gi       6.6Gi       5.9Gi
/dev/mmcblk0p2   29G   19G  9.0G  68% /
CONFIG_COMPAT: CONFIG_COMPAT=y
binfmt qemu-arm (little-endian arm32 emulation) present: no - 32-bit armv7 code must run natively on the aarch64 kernel
Movidius stick: Bus 003 Device 097: ID 03e7:2150 Intel Myriad VPU [Movidius Neural Compute Stick]
```

## 2. Docker / BuildKit

```text
Docker version 29.8.1, build 4a63305
server 29.8.1 linux/arm64
github.com/docker/buildx v0.37.1 0b265a9f62db554fa9aba6dd19e1bd5704bc7d8a
docker info: server=29.8.1 os=linux arch=aarch64 kernel=6.18.50+rpt-rpi-2712 cpus=4 containers=3 images=14 storage=overlayfs cgroup=systemd
image sha256:96cae438b33a80126a8b363bce8ed99146465b99c63740cf6986f1bc2c509004 linux/arm size=263647363
```

## 3. The container really runs 32-bit armv7 userspace

```text
OV install    : /opt/openvino (libs: /opt/openvino/inference_engine/lib/armv7l:/opt/openvino/ngraph/lib)
stick firmware: /opt/openvino/inference_engine/lib/armv7l/pcie-ma248x.mvcmd /opt/openvino/inference_engine/lib/armv7l/usb-ma2450.mvcmd /opt/openvino/inference_engine/lib/armv7l/usb-ma2x8x.mvcmd 
kernel   : aarch64          (the host kernel, arm32 processes run on it)
userspace: 32-bit
uname -P : unknown
hello_myriad ELF header:
   7f 45 4c 46 01 01 01 00 00 00 00 00 00 00 00 00
   03 00 28 00
  EI_CLASS=01 (ELF32), e_machine=28 00 (EM_ARM = 0x0028)
linked libraries (rpath resolved):
  	linux-vdso.so.1 (0xf718c000)
  	libinference_engine.so => /opt/openvino/inference_engine/lib/armv7l/libinference_engine.so (0xf6ff4000)
  	libinference_engine_legacy.so => /opt/openvino/inference_engine/lib/armv7l/libinference_engine_legacy.so (0xf6e00000)
  	libstdc++.so.6 => /usr/lib/arm-linux-gnueabihf/libstdc++.so.6 (0xf6cc4000)
  	libgcc_s.so.1 => /lib/arm-linux-gnueabihf/libgcc_s.so.1 (0xf6c98000)
  	libc.so.6 => /lib/arm-linux-gnueabihf/libc.so.6 (0xf6b98000)
  	libinference_engine_transformations.so => /opt/openvino/inference_engine/lib/armv7l/libinference_engine_transformations.so (0xf6aa8000)
  	libdl.so.2 => /lib/arm-linux-gnueabihf/libdl.so.2 (0xf6a94000)
  	libngraph.so => /opt/openvino/ngraph/lib/libngraph.so (0xf64dc000)
  	libm.so.6 => /lib/arm-linux-gnueabihf/libm.so.6 (0xf6474000)
  	libpthread.so.0 => /lib/arm-linux-gnueabihf/libpthread.so.0 (0xf644c000)
  	/lib/ld-linux-armhf.so.3 (0xf7190000)
```

## 4. Runtime image contents

```text
plugins shipped in plugins.xml:
plugin name="HETERO"
plugin name="MULTI"
plugin name="MYRIAD"
firmware next to libmyriadPlugin.so (getFirmwarePath uses dladdr):
/opt/openvino/inference_engine/lib/armv7l/pcie-ma248x.mvcmd
/opt/openvino/inference_engine/lib/armv7l/usb-ma2450.mvcmd
/opt/openvino/inference_engine/lib/armv7l/usb-ma2x8x.mvcmd
ngraph part of the install tree:
libinterpreter_backend.so
libngraph.so
```

## 5. Device enumeration through the Inference Engine

```text
>> docker run --rm --platform linux/arm/v7 --name ov203-smoke-320853 --network=host -v /dev:/dev --device-cgroup-rule=c 189:* rwm -e OV_ROOT=/opt/openvino openvino-2020.3-rpi5:latest --demo --list-only
OV install    : /opt/openvino (libs: /opt/openvino/inference_engine/lib/armv7l:/opt/openvino/ngraph/lib)
stick firmware: /opt/openvino/inference_engine/lib/armv7l/pcie-ma248x.mvcmd /opt/openvino/inference_engine/lib/armv7l/usb-ma2450.mvcmd /opt/openvino/inference_engine/lib/armv7l/usb-ma2x8x.mvcmd 

OpenVINO InferenceEngine smoke test
available devices : MYRIAD
  plugin MYRIAD   api 2.1 build custom__ (myriadPlugin)
RESULT: MYRIAD present
```

## 6. Compile + inference on the Movidius stick

```text
  USBDEVFS_RESET issued
RESULT: in boot mode 03e7:2150 at /dev/bus/usb/003/097
>> docker run --rm --platform linux/arm/v7 --name ov203-smoke-320941 --network=host -v /dev:/dev --device-cgroup-rule=c 189:* rwm -e OV_ROOT=/opt/openvino openvino-2020.3-rpi5:latest /opt/openvino/bin/hello_myriad --device MYRIAD --model /opt/openvino-demo/model/model.xml --weights /opt/openvino-demo/model/model.bin --iterations 10
OV install    : /opt/openvino (libs: /opt/openvino/inference_engine/lib/armv7l:/opt/openvino/ngraph/lib)
stick firmware: /opt/openvino/inference_engine/lib/armv7l/pcie-ma248x.mvcmd /opt/openvino/inference_engine/lib/armv7l/usb-ma2450.mvcmd /opt/openvino/inference_engine/lib/armv7l/usb-ma2x8x.mvcmd 
OpenVINO InferenceEngine smoke test
available devices : MYRIAD
  plugin MYRIAD   api 2.1 build custom__ (myriadPlugin)

reading model   : /opt/openvino-demo/model/model.xml + /opt/openvino-demo/model/model.bin
network         : tiny_conv_fp16
  input  input dims 1,3,32,32 -> FP16 NCHW
  output conv dims 1,8,32,32 FP32
loading on      : MYRIAD (first load compiles the graph on the VPU)
load+compile    : 1491.7 ms
  inference 1 : 1.82 ms
  inference 2 : 1.79 ms
  inference 3 : 1.84 ms
  inference 4 : 1.84 ms
  inference 5 : 1.75 ms
  inference 6 : 1.76 ms
  inference 7 : 1.76 ms
  inference 8 : 1.74 ms
  inference 9 : 1.77 ms
  inference 10 : 1.74 ms
mean inference  : 1.78 ms (560.91 fps)
  output conv (1,8,32,32, desc precision FP32, 4 B/elem) min -1.05859 max 2.64844 mean 1.02026
RESULT: PASS

after teardown the stick is back in ROM mode: Bus 003 Device 098: ID 03e7:2150 Intel Myriad VPU [Movidius Neural Compute Stick]
```

## 7. compile_tool from the same build (blob produced on the VPU)

```text
	Build .................. custom__
	Description ....... API
Done. LoadNetwork time elapsed: 1487 ms
2488440e8766d7053e84864058057e78  /out/verify_blob.bin
-rw-r--r-- 1 root root 1472 Sep 20 07:57 /out/verify_blob.bin
blob produced by the official Intel raspbian runtime (same IR, same -d MYRIAD):
2488440e8766d7053e84864058057e78  work/reference-check/blob/blob.bin
RESULT: in boot mode 03e7:2150 at /dev/bus/usb/003/099
```

## 8. Python policy (no Python libraries from APT)

```text
APT package lines in the Dockerfile (no python3-* library packages):
        build-essential
        cmake
        pkg-config
        patch
        git
        ca-certificates
        libusb-1.0-0-dev
        zlib1g-dev
        python3
        python3-venv
        libusb-1.0-0
        ca-certificates
        python3

python lines in the Dockerfile:
  50:        python3 \
  51:        python3-venv; \
  54:# Every Python *library* lives in this venv: no python3-* APT packages, no
  55:# system-wide pip installs.  python3-venv above is only the venv enabler.
  56:RUN python3 -m venv /opt/venv
  62:    python -c 'import sys; print("venv python:", sys.executable, sys.version)'
  205:    python3 /work/smoke/model/make_tiny_ir.py --outdir /work/smoke/model; \
  250:        python3; \
pinned pip libraries (requirements-build.txt):
  pip==21.3.1
  setuptools==59.6.0
  wheel==0.37.1
```

## 9. Vendor pin

```text
  openvino commit: 0b3773b7405d955d48642667ac5113289b9baab2
  tag: 2020.3.2
   cbe2db61a659c2cc304c3837406f95c39dfa938e inference-engine/thirdparty/ade (cbe2db6)
   1797d7fb712d2ffa6048ce5f5b3b560b84ab5ae4 ngraph (1797d7f)
```

## 10. MobileNet v2 converted by the vendored Model Optimizer, run on the stick

```text
corpus produced by ./scripts/prepare-mobilenet.sh:
     31675  labels/synset.txt
   6975776  mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.bin
     53064  mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.mapping
    137973  mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml
  13951520  mobilenet-v2-ov203/fp32/mobilenet-v2-ov203.bin
     53064  mobilenet-v2-ov203/fp32/mobilenet-v2-ov203.mapping
    138033  mobilenet-v2-ov203/fp32/mobilenet-v2-ov203.xml
  14246826  onnx/mobilenetv2-7.onnx
    602112  test_data/input_0.f32
      4000  test_data/output_0.f32
    plus vendor/models/images/*.ppm (224x224) and vendor/models/images/full/*.ppm

numerical check against the ONNX model zoo reference output:
>> docker run --rm --platform linux/arm/v7 --name ov203-smoke-321185 --network=host -v /dev:/dev --device-cgroup-rule=c 189:* rwm -e OV_ROOT=/opt/openvino -v /home/chris/workspace/flash-movidius-openvino/openvino-2020.3-rpi5/vendor/models:/models:ro openvino-2020.3-rpi5:latest /opt/openvino/bin/mobilenet_classify --device MYRIAD --model /models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml --weights /models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.bin --labels /models/labels/synset.txt --tensor /models/test_data/input_0.f32 --reference /models/test_data/output_0.f32 --iterations 10
model           : /models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml + /models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.bin
network         : mobilenet-v2-ov203
input           : data 1,3,224,224 FP16 NCHW
output          : mobilenetv20_output_flatten0_reshape0 1,1000
load+compile    : 1657.02 ms on MYRIAD
inference       : 44.52 ms mean over 10 run(s) (22.5 fps)
  max |diff|    : 0.0525  (reference logits up to 26.2702)
  mean |diff|   : 0.0122
  top-1         : device 556 vs reference 556
  top-1        : device 556  reference 556  = match
RESULT: PASS (top-1 matches, max |diff| 0.0525 vs tolerance 1.0000)

top-1 class per photo (ImageNet centre crop, FP16 weights):
dog        1.    444   0.5825  n02835271  bicycle-built-for-two, tandem bicycle, tandem  logit 12.141
cat        1.    281   0.2463  n02123045  tabby, tabby cat  logit 9.852
eagle      1.     21   0.8785  n01608432  kite  logit 16.359
banana     1.    954   0.7170  n07753592  banana  logit 13.211
cup        1.    968   0.9600  n07930864  cup  logit 16.109
```
