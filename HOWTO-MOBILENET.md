# MobileNet v2 on the Movidius MA2450

The MobileNet example is shared by ARMv7, native ARM64, and native x86_64 builds. It uses the OpenVINO 2020.3 C++ Inference Engine API; loading the network with device `MYRIAD` causes `libmyriadPlugin.so` to compile/deploy the graph and boot the stick through MVNC/XLink.

## 1. Build a runtime

Preferred Raspberry Pi 5 path:

```bash
./build.sh --platform arm64
```

Compatibility Pi path:

```bash
./build.sh --platform armv7
```

Ubuntu/x86_64:

```bash
./build.sh --platform amd64
```

## 2. Prepare the model corpus

```bash
./scripts/prepare-mobilenet.sh
```

This downloads the pinned ONNX MobileNet v2 model and reference tensors, stages the vendored OpenVINO 2020.3 Model Optimizer, creates FP32/FP16 IRs and prepares sample images.

Model Optimizer is Python code and runs in a native container matching the host: `linux/arm64` on a Pi 5 and `linux/amd64` on x86_64. That host-side conversion choice is independent of whether inference uses `arm64` or the `armv7` compatibility target.

Output is under:

```text
vendor/models/mobilenet-v2-ov203/fp16/
vendor/models/mobilenet-v2-ov203/fp32/
vendor/models/test_data/
vendor/models/labels/
vendor/models/images/
```

## 3. Numerical reference test on the stick

Pi 5 native ARM64:

```bash
./run.sh --platform arm64 mobilenet
```

ARMv7 fallback:

```bash
./run.sh --platform armv7 mobilenet
```

x86_64:

```bash
./run.sh --platform amd64 mobilenet
```

With no additional arguments the wrapper supplies the ONNX model-zoo input and reference output tensors to `mobilenet_classify`.

## 4. Classify an image

The runtime sees `vendor/models` mounted at `/models`:

```bash
./run.sh --platform arm64 mobilenet --image /models/images/cat.ppm
./run.sh --platform arm64 mobilenet --image /models/images/banana.ppm --topk 3
IR=fp32 ./run.sh --platform arm64 mobilenet
```

Use the same command shape with `armv7` or `amd64`.

## 5. Host-native mode

```bash
./scripts/pull-runtime.sh --platform arm64
./scripts/host-run.sh --platform arm64 mobilenet
./scripts/host-run.sh --platform arm64 mobilenet \
  --image vendor/models/images/banana.ppm --topk 3
```

Host-native paths are host filesystem paths rather than `/models/...` container paths.

## What uploads firmware and the network?

The example does not call MA2Host. Its relevant flow is:

```text
InferenceEngine::Core
  -> ReadNetwork(xml, bin)
  -> LoadNetwork(network, "MYRIAD")
  -> libmyriadPlugin.so
  -> MVNC / XLink
  -> usb-ma2450.mvcmd + compiled graph
  -> MA2450
```

The `.mvcmd` firmware sits beside `libmyriadPlugin.so` in the OpenVINO install tree on every host architecture.

## Troubleshooting

First separate build/package problems from USB hardware problems:

```bash
./scripts/verify.sh --platform arm64 --no-device
```

or use `armv7`/`amd64` as appropriate. Then check `lsusb`, udev permissions and Docker re-enumeration settings documented in `README.md`.

## Timing evidence

Only the original Raspberry Pi ARMv7 measurements are historical evidence in this repository. Native ARM64 and amd64 results remain pending physical hardware validation.

| Host/runtime | MobileNet FP16 load + compile | MA2450 inference | Evidence |
|---|---:|---:|---|
| Raspberry Pi 5 / ARMv7 | 1656.92 ms | 44.40 ms mean over 25 runs (22.5 fps) | `logs/mobilenet-run.log` |
| Raspberry Pi 5 / ARM64 | **pending hardware validation** | **pending hardware validation** | capture under `logs/rpi5-arm64/` |
| Ubuntu x86_64 / amd64 | **pending hardware validation** | **pending hardware validation** | capture under `logs/ubuntu-amd64/` |

Use identical models, firmware and iteration counts when adding comparison results.
