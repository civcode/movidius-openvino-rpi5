# MobileNet v2 on the Movidius stick

End-to-end demo on top of the runtime built by this project: a real network
(mobilenetv2-7) converted with the **vendored OpenVINO 2020.3.2 Model Optimizer**
and run on the NCS2 by an arm32v7 binary against the self-built libraries.

Generated: 2026-09-20 (host `edge`, Raspberry Pi 5, stick `03e7:2150` -> `03e7:f63b`)

Artifacts: `logs/mobilenet-run.log` (every command with its full output),
`logs/mo-fp16-transcript.txt` (Model Optimizer), `logs/mobilenet-build.log` (the Docker
build including the new stage), section 10 of `logs/VERIFICATION.md`.

Reproduce with two commands:

```bash
./scripts/prepare-mobilenet.sh      # download + convert + build the corpus
./run.sh mobilenet                  # numerical verification on the stick
./run.sh mobilenet --image /models/images/cat.ppm
```

---

## 1. Source model

| item | value |
| --- | --- |
| model | `mobilenetv2-7` from the ONNX model zoo (`validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx`) |
| tarball | `https://s3.amazonaws.com/download.onnx/models/opset_7/mobilenetv2-7.tar.gz` |
| tarball sha256 | `b463ad62dae99f13afd88549ca7d43e9bda6876614f3592ebb41177e1db0fcc5` |
| onnx sha256 | `c1c513582d56afceff8516c73804e484c81c6a830712ab6d682253f4a3cd042f` (14 246 826 B) |
| opset / ir_version | 7 / 6, producer `pytorch 1.8` |
| input | `data`-equivalent `input`, 1 x 3 x 224 x 224, float32 |
| output | `mobilenetv20_output_flatten0_reshape0`, 1 x 1000 logits |
| documented accuracy | Top-1 70.94 %, Top-5 89.99 % |
| documented preprocessing | RGB, NCHW, x/255, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, resize short side to 256, center crop 224, softmax |
| labels | `synset.txt` from the same zoo directory (1000 lines); the first word matches `https://pytorch.org/hub/torchvision_v2_classification/` class order 0/1000 mismatches |

Why opset 7 and not opset 10: `mobilenetv2-10.onnx` is exported with a dynamic batch
dimension (`batch_size` as a `dim_param`), and Model Optimizer 2020.3 stops shape
propagation at the final matmul:

```text
[ ERROR ] Failed to infer shape for Gemm_104/WithoutBiases: Partial inference ended with error
Stopped shape/value propagation at "Gemm_104/WithoutBiases" node.
```

`mobilenetv2-7.onnx` has static shapes and converts cleanly.

## 2. Conversion with the vendored Model Optimizer 2020.3.2

`scripts/prepare-mobilenet.sh` stages `vendor/openvino-2020.3.2/model-optimizer` into
`work/mo-2020.3` **without its unit tests** (897 of 1175 `.py` files) because
`mo/utils/import_extensions.py` imports every `.py` file it finds, and the vendored
`*_test.py` files import a test-only helper (`generator`) that is not part of the tree.

The converter runs in a **native arm64** `python:3.8-slim` container: Model Optimizer
is pure Python for the ONNX frontend, the Pi 5 kernel runs arm64 containers natively,
so no emulation layer is involved and no armhf wheel exists to be missing.  Pinned
libraries: `numpy==1.21.6 networkx==2.6.3 protobuf==3.19.6 defusedxml==0.7.1 onnx==1.12.0`.

```bash
python /mo/mo.py --input_model /models/onnx/mobilenetv2-7.onnx \
    --output_dir /models/mobilenet-v2-ov203/fp16 --model_name mobilenet-v2-ov203 \
    --data_type FP16
```

OpenVINO 2020.3 Model Optimizer has no `--compress_to_fp16`; the FP16 weights are
produced with `--data_type FP16` (`mo/utils/cli_parser.py`: `FP16/FP32/half/float`).
Full transcript of a fresh run: `logs/mo-fp16-transcript.txt`.

```text
Model Optimizer version: 	unknown version

[ SUCCESS ] Generated IR version 10 model.
[ SUCCESS ] XML file: /models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml
[ SUCCESS ] BIN file: /models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.bin
[ SUCCESS ] Total execution time: 14.35 seconds.
[ SUCCESS ] Memory consumed: 148 MB.
```

| artifact | FP32 variant | FP16 variant |
| --- | --- | --- |
| `mobilenet-v2-ov203.xml` | 138 033 B | 137 973 B |
| `mobilenet-v2-ov203.bin` | 13 951 520 B | 6 975 776 B |
| `mobilenet-v2-ov203.mapping` | 53 064 B | 53 064 B |

## 3. What the converted IR contains

266 layers, 275 edges, IR version 10, no custom extensions:

```text
fp16: {'Const': 109, 'Add': 63, 'Convolution': 37, 'ReLU': 36,
       'GroupConvolution': 17, 'Parameter': 1, 'ReduceMean': 1,
       'Reshape': 1, 'Result': 1}
fp32: identical layer types and count
```

```text
Parameter  data                                   1,3,224,224  element_type f16 (fp32 variant: f32)
Result     mobilenetv20_output_flatten0_reshape0  1,1000
```

Notes:

* Batch normalization is folded into `Mul`/`Add` pairs by MO (hence 63 `Add` layers and
  no `BatchNorm`), and the depthwise separable convolutions stay as `GroupConvolution`.
* This export uses plain `ReLU`, not `Clamp`; both exist in the 2020.3 IR frontend
  (`ie_format_parser.cpp` registers `ClampLayer`), which is what the tiny smoke-test
  model uses.
* `ReduceMean` (global average pooling) **is** accepted by the MYRIAD compiler of this
  build - the model compiles without any manual decomposition.

## 4. Numerical verification against the model zoo's own reference

The zoo ships `test_data_set_0/input_0.pb` (1 x 3 x 224 x 224 float32, random values)
and `test_data_set_0/output_0.pb` (1 x 1000 float32 logits).  Those are dumped as raw
little-endian files (`vendor/models/test_data/input_0.f32`, `output_0.f32`) and fed to
the stick through the demo binary, which converts the input to the precision the IR
declares and reads the output blob by measured element size (MYRIAD reports FP32).

```bash
./run.sh mobilenet --tensor /models/test_data/input_0.f32 \
    --reference /models/test_data/output_0.f32 --iterations 25
```

| IR | load + compile | inference (25 runs) | max abs diff | mean abs diff | top-5 identical? |
| --- | --- | --- | --- | --- | --- |
| FP16 weights | 1646 ms | 44.50 ms (22.5 fps) | 0.0525 | 0.0122 | yes |
| FP32 weights | 1696 ms | 44.40 ms (22.5 fps) | 0.0671 | 0.0148 | yes |

Largest reference logit is 26.2702, so the worst deviation is 0.2 % of the maximum.
Reference and device agree on every one of the top five classes:

```text
   1.    556   0.9999  n03347037  fire screen, fireguard        logit 26.219
   2.    818   0.0000  n04286575  spotlight, spot               logit 15.211
   3.    811   0.0000  n04265275  space heater                  logit 14.969
   4.    827   0.0000  n04330267  stove                         logit 14.336
   5.    918   0.0000  n06785654  crossword puzzle, crossword   logit 14.156

RESULT: PASS (top-1 matches, max |diff| 0.0525 vs tolerance 1.0000)
```

(The input is random noise, so the class names are meaningless - what matters is that
the numbers agree with ONNX's own reference run.)

## 5. Real photographs

`vendor/models/images/<name>.ppm` is the documented ImageNet preprocessing (short side
-> 256, center crop 224).  `vendor/models/images/full/<name>.ppm` is the same photo
squashed to 224 x 224, kept because these are detection photos where a center crop can
cut the subject out.  Top-1 with its softmax probability:

| photo | source | center crop | full-frame squash |
| --- | --- | --- | --- |
| `cat` | BVLC/caffe `examples/images/cat.jpg` | **tabby, tabby cat** 0.246 (tiger cat 0.183) | kit fox 0.291, red fox 0.284 |
| `dog` | darknet `data/dog.jpg` (two dogs **and a bicycle**) | tandem bicycle 0.583, malamute 0.104 | miniature schnauzer 0.173, standard schnauzer 0.165 |
| `eagle` | darknet `data/eagle.jpg` | kite 0.879, **bald eagle** 0.093 | **bald eagle** 0.519, kite 0.437 |
| `banana` | Wikimedia `Bananavarieties.jpg` | **banana** 0.717 (plate 0.042) | **banana** 0.956 |
| `cup` | Wikimedia `Coffee_at_Christmas_(Unsplash).jpg` | **cup** 0.960, espresso 0.011 | **cup** 0.836, espresso 0.074 |

Every run took 44.5 ms of inference.  The two "wrong" answers are both explainable and
are a property of the input, not of the toolchain: the crop of `dog.jpg` is dominated by
the bicycle in the middle of that photo, and the crop of `eagle.jpg` keeps mostly sky;
with the whole frame visible the eagle is top-1.  This is the usual MobileNet-v2
crop-sensitivity behaviour, and it matches the model zoo's stated 70.9 % top-1.

## 6. Demo binary

`mobilenet-test/main.cpp` -> `/opt/openvino/bin/mobilenet_classify` (ELF32 ARM, built by
the `mobilenet` stage of the Dockerfile against the install tree, linked with
`-Wl,--start-group` over `libinference_engine`, `..._legacy`, `..._transformations`,
`..._lp_transformations` and `libngraph`):

```text
mobilenet_classify --model <xml> [--weights <bin>] [--device MYRIAD]
    (--tensor <raw f32 1x3x224x224> | --image <224x224 P6 PPM>)
    [--labels <synset.txt>] [--reference <raw f32 1x1000>] [--tol 1.0]
    [--mean r,g,b] [--std r,g,b] [--topk 5] [--iterations N]
```

* no OpenCV, no extra Python libraries: the image reader is a 40-line binary PPM (P6)
  parser, the normalization is the mean/std above, and FP32 -> FP16 conversion is done
  in the program with round-to-nearest-even, subnormal and overflow handling.
* the output blob is interpreted by `byteSize() / size()` (4 B/elem on MYRIAD) instead
  of trusting the declared precision, same as the smoke test.
* exit codes: `0` ok, `2` no MYRIAD device, `3` model/IO failure, `4` bad arguments,
  `5` numerical comparison failed.

The full command log is `logs/mobilenet-run.log`; a missing input file exits 3 with
`ERROR: cannot open <path>` printed after the report (stdout is tied to stderr).

Useful invocations:

```bash
./run.sh mobilenet                                   # reference comparison, 1 iteration
./run.sh mobilenet --image /models/images/banana.ppm --topk 3
IR=fp32 ./run.sh mobilenet                           # the FP32-weight IR instead
./run.sh mobilenet --image /models/images/full/eagle.ppm
```

The corpus stays on the host and is mounted read-only at `/models` by `./run.sh
mobilenet`; the Docker image itself still contains no model data (75.6 MB content).

## 7. Cost summary

| step | cost |
| --- | --- |
| Model Optimizer conversion (per precision) | 14.35 s, 148 MB peak |
| `mobilenet_classify` compile time on MYRIAD | ~1.65-1.70 s |
| MobileNet v2 inference on the stick | 44.4-44.6 ms (22.4-22.5 fps) |
| tiny smoke-test model (for scale) | 1.78 ms (562 fps) |
| FP32 vs FP16 weights at runtime | identical: the VPU computes in FP16 internally, only the IR size differs (13.95 MB vs 6.98 MB) |
