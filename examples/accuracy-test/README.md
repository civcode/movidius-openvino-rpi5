# MobileNet v2 accuracy test (known-content images, on the MA2450)

`accuracy_test.py` checks that the MYRIAD MobileNet v2 pipeline actually
classifies correctly, by running a set of images with **known content**
through the same `mobilenet_server` backend the webcam example uses and
counting top-1 / top-5 hits.

## Dataset

[EliSchwartz/imagenet-sample-images](https://github.com/EliSchwartz/imagenet-sample-images):
1,000 JPEGs, exactly **one representative image per ImageNet-1k class**.
Each file is named `<synset>_<label>.JPEG` (e.g. `n02085620_Chihuahua.JPEG`),
so the ground truth is the synset in the file name, mapped to a class index
via `vendor/models/labels/synset.txt`.  No manual labeling involved.

Note: these are (research-licensed) ImageNet validation images; keep the
dataset inside this repo checkout and do not redistribute it.

## Usage

```bash
# once: download the 1000 images into examples/accuracy-test/images/
./examples/accuracy-test/fetch-sample-images.sh

# full run (all 1000 images, ~1 minute of stick time)
python3 examples/accuracy-test/accuracy_test.py

# quick subset, fp32 model, docker backend
python3 examples/accuracy-test/accuracy_test.py --limit 100 --ir fp32 --backend docker

# CI-style gate: fail unless top-1 >= 65%
python3 examples/accuracy-test/accuracy_test.py --fail-under 65
```

## What to expect

Verified on a MA2450 stick with the converted 2020.3 IR (both precisions):

| backend                          | top-1 | top-5 |
|---------------------------------|-------|-------|
| ONNX model, onnxruntime (CPU)   | 81.3% | 95.9% |
| 2020.3 IR, OpenVINO 2026 CPU    | 81.3% | 95.6% |
| **2020.3 IR, MYRIAD (fp16)**    | **81.4%** | **95.9%** |
| **2020.3 IR, MYRIAD (fp32)**    | **81.4%** | **95.9%** |

The MYRIAD numbers matching the CPU references within noise confirms the
whole pipeline (preprocessing, conversion, server protocol, VPU execution).
**A score far below ~80% is not a model problem — it means the pipeline is
broken** (preprocessing, tensor layout, sign handling, or output handling),
which is exactly what this test exists to catch.

Reported numbers:

* `top-1 accuracy` — predicted class equals the file-name synset
* `top-5 accuracy` — the truth synset is inside the 5 highest logits
* `throughput` — classified images per second in steady state
* up to `--show-errors` top-1 misses with the expected and predicted labels
  (a `[also a top-5 miss]` tag marks the rarer misses that failed top-5 too)

The test reuses the shared client in `examples/mobilenet_client.py`
(same binary stdio protocol as the webcam example) and the identical
preprocessing as `mobilenet-test/main.cpp`.

## Bug this test caught

The first full run scored only **41.9% top-1 / 65.9% top-5** on MYRIAD while
the same model and preprocessing scored 81.3% on CPU.  Instrumenting the
server showed every *negative* value in the input tensor arriving at the
network **positive**: a copy of `floatToHalf` in `mobilenet_server.cpp`
was missing the sign bit in two return paths.  The conversion now lives in
the repo-root `half.hpp` (shared, with a device-free unit test
`examples/webcam/test_half.cpp` that runs in the Docker build stage), and
the MYRIAD accuracy above was measured with the fix in place.
