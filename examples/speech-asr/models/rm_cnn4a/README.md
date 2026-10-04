# rm_cnn4a development fixture

`rm_cnn4a_smbr` is the initial known acoustic network used to develop the
OpenVINO/MYRIAD inference and measurement path.

## Important boundary

This public Intel package is **not a raw-audio end-to-end ASR model**. Its
development fixture is:

```text
Kaldi feature ARK -> rm_cnn4a acoustic network -> acoustic score ARK
```

The package ships `feat1_10.ark` and `score1_10.ark`, which make it useful
for deterministic CPU/MYRIAD numerical regression. It does not ship the full
raw-audio feature-transform/decoder bundle required to turn arbitrary AMI WAV
clips into text.

Full Kaldi WER decoding requires additional resources such as `HCLG.fst`,
`final.mdl`, and `words.txt`. Those are outside the first
hardware-qualification path.

Therefore:

- **AMI remains the canonical end-to-end dataset** for future custom models;
- **rm_cnn4a is the device/conversion regression fixture**;
- no guessed MFCC/log-mel frontend is attached to rm_cnn4a;
- vendor score error, not AMI WER, is the first acceptance metric for this model.

## Source and conversion

The source descriptor is `source-v1.json`. It pins the Intel
`models_contrib/speech/2021.2` release and the exact expected filenames.

Prepare it with:

```bash
./scripts/prepare-rm-cnn4a.sh
```

Intel's storage index does not publish SHA-256 values, so first acquisition
creates `vendor/models/rm_cnn4a_smbr/source/SOURCE-LOCK.sha256`; every
subsequent run verifies that content lock. The generated model spec records all
source and IR hashes.

Conversion uses the repository's pinned OpenVINO 2020.3.2 Model Optimizer with
the Kaldi frontend, removes the output softmax, and emits FP16 IR. The script
writes `ir-contract.json` by inspecting the actual generated IR instead of
hard-coding tensor names or dimensions.

## Historical MYRIAD constraint

Intel's published NCS2 test used the FP16 IR and batch size 1. That report also
showed substantial score divergence versus other targets, so numerical
comparison against `score1_10.ark` is a required metric.

## Non-goals

The eventual custom model does not inherit rm_cnn4a's frontend, acoustic
targets, decoder, topology, or training framework. The custom path remains
PyTorch/CUDA -> ONNX -> OpenVINO -> MYRIAD.


## Runtime regression

After rebuilding the runtime image with this checkpoint:

```bash
./build.sh --platform arm64
./run.sh --platform arm64 speech-regress
```

The Docker build includes the pinned OpenVINO 2020.3 `speech_sample` only; it
does not enable the full sample suite. The wrapper forces `MYRIAD`, FP16 IR and
batch size 1, and compares generated acoustic scores with the vendor
`score1_10.ark` reference.


## CPU reference regression

Phase 4 now has a hardware-independent OpenVINO reference path on amd64. It
uses the exact FP16 IR plus `feat1_10.ark` and `score1_10.ark` that the
MYRIAD path consumes:

```bash
./build.sh --platform amd64
./scripts/python.sh examples/speech-asr/evaluation/benchmark_rm_cnn4a.py \
    --backend cpu
```

The result is written to
`work/speech-asr/rm_cnn4a/result-cpu-reference.json`. The same evaluator can
later run the device path with `--backend myriad --platform arm64`.

Prepared bundles can be checked without network, Docker, or hardware:

```bash
./scripts/prepare-rm-cnn4a.sh --verify-only
```

That verification re-hashes the source lock and source artifacts, re-inspects
the generated IR, and verifies that `model-spec.json`, `ir-contract.json`,
the XML and the BIN all agree. The generated model spec also records the source
lock and license hashes so a completed reference run can be frozen from actual
evidence.


## Frozen Phase 4 CPU reference

The first completed amd64 CPU reference run is frozen in
`cpu-reference-v1.json`. It uses OpenVINO 2020.3.2, the FP16 IR identified by
XML SHA-256 `53a22c26746eeedf053864385606d64b20ce6c4020732a2a55f8ce2609152f27`
and BIN SHA-256
`a27a0f9ebf4a235e6789d746940ba07941c236b5930efdafdf8ffd771be79195`.

Observed reference results:

- 10 utterances / 3401 frames;
- 0 failures;
- weighted mean inference: 2.021289091443693 ms/frame;
- utterance p50: 2.019845 ms/frame;
- utterance p95: 2.0324755000000003 ms/frame;
- maximum absolute score error: 0.00880814;
- mean average error: 0.00137728;
- mean RMS error: 0.001708947;
- model load time: 150.739 ms.

These are evidence from the frozen run, not acceptance thresholds for MYRIAD.
Phase 5 should compare the device result against this exact artifact/fixture
identity and report the measured deltas.
