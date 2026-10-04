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
