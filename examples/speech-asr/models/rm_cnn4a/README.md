# rm_cnn4a development fixture

`rm_cnn4a` is the initial known acoustic network used to develop the
speech-ASR infrastructure on OpenVINO 2020.3/MYRIAD.

## Purpose

It should allow us to build and validate:

- model acquisition/version pinning;
- model-specific feature preparation;
- FP16 IR loading;
- MYRIAD inference;
- input/output tensor transport;
- timing and device measurement;
- deterministic evaluation;
- experiment artifact collection;
- DESIGN/EXECUTE/EVALUATE/REVIEW handoff.

## Non-goals

The project does **not** assume that the eventual custom model should copy
rm_cnn4a's:

- frontend;
- acoustic targets;
- decoder;
- topology;
- training framework.

The future custom-model path is expected to be designed independently, with a
PyTorch/CUDA training loop and ONNX/OpenVINO export.

## Adapter boundary

Any frontend or output semantics required specifically by rm_cnn4a belong in
this model package/adapter. They must not redefine the global 16 kHz mono audio
contract or the AMI benchmark.

Before implementation, model acquisition must be pinned by source/version and
checksum, and the exact input/output tensor contract must be recorded from the
actual model artifact used by this repository.
