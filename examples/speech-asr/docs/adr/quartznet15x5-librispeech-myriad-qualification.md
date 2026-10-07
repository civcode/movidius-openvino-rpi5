# ADR: qualify the clean QuartzNet reference on Pi 5 + MA2450/MYRIAD

Status: implemented; physical measurement pending
Date: 2026-10-07
Reference model: `quartznet15x5_nvidia_ref`

## Context

The pinned NVIDIA Multidataset QuartzNet15x5Base-En source recognizer is
qualified on the full LibriSpeech `dev-clean` corpus:

- PyTorch WER `0.037939781625675524`;
- CER `0.012506494000177396`;
- 2,703 / 2,703 utterances;
- exact original 29-class CTC head, blank index 28;
- zero training;
- PyTorch -> ONNX frame argmax agreement `1.0`.

Listening review established that LibriSpeech clean read speech is materially
closer to the intended product audio quality than AMI meeting speech. AMI
therefore remains a stress/domain benchmark rather than the primary deployment
quality target.

The repository has already demonstrated the related v19 QuartzNet topology on
Pi 5 + MA2450, but the exact clean 29-class reference model has not yet been
physically qualified.

## Decision

Port the already-qualified source recognizer unchanged through:

`PyTorch -> ONNX -> OpenVINO 2020.3 FP16 -> Pi 5 host runtime -> MA2450/MYRIAD`.

No training, vocabulary expansion, AMI fine-tuning, or beam/LM decoder is part
of this qualification. Greedy CTC remains the reference decoder.

## Frontend on Pi

The historical NeMo frontend is reproduced in a dependency-light NumPy module
for the Pi. It preserves:

- 16 kHz mono input;
- preemphasis 0.97;
- centered 512-point STFT;
- non-periodic 320-sample Hann window;
- 160-sample hop;
- 64 Slaney-normalized mel bins;
- additive `2^-24` log guard;
- per-feature sample-standard-deviation normalization (`ddof=1`);
- zero invalid frames;
- source time-axis padding to a multiple of 16.

The NumPy implementation must first requalify all 2,703 dev-clean utterances.
Its WER may differ from the qualified Torch frontend WER by at most `0.001`
absolute.

## Exact-time OpenVINO execution

The qualified source frontend is variable-length. Forcing every utterance into
v19's 512-frame input would truncate long clips and would no longer be the
qualified recognizer.

The deployment therefore exports two ONNX artifacts from the same untouched
29-class weights:

1. a dynamic-time ONNX used as the semantic reference;
2. a fixed 512-frame carrier ONNX converted by the pinned OpenVINO 2020.3 Model
   Optimizer to FP16 IR.

The carrier shape is not an inference policy. Before each MYRIAD network load,
`hello_myriad --reshape-time T` calls OpenVINO 2020.3
`CNNNetwork::reshape` so the input time axis exactly equals that utterance
group's source-frontend padded length.

Utterances are grouped only when their exact padded lengths are identical.
They are not truncated, extended to larger buckets, or otherwise changed.

One persistent MYRIAD session and one unmeasured warmup are used per exact time
shape. This makes the qualification slower to load than a production fixed
shape, but preserves source semantics.

## Semantic parity gate

Before accepting corpus accuracy, the first physical sample is compared against
the dynamic ONNX reference on the exact same NumPy feature tensor.

Required:

- valid decoded-frame argmax agreement exactly `1.0`;
- maximum valid decoded-frame total-variation distance <= `0.002`.

The full padded tensor comparison is also recorded diagnostically, but padded
tail frames are not decoded and therefore do not gate deployment quality.

This is the same probability-space tolerance used for pretrained QuartzNet
MYRIAD semantic checks elsewhere in the repository.

## Full clean-speech quality gate

The authoritative physical run covers all 2,703 LibriSpeech dev-clean
utterances.

Required:

- WER <= `0.05`;
- absolute WER difference from the qualified PyTorch result
  `0.037939781625675524` <= `0.005`;
- semantic parity gate passes;
- zero training;
- exact 29-class source vocabulary;
- host-native ARM64 OpenVINO runtime and MYRIAD backend.

A `--max-samples` run is diagnostic only and cannot pass the full-corpus
qualification.

## Latency scope

The result records separately:

- Pi CPU NumPy frontend latency;
- steady-state MYRIAD `Infer()` latency;
- MYRIAD inference-only realtime factor;
- per-shape OpenVINO/MYRIAD load+compile times.

The inference latency and realtime factor exclude frontend, IPC, model load,
and session warmup. Because exact-time qualification can create many shape
sessions, model-load time is not a production latency estimate.

## Runtime refresh

The generic `hello_myriad` binary gained `--reshape-time` support for this
qualification. An existing extracted Pi runtime built before this change will
fail `check-reshape` by design.

The edge controller accepts `--refresh-runtime`. When requested, it rebuilds
the ARM64 runtime image from the repository's pinned OpenVINO source and
re-extracts the host-native runtime before re-running the reshape capability
check.

## Run

From the development host:

```bash
git pull --ff-only

./ci/verify-static.sh
./scripts/test-speech-asr-quartznet-reference.sh
./scripts/test-speech-asr-quartznet-reference-myriad.sh

./scripts/run-quartznet15x5-reference-myriad-edge.sh \
  --device cuda \
  --refresh-runtime
```

The first physical run should include `--refresh-runtime`. Later runs can
omit it while the extracted host runtime remains current.

A short physical diagnostic may add `--max-samples 100`, but the final
qualification must run all 2,703 utterances.

Local evidence is pulled to:

```text
work/speech-asr/deployed-evaluations/quartznet15x5-reference-libri-dev-clean/
  result.json
  evaluator.log
  hypotheses.jsonl
  sessions/
```

Return to REVIEW after the full physical result. Do not tune the model from a
partial hardware run.
