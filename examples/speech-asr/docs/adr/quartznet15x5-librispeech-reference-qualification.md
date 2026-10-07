# ADR: qualify the pinned NVIDIA QuartzNet source on LibriSpeech before further ASR work

Status: qualified; source-domain reproduction complete; physical deployment deferred
Date: 2026-10-07
Source: NVIDIA Multidataset QuartzNet15x5Base-En version 2

## Context

The frozen cnn_ctc_v19 deployment is internally consistent but not yet useful
for general transcription: the selected decoder reaches WER
`0.5497732671129346` and CER `0.4634567759027818` on AMI ES2011.

That result does not establish that the pretrained QuartzNet implementation is
semantically identical to NVIDIA's original recognizer. The AMI transfer path
changed important source contracts before fine-tuning, most visibly by expanding
the original 29-class CTC projection to a 39-class AMI vocabulary with ten
randomly initialized digit rows.

NVIDIA publishes WER `0.0379` for the pinned Multidataset
QuartzNet15x5Base-En version-2 checkpoint on LibriSpeech dev-clean. That known
source-domain result is a better pipeline-debugging target than AMI.

## Decision

Freeze v19 unchanged and introduce a separate zero-training pretrained-reference
qualification.

The reference path must:

1. load the already-pinned `QuartzNet15x5-En-Base.nemo` archive by its frozen
   size and SHA-384;
2. preserve the original 29 CTC classes in source order, with blank at index
   28;
3. import all encoder tensors and all 29 decoder rows, leaving no task-specific
   randomly initialized projection rows;
4. reproduce the historical NeMo evaluation frontend rather than reuse the AMI
   fixed-shape frontend;
5. evaluate the full OpenSLR SLR12 LibriSpeech `dev-clean` corpus with greedy
   CTC decoding and no optimizer updates;
6. compare measured WER with NVIDIA's published `0.0379` dev-clean WER;
7. export the same zero-training PyTorch reconstruction to dynamic-time ONNX and
   require frame-argmax parity as a toolchain check.

The dev-clean archive is pinned by MD5
`42e2234ba48799c1f50f24a7926300a1` and expected to contain 2,703
utterances.

## Frontend contract

The reference frontend follows the historical NeMo FilterbankFeatures inference
semantics used by the source model:

- 16 kHz mono float audio;
- 20 ms / 320-sample analysis window;
- 10 ms / 160-sample hop;
- 512-point FFT;
- 64 Slaney-normalized mel bins from 0 to 8 kHz;
- preemphasis 0.97;
- non-periodic Hann window;
- centered STFT with constant padding;
- power spectrum;
- additive log guard `2^-24`;
- per-feature normalization over valid frames;
- sample standard deviation (`ddof=1`) plus `1e-5`;
- evaluation-time dither disabled;
- valid feature length `floor(audio_samples / 160)`;
- zero padding of the feature time axis to a multiple of 16.

These details are source-compatibility requirements, not new model
hyperparameters.

## Qualification gate

The published comparison point is WER `0.0379`. The first reproduction gate
is deliberately simple: the complete dev-clean run must reach WER <= `0.05`.
The result also records the exact absolute delta from `0.0379`.

A `--max-samples` run is diagnostic only and cannot pass the full-corpus gate.

## Qualification result

The full zero-training LibriSpeech dev-clean evaluation completed on 2026-10-07
with all 2,703 utterances:

- measured WER `0.037939781625675524` (3.793978%);
- published NVIDIA WER `0.0379` (3.79%);
- absolute WER delta `0.00003978162567552096`;
- CER `0.012506494000177396`;
- source decoder symbols loaded: 29/29;
- target-only randomly initialized symbols: 0;
- training performed: false;
- qualification status: `pass`.

This reproduces the published source-domain result essentially exactly. The
remaining difference is about 0.004 percentage points absolute WER, far below
the 5% reproduction ceiling.

The zero-training PyTorch -> ONNX parity check also passed:

- frame argmax agreement `1.0`;
- frame argmax mismatches `0 / 256`;
- maximum absolute logit error `6.0677528381347656e-05`;
- maximum per-frame total-variation distance
  `5.6905102316839574e-08`.

The source QuartzNet architecture, 29-class CTC head, pretrained tensor import,
historical NeMo frontend semantics, greedy decoder, and PyTorch/ONNX export path
are therefore qualified. Do not change these components to explain the AMI
error rate.

The earlier AMI result must now be treated as an adaptation/domain problem:
the source recognizer is correct before the AMI-specific 39-class head,
fixed-shape adaptation, and fine-tuning policy are introduced.

LibriSpeech test-clean remains available as an independent confirmation against
NVIDIA's published `0.0385` result, but it is not required to establish the
dev-clean source-pipeline reproduction.

## Hardware boundary

This phase performs no OpenVINO conversion and no MYRIAD execution.

That boundary describes this completed source-domain qualification only. After
the clean reference was reproduced and selected as the product-quality target,
a separate deployment qualification was created in
`quartznet15x5-librispeech-myriad-qualification.md`. The frozen 3.79% source
evidence remains unchanged.

The repository already has sufficient evidence that compatible acoustic graphs
can be carried through ONNX/OpenVINO to MA2450 once their reference semantics
are correct.

## Command

Run from the normal development host:

```bash
./scripts/test-speech-asr-quartznet-reference.sh
./scripts/qualify-quartznet15x5-reference.sh --device cuda
```

Use `--device auto` when CUDA is not available. For a short diagnostic before
the full 2,703-utterance run:

```bash
./scripts/qualify-quartznet15x5-reference.sh --device cuda --max-samples 100
```

No training occurs in either command.
