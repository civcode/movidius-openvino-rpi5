# ADR: isolate AMI adaptation loss from the qualified QuartzNet source recognizer

Status: complete; AMI loss attributed primarily to domain/adaptation
Date: 2026-10-07
Qualified source reference: `quartznet15x5_nvidia_ref`
Frozen AMI reference: `cnn_ctc_v19` from
`exp-f4adb44ab833e896/attempt-0002`

## Context

The pinned NVIDIA Multidataset QuartzNet15x5Base-En source recognizer has now
been reproduced on the full LibriSpeech dev-clean corpus at WER
`0.037939781625675524`, essentially identical to NVIDIA's published
`0.0379`. The source 29-class CTC head, historical NeMo frontend, pretrained
tensor import, greedy decoder, and PyTorch/ONNX path are therefore qualified.

The frozen AMI v19 result is much worse. Before decoding, the selected acoustic
checkpoint reaches validation WER `0.5983588857698121` / CER
`0.46201693255773774`. The source-domain result therefore rules out a broken
base QuartzNet reconstruction as the primary explanation.

The next question is where quality is lost when moving from the qualified
source recognizer to the AMI-specific pipeline.

## Decision

Measure the AMI adaptation boundary as four ordered PyTorch reference stages on
the exact frozen 1,273-record ES2011 validation manifest.

No new optimizer update is allowed.

### Stage 0: domain only

Use:

- the qualified original 29-class source CTC head;
- blank at source index 28;
- the historical NeMo source frontend;
- the untouched pinned pretrained weights;
- the AMI validation audio and existing text-v1 scorer.

This measures the source recognizer on AMI before any AMI-specific frontend,
head, or training change.

Do not interpret the numerical difference between LibriSpeech WER and stage-0
AMI WER as a frontend/head delta: the datasets themselves changed.

### Stage 1: fixed frontend

Keep the exact same 29-class source model and weights from stage 0, but replace
only the frontend with the AMI `logmel-v3` fixed-512-frame frontend used by
v18/v19.

The stage-0 -> stage-1 WER/CER delta is the frontend-adaptation effect under the
same acoustic weights and source head.

### Stage 2: AMI head, epoch zero

Keep the fixed frontend from stage 1 and introduce the v19 39-class head:

- blank moves to index 0 according to the project vocabulary;
- all 29 source rows are remapped from the pretrained checkpoint;
- digit rows 0-9 remain the same deterministic seed-1337 initialization used by
  v19 before training;
- no optimizer step occurs.

The stage-1 -> stage-2 delta isolates the 39-class head expansion/remap,
including competition from the ten randomly initialized digit rows.

As a self-check, the complete stage-2 run must reproduce the already-recorded
pretrained AMI epoch-zero evidence:

- WER `0.7393651479162168`;
- CER `0.5776651500316765`.

### Stage 3: selected v19 fine-tuning

Keep the same fixed frontend and 39-class vocabulary, then load the existing
selected best-epoch-5 v19 checkpoint. Do not retrain it.

The stage-2 -> stage-3 delta is the effect of the already-completed conservative
AMI fine-tuning.

As a self-check, the complete stage-3 run must reproduce:

- WER `0.5983588857698121`;
- CER `0.46201693255773774`.

## Frozen dataset and checkpoint guards

The evaluator requires the AMI validation manifest SHA-256

`fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088`

and exactly 1,273 records.

By default it searches the frozen experiment
`exp-f4adb44ab833e896` for an `attempt-0002` `checkpoint.pt`, verifies the
checkpoint format/model/spec/vocabulary identity, requires `best_epoch == 5`,
and refuses multiple distinct valid checkpoint hashes. An explicit
`--v19-checkpoint` may be supplied when necessary.

## Interpretation

Only adjacent stage deltas are causal attribution evidence:

- stage 0 -> 1: fixed frontend;
- stage 1 -> 2: 39-class head expansion/remap;
- stage 2 -> 3: existing v19 fine-tuning.

Stage 0 itself is the domain-transfer observation: how the untouched source
recognizer behaves on AMI.

The result includes per-stage hypothesis JSONL files so high-error utterances
can be compared across adjacent stages after the aggregate attribution is known.

## Attribution result

The full 1,273-utterance run completed with frozen-evidence self-checks passing
exactly.

Measured stages:

- stage 0, source head + source frontend on AMI:
  WER `0.7806089397538328`, CER `0.6027184242354432`;
- stage 1, source head + AMI fixed frontend:
  WER `0.7393651479162168`, CER `0.5776651500316765`;
- stage 2, 39-class AMI head at epoch zero:
  WER `0.7393651479162168`, CER `0.5776651500316765`;
- stage 3, selected v19 fine-tuned checkpoint:
  WER `0.5983588857698121`, CER `0.46201693255773774`.

Adjacent effects:

- fixed frontend: WER delta `-0.04124379183761606`, CER delta
  `-0.025053274203766684`;
- 39-class head expansion/remap: WER delta `0.0`, CER delta `0.0`;
- existing v19 fine-tuning: WER delta `-0.14100626214640466`, CER delta
  `-0.11564821747393877`.

The source-domain LibriSpeech dev-clean WER is
`0.037939781625675524`, while the untouched source recognizer reaches only
`0.7806089397538328` on the frozen AMI validation set. These values are not
a same-dataset causal delta, but together they show that the dominant quality
problem appears before the AMI frontend, expanded head, and fine-tuning are
introduced.

The fixed frontend is not the cause of the collapse; it improves WER by about
4.12 absolute percentage points. The 39-class head expansion/remap is neutral
at epoch zero. The existing fine-tuning is beneficial, improving WER by about
14.10 absolute percentage points, but it is insufficient to overcome the
source-to-AMI domain gap.

The next REVIEW should therefore focus on domain/data adaptation rather than
changing the qualified source architecture, reverting the fixed frontend, or
removing the 39-class head.

## Hardware boundary

This is a PyTorch reference analysis only. It performs no new training,
OpenVINO conversion, Docker execution, SSH, or MYRIAD inference.

The MA2450 deployment path remains deferred until the AMI quality loss is
understood at the CPU/CUDA reference level.

## Run

First run the static/unit gates:

```bash
./ci/verify-static.sh
./scripts/test-speech-asr-quartznet-reference.sh
./scripts/test-speech-asr-quartznet-ami-attribution.sh
```

A short diagnostic can be run with:

```bash
./scripts/evaluate-quartznet15x5-ami-attribution.sh \
  --device cuda \
  --max-samples 100
```

The authoritative attribution run is:

```bash
./scripts/evaluate-quartznet15x5-ami-attribution.sh --device cuda
```

Results are written under:

```text
work/speech-asr/analysis/quartznet15x5-ami-attribution-v1/
  result.json
  s0-source-head-source-frontend.hypotheses.jsonl
  s1-source-head-fixed-frontend.hypotheses.jsonl
  s2-ami-head-fixed-frontend-epoch0.hypotheses.jsonl
  s3-v19-finetuned.hypotheses.jsonl
```

Return to REVIEW after the full attribution result. Do not change the source
architecture, AMI frontend, vocabulary/head, or training recipe before that
evidence is recorded.
