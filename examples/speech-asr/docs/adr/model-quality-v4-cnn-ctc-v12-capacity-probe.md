# ADR: cnn_ctc_v12 large-capacity MYRIAD probe

Status: implemented; execution pending
Date: 2026-10-06
Comparison parent: `exp-00e6b1e434d187d8/attempt-0001`

## Why the strategy changes

The final small-width candidate, `cnn_ctc_v11`, completed as
`exp-4fb0f93165d0f76d/attempt-0001` with CER
`0.7865576225306686` and MA2450 p95 `14.9809824 ms`. It passed the frozen
screen CER gate, but the 112 -> 128 width increase only slightly improved v10.

The project therefore stops optimizing around the inherited 25 ms policy and
moves directly into a much larger acoustic-model regime. For this probe,
latency is evidence rather than an acceptance threshold.

## Frozen design

`cnn_ctc_v12` keeps the existing input/output and decoder boundary but makes
a large capacity jump:

- input: `[1,64,512]`;
- stems: 128 and 448 channels, both stride 2;
- output: `[1,128,39]`;
- residual width: 448;
- residual blocks: 16;
- residual kernel: 5;
- dilations: `[1,2,3,4,4,3,2,1]` repeated twice;
- normalization: none;
- activation: ReLU;
- dropout: 0.1 during training;
- residual final projection init: 1%-scaled Kaiming;
- parameters: `19,627,687`;
- fixed-input MACs: `2,515,673,088`;
- FP16 weight bytes: `39,255,374`;
- receptive-field estimate: 653 feature frames.

This is about 17.6x v11's parameter count and 17.3x its fixed-input MACs.

## Experiment policy

The probe reuses the exact frozen architecture-screen training manifest and
full ES2011 validation manifest for comparability and keeps the 12-epoch,
batch-1, validation-CER-selected screen budget.

The optimization policy is intentionally more aggressive than v11. Adam uses
a OneCycle schedule with a `3e-3` peak learning rate: it starts at `3e-4`,
reaches the peak during the first 10% of optimizer steps, then cosine-anneals
to `3e-5`. Gradient clipping remains at 5.0. This raises the peak global
learning rate by 10x rather than carrying the small-model `3e-4` ceiling
into the 20M-parameter probe.

Acceptance applies no numeric WER, CER, RTF or latency limit. Required evidence
is:

- training completed;
- ONNX export completed;
- OpenVINO conversion completed;
- physical MYRIAD execution completed;
- accuracy evaluation completed;
- final PyTorch/ONNX frame-argmax agreement is 1.0.

The initialized MYRIAD numerical compatibility probe still runs before
training. A conversion or physical-device failure is a useful result: it marks
a capacity boundary. If the graph runs, CER/emission quality and measured p95
determine the next design.

The sealed held-out benchmark remains unavailable for selection.

## Commands

```bash
./scripts/provision-speech-model-data-v4-edge.sh

INIT_JSON="$(./scripts/init-cnn-ctc-v12-capacity-probe.sh)"
EXP="$(printf '%s\n' "$INIT_JSON" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])')"

./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```
