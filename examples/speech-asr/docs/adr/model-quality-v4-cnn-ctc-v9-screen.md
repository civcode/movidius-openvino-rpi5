# ADR: cnn_ctc_v9 deep small-kernel dilated architecture screen

Status: implemented; execution pending
Date: 2026-10-06
Parent: `exp-00e6b1e434d187d8/attempt-0001`

## Evidence from v8

`cnn_ctc_v8` completed as `exp-aa5f9f4edd71919b/attempt-0001`.
The runtime path was healthy:

- initialized MYRIAD frame-argmax agreement: `1.0`;
- initialized MYRIAD max absolute error: `0.0007587336003780365`;
- final PyTorch/ONNX frame-argmax agreement: `1.0`;
- zero hardware failures.

The architecture result was rejected:

- CER: `0.8442089500662328` versus v3-screen `0.7984219316938317`;
- WER: `0.9784063917080544`;
- emitted/reference characters: `0.2442550250532742`;
- empty-hypothesis fraction: `0.3307148468185389`;
- MA2450 p95: `26.6580554 ms`, above the 25 ms ceiling;
- training duration: `522.7922753729981 s`;
- selected checkpoint: epoch 12.

The 256-frame CTC direction is therefore retired. It increased blank-path
opportunity and hardware cost without improving CER.

## v9 decision

v9 restores the proven v3 tensor rate and width while changing temporal
modeling depth:

- input: `[1,64,512]`;
- stems: 64 then 96 channels, both stride 2;
- output: `[1,128,39]`;
- residual width: 96;
- residual blocks: 8;
- residual kernel: 7 in every block;
- dilations: `[1,2,3,4,4,3,2,1]`;
- receptive field: 493 feature frames;
- parameters: 646,503;
- fixed-input MACs: 85,151,744.

Relative to v3, this keeps the same CTC time axis and channel width but uses
more nonlinear depth and dilation instead of five very large ordinary kernels.
The receptive field remains close to the complete 512-frame input while the
fixed-input MAC estimate falls from 174,804,992 to 85,151,744.

Frontend, vocabulary, standard CTC objective, greedy decoder, optimizer,
learning rate, screen manifest, full ES2011 validation and 12-epoch budget are
unchanged.

## Decision rule

Screen acceptance requires:

- CER <= `0.7984219316938317`;
- MA2450 p95 <= 25 ms;
- final PyTorch/ONNX frame-argmax agreement = 1.0;
- physical MYRIAD execution accepted.

Promotion to a larger-budget confirmation requires CER <=
`0.7744692737430167`, a 3% relative improvement over the v3 screen control,
plus non-regressing emission diagnostics.

Because v9 restores the 128-frame output, blank-frame fraction is directly
comparable to the v3 screen again.

## Commands

```bash
./scripts/provision-speech-model-data-v4-edge.sh

INIT_JSON="$(./scripts/init-cnn-ctc-v9-architecture-screen.sh)"
EXP="$(printf '%s\n' "$INIT_JSON" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])')"

./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

The sealed held-out benchmark remains unavailable for selection.
