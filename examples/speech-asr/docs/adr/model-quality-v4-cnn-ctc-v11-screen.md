# ADR: cnn_ctc_v11 128-channel deep-dilated architecture screen

Status: implemented; execution pending
Date: 2026-10-06
Parent: `exp-00e6b1e434d187d8/attempt-0001`

## Evidence from v10

`cnn_ctc_v10` completed as `exp-884807b8e192aa9c/attempt-0001` and passed
all frozen architecture-screen acceptance gates:

- physical CER: `0.7888613718827392`;
- v3-screen CER: `0.7984219316938317`;
- relative CER ratio: `0.9880256798672726`;
- physical WER: `1.0308788598574823`;
- emitted/reference characters: `0.4818867707193457`;
- empty-hypothesis fraction: `0.24509033778476041`;
- blank-frame fraction: `0.7500616218880947`;
- MA2450 p95: `13.6566998 ms`;
- final PyTorch/ONNX frame-argmax agreement: `1.0`;
- training duration: `671.3644759650015 s`;
- selected checkpoint: epoch 12.

v10 improved CER by about 1.20% relative to the v3 screen control but did not
reach the stronger promotion CER `0.7744692737430167`. Its best CER remained
at the final screen epoch, and the 96 -> 112 width step improved the v9 family
without consuming the available MA2450 latency budget.

## v11 decision

v11 changes only width within the v9/v10 family:

- input: `[1,64,512]`;
- stems: 64 then 128 channels, both stride 2;
- output: `[1,128,39]`;
- residual width: 128;
- residual blocks: 8;
- residual kernel: 7 in every block;
- dilations: `[1,2,3,4,4,3,2,1]`;
- receptive field: 493 feature frames;
- parameters: 1,117,287;
- fixed-input MACs: 145,342,464.

Frontend, vocabulary, standard CTC objective, greedy decoder, optimizer,
learning rate, exact screen manifest, full ES2011 validation and 12-epoch
budget are unchanged.

## Decision rule

Screen acceptance requires:

- CER <= `0.7984219316938317`;
- MA2450 p95 <= 25 ms;
- final PyTorch/ONNX frame-argmax agreement = 1.0;
- physical MYRIAD execution accepted.

Promotion to a larger-budget confirmation requires CER <=
`0.7744692737430167` plus healthy emission diagnostics.

## Commands

```bash
./scripts/provision-speech-model-data-v4-edge.sh

INIT_JSON="$(./scripts/init-cnn-ctc-v11-architecture-screen.sh)"
EXP="$(printf '%s\n' "$INIT_JSON" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])')"

./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

The sealed held-out benchmark remains unavailable for selection.
