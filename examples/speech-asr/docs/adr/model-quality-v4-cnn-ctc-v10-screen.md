# ADR: cnn_ctc_v10 wider deep-dilated architecture screen

Status: implemented; execution pending
Date: 2026-10-06
Parent: `exp-00e6b1e434d187d8/attempt-0001`

## Evidence from v9

`cnn_ctc_v9` completed as `exp-df5bbe0ea05c2238/attempt-0001`.

It was rejected by the strict screen CER gate, but it is a useful efficiency
Pareto point:

- physical CER: `0.8032598053331798`;
- v3-screen CER: `0.7984219316938317`;
- relative CER ratio: `1.0060592945249947`;
- physical WER: `1.005182465990067`;
- emitted/reference characters: `0.39399873293785637`;
- empty-hypothesis fraction: `0.2545168892380204`;
- blank-frame fraction: `0.7961900066903764`;
- MA2450 p95: `12.0927212 ms`;
- final PyTorch/ONNX frame-argmax agreement: `1.0`;
- training duration: `648.212656306001 s`;
- selected checkpoint: epoch 12.

The output/emission geometry is healthy relative to v8, and latency has about
4.7 ms of headroom versus the v3 screen reference. However, v9's training loss
remains higher than the v3 screen control at epoch 12, suggesting that the
narrower model may be capacity- or convergence-limited under the fixed proxy
budget.

## v10 decision

v10 changes only width within the v9 family:

- input: `[1,64,512]`;
- stems: 64 then 112 channels, both stride 2;
- output: `[1,128,39]`;
- residual width: 112;
- residual blocks: 8;
- residual kernel: 7 in every block;
- dilations: `[1,2,3,4,4,3,2,1]`;
- receptive field: 493 feature frames;
- parameters: 865,511;
- fixed-input MACs: 113,149,952.

Compared with v9, this is a pure capacity test. Compared with v3, it remains
substantially smaller in MACs while keeping the same CTC time axis.

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

INIT_JSON="$(./scripts/init-cnn-ctc-v10-architecture-screen.sh)"
EXP="$(printf '%s\n' "$INIT_JSON" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])')"

./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

The sealed held-out benchmark remains unavailable for selection.
