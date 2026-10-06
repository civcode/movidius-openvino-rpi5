# ADR: cnn_ctc_v8 high-resolution dilated architecture screen

Status: implemented; execution pending
Date: 2026-10-06
Parent: `exp-00e6b1e434d187d8/attempt-0001`

## Screen reference

The deterministic architecture-screen-v1 control completed with unchanged
`cnn_ctc_v3` on:

- 3,904 training records;
- 6,407.372 seconds of unique training audio;
- all 48 training meetings represented;
- 12 epochs / 46,848 optimizer steps;
- full 1,273-record ES2011 validation.

The selected checkpoint was epoch 12. Physical evaluation measured:

- CER `0.7984219316938317`;
- WER `0.9930900453465774`;
- empty-hypothesis fraction `0.26865671641791045`;
- emitted/reference character ratio `0.39250129585901056`;
- MA2450 p95 `16.7866342 ms`;
- training duration `404.94855711700075 s`.

This is the proxy architecture-ranking reference, not the full-budget promotion
baseline.

## Architecture decision

The rejected v7 experiment showed that widening the existing full-context v3
topology increases cost without improving CER. v3's calculated receptive field
already exceeds the 512-frame input, while its two stride-2 stems compress the
time axis to only 128 CTC frames before residual acoustic modeling.

v8 therefore reallocates compute from width into temporal resolution:

- stem 1: 64 channels, kernel 5, stride 1;
- stem 2: 72 channels, kernel 5, stride 2;
- residual width: 72;
- residual kernels: `[11,19,27,35,43]`;
- residual dilations: `[1,2,2,2,2]`;
- output time axis: 256 CTC frames;
- receptive field: 509 input feature frames;
- parameters: 772,983;
- fixed-input MACs: 202,897,408;
- input/output: `[1,64,512] -> [1,256,39]`.

The frontend, vocabulary, standard CTC objective, greedy decoder, optimizer,
learning rate, checkpoint-selection metric and screen data remain unchanged.

## Screen decision rule

Execution acceptance requires:

- CER no worse than the v3 screen control:
  `0.7984219316938317`;
- MA2450 p95 no greater than 25 ms;
- final PyTorch/ONNX frame-argmax agreement 1.0;
- physical MYRIAD execution accepted.

Promotion to a larger-budget confirmation is intentionally stricter: screen CER
must improve by at least 3% relative, i.e. CER <=
`0.7744692737430167`. Empty-hypothesis fraction and
emitted/reference-character ratio are reviewed as secondary evidence.

Because v8 doubles the output time axis, raw blank-frame fraction is not
directly comparable with v3 and is not a promotion criterion.

The initialized v8 graph is converted and physically probed before training.
This is important because OpenVINO 2020.3/MYRIAD compatibility of the dilated
Conv1D pattern is not assumed.

## Commands

```bash
./scripts/provision-speech-model-data-v4-edge.sh

INIT_JSON="$(./scripts/init-cnn-ctc-v8-architecture-screen.sh)"
EXP="$(printf '%s\n' "$INIT_JSON" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])')"

./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

The sealed held-out benchmark remains consumed and unavailable for this
architecture decision.
