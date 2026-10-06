# ADR: model-quality-v4 cnn_ctc_v7 112-channel capacity ablation

Status: executed; rejected
Date: 2026-10-06
Parent: `exp-63fdb8d218673527/attempt-0001`
Parent model: `cnn_ctc_v3`

## Evidence

The fresh model-quality-v4 baseline completed on the frozen ES2011 validation
boundary with:

- CER `0.7588550365720209`;
- WER `1.0427553444180522`;
- blank-frame fraction `0.6907285467798162`;
- empty-hypothesis fraction `0.15789473684210525`;
- emitted/reference character ratio `0.5895870529286413`;
- MA2450 p95 `16.7812226 ms`;
- 32-epoch wall-clock training time `3384.302443212 s`.

The deployment graph still has substantial room under the 25 ms p95 hardware
gate, while under-emission remains pronounced.

Previous controlled development on the older boundary rejected deterministic
SpecAugment, valid-frame CMVN, a 0.25 training-only blank-logit penalty and
InterCTC as model-quality directions. The next isolated variable is therefore
encoder capacity.

## Decision

Introduce `cnn_ctc_v7` as a width-only ablation:

- keep `logmel-v1`;
- keep input `[1,64,512]` and output `[1,128,39]`;
- keep the two stride-2 stems and residual kernel schedule
  `[11,19,27,35,43]`;
- keep normalization-free residual blocks, ReLU, dropout 0.1 and
  `kaiming_scaled_0.01` residual projection initialization;
- widen the second stem from 96 to 112 channels;
- widen all five residual blocks from 96 to 112 channels;
- increase parameters from 1,346,343 to 1,818,183;
- increase fixed-input MACs from 174,804,992 to 235,177,984;
- keep the 533-frame receptive field;
- keep standard CTC, greedy decoding, 32 epochs, batch size 1, Adam/cosine and
  validation-CER checkpoint selection;
- add no augmentation, blank bias, auxiliary objective or decoder change.

The training and validation manifests remain exactly:

- train:
  `6025d17f08c1815d1710c3ab56cea51a34365f398e9877b4aefec234e64e4096`;
- validation:
  `fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088`.

The sealed held-out corpus remains consumed and is forbidden for this decision.

## Decision rule

The experiment is compared directly against
`exp-63fdb8d218673527/attempt-0001` because both use the exact same ES2011
benchmark manifest.

Required gates:

- CER no worse than `0.7588550365720209`;
- ONNX/OpenVINO compatibility valid;
- physical MA2450 numerical gate accepted;
- final PyTorch/ONNX frame-argmax agreement `1.0`;
- MA2450 p95 no greater than `25 ms`.

WER, blank-frame fraction, empty-hypothesis fraction and
emitted/reference-character ratio are secondary review evidence. A capacity
direction should ideally improve CER together with emission diagnostics rather
than trading one collapse mode for another.

The initialized graph is physically probed on MA2450 before the full training
run.

## Result

`exp-ed00410b3da5d26c/attempt-0001` completed normally but failed the
same-manifest CER gate.

Selected-checkpoint / hardware evidence:

- best validation-CER epoch: 25;
- training validation CER at the selected epoch: `0.7669181593042677`;
- physical evaluation CER: `0.7670333467718712`;
- physical evaluation WER: `1.0462103217447636`;
- MA2450 p95: `19.7972674 ms`;
- PyTorch/ONNX frame-argmax agreement: `1.0`;
- total 32-epoch training duration: `3442.081941446 s`.

The candidate is rejected because CER is worse than the frozen v3 ES2011
baseline `0.7588550365720209`. Width increased hardware cost without fixing
the under-emission/generalization problem. Do not continue a width sweep from
v7.

The next development step is a deterministic reduced-budget architecture
screen, documented in `model-quality-v4-architecture-screen-v1.md`.

## Commands

After syncing the implementation and provisioning the already frozen ES2011
validation data to the same revision:

```bash
EXP="$(
  ./scripts/init-cnn-ctc-v7-wide.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"
echo "$EXP"

./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

The trainer streams progress, throughput and ETA during the run.
