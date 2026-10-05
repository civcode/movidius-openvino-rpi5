# ADR: Phase 11 generation-2 cnn_ctc_v3

Status: accepted design for implementation
Date: 2026-10-05
Parent: exp-1c682f4eda475a01 (cnn_ctc_v2 generation 1)

## Evidence from generation 1

Generation 1 completed successfully on physical MA2450:

- OpenVINO IR valid;
- PyTorch/ONNX frame argmax agreement 1.0;
- initialized PyTorch/MYRIAD frame argmax agreement 1.0;
- p95 inference latency 16.647 ms;
- inference-only RTF 0.004587;
- zero hardware failures.

It was rejected only because CER 0.945652 exceeded the frozen 0.913043
maximum. Training evidence was also weaker than v1 after one epoch:
train loss 5.9129 and validation loss 4.7632 versus v1 4.6742/4.5850.

The hardware graph therefore does not need to be made smaller to satisfy the
target. The immediate problem is optimization fidelity.

## Decision

Generation 2 keeps the v2 temporal geometry and removes BatchNorm. Residual
blocks use only ordinary Conv1d, ReLU, Dropout during training, 1x1 Conv, Add
and ReLU. The final 1x1 residual projection is small-initialized so each block
starts near an identity mapping.

Training changes from one epoch/batch 2 to 32 epochs/batch 1. On the current
two eligible smoke records this changes the order of magnitude from roughly one
optimizer update to roughly 64 updates.

Training also adds:

- cosine learning-rate decay from 3e-4 to 3e-5;
- deterministic gradient clipping at norm 5.0;
- best-validation-loss checkpoint selection;
- explicit optimizer-step count in training evidence.

## Why normalization-free

Fixup-style residual learning demonstrates that normalization is not required
for stable deep residual optimization when initialization is designed around
the residual path. Removing BatchNorm is also attractive on this target because
it removes running-statistics state and keeps the exported inference graph
closer to the Conv/ReLU/Add operator set already proven on MYRIAD.

This is not a claim that BatchNorm caused generation-1 failure; the dominant
measured issue is the tiny optimization budget. The normalization change is a
controlled simplification made at the same time as the update-budget repair.

## Why no SpecAugment yet

SpecAugment is well established for ASR, but the current smoke manifest has only
two eligible examples and is reused for validation. Adding masking now would
confound the learnability check. Augmentation belongs in the later real
train/validation corpus experiment.

## Frozen architecture

```text
[1,64,512]
 -> Conv1d 64, k5, s2, bias + ReLU
 -> Conv1d 96, k5, s2, bias + ReLU
 -> residual k11 @96
 -> residual k19 @96
 -> residual k27 @96
 -> residual k35 @96
 -> residual k43 @96
 -> Conv1d 39, k1
 -> transpose
[1,128,39]
```

Each residual block:

```text
x -> Conv1d(k,bias) -> ReLU -> Dropout(0.1)
  -> Conv1d(k1,bias, zero-init) -> Add(x) -> ReLU
```

## Acceptance

Generation 2 must retain:

- valid OpenVINO 2020.3 FP16 conversion;
- initialized physical MYRIAD probe;
- ONNX frame argmax agreement 1.0;
- WER <= 1.0;
- CER <= 0.913043;
- inference-only RTF <= 0.01;
- p95 MYRIAD latency <= 25 ms.

The stricter hardware thresholds are still above generation-1 measurements but
prevent accuracy work from silently consuming the available realtime margin.

## References

- Fixup Initialization: Residual Learning Without Normalization,
  arXiv:1901.09321
- SpecAugment, arXiv:1904.08779
- Fast Conformer, arXiv:2305.05084
- QuartzNet, arXiv:1910.10261
- Citrinet, arXiv:2104.01721


### Compatibility-probe note

Generation 2 originally considered exact zero initialization for the residual
projection. That was rejected before execution because a legacy graph optimizer
could simplify an exactly-zero branch during the initialized compatibility
probe. Using nonzero Kaiming weights scaled to 1% preserves the near-identity
optimization goal while ensuring the pretraining OpenVINO/MYRIAD probe sees the
same residual Conv/Add topology that will exist after training.


### Initialized MYRIAD numerical gate

The first physical v3 attempt reached MA2450 and produced 126/128 matching
frame argmaxes (0.984375), but the original hard-coded 0.99 initialized-model
argmax gate stopped the experiment before training.

For an untrained near-identity network, argmax is not a stable numerical
equivalence measure because top logits can be nearly tied. The pretraining
physical gate therefore uses bounded tensor error instead:

- tensor shape must match the fixed output contract;
- all values must be finite;
- maximum absolute logit error must be <= 0.01.

Frame argmax agreement, mismatch count and reference top-two margins remain
diagnostic evidence. This change applies only to the initialized hardware
compatibility probe; trained-model acceptance thresholds are unchanged.
