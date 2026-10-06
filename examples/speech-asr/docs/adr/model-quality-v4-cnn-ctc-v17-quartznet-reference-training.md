# ADR: cnn_ctc_v17 QuartzNet reference-training alignment

Status: implemented; execution pending
Date: 2026-10-06
Evidence parent: `exp-4d7f92c301064c45/attempt-0001`

## v16 evidence

v16 did not fail because QuartzNet was unsupported by MA2450. ONNX/OpenVINO
conversion succeeded, the initialized physical MYRIAD gate had 100% frame
argmax agreement, and the full evaluator processed more than 800 samples before
the persistent server session died.

The acoustic training itself was poor and unstable. The best validation CER
was `0.9297356447618499` at epoch 5; by epoch 12 CER was back to `1.0`
with validation loss `43.97023439682033`. The first epoch also observed an
extreme pre-clip gradient norm.

## Training mismatch

The v16 architecture was based on QuartzNet-15x5, but its inherited training
recipe was not. v16 used Adam, LR `3e-4`, dropout `0.2` and no warmup.

QuartzNet literature/configuration uses NovoGrad and cosine scheduling, and
published NeMo QuartzNet-15x5 configurations use NovoGrad with LR `0.01`,
betas `0.8/0.5`, weight decay `0.001` and dropout `0.0`. NVIDIA QuartzNet
training guidance also reports warmup as useful for stable early optimization.

## Decision

v17 keeps the v16 inference topology but changes the training recipe:

- dropout `0.0`;
- NovoGrad;
- peak LR `0.01`;
- betas `0.8/0.5`;
- epsilon `1e-8`;
- weight decay `0.001`;
- 12% step-based linear warmup;
- cosine decay to `1e-5`;
- existing gradient clip `5.0`.

The data split, batch size 1, 12-epoch screen budget, validation-CER checkpoint
selection, frontend, vocabulary, CTC objective and greedy decoder stay frozen.

The physical evaluator is also made resilient to persistent-server drops. That
is an execution reliability fix, not a model-quality change.

If v17 remains far behind v11/v14 after the aligned training recipe, stop
spending screen budget on QuartzNet-from-scratch with this small AMI training
set and move to pretrained transfer/fine-tuning rather than another optimizer
or architecture micro-tweak.
