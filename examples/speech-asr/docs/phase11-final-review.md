# Phase 11 final review

Date: 2026-10-05
Status: complete

## Purpose

Phase 11 tested whether the qualified OpenVINO 2020.3 / MA2450 execution
platform could support evidence-driven custom ASR architecture iteration.

The phase exit criterion required at least two architecture generations to be
proposed, trained, physically evaluated and reviewed with enough lineage to
explain the next decision.

That criterion is satisfied.

## Generation 1: cnn_ctc_v2

Experiment: `exp-1c682f4eda475a01/attempt-0001`

Generation 1 replaced the tiny bring-up encoder with a 1.35M-parameter
large-kernel residual temporal CTC model.

Review evidence:

- OpenVINO IR valid;
- PyTorch/ONNX frame argmax agreement 1.0;
- initialized PyTorch/MYRIAD frame argmax agreement 1.0;
- MYRIAD p95 latency 16.647 ms;
- inference-only RTF 0.004587;
- WER 1.0;
- CER 0.945652;
- one-epoch train/validation loss 5.9129 / 4.7632.

The frozen CER gate rejected the model because 0.945652 exceeded 0.913043.

The review conclusion was that hardware compatibility and realtime capacity were
not the bottleneck; the training budget was too small to evaluate a deeper
model meaningfully.

## Generation 2: cnn_ctc_v3

Experiment: `exp-a6f83c0451532122/attempt-0001`

Generation 2 kept the successful v2 temporal geometry and hardware workload,
removed BatchNorm, used near-identity residual initialization, and expanded the
optimization budget to 32 epochs / 64 optimizer steps.

Execution evidence:

- repository static checks passed;
- 197 speech tests passed with 4 expected NumPy-environment skips;
- requested training device: CUDA;
- actual model/logits device: `cuda:0`;
- CUDA device: NVIDIA GeForce RTX 4070 Ti SUPER;
- peak CUDA allocated/reserved memory: 168,837,120 / 316,669,952 bytes;
- deterministic CTC loss device: CPU;
- OpenVINO IR valid;
- PyTorch/ONNX frame argmax agreement 1.0;
- initialized MYRIAD numerical gate accepted;
- initialized MYRIAD max absolute error 0.000905376;
- MYRIAD p95 latency 16.59 ms;
- inference-only RTF 0.004579;
- zero hardware failures.

Training evidence:

- 32 epochs;
- 64 optimizer steps;
- best epoch 32;
- best validation loss 2.66385;
- final train/validation loss 2.66728 / 2.66385.

Accuracy evidence:

- WER 1.0;
- CER 1.0.

The frozen CER gate rejected the model.

## Interpretation

Generation 2 proves that generation 1's failure was not simply caused by too few
optimizer updates. The larger optimization budget substantially reduced CTC
loss, but greedy transcript quality became worse.

This is not sufficient evidence to reject the v2/v3 temporal encoder family.
The active limitation is the evaluation population:

- `ami-smoke-v1` has only two eligible records for this fixed-shape CTC path;
- the same smoke population is reused for training and validation;
- WER is saturated at 1.0 for v1, v2 and v3;
- a small number of character edits causes large CER movement;
- validation loss and greedy CER are demonstrably not ranking candidates
  consistently on this population.

Therefore further architecture selection on this smoke manifest would be
overfitting the experiment harness, not meaningful ASR research.

## Decision

Phase 11 is complete.

The qualified execution/model-development platform is retained. `cnn_ctc_v2`
and `cnn_ctc_v3` remain useful hardware-compatible research baselines, but
neither is accepted as an accuracy model.

No generation-3 architecture should be selected from the smoke results.

Before architecture research resumes, the project must establish a meaningful
model-quality dataset boundary:

1. disjoint training and validation manifests;
2. enough eligible speech to make CER/WER changes statistically interpretable;
3. frozen manifest hashes and text normalization;
4. explicit speaker/session separation where the source corpus permits it;
5. decoder diagnostics including blank rate, emitted-token count, hypothesis
   length and per-sample transcript/error evidence;
6. a training budget expressed in optimizer steps or processed audio duration,
   not only epochs.

Only after that boundary is accepted should augmentation, decoder improvements,
or a third architecture generation be reviewed.
