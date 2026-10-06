# ADR: cnn_ctc_v15 large-capacity BatchNorm conditioning

Status: implemented; execution pending
Date: 2026-10-06
Parent: `exp-1e2478b84317ab02/attempt-0001`

## Evidence

cnn_ctc_v14 completed the final learning-rate-only control for the 19.6M
parameter graph. The cosine schedule eliminated the v12/v13 blank-collapse
failure and selected epoch 8 at validation CER `0.7898404653573691`.

The remaining failure mode is generalization rather than optimizer explosion.
At epoch 8, train/validation loss was about `2.1325 / 4.2863`. By epoch 12,
train loss had fallen to `0.7465` while validation loss had risen to
`7.1361` and validation CER had worsened to `0.8242239244370213`.
The deployed v14 checkpoint also remains essentially tied with the much smaller
cnn_ctc_v10/v11 accuracy points while requiring about 104 ms p95 on MA2450.

The v14 ADR closed further LR-only tuning and called for an architectural
conditioning change.

## Decision

Introduce `cnn_ctc_v15` as a single controlled BatchNorm experiment:

- keep v14's stem widths, 16 residual blocks, kernel/dilation schedule,
  receptive field, output shape and decoder unchanged;
- keep Adam, the `3e-4 -> 3e-5` cosine schedule, clipping at 5.0, batch 1,
  12 architecture-screen epochs and validation-CER checkpoint selection;
- add BatchNorm after each stem convolution;
- add BatchNorm after each residual temporal convolution and residual projection;
- omit convolution bias terms where BatchNorm immediately follows;
- keep the final CTC projection bias-bearing;
- add no augmentation, auxiliary CTC loss, blank penalty, decoder change or
  additional capacity.

The resulting deterministic estimate is 19,642,599 trainable parameters,
2,515,673,088 convolution MACs and the same 653-feature-frame receptive field.

## Hardware risk

BatchNorm is not a new operator family for this repository: cnn_ctc_v2 already
qualified BatchNorm through ONNX, OpenVINO 2020.3 and physical MYRIAD execution.
v15 still must pass the existing initialized compatibility/numerical probe
before full training.

## Decision rule

Compare directly with v14 on the identical architecture-screen training and
ES2011 validation manifests. CER is primary, followed by decoder emission
diagnostics and WER. Because this graph is already much slower than v10/v11,
latency is recorded rather than used to explain away a quality regression.

If BatchNorm does not materially beat v14, do not continue tuning the 20M graph
with more optimizer variants. Reconsider the capacity/conditioning tradeoff
against the v10/v11 Pareto points.

The sealed held-out benchmark remains unavailable for selection.
