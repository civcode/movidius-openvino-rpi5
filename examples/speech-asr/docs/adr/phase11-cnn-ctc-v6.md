# ADR: cnn_ctc_v6 v3-capacity InterCTC ablation

Status: executed and rejected as a quality candidate
Date: 2026-10-05
Reference parent: exp-3c7727ca3f37ba2c (`cnn_ctc_v3`)
Preceding evidence: exp-fc2f3424d95c843e (`cnn_ctc_v5`)

## Evidence

The compact v5 objective/encoder redesign passed all compatibility and hardware
gates and cut MA2450 p95 latency from 16.64 to 8.11 ms. It failed the primary
quality rule:

- CER 0.91734 versus 0.86694 for v3;
- blank-frame fraction 79.50% versus 61.33%;
- empty hypotheses 39/95 versus 19/95;
- emitted/reference characters 23.59% versus 47.98%.

Its best CER occurred at epoch 31, and final/intermediate training CTC losses
were still decreasing. This differs from v3's selected epoch 21 and indicates
that aggressive encoder downsizing confounded the test of intermediate
supervision.

## Decision

Introduce `cnn_ctc_v6` as a controlled objective ablation:

- restore v3's 96-channel encoder and `[11,19,27,35,43]` residual kernels;
- restore 1,346,343 deployed parameters, 174,804,992 MACs and 533-frame
  receptive field;
- preserve v3's deployed parameter initialization order at seed 1337;
- add one training-only CTC projection after residual block 3;
- optimize `0.7 * final_CTC + 0.3 * intermediate_CTC`;
- retain v3 data, logmel-v1 frontend, schedule, final-head CER checkpoint
  selection, output contract and greedy decoder.

The auxiliary projection is created after the deployed output projection so all
deployed parameters receive the same seed-1337 initialization as v3. It is not
called by `forward()` and is excluded from ONNX and OpenVINO inference.

## Decision rule

Compare directly with `exp-3c7727ca3f37ba2c`. CER is primary, followed by
blank-frame fraction, empty hypotheses and emitted/reference characters. If v6
does not beat v3, reject InterCTC for this data/model boundary; do not tune its
weight. Preserve v5 only as a latency/size Pareto point.

## Reference

- J. Lee and S. Watanabe, “Intermediate Loss Regularization for CTC-based
  Speech Recognition,” ICASSP 2021, https://arxiv.org/abs/2102.03216

## Result

`exp-1d4836d8b6639c0d/attempt-0001` passed all compatibility and hardware gates
but did not beat v3. CER was 0.89819, blank-frame fraction 71.53%, empty
hypotheses 25/95 and emitted/reference characters 35.28%. P95 latency remained
16.64 ms. This rejects InterCTC at weight 0.3 for the current data/model
boundary; do not tune the weight.
