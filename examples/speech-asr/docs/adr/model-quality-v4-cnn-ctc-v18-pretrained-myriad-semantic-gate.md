# ADR: cnn_ctc_v18 pretrained MYRIAD semantic compatibility gate

Status: implemented
Date: 2026-10-07
Evidence experiment: `exp-26d9cd485975e897/attempt-0001`

## Context

The first cnn_ctc_v18 physical compatibility probe used the historical
initialized-model MYRIAD rule: maximum absolute logit error <= 0.01.

That rule was introduced for randomly initialized/near-identity acoustic
models, where frame argmax can be unstable because top logits are nearly tied.
For those models, bounding raw tensor error is a useful graph-fidelity proxy.

cnn_ctc_v18 is different: its probe executes a pretrained QuartzNet15x5 model.
Its logits have meaningful scale and large class margins. FP16 accumulation can
therefore move raw logits by much more than 0.01 while preserving the class
posterior and every CTC frame decision.

## Physical evidence

The first v18 MYRIAD probe produced:

- maximum absolute logit error: 0.7036104202270508;
- maximum reference absolute logit: 32.91058349609375;
- maximum error / maximum reference magnitude: 0.021379457471805818;
- frame argmax agreement: 1.0;
- frame argmax mismatches: 0 / 256;
- minimum reference top-two margin: 5.668647766113281;
- mean softmax absolute error: 1.2057904162574852e-06;
- maximum softmax absolute error: 0.0008063222413164928;
- mean per-frame total-variation distance: 2.351291311702096e-05;
- maximum per-frame total-variation distance: 0.0009230391151051188.

The raw-logit rule rejected this probe even though all 256 frame decisions were
identical and the largest probability-mass drift was below 0.1%.

## Decision

Keep the historical raw-logit gate unchanged for v1-v17.

For cnn_ctc_v18, use a pretrained semantic-parity gate:

- frame argmax agreement must be exactly 1.0;
- frame argmax mismatch count must be exactly 0;
- maximum per-frame total-variation distance between PyTorch and MYRIAD
  softmax distributions must be <= 0.002.

Raw-logit maximum error, softmax maximum error, and top-two margins remain
recorded as diagnostic evidence but do not independently accept the probe.

The 0.002 TV limit bounds worst-frame probability-mass movement to 0.2%, while
the exact argmax requirement prevents a probability-close but decision-changing
deployment from passing.

## Scope

This change applies only to the pre-training physical compatibility probe for
cnn_ctc_v18. It does not relax the legacy initialized-model gate, ONNX parity,
OpenVINO IR validation, persistent MYRIAD parity, corpus accuracy evaluation,
or final experiment acceptance policy.
