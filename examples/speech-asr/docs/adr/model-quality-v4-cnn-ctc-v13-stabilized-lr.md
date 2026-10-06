# ADR: cnn_ctc_v13 stabilized large-capacity training retry

Status: implemented; execution pending
Date: 2026-10-06
Parent: `exp-4e16fd348004571b/attempt-0001`

v13 keeps the complete v12 19,627,687-parameter / 2,515,673,088-MAC
inference graph. v12 proved that graph physically runs on MA2450 at
103.8896558 ms p95 and RTF 0.058019043467865204, but its 3e-3 OneCycle peak
caused permanent blank collapse.

The only intended training change is the LR trajectory:

- Adam, unchanged;
- initial LR `3e-4`, unchanged;
- peak LR `1.2e-3` instead of `3e-3`;
- peak at 30% instead of 10% of optimizer steps;
- final LR `3e-5`, unchanged;
- gradient clipping `5.0`, unchanged;
- 12 epochs, batch 1, validation-CER checkpoint selection, unchanged.

This retains a peak four times the historical small-model learning rate while
giving the large network several epochs to stabilize before maximum LR.

There are no numeric CER/WER/RTF/p95 rejection thresholds. The objective of
this retry is to determine whether the physically viable large graph can train
without falling into the CTC blank attractor. The sealed held-out benchmark
remains unavailable for selection.
