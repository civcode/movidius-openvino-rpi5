# ADR: model-quality-v3 scaled AMI training boundary

Status: executed and accepted as quality reference
Date: 2026-10-05
Accuracy reference: exp-aa5380b542b0d784 (`cnn_ctc_v3`)

## Evidence

The first meeting-disjoint expansion was the first intervention to improve all
transcript-level metrics: CER 0.84476, WER 1.04331, 14/95 empty hypotheses and
50.60% emitted/reference characters, with unchanged MA2450 latency. This is
evidence to scale the successful data variable before changing the model.

## Decision

Extend training from ES2005a-d to ES2005a-d, ES2006a-d and ES2007a-d. These
meetings are in AMI's official scenario training partition. Retain the exact
ES2002a speaker-D validation manifest and all v3 model, frontend, optimizer,
checkpoint-selection, export and hardware contracts.

The qualified boundary contains 4,429 utterances, 7,150.12 seconds, 14,753
words and 56,163 non-space characters. This is 2.99x the accepted v2 training
duration and 2.98x its record count. Record and meeting overlap with validation
are both zero.

## Scope

This remains a controlled subset, not the full approximately 50-hour AMI
scenario training partition. It spans three scenario teams and twelve sessions.
Validation remains a checkpoint-selection set, not an unbiased final test.

## Result

`exp-87538823d2bf1562/attempt-0001` passed CUDA, ONNX, OpenVINO and physical
MA2450 gates. Against `exp-aa5380b542b0d784`, CER improved from 0.84476 to
0.78629, empty hypotheses from 14/95 to 12/95, emitted/reference characters
from 50.60% to 72.08% and blank-frame fraction from 73.34% to 63.82%. WER
regressed slightly from 1.04331 to 1.07087. P95 latency remained 16.63 ms.

Accept the candidate because the primary CER and both under-emission measures
improved materially without a latency regression. Before further promotion or
architecture work, measure this frozen checkpoint once on a held-out official
AMI scenario evaluation partition that has never selected a checkpoint.

## Source authority

- https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml
- https://groups.inf.ed.ac.uk/ami/download/
