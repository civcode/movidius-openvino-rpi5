# ADR: model-quality-v2 expanded AMI training boundary

Status: accepted for controlled execution
Date: 2026-10-05
Accuracy reference: exp-3c7727ca3f37ba2c (`cnn_ctc_v3`)

## Evidence

The model-quality-v1 training set contains only 125 eligible utterances and
219.998 seconds from ES2002a. Controlled changes to augmentation, frontend
normalization, blank pressure, encoder size and intermediate CTC all failed to
beat v3. The v6 objective ablation reached CER 0.89819 versus 0.86694 for v3,
with 71.53% blank frames and 25/95 empty hypotheses.

Further loss/encoder micro-tuning on 220 seconds is not justified.

## Decision

Create `model-quality-v2` with:

- training source: all annotated A-D segments from ES2005a/b/c/d;
- validation: the byte-identical model-quality-v1 ES2002a speaker-D manifest;
- v3 fixed-shape CTC eligibility filtering;
- zero record overlap and zero meeting overlap;
- official AMI manual annotations v1.6.2 and pinned Mix-Headset WAV hashes.

The qualified training set has 1,487 utterances, 2,393.989 seconds, 4,638 words
and 17,562 non-space characters. This is 11.9x the audio duration and 11.9x the
record count of v1. The unchanged validation manifest keeps results directly
comparable with the accepted v3 reference.

The first experiment changes only training data. It restores standard v3 CTC,
uses no augmentation or auxiliary objective, and retains 32 epochs, Adam/cosine
and validation-CER checkpoint selection.

## Scope and limitations

ES2005a-d are four sessions from one scenario team, not the full official AMI
training partition. This bounded 2,394-second eligible corpus is a controlled
step before downloading the full approximately 50-hour scenario training set.
The ES2002a validation meeting is disjoint, but it remains a model-selection
set rather than an unbiased final test set.

## Source authority

AMI's official scenario partition places ES2005 in training and documents
Mix-Headset audio and manual annotations:

- https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml
- https://groups.inf.ed.ac.uk/ami/download/
- https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml
