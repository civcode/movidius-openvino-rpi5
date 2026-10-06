# ADR: model-quality-v4 fresh Full-corpus-ASR development boundary

Status: implemented; awaiting first source freeze and baseline execution
Date: 2026-10-06
Model baseline: `cnn_ctc_v3`
Parent quality reference: `exp-87538823d2bf1562/attempt-0001`

## Evidence

The sealed Full-corpus-ASR evaluation established two separate facts:

- the MA2450/OpenVINO execution path reproduces the frozen reference closely;
- the frozen v3 model still has poor generalization quality.

The hardware path therefore has enough fidelity and latency margin that the
next development step should improve the training/validation evidence rather
than change the deployment graph.

The previous model-quality boundary is no longer suitable for new model
selection. ES2002a has participated repeatedly in checkpoint, architecture and
data decisions. It remains historical model-selection evidence, but it is
retired from future development.

## Partition decision

Use the official AMI **Full-corpus-ASR** partition as the authority for the next
development boundary.

Restrict this generation to Edinburgh scenario meetings so the corpus domain
remains comparable while substantially expanding the number of independent
scenario teams.

Training uses the official Full-corpus-ASR training groups:

- ES2003;
- ES2005, ES2006, ES2007;
- ES2008, ES2009, ES2010;
- ES2012, ES2013, ES2014, ES2015, ES2016.

Every a/b/c/d meeting and every A/B/C/D annotated speaker segment is included
before the frozen cnn_ctc_v3 eligibility filter is applied.

ES2002 is deliberately omitted even though it belongs to the official
Full-corpus-ASR training partition because this project has already reused
ES2002a for model and checkpoint selection.

Validation uses the official Full-corpus-ASR development group **ES2011a-d**.
It is training-forbidden and checkpoint-selection-allowed.

The already sealed Full-corpus-ASR test groups remain excluded from development:

- EN2002;
- ES2004;
- IS1009;
- TS3003.

No metric from that sealed test boundary may select a model, objective,
frontend, decoder, threshold, hyperparameter or training-data change.

## Source locking

Existing AMI split specs pin source WAV SHA-256 values. The newly introduced
meetings must receive the same treatment before preparation.

`freeze_model_quality_v4_sources.py` performs a one-time official HTTPS
acquisition into the normal AMI download cache, computes SHA-256 for every
selected Mix-Headset WAV, and writes:

```text
work/speech-asr/ami/model-quality-v4-source-lock/
├── source-lock.json
├── train.split.json
└── validation.split.json
```

Once any of these files exists, the tool refuses a partial or changed lock.
Subsequent runs verify the cached source bytes against the frozen lock and do
not silently refresh them.

The pinned manual annotation archive remains v1.6.2 with its existing repository
SHA-256.

## Qualification

The normal fixed-shape cnn_ctc_v3 eligibility contract is applied independently
to training and validation. Qualification requires:

- the exact reviewed train meeting set;
- exactly ES2011a-d for validation;
- zero record overlap;
- zero meeting overlap;
- no ES2002 development records;
- no sealed held-out meetings;
- a training population larger than model-quality-v3's 4,429 eligible records;
- a training duration larger than model-quality-v3's 7,150.12 seconds.

The qualification records the policy hash, source-lock hash and both frozen
split-spec hashes.

### Reviewed ES2011c annotation anomaly

The pinned AMI v1.6.2 annotations contain one observed ES2011c segment whose
`transcriber_start` is after its `transcriber_end`. At 16 kHz the annotated
interval is `1249952:1248016` samples (78.122 s to 78.001 s). AMI's public
data-problems material does not publish a correction for this interval, so the
pipeline does not swap, infer, or otherwise invent segment boundaries.

The generic AMI preparer classifies a non-positive annotated segment interval
and records the excluded segment in preparation provenance. For
model-quality-v4, qualification is stricter:

- training must contain zero such exclusions;
- validation must contain exactly one exclusion;
- that exclusion must be ES2011c with the exact reviewed sample interval above;
- any additional or changed malformed interval stops qualification for review.

This preserves every valid ES2011 segment unchanged while making the single
source annotation defect explicit and auditable.

## Baseline experiment

The first model-quality-v4 experiment deliberately changes **data only**.

It keeps:

- model `cnn_ctc_v3`;
- frontend `logmel-v1`;
- vocabulary unchanged;
- standard CTC;
- no SpecAugment;
- no blank-logit penalty;
- 32 epochs;
- batch size 1;
- Adam/cosine training;
- validation-CER checkpoint selection;
- fixed ONNX/OpenVINO/MYRIAD contracts.

The parent is the accepted scaled-data v3 experiment
`exp-87538823d2bf1562`. The new validation population defines a new
development baseline, so WER/CER acceptance ceilings are deliberately unset.
Compatibility and physical p95 latency gates remain active.

## Commands

Prepare and freeze the new development data on oberon:

```bash
./scripts/prepare-speech-model-data-v4.sh
./scripts/prepare-speech-model-data-v4.sh --verify-only
```

Review:

```bash
cat work/speech-asr/ami/model-quality-v4-source-lock/source-lock.json
cat work/speech-asr/ami/model-quality-v4/qualification.json
```

Provision the fresh validation data to edge:

```bash
./scripts/provision-speech-model-data-v4-edge.sh
```

Initialize and execute the unchanged v3 baseline:

```bash
EXP="$(
  ./scripts/init-cnn-ctc-v3-model-quality-v4.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"
./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

Only after this baseline is reviewed should a new model/data/objective
intervention be designed.
