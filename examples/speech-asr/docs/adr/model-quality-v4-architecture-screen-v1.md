# ADR: model-quality-v4 architecture-screen-v1

Status: implemented; v3 control pending
Date: 2026-10-06

## Problem

Full model-quality-v4 training runs consume roughly one hour for 32 epochs.
Current CER remains far from the desired quality level, so architecture
development needs a cheaper ranking loop before paying the full training cost.

A proxy must not introduce a new model-selection leak. In particular, it must
not choose examples based on model errors, transcript difficulty, duration or
held-out behavior.

## Decision

Create a deterministic architecture screen derived only from the already
frozen model-quality-v4 qualified training manifest.

The screen selects a record when:

```text
u64_be(sha256(sample_id)[0:8]) % 4 == 0
```

The rule is identified as
`sha256-id-first-u64-be-mod4-eq0-v1`.

Properties:

- source training manifest is frozen at
  `6025d17f08c1815d1710c3ab56cea51a34365f398e9877b4aefec234e64e4096`;
- target training fraction is approximately 25%;
- every one of the 48 training meetings must remain represented;
- the full frozen 1,273-record ES2011 validation manifest remains unchanged at
  `fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088`;
- training budget is 12 epochs, batch size 1;
- checkpoint selection remains validation CER;
- sealed held-out data remains forbidden.

The screen manifest and its provenance are generated under:

```text
work/speech-asr/ami/model-quality-v4-architecture-screen-v1/
  train.manifest.jsonl
  screen.json
```

The generator is deterministic and supports `--verify-only`.

## Ranking protocol

First establish a `cnn_ctc_v3` control using the screen. New architectures
must use the exact same screen manifest, 12-epoch budget and full ES2011
validation manifest.

A screen result is directional evidence, not a final promotion result.
Candidates that materially beat the screen control should be confirmed at a
larger budget before a full 32-epoch / 100%-data run.

No fixed relative-CER promotion threshold is frozen before the v3 screen
control exists. The first control determines the proxy baseline and the
observed variance/trajectory available for later promotion rules.

## Commands

```bash
./scripts/prepare-speech-architecture-screen-v1.sh
./scripts/prepare-speech-architecture-screen-v1.sh --verify-only

cat work/speech-asr/ami/model-quality-v4-architecture-screen-v1/screen.json

EXP="$(
  ./scripts/init-cnn-ctc-v3-architecture-screen.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"

./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

No new validation provisioning is required if the frozen ES2011 validation
boundary is already present on the edge worker.
