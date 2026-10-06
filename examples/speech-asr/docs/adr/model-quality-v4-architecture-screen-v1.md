# ADR: model-quality-v4 architecture-screen-v1

Status: implemented; v3 control completed
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

The v3 control completed as `exp-00e6b1e434d187d8/attempt-0001` with:

- screen train manifest SHA-256:
  `0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b`;
- 3,904 training records / 6,407.372 seconds;
- all 48 training meetings represented;
- best checkpoint: epoch 12;
- hardware CER: `0.7984219316938317`;
- hardware WER: `0.9930900453465774`;
- empty-hypothesis fraction: `0.26865671641791045`;
- emitted/reference character ratio: `0.39250129585901056`;
- MA2450 p95: `16.7866342 ms`;
- training duration: `404.94855711700075 s`.

A screen candidate must at minimum beat the v3 CER to pass the screen
acceptance gate. Promotion to a larger-budget confirmation requires a stronger
3% relative CER improvement, i.e. CER <= `0.7744692737430167`, together with
healthy emission diagnostics.

The first candidate, `cnn_ctc_v8`, was rejected at CER
`0.8442089500662328` and MA2450 p95 `26.6580554 ms`. Its 256-frame CTC
output also worsened under-emission. That direction is retired.

The active second candidate is `cnn_ctc_v9`, which restores v3's 128-frame
CTC rate and 96-channel width and replaces the five large residual kernels with
eight kernel-7 dilated blocks. See
`model-quality-v4-cnn-ctc-v9-screen.md`.

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
