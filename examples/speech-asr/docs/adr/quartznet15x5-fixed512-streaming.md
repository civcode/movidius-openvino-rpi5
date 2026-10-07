# ADR: fixed-T=512 QuartzNet streaming on MA2450

Status: implemented; physical WER qualification pending
Date: 2026-10-07

## Context

Whole-utterance qualification preserved the source model's variable time axis
and therefore created 152 exact MYRIAD shape sessions on LibriSpeech dev-clean.
That methodology exposed a catastrophic MA2450 execution boundary between
`T=3088` and `T=3104`, but it is not a production requirement.

The physical residency experiment also proved that at least four QuartzNet
graphs can coexist on one MA2450. Multiple resident shapes are therefore
available for future applications, but they are unnecessary if one fixed
streaming shape preserves ASR quality.

## Decision

Qualify a production-oriented policy using exactly one `[1,64,512]` QuartzNet
graph loaded once and retained for the full evaluation.

The source frontend remains unchanged. Each inference window contains 81,760
audio samples (5.11 s), which produces 511 valid frontend frames plus one zero
pad frame for an exact 512-frame tensor.

Default window starts advance by 128 QuartzNet output frames, or 40,960 audio
samples (2.56 s). Consecutive full windows therefore overlap by 40,800 samples
(2.55 s).

Window starts are aligned to the network's output lattice. Overlapping logits
are stitched before decoding: each global CTC frame is owned by the window
where that frame is closest to the fixed 256-frame window center. Ties retain
the earlier window. The stitched global logits are decoded with one ordinary
greedy CTC collapse, so duplicate symbols spanning a window boundary remain
subject to normal CTC semantics.

The last partial waveform window is passed through the unchanged source
frontend and its feature tensor is zero-padded to `T=512`. Raw audio is not
zero-extended before per-feature normalization.

## Evidence required

For the physical MA2450 run:

- one persistent fixed-`T=512` MYRIAD network;
- one unmeasured warmup;
- first-window ONNX/MYRIAD valid-frame argmax agreement at least `0.99` as a non-catastrophic hardware smoke gate;
- all 2,703 LibriSpeech dev-clean utterances;
- full-corpus WER/CER after logit stitching, which remains the authoritative semantic acceptance gate;
- WER <= `0.05` as the initial fixed-window acceptance ceiling;
- inference RTF includes all overlapping-window inference work;
- model load time is reported separately.

The evaluator also supports `--engine onnx` so the same fixed-window policy can
be run without MYRIAD. This isolates chunking/stitching accuracy from VPU
numerical execution if the physical result regresses.

## Commands

Local ONNX policy diagnostic:

```bash
./scripts/evaluate-quartznet15x5-reference-fixed512.sh \
  --engine onnx \
  --max-samples 100
```

Physical Pi 5 + MA2450 run from the controller host:

```bash
./scripts/run-quartznet15x5-reference-fixed512-edge.sh
```

Use `--max-samples N` for a non-authoritative diagnostic. Full qualification
must run all 2,703 dev-clean utterances.
