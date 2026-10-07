# Speech runtime boundary

The speech runtime is the Pi/host-side execution layer between normalized model
inputs and model outputs.

## Canonical audio

`python/speech_asr/audio.py` implements the `audio-v1` boundary:

- uncompressed integer PCM WAV input (8/16/24/32-bit);
- deterministic channel averaging to mono;
- deterministic linear resampling to 16 kHz when required;
- canonical raw little-endian float32 (`f32le`) I/O;
- sample-index slicing.

The AMI smoke source is already 16 kHz, so its corpus adapter preserves exact
annotation sample coordinates rather than resampling before segment extraction.

## Frontend profiles

`python/speech_asr/features.py` defines a narrow `FrontendProfile` protocol:

```text
CanonicalAudio -> FeatureTensor
```

A `FeatureTensor` declares dtype, layout, shape and values and can be dumped as
f32le plus canonical metadata/hash for golden-tensor comparisons.

There is intentionally no global log-mel/MFCC default. A model package owns its
frontend profile. `RawAudioFrontend` exists only as a plumbing/test fixture.

## Runtime authority

The runtime may:

- load a declared model package;
- prepare tensors from the model's declared frontend profile;
- invoke CPU/reference or MYRIAD inference;
- measure load/inference timing;
- expose deterministic diagnostics;
- return model outputs to the decoder/evaluator.

It may not:

- redesign a model;
- change dataset splits;
- change scoring rules;
- silently change a model frontend;
- tune parameters while reporting the result as the original experiment.

Model-specific behavior is selected through the model contract/adapter.

## cnn_ctc_v19 CPU decoder

The frozen cnn_ctc_v19 acoustic model still ends at CTC logits on MYRIAD. The
selected post-acoustic runtime decoder is a separate, hashed artifact so the
validated v19 model spec and OpenVINO IR identity do not change.

The selected decoder is CTC prefix beam width 8 with per-frame token top-k 12,
a suffix-backoff character 5-gram LM, LM weight 0.30 and word-boundary bonus
-0.20. Its development sweep reduced validation WER from 0.598575 (greedy) to
0.549773.

Build the runtime artifact from the frozen sweep result with
`scripts/freeze-cnn-ctc-v19-decoder.sh`. The artifact serializes the trained LM
counts and all selected decoder parameters, so the Raspberry Pi does not need
the AMI training manifest at runtime.

`scripts/decode-cnn-ctc-v19-logits.sh` is the model-backed decoder boundary for
one physical logits tensor. `scripts/benchmark-cnn-ctc-v19-decoder.sh` measures
the same decoder over a cached physical-logit corpus and reports CPU decoder
p50/p95 latency plus WER/CER. Greedy remains available in the physical evaluator
when no decoder artifact is supplied, preserving historical acoustic-baseline
comparability.

For the final paired Pi deployment measurement, run the controller from the
main Oberon checkout after defining the source experiment, attempt and frozen
decoder artifact:

```bash
git pull --ff-only

EXP=work/speech-asr/experiments/exp-f4adb44ab833e896
ATTEMPT=attempt-0002

./scripts/run-cnn-ctc-v19-deployed-edge.sh \
  --experiment "$EXP" \
  --attempt "$ATTEMPT"
```

By default the controller uses
`$EXP/attempts/$ATTEMPT/results/decoder-artifact-v1.json`; pass
`--decoder-artifact` only to override that path.

The controller requires both the Oberon checkout and the edge human checkout to
remain on `main` at the same commit. It verifies the frozen 1,273-record
validation-manifest hash, validates the decoder artifact, stages the selected
v19 FP16 IR plus decoder artifact, and mirrors the frozen validation manifest
with every referenced audio clip into the isolated edge run directory. The
dataset mirror is built with hard links on Oberon when possible, so it does not
duplicate the local corpus before rsync. The controller then runs both the
standard edge runtime preflight and a deployed-input/audio preflight before
invoking a fresh edge evaluation under the existing exclusive MYRIAD lock.
Decoder CPU affinity defaults to
cores 0-3 with OpenMP/MKL/OpenBLAS restricted to one thread.

The edge evaluator recomputes every physical MYRIAD inference for this
measurement rather than reusing the previous acoustic cache. Evidence is pulled
back to:

```text
work/speech-asr/deployed-evaluations/
  exp-f4adb44ab833e896-attempt-0002-decoder-v1/
    result.json
    evaluator.log
```

The controller prints the measured MYRIAD p50/p95, decoder p50/p95/mean,
paired acoustic-plus-decoder p50/p95/mean, combined realtime factor, and whether
WER/CER still exactly match the frozen decoder result.


## Evaluation worker

Phase 6 adds a host-side non-interactive worker:

```bash
./scripts/benchmark-speech.sh --model rm_cnn4a --backend myriad --platform arm64
```

The worker refuses a non-native target, runs the frozen 1+5 measurement policy,
preserves every raw log/single-run result, checks model/fixture/runtime identity
between iterations, and emits one aggregate result plus a summary derived from
that result. It does not change the model, fixture, scoring rules or measurement
counts.


## Recorded-audio streaming

Phase 7 is implemented in `python/speech_asr/streaming.py`.

The chunker partitions overlap at deterministic midpoint boundaries: inference
windows may overlap and may include extra left/right context, but each source
sample belongs to exactly one emission region. This is the boundary future
model adapters use to avoid duplicate output from overlapping chunks.

The module also provides:

- a bounded absolute-index sample ring buffer;
- an energy-VAD interface;
- cumulative partial/final transcript events;
- consecutive-prefix token stabilization;
- sample-index-derived event/stabilization latency;
- offline-versus-streaming WER/CER comparison.

The current recorded replay adapter uses scripted cumulative hypotheses because
`rm_cnn4a` is not a raw-audio ASR model. Model-backed decoding will plug into
the same `ChunkDecoder` interface once the custom model exists.
