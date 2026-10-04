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
