# Recorded-audio streaming

Phase 7 builds the streaming mechanics before there is a custom raw-audio ASR
model.

## Boundary

The runtime consumes canonical 16 kHz mono f32le audio and produces deterministic
chunk metadata plus cumulative transcript events:

```text
CanonicalAudio
  -> overlapping nominal chunks
  -> left/right inference context
  -> midpoint emission ownership
  -> ChunkDecoder
  -> cumulative partial hypotheses
  -> stabilization tracker
  -> final event
  -> offline/streaming comparison
```

`ChunkDecoder` is the model adapter boundary. The current
`ScriptedCumulativeDecoder` is only a timing/mechanics fixture.

## Chunk semantics

For adjacent overlapping nominal chunks, the overlap is divided at one midpoint.
The left chunk owns samples before the midpoint and the right chunk owns samples
from the midpoint onward. Therefore emission regions are contiguous and cover
the complete recording exactly once.

Left/right context never changes emission ownership. It only changes the samples
available to inference. A chunk becomes observable at
`inference_end_sample`, which is used as the deterministic replay event time.

## Transcript events

Decoder hypotheses are cumulative and normalized with `text-v1`. The replay
emits partial events whenever the cumulative hypothesis changes and exactly one
final event at end of recording.

A token is stabilized after it appears in the common prefix of the configured
number of consecutive cumulative hypotheses. Stabilized tokens may not later be
revised. Stabilization delay is measured from the token's first observed
appearance to its stable event, using sample coordinates.

## VAD

`VoiceActivityDetector` is an interface. `EnergyVad` supplies the initial RMS
implementation. VAD is disabled by default and decoder gating is explicit; this
avoids silently changing transcript semantics during baseline replay.

## AMI replay mechanics

Prepared AMI manifests already contain clip-relative word timings. Generate a
scripted fixture from one record:

```bash
./scripts/python.sh examples/speech-asr/tools/make_streaming_updates.py \
    work/speech-asr/ami/ami-smoke-v1/manifest.jsonl \
    --output work/speech-asr/streaming/updates.json
```

Then run the command printed by that tool. The expected final transcript comes
from the normalized record and offline-versus-streaming comparison uses the
normal deterministic scorer.

This does **not** claim reference words are ASR output. It validates replay,
chunking, timing and event semantics until Phase 8 provides a model-backed
decoder.

## Acceptance

Phase 7 code is considered implemented when hardware-free tests cover:

- exact emission coverage with overlap;
- context invariance of emission ownership;
- ring-buffer absolute indexing;
- VAD interface behavior;
- partial/final event ordering;
- token stabilization and revision rejection;
- deterministic repeated replay;
- offline-versus-streaming comparison;
- manifest-to-script timing conversion.

A real prepared AMI clip replay remains the final acceptance evidence before the
phase is marked complete.
