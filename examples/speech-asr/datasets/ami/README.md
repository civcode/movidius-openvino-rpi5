# AMI dataset adapter

AMI is the canonical corpus for speech-ASR development and benchmarking.

This directory will contain preparation code and metadata only. Full corpus
audio should be downloaded/prepared locally and should not be committed by
default.

## Normalized representation

AMI-native annotations must be converted into a project JSONL manifest before
runtime/evaluation code consumes them.

Each normalized record should contain at least:

```json
{
  "id": "stable-sample-id",
  "audio": "relative/or/resolved/path.wav",
  "sample_rate": 16000,
  "start_sample": 0,
  "end_sample": 16000,
  "text": "normalized transcript",
  "words": [
    {"text": "normalized", "start_sample": 0, "end_sample": 4000}
  ]
}
```

Timing is stored as integer sample offsets in normalized 16 kHz audio.

## Planned benchmark tiers

- **smoke**: very small, used for plumbing/regression checks;
- **benchmark**: roughly tens of minutes, used during development optimization;
- **validation**: larger held-out set used before accepting architecture changes;
- **full**: corpus-scale experiments/training where appropriate.

The exact sample lists are versioned manifests. Once a benchmark version is
frozen, an agent may not alter those lists to improve a score.

## Responsibilities of the adapter

The adapter will:

1. locate/download the requested AMI material;
2. normalize selected audio to the audio-v1 contract;
3. parse AMI annotations;
4. convert annotation timing to sample indices;
5. apply the versioned text-normalization policy;
6. emit reproducible manifests and content/provenance hashes.
