# AMI dataset adapter

AMI is the canonical corpus for speech-ASR development and benchmarking. The
adapter converts pinned AMI NXT annotations and source audio into the project's
normalized speech-sample contract. Corpus audio and generated clips live under
`work/` and are not committed.

## Pinned smoke subset

`splits/smoke-v1.json` currently pins:

- AMI manual annotations **v1.6.2**;
- meeting **ES2002a**;
- the official **Mix-Headset** WAV;
- speaker A segments `sync.4`, `sync.6`, `sync.25`, and `sync.27`;
- SHA-256 for both the annotation archive and source WAV.

The source URLs point to the official AMI corpus servers. Downloads are rejected
when their SHA-256 differs from the split specification.

The AMI signals/transcription are CC BY 4.0; downstream use must preserve AMI
attribution.

## Prepare the smoke data

From the repository root:

```bash
python3 examples/speech-asr/datasets/ami/prepare_ami.py
```

Defaults:

```text
download cache: work/speech-asr/ami/downloads/
output:         work/speech-asr/ami/smoke-v1/
```

The output is:

```text
smoke-v1/
├── audio/
│   └── *.f32
├── manifest.jsonl
└── provenance.json
```

The `.f32` files are raw little-endian float32 mono samples at 16 kHz. Each
manifest record has clip-relative integer sample timing, `text-v1` normalized
transcript/word text, source hashes, and the normalized clip SHA-256.

## Determinism rules

- annotation times are parsed with `Decimal`, not binary floating point;
- seconds-to-sample conversion uses round-half-up at 16 kHz;
- AMI segment annotations define clip boundaries;
- the segment's NXT word range defines the lexical words;
- punctuation and non-lexical NXT elements are omitted from lexical word
  timing;
- text is normalized by the shared `text-v1` implementation;
- PCM16 source samples are converted deterministically to f32le by
  `sample / 32768.0`;
- JSONL keys and separators are canonicalized before hashing.

A synthetic NXT/WAV fixture tests the complete preparation path without
requiring network access.

## Normalized representation

Runtime/evaluation code consumes `manifest.jsonl`, not AMI-native XML. A
record follows `contracts/speech-sample-v1.schema.json`, for example:

```json
{
  "schema": "speech-asr/sample",
  "version": 1,
  "id": "ami-ES2002a-A-4",
  "audio": {
    "path": "audio/ES2002a.A.4.f32",
    "sample_rate_hz": 16000,
    "channels": 1,
    "sample_type": "float32",
    "encoding": "f32le",
    "start_sample": 0,
    "end_sample": 56752
  },
  "transcript": {
    "text": "hi i'm david and i'm supposed to be an industrial designer",
    "normalization": "text-v1",
    "words": []
  }
}
```

## Benchmark tiers

- **smoke**: small plumbing/regression set (implemented);
- **benchmark**: tens of minutes for optimization (planned);
- **validation**: held-out acceptance set (planned);
- **full**: corpus-scale training/experiments (planned).

Once a benchmark version is frozen, an agent must not alter its selected
meeting/segment list to improve a score.
