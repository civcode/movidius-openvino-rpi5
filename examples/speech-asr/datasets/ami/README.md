# AMI dataset adapter

AMI is the canonical corpus for speech-ASR development and benchmarking. The
adapter converts pinned AMI NXT annotations and source audio into the project's
normalized speech-sample contract. Corpus audio and generated clips live under
`work/` and are not committed.

## Version-1 split definitions

Two frozen split specifications are checked in:

- `splits/smoke-v1.json`: four explicit speaker-A segments from ES2002a for
  fast plumbing/regression tests;
- `splits/benchmark-v1.json`: every annotated segment for speakers A-D in
  ES2002a, using the same source WAV and annotation archive.

Using all segments for the benchmark avoids a hand-curated list while remaining
fully frozen: the annotation archive version/hash and the source audio hash are
part of the split spec.

The sources are the official AMI manual annotations **v1.6.2** and the official
ES2002a **Mix-Headset** WAV. Downloads are rejected when SHA-256 differs from
the split specification. AMI signals/transcription are CC BY 4.0; downstream
use must preserve AMI attribution.

## Python environment

The adapter is stdlib-only, but it is still invoked through the repository-wide uv-managed tools environment so host Python remains untouched.

## Prepare data

From the repository root:

```bash
./scripts/python.sh examples/speech-asr/datasets/ami/prepare_ami.py --subset smoke
./scripts/python.sh examples/speech-asr/datasets/ami/prepare_ami.py --subset benchmark
```

An explicit split file can be used with `--spec`. Relative cache/output paths
are resolved from the repository root.

Defaults:

```text
download cache: work/speech-asr/ami/downloads/
smoke output:   work/speech-asr/ami/ami-smoke-v1/
benchmark:      work/speech-asr/ami/ami-benchmark-v1/
```

Each output contains:

```text
<split-id>/
├── audio/
│   └── *.f32
├── manifest.jsonl
└── provenance.json
```

The `.f32` files are raw little-endian float32 mono samples at 16 kHz.

## Offline verification

After preparation, or after copying a prepared dataset to another machine:

```bash
./scripts/python.sh examples/speech-asr/datasets/ami/prepare_ami.py \
    --subset smoke \
    --verify-only
```

Verification checks:

- split-spec hash;
- manifest hash and record count;
- every normalized record against the speech-sample contract;
- unique sample IDs;
- each clip's expected byte length;
- each clip SHA-256 against manifest metadata;
- the complete per-clip hash map;
- a logical-tree SHA-256 covering manifest + normalized clips.

No network access is needed for `--verify-only`.

## Determinism rules

- split specs are validated before network/filesystem work;
- annotation times are parsed with `Decimal`, not binary floating point;
- seconds-to-sample conversion uses round-half-up at 16 kHz;
- AMI segment annotations define clip boundaries;
- the segment's NXT word range defines lexical words;
- `all_segments=true` expands in annotation-file order;
- punctuation and non-lexical NXT elements are omitted from lexical word timing;
- text is normalized by the shared `text-v1` implementation;
- PCM16 samples are converted deterministically to f32le by
  `sample / 32768.0`;
- JSONL keys/separators are canonicalized;
- provenance includes source, manifest, per-clip and logical-tree hashes.

Synthetic NXT/WAV fixtures exercise this complete path without network access.

## Normalized representation

Runtime/evaluation code consumes `manifest.jsonl`, never AMI-native XML.
Records follow `contracts/speech-sample-v1.schema.json` and carry
clip-relative integer word/sample timing plus source and normalized-audio
provenance.

## Scope

Phase 1 intentionally prepares dataset material only. Model inference, feature
extraction beyond canonical audio normalization, and ASR quality are later
phases.


## Real-source acceptance

Phase 1 was accepted against the pinned official AMI files, not only the
synthetic fixtures. The source files matched the split-spec hashes exactly and
both splits were prepared twice from clean output directories.

Frozen output identities:

| Split | Records | manifest SHA-256 | logical-tree SHA-256 |
|---|---:|---|---|
| `ami-smoke-v1` | 4 | `2b38aba98d0af830f5f0cfecdbda819f3b4f339f77c89c2f5f8090217bfc8922` | `21c529d2d5a02f4436d929026538f1a7e1c602e34d8723a6c09ad9acd3123e2d` |
| `ami-benchmark-v1` | 277 | `9f4444c6c0e54cf25d92a69723727bb46a73632e34ec33402c71beca1b64cd3e` | `25884bf377243ae1316e0b4a17d18a032325c4c00229deba05ce4465bee93f41` |

Those expected values are stored in the split specifications themselves.
`--verify-only` therefore verifies not only internal consistency but also that
the prepared output is exactly the frozen Phase 1 corpus identity.
