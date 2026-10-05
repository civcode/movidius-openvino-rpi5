# Speech ASR contracts

These files define the stable machine-readable boundary around datasets, models,
runtime provenance and benchmark results.

- `audio-v1.yaml` — canonical normalized audio/timing contract.
- `speech-sample-v1.schema.json` — normalized dataset record.
- `text-v1.json` — frozen transcript/scoring-normalization policy.
- `benchmark-v1.yaml` — benchmark identity, metrics and measurement policy.
- `experiment-result-v1.schema.json` — transcript/custom-model evaluation result.
- `acoustic-regression-result-v1.schema.json` — one rm_cnn4a vendor score regression run.
- `acoustic-benchmark-result-v1.schema.json` — repeated Phase 6 hardware benchmark result.
- `streaming-v1.json` — Phase 7 chunk/event/timing semantics.
- `streaming-replay-result-v1.schema.json` — deterministic recorded-audio replay result.
- `model-v1.yaml` — model-package and provenance boundary.
- `experiment-proposal-v1.schema.json` — reviewed architecture hypothesis/change request.
- `experiment-model-spec-v1.schema.json` — reviewed model/frontend/export declaration.
- `train-config-v1.schema.json` — deterministic training and dataset inputs.
- `acceptance-policy-v1.schema.json` — required gates, thresholds and retry policy.
- `experiment-lifecycle-v1.schema.json` — immutable request, attempts, summaries and history.
- `deployment-manifest-v1.schema.json` — hash-bound controller-to-edge deployment handoff.

The three `.yaml` files intentionally contain JSON syntax. JSON is valid YAML
1.2, and this keeps the root contracts parseable with the Python standard
library instead of adding a YAML dependency solely for contract validation.

The JSON Schema files remain portable descriptions for tools that support JSON
Schema draft 2020-12. Runtime validation does **not** require a third-party JSON
Schema library: `python/speech_asr/contracts.py` enforces the core schema and
cross-field invariants.

Run:

```bash
python3 examples/speech-asr/tools/validate_contract.py \
    examples/speech-asr/contracts/audio-v1.yaml
python3 examples/speech-asr/tools/validate_contract.py \
    examples/speech-asr/contracts/benchmark-v1.yaml
python3 examples/speech-asr/tools/validate_contract.py sample.json
python3 examples/speech-asr/tools/validate_contract.py result.json
```

The validator infers the document kind from `$.schema`, prints the canonical
JSON SHA-256 on success, and exits non-zero with field-level errors on failure.

## Phase 0 invariants

Contract validation freezes the following assumptions for version 1:

- canonical audio is 16 kHz mono f32le;
- canonical timing is integer sample index from normalized-audio start;
- scoring uses `text-v1`;
- benchmark-v1 uses one warmup and five measured iterations, with p50/p95
  latency aggregation;
- model frontends and decoders are declared by each model package;
- model packages declare source/spec/artifact provenance;
- hardware results identify repository revision, runtime, target and model/data
  hashes.


## Phase 6 measurement policy

`benchmark-v1` freezes one warmup plus five measured iterations. The semantic
validator now rejects any other count for benchmark version 1. For rm_cnn4a,
each iteration is an independent full vendor-fixture invocation; warmup is
excluded from aggregate statistics. The repeated worker result records
cross-run repeatability as measured evidence rather than applying an invented
stability threshold.


## Phase 7 streaming contract

`streaming-v1.json` fixes the semantics of sample-index timing, midpoint overlap
ownership, left/right context expansion, cumulative transcript events and
consecutive-prefix stabilization. Chunk duration, overlap, context and VAD
threshold remain explicit runtime parameters and are written into every replay
result.

The scripted update fixture is deliberately separate from a model contract. It
lets the project validate streaming mechanics before a raw-audio model exists.

## Phase 9 experiment lifecycle

Phase 9 experiment identity is content-derived from the immutable reviewed
request. Semantic validation additionally enforces parent lineage, request/document
hashes, legal attempt state transitions, immutable artifact/result references,
retry-policy structure, deterministic history ordering, and exact controller /
worker Git revision matching for a deployment manifest.

The JSON schemas describe the portable shape; `speech_asr.contracts` and
`speech_asr.experiment` enforce cross-field identity and lifecycle invariants.
