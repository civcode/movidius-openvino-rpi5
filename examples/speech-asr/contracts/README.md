# Speech ASR contracts

These files define the stable machine-readable boundary around datasets, models,
runtime provenance and benchmark results.

- `audio-v1.yaml` — canonical normalized audio/timing contract.
- `speech-sample-v1.schema.json` — normalized dataset record.
- `text-v1.json` — frozen transcript/scoring-normalization policy.
- `benchmark-v1.yaml` — benchmark identity, metrics and measurement policy.
- `experiment-result-v1.schema.json` — transcript/custom-model evaluation result.
- `acoustic-regression-result-v1.schema.json` — rm_cnn4a vendor score regression result.
- `model-v1.yaml` — model-package and provenance boundary.

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
