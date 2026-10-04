# Speech ASR contracts

These files define the stable machine-readable boundary around datasets, models
and benchmark results.

- `audio-v1.yaml` — canonical normalized audio/timing contract.
- `speech-sample-v1.schema.json` — normalized dataset record.
- `text-v1.json` — frozen transcript/scoring-normalization policy.
- `benchmark-v1.yaml` — benchmark identity and metrics contract.
- `experiment-result-v1.schema.json` — machine-readable evaluation result.
- `model-v1.yaml` — model-package boundary.

The JSON Schema files are portable descriptions for tools that support JSON
Schema draft 2020-12. Runtime validation does **not** require a third-party JSON
Schema library: `python/speech_asr/contracts.py` enforces the core schema and
cross-field invariants with the Python standard library.

Run:

```bash
python3 examples/speech-asr/tools/validate_contract.py sample.json
python3 examples/speech-asr/tools/validate_contract.py result.json
```

The validator prints the canonical JSON SHA-256 on success and exits non-zero
with field-level errors on failure.
