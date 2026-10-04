# Experiment records

Every model-development attempt is represented as a self-contained experiment.

Suggested layout:

```text
experiments/<id>/
├── proposal.yaml
├── hypothesis.md
├── model_spec.yaml
├── train_config.yaml
├── acceptance.yaml
├── results/
│   ├── training.json
│   ├── compatibility.json
│   ├── accuracy.json
│   ├── hardware.json
│   └── result.json
└── artifacts/              generated; normally stored outside Git
```

Each experiment has a stable ID and may name a parent experiment.

The compact `result.json` is the primary handoff from local execution to the
frontier-model REVIEW phase.

Generated checkpoints, ONNX models and OpenVINO XML/BIN artifacts can be large;
their hashes and storage references belong in the experiment record even when
the binaries themselves are not committed.
