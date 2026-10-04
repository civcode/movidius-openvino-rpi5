# Agent authority boundary

This directory documents the automation boundary for agent-driven development.

The core rule is simple:

**Frontier models design and review architectures. Local models execute and
measure declared experiments.**

The local execution side may perform repetitive tasks such as:

- training;
- checkpoint management;
- ONNX export;
- OpenVINO conversion;
- Pi/MYRIAD deployment;
- benchmark execution;
- log parsing;
- deterministic result aggregation;
- declared retries and routine diagnostics.

It may not independently change model topology in response to a failed or poor
experiment.

The project state should explicitly identify the current phase
(`DESIGN`, `APPROVED`, `EXECUTE`, `EVALUATE`, `AWAIT_REVIEW`,
`REVIEW`) so Pi virtual-model routing can later choose a suitable physical
model based on authority and task class rather than prompt wording alone.

See `../docs/agent-workflow.md` for the full state machine and handoff rules.
