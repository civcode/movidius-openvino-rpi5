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


## Phase 10 implementation

The execution side is now concrete:

- `init_cnn_ctc_v1_experiment.py` creates the frozen baseline acceptance
  request without mutating architecture;
- `run_experiment.py` is the oberon controller;
- `edge_worker.py` is the exact-commit, hash-validating MYRIAD worker.

The controller accepts only reviewed Phase 9 files. The executor registry now
contains the qualified `cnn_ctc_v1` baseline and the explicitly reviewed
Phase 11 `cnn_ctc_v2` generation. Unknown model IDs or undeclared architecture
parameters fail instead of being interpreted by a local model.

The final controller artifact is the compact Phase 9 `results/result.json`,
which always returns to `AWAIT_REVIEW`. A rejected acceptance policy is
reported as evidence; it does not authorize the execution side to enter DESIGN.


## Phase 10 acceptance

The local execution boundary was physically accepted on 2026-10-05 using
`exp-f915ec624a63caf6/attempt-0001`. The controller completed the approved
baseline through the exact-commit edge worker, physical MYRIAD evaluation,
result collection and acceptance evaluation, then stopped at
`AWAIT_REVIEW` as designed.

Phase 10 is therefore qualified as execution infrastructure. Phase 11 may add
new reviewed model executors, but it must not broaden this component's design
authority.


## Phase 11 generation 1

`cnn_ctc_v2` is the first new executor added after REVIEW. Its experiment
model declaration includes the complete frozen architecture tuple, and the
controller checks exact equality before execution.

For this generation, compatibility authority is stricter than Phase 10:
initialized ONNX/OpenVINO conversion and one physical MYRIAD inference must pass
before the attempt enters full CUDA training. This remains deterministic
execution logic; the controller still cannot alter kernels, width, depth,
frontend, thresholds, or training policy in response to a result.
