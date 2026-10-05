# Agent-driven model development workflow

## Principle

Architecture design and experiment execution are separate phases with separate
authority.

Frontier models are used for high-leverage architecture design and review.
Local models are used for repetitive training, export, test orchestration,
benchmark execution, log reduction and routine diagnostics.

The local execution agent must not silently redesign the network.

## State machine

```text
DESIGN
  |  frontier architect produces proposal/spec
  v
APPROVED
  |  explicit handoff
  v
EXECUTE
  |  local agent + deterministic tools
  v
EVALUATE
  |  deterministic scoring + local orchestration
  v
AWAIT_REVIEW
  |  summarized evidence
  v
REVIEW
  |  frontier model accepts/rejects/derives next hypothesis
  +-------------------------------> DESIGN
```

An execution agent may move work toward `AWAIT_REVIEW`. It may not promote
itself from `AWAIT_REVIEW` into a new architecture DESIGN decision.

## Roles and authority

| Role | Preferred intelligence | Architecture changes | Run training/tests | Change benchmark |
|---|---|---:|---:|---:|
| Architect | frontier | yes | normally no | no |
| Reviewer | frontier | yes, for next proposal | no | no |
| Experiment planner | frontier or strong local | declared training/runtime parameters only | no | no |
| Executor | local | no | yes | no |
| Evaluator/orchestrator | local + deterministic code | no | yes | no |
| Metric implementation | deterministic code | no | n/a | only through reviewed benchmark-version change |

Pi virtual-model routing can later map these phases to different physical
models. The phase itself should be explicit project state, not guessed only
from prompt wording.

## Handoff artifacts

Natural-language chat is not the API between phases. Each experiment is
represented by versioned files.

A design handoff is the reviewed Phase 9 request bundle:

```text
request/experiment.json
request/proposal.json
request/model-spec.json
request/train-config.json
request/acceptance.json
```

The proposal document contains the hypothesis/rationale directly. The immutable
experiment ID is derived from the parent, source revision, benchmark identity
and canonical hashes of the four reviewed request documents.

Execution/evaluation adds per-attempt evidence:

```text
attempts/attempt-NNNN/attempt.json
attempts/attempt-NNNN/deployment-manifest.json
attempts/attempt-NNNN/results/training.json
attempts/attempt-NNNN/results/compatibility.json
attempts/attempt-NNNN/results/hardware.json
attempts/attempt-NNNN/results/accuracy.json
attempts/attempt-NNNN/results/result.json
```

Large model artifacts remain in controller storage and are referenced by
repository-relative path plus SHA-256 from the attempt record.

Large generated artifacts are storage concerns and do not need to live in Git.

The frontier reviewer should normally receive the model spec, compact result
summary, relevant failure excerpts and experiment-history summary rather than
raw training logs.

## Execution pipeline

The controller/worker split for CUDA training on oberon and physical MYRIAD
execution on edge is specified in
[`remote-experiment-orchestration.md`](remote-experiment-orchestration.md).

Before expensive training, use a compatibility gate:

```text
instantiate model
 -> dummy forward
 -> export ONNX
 -> OpenVINO conversion
 -> MYRIAD compatibility/load probe where practical
 -> only then spend full training budget
```

Training should also support staged budgets:

```text
small smoke training
 -> reject obviously bad/broken candidates
 -> medium training
 -> full training
 -> hardware evaluation
```

The exact promotion thresholds are experiment-policy parameters and must be
recorded.

## Phase 10 implementation

The local execution agent is implemented as deterministic repository tooling,
not a free-form shell agent. Oberon invokes
`scripts/run-speech-experiment.sh`; the controller consumes only a validated
Phase 9 experiment request, uses the fixed executor registry, and hands a
hash-bound deployment manifest to the edge worker.

The initial executor registry contains only `cnn_ctc_v1`. This restriction is
part of the authority boundary: a new model ID or additional architecture fields
are rejected until REVIEW/DESIGN adds an explicit executable model path.

Transient retries remain separate Phase 9 attempt IDs. Re-running the same
top-level experiment command after a retryable terminal failure is permitted
only when the frozen retry policy allows another attempt.

## Failure behavior

The local executor may:

- retry declared transient operations;
- collect diagnostics;
- apply mechanical fixes that do not change model semantics, when allowed by
  the experiment policy;
- mark an experiment failed or blocked.

It may not respond to a poor WER, unsupported operation or slow benchmark by
inventing a new architecture. That evidence goes back to REVIEW.

## Immutable evidence

Each experiment result should record enough provenance to reproduce it:

- experiment ID and parent ID
- Git commit
- model-spec hash
- training-config hash
- dataset/manifest version
- feature-profile version
- random seed
- training framework version
- CUDA version and GPU identity
- exported ONNX hash
- OpenVINO IR hashes
- repository/runtime revision
- firmware hash where available
- hardware target identity/class
- benchmark version

## Optimization objective

The reviewer should consider a Pareto frontier across accuracy and device cost.
A candidate that improves WER but cannot operate in real time is not
automatically better.

Hard constraints can include:

- exports successfully;
- converts with OpenVINO 2020.3;
- loads/executes on MYRIAD;
- remains within device/runtime resource limits.

Soft objectives include:

- WER/CER
- RTF
- latency
- model size
- stability/numerical agreement
