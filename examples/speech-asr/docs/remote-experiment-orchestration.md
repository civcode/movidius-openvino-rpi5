# Remote experiment orchestration specification

Status: **accepted for Phase 9/10 implementation**

This document defines the multi-host execution architecture for speech-ASR
experiments. It separates experiment control and model production on the CUDA
workstation from physical MYRIAD execution on the edge worker.

The initial deployment is:

- **oberon** — experiment controller, CUDA training host, export/conversion host,
  artifact authority and result collector;
- **edge** — Raspberry Pi / arm64 hardware worker with the physical MA2450
  MYRIAD device;
- **Git** — source-code identity and reviewed configuration authority.

The architecture is intentionally compatible with the Phase 9 experiment
lifecycle and the later Phase 10 local execution agent. SSH and rsync are
transport mechanisms, not experiment-design APIs.

## Goals

The orchestration layer must make a complete declared experiment executable
from oberon without a human switching between terminals.

A successful run must be able to:

1. validate the declared experiment and source revision;
2. train/export/convert on oberon;
3. freeze the exact deployment artifacts and their hashes;
4. verify that edge is running the required worker revision and runtime;
5. transfer only the declared deployment bundle;
6. execute the declared hardware evaluation on edge under an exclusive device
   lock;
7. collect result files and relevant logs back to oberon;
8. validate the returned result and artifact identities;
9. leave one self-contained experiment record ready for REVIEW.

## Non-goals

This layer does not:

- choose or mutate model architecture;
- modify benchmark or scoring rules;
- reinterpret acceptance thresholds;
- regenerate ONNX or OpenVINO IR on edge;
- use edge as a training host;
- allow experiment specifications to contain arbitrary remote shell commands;
- automatically move an experiment from REVIEW into a new DESIGN.

## Host responsibilities

### Oberon: controller and artifact authority

Oberon owns:

- experiment state;
- model specification and training configuration selected for the run;
- CUDA training;
- framework checkpoint;
- ONNX export;
- OpenVINO Model Optimizer conversion;
- local compatibility/reference comparisons;
- immutable hashes of all deployment artifacts;
- remote-worker orchestration;
- result collection;
- final experiment bundle and transition to `AWAIT_REVIEW`.

An experiment is not allowed to replace its own model specification after
execution begins.

### Edge: hardware worker

Edge owns:

- access to the physical MA2450 device;
- pinned OpenVINO/MYRIAD runtime;
- worker-local dataset material required by the declared benchmark;
- execution of deterministic hardware evaluation;
- raw device logs;
- hardware result production.

Edge must not train, export or convert a model for a remotely orchestrated
experiment. It receives the exact deployment artifacts prepared on oberon.

### Git: source identity

Every run records the Git revision used by the controller and worker. Source
revision changes are deployment operations, not implicit side effects of an
experiment.

A worker must never perform an unbounded `git pull` merely because an
experiment starts.

## Authority and state ownership

| State | Authority |
|---|---|
| experiment ID / lineage | oberon experiment record |
| proposal / hypothesis | reviewed experiment record |
| model spec | reviewed experiment record |
| training config | reviewed experiment record |
| acceptance policy | reviewed experiment record |
| checkpoint | oberon |
| ONNX | oberon |
| OpenVINO XML/BIN | oberon |
| dataset/manifest identity | declared benchmark contract |
| physical MYRIAD execution | edge |
| hardware logs | edge, copied back to oberon |
| authoritative compact result | oberon after validating returned evidence |
| next architecture decision | REVIEW / frontier authority |

## Experiment lifecycle

The remote orchestration path implements only the execution side of the existing
state machine:

```text
APPROVED
   |
   v
EXECUTE
   |  oberon: train -> export -> convert -> freeze
   v
EVALUATE
   |  oberon -> edge: deploy immutable bundle
   |  edge: verify -> MYRIAD run -> result/logs
   |  edge -> oberon: collect evidence
   v
AWAIT_REVIEW
```

A failed remote operation transitions to a declared failed/blocked result. It
does not trigger architecture changes.

## Experiment directory contract

The Phase 9 experiment directory is the controller-side authority:

```text
work/speech-asr/experiments/<experiment-id>/
├── request/
│   ├── experiment.json
│   ├── model-spec.json
│   ├── train-config.json
│   └── acceptance.json
├── artifacts/
│   ├── checkpoint.pt
│   ├── model.onnx
│   ├── model.xml
│   ├── model.bin
│   └── manifest.json
├── results/
│   ├── training.json
│   ├── compatibility.json
│   ├── hardware.json
│   └── result.json
└── logs/
    ├── controller.log
    └── edge/
```

Generated binaries remain outside Git by default. Reviewed declarative
experiment records may later have a Git representation, but one experiment ID
must always identify one immutable execution lineage.

## Deployment bundle

Oberon produces a deployment manifest before contacting edge. At minimum it
contains:

- schema/version;
- experiment ID;
- controller Git commit;
- required worker Git commit;
- model ID;
- model-spec SHA-256;
- benchmark/manifest ID and SHA-256;
- checkpoint SHA-256;
- ONNX SHA-256;
- OpenVINO XML SHA-256;
- OpenVINO BIN SHA-256;
- required OpenVINO version;
- target platform;
- evaluator identifier/version;
- expected output/result path.

The bundle transferred to edge contains only files required for the hardware
evaluation. The deployment manifest is transferred with the model artifacts.

Edge verifies the bundle before opening the MYRIAD device.

## Worker source revision

The recommended worker model is a **dedicated automation checkout/worktree** on
edge, separate from the human interactive checkout.

The controller requests an exact Git commit. The worker checkout may fetch that
commit explicitly, but it must not merge, rebase or follow a moving branch as
part of the run.

The human checkout on edge remains untouched by experiment orchestration.

## Dataset policy

Corpus material is provisioned independently on edge. Ordinary experiments do
not rsync AMI audio from oberon.

The deployment request names the required benchmark/manifest identity. Edge
verifies the local manifest hash before execution. A mismatch is a preflight
failure, not an invitation to silently rebuild or replace the dataset.

Dataset provisioning remains a separate explicit operation.

## Remote protocol

The controller uses non-interactive SSH with public-key authentication.

### 1. Connectivity preflight

Oberon verifies:

- `ssh -o BatchMode=yes` succeeds;
- the expected worker hostname/identity responds;
- the worker checkout can resolve the requested Git commit;
- the target platform is arm64;
- the required OpenVINO/MYRIAD runtime is available;
- the declared dataset manifest exists and matches its hash.

No model artifacts are sent until this preflight passes.

### 2. Exclusive device reservation

The worker acquires an exclusive filesystem lock before hardware execution.

Only one experiment may own the MYRIAD device at a time. A busy device produces
a distinct blocked/busy result or exit status rather than concurrent access.

The lock covers device preflight, model load/compile and all declared inference
steps.

### 3. Artifact deployment

Oberon copies the deployment manifest and exact declared artifacts into a
per-experiment staging directory on edge.

The worker recomputes hashes after transfer and rejects any mismatch.

Deployment is append/new-directory oriented. A new experiment must not
overwrite another experiment's evidence.

### 4. Hardware execution

The worker invokes a repository-owned evaluator using structured arguments from
the validated deployment request.

The request may select declared evaluator options but may not provide arbitrary
shell fragments.

The worker records:

- start/end status;
- worker Git revision;
- runtime/OpenVINO identity;
- target/platform identity;
- input artifact hashes;
- dataset/manifest hash;
- evaluator command/structured request;
- MYRIAD logs;
- deterministic metric/result files.

### 5. Evidence collection

Oberon pulls the result JSON and declared logs into the controller-side
experiment directory.

The controller validates:

- result schema;
- experiment ID;
- source/worker revisions;
- benchmark/manifest identity;
- model/artifact hashes;
- successful hardware backend/target identity.

Only after validation is the result promoted to the experiment's authoritative
`results/result.json`.

## SSH/security contract

Remote execution must use:

- public-key authentication;
- `BatchMode=yes`;
- a non-root worker account;
- normal SSH host-key verification;
- no private key material copied into experiment directories;
- no password prompts;
- no experiment-controlled arbitrary shell command;
- no forwarding of agent credentials to edge unless separately reviewed.

The orchestration layer may use an SSH host alias such as `edge`; host/address
details are deployment configuration, not experiment identity.

## Failure classes

The controller distinguishes at least:

1. **request/configuration failure** — invalid experiment or missing declared
   artifact;
2. **controller execution failure** — training/export/conversion failure;
3. **transport/preflight failure** — SSH, source revision, dataset or artifact
   transfer problem;
4. **worker busy** — MYRIAD lock unavailable;
5. **hardware execution failure** — model load/compile/inference failure;
6. **result-contract failure** — returned evidence cannot be validated.

A retry may repeat transport or explicitly transient hardware operations when
policy permits. It must not change architecture, model spec, benchmark, scoring
rules or acceptance policy.

Retries are recorded in the experiment evidence.

## Logging and result retention

The controller log is the chronological top-level execution log.

Edge logs are collected under the experiment rather than left as the only copy
on the worker. Raw logs and compact structured results have separate roles:
raw logs diagnose failures; structured JSON is authoritative for metrics.

Worker-side staging may be garbage-collected later according to a retention
policy only after successful collection and hash verification.

## Idempotency

Re-running a completed experiment ID must not silently overwrite its existing
result.

The controller must either:

- detect that the same immutable request has already completed and return the
  existing result; or
- require a new run/attempt identifier.

Changing model spec, train config, acceptance policy, benchmark identity or
source revision requires a new immutable experiment request identity.

## Concurrency

Oberon may prepare or train multiple experiments concurrently if local resource
policy permits.

Edge has one physical MYRIAD execution slot unless the hardware topology is
explicitly expanded. Remote runs therefore serialize at the device lock.

The later execution agent should queue work rather than bypass this constraint.

## Relationship to Phase 9 and Phase 10

Phase 9 defines the declarative experiment records and result identities that
this orchestration consumes.

Phase 10 may invoke the orchestrator automatically, inspect declared failure
classes and perform allowed retries. Phase 10 does not gain authority to change
architecture merely because remote execution is automated.

The intended final controller command shape is:

```bash
./scripts/run-speech-experiment.sh \
  --experiment <experiment-id-or-request> \
  --worker edge
```

The implementation should keep local training, remote deployment, remote
execution and evidence collection as separable modules behind that command.

## Frozen contract decisions

The following choices affect experiment identity or reproducibility and were
accepted on 2026-10-05. They are normative for the orchestration implementation.

### D1. Worker source deployment

**Decision:** maintain a dedicated automation checkout/worktree on edge and
pin it to the exact requested commit. Do not mutate the human interactive
checkout.

### D2. Experiment/run identity

**Decision:** one immutable experiment ID identifies the declared model,
training config, benchmark, acceptance policy and source revision; repeated
executions use explicit attempt IDs beneath that experiment.

### D3. Deployment artifact scope

**Decision:** transfer a minimal hash-manifested deployment bundle
(OpenVINO XML/BIN plus model/benchmark metadata and any evaluator-required
files), not the entire model work directory.

### D4. Dataset provisioning

**Decision:** datasets/manifests are provisioned independently on edge and
verified by hash during each run. Experiment orchestration never silently syncs
or prepares corpus data.

### D5. Busy-worker behavior

**Decision:** fail/return `blocked: worker_busy` immediately and let the
controller/agent queue or retry according to policy.

### D6. Code/runtime update authority

**Decision:** source checkout synchronization to an exact Git commit may be
performed by the dedicated worker-management layer; rebuilding/updating the
OpenVINO runtime image remains an explicit infrastructure operation and is not
performed automatically by an ordinary model experiment.

These six decisions are intentionally explicit. Details such as rsync flags,
temporary filenames, SSH ControlMaster use, compression and log formatting are
implementation choices and do not belong in the experiment contract.
