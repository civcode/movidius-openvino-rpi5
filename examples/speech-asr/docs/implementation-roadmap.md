# Speech ASR implementation roadmap

This roadmap defines the implementation order for the speech-ASR subproject.
The sequence is intentionally conservative: first build a trustworthy
measurement system, then prove model execution on the Movidius target, then add
custom training, and only after that automate the architecture-development loop.

The central rule is:

> Do not automate model research before the dataset, metrics and hardware
> benchmark are deterministic.

## Implementation status

| Phase | Status |
|---|---|
| 0 — contracts and reproducibility | **complete** |
| 1 — AMI dataset pipeline | **complete** — real pinned sources prepared twice with frozen golden hashes |
| 2 — deterministic evaluation | implemented |
| 3 — audio/frontend framework | implemented |
| 4 — rm_cnn4a reference path | model preparation implemented; hardware-independent execution still needs artifact run |
| 5 — MYRIAD deployment | wired; physical device validation pending |
| 6 — benchmark harness | implemented; physical device validation pending |
| 7+ | not started |

Phase 0 was re-reviewed after later implementation work. The root audio,
benchmark, model and text contracts now have executable semantic validators,
benchmark measurement counts are explicit rather than runner-defined, and all
contract documents have deterministic canonical hashes.

## Milestone overview

The work is grouped into three major milestones.

| Milestone | Phases | Outcome |
|---|---:|---|
| **A. Measurement platform** | 0-6 | A model artifact can be evaluated reproducibly on AMI and the Pi/Movidius target |
| **B. Trainable deployment platform** | 7-8 | A custom PyTorch model can be trained, exported, converted and evaluated on MYRIAD |
| **C. Agent-driven model development** | 9-11 | Frontier models can design architectures while local agents execute experiments and return deterministic evidence |

The dependency chain is:

```text
contracts
   |
   v
AMI pipeline
   |
   v
deterministic evaluation
   |
   v
audio/frontend framework
   |
   v
rm_cnn4a reference path
   |
   v
rm_cnn4a MYRIAD path
   |
   v
benchmark harness
   |
   v
streaming
   |
   v
custom PyTorch model
   |
   v
experiment lifecycle
   |
   v
agent orchestration
   |
   v
architecture optimization
```

Later phases may prepare scaffolding early, but they must not become acceptance
dependencies before the preceding phase is complete.

---

# Milestone A: measurement platform

## Phase 0 - contracts and reproducibility

### Goal

Freeze the boundaries that should remain stable while implementations and
models change.

### Deliverables

- validate and refine:
  - `contracts/audio-v1.yaml`
  - `contracts/benchmark-v1.yaml`
  - `contracts/model-v1.yaml`
- define a normalized speech-sample JSON schema;
- define the experiment-result schema used by later automation;
- define text-normalization version `text-v1`;
- define artifact/provenance fields;
- add schema validation tests.

### Required invariants

- 16 kHz normalized audio;
- mono;
- float32 internal samples;
- integer sample indices for canonical timing;
- benchmark manifests are versioned;
- scoring rules are versioned;
- model-specific frontend rules remain inside model packages;
- hardware results record runtime/model provenance.

### Exit criteria

- invalid contracts fail validation with useful errors;
- one synthetic sample can be represented without AMI-specific fields;
- experiment metadata can identify dataset, code, model and runtime versions;
- no implementation code needs to infer missing contract values.

### Not in scope

- downloading AMI;
- running `rm_cnn4a`;
- MYRIAD inference;
- training.

---

## Phase 1 - AMI dataset pipeline

### Goal

Turn selected AMI material into deterministic local audio and normalized
manifests.

### Deliverables

- `datasets/ami/prepare_ami.py` or equivalent preparation tool;
- downloader/source verification strategy;
- AMI annotation parser;
- audio normalization to the audio-v1 contract;
- word timing conversion to integer 16 kHz sample indices;
- versioned text normalization;
- JSONL manifest writer;
- deterministic corpus hashes/provenance;
- frozen smoke subset manifest;
- initial benchmark subset manifest.

### Target command shape

```bash
python3 examples/speech-asr/datasets/ami/prepare_ami.py \
    --subset smoke \
    --output work/speech-asr/ami
```

A wrapper script can be added later if that improves consistency with the rest
of the repository.

### Smoke corpus

The smoke corpus should be deliberately small:

- enough speakers/utterances to exercise parsing and timing;
- small enough to rebuild and score frequently;
- fixed once benchmark-v1 is frozen.

The full AMI corpus must not be required for ordinary development tests.

### Exit criteria

Two clean preparations of the same source and configuration produce identical:

- normalized transcript records;
- sample boundaries;
- manifest ordering;
- content hashes for generated normalized audio, subject to the chosen
  normalization implementation.

The resulting manifest contains enough information for evaluation without
reading AMI-native annotation formats again.

### Not in scope

- model inference;
- feature extraction beyond audio normalization;
- ASR quality.

---

## Phase 2 - deterministic evaluation

### Goal

Build the authoritative scorer before there is a model whose quality we care
about.

### Deliverables

- text normalization implementation for `text-v1`;
- WER;
- CER;
- timing-error utilities;
- latency summary utilities;
- real-time-factor calculation;
- machine-readable result writer;
- unit tests with hand-computed expected metrics.

### Design rule

LLMs may summarize metric results but may not calculate or modify authoritative
values.

### Test fixtures

Include cases for:

- exact transcript match;
- insertion;
- deletion;
- substitution;
- punctuation/case normalization;
- empty reference/hypothesis handling;
- known timing offsets;
- known RTF and percentile inputs.

### Exit criteria

Synthetic fixtures produce known expected values and remain independent of:

- AMI parser implementation;
- OpenVINO;
- MYRIAD;
- `rm_cnn4a`.

---

## Phase 3 - audio and frontend framework

### Goal

Create the boundary between canonical audio and model-specific feature tensors.

### Deliverables

- canonical WAV/audio reader;
- resampling path if source audio is not already 16 kHz;
- mono conversion;
- float32 normalization;
- sample-index-preserving slicing;
- frontend/profile interface;
- tensor serialization/debug dump support;
- golden frontend test fixtures.

### Architecture

```text
source audio
    |
    v
canonical audio
16 kHz / mono / float32 / sample timing
    |
    v
FrontendProfile
    |
    v
model-specific feature tensor
```

### Important constraint

Do not declare one feature frontend globally.

The future custom model may use log-mel features, while `rm_cnn4a` must use the
frontend required by its actual published/model artifact contract.

### Exit criteria

- canonical audio is reproducible;
- feature profiles can be selected without changing dataset code;
- golden inputs produce reproducible tensor shapes and values;
- no model-specific constants are hard-coded in the AMI adapter.

---

## Phase 4 - rm_cnn4a reference path

### Goal

Prove the first real speech model path independently of the Movidius device.

### Deliverables

- pin the exact `rm_cnn4a` source/version;
- record source checksum/license/provenance;
- document actual input/output tensor names, shapes and semantics;
- implement the required `rm_cnn4a` frontend adapter;
- implement output handling/decoder required for meaningful evaluation;
- create a CPU/reference execution path where practical;
- create golden model I/O fixtures.

### Research requirement

Do not implement the adapter from assumptions about the model name. The exact
artifact used by this repository must be inspected and its contract documented.

### Exit criteria

A fixed smoke sample can run through:

```text
canonical audio
 -> rm_cnn4a frontend
 -> reference inference
 -> output adapter/decoder
 -> normalized hypothesis
 -> deterministic scorer
```

with stable results.

### Not in scope

- redesigning `rm_cnn4a`;
- training it;
- treating its frontend/decoder as the future custom-model standard.

---

## Phase 5 - MYRIAD deployment for rm_cnn4a

### Goal

Prove the model on the actual OpenVINO 2020.3/MYRIAD stack.

### Deliverables

- reproducible FP16 IR preparation/acquisition;
- `rm_cnn4a` MYRIAD runner;
- model load/compile diagnostics;
- repeated inference test;
- reference-versus-MYRIAD tensor comparison;
- device timing collection;
- failure/timeout reporting.

### Measurements

At minimum record:

- model load success;
- model load/compile time;
- steady-state inference latency;
- p50/p95 inference latency;
- RTF for benchmarked audio;
- inference failures;
- numerical divergence against the chosen reference path where comparable.

### Exit criteria

- the exact pinned model loads repeatedly;
- the smoke subset completes without manual intervention;
- numerical/device differences are quantified rather than described
  subjectively;
- failures produce machine-readable diagnostics.

---

## Phase 6 - benchmark harness

### Goal

Turn the Pi/Movidius system into a reproducible evaluation worker.

### Deliverables

- one command that evaluates a declared model/experiment against a declared
  benchmark manifest;
- warmup policy;
- iteration/measurement policy;
- provenance capture;
- compact `result.json`;
- stable non-zero exit statuses for invalid/failed experiments;
- human-readable summary generated from the same authoritative result data.

### Target command shape

```bash
./run.sh --platform arm64 speech benchmark \
    --model rm_cnn4a \
    --manifest work/speech-asr/ami/manifests/ami-smoke-v1.jsonl
```

The final CLI may differ if another shape fits the repository better, but the
important property is that the command is non-interactive and deterministic.

### Target result shape

```json
{
  "benchmark": "ami-smoke-v1",
  "model": "rm_cnn4a",
  "backend": "MYRIAD",
  "wer": 0.0,
  "cer": 0.0,
  "rtf": 0.0,
  "inference_latency_p50_ms": 0.0,
  "inference_latency_p95_ms": 0.0,
  "failures": 0,
  "provenance": {}
}
```

The numbers above are placeholders only; the schema is the important part.

### Exit criteria for Milestone A

Milestone A is complete when a fresh Pi setup can:

1. prepare the frozen AMI smoke corpus;
2. obtain/prepare the pinned development model;
3. run it through MYRIAD;
4. score the output;
5. emit a reproducible machine-readable result;

without a human interpreting intermediate logs.

At this point the system is ready to evaluate models, even though it cannot yet
train its own.

---

# Milestone B: trainable deployment platform

## Phase 7 - streaming layer

### Goal

Add real-time behavior without changing the measurement contract or model
semantics.

### Deliverables

- ring-buffer/chunk abstraction;
- chunk duration and overlap handling;
- left/right context policy;
- VAD interface;
- partial/final transcript events;
- token/word stabilization measurements;
- offline-versus-streaming comparison tests.

### Tunable runtime parameters

Examples:

- chunk duration;
- overlap;
- lookahead;
- VAD threshold;
- end-of-speech timeout;
- stabilization policy.

These must be explicit experiment parameters.

### Exit criteria

- the same recorded audio can be replayed through the streaming path
  deterministically;
- streaming results can be compared to offline results;
- latency metrics are measured from known sample/time origins;
- chunking parameters are included in result provenance.

Live microphone capture can be added after recorded-audio replay is reliable.

---

## Phase 8 - custom PyTorch training skeleton

### Goal

Prove the full custom-model lifecycle before doing serious architecture
optimization.

### Initial model

Implement a deliberately small `cnn_ctc_v1`-style model whose purpose is to
exercise the toolchain, not win accuracy benchmarks.

### Deliverables

- PyTorch dataset/data-loader using normalized manifests;
- model-definition interface driven by `model_spec.yaml`;
- CUDA training loop;
- checkpointing;
- deterministic validation;
- ONNX export;
- OpenVINO 2020.3 conversion;
- MYRIAD compatibility gate;
- reference/ONNX/OpenVINO comparison tooling;
- minimal CTC decoder path for the custom model family.

### Compatibility gate

Before spending a full training budget:

```text
instantiate
 -> dummy forward
 -> ONNX export
 -> OpenVINO conversion
 -> MYRIAD load/compile probe where practical
 -> short training probe
 -> full training
```

This is intended to catch unsupported graph choices before expensive training.

### Exit criteria for Milestone B

A tiny custom architecture can:

1. train through the normal PyTorch/CUDA workflow;
2. save a reproducible checkpoint;
3. export to ONNX;
4. convert through the pinned OpenVINO 2020.3 toolchain;
5. execute on MYRIAD;
6. return a standard benchmark `result.json`.

Accuracy does not need to be competitive yet.

---

# Milestone C: agent-driven model development

## Phase 9 - experiment lifecycle

### Goal

Make every architecture attempt a self-contained, reproducible experiment.

### Deliverables

- experiment ID scheme;
- parent/lineage tracking;
- proposal schema;
- model-spec schema;
- training-config schema;
- acceptance-policy schema;
- compatibility/training/hardware/accuracy result files;
- artifact hash/storage references;
- experiment-history index.

### State machine

```text
DESIGN
 -> APPROVED
 -> EXECUTE
 -> EVALUATE
 -> AWAIT_REVIEW
 -> REVIEW
 -> DESIGN
```

### Exit criteria

A human can create an approved model spec, hand it to the execution system, and
receive a complete result bundle without additional design decisions during the
run.

---

## Phase 10 - local execution agent and frontier handoff

### Goal

Automate repetitive experiment work while preserving the architecture authority
boundary.

### Frontier-model responsibilities

- architecture proposals;
- hypothesis formation;
- trade-off analysis;
- interpretation of completed experiment evidence;
- next-generation design.

### Local-model responsibilities

- training;
- retries allowed by policy;
- checkpoint handling;
- ONNX export;
- OpenVINO conversion;
- deployment;
- Pi/MYRIAD testing;
- benchmark execution;
- log reduction;
- result packaging;
- routine diagnostics.

### Forbidden local-agent behavior

The local execution agent must not respond to a poor result by silently:

- adding/removing layers;
- changing channel widths;
- changing feature geometry;
- changing benchmark data;
- changing scoring rules;
- changing acceptance targets.

It reports the evidence and transitions to `AWAIT_REVIEW`.

### Pi virtual-model integration

Pi virtual-model routing should eventually use explicit workflow state rather
than infer authority from natural-language prompts.

Conceptually:

```text
DESIGN / REVIEW     -> frontier model
EXECUTE / EVALUATE -> local model
```

### Exit criteria

A declared experiment can be executed end-to-end by the local agent, while a
new architecture cannot be created without crossing the explicit frontier
design/review boundary.

---

## Phase 11 - architecture optimization loop

### Goal

Begin actual agent-driven network research against the fixed evaluation
platform.

### Search dimensions

Initially permit controlled changes to:

- network depth;
- channel widths;
- temporal kernels;
- dilation;
- temporal stride;
- residual topology;
- normalization/activation;
- receptive field;
- custom-model frontend parameters;
- training hyperparameters.

Streaming/decoder tuning should be staged separately when possible so model
gains remain attributable.

### Objectives

Use multi-objective selection rather than WER alone.

Track a Pareto frontier across:

- WER;
- CER;
- RTF;
- p50/p95 latency;
- model size;
- MYRIAD compatibility/stability;
- streaming delay when available.

Hard constraints can include successful OpenVINO conversion and MYRIAD
execution.

### Exit criteria

At least two architecture generations have been proposed, trained, evaluated
and reviewed through the formal workflow, with lineage and results sufficient
to explain why the next architecture was chosen.

---

# Implementation checkpoints

The following checkpoints are intentionally small enough to commit and preserve
independently:

1. schemas and schema tests;
2. AMI smoke preparation;
3. deterministic scorer;
4. canonical audio/frontend framework;
5. pinned `rm_cnn4a` package and documented tensor contract;
6. reference inference;
7. MYRIAD inference;
8. benchmark/result harness;
9. recorded-audio streaming;
10. PyTorch/CUDA training skeleton;
11. ONNX/OpenVINO compatibility gate;
12. custom model on MYRIAD;
13. experiment-state machinery;
14. local execution agent;
15. frontier/local routing;
16. first architecture iteration.

Large or risky work should start only after the preceding checkpoint has been
committed and pushed.

# What should not be built yet

Until Milestone A is complete, avoid spending significant effort on:

- sophisticated neural architecture search;
- live microphone UI;
- large language models in the decoder;
- beam-search optimization;
- distributed training;
- multi-GPU orchestration;
- automatic architecture mutation;
- broad hyperparameter sweeps;
- dashboards;
- cloud experiment infrastructure.

Those may become useful later, but none solves the first problem: obtaining a
trustworthy measurement from the Pi/Movidius target.

# Immediate next work

Start Phase 0 and Phase 1 together only where they are tightly coupled:

1. define the normalized speech-sample and text-normalization schemas;
2. implement their validation tests;
3. identify and pin the exact AMI material needed for the smoke subset;
4. implement the AMI annotation-to-manifest conversion;
5. produce the first deterministic smoke manifest.

Do **not** start `rm_cnn4a` integration until the smoke corpus and deterministic
scoring path are stable enough to serve as its acceptance harness.
