# Speech ASR on Movidius

This directory is the development area for automatic speech recognition on the
Movidius/OpenVINO 2020.3.2 platform provided by this repository.

The first development model is **rm_cnn4a**. It is a conversion/device
qualification fixture: its published Intel package provides Kaldi feature and
reference-score ARKs, so it exercises FP16 conversion, tensor transport,
MYRIAD inference, numerical comparison, timing and result/provenance handling.
It is **not** the raw-audio AMI model and is not the long-term architecture.

The long-term workflow is designed for custom acoustic models trained with a
normal PyTorch/CUDA toolchain, exported through ONNX, converted to OpenVINO
2020.3 FP16 IR, and evaluated on the Raspberry Pi/Movidius target.

## Project boundary

The repository root owns the generic platform:

- OpenVINO 2020.3.2 build and runtime
- MYRIAD plugin and Movidius firmware
- USB/device access
- generic model loading and deployment support
- platform build/container infrastructure

This subproject owns the speech-specific work:

- AMI corpus preparation and manifests
- audio, benchmark and model contracts
- speech feature extraction and chunking
- model-specific adapters
- decoding
- WER/CER/timing evaluation
- experiment records
- training/export workflow
- agent-driven model development

Speech-specific assumptions must not leak into the generic Movidius runtime.

## Development lifecycle

The lifecycle has a hard boundary between architecture design and experiment
execution:

```text
DESIGN (frontier model)
  |
  v
APPROVED model/experiment spec
  |
  v
EXECUTE (local model + deterministic tools)
  |
  v
EVALUATE (deterministic scoring + local orchestration)
  |
  v
AWAIT_REVIEW
  |
  v
REVIEW / next DESIGN (frontier model)
```

The execution side may train, export, convert, test, benchmark, retry declared
operations and collect diagnostics. It may not silently redesign the network or
change the benchmark contract.

See [docs/agent-workflow.md](docs/agent-workflow.md).

The implementation sequence and phase exit criteria are defined in
[docs/implementation-roadmap.md](docs/implementation-roadmap.md).

## Data and measurement baseline

AMI is the canonical development corpus for end-to-end transcript evaluation.
Corpus-specific input is normalized to the internal speech sample format before
it reaches inference code.

Core invariants:

- 16 kHz audio
- mono
- float32 f32le normalized sample representation
- integer sample indices as the canonical timing coordinate
- versioned dataset splits/manifests
- versioned text normalization and scoring rules
- pinned OpenVINO/MYRIAD runtime for hardware comparisons

Feature geometry, chunking and future model architecture are explicit,
versioned experiment parameters rather than hidden constants.

See [docs/parameters.md](docs/parameters.md).

## Model paths

Initial device qualification:

```text
Intel rm_cnn4a Kaldi feature ARK
  -> OpenVINO 2020.3 FP16 IR
  -> MYRIAD
  -> acoustic score ARK
  -> deterministic comparison to Intel reference score ARK
```

AMI/custom-model path:

```text
AMI/WAV
  -> canonical 16 kHz mono audio
  -> model-declared frontend
  -> custom acoustic model
  -> decoder
  -> normalized hypothesis
  -> deterministic WER/CER + timing evaluation
```

Future custom-model loop:

```text
architecture spec
  -> PyTorch/CUDA training
  -> checkpoint
  -> ONNX
  -> OpenVINO 2020.3 FP16 IR
  -> Pi + Movidius evaluation
  -> results
  -> architecture review
  -> next architecture
```

The benchmark, dataset and hardware measurement contracts remain stable while
model architectures evolve.

## Python environment

Host-side speech tooling follows the repository-wide uv policy. Dependency-free speech utilities run through `./scripts/python.sh`; Python packages must not be installed into system Python.

## Tests

The speech-specific fixture tests are deliberately independent of network and
MYRIAD hardware:

```bash
./scripts/test-speech-asr.sh
```

Physical-device qualification, the repeated Phase 6 worker, and Phase 7
recorded replay remain explicit operations:

```bash
./scripts/prepare-rm-cnn4a.sh
./scripts/python.sh examples/speech-asr/evaluation/benchmark_rm_cnn4a.py \
    --backend myriad --platform arm64

./scripts/benchmark-speech.sh \
    --model rm_cnn4a --backend myriad --platform arm64

./scripts/python.sh examples/speech-asr/tools/make_streaming_updates.py \
    work/speech-asr/ami/ami-smoke-v1/manifest.jsonl \
    --output work/speech-asr/streaming/updates.json
# The helper prints the corresponding ./scripts/replay-speech.sh command.
```

## Directory layout

```text
speech-asr/
├── README.md
├── agent/                 execution/design authority notes
├── contracts/             versioned machine-readable contracts
├── datasets/
│   └── ami/               AMI preparation and manifest rules
├── docs/
│   ├── architecture.md
│   ├── parameters.md
│   ├── agent-workflow.md
│   ├── implementation-roadmap.md
│   └── adr/
├── evaluation/            deterministic scoring/measurement
├── experiments/           experiment artifact conventions
├── models/
│   └── rm_cnn4a/          initial development fixture
├── runtime/               Pi/MYRIAD speech runtime boundary
└── training/              future PyTorch/CUDA training/export path
```

No corpus audio, checkpoints or generated model binaries are intended to be
committed here unless a later decision explicitly says otherwise.


## Trainable custom-model skeleton

Milestone B is complete with `cnn_ctc_v1`, the first custom PyTorch/CTC
skeleton. The model keeps log-mel extraction on the host so the deployment
graph remains small and fixed-shape for OpenVINO 2020.3/MYRIAD. The accepted
2026-10-05 lifecycle includes deterministic CUDA training, ONNX export,
OpenVINO 2020.3 FP16 conversion, physical Pi 5/arm64 + MA2450 execution, and a
contract-valid AMI smoke result.

Always run compatibility before training:

```bash
./scripts/probe-cnn-ctc-v1.sh --platform amd64 --no-device
./scripts/train-cnn-ctc-v1.sh --device cuda
```

On the Pi, run the physical compatibility probe and trained-model evaluator.

Phase 9 is complete and provides the explicit immutable experiment lifecycle
used for further model development. Reviewed proposal/model/training/acceptance
documents are managed with:

```bash
./scripts/python.sh examples/speech-asr/tools/manage_experiment.py --help
```

Phase 10 is complete and implements the accepted oberon -> edge SSH/rsync execution path
using those experiment and deployment-manifest contracts. For the frozen
`cnn_ctc_v1` baseline:

```bash
./scripts/init-cnn-ctc-v1-experiment.sh
./scripts/run-speech-experiment.sh \
  --experiment work/speech-asr/experiments/exp-... \
  --worker edge
```

The executor rejects unknown architecture declarations rather than silently
changing or ignoring them. Runtime-image rebuilds and benchmark/acceptance
changes remain outside ordinary experiment execution.


The Phase 10 physical acceptance run on 2026-10-05 used
`exp-f915ec624a63caf6/attempt-0001` and completed with
`acceptance: accepted` at `AWAIT_REVIEW`. The project is ready to move into
Phase 11 architecture optimization while keeping this execution path fixed.


## Phase 11 generation 1

The first architecture optimization candidate is `cnn_ctc_v2`: a residual
large-kernel temporal CTC encoder inspired by the low-rate computation of
FastConformer/Zipformer and the residual temporal-convolution approach of
QuartzNet/Citrinet, constrained to the legacy OpenVINO/MYRIAD deployment
envelope.

It preserves `[1,64,512] -> [1,128,39]` and the existing frontend/decoder, but
moves to two stride-2 stems plus five 96-channel residual blocks with kernels
`11,19,27,35,43`. The fixed estimate is 1,347,463 parameters and
174,804,992 MACs.

Run it through the formal lifecycle with:

```bash
EXP="$(
  ./scripts/init-cnn-ctc-v2-experiment.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"
./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
```

The controller physically probes the initialized v2 graph on MA2450 before full
CUDA training. Phase 11 remains open until generation 1 is reviewed and a
second architecture generation is selected and evaluated from that evidence.


## Phase 11 generation 2

Generation-1 review showed that `cnn_ctc_v2` was hardware-safe but
optimization-starved: only CER failed, while OpenVINO/MYRIAD compatibility and
realtime performance retained large margin.

`cnn_ctc_v3` keeps v2's 4x reduction, 96-channel residual stack, kernel
schedule and 174,804,992 MAC envelope, but removes BatchNorm and small-initializes
each residual block's final 1x1 projection with 1%-scaled Kaiming weights. Training is increased to 32 epochs
with batch size 1, Adam at 3e-4, cosine decay to 3e-5, gradient clipping and
best-validation-loss checkpoint selection.

Run generation 2 with:

```bash
EXP="$(
  ./scripts/init-cnn-ctc-v3-experiment.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"
./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

The acceptance policy retains the v1 CER ceiling while tightening the already
comfortable v2 hardware limits to RTF <= 0.01 and p95 <= 25 ms.


## Phase 11 completion and next data boundary

Phase 11 completed after two physical architecture generations. `cnn_ctc_v2`
and `cnn_ctc_v3` both remained comfortably realtime on MA2450, but the
two-eligible-record smoke population could not rank model changes reliably:
v3 substantially reduced CTC loss while greedy CER regressed to 1.0.

The final review is in
`docs/phase11-final-review.md`.

Before another architecture generation, derive and review the larger
speaker-disjoint model-quality manifests:

```bash
./scripts/python.sh examples/speech-asr/datasets/ami/prepare_ami.py \
  --subset benchmark
./scripts/qualify-speech-model-data.sh
```

Future custom-model evaluation results include per-sample CTC collapse and edit
diagnostics so blank collapse or pathological emission can be distinguished
from ordinary substitution/deletion errors.


## Model-quality v1 baseline

The post-Phase-11 data qualification is reviewed and accepted for the next
controlled baseline. It uses the frozen ES2002a benchmark preparation but
partitions by speaker:

- train A/B/C: 125 eligible records, 219.998 seconds;
- validation D: 95 eligible records, 122.52 seconds;
- zero record and speaker overlap.

This is intentionally an interim speaker-holdout benchmark, not cross-meeting
generalization evidence.

Provision the reviewed dataset once on edge, then run the frozen v3 baseline:

```bash
./scripts/provision-speech-model-data-edge.sh

EXP="$(
  ./scripts/init-cnn-ctc-v3-quality-baseline.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"
./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

The experiment keeps `cnn_ctc_v3` unchanged and removes smoke-derived WER/CER
acceptance ceilings. It establishes the larger-data accuracy reference while
retaining the proven OpenVINO/MYRIAD realtime gates.


## cnn_ctc_v4 valid-frame frontend

After the deterministic SpecAugment diagnostic regressed decoder emission,
model development moved to a structural frontend correction rather than
stronger regularization.

`cnn_ctc_v4` keeps the v3 Conv1D/ReLU inference graph and MA2450 workload but
changes host feature normalization from padding-inclusive `logmel-v1` to
valid-frame `logmel-v2`. Padding no longer contributes to CMVN and padded
normalized frames are zero.

Run the reviewed child with:

```bash
./scripts/init-cnn-ctc-v4-valid-cmvn.sh
```


## Held-out promotion boundary

The current quality reference is the scaled-data `cnn_ctc_v3` checkpoint
`exp-87538823d2bf1562/attempt-0001`. Before another model-development
decision, it must be measured once on the sealed AMI Full-corpus-ASR unseen
scenario-component evaluation boundary.

The workflow is:

```bash
./scripts/prepare-speech-heldout-eval.sh
./scripts/provision-speech-heldout-eval-edge.sh
./scripts/run-speech-heldout-eval.sh
./scripts/review-speech-heldout-eval.sh
```

This path performs no training and no checkpoint selection. It verifies and
reuses the exact recorded checkpoint/ONNX/OpenVINO artifacts from the accepted
source experiment.
