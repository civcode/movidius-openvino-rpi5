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

Phase 9 now provides the explicit immutable experiment lifecycle used for
further model development. Reviewed proposal/model/training/acceptance documents
are managed with:

```bash
./scripts/python.sh examples/speech-asr/tools/manage_experiment.py --help
```

Phase 10 will automate the accepted oberon -> edge SSH/rsync execution path
using those experiment and deployment-manifest contracts; it must not change
architecture, benchmark or acceptance policy implicitly.
