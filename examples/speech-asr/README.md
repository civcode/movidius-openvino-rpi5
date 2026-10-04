# Speech ASR on Movidius

This directory is the development area for automatic speech recognition on the
Movidius/OpenVINO 2020.3.2 platform provided by this repository.

The first development model is **rm_cnn4a**. It is a bring-up and measurement
fixture: its purpose is to exercise audio ingestion, model-specific feature
preparation, MYRIAD inference, timing, evaluation, experiment recording and the
agent handoff workflow. It is not the long-term target architecture.

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

## Data and measurement baseline

AMI is the canonical development corpus. Corpus-specific input is normalized to
the internal speech sample format before it reaches inference code.

Core invariants:

- 16 kHz audio
- mono
- float32 internal sample representation
- integer sample indices as the canonical timing coordinate
- versioned dataset splits/manifests
- versioned text normalization and scoring rules
- pinned OpenVINO/MYRIAD runtime for hardware comparisons

Feature geometry, chunking and future model architecture are explicit,
versioned experiment parameters rather than hidden constants.

See [docs/parameters.md](docs/parameters.md).

## Model path

Initial bring-up:

```text
AMI/WAV
  -> normalized audio
  -> rm_cnn4a-specific frontend
  -> OpenVINO 2020.3 FP16
  -> MYRIAD
  -> rm_cnn4a-specific output handling
  -> deterministic evaluation
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
