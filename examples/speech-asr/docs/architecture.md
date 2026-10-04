# Speech ASR architecture

## Goals

The system must support two things without conflating them:

1. reliable speech inference and measurement on the Pi/Movidius platform;
2. a repeatable model-development loop in which architectures can be designed,
   trained elsewhere, exported, and evaluated on the physical target.

The benchmark and measurement system is therefore treated as infrastructure,
not as part of a particular model.

## System shape

```text
audio / dataset
      |
      v
normalization
16 kHz, mono, float32
      |
      v
model frontend / feature profile
      |
      v
chunking / context policy
      |
      v
AcousticModel adapter
      |
      +---- CPU/reference implementation
      |
      +---- MYRIAD/OpenVINO implementation
      |
      v
model output adapter
      |
      v
decoder
      |
      v
transcript + timing
      |
      v
deterministic scorer
```

The NCS/MYRIAD device is an inference accelerator. Audio capture, corpus
handling, decoding, language-model logic, scoring and orchestration remain host
responsibilities.

## Stable boundaries

### Audio boundary

All sources are normalized before feature extraction:

- sample rate: 16,000 Hz
- channels: one
- internal sample type: float32
- timing: integer sample indices

Source bit depth and container format are input concerns, not model concerns.

### Dataset boundary

Dataset adapters emit a normalized manifest. Runtime code must not understand
AMI XML or another corpus's native metadata format.

A normalized sample carries at least:

```text
id
audio path
sample rate
start sample
end sample
normalized transcript
word timing records when available
```

### Model boundary

A model package declares its frontend, input tensor contract, output semantics,
decoder kind and deployment requirements. The runtime consumes that declaration
rather than encoding model-specific constants globally.

The initial `rm_cnn4a` package is allowed to use its own compatible frontend
and output adapter. Future custom models are expected to converge on a
CNN/CTC-style contract, but the benchmark code must not require that today.

### Evaluation boundary

Evaluation consumes normalized references plus model outputs. WER, CER, timing,
RTF and latency statistics are computed by deterministic code. An LLM may
summarize results but may not decide metric values.

## Development model versus product model

`rm_cnn4a` is experiment zero. It qualifies the speech data path and the
OpenVINO/MYRIAD hardware path.

It is explicitly not a commitment to:

- its training framework;
- its acoustic target representation;
- its decoder architecture;
- its frontend as the eventual optimal frontend;
- its network topology as the custom model family.

Custom models later use the same dataset, experiment, export, hardware and
evaluation infrastructure.

## Future custom-model path

The canonical training path is intended to be PyTorch first:

```text
model_spec.yaml
      |
      v
PyTorch model
      |
      +--> CPU reference inference
      |
      +--> CUDA training
      |
      v
ONNX export
      |
      v
OpenVINO 2020.3 Model Optimizer
      |
      v
FP16 XML + BIN
      |
      v
MYRIAD evaluation
```

TensorFlow may be supported later through the same exported model contract, but
the project should not maintain two equal training implementations unless there
is a concrete need.

## Golden-tensor validation

For selected AMI samples the project should eventually preserve enough metadata
to compare stages:

```text
PCM
 -> feature tensor
 -> framework/reference outputs
 -> ONNX outputs
 -> OpenVINO outputs
 -> MYRIAD FP16 outputs
```

Comparisons use tolerances appropriate to each backend. This separates frontend
drift, export/conversion drift and device numerical drift from actual model
quality changes.
