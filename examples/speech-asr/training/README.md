# Training and export

Custom-model training is intentionally separate from the Pi/MYRIAD runtime.

The intended canonical workflow is:

```text
model spec
 -> PyTorch
 -> CUDA training
 -> checkpoint
 -> ONNX
 -> OpenVINO 2020.3 conversion
 -> FP16 IR
 -> Pi/MYRIAD evaluation
```

PyTorch is the initial canonical training implementation. TensorFlow support may
be added later through the same model/export contract when there is a concrete
need.

Training code owns optimization, augmentation and checkpointing. It must not
own benchmark scoring rules or hardware acceptance decisions.

Before a full training budget is spent, the execution workflow should perform a
cheap compatibility probe where possible: instantiate the model, run a dummy
forward pass, export ONNX, convert with the pinned OpenVINO toolchain, and test
the resulting deployment path sufficiently to catch unsupported graph choices.

Staged training budgets should reject broken or clearly unpromising candidates
before full training while preserving the declared architecture.


## Phase 8: cnn_ctc_v1

The first trainable skeleton is declared in
`models/cnn_ctc_v1/model_spec.json`. It accepts a fixed host-computed
`[1,64,512]` log-mel tensor and emits `[1,128,39]` CTC logits.

Prepare the isolated training environment:

```bash
./scripts/prepare-python-env.sh training
```

Run the cheap graph/toolchain gate **before training**. The initialized-model
probe is dataset-independent; it does not read or require an AMI manifest:

```bash
# workstation: deterministic init -> ONNX -> ONNX Runtime -> MO FP16
./scripts/probe-cnn-ctc-v1.sh --platform amd64 --no-device
```

After pulling the Phase 8 runner changes, rebuild the target runtime image so
`hello_myriad` contains the generic f32 `--tensor/--output` path, then run the
same initialized-model probe on the physical stick:

```bash
./build.sh --platform arm64
./scripts/probe-cnn-ctc-v1.sh --platform arm64
```

The probe spends no training budget. It records PyTorch-vs-ONNX and
PyTorch-vs-MYRIAD numerical differences without applying an invented device
tolerance. The physical Pi 5/arm64 + MA2450 compatibility gate for this fixed
graph passed on 2026-10-05; full trained-checkpoint evaluation remains the
Milestone B acceptance step.

Full skeleton training/export/conversion on a CUDA workstation:

```bash
./scripts/train-cnn-ctc-v1.sh --device cuda
```

The default model spec is deliberately small and defaults to one epoch. This is
a lifecycle proof, not an accuracy target. Unless `--validation-manifest` is
provided, the smoke manifest is reused for validation; that is acceptable for
toolchain bring-up only and is not generalization evidence.

Evaluate the trained FP16 IR on the Pi after copying the generated
`work/speech-asr/cnn_ctc_v1` artifacts without changing their hashes:

```bash
./scripts/evaluate-cnn-ctc-v1.sh \
    --platform arm64 \
    --manifest work/speech-asr/ami/ami-smoke-v1/manifest.jsonl
```

The evaluator computes the declared log-mel frontend on the Pi, sends one fixed
feature tensor per sample to MYRIAD, greedily decodes CTC logits, and emits the
existing `speech-asr/experiment-result` contract with real WER/CER plus
inference-only RTF and p50/p95 device latency.

Generated checkpoints, ONNX, IR and evaluation files live below `work/` and
are not committed by default.
