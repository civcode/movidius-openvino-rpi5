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
graph passed on 2026-10-05.

Full skeleton training/export/conversion on a CUDA workstation:

```bash
./scripts/train-cnn-ctc-v1.sh --device cuda
```

The pinned PyTorch 2.2.x CUDA backend does not provide a deterministic
`ctc_loss_backward_gpu`. The training loop therefore keeps the model forward
pass and optimizer on CUDA but computes CTC loss on CPU from the small logits
tensor. Autograd carries that gradient back to the CUDA model, preserving the
strict deterministic-algorithm policy rather than weakening it to warnings.

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
inference-only RTF and p50/p95 device latency. It applies the same fixed-shape
eligibility policy as training: over-limit audio and transcripts whose CTC
target cannot fit the available output frames are skipped consistently. The
result records the full manifest hash, evaluated-sample count, and skipped IDs,
so the scored population remains explicit.

Generated checkpoints, ONNX, IR and evaluation files live below `work/` and
are not committed by default.

## Milestone B acceptance

The complete trained lifecycle was accepted on 2026-10-05. One-epoch CUDA
training, ONNX export/reference comparison, OpenVINO 2020.3 FP16 conversion,
physical MA2450 execution, and the contract-valid AMI smoke evaluation all
completed. The smoke evaluation scored 2 eligible records from the 4-record
manifest, with WER 1.000000, CER 0.913043, inference-only RTF 0.001209, and
MYRIAD latency p50/p95 4.380/4.407 ms.

The next project phase is the experiment lifecycle; further architecture work
should be expressed as explicit experiment records rather than extending this
bring-up skeleton ad hoc.


## Phase 11 generation 1: cnn_ctc_v2

`cnn_ctc_v2` is the first accuracy-oriented architecture generation. It keeps
the accepted frontend, CTC vocabulary, output rate and benchmark fixed while
replacing the bring-up encoder with a 1.35M-parameter residual temporal stack.

Direct local graph probe:

```bash
./scripts/probe-cnn-ctc-v2.sh
```

Direct CUDA training/export path:

```bash
./scripts/train-cnn-ctc-v2.sh --device cuda
```

The normal Phase 11 workflow is the experiment controller rather than these
manual commands:

```bash
EXP="$(
  ./scripts/init-cnn-ctc-v2-experiment.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"
./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
```

For v2 the controller performs both OpenVINO conversion and a one-inference
physical MA2450 graph probe from an initialized checkpoint before entering the
full training state. The probe output is numerically compared against the
PyTorch golden tensor and retained in the attempt compatibility evidence.

Generation 1 deliberately avoids grouped/depthwise convolution and attention
despite their use in efficient modern ASR models. The first goal is to learn how
much accuracy can be recovered with a large-receptive-field residual encoder
while staying inside an operator family already close to the proven v1 graph.


## Phase 11 generation 2: cnn_ctc_v3

Generation 2 preserves the generation-1 temporal/receptive-field design but
changes optimization dynamics based on the rejected v2 evidence.

The inference model is normalization-free and contains ordinary Conv1d, ReLU,
Add and the final transpose only. Residual projections are zero-initialized so
the residual stack begins close to identity.

The reviewed default training budget is 32 epochs, batch size 1, Adam 3e-4,
cosine decay to 3e-5, gradient clipping at norm 5.0, and best-validation-loss
checkpoint selection. Training evidence records optimizer-step count and the
selected epoch.

Use the experiment lifecycle rather than editing those defaults during a run:

```bash
./scripts/init-cnn-ctc-v3-experiment.sh
./scripts/run-speech-experiment.sh --experiment work/.../exp-... --worker edge
```
