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
