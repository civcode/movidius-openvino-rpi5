# Speech runtime boundary

The speech runtime is the Pi/host-side execution layer between normalized model
inputs and model outputs.

It may:

- load a declared model package;
- prepare fixed-shape tensors from an already-defined feature profile;
- invoke CPU/reference or MYRIAD inference;
- measure load/inference timing;
- expose deterministic diagnostics;
- return model outputs to the decoder/evaluator.

It may not:

- redesign a model;
- change dataset splits;
- change scoring rules;
- silently change a model frontend;
- tune parameters while reporting the result as the original experiment.

Model-specific behavior is selected through the model contract/adapter.

The implementation should keep the existing repository's generic MYRIAD
runtime reusable and add speech-specific behavior only under this subproject
unless a component is generalized first.
