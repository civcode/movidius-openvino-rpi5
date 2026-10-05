# cnn_ctc_v3

Phase 11 generation 2 responds directly to the generation-1 evidence.

`cnn_ctc_v2` converted cleanly, matched ONNX/MYRIAD numerically, and ran at
~16.65 ms p95 on MA2450, but one epoch/batch-size 2 provided only a tiny
optimization budget and produced worse CER than the v1 baseline.

Generation 2 therefore keeps the successful hardware envelope while changing
training dynamics:

- same `[1,64,512] -> [1,128,39]` contract;
- same 4x temporal reduction;
- same 96-channel, five-block, full-window receptive field;
- remove BatchNorm entirely;
- keep only ordinary Conv1d, ReLU, Add and final Transpose at inference;
- initialize each residual block's final 1x1 projection with 1%-scaled Kaiming weights so the residual
  stack begins as an identity perturbation;
- train 32 epochs with batch size 1 and Adam at 3e-4;
- cosine-decay learning rate to 3e-5;
- clip gradient norm at 5.0;
- export the checkpoint with the best validation loss.

The fixed estimate is 1,346,343 parameters and 174,804,992 MACs. Removing
BatchNorm reduces state/parameters slightly without changing convolution MACs.

The smoke manifest is still only lifecycle/relative evidence, not a
generalization benchmark. SpecAugment is deliberately deferred until a larger
training/validation split is used.
