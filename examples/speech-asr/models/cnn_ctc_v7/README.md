# cnn_ctc_v7

`cnn_ctc_v7` is the model-quality-v4 capacity ablation.

It keeps the deployed `cnn_ctc_v3` operator topology, frontend, receptive
field, fixed tensor contracts and training policy, and changes only encoder
width:

- stem channels: `[64, 112]` instead of `[64, 96]`;
- residual channels: `112` instead of `96`;
- residual kernels: unchanged at `[11, 19, 27, 35, 43]`;
- receptive field: unchanged at 533 feature frames;
- parameters: 1,818,183;
- fixed-input MACs: 235,177,984;
- output: unchanged at `[1,128,39]`.

The experiment uses the frozen model-quality-v4 training manifest and ES2011
validation manifest. It must not use the sealed held-out benchmark for model
selection.
