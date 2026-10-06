# cnn_ctc_v14

Cosine-control retry for the physically proven large-capacity graph.

The inference architecture is identical to cnn_ctc_v12/v13:

- input: `[1,64,512]`;
- stems: 128 then 448 channels, both stride 2;
- 16 normalization-free 448-channel kernel-5 residual blocks;
- dilations: `[1,2,3,4,4,3,2,1]` repeated twice;
- output: `[1,128,39]`;
- parameters: 19,627,687;
- fixed-input MACs: 2,515,673,088.

v13 avoided the catastrophic v12 explosion but still degraded as OneCycle
raised the learning rate. Its best CER was epoch 1 while LR was still near the
starting range, and it reached full blank collapse by epoch 6.

v14 therefore restores the last proven-stable small-model optimizer policy:

- Adam;
- initial LR: `3e-4`;
- cosine decay only; no LR rise;
- final LR: `3e-5`;
- gradient clipping: 5.0;
- screen checkpoint selection: validation CER.

If this control still collapses, the next change should target the large
normalization-free architecture itself rather than continue LR tuning.
