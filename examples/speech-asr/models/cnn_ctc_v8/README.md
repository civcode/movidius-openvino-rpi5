# cnn_ctc_v8

`cnn_ctc_v8` is the first architecture-screen-v1 redesign after the rejected
v7 width sweep.

It reallocates compute from channel width into temporal resolution:

- input: `[1,64,512]`;
- stems: 64 channels at stride 1, then 72 channels at stride 2;
- output rate: 256 CTC frames instead of 128;
- residual width: 72 channels;
- residual kernels: `[11,19,27,35,43]`;
- residual dilations: `[1,2,2,2,2]`;
- effective receptive field: 509 input feature frames;
- parameters: 772,983;
- fixed-input MACs: 202,897,408;
- output: `[1,256,39]`.

The first experiment uses the frozen architecture-screen-v1 training manifest,
12 epochs and the full ES2011 validation manifest. No decoder, frontend,
objective or augmentation change is allowed in that screen.
