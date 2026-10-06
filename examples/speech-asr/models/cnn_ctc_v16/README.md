# cnn_ctc_v16

`cnn_ctc_v16` replaces the unsuccessful home-grown v12-v15 large-capacity
residual family with a faithful QuartzNet-15x5 acoustic topology.

The architectural source is Kriman et al., *QuartzNet: Deep Automatic Speech
Recognition with 1D Time-Channel Separable Convolutions* (ICASSP 2020,
arXiv:1910.10261), plus the NeMo/Open Model Zoo deployment configuration.

Frozen graph contract:

- input: `[1,64,512]`;
- C1: 256 channels, kernel 33, stride 2, time-channel separable;
- block groups B1-B5: kernels `33,39,51,63,75`, channels
  `256,256,512,512,512`;
- each block type is repeated 3 times; each block contains 5 TCS modules;
- every residual branch uses pointwise Conv1D + BatchNorm;
- C2: 512 channels, kernel 87, dilation 2, separable;
- C3: 1024 channels, kernel 1;
- output head: project to the frozen 39-token CTC vocabulary;
- output: `[1,256,39]`;
- trainable parameters: `18,934,631`;
- fixed-input convolution MACs: `4,827,463,680`;
- receptive field: `8,057` feature frames (larger than the fixed input, so
  every output position can cover the full chunk).

The original architecture is reported at about 18.9M parameters. The small
count difference here reflects the project 39-token head and trainable
BatchNorm affine parameters/counting conventions.

## VPU / CPU split

v16 intentionally ends at CTC logits on the accelerator. The initial screen
continues to use greedy CTC only so acoustic quality remains directly comparable
with prior experiments. A CPU CTC prefix-beam + language-model decoder is the
next system layer after the acoustic model proves itself; it is not embedded in
the OpenVINO graph.

## Gate order

The initialized graph must first export to ONNX, convert with OpenVINO 2020.3,
and execute numerically on physical MYRIAD. Only then does the experiment runner
start the 12-epoch architecture screen.
