# cnn_ctc_v2

Phase 11 generation 1 is a residual temporal CTC encoder designed around the
actual OpenVINO 2020.3 + MA2450 deployment envelope rather than around current
server/GPU assumptions.

## Design

The model keeps the accepted 64-bin log-mel frontend, fixed 512-frame input,
39-symbol character CTC vocabulary and 128-frame output so comparisons against
`cnn_ctc_v1` isolate the encoder.

The encoder uses:

- two stride-2 ordinary temporal convolutions, reducing 512 -> 256 -> 128 frames;
- five 96-channel residual temporal blocks;
- large ordinary kernels `11,19,27,35,43`;
- BatchNorm + ReLU;
- a 1x1 convolution inside each residual branch;
- a final 1x1 CTC projection.

The fixed-input estimate is 1,347,463 trainable/inference parameters and
174,804,992 multiply-accumulates. FP16 weights are about 2.6 MiB. The effective
receptive field is 533 input feature frames, slightly larger than the fixed
512-frame window.

## Why this shape

Modern efficient ASR systems repeatedly exploit temporal reduction and
multi-scale context. FastConformer reduces time resolution aggressively;
Zipformer moves substantial computation to lower frame rates; QuartzNet and
Citrinet show that residual temporal convolution + CTC can be highly effective.

The legacy VPU changes which parts are copied:

- keep early temporal reduction;
- keep residual temporal context;
- keep large receptive fields;
- keep CTC/non-autoregressive decoding;
- do **not** copy self-attention, LayerNorm-heavy transformer blocks, Swish,
  squeeze/excitation, grouped/depthwise convolution, dynamic sequence shapes,
  or autoregressive decoders in generation 1.

Ordinary Conv/ReLU/Add have already been exercised successfully by the project,
and the remaining BatchNorm path is deliberately behind the compatibility gate.

## Phase 11 hypothesis

Compared with `cnn_ctc_v1`, substantially more context and residual depth at
the same 4x frame reduction should improve learnability/accuracy while retaining
large real-time headroom on MA2450.

The first acceptance target is deliberately hardware-first:

- ONNX opset 11 conversion succeeds;
- OpenVINO 2020.3 FP16 conversion succeeds;
- fixed input/output contracts remain `[1,64,512] -> [1,128,39]`;
- PyTorch vs ONNX frame argmax agreement is 1.0;
- physical MYRIAD execution succeeds;
- inference-only RTF <= 0.05;
- p95 MYRIAD inference latency <= 100 ms.

WER/CER are recorded and compared to lineage, but are not allowed to redefine
the benchmark after execution. Generation 2 is chosen only after REVIEW.
