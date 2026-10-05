# ADR: Phase 11 generation-1 ASR architecture

Status: accepted design for implementation
Date: 2026-10-05

## Context

The project now has a physically qualified training/export/deployment loop on
OpenVINO 2020.3.2 and Intel Movidius MA2450. `cnn_ctc_v1` proved the lifecycle
but is intentionally too small to be an accuracy architecture.

Current high-performing ASR encoders are dominated by Conformer/Zipformer-like
families, while efficient convolutional systems such as QuartzNet, ContextNet
and Citrinet remain useful references. The deployment target is materially
older and smaller than the accelerators those architectures normally assume.

## Decision

Generation 1 will be `cnn_ctc_v2`, a residual large-kernel temporal CTC model.

Borrowed principles:

1. **low-rate middle encoder** — from FastConformer/Zipformer efficiency work;
2. **residual temporal convolution + CTC** — from QuartzNet/Citrinet;
3. **wide temporal context** — using increasing ordinary kernels rather than
   attention or recurrent state.

Deliberately excluded from generation 1:

- multi-head attention;
- LayerNorm/RMSNorm;
- Swish/GELU/Swoosh;
- grouped/depthwise convolution;
- squeeze/excitation;
- dilation;
- dynamic axes;
- autoregressive/transducer decoder.

The exclusions reduce operator risk on the pinned legacy Model Optimizer and
MYRIAD plugin. Historical VPU grouped-convolution behavior is specifically a
reason not to reproduce QuartzNet/Citrinet time-channel separable convolution
literally.

## Fixed architecture

```text
[1,64,512]
  -> Conv1d 64, k5, s2 + BN + ReLU
  -> Conv1d 96, k5, s2 + BN + ReLU
  -> residual k11 @ 96
  -> residual k19 @ 96
  -> residual k27 @ 96
  -> residual k35 @ 96
  -> residual k43 @ 96
  -> Conv1d 39, k1
  -> transpose
[1,128,39]
```

Each residual block is:

```text
x -> Conv1d(k) -> BN -> ReLU -> Dropout(0.1) -> Conv1d(k1) -> BN -> Add(x) -> ReLU
```

Dropout is training-only and disappears in inference.

## Resource model

For fixed `[1,64,512]` input:

- parameters: 1,347,463;
- MACs: 174,804,992;
- FP16 weights: ~2.6 MiB;
- temporal reduction: 4x;
- receptive field: 533 feature frames.

The measured v1 MYRIAD p95 latency was about 4.4 ms at roughly 13.5M MACs.
Linear scaling is not assumed, but this gives enough margin to set a strict
generation-1 hardware gate of 100 ms p95 / 0.05 inference-only RTF.

## References

- QuartzNet, arXiv:1910.10261
- ContextNet, arXiv:2005.03191
- Citrinet, arXiv:2104.01721
- Fast Conformer, arXiv:2305.05084
- Zipformer, arXiv:2310.11230
- OpenVINO VPU grouped-convolution issue #4073
