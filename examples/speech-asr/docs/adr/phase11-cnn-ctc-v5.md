# ADR: cnn_ctc_v5 compact encoder with intermediate CTC

Status: accepted design for implementation
Date: 2026-10-05
Parent: exp-3c7727ca3f37ba2c (`cnn_ctc_v3` CER-selected reference)

## Evidence

The v3 reference passes export, OpenVINO 2020.3 and physical MA2450 gates, but
its held-out CER is 0.86694, blank-frame fraction is 61.33%, 19/95 hypotheses
are empty and emitted/reference non-space characters are 47.98%.

Three controlled siblings failed to improve the reference:

- SpecAugment: CER 0.90323 and 83.24% blank frames;
- valid-frame CMVN: CER 0.88407 and 68.58% blank frames;
- training-only 0.25 blank-logit penalty: CER 0.87702, 70.74% blank frames,
  22/95 empty hypotheses and 39.31% emitted/reference characters.

The final result rejects further CTC-local blank-bias tuning. The v3 encoder
also has 1,346,343 parameters and a 533-feature-frame receptive field while the
reviewed training set contains only 125 utterances / 220 seconds. Its training
loss continues falling after held-out CER has selected an earlier checkpoint,
which is consistent with excess capacity for this data boundary.

## Decision

Introduce `cnn_ctc_v5`, changing the encoder and objective together as one
reviewed generation:

1. reduce the residual width from 96 to 64 channels;
2. replace the `[11,19,27,35,43]` residual kernels with
   `[9,9,13,13,17]`;
3. preserve two stride-2 stem convolutions and the fixed `[1,128,39]` logits;
4. attach a training-only CTC projection after residual block 3;
5. train with `0.7 * final_CTC + 0.3 * intermediate_CTC`;
6. select the checkpoint by final-head validation CER;
7. retain the v3 data, frontend, optimizer schedule, vocabulary and greedy
   decoder.

The auxiliary projection is absent from the exported inference graph. The
physical graph continues to use only ordinary Conv1D, ReLU and Add operations
already qualified on MA2450.

## Resource estimate

For fixed `[1,64,512]` input:

- deployed parameters: 304,343 (77.4% below v3);
- training parameters including the auxiliary head: 306,878;
- convolution MACs: 40,820,736 (76.6% below v3);
- receptive field: 237 feature frames (about 2.37 seconds);
- temporal reduction and output shape: unchanged at 4x / `[1,128,39]`.

The smaller receptive field still spans substantially more context than the
median utterance while avoiding a whole-input receptive field at every output.

## Why intermediate CTC

Lee and Watanabe's InterCTC objective applies the same transcript at an
intermediate encoder layer and combines it with final CTC using weight 0.3.
Their experiments show that it regularizes lower encoder layers and improves
CTC greedy recognition, with no inference overhead because the intermediate
head is discarded. This is a structural objective change, unlike a constant
blank-logit bias.

CTC blank-heavy spike behavior is intrinsic to its alignment topology; the
analysis by Zeyer, Schlüter and Ney also warns that local-context encoders can
be suboptimal. This experiment therefore preserves multi-second temporal
context while reducing global over-parameterization and directly supervising
an internal representation.

## Decision rule

Compare only against `exp-3c7727ca3f37ba2c` on the identical validation
manifest. Review, in order:

1. CER (primary);
2. blank-frame fraction, empty hypotheses, and emitted/reference characters;
3. WER;
4. ONNX agreement and initialized MYRIAD numerical gate;
5. physical latency and RTF.

The candidate advances only if CER improves materially without hardware or
compatibility regression. The validation population is also used for
checkpoint selection, so the result remains model-selection evidence rather
than an unbiased final test estimate.

## References

- J. Lee and S. Watanabe, “Intermediate Loss Regularization for CTC-based
  Speech Recognition,” ICASSP 2021, https://arxiv.org/abs/2102.03216
- A. Zeyer, R. Schlüter and H. Ney, “Why does CTC result in peaky behavior?”,
  2021, https://arxiv.org/abs/2105.14849
- S. Kriman et al., “QuartzNet: Deep Automatic Speech Recognition with 1D
  Time-Channel Separable Convolutions,” 2019,
  https://arxiv.org/abs/1910.10261
