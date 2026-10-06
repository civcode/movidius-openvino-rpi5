# ADR: cnn_ctc_v16 QuartzNet-15x5 architecture reset

Status: implemented; execution pending
Date: 2026-10-06
Parent: `exp-5d1209f5120388f8/attempt-0001`

## Evidence

cnn_ctc_v15 completed at physical CER `0.8428267004549905`, WER
`0.9941697257611747`, blank-frame fraction `0.8668262967005881`, and
MA2450 p95 `103.8926016 ms`. It regressed against v14 CER
`0.7898404653573691` and against the much smaller v11 CER
`0.7865576225306686`.

The v15 decision rule closed further tuning of the home-grown 20M residual
graph if BatchNorm did not materially beat v14. That condition is met.

## External architecture evidence

QuartzNet-15x5 is a published character-CTC acoustic architecture built from
1D time-channel separable convolutions, BatchNorm, ReLU, dropout and projected
residuals. Its five block types use kernels 33/39/51/63/75, each block contains
five modules, and each block type is repeated three times. The published model
is about 18.9M parameters.

Open Model Zoo publishes `quartznet-15x5-en` and documents conversion from
the NeMo model to OpenVINO IR with a 64-bin acoustic input. This is strong
evidence for the operator family, but it is not proof that this exact fixed
512-frame adaptation executes on MA2450/OpenVINO 2020.3; the physical initialized
probe remains mandatory.

Sources:

- https://arxiv.org/abs/1910.10261
- https://docs.openvino.ai/2023.3/omz_models_model_quartznet_15x5_en.html
- https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/asr/configs.html

## Decision

Implement v16 as the QuartzNet-15x5 topology, adapting only project interfaces:

- retain frozen logmel-v1 `[1,64,512]` input and 39-token vocabulary;
- use the released NeMo-style separable C1/C2 encoder configuration;
- preserve the QuartzNet 15x5 block/channel/kernel schedule;
- emit 256 CTC frames because QuartzNet downsamples time once at C1;
- keep the CTC acoustic graph on MYRIAD and leave decoding on CPU;
- keep greedy decoding for this acoustic screen;
- keep the screen's Adam cosine schedule and 12-epoch validation-CER selection
  so this run primarily answers the architecture question;
- do not add SpecAugment, LM scoring, beam-search tuning or pretrained weights
  in the same experiment.

If the initialized graph cannot convert/execute on MYRIAD, stop before training.
If it executes and materially improves acoustic CER, add CPU CTC prefix beam
search and LM scoring as the next isolated system experiment.
