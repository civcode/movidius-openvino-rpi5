# ADR: cnn_ctc_v19 conservative pretrained fine-tuning

Status: implemented; selected execution complete
Date: 2026-10-07
Evidence parent: `exp-7c7ac81bc57f8f3b/attempt-0001`

## v18 evidence

cnn_ctc_v18 proved that pretrained transfer changes the acoustic-quality regime.

Before any AMI optimizer update, the pinned QuartzNet15x5Base-En initialization
reached validation CER `0.5776651500316765` and WER
`0.7393651479162168`. The decoder emitted approximately the reference
character volume rather than collapsing to blank.

The v18 fine-tuning recipe then damaged that starting point:

- batch size 1;
- peak NovoGrad learning rate `0.001`;
- 12% warmup toward that peak;
- BatchNorm modules left in training mode, so pretrained running mean/variance
  were updated from single-sample AMI batches;
- epoch-zero validation was recorded but was not eligible to win checkpoint
  selection.

Epoch 1 validation CER rose to `0.9870990036284052`. The best post-update
validation CER was only `0.9003628405229511` at epoch 3. The selected
physical checkpoint reached CER `0.8975407475666648` and WER
`1.22219822932412`.

The first epoch also observed a maximum pre-clip gradient norm of
`222098.734375`.

## Decision

v19 keeps the exact v18 inference topology, frontend, vocabulary, pretrained
source, and deployment path. Only the fine-tuning policy changes:

- batch size 8;
- peak learning rate `1e-4`;
- warmup ratio 0;
- cosine decay to `1e-6`;
- freeze BatchNorm running statistics at their pretrained values after every
  call to `model.train()`;
- keep BatchNorm affine parameters trainable;
- make epoch-zero validation a real checkpoint candidate.

The final checkpoint therefore cannot be replaced by a worse post-update model
under validation-CER selection: if no fine-tuned epoch beats the pretrained
baseline, v19 reloads and deploys epoch 0.

## v19 result

The selected execution completed as `exp-f4adb44ab833e896/attempt-0002`.
Epoch 5 beat the pretrained epoch-zero baseline and became the frozen acoustic
reference:

- validation CER `0.46201693255773774`;
- validation WER `0.5983588857698121`;
- physical MYRIAD CER `0.46230490122674656`;
- physical MYRIAD WER `0.5985748218527316`;
- inference-only RTF `0.21930080610559416`;
- physical inference p50/p95 `392.428065 / 395.7003068 ms`;
- 1,273/1,273 validation utterances completed with zero failures and zero
  persistent-server restarts.

The conservative fine-tuning policy therefore succeeded: AMI adaptation
improved materially over the pretrained starting point without changing the
v18/v19 inference topology.

## Physical compatibility

v19 retains the pretrained semantic MYRIAD gate introduced for v18:

- frame argmax agreement exactly 1.0;
- frame argmax mismatches exactly 0;
- maximum per-frame total-variation distance <= 0.002.

The v18 initialized physical probe measured maximum frame TV
`0.0009230391151051188` with zero frame mismatches.

## Interpretation

This is a conservative transfer experiment, not another architecture search.
It tests whether catastrophic forgetting can be avoided while preserving the
already-useful pretrained acoustic representation. If no AMI update beats
epoch zero, the pretrained baseline itself remains valid evidence and should be
the basis for the next deployment/decoder experiment.
