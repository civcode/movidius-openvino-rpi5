# ADR: cnn_ctc_v14 large-capacity cosine control

Status: implemented; execution pending
Date: 2026-10-06
Parent: `exp-968150d44c4479ac/attempt-0001`

v13 removed the catastrophic v12 learning-rate explosion, but the large graph
still degraded as OneCycle increased LR. Its best validation CER was
`0.9755802568680527` at epoch 1, when LR had only risen from `3e-4` to
about `4.61e-4`. Emission then deteriorated and validation became fully blank
by epoch 6.

v14 keeps the complete 19,627,687-parameter inference graph unchanged and
restores the historically stable optimizer schedule used by the successful
small-model family:

- Adam;
- initial/maximum LR `3e-4`;
- cosine decay with no LR increase;
- final LR `3e-5`;
- gradient clipping `5.0`;
- 12 epochs, batch 1, validation-CER checkpoint selection.

This is the final optimizer-only control for the 20M normalization-free graph.
If it still collapses or remains far behind v10/v11, further LR sweeps are not
justified. The next controlled change should target architectural optimization
and conditioning rather than capacity or learning rate.

The sealed held-out benchmark remains unavailable for selection.
