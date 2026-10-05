# cnn_ctc_v6

`cnn_ctc_v6` isolates intermediate CTC supervision from the encoder downsizing
that caused v5 to regress. Its deployed encoder geometry and seed-1337
initialization match `cnn_ctc_v3`. A training-only CTC head after residual block
3 contributes 30% of the objective and is excluded from export.

See [`../../docs/adr/phase11-cnn-ctc-v6.md`](../../docs/adr/phase11-cnn-ctc-v6.md).

```bash
./scripts/init-cnn-ctc-v6-interctc.sh
./scripts/probe-cnn-ctc-v6.sh
```
