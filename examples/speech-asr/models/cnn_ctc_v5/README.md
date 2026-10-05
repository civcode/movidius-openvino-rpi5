# cnn_ctc_v5

`cnn_ctc_v5` is the reviewed objective/encoder redesign after the v3
blank-penalty diagnostic failed. It preserves the qualified logmel-v1 tensor
contracts and MA2450 operator set while reducing the deployed encoder from
1,346,343 to 304,343 parameters.

The five residual kernels are `[9, 9, 13, 13, 17]` at 64 channels. During
training, an auxiliary CTC head after block 3 contributes 30% of the objective.
The auxiliary head is not called by `forward()` and is absent from ONNX,
OpenVINO IR and device inference.

See [`../../docs/adr/phase11-cnn-ctc-v5.md`](../../docs/adr/phase11-cnn-ctc-v5.md)
for the evidence and decision.

```bash
./scripts/init-cnn-ctc-v5-interctc.sh
./scripts/probe-cnn-ctc-v5.sh
```
