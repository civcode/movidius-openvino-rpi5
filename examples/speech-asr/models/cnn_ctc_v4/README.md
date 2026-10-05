# cnn_ctc_v4

`cnn_ctc_v4` is a frontend-correction generation. It deliberately keeps the
qualified `cnn_ctc_v3` residual temporal inference graph unchanged.

The change is host-side feature normalization:

- `logmel-v1`: per-mel-bin mean includes all 512 fixed-shape frames, including
  frames derived entirely from zero-padded audio.
- `logmel-v2`: compute the mean from valid utterance frames only, subtract it
  from the fixed feature tensor, then set invalid/padded feature frames exactly
  to zero.

The tensor and deployment contracts remain:

```text
input   [1,64,512] float32 NCT
output  [1,128,39] float32 NTV
ONNX    opset 11, fixed shape
IR      OpenVINO 2020.3 FP16
device  MYRIAD / Intel Movidius MA2450
```

The network geometry, parameter count, MAC estimate, vocabulary, optimizer
schedule and CER-aligned checkpoint-selection policy are inherited unchanged
from the accepted v3 model-quality reference.

The generation exists because fixed-shape padding is an execution artifact and
must not influence utterance CMVN statistics. For short clips, the historical
frontend allowed padding to dominate the per-bin mean and made the feature
distribution depend on clip duration.

Initialize the reviewed experiment with:

```bash
./scripts/init-cnn-ctc-v4-valid-cmvn.sh
```

This experiment branches from `exp-3c7727ca3f37ba2c`, not from the
SpecAugment sibling.
