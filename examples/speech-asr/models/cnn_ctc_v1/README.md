# cnn_ctc_v1

`cnn_ctc_v1` is the first trainable custom-model skeleton. Its purpose is to
prove the PyTorch -> ONNX -> OpenVINO 2020.3 -> MYRIAD lifecycle, not to establish
an accuracy baseline.

The model accepts a fixed host-computed log-mel tensor:

```text
canonical audio
 -> logmel-v1 (64 bins, 25 ms window, 10 ms hop)
 -> [1,64,512] features
 -> Conv1D/ReLU temporal network
 -> [1,128,39] CTC logits
 -> greedy CTC decoder
```

The exported network intentionally excludes STFT/mel operations. This keeps the
deployment graph small and legacy-friendly while the exact frontend remains
declared in `model_spec.json`.

The vocabulary contains blank, space, apostrophe, lowercase a-z and digits 0-9.
It matches the character set preserved by `text-v1`.

The fixed 512-frame input corresponds to 82,160 canonical audio samples
(~5.135 s). Shorter clips are zero-padded. Longer clips are rejected by the
training/evaluation baseline rather than silently changing transcript alignment;
streaming/windowed training is a later optimization.


## Compatibility gate

Use `scripts/probe-cnn-ctc-v1.sh` before full training:

```text
deterministic initialization
 -> fixed PyTorch golden input/output
 -> ONNX opset 11 export + checker
 -> ONNX Runtime comparison
 -> OpenVINO 2020.3 FP16 Model Optimizer conversion
 -> IR contract inspection
 -> optional MYRIAD load/compile/infer
 -> MYRIAD-vs-PyTorch tensor comparison
```

The tensor comparison reports signed mean error, MAE, RMS, maximum absolute
error and framewise CTC argmax agreement. It is evidence-only until this custom
model family has an explicitly frozen numerical acceptance policy.

If the legacy MYRIAD compiler rejects the current rank-3 Conv1D graph, treat
that as a compatibility failure before training. The architecture can then be
re-expressed as an equivalent fixed Conv2D-over-time graph in the next design
revision rather than spending a CUDA training budget on an undeployable model.


### OpenVINO output naming

OpenVINO 2020.3 Model Optimizer may not preserve the ONNX logical output name
`logits` in IR v10. The converted graph can expose the sole output producer
under a generated name such as `/Transpose`.

For `cnn_ctc_v1`, IR acceptance therefore requires exactly one output with the
declared fixed shape `[1,128,39]`; the logical package name remains
`logits`, while the actual IR producer/result names are recorded as
provenance. A generated serializer name is not treated as a model-semantic
difference.
