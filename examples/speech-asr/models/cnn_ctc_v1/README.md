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
