# Python dependency sets

These files are inputs to the repository's staged Python tooling. OpenVINO
2020.3 Model Optimizer is legacy software, so ONNX, TensorFlow and Kaldi
frontends intentionally use separate virtual environments.

- `mo-onnx.txt` — ONNX Model Optimizer frontend.
- `mo-tensorflow.txt` — TensorFlow Model Optimizer frontend.
- `mo-kaldi.txt` — Kaldi nnet1/nnet2 frontend used by speech-ASR rm_cnn4a.
- `mo-common.txt` — dependencies shared by the MO frontends.

The standalone client environment (`apps.txt`) includes ONNX Runtime for
MobileNet CPU fallback. SSD/DeepLab CPU fallback uses the separate
`cpu-tensorflow.txt` environment so MYRIAD-only users can skip the large
TensorFlow installation with `setup.sh --skip-cpu-fallback`.
