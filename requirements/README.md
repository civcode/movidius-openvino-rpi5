# Python dependency sets

These files are inputs to `prepare-standalone-python.sh`. The wheelhouse is built on the target architecture and its resolved package versions are recorded in a manifest. OpenVINO 2020.3 Model Optimizer is legacy software, so ONNX and TensorFlow frontends intentionally use separate virtual environments.

The standalone client environment (`apps.txt`) includes ONNX Runtime for MobileNet CPU fallback. SSD/DeepLab CPU fallback uses the separate `cpu-tensorflow.txt` environment so MYRIAD-only users can skip the large TensorFlow installation with `setup.sh --skip-cpu-fallback`.
