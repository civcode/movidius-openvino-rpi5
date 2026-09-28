# Troubleshooting

- `libmyriadPlugin.so` missing: rerun `pull-runtime.sh` and export.
- device not visible: check `lsusb | grep 03e7`, udev permissions, and `/tmp/mvnc.mutex` ownership.
- Python setup fails offline: prepare/export a wheelhouse on the same CPU architecture and compatible Python ABI.
- Model Optimizer import errors: use the separate `mo-onnx` or `mo-tf` environment.
- Relocation problems: run `./verify.sh --static` from the new path and inspect `ldd runtime/openvino/bin/*`.
