# Python environment policy

All **host-side Python dependencies** in this repository are installed into
uv-managed virtual environments. System Python is never mutated.

## Rules

- Do not use `pip install` against system Python.
- Do not create host environments with `python -m venv`.
- Do not use `uv pip install --system`.
- Generated environments live below `work/` and are not committed.
- Dependency-free repository tools run through `scripts/python.sh`.
- Legacy OpenVINO 2020.3 Model Optimizer environments stay isolated from
  application and future training environments.

`uv` itself is an external executable, not a Python package dependency.

Host wrappers resolve `uv` from the current `PATH` first, then from
`UV_INSTALL_DIR`, `XDG_BIN_HOME`, `~/.local/bin`, and `~/.cargo/bin`.
This matters for non-interactive SSH workers, whose PATH may be narrower than an
interactive login shell.

## Named environments

| Purpose | Environment | Python |
|---|---|---:|
| repository tools | `work/venv-tools` | 3.11 |
| example clients / ONNX CPU fallback | `work/venv-apps` | 3.11 |
| TensorFlow CPU fallback | `work/venv-cpu` | 3.11 |
| PyTorch training/export | `work/venv-training` | 3.11 |
| OpenVINO MO ONNX | `work/venv-mo-onnx` | 3.10 |
| OpenVINO MO TensorFlow | `work/venv-mo-tensorflow` | 3.11 |
| OpenVINO MO Kaldi | `work/venv-mo-kaldi` | 3.10 |

Examples:

```bash
./scripts/prepare-python-env.sh tools
./scripts/prepare-python-env.sh apps
./scripts/prepare-python-env.sh cpu
./scripts/prepare-python-env.sh training
./scripts/prepare-python-env.sh mo-kaldi

./scripts/python.sh examples/speech-asr/datasets/ami/prepare_ami.py --subset smoke
```

`scripts/run-mo.sh` automatically creates/synchronizes the appropriate
Model Optimizer environment through uv.

## Why separate environments

OpenVINO 2020.3 is legacy software. Its ONNX/Kaldi frontends use older NumPy
era dependency sets that should not constrain application code or future
PyTorch/CUDA training. The separate uv environments make that boundary
explicit and reproducible.

## Docker exception

Docker build/runtime images are already isolated operating-system
environments. Existing package installation inside those images may continue
where required by the pinned legacy build. The repository rule is that host
tooling must never mutate system Python.
