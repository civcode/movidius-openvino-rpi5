# Python dependency sets

Host Python dependency installation is managed exclusively by **uv**. These
requirement files describe separate virtual environments; they are never
installed into system Python.

- `apps.txt` -> `work/venv-apps` (Python 3.11)
- `cpu-tensorflow.txt` -> `work/venv-cpu` (Python 3.11)
- `training.txt` -> `work/venv-training` (Python 3.11)
- `mo-onnx.txt` -> `work/venv-mo-onnx` (Python 3.10)
- `mo-tensorflow.txt` -> `work/venv-mo-tensorflow` (Python 3.11)
- `mo-kaldi.txt` -> `work/venv-mo-kaldi` (Python 3.10)
- `mo-common.txt` is shared by the Model Optimizer environments.

Prepare an environment with:

```bash
./scripts/prepare-python-env.sh apps
./scripts/prepare-python-env.sh cpu
./scripts/prepare-python-env.sh training
./scripts/prepare-python-env.sh mo-kaldi
```

Dependency-free repository tools use `work/venv-tools` via
`./scripts/python.sh`. See `docs/PYTHON-ENVIRONMENTS.md` for the full policy.
