#!/usr/bin/env bash
# Prepare named host Python environments without modifying system Python.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lib/python-env.sh"

profile="${1:-}"
case "$profile" in
    tools) python_env_ensure_venv "$ROOT/work/venv-tools" 3.11 ;;
    apps) python_env_ensure_requirements "$ROOT/work/venv-apps" 3.11 "$ROOT/requirements/apps.txt" "import numpy, cv2, onnxruntime" ;;
    cpu) python_env_ensure_requirements "$ROOT/work/venv-cpu" 3.11 "$ROOT/requirements/cpu-tensorflow.txt" "import numpy, cv2, tensorflow" ;;
    mo-onnx) python_env_ensure_requirements "$ROOT/work/venv-mo-onnx" 3.10 "$ROOT/requirements/mo-onnx.txt" "import numpy, onnx, networkx, defusedxml, google.protobuf" ;;
    mo-tensorflow) python_env_ensure_requirements "$ROOT/work/venv-mo-tensorflow" 3.11 "$ROOT/requirements/mo-tensorflow.txt" "import numpy, tensorflow, networkx, defusedxml, google.protobuf" ;;
    mo-kaldi) python_env_ensure_requirements "$ROOT/work/venv-mo-kaldi" 3.10 "$ROOT/requirements/mo-kaldi.txt" "import numpy, networkx, defusedxml" ;;
    training) python_env_ensure_requirements "$ROOT/work/venv-training" 3.11 "$ROOT/requirements/training.txt" "import numpy, torch, onnx" ;;
    all) "$0" tools; "$0" apps; "$0" cpu ;;
    *) echo "usage: $0 tools|apps|cpu|training|mo-onnx|mo-tensorflow|mo-kaldi|all" >&2; exit 2 ;;
esac
