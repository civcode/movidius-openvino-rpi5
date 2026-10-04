#!/usr/bin/env bash
# Run the staged OpenVINO 2020.3 Model Optimizer in a framework-specific venv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lib/python-env.sh"
FRAMEWORK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --framework) [[ $# -ge 2 ]] || { echo "--framework needs onnx|tf|kaldi" >&2; exit 2; }; FRAMEWORK="$2"; shift 2 ;;
        --) shift; break ;;
        -h|--help)
            echo "usage: $0 --framework onnx|tf|kaldi -- <Model Optimizer arguments>"
            exit 0
            ;;
        *) break ;;
    esac
done
case "$FRAMEWORK" in onnx|tf|kaldi) ;; *) echo "--framework must be onnx, tf or kaldi" >&2; exit 2 ;; esac

MO_ROOT="$ROOT/work/model-optimizer"
if [[ ! -f "$MO_ROOT/mo.py" ]]; then
    "$ROOT/scripts/prepare-model-optimizer.sh"
fi

case "$FRAMEWORK" in
    onnx)
        VENV="$ROOT/work/venv-mo-onnx"
        REQ="$ROOT/requirements/mo-onnx.txt"
        IMPORTS='import numpy, onnx, networkx, defusedxml, google.protobuf'
        PYTHON_REQUEST=3.10
        ;;
    tf)
        VENV="$ROOT/work/venv-mo-tensorflow"
        REQ="$ROOT/requirements/mo-tensorflow.txt"
        IMPORTS='import numpy, tensorflow, networkx, defusedxml, google.protobuf'
        PYTHON_REQUEST=3.11
        ;;
    kaldi)
        VENV="$ROOT/work/venv-mo-kaldi"
        REQ="$ROOT/requirements/mo-kaldi.txt"
        IMPORTS='import numpy, networkx, defusedxml'
        PYTHON_REQUEST=3.10
        ;;
esac

python_env_ensure_requirements "$VENV" "$PYTHON_REQUEST" "$REQ" "$IMPORTS"

export OV203_MO_ROOT="$MO_ROOT"
exec "$VENV/bin/python" "$ROOT/scripts/mo_compat_run.py" "$@"
