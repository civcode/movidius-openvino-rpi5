#!/usr/bin/env bash
# Run the staged OpenVINO 2020.3 Model Optimizer in a framework-specific venv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMEWORK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --framework) [[ $# -ge 2 ]] || { echo "--framework needs onnx|tf" >&2; exit 2; }; FRAMEWORK="$2"; shift 2 ;;
        --) shift; break ;;
        -h|--help)
            echo "usage: $0 --framework onnx|tf -- <Model Optimizer arguments>"
            exit 0
            ;;
        *) break ;;
    esac
done
case "$FRAMEWORK" in onnx|tf) ;; *) echo "--framework must be onnx or tf" >&2; exit 2 ;; esac

MO_ROOT="$ROOT/work/model-optimizer"
if [[ ! -f "$MO_ROOT/mo.py" ]]; then
    "$ROOT/scripts/prepare-model-optimizer.sh"
fi

case "$FRAMEWORK" in
    onnx)
        VENV="$ROOT/work/venv-mo-onnx"
        REQ="$ROOT/requirements/mo-onnx.txt"
        IMPORTS='import numpy, onnx, networkx, defusedxml, google.protobuf'
        ;;
    tf)
        VENV="$ROOT/work/venv-mo-tensorflow"
        REQ="$ROOT/requirements/mo-tensorflow.txt"
        IMPORTS='import numpy, tensorflow, networkx, defusedxml, google.protobuf'
        ;;
esac

if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c "$IMPORTS" >/dev/null 2>&1; then
    echo "installing Model Optimizer $FRAMEWORK dependencies into $VENV" >&2
    "$VENV/bin/python" -m pip install --disable-pip-version-check -r "$REQ"
fi

export OV203_MO_ROOT="$MO_ROOT"
exec "$VENV/bin/python" "$ROOT/scripts/mo_compat_run.py" "$@"
