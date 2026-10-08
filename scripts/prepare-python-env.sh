#!/usr/bin/env bash
# Prepare named host Python environments without modifying system Python.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lib/python-env.sh"

prepare_training_rocm() {
    local venv="$ROOT/work/venv-training-rocm"
    local req="$ROOT/requirements/training-rocm.txt"
    python_env_ensure_requirements "$venv" 3.11 "$req" "import numpy, onnx, onnxruntime, soundfile"

    local rocm_version="${SPEECH_ROCM_VERSION:-}"
    if [[ -z "$rocm_version" ]]; then
        local hipconfig=""
        if command -v hipconfig >/dev/null 2>&1; then
            hipconfig="$(command -v hipconfig)"
        elif [[ -x /opt/rocm/bin/hipconfig ]]; then
            hipconfig=/opt/rocm/bin/hipconfig
        fi
        if [[ -n "$hipconfig" ]]; then
            rocm_version="$("$hipconfig" --version 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)"
        fi
    fi

    local torch_version="${SPEECH_ROCM_TORCH_VERSION:-}"
    local index_url="${SPEECH_ROCM_INDEX_URL:-}"
    if [[ -z "$index_url" || -z "$torch_version" ]]; then
        case "$rocm_version" in
            7.2*) torch_version="${torch_version:-2.13.0}"; index_url="${index_url:-https://download.pytorch.org/whl/rocm7.2}" ;;
            7.1*) torch_version="${torch_version:-2.13.0}"; index_url="${index_url:-https://download.pytorch.org/whl/rocm7.1}" ;;
            6.4*) torch_version="${torch_version:-2.9.1}"; index_url="${index_url:-https://download.pytorch.org/whl/rocm6.4}" ;;
            6.2*) torch_version="${torch_version:-2.6.0}"; index_url="${index_url:-https://download.pytorch.org/whl/rocm6.2.4}" ;;
            6.1*) torch_version="${torch_version:-2.6.0}"; index_url="${index_url:-https://download.pytorch.org/whl/rocm6.1}" ;;
            6.0*) torch_version="${torch_version:-2.3.1}"; index_url="${index_url:-https://download.pytorch.org/whl/rocm6.0}" ;;
            5.7*) torch_version="${torch_version:-2.2.2}"; index_url="${index_url:-https://download.pytorch.org/whl/rocm5.7}" ;;
            *)
                echo "Unable to select a PyTorch ROCm wheel for ROCm version '${rocm_version:-unknown}'." >&2
                echo "Set SPEECH_ROCM_VERSION, or explicitly set both SPEECH_ROCM_TORCH_VERSION and SPEECH_ROCM_INDEX_URL." >&2
                return 2
                ;;
        esac
    fi

    if ! "$venv/bin/python" - "$torch_version" <<'PY' >/dev/null 2>&1
import sys
try:
    import torch
except Exception:
    raise SystemExit(1)
want = sys.argv[1]
raise SystemExit(
    0
    if torch.__version__.split("+", 1)[0] == want and getattr(torch.version, "hip", None)
    else 1
)
PY
    then
        echo "syncing ROCm PyTorch $torch_version from $index_url" >&2
        uv pip install --python "$venv/bin/python" --index-url "$index_url" "torch==$torch_version"
    fi

    "$venv/bin/python" - <<'PY'
import torch
if not getattr(torch.version, "hip", None):
    raise SystemExit("installed torch is not a ROCm/HIP build")
print(
    f"ROCm PyTorch ready: torch={torch.__version__} "
    f"hip={torch.version.hip} available={torch.cuda.is_available()}"
)
PY
}

profile="${1:-}"
case "$profile" in
    tools) python_env_ensure_venv "$ROOT/work/venv-tools" 3.11 ;;
    apps) python_env_ensure_requirements "$ROOT/work/venv-apps" 3.11 "$ROOT/requirements/apps.txt" "import numpy, cv2, onnxruntime, soundfile" ;;
    cpu) python_env_ensure_requirements "$ROOT/work/venv-cpu" 3.11 "$ROOT/requirements/cpu-tensorflow.txt" "import numpy, cv2, tensorflow" ;;
    mo-onnx) python_env_ensure_requirements "$ROOT/work/venv-mo-onnx" 3.10 "$ROOT/requirements/mo-onnx.txt" "import numpy, onnx, networkx, defusedxml, google.protobuf" ;;
    mo-tensorflow) python_env_ensure_requirements "$ROOT/work/venv-mo-tensorflow" 3.11 "$ROOT/requirements/mo-tensorflow.txt" "import numpy, tensorflow, networkx, defusedxml, google.protobuf" ;;
    mo-kaldi) python_env_ensure_requirements "$ROOT/work/venv-mo-kaldi" 3.10 "$ROOT/requirements/mo-kaldi.txt" "import numpy, networkx, defusedxml" ;;
    training) python_env_ensure_requirements "$ROOT/work/venv-training" 3.11 "$ROOT/requirements/training.txt" "import numpy, torch, onnx, onnxruntime, soundfile" ;;
    training-rocm) prepare_training_rocm ;;
    all) "$0" tools; "$0" apps; "$0" cpu ;;
    *) echo "usage: $0 tools|apps|cpu|training|training-rocm|mo-onnx|mo-tensorflow|mo-kaldi|all" >&2; exit 2 ;;
esac
