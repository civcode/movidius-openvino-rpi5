#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

backend="${SPEECH_TORCH_BACKEND:-}"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--device" && $((i + 1)) -lt ${#args[@]} ]]; then
        case "${args[$((i + 1))]}" in
            rocm) backend=rocm ;;
            cuda) backend=cuda ;;
        esac
    fi
done
[[ -n "$backend" ]] && export SPEECH_TORCH_BACKEND="$backend"

exec "$ROOT/scripts/python-training.sh" \
  "$ROOT/examples/speech-asr/evaluation/evaluate_quartznet15x5_reference.py" "$@"
