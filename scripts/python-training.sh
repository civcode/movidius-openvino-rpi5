#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

backend="${SPEECH_TORCH_BACKEND:-cuda}"
case "$backend" in
    rocm)
        "$ROOT/scripts/prepare-python-env.sh" training-rocm >/dev/null
        exec "$ROOT/work/venv-training-rocm/bin/python" "$@"
        ;;
    auto|cpu|cuda)
        "$ROOT/scripts/prepare-python-env.sh" training >/dev/null
        exec "$ROOT/work/venv-training/bin/python" "$@"
        ;;
    *)
        echo "invalid SPEECH_TORCH_BACKEND: $backend" >&2
        exit 2
        ;;
esac
