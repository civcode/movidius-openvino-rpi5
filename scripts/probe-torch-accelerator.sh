#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEVICE=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            [[ $# -ge 2 ]] || { echo "--device needs cuda|rocm" >&2; exit 2; }
            DEVICE="$2"
            ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

case "$DEVICE" in
    cuda|rocm) ;;
    *) echo "usage: $0 --device cuda|rocm [--device-index N] [--matrix-size N] [--iterations N]" >&2; exit 2 ;;
esac

export SPEECH_TORCH_BACKEND="$DEVICE"
exec "$ROOT/scripts/python-training.sh" \
    "$ROOT/examples/speech-asr/tools/probe_torch_accelerator.py" \
    "${ARGS[@]}"
