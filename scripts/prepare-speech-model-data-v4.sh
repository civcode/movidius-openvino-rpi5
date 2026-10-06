#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_DIR="$ROOT/work/speech-asr/ami/model-quality-v4-source-lock"
TRAIN_DIR="$ROOT/work/speech-asr/ami/ami-model-quality-v4-train-v1"
VALIDATION_DIR="$ROOT/work/speech-asr/ami/ami-model-quality-v4-validation-v1"

VERIFY_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify-only) VERIFY_ONLY=1; shift ;;
        -h|--help)
            echo "usage: $0 [--verify-only]"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ "$VERIFY_ONLY" == 1 ]]; then
    "$ROOT/scripts/python.sh"         "$ROOT/examples/speech-asr/datasets/ami/freeze_model_quality_v4_sources.py"         --verify-only
    "$ROOT/scripts/python.sh"         "$ROOT/examples/speech-asr/datasets/ami/prepare_ami.py"         --spec "$LOCK_DIR/train.split.json"         --output "$TRAIN_DIR"         --verify-only
    "$ROOT/scripts/python.sh"         "$ROOT/examples/speech-asr/datasets/ami/prepare_ami.py"         --spec "$LOCK_DIR/validation.split.json"         --output "$VALIDATION_DIR"         --verify-only
    exec "$ROOT/scripts/qualify-speech-model-data-v4.sh" --verify-only
fi

"$ROOT/scripts/python.sh"     "$ROOT/examples/speech-asr/datasets/ami/freeze_model_quality_v4_sources.py"
"$ROOT/scripts/python.sh"     "$ROOT/examples/speech-asr/datasets/ami/prepare_ami.py"     --spec "$LOCK_DIR/train.split.json"     --output "$TRAIN_DIR"
"$ROOT/scripts/python.sh"     "$ROOT/examples/speech-asr/datasets/ami/prepare_ami.py"     --spec "$LOCK_DIR/validation.split.json"     --output "$VALIDATION_DIR"
exec "$ROOT/scripts/qualify-speech-model-data-v4.sh"
