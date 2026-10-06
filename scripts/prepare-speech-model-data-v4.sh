#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_DIR="$ROOT/work/speech-asr/ami/model-quality-v4-source-lock"
TRAIN_DIR="$ROOT/work/speech-asr/ami/ami-model-quality-v4-train-v1"
VALIDATION_DIR="$ROOT/work/speech-asr/ami/ami-model-quality-v4-validation-v1"

VERIFY_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify-only)
            VERIFY_ONLY=1
            shift
            ;;
        -h|--help)
            echo "usage: $0 [--verify-only]"
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            exit 2
            ;;
    esac
done

verify_split() {
    local spec="$1"
    local output="$2"
    "$ROOT/scripts/python.sh" \
        "$ROOT/examples/speech-asr/datasets/ami/prepare_ami.py" \
        --spec "$spec" \
        --output "$output" \
        --verify-only
}

prepare_or_verify() {
    local spec="$1"
    local output="$2"
    if verify_split "$spec" "$output" >/dev/null 2>&1; then
        printf '[model-quality-v4] prepared split verified; reusing %s\n' "$output"
        return
    fi
    "$ROOT/scripts/python.sh" \
        "$ROOT/examples/speech-asr/datasets/ami/prepare_ami.py" \
        --spec "$spec" \
        --output "$output"
}

if [[ "$VERIFY_ONLY" == 1 ]]; then
    "$ROOT/scripts/python.sh" \
        "$ROOT/examples/speech-asr/datasets/ami/freeze_model_quality_v4_sources.py" \
        --verify-only
    verify_split "$LOCK_DIR/train.split.json" "$TRAIN_DIR"
    verify_split "$LOCK_DIR/validation.split.json" "$VALIDATION_DIR"
    exec "$ROOT/scripts/qualify-speech-model-data-v4.sh" --verify-only
fi

"$ROOT/scripts/python.sh" \
    "$ROOT/examples/speech-asr/datasets/ami/freeze_model_quality_v4_sources.py"

prepare_or_verify "$LOCK_DIR/train.split.json" "$TRAIN_DIR"
prepare_or_verify "$LOCK_DIR/validation.split.json" "$VALIDATION_DIR"

exec "$ROOT/scripts/qualify-speech-model-data-v4.sh"
