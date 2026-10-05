#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/python.sh" \
    "$ROOT/examples/speech-asr/tools/qualify_expanded_model_quality_manifests.py" \
    --train-source "$ROOT/work/speech-asr/ami/ami-train-es2005-es2007-v1/manifest.jsonl" \
    --output "$ROOT/work/speech-asr/ami/model-quality-v3" "$@"
