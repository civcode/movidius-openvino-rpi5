#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/work/speech-asr/pretrained/quartznet15x5-en-base-v2/QuartzNet15x5-En-Base.nemo"
exec "$ROOT/scripts/python.sh"   "$ROOT/examples/speech-asr/training/prepare_cnn_ctc_v18_pretrained.py"   "$OUT" "$@"
