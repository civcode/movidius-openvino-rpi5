#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/python-training.sh"   "$ROOT/examples/speech-asr/datasets/librispeech/prepare_dev_clean.py" "$@"
