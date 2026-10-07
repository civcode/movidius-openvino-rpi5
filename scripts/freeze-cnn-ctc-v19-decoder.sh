#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/python-training.sh" \
  "$ROOT/examples/speech-asr/tools/freeze_cnn_ctc_v19_decoder.py" "$@"
