#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/python.sh" \
    "$ROOT/examples/speech-asr/agent/init_cnn_ctc_v5_interctc.py" "$@"
