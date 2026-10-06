#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/python.sh" \
    "$ROOT/examples/speech-asr/tools/prepare_architecture_screen_v1.py" "$@"
