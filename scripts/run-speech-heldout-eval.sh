#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/python.sh"   "$ROOT/examples/speech-asr/agent/run_frozen_heldout_evaluation.py" "$@"
