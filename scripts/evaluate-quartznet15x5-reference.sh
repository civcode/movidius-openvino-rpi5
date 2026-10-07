#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/python-training.sh"   "$ROOT/examples/speech-asr/evaluation/evaluate_quartznet15x5_reference.py" "$@"
