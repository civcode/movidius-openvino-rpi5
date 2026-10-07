#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/python.sh" \
  "$ROOT/examples/speech-asr/evaluation/analyze_quartznet15x5_reference_myriad_lengths.py" "$@"
