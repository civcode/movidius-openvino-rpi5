#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/python-apps.sh" \
  "$ROOT/examples/speech-asr/evaluation/test_quartznet15x5_reference_myriad_residency.py" \
  "$@"
