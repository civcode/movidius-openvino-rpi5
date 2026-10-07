#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ctc_beam imports NumPy, so run this test in the uv-managed training
# environment rather than the dependency-free tools environment.
exec "$ROOT/scripts/python-training.sh" \
  "$ROOT/tests/python/test_speech_asr_ctc_beam.py" "$@"
