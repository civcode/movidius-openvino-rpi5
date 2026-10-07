#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/python.sh"   "$ROOT/tests/python/test_speech_asr_quartznet_reference.py" "$@"
