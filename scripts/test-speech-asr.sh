#!/usr/bin/env bash
# Run the speech-ASR unit/integration-with-fixtures test suite.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec python3 -m unittest discover     -s tests/python     -p 'test_speech_asr_*.py'     -v
