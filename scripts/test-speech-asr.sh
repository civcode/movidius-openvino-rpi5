#!/usr/bin/env bash
# Run the speech-ASR unit/integration-with-fixtures test suite.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-$ROOT/work/venv-tools/bin/python}"
if [[ ! -x "$PY" ]]; then
    "$ROOT/scripts/prepare-python-env.sh" tools
fi
exec "$PY" -m unittest discover -s tests/python -p 'test_speech_asr_*.py' -v
