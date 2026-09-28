#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
find scripts examples standalone -name '*.sh' -type f -print0 | xargs -0 -n1 bash -n
python3 -m compileall -q examples python tests standalone/tools scripts/mo_compat_run.py scripts/patch-model-optimizer.py
python3 -m unittest discover -s tests/python -p 'test_*.py'
bash tests/shell/test-runtime.sh
timeout 45s bash scripts/test-python-clients.sh
