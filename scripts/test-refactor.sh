#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-$ROOT/work/venv-apps/bin/python}"
[[ -x "$PY" ]] || { echo "test environment missing: run ./scripts/prepare-python-env.sh apps" >&2; exit 2; }
find scripts examples standalone -name '*.sh' -type f -print0 | xargs -0 -n1 bash -n
for cmd in standalone/bin/* standalone/setup.sh standalone/verify.sh; do
    [[ -f "$cmd" ]] || continue
    head -1 "$cmd" | grep -qE '(^#!.*(bash|sh))' || continue
    bash -n "$cmd"
done
"$PY" -m compileall -q examples python tests standalone/tools scripts/mo_compat_run.py scripts/patch-model-optimizer.py
"$PY" -m unittest discover -s tests/python -p 'test_*.py'
timeout 10s "$PY" examples/webcam/webcam_mobilenet.py --help >/dev/null
timeout 10s "$PY" examples/ssd-detect/ssd_stream.py --help >/dev/null
timeout 10s "$PY" examples/deeplab-seg/seg_stream.py --help >/dev/null
wire_test="$(mktemp)"
device_test="$(mktemp)"
trap 'rm -f "$wire_test" "$device_test"' EXIT
g++ -std=c++14 -Wall -Wextra -I. tests/cpp/test_wire.cpp -o "$wire_test"
g++ -std=c++14 -Wall -Wextra -I. tests/cpp/test_device_spec.cpp -o "$device_test"
"$wire_test"
"$device_test"
bash tests/shell/test-runtime.sh
bash tests/shell/test-export-standalone.sh
PYTHON="$PY" timeout 45s bash scripts/test-python-clients.sh

if grep -R -n 'work/mo-2020\.3' scripts standalone python examples \
     --exclude='test-refactor.sh' >/dev/null 2>&1; then
    echo "legacy work/mo-2020.3 reference remains in active code" >&2
    exit 1
fi
