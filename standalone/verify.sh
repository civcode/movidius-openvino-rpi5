#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; STATIC=0
[[ "${1:-}" == --static ]] && STATIC=1
fail=0
need(){ [[ -e "$ROOT/$1" ]] || { echo "MISSING: $1" >&2; fail=1; }; }
for p in runtime/openvino/inference_engine runtime/openvino/ngraph/lib runtime/openvino/bin/hello_myriad python/ov203 tools/model-optimizer/mo.py bin/compile-model bin/build-model bin/mo udev/99-movidius.rules MANIFEST.env; do need "$p"; done
source "$ROOT/bin/ov-env" || fail=1
for b in hello_myriad mobilenet_server ssd_detect seg_detect; do
  x="$OV_ROOT/bin/$b"; [[ -x "$x" ]] || { echo "MISSING executable: $x" >&2; fail=1; continue; }
  if command -v ldd >/dev/null 2>&1; then bad="$(LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ldd "$x" 2>/dev/null | grep 'not found' || true)"; [[ -z "$bad" ]] || { echo "$bad" >&2; fail=1; }; fi
done
[[ -x "$OV_IE_LIB/compile_tool" || -x "$OV_IE_LIB/myriad_compile" ]] || { echo "MYRIAD compiler missing" >&2; fail=1; }
if [[ -f "$ROOT/SHA256SUMS" ]]; then (cd "$ROOT" && sha256sum -c SHA256SUMS >/dev/null) || { echo "checksum verification failed" >&2; fail=1; }; fi
PYTHONPATH="$ROOT/python" python3 - <<'PY' || fail=1
import ov203
from ov203.device import parse_device_spec
assert parse_device_spec('HETERO:MYRIAD').uses_myriad
print('python package: OK')
PY
(( fail == 0 )) || exit 1
if (( ! STATIC )); then
  if command -v lsusb >/dev/null 2>&1 && lsusb | grep -qi '03e7'; then "$ROOT/bin/ov-device-list"; else echo "hardware test skipped: no 03e7 MYRIAD USB device visible"; fi
fi
echo "standalone verification: PASS"
