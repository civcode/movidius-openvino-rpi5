#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATIC=0
[[ "${1:-}" == --static ]] && STATIC=1
fail=0
need(){ [[ -e "$ROOT/$1" ]] || { echo "MISSING: $1" >&2; fail=1; }; }
note(){ echo "verify: $*"; }

for p in MANIFEST.env SHA256SUMS bin/ov-env bin/ov-run bin/ov-device-list          bin/classify bin/detect bin/segment runtime/openvino/inference_engine          runtime/openvino/ngraph/lib runtime/openvino/bin/hello_myriad          python/ov203 udev/99-movidius.rules; do
  need "$p"
done
source "$ROOT/MANIFEST.env"

if [[ "${MODEL_OPTIMIZER_INCLUDED:-0}" == 1 ]]; then
  for p in tools/model-optimizer/mo.py tools/mo-launcher.py bin/mo bin/mo-onnx            bin/mo-tf bin/compile-model bin/build-model; do need "$p"; done
fi

if ! source "$ROOT/bin/ov-env"; then
  echo "cannot initialize standalone OpenVINO environment" >&2
  fail=1
else
  for p in libinference_engine.so libmyriadPlugin.so plugins.xml usb-ma2450.mvcmd; do
    [[ -f "$OV_IE_LIB/$p" ]] || { echo "MISSING runtime file: $OV_IE_LIB/$p" >&2; fail=1; }
  done

  hello="$OV_ROOT/bin/hello_myriad"
  if command -v readelf >/dev/null 2>&1 && [[ -f "$hello" ]]; then
    machine="$(readelf -h "$hello" 2>/dev/null | awk -F: '/Machine:/{gsub(/^[ 	]+/,"",$2); print $2}')"
    case "${TARGET:-}" in
      arm64) [[ "$machine" == *AArch64* ]] || { echo "ELF target mismatch: expected AArch64, got $machine" >&2; fail=1; } ;;
      armv7) [[ "$machine" == ARM* ]] || { echo "ELF target mismatch: expected ARM, got $machine" >&2; fail=1; } ;;
      amd64) [[ "$machine" == *X86-64* || "$machine" == *x86-64* ]] || { echo "ELF target mismatch: expected x86-64, got $machine" >&2; fail=1; } ;;
    esac
  fi

  for b in hello_myriad mobilenet_server ssd_detect seg_detect; do
    x="$OV_ROOT/bin/$b"
    [[ -x "$x" ]] || { echo "MISSING executable: $x" >&2; fail=1; continue; }
    if command -v ldd >/dev/null 2>&1; then
      bad="$(LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ldd "$x" 2>/dev/null | grep 'not found' || true)"
      [[ -z "$bad" ]] || { echo "$bad" >&2; fail=1; }
    fi
  done

  if [[ "${MODEL_OPTIMIZER_INCLUDED:-0}" == 1 ]]; then
    [[ -x "$OV_IE_LIB/compile_tool" || -x "$OV_IE_LIB/myriad_compile" ]] || {
      echo "MYRIAD compiler missing" >&2; fail=1;
    }
  fi
fi

(cd "$ROOT" && sha256sum -c SHA256SUMS >/dev/null) || {
  echo "checksum verification failed" >&2; fail=1;
}

if grep -R -n -E 'work/host-runtime|work/mo-2020\.3|vendor/openvino|/opt/openvino-demo'      "$ROOT/bin" "$ROOT/tools" "$ROOT/python" "$ROOT/setup.sh" "$ROOT/verify.sh"      2>/dev/null; then
  echo "active standalone files contain source-tree/container-only path references" >&2
  fail=1
fi

PYTHONPATH="$ROOT/python" python3 - <<'PY' || fail=1
import ov203
from ov203.device import parse_device_spec
assert parse_device_spec("HETERO:MYRIAD").uses_myriad
print("python package: OK")
PY

check_env(){
  local env="$1"; shift
  local py="$ROOT/.envs/$env/bin/python"
  [[ -x "$py" ]] || { note "$env environment not installed; import smoke skipped"; return 0; }
  PYTHONPATH="$ROOT/python" "$py" - "$@" <<'PY' || return 1
import importlib,sys
for name in sys.argv[1:]:
    importlib.import_module(name)
print("environment imports OK:", ", ".join(sys.argv[1:]))
PY
}
check_env apps numpy cv2 ov203 || fail=1
if [[ "${TARGET:-}" == arm64 && "${CPU_FALLBACK_INCLUDED:-0}" == 1 ]]; then
  check_env cpu-tensorflow numpy tensorflow || fail=1
fi
if [[ "${MODEL_OPTIMIZER_INCLUDED:-0}" == 1 ]]; then
  check_env mo-onnx numpy onnx networkx defusedxml || fail=1
  check_env mo-tensorflow numpy tensorflow networkx defusedxml || fail=1
fi

(( fail == 0 )) || exit 1
if (( ! STATIC )); then
  if command -v lsusb >/dev/null 2>&1 && lsusb | grep -qi '03e7'; then
    "$ROOT/bin/ov-device-list"
  else
    echo "hardware test skipped: no 03e7 MYRIAD USB device visible"
  fi
fi
echo "standalone verification: PASS"
