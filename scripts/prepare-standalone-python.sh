#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/platform.sh"
TARGET_REQUEST="$(platform_default_request)"; PYTHON="${PYTHON:-python3}"
while [[ $# -gt 0 ]]; do case "$1" in --platform) TARGET_REQUEST="$2"; shift 2;; --python) PYTHON="$2"; shift 2;; -h|--help) echo "usage: $0 [--platform arm64|armv7|amd64] [--python python3]"; exit 0;; *) echo "unknown option: $1" >&2; exit 2;; esac; done
platform_load "$TARGET_REQUEST"
OUT="$ROOT/work/standalone-python/$TARGET"; rm -rf "$OUT"; mkdir -p "$OUT/wheelhouse" "$OUT/manifests"
case "$(uname -m):$TARGET" in aarch64:arm64|arm64:arm64|x86_64:amd64|amd64:amd64) ;; *) echo "wheelhouses must be prepared natively for $TARGET; host=$(uname -m)" >&2; exit 1;; esac
for env in apps cpu-tensorflow mo-onnx mo-tensorflow; do req="$ROOT/requirements/$env.txt"; dest="$OUT/wheelhouse/$env"; mkdir -p "$dest"; "$PYTHON" -m pip download --dest "$dest" -r "$req"; done
{
  echo "TARGET=$TARGET"
  "$PYTHON" - <<'PY'
import platform,sysconfig
print('PYTHON_VERSION='+platform.python_version())
print('PYTHON_ABI='+str(sysconfig.get_config_var('SOABI') or 'unknown'))
print('PLATFORM='+sysconfig.get_platform())
PY
} > "$OUT/manifests/python.env"
find "$OUT/wheelhouse" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/manifests/wheelhouse.sha256"
echo "prepared offline wheelhouse: $OUT"
