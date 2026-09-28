#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK=0; UDEV=0; SKIP_PY=0
while [[ $# -gt 0 ]]; do case "$1" in --check) CHECK=1; shift;; --install-udev) UDEV=1; shift;; --skip-python) SKIP_PY=1; shift;; -h|--help) echo "usage: ./setup.sh [--check] [--install-udev] [--skip-python]"; exit 0;; *) echo "unknown option: $1" >&2; exit 2;; esac; done
[[ -f "$ROOT/MANIFEST.env" ]] && source "$ROOT/MANIFEST.env"
arch="$(uname -m)"; case "${TARGET:-}" in arm64) [[ "$arch" =~ ^(aarch64|arm64)$ ]] || { echo "bundle target arm64 does not match host $arch" >&2; exit 1; };; amd64) [[ "$arch" =~ ^(x86_64|amd64)$ ]] || { echo "bundle target amd64 does not match host $arch" >&2; exit 1; };; esac
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
command -v ldconfig >/dev/null && { ldconfig -p 2>/dev/null | grep -q 'libusb-1.0.so.0' || echo "warning: libusb-1.0 runtime not visible in ldconfig" >&2; }
if (( CHECK )); then "$ROOT/verify.sh" --static; exit $?; fi
if (( ! SKIP_PY )); then
  for env in apps mo-onnx mo-tensorflow; do
    req="$ROOT/requirements/$env.txt"; wheels="$ROOT/wheelhouse/$env"
    [[ -f "$req" ]] || continue
    python3 -m venv "$ROOT/.envs/$env"
    pip=("$ROOT/.envs/$env/bin/python" -m pip install)
    if [[ -d "$wheels" && -n "$(find "$wheels" -type f -print -quit 2>/dev/null)" ]]; then "${pip[@]}" --no-index --find-links "$wheels" -r "$req"; else echo "warning: no offline wheelhouse for $env; installing from configured pip indexes" >&2; "${pip[@]}" -r "$req"; fi
  done
fi
if (( UDEV )); then
  sudo install -m 0644 "$ROOT/udev/99-movidius.rules" /etc/udev/rules.d/99-movidius.rules
  sudo udevadm control --reload-rules; sudo udevadm trigger
fi
"$ROOT/verify.sh" --static
