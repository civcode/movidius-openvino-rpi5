#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET=amd64
RT="$ROOT/work/host-runtime/$TARGET"
MO="$ROOT/work/model-optimizer"
TMP="$(mktemp -d)"
CREATED_RT=0
CREATED_MO=0

cleanup() {
    (( CREATED_RT )) && rm -rf "$RT"
    (( CREATED_MO )) && rm -rf "$MO"
    rm -rf "$TMP"
}
trap cleanup EXIT

if [[ ! -d "$RT/openvino" ]]; then
    CREATED_RT=1
    mkdir -p "$RT/openvino"
fi
if [[ ! -d "$MO" ]]; then
    CREATED_MO=1
    mkdir -p "$MO"
    printf '# fake staged MO for exporter structure test\n' > "$MO/mo.py"
fi

runtime_out="$TMP/runtime"
"$ROOT/scripts/export-standalone.sh"     --profile runtime --platform "$TARGET" --output "$runtime_out" >/dev/null

[[ -f "$runtime_out/MANIFEST.env" ]]
grep -q '^MODEL_OPTIMIZER_INCLUDED=0$' "$runtime_out/MANIFEST.env"
grep -q '^CPU_FALLBACK_INCLUDED=1$' "$runtime_out/MANIFEST.env"
[[ ! -e "$runtime_out/bin/mo" ]]
[[ ! -e "$runtime_out/bin/build-model" ]]
[[ -x "$runtime_out/bin/classify" ]]
[[ -x "$runtime_out/setup.sh" ]]
(cd "$runtime_out" && sha256sum -c SHA256SUMS >/dev/null)

relocated="$TMP/path with spaces/runtime bundle"
mkdir -p "$(dirname "$relocated")"
mv "$runtime_out" "$relocated"
(cd "$relocated" && sha256sum -c SHA256SUMS >/dev/null)
"$relocated/bin/ov-info" | grep -F "BUNDLE_ROOT=$relocated" >/dev/null

dev_out="$TMP/development"
"$ROOT/scripts/export-standalone.sh"     --profile development --platform "$TARGET" --output "$dev_out" >/dev/null

grep -q '^MODEL_OPTIMIZER_INCLUDED=1$' "$dev_out/MANIFEST.env"
[[ -f "$dev_out/tools/model-optimizer/mo.py" ]]
[[ -x "$dev_out/bin/mo" ]]
[[ -x "$dev_out/bin/compile-model" ]]
[[ -x "$dev_out/bin/build-model" ]]
(cd "$dev_out" && sha256sum -c SHA256SUMS >/dev/null)

echo "standalone exporter shell tests: PASS"
