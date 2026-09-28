#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/platform.sh"
TARGET_REQUEST="$(platform_default_request)"; OUTPUT=""; MODELS=0; WHEELS=0; PROFILE=development
while [[ $# -gt 0 ]]; do case "$1" in --platform) TARGET_REQUEST="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;; --include-example-models) MODELS=1; shift;; --with-wheelhouse) WHEELS=1; shift;; --profile) PROFILE="$2"; shift 2;; -h|--help) echo "usage: $0 --platform arm64|armv7|amd64 [--profile runtime|development] --output DIR [--include-example-models] [--with-wheelhouse]"; exit 0;; *) echo "unknown option: $1" >&2; exit 2;; esac; done
platform_load "$TARGET_REQUEST"; [[ "$PROFILE" == runtime || "$PROFILE" == development ]] || { echo "profile must be runtime|development" >&2; exit 2; }
OUTPUT="${OUTPUT:-$ROOT/dist/openvino-movidius-${PROFILE}-${TARGET}}"; RT="$ROOT/work/host-runtime/$TARGET"
[[ -d "$RT/openvino" ]] || { echo "missing $RT/openvino; run scripts/pull-runtime.sh --platform $TARGET" >&2; exit 1; }
if [[ "$PROFILE" == development ]]; then [[ -d "$ROOT/work/model-optimizer" ]] || { echo "missing work/model-optimizer; run scripts/prepare-model-optimizer.sh" >&2; exit 1; }; fi
rm -rf "$OUTPUT"; mkdir -p "$OUTPUT"/{bin,runtime,tools,python,apps,models,requirements,wheelhouse,udev,docs,tests/smoke}
cp -a "$RT/openvino" "$OUTPUT/runtime/openvino"
[[ -d "$RT/openvino-demo" ]] && cp -a "$RT/openvino-demo" "$OUTPUT/runtime/openvino-demo"
cp -a "$ROOT/python/ov203" "$OUTPUT/python/ov203"
cp "$ROOT/examples/mobilenet_client.py" "$OUTPUT/apps/mobilenet_client.py"
cp -a "$ROOT/examples/webcam" "$OUTPUT/apps/classification"
cp -a "$ROOT/examples/ssd-detect" "$OUTPUT/apps/detection"
cp -a "$ROOT/examples/deeplab-seg" "$OUTPUT/apps/segmentation"
rm -rf "$OUTPUT/apps"/*/__pycache__ "$OUTPUT/python"/*/__pycache__
cp "$ROOT/scripts/99-movidius.rules" "$OUTPUT/udev/99-movidius.rules"
cp -a "$ROOT/requirements/." "$OUTPUT/requirements/"
cp "$ROOT/docs/PROTOCOLS.md" "$OUTPUT/docs/PROTOCOLS.md"
cp -a "$ROOT/standalone/bin/." "$OUTPUT/bin/"
cp "$ROOT/standalone/setup.sh" "$OUTPUT/setup.sh"; cp "$ROOT/standalone/verify.sh" "$OUTPUT/verify.sh"
cp -a "$ROOT/standalone/docs/." "$OUTPUT/docs/" 2>/dev/null || true
if [[ "$PROFILE" == development ]]; then cp -a "$ROOT/work/model-optimizer" "$OUTPUT/tools/model-optimizer"; cp "$ROOT/standalone/tools/mo-launcher.py" "$OUTPUT/tools/mo-launcher.py"; else rm -f "$OUTPUT/bin/mo" "$OUTPUT/bin/mo-onnx" "$OUTPUT/bin/mo-tf" "$OUTPUT/bin/compile-model" "$OUTPUT/bin/build-model"; fi
if (( MODELS )); then [[ -d "$ROOT/vendor/models" ]] || { echo "vendor/models missing" >&2; exit 1; }; cp -a "$ROOT/vendor/models/." "$OUTPUT/models/"; else echo 'Example models were not included. Set OV_MODELS_DIR or re-export with --include-example-models.' > "$OUTPUT/models/README.md"; fi
if (( WHEELS )); then W="$ROOT/work/standalone-python/$TARGET/wheelhouse"; [[ -d "$W" ]] || { echo "missing $W; run scripts/prepare-standalone-python.sh --platform $TARGET" >&2; exit 1; }; cp -a "$W/." "$OUTPUT/wheelhouse/"; fi
chmod +x "$OUTPUT/bin"/* "$OUTPUT/setup.sh" "$OUTPUT/verify.sh" "$OUTPUT/tools/mo-launcher.py" 2>/dev/null || true
rev=unknown; git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1 && rev="$(git -C "$ROOT" rev-parse HEAD)"
cat > "$OUTPUT/MANIFEST.env" <<M
PRODUCT=openvino-movidius-$PROFILE
PRODUCT_VERSION=1
TARGET=$TARGET
OPENVINO_VERSION=2020.3.2
PROJECT_REVISION=$rev
MODEL_OPTIMIZER_INCLUDED=$([[ "$PROFILE" == development ]] && echo 1 || echo 0)
CPU_FALLBACK_INCLUDED=$([[ "$TARGET" == armv7 ]] && echo 0 || echo 1)
EXAMPLE_MODELS_INCLUDED=$MODELS
WHEELHOUSE_INCLUDED=$WHEELS
M
printf 'openvino-movidius-%s-%s\n' "$PROFILE" "$TARGET" > "$OUTPUT/VERSION"
(cd "$OUTPUT" && find . -type f ! -name SHA256SUMS ! -path './.envs/*' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "standalone export complete: $OUTPUT"
