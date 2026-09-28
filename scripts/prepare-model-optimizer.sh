#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${OV203_MO_SOURCE:-$ROOT/vendor/openvino-2020.3.2/model-optimizer}"
DEST="${OV203_MO_STAGE:-$ROOT/work/model-optimizer}"
[[ -d "$SRC" && -f "$SRC/mo.py" ]] || { echo "Model Optimizer source not found: $SRC" >&2; exit 1; }
rm -rf "$DEST"; mkdir -p "$DEST"; cp -a "$SRC/." "$DEST/"
python3 "$ROOT/scripts/patch-model-optimizer.py" "$DEST"
commit="unknown"
if [[ -d "$ROOT/vendor/openvino-2020.3.2/.git" ]]; then commit="$(git -C "$ROOT/vendor/openvino-2020.3.2" rev-parse HEAD)"; fi
{
  echo 'OPENVINO_VERSION=2020.3.2'
  echo "OPENVINO_COMMIT=$commit"
  echo 'COMPAT_PATCH_REVISION=1'
  find "$DEST" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | sed 's/^/MODEL_OPTIMIZER_TREE_SHA256=/'
} > "$DEST/MANIFEST.env"
echo "staged Model Optimizer: $DEST"
