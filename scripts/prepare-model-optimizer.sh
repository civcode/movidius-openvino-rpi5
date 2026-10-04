#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${OV203_MO_SOURCE:-$ROOT/vendor/openvino-2020.3.2/model-optimizer}"
DEST="${OV203_MO_STAGE:-$ROOT/work/model-optimizer}"
[[ -d "$SRC" && -f "$SRC/mo.py" ]] || {
    echo "Model Optimizer source not found: $SRC" >&2
    exit 1
}

rm -rf "$DEST"
mkdir -p "$DEST"
# The 2020.3 extension loader recursively imports Python modules. Unit-test and
# automation modules depend on helpers that are not part of the distributable
# MO tree, so stage the same production-only subset the original recipes used.
rsync -a \
    --exclude='*_test.py' \
    --exclude='automation/' \
    --exclude='install_prerequisites/' \
    "$SRC/" "$DEST/"

"$ROOT/scripts/python.sh" "$ROOT/scripts/patch-model-optimizer.py" "$DEST"

commit=unknown
if [[ -d "$ROOT/vendor/openvino-2020.3.2/.git" ]]; then
    commit="$(git -C "$ROOT/vendor/openvino-2020.3.2" rev-parse HEAD)"
fi

# Hash file contents using paths relative to the staged root so the manifest is
# stable when the repository is checked out in a different absolute directory.
tree_hash="$(
    cd "$DEST"
    find . -type f ! -name MANIFEST.env -print0 |
        sort -z |
        xargs -0 sha256sum |
        sha256sum |
        awk '{print $1}'
)"
{
    echo 'OPENVINO_VERSION=2020.3.2'
    echo "OPENVINO_COMMIT=$commit"
    echo 'COMPAT_PATCH_REVISION=2'
    echo "MODEL_OPTIMIZER_TREE_SHA256=$tree_hash"
} > "$DEST/MANIFEST.env"

echo "staged Model Optimizer: $DEST"
