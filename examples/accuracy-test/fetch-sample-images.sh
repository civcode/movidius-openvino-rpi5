#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fetch-sample-images.sh - download the known-content ImageNet sample set
# used by accuracy_test.py: 1000 JPEGs, exactly one per ImageNet-1k class,
# named '<synset>_<label>.JPEG' (ground truth in the file name).
#
# Source: https://github.com/EliSchwartz/imagenet-sample-images
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/images"
URL="${IMAGENET_SAMPLE_REPO:-https://github.com/EliSchwartz/imagenet-sample-images.git}"

if [[ -d "${DEST}/.git" ]]; then
    echo "already cloned - updating"
    git -C "${DEST}" pull --ff-only
else
    rm -rf "${DEST}"
    git clone --depth 1 "${URL}" "${DEST}"
fi

count="$(find "${DEST}" -maxdepth 1 -type f \( -name '*.JPEG' -o -name '*.jpg' -o -name '*.png' \) | wc -l)"
echo "dataset ready: ${count} images in ${DEST}"
if (( count < 1000 )); then
    echo "warning: expected 1000 images, found ${count}" >&2
fi
