#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fetch-sample-images.sh - download the known-content ImageNet sample set
# used by accuracy_test.py: 1000 JPEGs, exactly one per ImageNet-1k class,
# named '<synset>_<label>.JPEG' (ground truth in the file name).
#
# Source: https://github.com/EliSchwartz/imagenet-sample-images
# ---------------------------------------------------------------------------
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: fetch-sample-images.sh

Downloads the known-content ImageNet sample set used by
accuracy_test.py: 1000 JPEGs, exactly one per ImageNet-1k class, named
'<synset>_<label>.JPEG' (ground truth is the file name).
Source: https://github.com/EliSchwartz/imagenet-sample-images

Arguments: none (--help shows this text)

Environment overrides:
  IMAGENET_SAMPLE_REPO=<url>   git repository to clone
                               (default: the EliSchwartz/imagenet-sample-images repo)

Examples:
  examples/accuracy-test/fetch-sample-images.sh
  python3 examples/accuracy-test/accuracy_test.py --fail-under 80
EOF
}
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

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
