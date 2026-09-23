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
# -h/--help is accepted at any position
for _arg in "$@"; do
    if [[ "${_arg}" == "-h" || "${_arg}" == "--help" ]]; then
        usage
        exit 0
    fi
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/images"
URL="${IMAGENET_SAMPLE_REPO:-https://github.com/EliSchwartz/imagenet-sample-images.git}"

if [[ -d "${DEST}/.git" ]]; then
    echo "already cloned - updating"
    git -C "${DEST}" pull --ff-only
elif [[ -e "${DEST}" && -n "$(ls -A "${DEST}" 2>/dev/null)" ]]; then
    # a non-empty directory that is not our clone: never delete user data
    echo "error: ${DEST} exists, is non-empty and is not a git clone;" >&2
    echo "       refusing to overwrite it - move or rename it first" >&2
    exit 1
else
    rm -rf "${DEST}"
    git clone --depth 1 "${URL}" "${DEST}"
fi

count="$(find "${DEST}" -maxdepth 1 -type f \( -name '*.JPEG' -o -name '*.jpg' -o -name '*.png' \) | wc -l)"
echo "dataset ready: ${count} images in ${DEST}"
if (( count < 1000 )); then
    echo "warning: expected 1000 images, found ${count}" >&2
fi
