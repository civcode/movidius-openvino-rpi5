#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# clone-openvino.sh - fetch the pinned OpenVINO source tree used by this build
#
# Pinned revision: tag 2020.3.2 (commit 0b3773b7405d955d48642667ac5113289b9baab2)
# Submodules     : ngraph                  @ 1797d7fb712d2ffa6048ce5f5b3b560b84ab5ae4
#                  inference-engine/thirdparty/ade @ cbe2db61a659c2cc304c3837406f95c39dfa938e
#
# 2020.3 LTS is the last OpenVINO series that ships the MYRIAD plugin, i.e. the
# last one that can drive the Intel Movidius Neural Compute Stick, so the
# version is not negotiable here.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="2020.3.2"
EXPECTED_COMMIT="0b3773b7405d955d48642667ac5113289b9baab2"
DEST="${ROOT}/vendor/openvino-${TAG}"
URL="https://github.com/openvinotoolkit/openvino.git"

if [[ -d "${DEST}/.git" ]]; then
    echo "source tree already present: ${DEST}"
else
    echo "cloning ${URL} (tag ${TAG}, shallow, with submodules) -> ${DEST}"
    git clone --recursive --depth 1 --branch "${TAG}" "${URL}" "${DEST}"
fi

actual="$(git -C "${DEST}" rev-parse HEAD)"
echo "openvino commit : ${actual}"
if [[ "${actual}" != "${EXPECTED_COMMIT}" ]]; then
    echo "WARNING: commit differs from the pinned one (${EXPECTED_COMMIT})" >&2
fi
git -C "${DEST}" submodule status
echo "tree size       : $(du -sh "${DEST}" | cut -f1)"
