#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fetch-runtime.sh - download the official OpenVINO 2020.3.2 raspbian runtime
#
# Two reasons this is part of the project:
#   1. It is the authoritative source of the MA2450 firmware blob
#      (usb-ma2450.mvcmd) that this build needs.  OpenVINO 2020.3's own
#      download server (download.01.org) is retired, so the firmware has to be
#      supplied from the release package instead - see scripts/prepare-deps.sh.
#   2. The armv7l libraries inside it (libmyriadPlugin.so etc.) are the exact
#      upstream reference for what our build should produce; they are kept in
#      vendor/reference-runtime/ for comparison only and are NOT used by the
#      container image.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="l_openvino_toolkit_runtime_raspbian_p_2020.3.355"
URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/2020.3.2/${PKG}.tgz"
TARBALL="${ROOT}/vendor/${PKG}-runtime-raspbian.tgz"
UNPACK="${ROOT}/vendor/reference-runtime"

mkdir -p "${ROOT}/vendor"
if [[ ! -f "${TARBALL}" ]]; then
    echo "downloading ${URL}"
    curl -fL --retry 3 -o "${TARBALL}" "${URL}"
fi
echo "tarball: ${TARBALL} ($(du -h "${TARBALL}" | cut -f1))"

if [[ ! -d "${UNPACK}/${PKG}" ]]; then
    mkdir -p "${UNPACK}"
    tar -xzf "${TARBALL}" -C "${UNPACK}"
fi

IE_LIBS="${UNPACK}/${PKG}/deployment_tools/inference_engine/lib/armv7l"
mkdir -p "${ROOT}/vendor/firmware"
for f in usb-ma2450 usb-ma2x8x pcie-ma248x; do
    cp -f "${IE_LIBS}/${f}.mvcmd" "${ROOT}/vendor/firmware/${f}.mvcmd"
done
echo "firmware copied to ${ROOT}/vendor/firmware:"
md5sum "${ROOT}"/vendor/firmware/*.mvcmd
