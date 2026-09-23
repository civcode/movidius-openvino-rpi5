#!/usr/bin/env bash
# Download Intel's OpenVINO 2020.3.2 Raspbian runtime as a pinned source for the
# MA2450 device firmware. The host-side ARM libraries are reference material only;
# all armv7, arm64 and amd64 project images build libmyriadPlugin.so from source.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="l_openvino_toolkit_runtime_raspbian_p_2020.3.355"
URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/2020.3.2/${PKG}.tgz"
TARBALL="${ROOT}/vendor/${PKG}-runtime-raspbian.tgz"
UNPACK="${ROOT}/vendor/reference-runtime"
FW_DIR="${ROOT}/vendor/firmware"

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
mkdir -p "${FW_DIR}"
for f in usb-ma2450 usb-ma2x8x pcie-ma248x; do
    test -f "${IE_LIBS}/${f}.mvcmd"
    cp -f "${IE_LIBS}/${f}.mvcmd" "${FW_DIR}/${f}.mvcmd"
done

# The package revision used by this project is known to contain these sizes.
# This catches an obviously wrong/truncated payload even on first acquisition.
[[ "$(stat -c%s "${FW_DIR}/usb-ma2450.mvcmd")" == 1795780 ]] || { echo "unexpected usb-ma2450.mvcmd size" >&2; exit 1; }
[[ "$(stat -c%s "${FW_DIR}/usb-ma2x8x.mvcmd")" == 2040216 ]] || { echo "unexpected usb-ma2x8x.mvcmd size" >&2; exit 1; }
[[ "$(stat -c%s "${FW_DIR}/pcie-ma248x.mvcmd")" == 1800408 ]] || { echo "unexpected pcie-ma248x.mvcmd size" >&2; exit 1; }

(cd "${FW_DIR}" && sha256sum pcie-ma248x.mvcmd usb-ma2450.mvcmd usb-ma2x8x.mvcmd > SHA256SUMS && sha256sum -c SHA256SUMS)
echo "firmware copied to ${FW_DIR}; it is device-side and shared by both host targets:"
cat "${FW_DIR}/SHA256SUMS"
