#!/usr/bin/env bash
# Extract a built runtime for host-native execution.
# armv7 also extracts the ARMHF loader/sysroot; arm64 and amd64 use native host execution.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=platform.sh
source "${ROOT}/scripts/platform.sh"

TARGET_REQUEST="$(platform_default_request)"
IMAGE_OVERRIDE="${IMAGE:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) TARGET_REQUEST="$2"; shift 2 ;;
        --image) IMAGE_OVERRIDE="$2"; shift 2 ;;
        --print-platform) platform_load "${TARGET_REQUEST}"; IMAGE="${IMAGE_OVERRIDE:-${DEFAULT_IMAGE}}"; platform_print; exit 0 ;;
        -h|--help) echo "usage: $0 [--platform armv7|arm64|amd64] [--image TAG]"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
platform_load "${TARGET_REQUEST}"
IMAGE="${IMAGE_OVERRIDE:-${DEFAULT_IMAGE}}"
RT="${ROOT}/work/host-runtime/${TARGET}"

rm -rf "${RT}"
mkdir -p "${RT}"

if [[ "${TARGET}" == armv7 ]]; then
    mkdir -p "${RT}/sysroot"
    docker run --rm --platform "${DOCKER_PLATFORM}" -v "${RT}:/host" "${IMAGE}" bash -c '
set -e
files="
/lib/ld-linux-armhf.so.3
/lib/arm-linux-gnueabihf/libc.so.6
/lib/arm-linux-gnueabihf/libm.so.6
/lib/arm-linux-gnueabihf/libdl.so.2
/lib/arm-linux-gnueabihf/libpthread.so.0
/lib/arm-linux-gnueabihf/libgcc_s.so.1
/lib/arm-linux-gnueabihf/librt.so.1
/usr/lib/arm-linux-gnueabihf/libstdc++.so.6
/usr/lib/arm-linux-gnueabihf/libusb-1.0.so.0
/usr/lib/arm-linux-gnueabihf/libudev.so.1
"
tar chzf /host/sysroot.tgz -C / $files
' 2>/dev/null
    tar xzf "${RT}/sysroot.tgz" -C "${RT}/sysroot"
    rm -f "${RT}/sysroot.tgz"
fi

cid="$(docker create --platform "${DOCKER_PLATFORM}" "${IMAGE}" true)"
trap 'docker rm -f "${cid}" >/dev/null 2>&1 || true' EXIT
docker cp "${cid}:/opt/openvino" "${RT}/openvino" >/dev/null
docker cp "${cid}:/opt/openvino-demo" "${RT}/openvino-demo" >/dev/null
docker rm "${cid}" >/dev/null
trap - EXIT
rm -rf "${RT}/openvino-demo/model/__pycache__" 2>/dev/null || true

cat > "${RT}/host-runtime.env" <<EOF_MANIFEST
TARGET=${TARGET}
DOCKER_PLATFORM=${DOCKER_PLATFORM}
IMAGE=${IMAGE}
HOST_RUNTIME_KIND=${HOST_RUNTIME_KIND}
EOF_MANIFEST

IE_LIB="$(platform_find_ie_libdir "${RT}/openvino/inference_engine")"
if command -v readelf >/dev/null 2>&1; then
    readelf -h "${RT}/openvino/bin/hello_myriad" > "${RT}/hello_myriad.elf.txt"
    grep -q "Class:.*${EXPECTED_ELF_CLASS}" "${RT}/hello_myriad.elf.txt" || {
        echo "extracted hello_myriad has the wrong ELF class for ${TARGET}" >&2; exit 1; }
    grep -Eq "Machine:.*(${EXPECTED_ELF_MACHINE_REGEX})" "${RT}/hello_myriad.elf.txt" || {
        echo "extracted hello_myriad has the wrong machine type for ${TARGET}" >&2; exit 1; }
else
    echo "warning: readelf not installed on host; extracted ELF architecture check skipped" >&2
fi
printf 'IE_LIB=%s\n' "${IE_LIB}" >> "${RT}/host-runtime.env"

printf 'host runtime ready: %s\n' "${RT}"
printf '  target   %s (%s)\n' "${TARGET}" "${DOCKER_PLATFORM}"
printf '  openvino %s\n' "$(du -sh "${RT}/openvino" | cut -f1)"
printf '  ie libs  %s\n' "${IE_LIB}"
if [[ "${TARGET}" == armv7 ]]; then
    printf '  sysroot  %s\n' "$(du -sh "${RT}/sysroot" | cut -f1)"
fi
echo "run it with: ./scripts/host-run.sh --platform ${TARGET} list"
