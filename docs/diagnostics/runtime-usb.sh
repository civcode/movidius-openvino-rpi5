#!/usr/bin/env bash
# Platform-aware USB/MYRIAD diagnostic using the self-built runtime image.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/platform.sh"
TARGET_REQUEST="$(platform_default_request)"
IMAGE_OVERRIDE="${IMAGE:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) TARGET_REQUEST="$2"; shift 2 ;;
        --image) IMAGE_OVERRIDE="$2"; shift 2 ;;
        -h|--help) echo "usage: $0 [--platform armv7|arm64|amd64] [--image TAG]"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
platform_load "${TARGET_REQUEST}"
IMAGE="${IMAGE_OVERRIDE:-${DEFAULT_IMAGE}}"

echo "target=${TARGET} docker_platform=${DOCKER_PLATFORM} image=${IMAGE}"
if command -v lsusb >/dev/null 2>&1; then
    echo 'host USB before:'
    lsusb | grep -i -E '03e7|myriad' || echo '  no Movidius device visible'
fi

docker run --rm --platform "${DOCKER_PLATFORM}" \
    --network=host -v /dev:/dev --device-cgroup-rule='c 189:* rwm' \
    --entrypoint bash "${IMAGE}" -lc '
set -e
IE_LIB="$(dirname "$(find /opt/openvino/inference_engine/lib -mindepth 2 -maxdepth 2 -name libmyriadPlugin.so -print -quit)")"
export LD_LIBRARY_PATH="$IE_LIB:/opt/openvino/ngraph/lib"
echo "runtime manifest:"; cat /opt/openvino/runtime-manifest.env
echo "IE_LIB=$IE_LIB"
echo "firmware:"; ls -l "$IE_LIB"/*.mvcmd
echo "USB nodes:"; ls -l /dev/bus/usb/*/* 2>/dev/null | tail -20 || true
echo "MYRIAD enumeration:"
/opt/openvino/bin/hello_myriad --list-only
'

if command -v lsusb >/dev/null 2>&1; then
    echo 'host USB after:'
    lsusb | grep -i -E '03e7|myriad' || echo '  no Movidius device visible'
fi
