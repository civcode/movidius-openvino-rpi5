#!/usr/bin/env bash
# Shared setup for active, platform-aware low-level diagnostics.
DIAG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../../scripts/platform.sh
source "${DIAG_ROOT}/scripts/platform.sh"

diag_init() {
    platform_load "${1:-auto}"
    DIAG_IMAGE="ov203-myriad-diagnostics-${TARGET}:latest"
    MV="${DIAG_ROOT}/vendor/openvino-2020.3.2/inference-engine/thirdparty/movidius"
    OUT="${DIAG_ROOT}/work/diagnostics/${TARGET}"
    mkdir -p "${OUT}"
    if [[ ! -d "${MV}" ]]; then
        echo "OpenVINO source not present; run ./scripts/clone-openvino.sh" >&2
        return 1
    fi
    if ! docker image inspect "${DIAG_IMAGE}" >/dev/null 2>&1; then
        docker build --platform "${DOCKER_PLATFORM}" --provenance=false \
            --build-arg "DOCKER_PLATFORM=${DOCKER_PLATFORM}" \
            --build-arg "BASE_IMAGE=${DEFAULT_BASE_IMAGE}" \
            -t "${DIAG_IMAGE}" -f "${DIAG_ROOT}/docs/diagnostics/diagnostic.Dockerfile" \
            "${DIAG_ROOT}/docs/diagnostics"
    fi
}

diag_reset_stick() {
    sudo python3 "${DIAG_ROOT}/scripts/reset-stick.py" >/dev/null 2>&1 || true
}

diag_host_usb() {
    if command -v lsusb >/dev/null 2>&1; then
        lsusb | grep -i -E '03e7|myriad' || echo 'no Movidius device visible'
    else
        echo 'lsusb not installed'
    fi
}
