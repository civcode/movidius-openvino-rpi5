#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# infer-server.sh - start the mobilenet_server MYRIAD backend for the webcam
# example (webcam_mobilenet.py).
#
#   examples/webcam/infer-server.sh [backend] [ir] [device]
#     backend : host | docker | auto   (default: auto)
#               host  - native host binaries from work/host-runtime (no Docker)
#               docker- inside the runtime image built by ./build.sh
#               auto  - host if the runtime was pulled, otherwise docker
#     ir      : fp16 | fp32            (default: fp16, override with IR=...)
#     device  : MYRIAD                 (default: MYRIAD)
#
# The server inherits this script's stdin/stdout (the binary tensor protocol
# of mobilenet_server) and prints startup diagnostics on stderr.  Target
# selection follows scripts/platform.sh (OV_PLATFORM / TARGET or host CPU).
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/platform.sh
source "${ROOT}/scripts/platform.sh"
platform_load "$(platform_default_request)"

BACKEND="${1:-auto}"
IR="${2:-${IR:-fp16}}"
DEVICE="${3:-MYRIAD}"
case "${IR}" in
    fp16|fp32) ;;
    *) echo "ir must be fp16 or fp32 (got '${IR}')" >&2; exit 2 ;;
esac

RT="${ROOT}/work/host-runtime"
OV="${RT}/openvino"
SERVER="${OV}/bin/mobilenet_server"
LIBDIR="${OV}/inference_engine/${TARGET_LIB_DIR}"

host_runtime_ready() {
    [[ -x "${SERVER}" ]]
}

host_backend() {
    local host_cpu accepted=0 pair
    # armv7 binaries run on an aarch64 host in compat mode (CONFIG_COMPAT=y).
    for pair in "armv7:armv7l" "armv7:aarch64" "arm64:aarch64" "amd64:x86_64"; do
        [[ "${pair%%:*}" == "${TARGET}" && "${pair#*:}" == "$(uname -m)" ]] && accepted=1
    done
    if (( ! accepted )); then
        echo "target '${TARGET}' cannot run on host CPU '$(uname -m)' - use backend 'docker' or" >&2
        echo "rebuild/re-pull with the matching platform" >&2
        exit 1
    fi
    if ! host_runtime_ready; then
        # Fallback: we are already inside the runtime image (this example
        # running via docker run on the built image, camera and stick passed
        # through as devices; /opt/openvino is the runtime itself).
        local ov_root="${OV_ROOT:-/opt/openvino}"
        if [[ -x "${ov_root}/bin/mobilenet_server" \
              && -d "${ov_root}/inference_engine/${TARGET_LIB_DIR}" ]]; then
            local libdir="${ov_root}/inference_engine/${TARGET_LIB_DIR}"
            echo "infer-server: backend=host target=${TARGET} in-image ${ov_root}" >&2
            LD_LIBRARY_PATH="${libdir}:${ov_root}/ngraph/lib" \
                exec "${ov_root}/bin/mobilenet_server" \
                --model "${ROOT}/vendor/models/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.xml" \
                --weights "${ROOT}/vendor/models/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.bin" \
                --device "${DEVICE}"
        fi
        echo "host runtime missing ${SERVER} - run ./scripts/pull-runtime.sh --platform ${TARGET}" >&2
        exit 1
    fi

    # mvnc's global lock: warn about foreign ownership in world-writable /tmp
    # (see logs/HOST-RUN.md).
    local mutex="${MVNC_MUTEX:-/tmp/mvnc.mutex}"
    if [[ -e "${mutex}" ]]; then
        local owner
        owner="$(stat -c '%u %a' "${mutex}")"
        if [[ "${owner%% *}" != "$(id -u)" ]]; then
            echo "note: ${mutex} is owned by '${owner}' - mvnc may fail to open it;" >&2
            echo "      fix with:  sudo rm ${mutex}" >&2
        fi
    fi

    if [[ "${TARGET}" == armv7 ]]; then
        # armhf glibc resolves SONAMEs only from directories it can search and
        # the PT_INTERP /lib/ld-linux-armhf.so.3 does not exist on an aarch64
        # host, so the armhf binaries run through the loader (same trick as
        # scripts/host-run.sh).
        local LD="${RT}/sysroot/lib/ld-linux-armhf.so.3"
        [[ -x "${LD}" ]] || { echo "armhf sysroot missing - re-run ./scripts/pull-runtime.sh --platform armv7" >&2; exit 1; }
        local LP="${RT}/sysroot/lib/arm-linux-gnueabihf:${RT}/sysroot/usr/lib/arm-linux-gnueabihf:${LIBDIR}:${OV}/ngraph/lib"
        echo "infer-server: backend=host target=${TARGET} loader=${LD}" >&2
        exec "${LD}" --library-path "${LP}" "${SERVER}" \
            --model "${ROOT}/vendor/models/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.xml" \
            --weights "${ROOT}/vendor/models/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.bin" \
            --device "${DEVICE}"
    fi

    local LP="${LIBDIR}:${OV}/ngraph/lib"
    echo "infer-server: backend=host target=${TARGET} native" >&2
    LD_LIBRARY_PATH="${LP}" exec "${SERVER}" \
        --model "${ROOT}/vendor/models/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.xml" \
        --weights "${ROOT}/vendor/models/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.bin" \
        --device "${DEVICE}"
}

docker_backend() {
    IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"
    docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
        echo "image ${IMAGE} not built yet - run ./build.sh --platform ${TARGET}" >&2; exit 1; }
    echo "infer-server: backend=docker target=${TARGET} image=${IMAGE}" >&2
    exec docker run --rm -i \
        --platform "${DOCKER_PLATFORM}" \
        --name "ov203-webcam-$$" \
        --network=host \
        -v /dev:/dev \
        --device-cgroup-rule='c 189:* rwm' \
        -v "${ROOT}/vendor/models:/models:ro" \
        -e OV_QUIET=1 \
        "${IMAGE}" \
        /opt/openvino/bin/mobilenet_server \
            --model "/models/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.xml" \
            --weights "/models/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.bin" \
            --device "${DEVICE}"
}

case "${BACKEND}" in
    host)   host_backend ;;
    docker) docker_backend ;;
    auto)
        if host_runtime_ready; then
            host_backend
        else
            docker_backend
        fi
        ;;
    *)
        echo "backend must be host, docker or auto (got '${BACKEND}')" >&2; exit 2 ;;
esac
