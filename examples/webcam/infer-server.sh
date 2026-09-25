#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# infer-server.sh - start the mobilenet_server MYRIAD backend for the webcam
# example (webcam_mobilenet.py).
#
#   examples/webcam/infer-server.sh [backend] [ir] [device]
#     backend : host | docker | auto   (default: auto)
#               host  - native host binaries from work/host-runtime/<target> (no Docker)
#               docker- inside the runtime image built by ./build.sh
#               auto  - host if the runtime was pulled, else docker, else
#                       in-image fallback (running inside the runtime image)
#     ir      : fp16 | fp32            (default: fp16, override with IR=...)
#     device  : MYRIAD | CPU           (default: MYRIAD)
#               MYRIAD - OV server (all targets)
#               CPU    - amd64: OV server + FP32 IR;  arm64: Python ONNX
#                        Runtime server (host, or in-image copy under docker); armv7: error
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

usage() {
    cat <<EOF
Usage: $(basename "$0") [backend] [ir] [device]

Starts mobilenet_server, the MYRIAD inference backend for the webcam
example.  The server inherits this script's stdin/stdout (the binary
tensor protocol of mobilenet_server) and prints startup diagnostics on
stderr.  Normally you do not run it directly - webcam_mobilenet.py
starts it for you.

Arguments (all optional, positional):
  backend    host | docker | auto      default: auto
               host   - native binaries from work/host-runtime/<target> (no Docker)
               docker - inside the runtime image built by ./build.sh
               auto   - host if pulled, else docker, else in-image fallback
  ir         fp16 | fp32               default: fp16
  device     MYRIAD | CPU              default: MYRIAD
               CPU on arm64 uses the Python ONNX Runtime server
               (examples/webcam/mobilenet_cpu_server.py, host or in-image copy under docker); see
               docs/CPU-BACKENDS.md.

Environment overrides:
  IR=<fp16|fp32>            inference precision
  IMAGE=<name>              Docker image (default: ${DEFAULT_IMAGE})
  OV_PLATFORM=<p>           target platform (see scripts/platform.sh)
  MVNC_MUTEX=<path>         mvnc global lock file (default /tmp/mvnc.mutex)

Examples:
  # run the client (it starts the server itself):
  python3 examples/webcam/webcam_mobilenet.py

  # drive the server directly with the reference fp32 input tensor:
  examples/webcam/infer-server.sh docker fp16 MYRIAD \\
      < vendor/models/test_data/input_0.f32

  # CPU on arm64 (Python ONNX Runtime server):
  examples/webcam/infer-server.sh host fp32 CPU \\
      < vendor/models/test_data/input_0.f32
EOF
}
# -h/--help is accepted at any position
for _arg in "$@"; do
    if [[ "${_arg}" == "-h" || "${_arg}" == "--help" ]]; then
        usage
        exit 0
    fi
done

BACKEND="${1:-auto}"
IR="${2:-${IR:-fp16}}"
DEVICE="${3:-MYRIAD}"
case "${IR}" in
    fp16|fp32) ;;
    *) echo "ir must be fp16 or fp32 (got '${IR}')" >&2; exit 2 ;;
esac
case "${DEVICE}" in
    MYRIAD|CPU) ;;
    *) echo "device must be MYRIAD or CPU (got '${DEVICE}')" >&2; exit 2 ;;
esac

# --device CPU: per-target CPU backend (docs/CPU-BACKENDS.md).
#   amd64: the OV C++ server below with the FP32 IR (unchanged path).
#   arm64: the Python ONNX Runtime server (host, or in-image copy under docker).
#   armv7: not available - no 32-bit inference wheels (MYRIAD only).
# Resolve a python3 interpreter that can import <module>: the system
# python3 first, then the project venv at work/venv-cpu (PEP 668 system
# pythons refuse plain pip installs, so the venv is the supported host
# path: python3 -m venv work/venv-cpu && work/venv-cpu/bin/pip install <pkg>)
cpu_python_interpreter() {
    local mod="$1" venv="${ROOT}/work/venv-cpu"
    if command -v python3 >/dev/null 2>&1 && python3 -c "import ${mod}" >/dev/null 2>&1; then
        echo "python3"
        return 0
    fi
    if [[ -x "${venv}/bin/python" ]] && "${venv}/bin/python" -c "import ${mod}" >/dev/null 2>&1; then
        echo "${venv}/bin/python"
        return 0
    fi
    return 1
}

cpu_python_server() {
    local onnx="${ROOT}/vendor/models/onnx/mobilenetv2-7.onnx"
    if [[ ! -f "${onnx}" ]]; then
        echo "ONNX model missing: ${onnx}" >&2
        echo "run ./scripts/prepare-mobilenet.sh first" >&2
        exit 1
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 not found - install Python 3 to use --device CPU on arm64" >&2
        exit 1
    fi
    local py
    py="$(cpu_python_interpreter onnxruntime)" || {
        echo "onnxruntime is not installed - run:  pip install onnxruntime  (or the venv: python3 -m venv ${ROOT}/work/venv-cpu && ${ROOT}/work/venv-cpu/bin/pip install onnxruntime)" >&2
        exit 1
    }
    echo "infer-server: backend=host target=${TARGET} python-onnxruntime" >&2
    exec "${py}" "${ROOT}/examples/webcam/mobilenet_cpu_server.py" --model "${onnx}"
}

if [[ "${DEVICE}" == CPU ]]; then
    case "${TARGET}" in
        arm64)
            # host: the local Python server; docker: the in-image copy
            # (handled in docker_backend)
            [[ "${BACKEND}" == docker ]] || cpu_python_server
            ;;
        amd64) ;;   # fall through to the OV server with the FP32 IR
        *)
            echo "--device CPU is not available on target '${TARGET}' (no 32-bit CPU" >&2
            echo "runtime for armv7) - use MYRIAD" >&2
            exit 1
            ;;
    esac
fi

RT="${ROOT}/work/host-runtime/${TARGET}"
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
    if [[ "${DEVICE}" == CPU && "${TARGET}" == arm64 ]]; then
        # Python ONNX Runtime server from the runtime image (the image must
        # have been built with the cpu-servers step, see Dockerfile).
        exec docker run --rm -i \
            --platform "${DOCKER_PLATFORM}" \
            --name "ov203-webcam-cpu-$$" \
            -e OV_QUIET=1 \
            -v "${ROOT}/vendor/models:/models:ro" \
            "${IMAGE}" \
            python3 /opt/openvino-demo/cpu-servers/mobilenet_cpu_server.py \
                --model "/models/onnx/mobilenetv2-7.onnx"
    fi
    exec docker run --rm -i \
        --platform "${DOCKER_PLATFORM}" \
        --name "ov203-webcam-$$" \
        --network=host \
        -v /dev:/dev \
        --device-cgroup-rule='c 189:* rwm' \
        --device-cgroup-rule='c 81:* rwm' \
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
        elif command -v docker >/dev/null 2>&1; then
            docker_backend
        else
            # no docker CLI - we are probably inside the runtime image
            # itself; host_backend falls back to ${OV_ROOT}/bin/mobilenet_server
            host_backend
        fi
        ;;
    *)
        echo "backend must be host, docker or auto (got '${BACKEND}')" >&2; exit 2 ;;
esac
