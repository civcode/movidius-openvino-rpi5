#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# infer-ssd-server.sh - start the ssd_detect stream backend for the webcam
# detection example (ssd_stream.py).
#
#   examples/ssd-detect/infer-ssd-server.sh [backend] [device] [min-conf]
#     backend : host | docker | auto   (default: auto)
#               host  - native host binaries from work/host-runtime/<target> (no Docker)
#               docker- inside the runtime image built by ./build.sh
#               auto  - host if the runtime was pulled, otherwise docker
#     device  : MYRIAD | CPU           (default: MYRIAD; CPU on amd64 selects the
#                                       FP32 IRs; CPU on arm64 = Python full-TF server)
#     min-conf: 0.5                    (default: 0.5)
#
# The server inherits this script's stdin/stdout (the frame protocol of
# ssd_detect --stdin) and prints startup diagnostics on stderr.  Target
# selection follows scripts/platform.sh (OV_PLATFORM / TARGET or host CPU).
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/platform.sh
source "${ROOT}/scripts/platform.sh"
platform_load "$(platform_default_request)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [backend] [device] [min-conf]

Starts ssd_detect --stdin, the MYRIAD inference server for the SSDLite
webcam detection example.  The server inherits this script's
stdin/stdout (the binary frame protocol of ssd_detect --stdin) and prints
startup diagnostics on stderr.  Normally you do not run it directly -
ssd_stream.py starts it for you.

Arguments (all optional, positional):
  backend    host | docker | auto      default: auto
               host   - native binaries from work/host-runtime/<target> (no Docker)
               docker - inside the runtime image built by ./build.sh
               auto   - host if pulled, else docker, else in-image fallback
  device     MYRIAD | CPU              default: MYRIAD.  CPU per target:
                                            amd64 = OV server + FP32 IR;
                                            arm64 = Python full-TensorFlow server
                                            (host or docker); armv7 = error.  See
                                            docs/CPU-BACKENDS.md.
  min-conf   float                     default: 0.5
               keep detections with score >= min-conf

Environment overrides:
  IMAGE=<name>              Docker image (default: ${DEFAULT_IMAGE})
  OV_PLATFORM=<p>           target platform (see scripts/platform.sh)
  MVNC_MUTEX=<path>         mvnc global lock file (default /tmp/mvnc.mutex)

Protocol (stdin -> stdout):
  in : repeated frames, each = uint32 width + uint32 height (little endian)
       + width*height*3 RGB bytes
  out: per frame:  "FRAME <w> <h> <infer_ms>"
               "DET <label> <score> <x1> <y1> <x2> <y2>"   (0..N lines)
               "END"

Examples:
  # run the client (it starts the server itself):
  python3 examples/ssd-detect/ssd_stream.py --headless

  # drive the server directly with one binary frame:
  examples/ssd-detect/infer-ssd-server.sh docker MYRIAD 0.5 < /tmp/frame.bin
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
DEVICE="${2:-MYRIAD}"
MINCONF="${3:-0.5}"

# The CPU plugin of the amd64 runtime (OpenVINO 2020.3) does not accept FP16
# input tensors, so --device CPU runs against the FP32 IR instead.
if [[ "${DEVICE}" == "CPU" ]]; then
	MODEL_IR_DIR="openvino_fp32"
else
	MODEL_IR_DIR="openvino"
fi
MODEL_XML="${ROOT}/vendor/models/ssdlite_mobilenet_v2/${MODEL_IR_DIR}/ssdlite_mobilenet_v2.xml"
MODEL_BIN="${ROOT}/vendor/models/ssdlite_mobilenet_v2/${MODEL_IR_DIR}/ssdlite_mobilenet_v2.bin"
LABELS="${ROOT}/vendor/models/labels/coco.txt"

# --device CPU: per-target CPU backend (docs/CPU-BACKENDS.md).
#   amd64: the OV C++ server below with the FP32 IR (unchanged path).
#   arm64: the Python full-TensorFlow server (host, or in-image copy under docker).
#   armv7: not available - no 32-bit inference wheels (MYRIAD only).
cpu_python_server() {
    local pb="${ROOT}/vendor/models/ssdlite_mobilenet_v2/source/frozen_inference_graph.pb"
    if [[ ! -f "${pb}" ]]; then
        echo "frozen graph missing: ${pb}" >&2
        echo "run ./scripts/prepare-ssdlite.sh first" >&2
        exit 1
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 not found - install Python 3 to use --device CPU on arm64" >&2
        exit 1
    fi
    if ! python3 -c 'import tensorflow' >/dev/null 2>&1; then
        echo "tensorflow is not installed - run:  pip install tensorflow" >&2
        exit 1
    fi
    echo "infer-ssd-server: backend=host target=${TARGET} python-tensorflow" >&2
    exec python3 "${ROOT}/examples/ssd-detect/ssd_cpu_server.py" \
        --model "${pb}" --labels "${LABELS}" --min-conf "${MINCONF}"
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
SERVER="${OV}/bin/ssd_detect"
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
        if [[ -x "${ov_root}/bin/ssd_detect" \
              && -d "${ov_root}/inference_engine/${TARGET_LIB_DIR}" ]]; then
            local libdir="${ov_root}/inference_engine/${TARGET_LIB_DIR}"
            echo "infer-ssd-server: backend=host target=${TARGET} in-image ${ov_root}" >&2
            LD_LIBRARY_PATH="${libdir}:${ov_root}/ngraph/lib" \
                exec "${ov_root}/bin/ssd_detect" \
                    --model "${MODEL_XML}" \
                    --weights "${MODEL_BIN}" \
                    --labels "${LABELS}" \
                    --device "${DEVICE}" --min-conf "${MINCONF}" --stdin
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
        echo "infer-ssd-server: backend=host target=${TARGET} loader=${LD}" >&2
        exec "${LD}" --library-path "${LP}" "${SERVER}" \
            --model "${MODEL_XML}" --weights "${MODEL_BIN}" \
            --labels "${LABELS}" \
            --device "${DEVICE}" --min-conf "${MINCONF}" --stdin
    fi

    local LP="${LIBDIR}:${OV}/ngraph/lib"
    echo "infer-ssd-server: backend=host target=${TARGET} native" >&2
    LD_LIBRARY_PATH="${LP}" exec "${SERVER}" \
        --model "${MODEL_XML}" --weights "${MODEL_BIN}" \
        --labels "${LABELS}" \
        --device "${DEVICE}" --min-conf "${MINCONF}" --stdin
}

docker_backend() {
    IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"
    docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
        echo "image ${IMAGE} not built yet - run ./build.sh --platform ${TARGET}" >&2; exit 1; }
    echo "infer-ssd-server: backend=docker target=${TARGET} image=${IMAGE}" >&2
    if [[ "${DEVICE}" == CPU && "${TARGET}" == arm64 ]]; then
        # Python full-TensorFlow server from the runtime image (the image
        # must have been built with the cpu-servers step, see Dockerfile).
        exec docker run --rm -i \
            --platform "${DOCKER_PLATFORM}" \
            --name "ov203-ssd-cpu-$$" \
            -e OV_QUIET=1 \
            -v "${ROOT}/vendor/models:/models:ro" \
            "${IMAGE}" \
            python3 /opt/openvino-demo/cpu-servers/ssd_cpu_server.py \
                --model /models/ssdlite_mobilenet_v2/source/frozen_inference_graph.pb \
                --labels /models/labels/coco.txt \
                --min-conf "${MINCONF}"
    fi
    exec docker run --rm -i \
        --platform "${DOCKER_PLATFORM}" \
        --name "ov203-ssd-$$" \
        --network=host \
        -v /dev:/dev \
        --device-cgroup-rule='c 189:* rwm' \
        -v "${ROOT}/vendor/models:/models:ro" \
        -e OV_QUIET=1 \
        "${IMAGE}" \
        /opt/openvino/bin/ssd_detect \
            --model "/models/ssdlite_mobilenet_v2/${MODEL_IR_DIR}/ssdlite_mobilenet_v2.xml" \
            --weights "/models/ssdlite_mobilenet_v2/${MODEL_IR_DIR}/ssdlite_mobilenet_v2.bin" \
            --labels "/models/labels/coco.txt" \
            --device "${DEVICE}" --min-conf "${MINCONF}" --stdin
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
            # itself; host_backend falls back to ${OV_ROOT}/bin/ssd_detect
            host_backend
        fi
        ;;
    *)
        echo "backend must be host, docker or auto (got '${BACKEND}')" >&2; exit 2 ;;
esac
