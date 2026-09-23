#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh - host-side wrapper that starts the OpenVINO 2020.3.2 container with
# the MA2450 stick attached.
#
#   ./run.sh                 # device enumeration + tiny-model inference
#   ./run.sh list            # plugin/device enumeration only
#   ./run.sh shell           # bash inside the runtime image
#   ./run.sh bench --iterations 20
#   ./run.sh custom --model /work/model.xml --weights /work/model.bin
#   ./run.sh mobilenet                     # MobileNet v2 on the stick, checked against
#                                          # the ONNX model zoo reference output
#   ./run.sh mobilenet --image /models/images/dog.ppm
#   IR=fp32 ./run.sh mobilenet             # use the FP32 weights instead of FP16
#   ./run.sh ssd --image /models/images/dog_ssd.ppm
#                                            # SSDLite detection (prepare-ssdlite.sh)
#   ./run.sh seg [--image /models/images/dog_ssd.ppm]
#                                            # DeepLabV3 segmentation (prepare-deeplabv3.sh)
#
# USB access (measured on this Pi 5 - see README "USB access" section):
#   --network=host        libusb, which OpenVINO's MYRIAD plugin uses through
#                         XLink/mvnc, caches its device list and only refreshes it
#                         from kernel uevents.  In a private netns the stick that
#                         re-enumerates during firmware boot (bus 3 / 03e7:2150 ->
#                         bus 4 / 03e7:f63b on the Pi 5) is never seen and
#                         ncDeviceOpen() times out.
#   -v /dev:/dev          live /dev/bus/usb nodes.  --device only publishes the
#                         nodes that exist when the container starts, so the
#                         post-boot device node would be missing.
#   --device-cgroup-rule  the new node is char major 189; it must be writable.
# No --privileged.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/platform.sh
source "${ROOT}/scripts/platform.sh"

# Consume a leading --platform/-p <target> before mode parsing; it selects the
# image but must not be forwarded into the container as binary arguments.
_req="$(platform_default_request)"
_reargs=()
_args=("$@")
for (( _i=0; _i<${#_args[@]}; _i++ )); do
    case "${_args[_i]}" in
        --platform|-p)
            (( _i < ${#_args[@]} - 1 )) || { echo "missing value for ${_args[_i]}" >&2; exit 2; }
            _req="${_args[_i+1]}"
            _i=$((_i + 1))
            ;;
        *) _reargs+=("${_args[_i]}") ;;
    esac
done
set -- "${_reargs[@]+"${_reargs[@]}"}"
platform_load "${_req}"
IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"
MODE="${1:-demo}"
case "${MODE}" in
    demo|--demo|list|demo-list|shell|bench|custom|mobilenet|ssd|seg)
        if [[ $# -gt 0 ]]; then shift; fi
        ;;
    *) MODE="demo" ;;   # extra hello_myriad flags with the default demo mode
esac
if [[ "${MODE}" == "list" || "${MODE}" == "demo-list" ]]; then
    MODE="demo"
    set -- --list-only
fi

DOCKER_ARGS=(
    --rm
    --platform "${DOCKER_PLATFORM}"
    --name "ov203-smoke-$$"
    --network=host
    -v /dev:/dev
    --device-cgroup-rule='c 189:* rwm'
    -e OV_ROOT=/opt/openvino
)

# The image's ENTRYPOINT is /opt/openvino-demo/run.sh, so everything here is
# passed as *arguments* to it (passing the script path again would make the
# entrypoint re-invoke itself).  Mode handling lives in container-entry.sh:
#   --demo        enumerate devices and run the bundled tiny model
#   <anything>    exec'd directly (used by bench/custom/shell)
case "${MODE}" in
    demo)
        ENTRY=("--demo" "$@")
        ;;
    shell)
        DOCKER_ARGS+=(--entrypoint bash)
        if [[ -t 0 ]]; then
            DOCKER_ARGS+=(-it)
            ENTRY=("bash")
        else
            ENTRY=("bash" "-c" "echo 'no TTY: this container has no daemon; use bench/custom/list modes or run a single command'")
        fi
        ;;
    bench)
        ENTRY=("/opt/openvino/bin/hello_myriad" "--device" "MYRIAD" "$@")
        ;;
    custom)
        DOCKER_ARGS+=(-v "${ROOT}/work:/work:ro")
        ENTRY=("/opt/openvino/bin/hello_myriad" "--device" "MYRIAD" "$@")
        ;;
    mobilenet)
        # the corpus prepared by ./scripts/prepare-mobilenet.sh stays on the host
        IR="${IR:-fp16}"
        MODEL_DIR="${ROOT}/vendor/models/mobilenet-v2-ov203/${IR}"
        if [[ ! -f "${MODEL_DIR}/mobilenet-v2-ov203.xml" ]]; then
            echo "no IR in ${MODEL_DIR} - run ./scripts/prepare-mobilenet.sh first" >&2
            exit 1
        fi
        DOCKER_ARGS+=(-v "${ROOT}/vendor/models:/models:ro")
        # with no arguments: verify against the model zoo's own reference output
        if [[ $# -eq 0 ]]; then
            set -- --tensor /models/test_data/input_0.f32 --reference /models/test_data/output_0.f32
        fi
        ENTRY=("/opt/openvino/bin/mobilenet_classify" "--device" "MYRIAD"
               "--model" "/models/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.xml"
               "--weights" "/models/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.bin"
               "--labels" "/models/labels/synset.txt" "$@")
        ;;
    ssd)
        # the corpus prepared by ./scripts/prepare-ssdlite.sh stays on the host
        MODEL_DIR="${ROOT}/vendor/models/ssdlite_mobilenet_v2"
        if [[ ! -f "${MODEL_DIR}/openvino/ssdlite_mobilenet_v2.xml" ]]; then
            echo "no IR in ${MODEL_DIR} - run ./scripts/prepare-ssdlite.sh first" >&2
            exit 1
        fi
        DOCKER_ARGS+=(-v "${ROOT}/vendor/models:/models:ro")
        ENTRY=("/opt/openvino/bin/ssd_detect" "--device" "MYRIAD"
               "--model" "/models/ssdlite_mobilenet_v2/openvino/ssdlite_mobilenet_v2.xml"
               "--weights" "/models/ssdlite_mobilenet_v2/openvino/ssdlite_mobilenet_v2.bin"
               "--labels" "/models/labels/coco.txt" "$@")
        ;;
    seg)
        # the corpus prepared by ./scripts/prepare-deeplabv3.sh stays on the host
        MODEL_DIR="${ROOT}/vendor/models/deeplabv3"
        if [[ ! -f "${MODEL_DIR}/openvino/deeplabv3.xml" ]]; then
            echo "no IR in ${MODEL_DIR} - run ./scripts/prepare-deeplabv3.sh first" >&2
            exit 1
        fi
        DOCKER_ARGS+=(-v "${ROOT}/vendor/models:/models:ro")
        if [[ $# -eq 0 ]]; then
            set -- --image /models/images/dog_ssd.ppm
        fi
        ENTRY=("/opt/openvino/bin/seg_detect" "--device" "MYRIAD"
               "--model" "/models/deeplabv3/openvino/deeplabv3.xml"
               "--weights" "/models/deeplabv3/openvino/deeplabv3.bin"
               "--labels" "/models/labels/pascal_voc.txt" "$@")
        ;;
esac

echo ">> docker run ${DOCKER_ARGS[*]} ${IMAGE} ${ENTRY[*]}"
exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${ENTRY[@]}"
