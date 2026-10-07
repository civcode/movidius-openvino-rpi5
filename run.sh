#!/usr/bin/env bash
# Run the MA2450 OpenVINO image on any supported Linux target.
#
#   ./run.sh                 # device enumeration + tiny-model inference
#   ./run.sh list            # plugin/device enumeration only
#   ./run.sh shell           # bash inside the runtime image
#   ./run.sh bench --iterations 20
#   ./run.sh custom --model /work/model.xml --weights /work/model.bin
#   ./run.sh custom-server --model /work/model.xml --weights /work/model.bin
#                                            # persistent binary tensor stream
#   ./run.sh mobilenet                     # MobileNet v2 on the stick, checked against
#                                          # the ONNX model zoo reference output
#   ./run.sh mobilenet --image /models/images/dog.ppm
#   IR=fp32 ./run.sh mobilenet             # use the FP32 weights instead of FP16
#   ./run.sh ssd --image /models/images/dog_ssd.ppm
#                                            # SSDLite detection (prepare-ssdlite.sh)
#   ./run.sh seg [--image /models/images/dog_ssd.ppm]
#                                            # DeepLabV3 segmentation (prepare-deeplabv3.sh)
#   ./run.sh speech-reference
#                                            # rm_cnn4a vendor ARK regression on amd64 CPU
#   ./run.sh speech-regress
#                                            # rm_cnn4a vendor ARK regression on MYRIAD
#
# Global options (before the mode):
#   --platform|-p armv7|arm64|amd64    select the target (default: auto-detect)
#   --image <tag>                      run a specific Docker image
#   --verbose|-v                       print the resolved platform configuration
#   --print-platform                   print it and exit
#   -h|--help                          show this help
#
# Global options must appear before the mode. USB access deliberately uses the
# same re-enumeration-safe strategy on all targets: host netns, live /dev and a
# cgroup rule for USB character devices. No --privileged is required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/platform.sh
source "${ROOT}/scripts/platform.sh"

TARGET_REQUEST="$(platform_default_request)"
IMAGE_OVERRIDE="${IMAGE:-}"
VERBOSE=0
PRINT_PLATFORM=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform|-p) [[ $# -ge 2 ]] || { echo "--platform needs armv7|arm64|amd64" >&2; exit 2; }; TARGET_REQUEST="$2"; shift 2 ;;
        --image) [[ $# -ge 2 ]] || { echo "--image needs a Docker image tag" >&2; exit 2; }; IMAGE_OVERRIDE="$2"; shift 2 ;;
        --verbose|-v) VERBOSE=1; shift ;;
        --print-platform) PRINT_PLATFORM=1; shift ;;
        -h|--help)
            awk '/^set -euo/{exit} {print}' "$0"
            exit 0
            ;;
        *) break ;;
    esac
done

platform_load "${TARGET_REQUEST}"
IMAGE="${IMAGE_OVERRIDE:-${DEFAULT_IMAGE}}"
export IMAGE
if (( PRINT_PLATFORM )); then platform_print; exit 0; fi

MODE="${1:-demo}"
case "${MODE}" in
    demo|--demo|list|demo-list|shell|bench|custom|custom-server|mobilenet|ssd|seg|speech-reference|speech-regress)
        [[ $# -gt 0 ]] && shift
        ;;
    *) MODE=demo ;;
esac
if [[ "${MODE}" == list || "${MODE}" == demo-list ]]; then
    MODE=demo
    set -- --list-only
fi

# Fail before USB/device diagnostics when the requested runtime image is not
# available locally. These project images are built locally; Docker Hub does not
# contain them. PLATFORM selects the host/runtime architecture, not the Movidius
# stick architecture.
if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to run ${MODE}" >&2
    exit 2
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "local Docker image not found: ${IMAGE}" >&2
    echo "selected runtime platform: ${TARGET} (${DOCKER_PLATFORM})" >&2
    echo "build it first with: ./build.sh --platform ${TARGET}" >&2
    echo "note: --platform selects the host/runtime architecture, not the Movidius device" >&2
    exit 2
fi

# Catch the common mistake of selecting a target that does not match a locally
# built image before USB debugging obscures the real problem.
image_arch="$(docker image inspect "${IMAGE}" --format '{{.Architecture}}' 2>/dev/null || true)"
expected_arch=amd64
[[ "${TARGET}" == armv7 ]] && expected_arch=arm
[[ "${TARGET}" == arm64 ]] && expected_arch=arm64
if [[ -n "${image_arch}" && "${image_arch}" != "${expected_arch}" ]]; then
    echo "image architecture mismatch: ${IMAGE} is ${image_arch}, selected target ${TARGET} expects ${expected_arch} (${DOCKER_PLATFORM})" >&2
    echo "rebuild with: ./build.sh --platform ${TARGET}" >&2
    exit 1
fi

DOCKER_ARGS=(
    --rm
    --platform "${DOCKER_PLATFORM}"
    --name "ov203-${MODE}-${TARGET}-$$"
    -e OV_ROOT=/opt/openvino
)
if [[ "${MODE}" != speech-reference ]]; then
    DOCKER_ARGS+=(
        --network=host
        -v /dev:/dev
        --device-cgroup-rule='c 189:* rwm'
    )
fi

case "${MODE}" in
    demo)
        ENTRY=(--demo "$@")
        ;;
    shell)
        DOCKER_ARGS+=(--entrypoint bash)
        if [[ -t 0 ]]; then
            DOCKER_ARGS+=(-it)
            ENTRY=(bash)
        else
            ENTRY=(bash -c "echo 'no TTY: use bench/custom/list modes or run an explicit command'")
        fi
        ;;
    bench)
        ENTRY=(/opt/openvino/bin/hello_myriad --device MYRIAD "$@")
        ;;
    custom)
        DOCKER_ARGS+=(-v "${ROOT}/work:/work")
        custom_device_args=(--device MYRIAD)
        for arg in "$@"; do
            if [[ "${arg}" == "--device" ]]; then
                custom_device_args=()
                break
            fi
        done
        ENTRY=(/opt/openvino/bin/hello_myriad "${custom_device_args[@]}" "$@")
        ;;
    custom-server)
        # Binary tensor protocol: container-entry.sh must not write its banner
        # to stdout or the client will consume text bytes as float32 logits.
        DOCKER_ARGS+=(-i -e OV_QUIET=1 -v "${ROOT}/work:/work")
        ENTRY=(/opt/openvino/bin/hello_myriad --device MYRIAD --stdin "$@")
        ;;
    speech-reference|speech-regress)
        MODEL_DIR="${ROOT}/vendor/models/rm_cnn4a_smbr"
        XML="${MODEL_DIR}/openvino/fp16/rm_cnn4a_fp16.xml"
        FEATURES="${MODEL_DIR}/source/feat1_10.ark"
        SCORES="${MODEL_DIR}/source/score1_10.ark"
        if [[ ! -f "${XML}" || ! -f "${FEATURES}" || ! -f "${SCORES}" ]]; then
            echo "rm_cnn4a fixture is not prepared - run ./scripts/prepare-rm-cnn4a.sh first" >&2
            exit 1
        fi
        DEVICE=MYRIAD
        OUT=/tmp/rm_cnn4a_myriad_scores.ark
        if [[ "${MODE}" == speech-reference ]]; then
            if [[ "${TARGET}" != amd64 ]]; then
                echo "speech-reference requires --platform amd64 because the pinned OpenVINO CPU plugin is built only for amd64" >&2
                exit 2
            fi
            DEVICE=CPU
            OUT=/tmp/rm_cnn4a_cpu_scores.ark
        fi
        DOCKER_ARGS+=(-v "${ROOT}/vendor/models:/models:ro")
        ENTRY=(/opt/openvino/bin/speech_sample
               -m /models/rm_cnn4a_smbr/openvino/fp16/rm_cnn4a_fp16.xml
               -i /models/rm_cnn4a_smbr/source/feat1_10.ark
               -r /models/rm_cnn4a_smbr/source/score1_10.ark
               -o "${OUT}"
               -d "${DEVICE}"
               -bs 1
               "$@")
        ;;
    mobilenet)
        IR="${IR:-fp16}"
        MODEL_DIR="${ROOT}/vendor/models/mobilenet-v2-ov203/${IR}"
        if [[ ! -f "${MODEL_DIR}/mobilenet-v2-ov203.xml" ]]; then
            echo "no IR in ${MODEL_DIR} - run ./scripts/prepare-mobilenet.sh first" >&2
            exit 1
        fi
        DOCKER_ARGS+=(-v "${ROOT}/vendor/models:/models:ro")
        if [[ $# -eq 0 ]]; then
            set -- --tensor /models/test_data/input_0.f32 --reference /models/test_data/output_0.f32
        fi
        ENTRY=(/opt/openvino/bin/mobilenet_classify --device MYRIAD
               --model "/models/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.xml"
               --weights "/models/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.bin"
               --labels /models/labels/synset.txt "$@")
        ;;
    ssd)
        # the corpus prepared by ./scripts/prepare-ssdlite.sh stays on the host
        MODEL_DIR="${ROOT}/vendor/models/ssdlite_mobilenet_v2"
        if [[ ! -f "${MODEL_DIR}/openvino/ssdlite_mobilenet_v2.xml" ]]; then
            echo "no IR in ${MODEL_DIR} - run ./scripts/prepare-ssdlite.sh first" >&2
            exit 1
        fi
        DOCKER_ARGS+=(-v "${ROOT}/vendor/models:/models:ro")
        if [[ $# -eq 0 ]]; then
            set -- --image /models/images/dog_ssd.ppm
        fi
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

if (( VERBOSE )); then
    platform_print
fi
if [[ "${MODE}" == custom-server ]]; then
    echo ">> target=${TARGET} image=${IMAGE} platform=${DOCKER_PLATFORM}" >&2
    { printf '>> docker run'; printf ' %q' "${DOCKER_ARGS[@]}" "${IMAGE}" "${ENTRY[@]}"; printf '\n'; } >&2
else
    echo ">> target=${TARGET} image=${IMAGE} platform=${DOCKER_PLATFORM}"
    printf '>> docker run'; printf ' %q' "${DOCKER_ARGS[@]}" "${IMAGE}" "${ENTRY[@]}"; printf '\n'
fi
exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${ENTRY[@]}"
