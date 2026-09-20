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
IMAGE="${IMAGE:-openvino-2020.3-rpi5:latest}"
MODE="${1:-demo}"
case "${MODE}" in
    demo|--demo|list|demo-list|shell|bench|custom|mobilenet)
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
    --platform linux/arm/v7
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
esac

echo ">> docker run ${DOCKER_ARGS[*]} ${IMAGE} ${ENTRY[*]}"
exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${ENTRY[@]}"
