#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# validate-reference.sh - prove the premise before building anything
#
# Uses the OFFICIAL OpenVINO 2020.3.2 raspbian runtime (vendor/reference-runtime,
# armv7l binaries) inside an arm32v7 container to check, on this Pi 5 and with
# this MA2450 stick:
#
#   1. an arm32v7 (armhf) container can talk to the stick over /dev/bus/usb and
#      boot it out of ROM mode with OpenVINO's own usb-ma2450.mvcmd
#   2. the official MYRIAD compiler accepts the tiny IR from smoke-test/model
#   3. smoke-test/main.cpp compiles against the 2020.3.2 C++ API
#
# Everything that passes here is later compared against the image built by
# build.sh, so a failure in our own build is easy to tell apart from a host or
# stick problem.
#
#   ./scripts/validate-reference.sh              # full check
#   ./scripts/validate-reference.sh --no-device  # IR/compiler only, no USB
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck source=platform.sh
source "${ROOT}/scripts/platform.sh"
# This validator intentionally uses Intel's ARMv7 Raspbian package. It is a
# Pi/ARMv7 reference check, not the native arm64/amd64 self-built runtime validator.
# Use scripts/verify.sh --platform arm64 or --platform amd64 for native images.

IMAGE="ov203-reference-check:latest"
WITH_DEVICE="yes"
TARGET_REQUEST="$(platform_default_request)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-device) WITH_DEVICE="no"; shift ;;
        --platform) TARGET_REQUEST="${2:-}"; shift 2 ;;
        -h|--help) echo "usage: $0 [--platform armv7] [--no-device]"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

platform_load "${TARGET_REQUEST}"
if [[ "${TARGET}" != armv7 ]]; then
    echo "SKIP: validate-reference.sh uses Intel's ARMv7 Raspbian reference runtime; selected target is ${TARGET}."
    echo "Use ./scripts/verify.sh --platform ${TARGET} for the self-built runtime."
    exit 0
fi

REF_DIR="$(ls -d vendor/reference-runtime/*/deployment_tools/inference_engine | head -1)"
if [[ ! -d "${REF_DIR}" ]]; then
    echo "reference runtime not found - run ./scripts/fetch-runtime.sh first" >&2
    exit 1
fi
REF_DIR="$(cd "${REF_DIR}" && pwd)"

log() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

log "1/4 generate the tiny IR (host python3)"
python3 smoke-test/model/make_tiny_ir.py --outdir smoke-test/model
mkdir -p work/reference-check

log "2/4 build the arm32v7 check image (g++ + libusb-dev)"
# --provenance=false: with Docker 29 the default attestation turns the image into
# an OCI index, and `docker run --platform linux/arm/v7` then refuses to use it.
docker build --platform linux/arm/v7 --provenance=false -t "${IMAGE}" \
    -f scripts/reference-check.Dockerfile scripts

NGRAPH_DIR="$(dirname "${REF_DIR}")/ngraph"
DOCKER_ARGS=(--rm --platform linux/arm/v7 --name "ov203-refcheck-$$"
             -v "${REF_DIR}:/ov:ro"
             -v "${NGRAPH_DIR}:/ng:ro"
             -v "${ROOT}/smoke-test:/work/smoke:ro"
             -v "${ROOT}/half.hpp:/work/half.hpp:ro"
             -v "${ROOT}/work/reference-check:/out")
# USB requirements, measured on this Pi 5 (see README "USB access" section):
#   --network=host            libusb refreshes its device list from kernel uevents;
#                             without the host netns the stick that re-enumerates
#                             (bus 3/2150 -> bus 4/f63b during boot) is never seen
#                             and ncDeviceOpen times out.
#   -v /dev:/dev              live device nodes; --device only copies the nodes that
#                             exist when the container starts.
#   --device-cgroup-rule      the new node (char major 189) must be writable.
[[ "${WITH_DEVICE}" == "yes" ]] && DOCKER_ARGS+=(
    --network=host
    -v /dev:/dev
    --device-cgroup-rule='c 189:* rwm')

log "3/4 compile the tiny IR with the official MYRIAD compiler"
docker run "${DOCKER_ARGS[@]}" "${IMAGE}" bash -c '
    set -euo pipefail
    export LD_LIBRARY_PATH=/ov/lib/armv7l:/ng/lib
    echo "kernel arch: $(uname -m)  (userland is armhf: $(dpkg --print-architecture 2>/dev/null || true))"
    /ov/lib/armv7l/compile_tool -h 2>&1 | head -14 || true
    mkdir -p /out/blob
    # 2020.3 compile_tool has no -w: the .bin is derived from the .xml name
    /ov/lib/armv7l/compile_tool -m /work/smoke/model/model.xml -d MYRIAD -o /out/blob/blob.bin
    ls -l /out/blob
'

log "4/4 build + run the smoke test app against the reference runtime"
docker run "${DOCKER_ARGS[@]}" "${IMAGE}" bash -c '
    set -euo pipefail
    export LD_LIBRARY_PATH=/ov/lib/armv7l:/ng/lib
    # -linference_engine_legacy: Data::setPrecision and the legacy CNNNetwork
    # symbols live in that library in 2020.3.
    g++ -std=c++14 -O1 -I/ov/include -I/ng/include /work/smoke/main.cpp \
        -o /out/hello_myriad_ref -L/ov/lib/armv7l -L/ng/lib \
        -linference_engine -linference_engine_legacy \
        -linference_engine_transformations -linference_engine_lp_transformations \
        -lngraph \
        -ldl -Wl,-rpath,/ov/lib/armv7l -Wl,-rpath,/ng/lib
    /out/hello_myriad_ref --device MYRIAD --iterations 3 \
        --model /work/smoke/model/model.xml --weights /work/smoke/model/model.bin
'

log "done"
echo "artifacts in work/reference-check/ (hello_myriad_ref, blob/)"
echo "remove the throwaway image with: docker rmi ${IMAGE}"
