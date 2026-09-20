#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build.sh - one-command build of the OpenVINO 2020.3.2 MYRIAD runtime image
#
#   ./build.sh                 # full build (builder -> smoke -> runtime)
#   ./build.sh --target configure   # stop after CMake configure (fast)
#   ./build.sh --target builder
#   ./build.sh --no-cache
#   ./build.sh --no-patches      # build the unpatched tree on purpose: reproduces
#                                # the protoc / ade -Werror failures documented in README
#   BUILD_JOBS=1 ./build.sh    # if the Pi runs short on memory
#
# What it does before calling docker build:
#   1. makes sure the vendor payload is present (source tree, runtime package,
#      firmware, local dependency mirror) - see scripts/
#   2. resets the pinned OpenVINO tree to pristine and re-applies patches/NNNN-*
#      in sorted order inside the build, so a rebuild is always reproducible
#   3. computes a SYNC_STAMP (pinned commit + patch hashes) which the Dockerfile
#      uses to decide whether its /work/src cache mount must be refreshed
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

SRC_DIR="vendor/openvino-2020.3.2"
SRC_COMMIT="${SRC_COMMIT:-0b3773b7405d955d48642667ac5113289b9baab2}"   # tag 2020.3.2
BASE_IMAGE="${BASE_IMAGE:-arm32v7/debian:bullseye}"
BUILD_JOBS="${BUILD_JOBS:-2}"
IMAGE="${IMAGE:-openvino-2020.3-rpi5:latest}"
BUILD_TARGET=""
DOCKER_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) BUILD_TARGET="$2"; shift 2 ;;
        --no-cache) DOCKER_EXTRA+=(--no-cache); shift ;;
        --no-patches) APPLY_PATCHES=0; shift ;;
        --image) IMAGE="$2"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

log() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. vendor payload
# ---------------------------------------------------------------------------
if [[ ! -d "${SRC_DIR}/inference-engine" ]]; then
    log "clone pinned OpenVINO ${SRC_COMMIT}"
    ./scripts/clone-openvino.sh
fi
if ! ls vendor/reference-runtime/*/deployment_tools/inference_engine/include/ie_core.hpp >/dev/null 2>&1; then
    log "download the official 2020.3.2 raspbian runtime (headers + firmware)"
    ./scripts/fetch-runtime.sh
fi
log "local dependency mirror"
./scripts/prepare-deps.sh

# ---------------------------------------------------------------------------
# 2. pristine source + patches
# ---------------------------------------------------------------------------
log "reset ${SRC_DIR} to pristine and apply patches/"
if [[ -e "${SRC_DIR}/.git" ]]; then
    git -C "${SRC_DIR}" reset --hard HEAD >/dev/null
    git -C "${SRC_DIR}" clean -fd >/dev/null
    # patches may touch submodule working trees (ade is a submodule), and the
    # superproject ignores submodule dirtiness ("ignore = dirty"), so reset those
    # too - otherwise a patch would be applied twice on the next build.
    git -C "${SRC_DIR}" submodule foreach --recursive \
        'git reset --hard HEAD >/dev/null; git clean -fdq >/dev/null' >/dev/null
else
    echo "   (no .git in ${SRC_DIR}; assuming it is pristine)"
fi
# The vendor tree stays pristine: the Dockerfile copies it into a cache mount and
# applies patches/NNNN-*.patch there.  Here we only verify that they apply.
# --no-patches builds the unpatched tree on purpose (it reproduces the protoc and
# ade -Werror failures) - see README "Patches".
APPLY_PATCHES="${APPLY_PATCHES:-1}"
if [[ "${APPLY_PATCHES}" != "1" ]]; then
    echo "   skipping patches by request (--no-patches): expect the build to fail"
elif compgen -G "patches/*.patch" >/dev/null; then
    for p in $(ls patches/*.patch | sort); do
        if patch -p1 -d "${SRC_DIR}" --dry-run --quiet < "${p}" >/dev/null; then
            echo "   ${p} applies cleanly (applied inside the build)"
        else
            echo "   PATCH FAILED: ${p}" >&2
            patch -p1 -d "${SRC_DIR}" --dry-run < "${p}" 2>&1 | head -20 >&2
            exit 1
        fi
    done
else
    echo "   (no patches/*.patch)"
fi

# ---------------------------------------------------------------------------
# 3. stamp + docker build
# ---------------------------------------------------------------------------
SRC_SHA="$(git -C "${SRC_DIR}" rev-parse HEAD 2>/dev/null || echo "${SRC_COMMIT}")"
if [[ "${SRC_SHA}" != "${SRC_COMMIT}" ]]; then
    echo "WARNING: ${SRC_DIR} is at ${SRC_SHA}, expected pinned ${SRC_COMMIT}" >&2
fi
PATCH_SHA="$(cat patches/*.patch 2>/dev/null | sha256sum | cut -d' ' -f1)"
if [[ "${APPLY_PATCHES}" == "1" ]]; then
    SYNC_STAMP="${SRC_SHA}+${PATCH_SHA}"
else
    SYNC_STAMP="${SRC_SHA}+unpatched"
fi
echo "SYNC_STAMP=${SYNC_STAMP}"

log "docker build -> ${IMAGE}"
# --provenance=false: Docker 29 adds build attestation by default, which turns
# the image into an OCI index; `docker run --platform linux/arm/v7` then refuses
# to use it.
ARGS=(--platform linux/arm/v7
      --provenance=false
      --build-arg "BASE_IMAGE=${BASE_IMAGE}"
      --build-arg "BUILD_JOBS=${BUILD_JOBS}"
      --build-arg "OPENVINO_SOURCE=${SRC_DIR}"
      --build-arg "SYNC_STAMP=${SYNC_STAMP}"
      --build-arg "APPLY_PATCHES=${APPLY_PATCHES}")
[[ -n "${BUILD_TARGET}" ]] && ARGS+=(--target "${BUILD_TARGET}")

docker build "${ARGS[@]}" "${DOCKER_EXTRA[@]+"${DOCKER_EXTRA[@]}"}" -t "${IMAGE}" -f Dockerfile .

log "built ${IMAGE}"
docker image ls "${IMAGE}"
echo
echo "next: ./run.sh            (device enumeration + tiny-model inference)"
echo "      ./run.sh --list-only"
echo "      ./run.sh shell"
