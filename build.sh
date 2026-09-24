#!/usr/bin/env bash
# Build OpenVINO 2020.3.2 + MYRIAD for ARMv7, native ARM64 or native amd64.
#
#   ./build.sh                         # auto-detect host target
#   ./build.sh --platform armv7        # Pi legacy fallback: linux/arm/v7
#   ./build.sh --platform arm64        # Pi 5 preferred: native linux/arm64
#   ./build.sh --platform amd64        # x86_64 Linux: native linux/amd64
#   ./build.sh --target configure      # stop after CMake configure
#   ./build.sh --target builder
#   ./build.sh --no-cache
#   ./build.sh --no-patches
#   ./build.sh --print-platform
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
# shellcheck source=scripts/platform.sh
source "${ROOT}/scripts/platform.sh"

SRC_DIR="vendor/openvino-2020.3.2"
SRC_COMMIT="${SRC_COMMIT:-0b3773b7405d955d48642667ac5113289b9baab2}"
BUILD_TARGET=""
TARGET_REQUEST="$(platform_default_request)"
IMAGE_OVERRIDE="${IMAGE:-}"
BASE_IMAGE_OVERRIDE="${BASE_IMAGE:-}"
BUILD_JOBS_OVERRIDE="${BUILD_JOBS:-}"
APPLY_PATCHES="${APPLY_PATCHES:-1}"
PRINT_PLATFORM=0
DOCKER_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) [[ $# -ge 2 ]] || { echo "--platform needs armv7|arm64|amd64" >&2; exit 2; }; TARGET_REQUEST="$2"; shift 2 ;;
        --target) [[ $# -ge 2 ]] || { echo "--target needs a Docker stage" >&2; exit 2; }; BUILD_TARGET="$2"; shift 2 ;;
        --no-cache) DOCKER_EXTRA+=(--no-cache); shift ;;
        --no-patches) APPLY_PATCHES=0; shift ;;
        --image) [[ $# -ge 2 ]] || { echo "--image needs a tag" >&2; exit 2; }; IMAGE_OVERRIDE="$2"; shift 2 ;;
        --print-platform) PRINT_PLATFORM=1; shift ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

platform_load "${TARGET_REQUEST}"
BASE_IMAGE="${BASE_IMAGE_OVERRIDE:-${DEFAULT_BASE_IMAGE}}"
IMAGE="${IMAGE_OVERRIDE:-${DEFAULT_IMAGE}}"
if [[ -n "${BUILD_JOBS_OVERRIDE}" ]]; then
    BUILD_JOBS="${BUILD_JOBS_OVERRIDE}"
elif [[ "${TARGET}" == armv7 ]]; then
    # ARMHF userspace: keep the 2 jobs the armv7 baseline was validated with.
    BUILD_JOBS=2
else
    # Native targets (arm64, amd64): use every host core, capped at 8.
    host_jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
    (( host_jobs > 8 )) && host_jobs=8
    BUILD_JOBS="${host_jobs}"
fi
export BASE_IMAGE IMAGE BUILD_JOBS

if (( PRINT_PLATFORM )); then
    platform_print
    exit 0
fi

log() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. vendor payload. Firmware is VPU firmware, not host-CPU code, so the same
# pinned Raspbian release package is used as the source for all host targets.
# ---------------------------------------------------------------------------
if [[ ! -d "${SRC_DIR}/inference-engine" ]]; then
    log "clone pinned OpenVINO ${SRC_COMMIT}"
    ./scripts/clone-openvino.sh
fi
if [[ ! -f vendor/firmware/usb-ma2450.mvcmd ]]; then
    log "download pinned OpenVINO 2020.3.2 firmware/reference payload"
    ./scripts/fetch-runtime.sh
fi
log "prepare local OpenVINO dependency mirror"
./scripts/prepare-deps.sh --platform "${TARGET}"

# ---------------------------------------------------------------------------
# 2. pristine source + deterministic patch validation
# ---------------------------------------------------------------------------
log "reset ${SRC_DIR} to pristine and validate patches"
if [[ -e "${SRC_DIR}/.git" ]]; then
    git -C "${SRC_DIR}" reset --hard HEAD >/dev/null
    git -C "${SRC_DIR}" clean -fd >/dev/null
    git -C "${SRC_DIR}" submodule foreach --recursive \
        'git reset --hard HEAD >/dev/null; git clean -fdq >/dev/null' >/dev/null
else
    echo "   (no .git in ${SRC_DIR}; assuming it is pristine)"
fi

if [[ "${APPLY_PATCHES}" != 1 ]]; then
    echo "   skipping patches by request (--no-patches)"
elif compgen -G 'patches/*.patch' >/dev/null; then
    while IFS= read -r p; do
        if patch -p1 -d "${SRC_DIR}" --dry-run --quiet < "${p}" >/dev/null; then
            echo "   ${p} applies cleanly"
        else
            echo "   PATCH FAILED: ${p}" >&2
            patch -p1 -d "${SRC_DIR}" --dry-run < "${p}" 2>&1 | head -20 >&2
            exit 1
        fi
    done < <(find patches -maxdepth 1 -type f -name '*.patch' | sort)
fi

# ---------------------------------------------------------------------------
# 3. target-qualified stamp + Docker build
# ---------------------------------------------------------------------------
SRC_SHA="$(git -C "${SRC_DIR}" rev-parse HEAD 2>/dev/null || echo "${SRC_COMMIT}")"
PROJECT_REVISION="$(git -C "${ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unversioned)"
if [[ "${SRC_SHA}" != "${SRC_COMMIT}" ]]; then
    echo "WARNING: ${SRC_DIR} is at ${SRC_SHA}, expected pinned ${SRC_COMMIT}" >&2
fi
PATCH_SHA="$(find patches -maxdepth 1 -type f -name '*.patch' -print0 | sort -z | xargs -0r cat | sha256sum | cut -d' ' -f1)"
if [[ "${APPLY_PATCHES}" == 1 ]]; then
    SYNC_STAMP="${TARGET}:${SRC_SHA}+${PATCH_SHA}"
else
    SYNC_STAMP="${TARGET}:${SRC_SHA}+unpatched"
fi

log "build configuration"
platform_print
printf 'build_jobs=%s\nproject_revision=%s\nopenvino_commit=%s\npatches=%s\nsync_stamp=%s\n' \
    "${BUILD_JOBS}" "${PROJECT_REVISION}" "${SRC_SHA}" "${APPLY_PATCHES}" "${SYNC_STAMP}"

log "docker build -> ${IMAGE}"
ARGS=(
    --platform "${DOCKER_PLATFORM}"
    --provenance=false
    --build-arg "TARGET=${TARGET}"
    --build-arg "DOCKER_PLATFORM=${DOCKER_PLATFORM}"
    --build-arg "BASE_IMAGE=${BASE_IMAGE}"
    --build-arg "PROJECT_REVISION=${PROJECT_REVISION}"
    --build-arg "OPENVINO_COMMIT=${SRC_SHA}"
    --build-arg "BUILD_JOBS=${BUILD_JOBS}"
    --build-arg "OPENVINO_SOURCE=${SRC_DIR}"
    --build-arg "SYNC_STAMP=${SYNC_STAMP}"
    --build-arg "APPLY_PATCHES=${APPLY_PATCHES}"
    --build-arg "USE_CMAKE_TOOLCHAIN=${USE_CMAKE_TOOLCHAIN}"
    --build-arg "EXPECTED_ELF_CLASS=${EXPECTED_ELF_CLASS}"
    --build-arg "EXPECTED_ELF_MACHINE_REGEX=${EXPECTED_ELF_MACHINE_REGEX}"
    --build-arg "EXPECTED_ELF_MACHINE_ID=${EXPECTED_ELF_MACHINE_ID}"
)
[[ -n "${BUILD_TARGET}" ]] && ARGS+=(--target "${BUILD_TARGET}")

docker build "${ARGS[@]}" "${DOCKER_EXTRA[@]+"${DOCKER_EXTRA[@]}"}" -t "${IMAGE}" -f Dockerfile .

log "built ${IMAGE} (${TARGET})"
docker image ls "${IMAGE}"
echo
echo "next: ./run.sh --platform ${TARGET}"
echo "      ./run.sh --platform ${TARGET} list"
echo "      ./run.sh --platform ${TARGET} mobilenet"
