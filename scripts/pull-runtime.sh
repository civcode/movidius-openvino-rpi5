#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/pull-runtime.sh - extract the runtime from the Docker image into
# work/host-runtime (no container is left running, no QEMU).  This is the
# stage that turns the Docker build into something a native host can use.
#
# The image and platform come from scripts/platform.sh:
#   armv7  also extracts the armhf sysroot (the glibc + libudev runtime the
#          armhf binaries link against), because the host does not have an
#          armhf /lib and the loader (sysroot/lib/ld-linux-armhf.so.3) needs
#          it to resolve SONAMEs.
#   arm64  AArch64 image; binaries run natively on the host, no sysroot.
#   amd64  x86_64 image; binaries run natively on the host, no sysroot.
#
# IMAGE=openvino-2020.3-movidius-<target>:latest ./scripts/pull-runtime.sh
# ---------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/platform.sh
source "${ROOT}/scripts/platform.sh"
platform_load "$(platform_default_request)"

IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"
RT="${ROOT}/work/host-runtime"
OV="${RT}/openvino"
SYSROOT="${RT}/sysroot"

echo "target       : ${TARGET}"
echo "docker       : ${DOCKER_PLATFORM}"
echo "image        : ${IMAGE}"
echo "extract into : ${RT}"

docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
    echo "image not built yet - run ./build.sh --platform ${TARGET} first" >&2; exit 1; }

cid="$(docker create --platform "${DOCKER_PLATFORM}" "${IMAGE}" true)"
echo "created $(docker inspect --format='{{.Id}}' "${cid}" | cut -c1-19)"

# Pull the runtime tree from the scratch container.
rm -rf "${OV}"
mkdir -p "${OV}"
docker cp "${cid}:/opt/openvino" "${OV}"

# armv7 only: the armhf sysroot the glibc loader needs (not on an aarch64
# host); the native targets ship their own glibc in the container image and
# run directly on the host.
if [[ "${TARGET}" == armv7 ]]; then
    rm -rf "${SYSROOT}"
    mkdir -p "${SYSROOT}"
    docker cp "${cid}:/lib/ld-linux-armhf.so.3" "${SYSROOT}/lib/ld-linux-armhf.so.3"
    docker cp "${cid}:/lib/arm-linux-gnueabihf" "${SYSROOT}/lib/arm-linux-gnueabihf"
    docker cp "${cid}:/usr/lib/arm-linux-gnueabihf/libudev.so.1" "${SYSROOT}/usr/lib/arm-linux-gnueabihf/libudev.so.1"
    ls -l "${SYSROOT}/lib" "${SYSROOT}/usr/lib/arm-linux-gnueabihf"
fi

# The demo model + weights that ship in the image (hello_myriad --model).
mkdir -p "${OV}/openvino-demo/model"
docker cp "${cid}:/opt/openvino-demo/model/model.xml"   "${OV}/openvino-demo/model/model.xml"
docker cp "${cid}:/opt/openvino-demo/model/model.bin"   "${OV}/openvino-demo/model/model.bin"

docker rm -f "${cid}" >/dev/null

echo
echo "extracted to: ${OV}"
find "${OV}" -maxdepth 2 -mindepth 1 -printf '  %3TY  %10s  %P\n' 2>/dev/null | sort -k3
[[ -x "${OV}/bin/hello_myriad" ]] && echo
echo "ready - next:  ./scripts/host-run.sh list"
