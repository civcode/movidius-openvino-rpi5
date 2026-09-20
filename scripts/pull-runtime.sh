#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/pull-runtime.sh - copy the built OpenVINO 2020.3.2 runtime (and the
# arm32v7 userspace it needs) OUT of the Docker image into work/host-runtime/,
# so the demo binaries and compile_tool can run directly on the aarch64 host.
#
#   ./scripts/pull-runtime.sh            # extract runtime + armhf sysroot
#
# Result (work/host-runtime/):
#   sysroot/lib/ld-linux-armhf.so.3       the arm32 dynamic loader, from the image
#   sysroot/{lib,usr/lib}/arm-linux-gnueabihf/*.so*   libc 2.31, libstdc++ 6.0.28,
#                                          libusb 1.0.24, libudev 247 - the only
#                                          non-OpenVINO libraries the binaries need
#   openvino/                             == image /opt/openvino (34 MB, plugins,
#                                          mvnc firmware .mvcmd next to the plugin)
#   openvino-demo/                        == image /opt/openvino-demo (tiny model)
#
# Nothing is installed on the host: no apt packages, no dpkg multiarch, no QEMU -
# the kernel executes the ELF32 binaries in compat mode (CONFIG_COMPAT=y).
# ---------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-openvino-2020.3-rpi5:latest}"
RT="${ROOT}/work/host-runtime"
mkdir -p "${RT}"
cd "${RT}"

rm -rf sysroot openvino openvino-demo sysroot.tgz
mkdir -p sysroot

# -h dereferences symlinks: every SONAME becomes a real file in the tarball.
docker run --rm --platform linux/arm/v7 -v "${RT}":/host "${IMAGE}" bash -c '
set -e
files="
/lib/ld-linux-armhf.so.3
/lib/arm-linux-gnueabihf/libc.so.6
/lib/arm-linux-gnueabihf/libm.so.6
/lib/arm-linux-gnueabihf/libdl.so.2
/lib/arm-linux-gnueabihf/libpthread.so.0
/lib/arm-linux-gnueabihf/libgcc_s.so.1
/lib/arm-linux-gnueabihf/librt.so.1
/usr/lib/arm-linux-gnueabihf/libstdc++.so.6
/usr/lib/arm-linux-gnueabihf/libusb-1.0.so.0
/usr/lib/arm-linux-gnueabihf/libudev.so.1
"
tar chzf /host/sysroot.tgz -C / $files
' 2>/dev/null
tar xzf sysroot.tgz -C sysroot
rm -f sysroot.tgz

cid="$(docker create --platform linux/arm/v7 "${IMAGE}" true)"
docker cp "${cid}":/opt/openvino      "${RT}/openvino"      >/dev/null
docker cp "${cid}":/opt/openvino-demo "${RT}/openvino-demo" >/dev/null
docker rm "${cid}" >/dev/null
rm -rf openvino-demo/model/__pycache__

echo "host runtime ready in ${RT}"
echo "  sysroot  $(du -sh sysroot | cut -f1),  $(find sysroot -name '*.so*' | wc -l) libraries"
echo "  openvino $(du -sh openvino | cut -f1)"
echo "run it with:  ./scripts/host-run.sh list"
