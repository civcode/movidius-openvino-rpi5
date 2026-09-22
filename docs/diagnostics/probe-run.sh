#!/usr/bin/env bash
# Build and run OpenVINO 2020.3.2's vendored MVNC/XLink stack natively for the
# selected target, bypassing libmyriadPlugin.so.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/docs/diagnostics/common.sh"
TARGET_REQUEST="$(platform_default_request)"
if [[ "${1:-}" == --platform ]]; then TARGET_REQUEST="$2"; shift 2; fi
diag_init "${TARGET_REQUEST}"
cp -f "${ROOT}"/vendor/firmware/*.mvcmd "${OUT}/" 2>/dev/null || { echo 'firmware missing; run scripts/fetch-runtime.sh' >&2; exit 1; }
diag_reset_stick
echo "target=${TARGET} host before: $(diag_host_usb)"

docker run --rm --platform "${DOCKER_PLATFORM}" --network=host \
    -v /dev:/dev --device-cgroup-rule='c 189:* rwm' \
    -v "${MV}:/mv:ro" -v "${ROOT}/docs/diagnostics:/work:ro" -v "${OUT}:/out" \
    "${DIAG_IMAGE}" bash -u -c '
      MV=/mv; mkdir -p /tmp/b
      C_SRC="$(ls $MV/mvnc/src/*.c) $(ls $MV/XLink/shared/src/*.c) $(ls $MV/XLink/pc/*.c) $(ls $MV/XLink/pc/protocols/*.c) $MV/XLink/pc/MacOS/pthread_semaphore.c"
      CPP_SRC="$(ls $MV/mvnc/src/watchdog/*.cpp)"
      COMMON="-g -O0 -D__PC__ -DHAVE_STRUCT_TIMESPEC -DUSE_USB_VSC -I$MV/mvnc/include -I$MV/mvnc/include/watchdog -I$MV/XLink/shared/include -I$MV/XLink/pc -I$MV/XLink/pc/protocols -I/usr/include/libusb-1.0"
      for f in $C_SRC; do gcc $COMMON -c "$f" -o "/tmp/b/$(basename "$f" .c).o"; done
      for f in $CPP_SRC; do g++ $COMMON -c "$f" -o "/tmp/b/$(basename "$f" .cpp).o"; done
      gcc $COMMON -c /work/mvnc_probe.c -o /tmp/b/probe.o
      g++ /tmp/b/*.o -o /out/mvnc_probe -lusb-1.0 -lpthread -ldl
      echo "probe ELF:"; readelf -h /out/mvnc_probe | grep -E "Class:|Machine:"
      echo "--- run (firmware alongside binary) ---"
      NC_LOG_LEVEL=3 /out/mvnc_probe
    '
echo "host after: $(diag_host_usb)"
