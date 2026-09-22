#!/usr/bin/env bash
# Run the direct XLink probe with caller-supplied Docker flags to isolate which
# namespace/device options affect re-enumeration on the selected host.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/docs/diagnostics/common.sh"
TARGET_REQUEST="$(platform_default_request)"
if [[ "${1:-}" == --platform ]]; then TARGET_REQUEST="$2"; shift 2; fi
diag_init "${TARGET_REQUEST}"
cp -f "${ROOT}"/vendor/firmware/*.mvcmd "${OUT}/" 2>/dev/null || { echo 'firmware missing' >&2; exit 1; }
diag_reset_stick
echo "target=${TARGET} flags: $*"
echo "host before: $(diag_host_usb)"

timeout 200 docker run --rm --platform "${DOCKER_PLATFORM}" "$@" \
    -v "${MV}:/mv:ro" -v "${ROOT}/docs/diagnostics:/work:ro" -v "${OUT}:/out" \
    "${DIAG_IMAGE}" bash -u -c '
      MV=/mv; mkdir -p /tmp/x
      C_SRC="$(ls $MV/XLink/shared/src/*.c) $(ls $MV/XLink/pc/*.c) $(ls $MV/XLink/pc/protocols/*.c) $MV/XLink/pc/MacOS/pthread_semaphore.c"
      COMMON="-g -O0 -D__PC__ -DHAVE_STRUCT_TIMESPEC -DUSE_USB_VSC -I$MV/XLink/shared/include -I$MV/XLink/pc -I$MV/XLink/pc/protocols -I/usr/include/libusb-1.0"
      for f in $C_SRC; do gcc $COMMON -c "$f" -o "/tmp/x/$(basename "$f" .c).o"; done
      gcc $COMMON -c /work/xlink_probe.c -o /tmp/x/probe.o
      g++ /tmp/x/*.o -o /out/xlink_probe -lusb-1.0 -lpthread -ldl
      /out/xlink_probe
      echo "--- fresh libusb view ---"
      gcc -O0 -I/usr/include/libusb-1.0 /work/usb_list.c -o /out/usb_list -lusb-1.0
      /out/usb_list | grep -E "03e7|devices$" || true
    '
echo "host after: $(diag_host_usb)"
