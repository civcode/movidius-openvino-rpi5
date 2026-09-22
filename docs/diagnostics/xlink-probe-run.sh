#!/usr/bin/env bash
# Direct XLink boot/enumeration probe, compiled natively for armv7, arm64 or amd64.
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
      MV=/mv; mkdir -p /tmp/x
      C_SRC="$(ls $MV/XLink/shared/src/*.c) $(ls $MV/XLink/pc/*.c) $(ls $MV/XLink/pc/protocols/*.c) $MV/XLink/pc/MacOS/pthread_semaphore.c"
      COMMON="-g -O0 -D__PC__ -DHAVE_STRUCT_TIMESPEC -DUSE_USB_VSC -I$MV/XLink/shared/include -I$MV/XLink/pc -I$MV/XLink/pc/protocols -I/usr/include/libusb-1.0"
      for f in $C_SRC; do gcc $COMMON -c "$f" -o "/tmp/x/$(basename "$f" .c).o"; done
      gcc $COMMON -c /work/xlink_probe.c -o /tmp/x/probe.o
      g++ /tmp/x/*.o -o /out/xlink_probe -lusb-1.0 -lpthread -ldl
      readelf -h /out/xlink_probe | grep -E "Class:|Machine:"
      /out/xlink_probe > /tmp/probe.out & PB=$!
      for t in 2 4 6 8 10; do
        sleep 2
        echo "=== container USB view t=${t}s ==="
        for d in /sys/bus/usb/devices/*; do
          [ -r "$d/idVendor" ] || continue
          [ "$(cat "$d/idVendor")" = 03e7 ] || continue
          printf "%s %s:%s\n" "$(basename "$d")" "$(cat "$d/idVendor")" "$(cat "$d/idProduct")"
        done
        find /dev/bus/usb -type c -maxdepth 3 -printf "%p " 2>/dev/null || true; echo
      done
      wait $PB
      cat /tmp/probe.out
    '
echo "host after: $(diag_host_usb)"
