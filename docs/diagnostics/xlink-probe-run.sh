#!/usr/bin/env bash
# Throwaway diagnostic: build + run a direct XLink-level probe (no mvnc) so the
# before/after-boot device names and counts are printed explicitly.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MV=$ROOT/vendor/openvino-2020.3.2/inference-engine/thirdparty/movidius
OUT=$ROOT/work/reference-check
mkdir -p "$OUT"
cp -f "$ROOT"/vendor/firmware/*.mvcmd "$OUT"/ 2>/dev/null

sudo python3 "$ROOT/scripts/reset-stick.py" >/dev/null 2>&1
echo "host before: $(lsusb | grep 03e7 || echo 'no myriad device')"

docker run --rm --privileged --platform linux/arm/v7 \
    -v /dev:/dev \
    -v "$MV":/mv:ro \
    -v "$ROOT/docs/diagnostics":/work:ro \
    -v "$OUT":/out \
    ov203-reference-check:latest bash -u -c '
      MV=/mv
      mkdir -p /tmp/x
      C_SRC="$(ls $MV/XLink/shared/src/*.c) $(ls $MV/XLink/pc/*.c) $(ls $MV/XLink/pc/protocols/*.c) $MV/XLink/pc/MacOS/pthread_semaphore.c"
      COMMON="-g -O0 -D__PC__ -DHAVE_STRUCT_TIMESPEC -DUSE_USB_VSC -I$MV/XLink/shared/include -I$MV/XLink/pc -I$MV/XLink/pc/protocols -I/usr/include/libusb-1.0"
      rc=0
      for f in $C_SRC; do gcc $COMMON -c "$f" -o "/tmp/x/$(basename "$f" .c).o" || { rc=1; break; }; done
      gcc $COMMON -c /work/xlink_probe.c -o /tmp/x/probe.o || rc=1
      g++ /tmp/x/*.o -o /out/xlink_probe -lusb-1.0 -lpthread -ldl || rc=1
      echo "build rc=$rc"
      [ $rc -eq 0 ] || exit 1
      /out/xlink_probe > /tmp/probe.out &
      PB=$!
      for t in 2 4 6 9 12; do
        sleep 2
        echo "=== container view t=${t}s ==="
        ls /sys/bus/usb/devices/ | tr "\n" " "; echo
        grep -H . /sys/bus/usb/devices/*/idVendor /sys/bus/usb/devices/*/idProduct 2>/dev/null | grep -A0 -H . | grep -i "03e7" || echo "  no 03e7 in container sysfs"
        ls /dev/bus/usb/003 /dev/bus/usb/004 2>/dev/null | tr "\n" " "; echo
      done
      wait $PB
      echo "=== probe output ==="
      head -12 /tmp/probe.out
    ' 2>&1 | tail -70

echo "host after: $(lsusb | grep 03e7 || echo 'no myriad device')"
