#!/usr/bin/env bash
# Throwaway: run the direct XLink probe under arbitrary docker flags and print
# what XLink sees before/after XLinkBoot.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MV=$ROOT/vendor/openvino-2020.3.2/inference-engine/thirdparty/movidius
OUT=$ROOT/work/reference-check
cp -f "$ROOT"/vendor/firmware/*.mvcmd "$OUT"/ 2>/dev/null
echo "flags: $*"

sudo python3 "$ROOT/scripts/reset-stick.py" >/dev/null 2>&1
echo "host before: $(lsusb | grep -o '03e7:[0-9a-f]*' || echo none)"

timeout 200 docker run --rm --platform linux/arm/v7 "$@" \
    -v "$MV":/mv:ro -v "$ROOT/docs/diagnostics":/work:ro -v "$OUT":/out \
    ov203-reference-check:latest bash -u -c '
      MV=/mv; mkdir -p /tmp/x
      C_SRC="$(ls $MV/XLink/shared/src/*.c) $(ls $MV/XLink/pc/*.c) $(ls $MV/XLink/pc/protocols/*.c) $MV/XLink/pc/MacOS/pthread_semaphore.c"
      COMMON="-g -O0 -D__PC__ -DHAVE_STRUCT_TIMESPEC -DUSE_USB_VSC -I$MV/XLink/shared/include -I$MV/XLink/pc -I$MV/XLink/pc/protocols -I/usr/include/libusb-1.0"
      rc=0
      for f in $C_SRC; do gcc $COMMON -c "$f" -o "/tmp/x/$(basename "$f" .c).o" || { rc=1; break; }; done
      gcc $COMMON -c /work/xlink_probe.c -o /tmp/x/probe.o || rc=1
      g++ /tmp/x/*.o -o /out/xlink_probe -lusb-1.0 -lpthread -ldl || rc=1
      [ $rc -eq 0 ] || { echo BUILD_FAIL; exit 1; }
      /out/xlink_probe 2>&1 | grep -v "AFTER find-by-old-name" | head -22
      echo "--- fresh libusb view at end of same container ---"
      gcc -O0 -I/usr/include/libusb-1.0 /work/usb_list.c -o /out/usb_list && /out/usb_list | grep -E "03e7|devices$"
    ' 2>&1 | tail -30
echo "host after: $(lsusb | grep -o '03e7:[0-9a-f]*' || echo none)"
