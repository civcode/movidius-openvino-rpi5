#!/usr/bin/env bash
# Throwaway diagnostic: build + run OpenVINO 2020.3.2's vendored mvnc/XLink as a
# standalone probe inside the armv7 container, with full NC_LOG_DEBUG verbosity.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MV=$ROOT/vendor/openvino-2020.3.2/inference-engine/thirdparty/movidius
OUT=$ROOT/work/reference-check
mkdir -p "$OUT"
cp -f "$ROOT"/vendor/firmware/*.mvcmd "$OUT"/ 2>/dev/null

# reset the stick to ROM (unbooted) state first
sudo python3 "$ROOT/scripts/reset-stick.py" >/dev/null 2>&1
echo "host before: $(lsusb | grep 03e7 || echo 'no myriad device')"

docker run --rm --privileged --platform linux/arm/v7 \
    -v /dev:/dev \
    -v "$MV":/mv:ro \
    -v "$ROOT/docs/diagnostics":/work:ro \
    -v "$OUT":/out \
    ov203-reference-check:latest bash -u -c '
      MV=/mv
      mkdir -p /tmp/b
      C_SRC="$(ls $MV/mvnc/src/*.c) $(ls $MV/XLink/shared/src/*.c) $(ls $MV/XLink/pc/*.c) $(ls $MV/XLink/pc/protocols/*.c) $MV/XLink/pc/MacOS/pthread_semaphore.c"
      CPP_SRC="$(ls $MV/mvnc/src/watchdog/*.cpp)"
      COMMON="-g -O0 -D__PC__ -DHAVE_STRUCT_TIMESPEC -DUSE_USB_VSC -I$MV/mvnc/include -I$MV/mvnc/include/watchdog -I$MV/XLink/shared/include -I$MV/XLink/pc -I$MV/XLink/pc/protocols -I/usr/include/libusb-1.0"
      for f in $C_SRC; do gcc $COMMON -c "$f" -o "/tmp/b/$(basename "$f" .c).o" || break; done
      for f in $CPP_SRC; do g++ $COMMON -c "$f" -o "/tmp/b/$(basename "$f" .cpp).o" || break; done
      gcc $COMMON -c /work/mvnc_probe.c -o /tmp/b/probe.o
      g++ /tmp/b/*.o -o /out/mvnc_probe -lusb-1.0 -lpthread -ldl
      echo "build rc=$?"
      tail -20 /tmp/build.log
      ls -l /out/mvnc_probe
      echo "--- run (firmware dir = binary dir) ---"
      ls /out/*.mvcmd
      /out/mvnc_probe; echo "probe rc=$?"
    ' 2>&1 | tail -60

echo "host after: $(lsusb | grep 03e7 || echo 'no myriad device')"
