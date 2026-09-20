#!/usr/bin/env bash
# Throwaway diagnostic: trace what OpenVINO 2020.3 compile_tool sees on USB while
# its MYRIAD plugin boots the MA2450 stick inside the armv7 container.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF=$ROOT/vendor/reference-runtime/l_openvino_toolkit_runtime_raspbian_p_2020.3.355/deployment_tools
OUT=$ROOT/work/diag
mkdir -p "$OUT"

sudo python3 "$ROOT/scripts/reset-stick.py" >/dev/null 2>&1
echo "host before: $(lsusb | grep 03e7)"

C=$(docker run -d --privileged --platform linux/arm/v7 \
      -v /dev:/dev \
      -v "$REF/inference_engine":/ov:ro \
      -v "$REF/ngraph":/ng:ro \
      -v "$ROOT/smoke-test":/work/smoke:ro \
      -v "$ROOT/work/reference-check":/out \
      ov203-reference-check:latest \
      bash -c 'export LD_LIBRARY_PATH=/ov/lib/armv7l:/ng/lib;
               IE_VPU_LOG_LEVEL=LOG_DEBUG /ov/lib/armv7l/compile_tool -m /work/smoke/model/model.xml -d MYRIAD \
                   -o /tmp/blob; echo "EXIT=$?"')
PID=$(docker inspect -f '{{.State.Pid}}' "$C")
echo "container pid=$PID"

sudo timeout 40 strace -f -tt -p "$PID" -e trace=openat,ioctl -o "$OUT/trace.txt" -q
sleep 1

docker logs "$C" > "$OUT/compile.log" 2>&1
docker rm -f "$C" >/dev/null 2>&1
echo "host after: $(lsusb | grep 03e7 || echo none)"
echo "compile_tool output:"; tail -5 "$OUT/compile.log"
echo "trace lines: $(wc -l < "$OUT/trace.txt")"
echo "enumeration passes (sysfs dir opens): $(grep -c 'sys/bus/usb/devices/"' "$OUT/trace.txt")"
echo "sysfs stick paths seen:"; grep -oE "devices/(3|4)-1" "$OUT/trace.txt" | sort | uniq -c
echo "stick node opens (/dev/bus/usb):"; grep -oE "/dev/bus/usb/[0-9]+/[0-9]+" "$OUT/trace.txt" | sort | uniq -c
