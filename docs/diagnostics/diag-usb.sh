#!/usr/bin/env bash
# Trace host syscalls made while the selected self-built runtime boots/opens the
# stick. Requires host strace and permission to ptrace the container process.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/platform.sh"
TARGET_REQUEST="$(platform_default_request)"; IMAGE_OVERRIDE="${IMAGE:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) TARGET_REQUEST="$2"; shift 2 ;;
    --image) IMAGE_OVERRIDE="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
platform_load "$TARGET_REQUEST"; IMAGE="${IMAGE_OVERRIDE:-${DEFAULT_IMAGE}}"
OUT="$ROOT/work/diagnostics/$TARGET"; mkdir -p "$OUT"
sudo python3 "$ROOT/scripts/reset-stick.py" >/dev/null 2>&1 || true
command -v lsusb >/dev/null && echo "host before: $(lsusb | grep -i 03e7 || echo none)"

C="$(docker run -d --platform "$DOCKER_PLATFORM" --network=host -v /dev:/dev \
      --device-cgroup-rule='c 189:* rwm' --entrypoint bash "$IMAGE" -lc '
        sleep 2
        IE_LIB="$(dirname "$(find /opt/openvino/inference_engine/lib -mindepth 2 -maxdepth 2 -name libmyriadPlugin.so -print -quit)")"
        export LD_LIBRARY_PATH="$IE_LIB:/opt/openvino/ngraph/lib"
        IE_VPU_LOG_LEVEL=LOG_DEBUG /opt/openvino/bin/hello_myriad --device MYRIAD --iterations 1 \
          --model /opt/openvino-demo/model/model.xml --weights /opt/openvino-demo/model/model.bin
      ')"
trap 'docker rm -f "$C" >/dev/null 2>&1 || true' EXIT
PID="$(docker inspect -f '{{.State.Pid}}' "$C")"
echo "target=$TARGET container=$C pid=$PID"
sudo timeout 90 strace -f -tt -p "$PID" -e trace=openat,ioctl -o "$OUT/trace.txt" -q || true
docker wait "$C" >/dev/null || true
docker logs "$C" >"$OUT/runtime.log" 2>&1 || true
docker rm "$C" >/dev/null 2>&1 || true; trap - EXIT
command -v lsusb >/dev/null && echo "host after: $(lsusb | grep -i 03e7 || echo none)"
echo "runtime output:"; tail -20 "$OUT/runtime.log"
echo "trace lines: $(wc -l < "$OUT/trace.txt")"
echo 'USB node opens:'; grep -oE '/dev/bus/usb/[0-9]+/[0-9]+' "$OUT/trace.txt" | sort | uniq -c || true
