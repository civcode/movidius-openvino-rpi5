#!/usr/bin/env bash
# Throwaway: find the minimal docker flag set that lets OpenVINO 2020.3 mvnc find
# the stick after it re-enumerates during boot.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF=$ROOT/vendor/reference-runtime/l_openvino_toolkit_runtime_raspbian_p_2020.3.355/deployment_tools
MV=$ROOT/vendor/openvino-2020.3.2/inference-engine/thirdparty/movidius
OUT=$ROOT/work/reference-check

run_case() {
  local name="$1"; shift
  sudo python3 "$ROOT/scripts/reset-stick.py" >/dev/null 2>&1
  echo "===== $name  flags: $*"
  echo "  before: $(lsusb | grep -o '03e7:[0-9a-f]*' || echo none)"
  timeout 160 docker run --rm --platform linux/arm/v7 "$@" \
      -v "$REF/inference_engine":/ov:ro -v "$REF/ngraph":/ng:ro -v "$ROOT/smoke-test":/work/smoke:ro \
      ov203-reference-check:latest bash -u -c '
        export LD_LIBRARY_PATH=/ov/lib/armv7l:/ng/lib
        /ov/lib/armv7l/compile_tool -m /work/smoke/model/model.xml -d MYRIAD -o /tmp/blob.bin 2>&1 | tail -2
        if [ -f /tmp/blob.bin ]; then echo "  RESULT=OK blob=$(stat -c %s /tmp/blob.bin)"; else echo "  RESULT=FAIL"; fi'
  echo "  after: $(lsusb | grep -o '03e7:[0-9a-f]*' || echo none)"
}

run_case "G privileged+nethost"            --privileged --network=host
run_case "F vdev+devcn+nethost"            -v /dev:/dev --device-cgroup-rule='c 189:* rwm' --network=host
run_case "H vdev+nethost (no devcn rule)"  -v /dev:/dev --network=host
