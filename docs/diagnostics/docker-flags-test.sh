#!/usr/bin/env bash
# Compare Docker flag sets using the selected self-built OpenVINO image.
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

run_case() {
  local name="$1"; shift
  sudo python3 "$ROOT/scripts/reset-stick.py" >/dev/null 2>&1 || true
  echo "===== ${name} target=${TARGET} flags: $*"
  command -v lsusb >/dev/null && { echo "before: $(lsusb | grep -o '03e7:[0-9a-f]*' || echo none)"; } || true
  timeout 180 docker run --rm --platform "$DOCKER_PLATFORM" "$@" --entrypoint bash "$IMAGE" -lc '
    IE_LIB="$(dirname "$(find /opt/openvino/inference_engine/lib -mindepth 2 -maxdepth 2 -name libmyriadPlugin.so -print -quit)")"
    export LD_LIBRARY_PATH="$IE_LIB:/opt/openvino/ngraph/lib"
    /opt/openvino/bin/hello_myriad --device MYRIAD --iterations 1 \
      --model /opt/openvino-demo/model/model.xml --weights /opt/openvino-demo/model/model.bin
  '
  command -v lsusb >/dev/null && { echo "after: $(lsusb | grep -o '03e7:[0-9a-f]*' || echo none)"; } || true
}

run_case 'privileged+nethost'           --privileged --network=host
run_case 'vdev+devcn+nethost'           -v /dev:/dev --device-cgroup-rule='c 189:* rwm' --network=host
run_case 'vdev+nethost (no cgroup rule)' -v /dev:/dev --network=host
