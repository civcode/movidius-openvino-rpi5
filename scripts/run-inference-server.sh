#!/usr/bin/env bash
# Common source-tree inference-server launcher for classification, detection and
# segmentation examples. Existing infer-*-server.sh files are compatibility
# wrappers around this command.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/platform.sh
source "$ROOT/scripts/platform.sh"
# shellcheck source=scripts/lib/runtime.sh
source "$ROOT/scripts/lib/runtime.sh"

APP=""
BACKEND=auto
DEVICE=MYRIAD
IR="${IR:-fp16}"
MIN_CONF=0.5
TARGET_REQUEST="$(platform_default_request)"
IMAGE_OVERRIDE="${IMAGE:-}"

usage() {
    cat <<'EOF'
Usage: run-inference-server.sh --app mobilenet|ssd|seg [options]

Options:
  --backend auto|host|docker   default: auto
  --device DEVICE             default: MYRIAD
  --ir fp16|fp32              MobileNet IR precision (default: fp16)
  --min-conf FLOAT            SSD score threshold (default: 0.5)
  --platform TARGET           armv7|arm64|amd64
  --image IMAGE               override Docker image

The server inherits stdin/stdout. Diagnostics are written to stderr.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --app) [[ $# -ge 2 ]] || { echo "--app needs a value" >&2; exit 2; }; APP="$2"; shift 2 ;;
        --backend) [[ $# -ge 2 ]] || { echo "--backend needs a value" >&2; exit 2; }; BACKEND="$2"; shift 2 ;;
        --device) [[ $# -ge 2 ]] || { echo "--device needs a value" >&2; exit 2; }; DEVICE="$2"; shift 2 ;;
        --ir) [[ $# -ge 2 ]] || { echo "--ir needs a value" >&2; exit 2; }; IR="$2"; shift 2 ;;
        --min-conf) [[ $# -ge 2 ]] || { echo "--min-conf needs a value" >&2; exit 2; }; MIN_CONF="$2"; shift 2 ;;
        --platform) [[ $# -ge 2 ]] || { echo "--platform needs a value" >&2; exit 2; }; TARGET_REQUEST="$2"; shift 2 ;;
        --image) [[ $# -ge 2 ]] || { echo "--image needs a value" >&2; exit 2; }; IMAGE_OVERRIDE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$APP" in mobilenet|ssd|seg) ;; *) echo "--app must be mobilenet, ssd or seg" >&2; exit 2 ;; esac
runtime_validate_backend "$BACKEND"
case "$IR" in fp16|fp32) ;; *) echo "--ir must be fp16 or fp32" >&2; exit 2 ;; esac
python3 - "$MIN_CONF" <<'PY' || exit 2
import sys
try:
    value=float(sys.argv[1])
except ValueError:
    raise SystemExit("--min-conf must be a number between 0 and 1")
if not 0 <= value <= 1:
    raise SystemExit("--min-conf must be between 0 and 1")
PY

platform_load "$TARGET_REQUEST"
IMAGE="${IMAGE_OVERRIDE:-$DEFAULT_IMAGE}"
MODELS="${OV_MODELS_DIR:-$ROOT/vendor/models}"
RT="$ROOT/work/host-runtime/$TARGET"
OV="$RT/openvino"

DEVICE_UPPER="${DEVICE^^}"
CPU_PY=0
if [[ "$TARGET" == arm64 && "$DEVICE_UPPER" == CPU ]]; then
    CPU_PY=1
fi

SERVER_NAME=""
CPU_MODULE=""
CPU_SCRIPT=""
CPU_SOURCE_MODEL=""
CPU_LABELS=""
CPP_MODEL_XML=""
CPP_MODEL_BIN=""
CPP_LABELS=""
DOCKER_CPP_MODEL=""
DOCKER_CPP_BIN=""
DOCKER_LABELS=""
DOCKER_CPU_SCRIPT=""
DOCKER_CPU_MODEL=""

case "$APP" in
    mobilenet)
        SERVER_NAME=mobilenet_server
        CPU_MODULE=onnxruntime
        CPU_SCRIPT="$ROOT/examples/webcam/mobilenet_cpu_server.py"
        CPU_SOURCE_MODEL="$MODELS/onnx/mobilenetv2-7.onnx"
        DOCKER_CPU_SCRIPT=/opt/openvino-demo/cpu-servers/mobilenet_cpu_server.py
        DOCKER_CPU_MODEL=/models/onnx/mobilenetv2-7.onnx
        if [[ "$DEVICE_UPPER" == CPU && "$TARGET" == amd64 ]]; then
            IR=fp32
        fi
        CPP_MODEL_XML="$MODELS/mobilenet-v2-ov203/$IR/mobilenet-v2-ov203.xml"
        CPP_MODEL_BIN="${CPP_MODEL_XML%.xml}.bin"
        DOCKER_CPP_MODEL="/models/mobilenet-v2-ov203/$IR/mobilenet-v2-ov203.xml"
        DOCKER_CPP_BIN="${DOCKER_CPP_MODEL%.xml}.bin"
        ;;
    ssd)
        SERVER_NAME=ssd_detect
        CPU_MODULE=tensorflow
        CPU_SCRIPT="$ROOT/examples/ssd-detect/ssd_cpu_server.py"
        CPU_SOURCE_MODEL="$MODELS/ssdlite_mobilenet_v2/source/frozen_inference_graph.pb"
        CPU_LABELS="$MODELS/labels/coco.txt"
        DOCKER_CPU_SCRIPT=/opt/openvino-demo/cpu-servers/ssd_cpu_server.py
        DOCKER_CPU_MODEL=/models/ssdlite_mobilenet_v2/source/frozen_inference_graph.pb
        if [[ "$DEVICE_UPPER" == CPU && "$TARGET" == amd64 ]]; then MODEL_IR_DIR=openvino_fp32; else MODEL_IR_DIR=openvino; fi
        CPP_MODEL_XML="$MODELS/ssdlite_mobilenet_v2/$MODEL_IR_DIR/ssdlite_mobilenet_v2.xml"
        CPP_MODEL_BIN="${CPP_MODEL_XML%.xml}.bin"
        CPP_LABELS="$MODELS/labels/coco.txt"
        DOCKER_CPP_MODEL="/models/ssdlite_mobilenet_v2/$MODEL_IR_DIR/ssdlite_mobilenet_v2.xml"
        DOCKER_CPP_BIN="${DOCKER_CPP_MODEL%.xml}.bin"
        DOCKER_LABELS=/models/labels/coco.txt
        ;;
    seg)
        SERVER_NAME=seg_detect
        CPU_MODULE=tensorflow
        CPU_SCRIPT="$ROOT/examples/deeplab-seg/seg_cpu_server.py"
        CPU_SOURCE_MODEL="$MODELS/deeplabv3/source/frozen_inference_graph.pb"
        CPU_LABELS="$MODELS/labels/pascal_voc.txt"
        DOCKER_CPU_SCRIPT=/opt/openvino-demo/cpu-servers/seg_cpu_server.py
        DOCKER_CPU_MODEL=/models/deeplabv3/source/frozen_inference_graph.pb
        if [[ "$DEVICE_UPPER" == CPU && "$TARGET" == amd64 ]]; then MODEL_IR_DIR=openvino_fp32; else MODEL_IR_DIR=openvino; fi
        CPP_MODEL_XML="$MODELS/deeplabv3/$MODEL_IR_DIR/deeplabv3.xml"
        CPP_MODEL_BIN="${CPP_MODEL_XML%.xml}.bin"
        CPP_LABELS="$MODELS/labels/pascal_voc.txt"
        DOCKER_CPP_MODEL="/models/deeplabv3/$MODEL_IR_DIR/deeplabv3.xml"
        DOCKER_CPP_BIN="${DOCKER_CPP_MODEL%.xml}.bin"
        DOCKER_LABELS=/models/labels/pascal_voc.txt
        ;;
esac

if [[ "$DEVICE_UPPER" == CPU && "$TARGET" == armv7 ]]; then
    echo "--device CPU is not available for armv7; use MYRIAD" >&2
    exit 1
fi

HOST_READY=0
HOST_PY=""
if (( CPU_PY )); then
    if HOST_PY="$(runtime_python_interpreter "$CPU_MODULE" "$ROOT" 2>/dev/null)" &&
       [[ -f "$CPU_SOURCE_MODEL" ]] &&
       { [[ -z "$CPU_LABELS" ]] || [[ -f "$CPU_LABELS" ]]; }; then
        HOST_READY=1
    fi
else
    if runtime_target_can_run_host "$TARGET" &&
       [[ -x "$OV/bin/$SERVER_NAME" ]]; then
        HOST_READY=1
    fi
    # Running from an already-built runtime image is also a valid host mode.
    if runtime_target_can_run_host "$TARGET" &&
       [[ -x "${OV_ROOT:-/opt/openvino}/bin/$SERVER_NAME" ]]; then
        HOST_READY=1
    fi
fi

RESOLVED="$(runtime_resolve_backend "$BACKEND" "$HOST_READY" "$IMAGE")"
echo "run-inference-server: app=$APP backend=$RESOLVED target=$TARGET device=$DEVICE" >&2

check_model_file() {
    [[ -f "$1" ]] || {
        echo "required model file missing: $1" >&2
        echo "run the corresponding scripts/prepare-*.sh recipe first" >&2
        exit 1
    }
}

host_backend() {
    if (( CPU_PY )); then
        check_model_file "$CPU_SOURCE_MODEL"
        [[ -z "$CPU_LABELS" ]] || check_model_file "$CPU_LABELS"
        case "$APP" in
            mobilenet)
                exec "$HOST_PY" "$CPU_SCRIPT" --model "$CPU_SOURCE_MODEL"
                ;;
            ssd)
                exec "$HOST_PY" "$CPU_SCRIPT" --model "$CPU_SOURCE_MODEL" \
                    --labels "$CPU_LABELS" --min-conf "$MIN_CONF"
                ;;
            seg)
                exec "$HOST_PY" "$CPU_SCRIPT" --model "$CPU_SOURCE_MODEL" \
                    --labels "$CPU_LABELS"
                ;;
        esac
    fi

    check_model_file "$CPP_MODEL_XML"
    check_model_file "$CPP_MODEL_BIN"
    [[ -z "$CPP_LABELS" ]] || check_model_file "$CPP_LABELS"
    runtime_device_uses "$DEVICE" MYRIAD && runtime_warn_mvnc_mutex || true

    local ov_root="$OV" rt_root="$RT" server="$OV/bin/$SERVER_NAME"
    if [[ ! -x "$server" ]]; then
        ov_root="${OV_ROOT:-/opt/openvino}"
        rt_root="$ROOT"
        server="$ov_root/bin/$SERVER_NAME"
    fi
    [[ -x "$server" ]] || { echo "host runtime server missing: $server" >&2; exit 1; }

    local args=(--model "$CPP_MODEL_XML" --weights "$CPP_MODEL_BIN" --device "$DEVICE")
    case "$APP" in
        mobilenet) ;;
        ssd) args+=(--labels "$CPP_LABELS" --stdin --min-conf "$MIN_CONF") ;;
        seg) args+=(--labels "$CPP_LABELS" --stdin) ;;
    esac
    runtime_exec_openvino "$TARGET" "$rt_root" "$ov_root" "$server" "${args[@]}"
}

docker_backend() {
    local name="ov203-${APP}-$$"
    if (( CPU_PY )); then
        check_model_file "$CPU_SOURCE_MODEL"
        [[ -z "$CPU_LABELS" ]] || check_model_file "$CPU_LABELS"
        local args=(docker run --rm -i --platform "$DOCKER_PLATFORM" --name "$name"
                    -e OV_QUIET=1 -v "$MODELS:/models:ro" "$IMAGE"
                    python3 "$DOCKER_CPU_SCRIPT" --model "$DOCKER_CPU_MODEL")
        case "$APP" in
            mobilenet) ;;
            ssd) args+=(--labels /models/labels/coco.txt --min-conf "$MIN_CONF") ;;
            seg) args+=(--labels /models/labels/pascal_voc.txt) ;;
        esac
        exec "${args[@]}"
    fi

    check_model_file "$CPP_MODEL_XML"
    check_model_file "$CPP_MODEL_BIN"
    [[ -z "$CPP_LABELS" ]] || check_model_file "$CPP_LABELS"
    local run=(docker run --rm -i --platform "$DOCKER_PLATFORM" --name "$name"
               --network=host -v "$MODELS:/models:ro" -e OV_QUIET=1)
    if runtime_device_uses "$DEVICE" MYRIAD; then
        run+=(-v /dev:/dev --device-cgroup-rule='c 189:* rwm')
    fi
    run+=("$IMAGE" "/opt/openvino/bin/$SERVER_NAME"
          --model "$DOCKER_CPP_MODEL" --weights "$DOCKER_CPP_BIN" --device "$DEVICE")
    case "$APP" in
        mobilenet) ;;
        ssd) run+=(--labels "$DOCKER_LABELS" --stdin --min-conf "$MIN_CONF") ;;
        seg) run+=(--labels "$DOCKER_LABELS" --stdin) ;;
    esac
    exec "${run[@]}"
}

case "$RESOLVED" in
    host) host_backend ;;
    docker) docker_backend ;;
    *) echo "internal error: unresolved backend '$RESOLVED'" >&2; exit 70 ;;
esac
