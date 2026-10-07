#!/usr/bin/env bash
# Run the generic hello_myriad tensor boundary via host runtime or Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/platform.sh
source "$ROOT/scripts/platform.sh"
# shellcheck source=scripts/lib/runtime.sh
source "$ROOT/scripts/lib/runtime.sh"

TARGET_REQUEST="$(platform_default_request)"
BACKEND=auto
IMAGE_OVERRIDE="${IMAGE:-}"

usage() {
    cat <<'EOF'
usage: run-myriad-tensor.sh [--platform TARGET] [--backend auto|host|docker] [--image IMAGE] MODE [args...]

MODE:
  check          verify the selected hello_myriad supports tensor streaming
  check-reshape  also require runtime time-axis reshape support
  check-resident also require multi-resident network test support
  custom         run one-shot tensor inference
  custom-server  run the persistent stdin/stdout tensor server

Paths below /work are mapped to the repository work/ tree for host mode and
passed through unchanged for Docker mode.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) TARGET_REQUEST="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --image) IMAGE_OVERRIDE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) break ;;
    esac
done

MODE="${1:-}"
[[ -n "$MODE" ]] || { usage >&2; exit 2; }
shift
case "$MODE" in check|check-reshape|check-resident|custom|custom-server) ;; *) echo "invalid mode: $MODE" >&2; exit 2 ;; esac
runtime_validate_backend "$BACKEND"
platform_load "$TARGET_REQUEST"
IMAGE="${IMAGE_OVERRIDE:-$DEFAULT_IMAGE}"

RT="$ROOT/work/host-runtime/$TARGET"
OV="$RT/openvino"
HOST_READY=0
IE_LIB=""
if runtime_target_can_run_host "$TARGET" &&
   [[ -f "$RT/host-runtime.env" ]] &&
   [[ -x "$OV/bin/hello_myriad" ]]; then
    manifest_target="$(platform_read_manifest_target "$RT/host-runtime.env" 2>/dev/null || true)"
    IE_LIB="$(runtime_find_ie_libdir "$OV" 2>/dev/null || true)"
    if [[ "$manifest_target" == "$TARGET" &&
          -n "$IE_LIB" &&
          -f "$IE_LIB/libmyriadPlugin.so" &&
          -f "$IE_LIB/usb-ma2450.mvcmd" ]]; then
        HOST_READY=1
    fi
fi

find_docker() {
    local candidate
    if candidate="$(command -v docker 2>/dev/null)" && [[ -x "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    for candidate in /usr/bin/docker /usr/local/bin/docker /snap/bin/docker; do
        [[ -x "$candidate" ]] || continue
        printf '%s\n' "$candidate"
        return 0
    done
    return 1
}

DOCKER_BIN="$(find_docker 2>/dev/null || true)"
DOCKER_READY=0
if [[ -n "$DOCKER_BIN" ]] &&
   "$DOCKER_BIN" info >/dev/null 2>&1 &&
   "$DOCKER_BIN" image inspect "$IMAGE" >/dev/null 2>&1; then
    DOCKER_READY=1
fi

case "$BACKEND" in
    host)
        (( HOST_READY == 1 )) || {
            echo "host tensor runtime is unavailable at $RT" >&2
            exit 1
        }
        RESOLVED=host
        ;;
    docker)
        (( DOCKER_READY == 1 )) || {
            echo "Docker tensor runtime is unavailable for image $IMAGE" >&2
            exit 1
        }
        RESOLVED=docker
        ;;
    auto)
        if (( HOST_READY == 1 )); then
            RESOLVED=host
        elif (( DOCKER_READY == 1 )); then
            RESOLVED=docker
        else
            echo "no usable MYRIAD tensor runtime found" >&2
            echo "checked host runtime: $RT" >&2
            echo "checked Docker image: $IMAGE" >&2
            exit 1
        fi
        ;;
esac

if [[ "$MODE" == check || "$MODE" == check-reshape || "$MODE" == check-resident ]]; then
    if [[ "$RESOLVED" == host ]]; then
        output="$(runtime_run_openvino "$TARGET" "$RT" "$OV" "$OV/bin/hello_myriad" --help 2>&1)"
    else
        output="$("$DOCKER_BIN" run --rm --platform "$DOCKER_PLATFORM" "$IMAGE" /opt/openvino/bin/hello_myriad --help 2>&1)"
    fi
    grep -q -- '--stdin' <<<"$output" || {
        printf '%s\n' "$output" >&2
        echo "selected hello_myriad lacks persistent tensor-stream support" >&2
        exit 1
    }
    grep -q -- '--tensor' <<<"$output" || {
        printf '%s\n' "$output" >&2
        echo "selected hello_myriad lacks tensor input support" >&2
        exit 1
    }
    grep -q -- '--output' <<<"$output" || {
        printf '%s\n' "$output" >&2
        echo "selected hello_myriad lacks tensor output support" >&2
        exit 1
    }
    if [[ "$MODE" == check-reshape || "$MODE" == check-resident ]]; then
        grep -q -- '--reshape-time' <<<"$output" || {
            printf '%s\n' "$output" >&2
            echo "selected hello_myriad lacks runtime reshape support" >&2
            echo "rebuild the runtime image and rerun scripts/pull-runtime.sh" >&2
            exit 1
        }
    fi
    if [[ "$MODE" == check-resident ]]; then
        grep -q -- '--resident-times' <<<"$output" || {
            printf '%s\n' "$output" >&2
            echo "selected hello_myriad lacks resident-network test support" >&2
            echo "rebuild the runtime image and rerun scripts/pull-runtime.sh" >&2
            exit 1
        }
    fi
    echo "runtime_backend=$RESOLVED"
    echo "runtime_target=$TARGET"
    exit 0
fi

map_host_path() {
    case "$1" in
        /work) printf '%s\n' "$ROOT/work" ;;
        /work/*) printf '%s/work/%s\n' "$ROOT" "${1#/work/}" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

map_docker_path() {
    case "$1" in
        "$ROOT/work") printf '/work\n' ;;
        "$ROOT/work/"*) printf '/work/%s\n' "${1#"$ROOT/work/"}" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

translated=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--weights|--tensor|--output|--reference)
            [[ $# -ge 2 ]] || { echo "$1 needs a path" >&2; exit 2; }
            option="$1"
            value="$2"
            if [[ "$RESOLVED" == host ]]; then
                value="$(map_host_path "$value")"
            else
                value="$(map_docker_path "$value")"
            fi
            translated+=("$option" "$value")
            shift 2
            ;;
        *)
            translated+=("$1")
            shift
            ;;
    esac
done

echo "run-myriad-tensor: backend=$RESOLVED target=$TARGET mode=$MODE" >&2
if [[ "$RESOLVED" == host ]]; then
    args=(--device MYRIAD)
    [[ "$MODE" == custom-server ]] && args+=(--stdin)
    args+=("${translated[@]}")
    runtime_exec_openvino "$TARGET" "$RT" "$OV" "$OV/bin/hello_myriad" "${args[@]}"
fi

docker_args=(
    run --rm --platform "$DOCKER_PLATFORM"
    --network=host
    -v /dev:/dev
    --device-cgroup-rule='c 189:* rwm'
    -v "$ROOT/work:/work"
)
if [[ "$MODE" == custom-server ]]; then
    docker_args+=(-i -e OV_QUIET=1)
fi
entry=(/opt/openvino/bin/hello_myriad --device MYRIAD)
[[ "$MODE" == custom-server ]] && entry+=(--stdin)
entry+=("${translated[@]}")
exec "$DOCKER_BIN" "${docker_args[@]}" "$IMAGE" "${entry[@]}"
