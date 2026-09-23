#!/usr/bin/env bash
# Architecture-neutral runtime entry point for armv7, arm64 and amd64 images.
set -euo pipefail

OV_ROOT="${OV_ROOT:-/opt/openvino}"
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -d "${OV_ROOT}" ]] || { echo "OpenVINO install tree not found at ${OV_ROOT}" >&2; exit 1; }

mapfile -t libdirs < <(
    find "${OV_ROOT}/inference_engine/lib" -mindepth 2 -maxdepth 2 \
        -name libinference_engine.so -print 2>/dev/null \
        | xargs -r -n1 dirname | sort -u
)
if [[ ${#libdirs[@]} -ne 1 ]]; then
    echo "expected exactly one OpenVINO architecture lib directory under ${OV_ROOT}/inference_engine/lib; found ${#libdirs[@]}" >&2
    printf '  %s\n' "${libdirs[@]:-}" >&2
    exit 1
fi
OV_LIB="${libdirs[0]}"
NGRAPH_LIB="${OV_ROOT}/ngraph/lib"
export LD_LIBRARY_PATH="${OV_LIB}:${NGRAPH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -f "${OV_LIB}/libmyriadPlugin.so" ]]; then
    echo "MYRIAD plugin missing: ${OV_LIB}/libmyriadPlugin.so" >&2
    echo "try: ldd ${OV_LIB}/libmyriadPlugin.so" >&2
    exit 1
fi
if [[ ! -f "${OV_LIB}/usb-ma2450.mvcmd" ]]; then
    echo "MA2450 firmware missing: ${OV_LIB}/usb-ma2450.mvcmd" >&2
    exit 1
fi

# OV_QUIET=1 suppresses the banner (tools that speak a binary protocol on
# stdout, e.g. examples/webcam/infer-server.sh, need a clean pipe).
if [[ -z "${OV_QUIET:-}" ]]; then
    echo "OV target     : ${OV_TARGET:-unknown}"
    echo "OV install    : ${OV_ROOT}"
    echo "OV libraries  : ${OV_LIB} (arch dir: $(basename "${OV_LIB}"))"
    echo "stick firmware: $(find "${OV_LIB}" -maxdepth 1 -type f -name '*.mvcmd' -printf '%f ' | sort)"
fi

if [[ $# -eq 0 || "${1:-}" == "--demo" ]]; then
    [[ "${1:-}" == "--demo" ]] && shift
    model_args=()
    if [[ -f "${DEMO_DIR}/model/model.xml" ]]; then
        model_args=(--model "${DEMO_DIR}/model/model.xml" --weights "${DEMO_DIR}/model/model.bin")
    fi
    exec "${OV_ROOT}/bin/hello_myriad" --device MYRIAD --iterations 3 "${model_args[@]}" "$@"
fi

exec "$@"
