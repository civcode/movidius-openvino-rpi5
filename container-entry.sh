#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Container entry point for the openvino-2020.3-rpi5 runtime image.
#
# It finds the architecture directory the build installed into (lib/armv7l for
# the arm32v7 container) and exports it in LD_LIBRARY_PATH, so nothing in the
# image has to hard-code the architecture name.  Then it runs the smoke test.
# ---------------------------------------------------------------------------
set -euo pipefail

OV_ROOT="${OV_ROOT:-/opt/openvino}"
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "${OV_ROOT}" ]]; then
    echo "OpenVINO install tree not found at ${OV_ROOT}" >&2
    exit 1
fi

mapfile -t libdirs < <(find "${OV_ROOT}/inference_engine/lib" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
if [[ ${#libdirs[@]} -eq 0 ]]; then
    echo "no library directory under ${OV_ROOT}/inference_engine/lib" >&2
    exit 1
fi
OV_LIB="${libdirs[0]}"
# ngraph is installed next to the inference engine (deployment_tools/ngraph/lib)
NGRAPH_LIB="${OV_ROOT}/ngraph/lib"
export LD_LIBRARY_PATH="${OV_LIB}:${NGRAPH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "OV install    : ${OV_ROOT} (libs: ${OV_LIB}:${NGRAPH_LIB})"
echo "stick firmware: $(ls "${OV_LIB}"/*.mvcmd 2>/dev/null | tr '\n' ' ')"

# plugins.xml / firmware are found relative to libmyriadPlugin.so, which lives
# in OV_LIB because CMAKE_INSTALL_PREFIX is the install tree root.

if [[ $# -eq 0 || "${1:-}" == "--demo" ]]; then
    [[ "${1:-}" == "--demo" ]] && shift
    model_args=()
    if [[ -f "${DEMO_DIR}/model/model.xml" ]]; then
        model_args=(--model "${DEMO_DIR}/model/model.xml" --weights "${DEMO_DIR}/model/model.bin")
    fi
    echo
    # later flags win in hello_myriad, so the caller can override --iterations
    exec "${OV_ROOT}/bin/hello_myriad" --device MYRIAD --iterations 3 "${model_args[@]}" "$@"
fi

exec "$@"
