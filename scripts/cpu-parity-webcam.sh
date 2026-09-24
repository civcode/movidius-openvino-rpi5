#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cpu-parity-webcam.sh - regression test: the Python ONNX Runtime CPU server
# (examples/webcam/mobilenet_cpu_server.py) must produce the same
# classification as the C++ OpenVINO CPU server (mobilenet_server, FP32 IR)
# for the same preprocessed frame.  See docs/CPU-BACKENDS.md (spike log).
#
#   scripts/cpu-parity-webcam.sh [image.ppm]
#     image.ppm  test image (default: vendor/models/images/banana.ppm)
#
# Sides:
#   C++ side:  work/host-runtime/<target>/openvino/bin/mobilenet_server
#              (FP32 IR, device CPU) - the amd64 OV-CPU path.
#   ORT side:  python3 mobilenet_cpu_server.py if `onnxruntime` is installed
#              locally, otherwise a python:3.11-slim container with it.
#
# PASS: max |logit diff| < 1e-3 and identical top-5 class indices.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-${ROOT}/vendor/models/images/banana.ppm}"
[[ -f "${IMAGE}" ]] || { echo "image not found: ${IMAGE}" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"; docker rm -f ov203-parity-ort 2>/dev/null || true' EXIT

echo "cpu-parity: building reference tensor from ${IMAGE}"
python3 - "${ROOT}" "${IMAGE}" "${TMP}/in.f32" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1] + "/examples")
from mobilenet_client import preprocess
import numpy as np, cv2
img = cv2.imread(sys.argv[2])
assert img is not None, f"cannot read {sys.argv[2]}"
open(sys.argv[3], "wb").write(preprocess(img).tobytes())
print(f"cpu-parity: tensor written ({3*224*224*4} bytes)")
EOF

# ---- C++ OpenVINO CPU side -------------------------------------------------
CXX_OUT="${TMP}/cpp_logits.bin"
cpp_model="${ROOT}/vendor/models/mobilenet-v2-ov203/fp32/mobilenet-v2-ov203.xml"
if [[ -x "${ROOT}/work/host-runtime/amd64/openvino/bin/mobilenet_server" \
      || -x "${ROOT}/work/host-runtime/arm64/openvino/bin/mobilenet_server" \
      || -x "${ROOT}/work/host-runtime/armv7/openvino/bin/mobilenet_server" ]]; then
    # source platform.sh the same way the launchers do
    # shellcheck source=scripts/platform.sh
    source "${ROOT}/scripts/platform.sh"
    platform_load "$(platform_default_request)"
    RT="${ROOT}/work/host-runtime/${TARGET}"
    echo "cpu-parity: C++ side = host runtime ${TARGET}"
    LD_LIBRARY_PATH="${RT}/openvino/inference_engine/${TARGET_LIB_DIR}:${RT}/openvino/ngraph/lib" \
        "${RT}/openvino/bin/mobilenet_server" \
            --model "${cpp_model}" \
            --device CPU < "${TMP}/in.f32" > "${CXX_OUT}" 2>"${TMP}/cpp.err"
else
    echo "cpu-parity: C++ side = docker image (amd64)"
    docker run --rm -i --platform linux/amd64 \
        -v "${ROOT}/vendor/models:/models:ro" \
        openvino-2020.3-movidius-amd64:latest \
        /opt/openvino/bin/mobilenet_server \
            --model /models/mobilenet-v2-ov203/fp32/mobilenet-v2-ov203.xml \
            --device CPU < "${TMP}/in.f32" > "${CXX_OUT}" 2>"${TMP}/cpp.err"
fi
[[ -s "${CXX_OUT}" ]] || { echo "C++ side produced no output" >&2; tail "${TMP}/cpp.err" >&2; exit 1; }

# ---- ONNX Runtime side -----------------------------------------------------
ORT_OUT="${TMP}/ort_logits.bin"
MODEL="${ROOT}/vendor/models/onnx/mobilenetv2-7.onnx"
[[ -f "${MODEL}" ]] || { echo "ONNX model missing: ${MODEL} - run ./scripts/prepare-mobilenet.sh" >&2; exit 1; }
if python3 -c 'import onnxruntime' >/dev/null 2>&1; then
    echo "cpu-parity: ORT side = local python3"
    python3 "${ROOT}/examples/webcam/mobilenet_cpu_server.py" --model "${MODEL}" \
        < "${TMP}/in.f32" > "${ORT_OUT}" 2>"${TMP}/ort.err"
else
    echo "cpu-parity: ORT side = python:3.11-slim container"
    docker run -d --name ov203-parity-ort python:3.11-slim sleep 300 >/dev/null
    docker exec ov203-parity-ort pip install -q onnxruntime >/dev/null 2>&1
    docker cp "${ROOT}/examples/webcam/mobilenet_cpu_server.py" ov203-parity-ort:/srv.py
    docker cp "${MODEL}" ov203-parity-ort:/model.onnx
    docker exec -i ov203-parity-ort python3 /srv.py --model /model.onnx \
        < "${TMP}/in.f32" > "${ORT_OUT}" 2>"${TMP}/ort.err"
fi
[[ -s "${ORT_OUT}" ]] || { echo "ORT side produced no output" >&2; tail "${TMP}/ort.err" >&2; exit 1; }

# ---- compare ---------------------------------------------------------------
python3 - "${CXX_OUT}" "${ORT_OUT}" <<'EOF'
import sys
import numpy as np
a = np.fromfile(sys.argv[1], dtype="<f4")
b = np.fromfile(sys.argv[2], dtype="<f4")
assert len(a) == 1000 and len(b) == 1000, f"unexpected sizes {len(a)} {len(b)}"
d = float(np.abs(a - b).max())
ta, tb = np.argsort(-a)[:5], np.argsort(-b)[:5]
print(f"cpu-parity: max |logit diff| = {d:.6f}")
print(f"cpu-parity: top5 C++ = {ta}   ORT = {tb}")
ok = d < 1e-3 and list(ta) == list(tb)
print(f"cpu-parity: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
