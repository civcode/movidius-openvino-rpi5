#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cpu-parity-ssd.sh - regression test: the Python full-TensorFlow CPU server
# (examples/ssd-detect/ssd_cpu_server.py) must produce the same detections
# as the C++ OpenVINO CPU server (ssd_detect, FP32 IR) for the same frame.
# See docs/CPU-BACKENDS.md (spike log).
#
#   scripts/cpu-parity-ssd.sh [image.ppm]
#     image.ppm  test image (default: vendor/models/images/dog_ssd.ppm)
#
# Sides:
#   C++ side:  work/host-runtime/<target>/openvino/bin/ssd_detect
#              (FP32 IR, device CPU); falls back to the amd64 docker image.
#   TF side:   python3 ssd_cpu_server.py if `tensorflow` is installed
#              locally, otherwise a python:3.10-slim container with it.
#
# PASS: same number of detections, same labels (matched by label+box),
#       all boxes within 2 px and scores within 0.05.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-${ROOT}/vendor/models/images/dog_ssd.ppm}"
[[ -f "${IMAGE}" ]] || { echo "image not found: ${IMAGE}" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"; docker rm -f ov203-parity-tf 2>/dev/null || true' EXIT

echo "cpu-parity: building frame from ${IMAGE}"
python3 - "${IMAGE}" "${TMP}/frame.bin" <<'EOF'
import sys, struct
import cv2, numpy as np
img = cv2.imread(sys.argv[1])
assert img is not None, f"cannot read {sys.argv[1]}"
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
with open(sys.argv[2], "wb") as f:
    f.write(struct.pack("<II", rgb.shape[1], rgb.shape[0]))
    f.write(np.ascontiguousarray(rgb).tobytes())
print(f"cpu-parity: frame written ({rgb.shape[1]}x{rgb.shape[0]})")
EOF

MINCONF=0.5

# ---- C++ OpenVINO CPU side -------------------------------------------------
CXX_OUT="${TMP}/cpp.out"
LABELS="${ROOT}/vendor/models/labels/coco.txt"
IR_DIR="${ROOT}/vendor/models/ssdlite_mobilenet_v2/openvino_fp32"
if [[ -x "${ROOT}/work/host-runtime/amd64/openvino/bin/ssd_detect" \
      || -x "${ROOT}/work/host-runtime/arm64/openvino/bin/ssd_detect" \
      || -x "${ROOT}/work/host-runtime/armv7/openvino/bin/ssd_detect" ]]; then
    # source platform.sh the same way the launchers do
    # shellcheck source=scripts/platform.sh
    source "${ROOT}/scripts/platform.sh"
    platform_load "$(platform_default_request)"
    RT="${ROOT}/work/host-runtime/${TARGET}"
    echo "cpu-parity: C++ side = host runtime ${TARGET}"
    LD_LIBRARY_PATH="${RT}/openvino/inference_engine/${TARGET_LIB_DIR}:${RT}/openvino/ngraph/lib" \
        "${RT}/openvino/bin/ssd_detect" \
            --model "${IR_DIR}/ssdlite_mobilenet_v2.xml" \
            --weights "${IR_DIR}/ssdlite_mobilenet_v2.bin" \
            --labels "${LABELS}" \
            --device CPU --min-conf "${MINCONF}" --stdin \
            < "${TMP}/frame.bin" > "${CXX_OUT}" 2>"${TMP}/cpp.err"
else
    echo "cpu-parity: C++ side = docker image (amd64)"
    docker run --rm -i --platform linux/amd64 \
        -v "${ROOT}/vendor/models:/models:ro" \
        openvino-2020.3-movidius-amd64:latest \
        /opt/openvino/bin/ssd_detect \
            --model /models/ssdlite_mobilenet_v2/openvino_fp32/ssdlite_mobilenet_v2.xml \
            --weights /models/ssdlite_mobilenet_v2/openvino_fp32/ssdlite_mobilenet_v2.bin \
            --labels /models/labels/coco.txt \
            --device CPU --min-conf "${MINCONF}" --stdin \
            < "${TMP}/frame.bin" > "${CXX_OUT}" 2>"${TMP}/cpp.err"
fi
grep -q '^FRAME ' "${CXX_OUT}" || { echo "C++ side produced no frame output" >&2; cat "${TMP}/cpp.err" >&2; exit 1; }

# ---- TensorFlow side --------------------------------------------------------
TF_OUT="${TMP}/tf.out"
PB="${ROOT}/vendor/models/ssdlite_mobilenet_v2/source/frozen_inference_graph.pb"
[[ -f "${PB}" ]] || { echo "frozen graph missing: ${PB} - run ./scripts/prepare-ssdlite.sh" >&2; exit 1; }
if python3 -c 'import tensorflow' >/dev/null 2>&1; then
    echo "cpu-parity: TF side = local python3"
    python3 "${ROOT}/examples/ssd-detect/ssd_cpu_server.py" \
        --model "${PB}" --labels "${LABELS}" --min-conf "${MINCONF}" \
        < "${TMP}/frame.bin" > "${TF_OUT}" 2>"${TMP}/tf.err"
else
    echo "cpu-parity: TF side = python:3.10-slim container (first run downloads TF)"
    docker rm -f ov203-parity-tf 2>/dev/null || true
    docker run -d --name ov203-parity-tf python:3.10-slim sleep 3600 >/dev/null
    docker exec ov203-parity-tf pip install -q tensorflow opencv-python-headless >/dev/null 2>&1
    docker cp "${ROOT}/examples/ssd-detect/ssd_cpu_server.py" ov203-parity-tf:/srv.py
    docker cp "${PB}" ov203-parity-tf:/model.pb
    docker cp "${LABELS}" ov203-parity-tf:/labels.txt
    docker exec -i ov203-parity-tf python3 /srv.py \
        --model /model.pb --labels /labels.txt --min-conf "${MINCONF}" \
        < "${TMP}/frame.bin" > "${TF_OUT}" 2>"${TMP}/tf.err"
fi
grep -q '^FRAME ' "${TF_OUT}" || { echo "TF side produced no frame output" >&2; cat "${TMP}/tf.err" >&2; exit 1; }

# ---- compare ---------------------------------------------------------------
python3 - "${CXX_OUT}" "${TF_OUT}" <<'EOF'
import re
import sys

def parse(path):
    dets = []
    for line in open(path):
        m = re.match(r"^DET (\S+) ([\d.]+) (\d+) (\d+) (\d+) (\d+)$", line)
        if m:
            dets.append((m.group(1), float(m.group(2)),
                         int(m.group(3)), int(m.group(4)),
                         int(m.group(5)), int(m.group(6))))
    return dets

a, b = parse(sys.argv[1]), parse(sys.argv[2])
print(f"cpu-parity: C++ {len(a)} detection(s), TF {len(b)} detection(s)")
if len(a) != len(b):
    print("cpu-parity: FAIL (different detection counts)")
    sys.exit(1)
# match by (label, x1, y1) ordering so near-tie score flips do not break parity
key = lambda d: (d[0], d[2], d[3])
a.sort(key=key)
b.sort(key=key)
ok = True
for da, db in zip(a, b):
    dbox = max(abs(da[2]-db[2]), abs(da[3]-db[3]), abs(da[4]-db[4]), abs(da[5]-db[5]))
    dscore = abs(da[1] - db[1])
    match = da[0] == db[0] and dbox <= 2 and dscore <= 0.05
    print(f"  {'ok ' if match else 'MISMATCH'} {da}  vs  {db}")
    ok = ok and match
print(f"cpu-parity: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
