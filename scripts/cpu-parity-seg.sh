#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cpu-parity-seg.sh - regression test: the Python full-TensorFlow CPU server
# (examples/deeplab-seg/seg_cpu_server.py) must produce the same segmentation
# as the C++ OpenVINO CPU server (seg_detect, FP32 IR) for the same frame.
# See docs/CPU-BACKENDS.md (spike log).
#
#   scripts/cpu-parity-seg.sh [image.ppm]
#     image.ppm  test image (default: vendor/models/images/dog_ssd.ppm)
#
# Sides:
#   C++ side:  work/host-runtime/<target>/openvino/bin/seg_detect
#              (FP32 IR, device CPU); falls back to the amd64 docker image.
#   TF side:   python3 seg_cpu_server.py if `tensorflow` is installed
#              locally, otherwise a python:3.10-slim container with it.
#
# PASS: same class set, per-class pixel deltas within 1% of the frame,
#       and >= 98% of mask pixels identical (ArgMax tie flips between the
#       two FP32 backends are expected at class boundaries).
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-${ROOT}/vendor/models/images/dog_ssd.ppm}"
[[ -f "${IMAGE}" ]] || { echo "image not found: ${IMAGE}" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"; docker rm -f ov203-parity-tf >/dev/null 2>&1 || true' EXIT

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

# ---- C++ OpenVINO CPU side -------------------------------------------------
CXX_OUT="${TMP}/cpp.out"
LABELS="${ROOT}/vendor/models/labels/pascal_voc.txt"
IR_DIR="${ROOT}/vendor/models/deeplabv3/openvino_fp32"
if [[ -x "${ROOT}/work/host-runtime/amd64/openvino/bin/seg_detect" \
      || -x "${ROOT}/work/host-runtime/arm64/openvino/bin/seg_detect" \
      || -x "${ROOT}/work/host-runtime/armv7/openvino/bin/seg_detect" ]]; then
    # source platform.sh the same way the launchers do
    # shellcheck source=scripts/platform.sh
    source "${ROOT}/scripts/platform.sh"
    platform_load "$(platform_default_request)"
    RT="${ROOT}/work/host-runtime/${TARGET}"
    echo "cpu-parity: C++ side = host runtime ${TARGET}"
    LD_LIBRARY_PATH="${RT}/openvino/inference_engine/${TARGET_LIB_DIR}:${RT}/openvino/ngraph/lib" \
        "${RT}/openvino/bin/seg_detect" \
            --model "${IR_DIR}/deeplabv3.xml" \
            --weights "${IR_DIR}/deeplabv3.bin" \
            --labels "${LABELS}" \
            --device CPU --stdin \
            < "${TMP}/frame.bin" > "${CXX_OUT}" 2>"${TMP}/cpp.err"
else
    echo "cpu-parity: C++ side = docker image (amd64)"
    docker run --rm -i --platform linux/amd64 \
        -v "${ROOT}/vendor/models:/models:ro" \
        openvino-2020.3-movidius-amd64:latest \
        /opt/openvino/bin/seg_detect \
            --model /models/deeplabv3/openvino_fp32/deeplabv3.xml \
            --weights /models/deeplabv3/openvino_fp32/deeplabv3.bin \
            --labels /models/labels/pascal_voc.txt \
            --device CPU --stdin \
            < "${TMP}/frame.bin" > "${CXX_OUT}" 2>"${TMP}/cpp.err"
fi
grep -q '^FRAME ' "${CXX_OUT}" || { echo "C++ side produced no frame output" >&2; cat "${TMP}/cpp.err" >&2; exit 1; }

# ---- TensorFlow side --------------------------------------------------------
TF_OUT="${TMP}/tf.out"
PB="${ROOT}/vendor/models/deeplabv3/source/frozen_inference_graph.pb"
[[ -f "${PB}" ]] || { echo "frozen graph missing: ${PB} - run ./scripts/prepare-deeplabv3.sh" >&2; exit 1; }
if python3 -c 'import tensorflow' >/dev/null 2>&1; then
    echo "cpu-parity: TF side = local python3"
    python3 "${ROOT}/examples/deeplab-seg/seg_cpu_server.py" \
        --model "${PB}" --labels "${LABELS}" \
        < "${TMP}/frame.bin" > "${TF_OUT}" 2>"${TMP}/tf.err"
else
    echo "cpu-parity: TF side = python:3.10-slim container (first run downloads TF)"
    docker rm -f ov203-parity-tf >/dev/null 2>&1 || true
    docker run -d --name ov203-parity-tf python:3.10-slim sleep 3600 >/dev/null
    docker exec ov203-parity-tf pip install -q tensorflow opencv-python-headless >/dev/null 2>&1
    docker cp "${ROOT}/examples/deeplab-seg/seg_cpu_server.py" ov203-parity-tf:/srv.py
    docker cp "${PB}" ov203-parity-tf:/model.pb
    docker cp "${LABELS}" ov203-parity-tf:/labels.txt
    docker exec -i ov203-parity-tf python3 /srv.py \
        --model /model.pb --labels /labels.txt \
        < "${TMP}/frame.bin" > "${TF_OUT}" 2>"${TMP}/tf.err"
fi
grep -q '^FRAME ' "${TF_OUT}" || { echo "TF side produced no frame output" >&2; cat "${TMP}/tf.err" >&2; exit 1; }

# ---- compare ---------------------------------------------------------------
python3 - "${CXX_OUT}" "${TF_OUT}" <<'EOF'
import re
import sys

import numpy as np

def parse(path):
    data = open(path, "rb").read()
    out = {"hist": {}, "mask": None}
    for m in re.finditer(rb"CLASS (\d+) \S+ (\d+)", data):
        out["hist"][int(m.group(1))] = int(m.group(2))
    i = data.find(b"END")
    j = data.rfind(b"MASK ", 0, i)
    nl = data.find(b"\n", j)
    out["mask"] = np.frombuffer(data[nl + 1:i], dtype="<u2")
    return out

a, b = parse(sys.argv[1]), parse(sys.argv[2])
assert a["mask"].size == b["mask"].size, "mask sizes differ"
n = a["mask"].size
agree = float((a["mask"] == b["mask"]).mean())
print(f"cpu-parity: masks {n} px, agreement {agree * 100:.2f}%")
keys = set(a["hist"]) | set(b["hist"])
if set(a["hist"]) != set(b["hist"]):
    print(f"cpu-parity: class sets differ: {sorted(a['hist'])} vs {sorted(b['hist'])}")
ok = agree >= 0.98 and set(a["hist"]) == set(b["hist"])
for k in sorted(keys):
    da, db = a["hist"].get(k, 0), b["hist"].get(k, 0)
    d = abs(da - db)
    fine = d <= 0.01 * n
    print(f"  class {k:2d}: C++ {da:7d}  TF {db:7d}  d {d:5d} ({d / n * 100:.2f}% of frame)"
          f"  {'ok' if fine else 'TOO BIG'}")
    ok = ok and fine
print(f"cpu-parity: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
