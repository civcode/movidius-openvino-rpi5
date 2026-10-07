#!/usr/bin/env bash
# Run the fixed-T=512 overlapping-window QuartzNet policy on Pi-hosted MA2450.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/work/speech-asr/librispeech/dev-clean/manifest.jsonl"
IR_DIR="$ROOT/work/speech-asr/quartznet15x5-reference/myriad/openvino/fp16"
DYNAMIC_ONNX="$ROOT/work/speech-asr/quartznet15x5-reference/myriad/dynamic/quartznet15x5_nvidia_ref.onnx"
WORK_DIR="$ROOT/work/speech-asr/quartznet15x5-reference/fixed512-evaluation"
OUTPUT="$WORK_DIR/result.json"
LOG="$WORK_DIR/evaluator.log"
CPUSET="${SPEECH_DECODER_CPUSET:-0-3}"
MAX_SAMPLES=""
HOP_OUTPUT_FRAMES=128
FRESH=0
PREFLIGHT_ONLY=0
LOCK_PATH="${SPEECH_MYRIAD_LOCK:-/tmp/movidius-speech-asr-myriad.lock}"

usage() {
    cat <<'EOF'
usage: evaluate-quartznet15x5-reference-fixed512-myriad.sh [options]

Options:
  --manifest PATH
  --ir-dir PATH
  --dynamic-onnx PATH
  --work-dir PATH
  --output PATH
  --log PATH
  --cpuset LIST
  --hop-output-frames N
  --max-samples N
  --fresh
  --preflight-only
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --ir-dir) IR_DIR="$2"; shift 2 ;;
        --dynamic-onnx) DYNAMIC_ONNX="$2"; shift 2 ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --log) LOG="$2"; shift 2 ;;
        --cpuset) CPUSET="$2"; shift 2 ;;
        --hop-output-frames) HOP_OUTPUT_FRAMES="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --fresh) FRESH=1; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$(uname -m)" =~ ^(aarch64|arm64)$ ]] || {
    echo "fixed512 MYRIAD evaluation requires arm64/aarch64 Pi host" >&2
    exit 2
}
[[ "$CPUSET" =~ ^[0-9,-]+$ ]] || { echo "invalid --cpuset: $CPUSET" >&2; exit 2; }
[[ "$HOP_OUTPUT_FRAMES" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid --hop-output-frames: $HOP_OUTPUT_FRAMES" >&2
    exit 2
}
(( HOP_OUTPUT_FRAMES <= 256 )) || {
    echo "--hop-output-frames must be <= 256" >&2
    exit 2
}
if [[ -n "$MAX_SAMPLES" && ! "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid --max-samples: $MAX_SAMPLES" >&2
    exit 2
fi
for command in flock taskset realpath; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "required command missing: $command" >&2
        exit 2
    }
done
for path in     "$MANIFEST"     "$IR_DIR/quartznet15x5_nvidia_ref.xml"     "$IR_DIR/quartznet15x5_nvidia_ref.bin"     "$IR_DIR/artifacts.json"     "$DYNAMIC_ONNX"; do
    [[ -s "$path" ]] || { echo "required file missing or empty: $path" >&2; exit 2; }
done

"$ROOT/scripts/python-apps.sh" - "$ROOT" <<'PY'
import pathlib
import sys

import numpy
import onnxruntime
import soundfile

root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "examples/speech-asr/python"))

from speech_asr.quartznet_fixed512 import FIXED_TENSOR_FRAMES

assert FIXED_TENSOR_FRAMES == 512
print("fixed512 edge Python preflight: PASS")
PY

"$ROOT/scripts/run-myriad-tensor.sh"     --platform arm64     --backend host     check

if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
    exit 0
fi

if [[ "$FRESH" -eq 1 ]]; then
    work_abs="$(realpath -m "$WORK_DIR")"
    case "$work_abs" in
        "$ROOT"/work/*) rm -rf -- "$work_abs" ;;
        *) echo "refusing --fresh outside repository work/: $work_abs" >&2; exit 2 ;;
    esac
    rm -f -- "$OUTPUT" "$LOG" "$(dirname "$OUTPUT")/hypotheses.jsonl"
fi
mkdir -p "$WORK_DIR" "$(dirname "$OUTPUT")" "$(dirname "$LOG")"

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo '{"status":"blocked","failure_class":"worker_busy"}' >&2
    exit 75
fi

args=(
    --engine myriad
    --platform arm64
    --runtime-backend host
    --manifest "$MANIFEST"
    --ir-dir "$IR_DIR"
    --dynamic-onnx "$DYNAMIC_ONNX"
    --hop-output-frames "$HOP_OUTPUT_FRAMES"
    --work-dir "$WORK_DIR"
    --output "$OUTPUT"
)
[[ -n "$MAX_SAMPLES" ]] && args+=(--max-samples "$MAX_SAMPLES")

set +e
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c "$CPUSET"     "$ROOT/scripts/python-apps.sh"     "$ROOT/examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_fixed512.py"     "${args[@]}" 2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
set -e
exit "$status"
