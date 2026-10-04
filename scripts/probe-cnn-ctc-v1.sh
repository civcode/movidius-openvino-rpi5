#!/usr/bin/env bash
# Cheap cnn_ctc_v1 compatibility gate: init -> ONNX -> MO -> optional MYRIAD.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/platform.sh
source "$ROOT/scripts/platform.sh"
TARGET_REQUEST="$(platform_default_request)"
WITH_DEVICE=1
MANIFEST="$ROOT/work/speech-asr/ami/ami-smoke-v1/manifest.jsonl"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform|-p) TARGET_REQUEST="$2"; shift 2 ;;
        --no-device) WITH_DEVICE=0; shift ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--platform armv7|arm64|amd64] [--no-device] [--manifest path]"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
platform_load "$TARGET_REQUEST"

[[ -s "$MANIFEST" ]] || {
    echo "AMI manifest missing: $MANIFEST" >&2
    echo "prepare it first with: ./scripts/python.sh examples/speech-asr/datasets/ami/prepare_ami.py --subset smoke" >&2
    exit 2
}

TRAIN="$ROOT/work/speech-asr/cnn_ctc_v1/training-probe"
EXPORT="$ROOT/work/speech-asr/cnn_ctc_v1/export"
IR="$ROOT/work/speech-asr/cnn_ctc_v1/openvino/fp16"
PROBE="$ROOT/work/speech-asr/cnn_ctc_v1/probe-$TARGET"
mkdir -p "$PROBE"

echo "== initialize deterministic checkpoint =="
"$ROOT/scripts/python-training.sh"     "$ROOT/examples/speech-asr/training/train_cnn_ctc_v1.py"     --manifest "$MANIFEST"     --output-dir "$TRAIN"     --device cpu     --max-samples 1     --init-only

echo "== export ONNX + PyTorch golden tensors =="
"$ROOT/scripts/python-training.sh"     "$ROOT/examples/speech-asr/training/export_cnn_ctc_v1.py"     --checkpoint "$TRAIN/checkpoint.pt"     --output-dir "$EXPORT"

echo "== compare ONNX Runtime to PyTorch golden =="
"$ROOT/scripts/python-apps.sh"     "$ROOT/examples/speech-asr/evaluation/compare_cnn_ctc_v1_onnx.py"     "$EXPORT"     --output "$PROBE/onnx-comparison.json"

echo "== convert with OpenVINO 2020.3 Model Optimizer =="
"$ROOT/scripts/prepare-cnn-ctc-v1.sh"     --onnx "$EXPORT/cnn_ctc_v1.onnx"     --output-dir "$IR"

if (( ! WITH_DEVICE )); then
    echo "compatibility gate completed through OpenVINO IR (--no-device)"
    exit 0
fi

echo "== validate runtime image supports tensor I/O =="
HELP_LOG="$PROBE/hello-myriad-help.txt"
set +e
"$ROOT/run.sh" --platform "$TARGET" custom --help >"$HELP_LOG" 2>&1
help_status=$?
set -e
cat "$HELP_LOG"
if (( help_status != 0 )) || ! grep -q -- '--tensor' "$HELP_LOG" || ! grep -q -- '--output' "$HELP_LOG"; then
    echo "runtime image does not contain the cnn_ctc_v1 tensor-I/O capable hello_myriad" >&2
    echo "rebuild it from this checkout with: ./build.sh --platform $TARGET" >&2
    echo "then rerun: ./scripts/probe-cnn-ctc-v1.sh --platform $TARGET" >&2
    exit 2
fi

echo "== MYRIAD load/compile/infer probe =="
OUT="$PROBE/myriad-output.f32"
LOG="$PROBE/myriad.log"
set +e
"$ROOT/run.sh" --platform "$TARGET" custom     --model /work/speech-asr/cnn_ctc_v1/openvino/fp16/cnn_ctc_v1.xml     --weights /work/speech-asr/cnn_ctc_v1/openvino/fp16/cnn_ctc_v1.bin     --tensor /work/speech-asr/cnn_ctc_v1/export/golden-input.f32     --output "/work/speech-asr/cnn_ctc_v1/probe-$TARGET/myriad-output.f32"     --iterations 1     >"$LOG" 2>&1
status=$?
set -e
cat "$LOG"
if (( status != 0 )); then
    echo "MYRIAD compatibility probe failed with status $status" >&2
    exit "$status"
fi
test -s "$OUT"

echo "== compare MYRIAD to PyTorch golden =="
"$ROOT/scripts/python-apps.sh"     "$ROOT/examples/speech-asr/evaluation/compare_cnn_ctc_v1_tensor.py"     "$EXPORT/golden-output.f32"     "$OUT"     --candidate-name "MYRIAD-$TARGET"     --output "$PROBE/myriad-comparison.json"

echo "compatibility gate: PASS"
echo "evidence: $PROBE"
