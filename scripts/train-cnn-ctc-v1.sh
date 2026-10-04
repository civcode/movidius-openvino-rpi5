#!/usr/bin/env bash
# Train cnn_ctc_v1, export ONNX, compare reference output, and convert FP16 IR.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/work/speech-asr/ami/ami-smoke-v1/manifest.jsonl"
VALIDATION=""
DEVICE=cuda
EPOCHS=""
MAX_SAMPLES=""
TRAIN="$ROOT/work/speech-asr/cnn_ctc_v1/training"
EXPORT="$ROOT/work/speech-asr/cnn_ctc_v1/export"
IR="$ROOT/work/speech-asr/cnn_ctc_v1/openvino/fp16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) [[ $# -ge 2 ]] || { echo "--manifest needs a path" >&2; exit 2; }; MANIFEST="$2"; shift 2 ;;
        --validation-manifest) [[ $# -ge 2 ]] || { echo "--validation-manifest needs a path" >&2; exit 2; }; VALIDATION="$2"; shift 2 ;;
        --device) [[ $# -ge 2 ]] || { echo "--device needs cuda|cpu|auto" >&2; exit 2; }; DEVICE="$2"; shift 2 ;;
        --epochs) [[ $# -ge 2 ]] || { echo "--epochs needs an integer" >&2; exit 2; }; EPOCHS="$2"; shift 2 ;;
        --max-samples) [[ $# -ge 2 ]] || { echo "--max-samples needs an integer" >&2; exit 2; }; MAX_SAMPLES="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--manifest path] [--validation-manifest path] [--device cuda|cpu|auto] [--epochs N] [--max-samples N]"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$DEVICE" in cuda|cpu|auto) ;; *) echo "--device must be cuda, cpu or auto" >&2; exit 2 ;; esac
[[ -s "$MANIFEST" ]] || { echo "training manifest missing: $MANIFEST" >&2; exit 2; }
if [[ -n "$VALIDATION" && ! -s "$VALIDATION" ]]; then
    echo "validation manifest missing: $VALIDATION" >&2
    exit 2
fi

args=(--manifest "$MANIFEST" --output-dir "$TRAIN" --device "$DEVICE")
[[ -n "$VALIDATION" ]] && args+=(--validation-manifest "$VALIDATION")
[[ -n "$EPOCHS" ]] && args+=(--epochs "$EPOCHS")
[[ -n "$MAX_SAMPLES" ]] && args+=(--max-samples "$MAX_SAMPLES")

echo "== train cnn_ctc_v1 =="
"$ROOT/scripts/python-training.sh"     "$ROOT/examples/speech-asr/training/train_cnn_ctc_v1.py"     "${args[@]}"

echo "== export fixed-shape ONNX =="
"$ROOT/scripts/python-training.sh"     "$ROOT/examples/speech-asr/training/export_cnn_ctc_v1.py"     --checkpoint "$TRAIN/checkpoint.pt"     --output-dir "$EXPORT"

echo "== compare ONNX Runtime vs PyTorch golden =="
"$ROOT/scripts/python-apps.sh"     "$ROOT/examples/speech-asr/evaluation/compare_cnn_ctc_v1_onnx.py"     "$EXPORT"     --output "$EXPORT/onnx-comparison.json"

echo "== convert OpenVINO 2020.3 FP16 IR =="
"$ROOT/scripts/prepare-cnn-ctc-v1.sh"     --onnx "$EXPORT/cnn_ctc_v1.onnx"     --output-dir "$IR"

echo "training/export pipeline: PASS"
echo "checkpoint: $TRAIN/checkpoint.pt"
echo "onnx:       $EXPORT/cnn_ctc_v1.onnx"
echo "ir:         $IR/cnn_ctc_v1.xml"
