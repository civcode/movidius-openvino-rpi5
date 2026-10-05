#!/usr/bin/env bash
# Phase 11 cheap graph gate: initialized cnn_ctc_v3 -> ONNX -> OpenVINO IR.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$ROOT/work/speech-asr/cnn_ctc_v3/compatibility-probe"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --work-dir) [[ $# -ge 2 ]] || { echo "--work-dir needs a path" >&2; exit 2; }; WORK_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--work-dir path]"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

TRAIN="$WORK_DIR/training-init"
EXPORT="$WORK_DIR/export"
IR="$WORK_DIR/openvino/fp16"
mkdir -p "$WORK_DIR"

echo "== initialize cnn_ctc_v3 =="
"$ROOT/scripts/python-training.sh"     "$ROOT/examples/speech-asr/training/train_cnn_ctc_v3.py"     --output-dir "$TRAIN"     --device cpu     --init-only

echo "== export initialized ONNX =="
"$ROOT/scripts/python-training.sh"     "$ROOT/examples/speech-asr/training/export_cnn_ctc_v3.py"     --checkpoint "$TRAIN/checkpoint.pt"     --output-dir "$EXPORT"

echo "== compare initialized ONNX =="
"$ROOT/scripts/python-apps.sh"     "$ROOT/examples/speech-asr/evaluation/compare_cnn_ctc_v3_onnx.py"     "$EXPORT"     --output "$EXPORT/onnx-comparison.json"

echo "== convert initialized graph with OpenVINO 2020.3 =="
"$ROOT/scripts/prepare-cnn-ctc-v2.sh"     --onnx "$EXPORT/cnn_ctc_v3.onnx"     --output-dir "$IR"

echo "cnn_ctc_v3 compatibility graph gate: PASS"
