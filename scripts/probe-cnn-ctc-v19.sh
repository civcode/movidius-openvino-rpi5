#!/usr/bin/env bash
# Phase 11 cheap graph gate: initialized cnn_ctc_v19 -> ONNX -> OpenVINO IR.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$ROOT/work/speech-asr/cnn_ctc_v19/compatibility-probe"
PRETRAINED="$ROOT/work/speech-asr/pretrained/quartznet15x5-en-base-v2/QuartzNet15x5-En-Base.nemo"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained) [[ $# -ge 2 ]] || { echo "--pretrained needs a path" >&2; exit 2; }; PRETRAINED="$2"; shift 2 ;;
        --work-dir) [[ $# -ge 2 ]] || { echo "--work-dir needs a path" >&2; exit 2; }; WORK_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--work-dir path]"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -s "$PRETRAINED" ]] || {
    echo "pretrained QuartzNet archive missing: $PRETRAINED" >&2
    echo "run ./scripts/prepare-cnn-ctc-v19-pretrained.sh first" >&2
    exit 2
}
TRAIN="$WORK_DIR/training-init"
EXPORT="$WORK_DIR/export"
IR="$WORK_DIR/openvino/fp16"
mkdir -p "$WORK_DIR"

echo "== initialize cnn_ctc_v19 =="
"$ROOT/scripts/python-training.sh"     "$ROOT/examples/speech-asr/training/train_cnn_ctc_v19.py"     --output-dir "$TRAIN"     --device cpu     --pretrained "$PRETRAINED"     --init-only

echo "== export initialized ONNX =="
"$ROOT/scripts/python-training.sh"     "$ROOT/examples/speech-asr/training/export_cnn_ctc_v19.py"     --checkpoint "$TRAIN/checkpoint.pt"     --output-dir "$EXPORT"

echo "== compare initialized ONNX =="
"$ROOT/scripts/python-apps.sh"     "$ROOT/examples/speech-asr/evaluation/compare_cnn_ctc_v19_onnx.py"     "$EXPORT"     --output "$EXPORT/onnx-comparison.json"

echo "== convert initialized graph with OpenVINO 2020.3 =="
"$ROOT/scripts/prepare-cnn-ctc-v19.sh"     --onnx "$EXPORT/cnn_ctc_v19.onnx"     --output-dir "$IR"

test -s "$IR/cnn_ctc_v19.xml"
test -s "$IR/cnn_ctc_v19.bin"
test -s "$IR/ir-validation.json"
test -s "$IR/artifacts.json"

echo "cnn_ctc_v19 compatibility graph gate: PASS"
