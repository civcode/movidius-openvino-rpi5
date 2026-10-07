#!/usr/bin/env bash
# Zero-training LibriSpeech dev-clean qualification for the pinned NVIDIA QuartzNet source.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/work/speech-asr/pretrained/quartznet15x5-en-base-v2/QuartzNet15x5-En-Base.nemo"
DEVICE="auto"
MAX_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) [[ $# -ge 2 ]] || { echo "--device needs auto|cpu|cuda" >&2; exit 2; }; DEVICE="$2"; shift 2 ;;
        --max-samples) [[ $# -ge 2 ]] || { echo "--max-samples needs a positive integer" >&2; exit 2; }; MAX_SAMPLES="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--device auto|cpu|cuda] [--max-samples N]"
            echo "Runs source preparation, LibriSpeech dev-clean preparation, zero-training PyTorch WER, and ONNX parity."
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$DEVICE" in auto|cpu|cuda) ;; *) echo "invalid --device: $DEVICE" >&2; exit 2 ;; esac
if [[ -n "$MAX_SAMPLES" && ! "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid --max-samples: $MAX_SAMPLES" >&2
    exit 2
fi

"$ROOT/scripts/python.sh"   "$ROOT/examples/speech-asr/training/prepare_cnn_ctc_v18_pretrained.py"   "$SOURCE"

"$ROOT/scripts/prepare-librispeech-dev-clean.sh"

eval_args=(--archive "$SOURCE" --device "$DEVICE")
[[ -n "$MAX_SAMPLES" ]] && eval_args+=(--max-samples "$MAX_SAMPLES")
"$ROOT/scripts/evaluate-quartznet15x5-reference.sh" "${eval_args[@]}"

"$ROOT/scripts/export-quartznet15x5-reference.sh" --archive "$SOURCE"
"$ROOT/scripts/compare-quartznet15x5-reference-onnx.sh"   --output "$ROOT/work/speech-asr/quartznet15x5-reference/onnx-parity.json"

echo "QuartzNet15x5 pretrained reference qualification: complete"
echo "No training, OpenVINO conversion, or MYRIAD execution was performed."
