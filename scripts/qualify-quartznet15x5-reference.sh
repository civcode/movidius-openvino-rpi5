#!/usr/bin/env bash
# Zero-training LibriSpeech dev-clean qualification for the pinned NVIDIA QuartzNet source.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/work/speech-asr/pretrained/quartznet15x5-en-base-v2/QuartzNet15x5-En-Base.nemo"
DEVICE="auto"
DEVICE_INDEX=0
MAX_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) [[ $# -ge 2 ]] || { echo "--device needs auto|cpu|cuda|rocm" >&2; exit 2; }; DEVICE="$2"; shift 2 ;;
        --device-index) [[ $# -ge 2 ]] || { echo "--device-index needs a non-negative integer" >&2; exit 2; }; DEVICE_INDEX="$2"; shift 2 ;;
        --max-samples) [[ $# -ge 2 ]] || { echo "--max-samples needs a positive integer" >&2; exit 2; }; MAX_SAMPLES="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--device auto|cpu|cuda|rocm] [--device-index N] [--max-samples N]"
            echo "Runs source preparation, LibriSpeech dev-clean preparation, zero-training PyTorch WER, and ONNX parity."
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$DEVICE" in auto|cpu|cuda|rocm) ;; *) echo "invalid --device: $DEVICE" >&2; exit 2 ;; esac
[[ "$DEVICE_INDEX" =~ ^[0-9]+$ ]] || { echo "invalid --device-index: $DEVICE_INDEX" >&2; exit 2; }
if [[ -n "$MAX_SAMPLES" && ! "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid --max-samples: $MAX_SAMPLES" >&2
    exit 2
fi

"$ROOT/scripts/python.sh"   "$ROOT/examples/speech-asr/training/prepare_cnn_ctc_v18_pretrained.py"   "$SOURCE"

"$ROOT/scripts/prepare-librispeech-dev-clean.sh"

eval_args=(--archive "$SOURCE" --device "$DEVICE" --device-index "$DEVICE_INDEX")
[[ -n "$MAX_SAMPLES" ]] && eval_args+=(--max-samples "$MAX_SAMPLES")

# A full-corpus WER miss is qualification evidence, not a reason to discard the
# independent PyTorch -> ONNX parity evidence. Preserve the evaluator status,
# finish all reference-only checks, then return the WER gate result.
set +e
"$ROOT/scripts/evaluate-quartznet15x5-reference.sh" "${eval_args[@]}"
eval_status=$?
set -e
if [[ "$eval_status" -ne 0 && "$eval_status" -ne 3 ]]; then
    exit "$eval_status"
fi

"$ROOT/scripts/export-quartznet15x5-reference.sh" --archive "$SOURCE"
"$ROOT/scripts/compare-quartznet15x5-reference-onnx.sh"   --output "$ROOT/work/speech-asr/quartznet15x5-reference/onnx-parity.json"

echo "QuartzNet15x5 pretrained reference qualification: complete"
echo "No training, OpenVINO conversion, or MYRIAD execution was performed."
if [[ "$eval_status" -eq 3 ]]; then
    echo "LibriSpeech dev-clean WER reproduction gate: FAIL" >&2
fi
exit "$eval_status"
