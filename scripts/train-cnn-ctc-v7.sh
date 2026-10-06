#!/usr/bin/env bash
# Train cnn_ctc_v7, export ONNX, compare reference output, and convert FP16 IR.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/work/speech-asr/ami/ami-smoke-v1/manifest.jsonl"
VALIDATION=""
DEVICE=cuda
EPOCHS=""
BATCH_SIZE=""
SEED=""
LEARNING_RATE=""
MAX_SAMPLES=""
CHECKPOINT_SELECTION=""
CTC_OBJECTIVE_KIND=""
BLANK_LOGIT_PENALTY=""
AUGMENTATION_KIND=""
FREQUENCY_MASKS=""
FREQUENCY_MAX_WIDTH=""
TIME_MASKS=""
TIME_MAX_WIDTH=""
TIME_MAX_FRACTION=""
AUGMENTATION_MASK_VALUE=""
AUGMENTATION_SEED_OFFSET=""
WORK_DIR="$ROOT/work/speech-asr/cnn_ctc_v7"
TRAIN="$WORK_DIR/training"
EXPORT="$WORK_DIR/export"
IR="$WORK_DIR/openvino/fp16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) [[ $# -ge 2 ]] || { echo "--manifest needs a path" >&2; exit 2; }; MANIFEST="$2"; shift 2 ;;
        --validation-manifest) [[ $# -ge 2 ]] || { echo "--validation-manifest needs a path" >&2; exit 2; }; VALIDATION="$2"; shift 2 ;;
        --device) [[ $# -ge 2 ]] || { echo "--device needs cuda|cpu|auto" >&2; exit 2; }; DEVICE="$2"; shift 2 ;;
        --epochs) [[ $# -ge 2 ]] || { echo "--epochs needs an integer" >&2; exit 2; }; EPOCHS="$2"; shift 2 ;;
        --batch-size) [[ $# -ge 2 ]] || { echo "--batch-size needs an integer" >&2; exit 2; }; BATCH_SIZE="$2"; shift 2 ;;
        --seed) [[ $# -ge 2 ]] || { echo "--seed needs an integer" >&2; exit 2; }; SEED="$2"; shift 2 ;;
        --learning-rate) [[ $# -ge 2 ]] || { echo "--learning-rate needs a number" >&2; exit 2; }; LEARNING_RATE="$2"; shift 2 ;;
        --max-samples) [[ $# -ge 2 ]] || { echo "--max-samples needs an integer" >&2; exit 2; }; MAX_SAMPLES="$2"; shift 2 ;;
        --checkpoint-selection) [[ $# -ge 2 ]] || { echo "--checkpoint-selection needs validation_loss|validation_cer" >&2; exit 2; }; CHECKPOINT_SELECTION="$2"; shift 2 ;;
        --ctc-objective-kind) [[ $# -ge 2 ]] || { echo "--ctc-objective-kind needs a value" >&2; exit 2; }; CTC_OBJECTIVE_KIND="$2"; shift 2 ;;
        --blank-logit-penalty) [[ $# -ge 2 ]] || { echo "--blank-logit-penalty needs a number" >&2; exit 2; }; BLANK_LOGIT_PENALTY="$2"; shift 2 ;;
        --augmentation-kind) [[ $# -ge 2 ]] || { echo "--augmentation-kind needs a value" >&2; exit 2; }; AUGMENTATION_KIND="$2"; shift 2 ;;
        --frequency-masks) [[ $# -ge 2 ]] || { echo "--frequency-masks needs an integer" >&2; exit 2; }; FREQUENCY_MASKS="$2"; shift 2 ;;
        --frequency-max-width) [[ $# -ge 2 ]] || { echo "--frequency-max-width needs an integer" >&2; exit 2; }; FREQUENCY_MAX_WIDTH="$2"; shift 2 ;;
        --time-masks) [[ $# -ge 2 ]] || { echo "--time-masks needs an integer" >&2; exit 2; }; TIME_MASKS="$2"; shift 2 ;;
        --time-max-width) [[ $# -ge 2 ]] || { echo "--time-max-width needs an integer" >&2; exit 2; }; TIME_MAX_WIDTH="$2"; shift 2 ;;
        --time-max-fraction) [[ $# -ge 2 ]] || { echo "--time-max-fraction needs a number" >&2; exit 2; }; TIME_MAX_FRACTION="$2"; shift 2 ;;
        --augmentation-mask-value) [[ $# -ge 2 ]] || { echo "--augmentation-mask-value needs a number" >&2; exit 2; }; AUGMENTATION_MASK_VALUE="$2"; shift 2 ;;
        --augmentation-seed-offset) [[ $# -ge 2 ]] || { echo "--augmentation-seed-offset needs an integer" >&2; exit 2; }; AUGMENTATION_SEED_OFFSET="$2"; shift 2 ;;
        --work-dir) [[ $# -ge 2 ]] || { echo "--work-dir needs a path" >&2; exit 2; }; WORK_DIR="$2"; TRAIN="$WORK_DIR/training"; EXPORT="$WORK_DIR/export"; IR="$WORK_DIR/openvino/fp16"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--manifest path] [--validation-manifest path] [--device cuda|cpu|auto] [--epochs N] [--batch-size N] [--seed N] [--learning-rate RATE] [--max-samples N] [--checkpoint-selection validation_loss|validation_cer] [--augmentation-kind specaugment-v1 ...] [--work-dir path]"
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
[[ -n "$BATCH_SIZE" ]] && args+=(--batch-size "$BATCH_SIZE")
[[ -n "$SEED" ]] && args+=(--seed "$SEED")
[[ -n "$LEARNING_RATE" ]] && args+=(--learning-rate "$LEARNING_RATE")
[[ -n "$MAX_SAMPLES" ]] && args+=(--max-samples "$MAX_SAMPLES")
[[ -n "$CHECKPOINT_SELECTION" ]] && args+=(--checkpoint-selection "$CHECKPOINT_SELECTION")
if [[ -n "$CTC_OBJECTIVE_KIND" ]]; then
    [[ -n "$BLANK_LOGIT_PENALTY" ]] || { echo "--ctc-objective-kind requires --blank-logit-penalty" >&2; exit 2; }
    args+=(--ctc-objective-kind "$CTC_OBJECTIVE_KIND" --blank-logit-penalty "$BLANK_LOGIT_PENALTY")
elif [[ -n "$BLANK_LOGIT_PENALTY" ]]; then
    echo "--blank-logit-penalty requires --ctc-objective-kind" >&2
    exit 2
fi
if [[ -n "$AUGMENTATION_KIND" ]]; then
    args+=(
        --augmentation-kind "$AUGMENTATION_KIND"
        --frequency-masks "$FREQUENCY_MASKS"
        --frequency-max-width "$FREQUENCY_MAX_WIDTH"
        --time-masks "$TIME_MASKS"
        --time-max-width "$TIME_MAX_WIDTH"
        --time-max-fraction "$TIME_MAX_FRACTION"
        --augmentation-mask-value "$AUGMENTATION_MASK_VALUE"
        --augmentation-seed-offset "$AUGMENTATION_SEED_OFFSET"
    )
fi

echo "== train cnn_ctc_v7 =="
"$ROOT/scripts/python-training.sh"     "$ROOT/examples/speech-asr/training/train_cnn_ctc_v7.py"     "${args[@]}"

echo "== export fixed-shape ONNX =="
"$ROOT/scripts/python-training.sh"     "$ROOT/examples/speech-asr/training/export_cnn_ctc_v7.py"     --checkpoint "$TRAIN/checkpoint.pt"     --output-dir "$EXPORT"

echo "== compare ONNX Runtime vs PyTorch golden =="
"$ROOT/scripts/python-apps.sh"     "$ROOT/examples/speech-asr/evaluation/compare_cnn_ctc_v7_onnx.py"     "$EXPORT"     --output "$EXPORT/onnx-comparison.json"

echo "== convert OpenVINO 2020.3 FP16 IR =="
"$ROOT/scripts/prepare-cnn-ctc-v7.sh"     --onnx "$EXPORT/cnn_ctc_v7.onnx"     --output-dir "$IR"

test -s "$IR/cnn_ctc_v7.xml"
test -s "$IR/cnn_ctc_v7.bin"
test -s "$IR/ir-validation.json"
test -s "$IR/artifacts.json"

echo "training/export pipeline: PASS"
echo "checkpoint: $TRAIN/checkpoint.pt"
echo "onnx:       $EXPORT/cnn_ctc_v7.onnx"
echo "ir:         $IR/cnn_ctc_v7.xml"
