#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/work/speech-asr/pretrained/quartznet15x5-en-base-v2/QuartzNet15x5-En-Base.nemo"
VERIFY="$ROOT/work/speech-asr/pretrained/quartznet15x5-en-base-v2/import-verification.json"

"$ROOT/scripts/python.sh" \
  "$ROOT/examples/speech-asr/training/prepare_cnn_ctc_v18_pretrained.py" \
  "$OUT" "$@"

"$ROOT/scripts/python-training.sh" \
  "$ROOT/examples/speech-asr/training/verify_cnn_ctc_v18_pretrained.py" \
  "$OUT" \
  --output "$VERIFY"

echo "cnn_ctc_v18 pretrained source + import verification: PASS"
echo "source: $OUT"
echo "import: $VERIFY"
