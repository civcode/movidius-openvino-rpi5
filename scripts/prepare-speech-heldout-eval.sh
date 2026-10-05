#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC="$ROOT/examples/speech-asr/datasets/ami/splits/eval-full-corpus-asr-sc-v1.json"

"$ROOT/scripts/python.sh"   "$ROOT/examples/speech-asr/datasets/ami/prepare_ami.py"   --spec "$SPEC"

exec "$ROOT/scripts/qualify-speech-heldout-eval.sh"
