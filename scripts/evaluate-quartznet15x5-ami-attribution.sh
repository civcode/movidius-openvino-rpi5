#!/usr/bin/env bash
# Attribute AMI quality loss across source frontend -> fixed frontend -> AMI head -> v19 fine-tuning.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/python-training.sh"   "$ROOT/examples/speech-asr/evaluation/evaluate_quartznet15x5_ami_attribution.py" "$@"
