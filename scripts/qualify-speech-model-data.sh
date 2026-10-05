#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/python.sh"     "$ROOT/examples/speech-asr/tools/qualify_model_quality_manifests.py" "$@"
