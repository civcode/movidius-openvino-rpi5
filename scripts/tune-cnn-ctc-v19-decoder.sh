#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The sweep parallelizes across utterances with --jobs. Keep numerical-library
# thread pools at one thread per worker to avoid 16x16 oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

exec "$ROOT/scripts/python-training.sh" \
  "$ROOT/examples/speech-asr/evaluation/tune_cnn_ctc_v19_decoder.py" "$@"
