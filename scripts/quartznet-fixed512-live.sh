#!/usr/bin/env bash
# Live fixed-T=512 QuartzNet ASR on Pi 5 + MA2450.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPUSET="${SPEECH_LIVE_CPUSET:-0-3}"
THREADS="${SPEECH_LIVE_THREADS:-4}"

[[ "$CPUSET" =~ ^[0-9,-]+$ ]] || {
    echo "invalid SPEECH_LIVE_CPUSET: $CPUSET" >&2
    exit 2
}
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid SPEECH_LIVE_THREADS: $THREADS" >&2
    exit 2
}
command -v taskset >/dev/null 2>&1 || {
    echo "taskset is required" >&2
    exit 2
}

exec env     OMP_NUM_THREADS="$THREADS"     MKL_NUM_THREADS="$THREADS"     OPENBLAS_NUM_THREADS="$THREADS"     taskset -c "$CPUSET"     "$ROOT/scripts/python-apps.sh"     "$ROOT/examples/speech-asr/runtime/quartznet_fixed512_live.py"     "$@"
