#!/usr/bin/env bash
# Live fixed-T=512 QuartzNet ASR on Pi 5 + MA2450.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPUSET="${SPEECH_LIVE_CPUSET:-0-3}"
THREADS="${SPEECH_LIVE_THREADS:-4}"
LOCK_PATH="${SPEECH_MYRIAD_LOCK:-/tmp/movidius-speech-asr-myriad.lock}"

[[ "$CPUSET" =~ ^[0-9,-]+$ ]] || {
    echo "invalid SPEECH_LIVE_CPUSET: $CPUSET" >&2
    exit 2
}
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid SPEECH_LIVE_THREADS: $THREADS" >&2
    exit 2
}
for command in taskset flock; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "$command is required" >&2
        exit 2
    }
done

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo "MYRIAD is busy: lock held at $LOCK_PATH" >&2
    exit 75
fi

exec env \
    OMP_NUM_THREADS="$THREADS" \
    MKL_NUM_THREADS="$THREADS" \
    OPENBLAS_NUM_THREADS="$THREADS" \
    taskset -c "$CPUSET" \
    "$ROOT/scripts/python-apps.sh" \
    "$ROOT/examples/speech-asr/runtime/quartznet_fixed512_live.py" \
    "$@"
