#!/usr/bin/env bash
# Serialize access to the single physical MYRIAD device and execute the edge worker.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_PATH="${SPEECH_MYRIAD_LOCK:-/tmp/movidius-speech-asr-myriad.lock}"

command -v flock >/dev/null 2>&1 || {
    echo "flock is required for exclusive MYRIAD execution" >&2
    exit 2
}
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo '{"status":"blocked","failure_class":"worker_busy"}'
    exit 75
fi

exec "$ROOT/scripts/python.sh"     "$ROOT/examples/speech-asr/agent/edge_worker.py" "$@"
