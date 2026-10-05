#!/usr/bin/env bash
# Compile and run one staged fixed-shape model on the physical MYRIAD device.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_PATH="${SPEECH_MYRIAD_LOCK:-/tmp/movidius-speech-asr-myriad.lock}"
MODEL=""
WEIGHTS=""
TENSOR=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --weights) WEIGHTS="$2"; shift 2 ;;
        --tensor) TENSOR="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

for value in "$MODEL" "$WEIGHTS" "$TENSOR" "$OUTPUT"; do
    [[ "$value" == "$ROOT/work/"* ]] || {
        echo "probe paths must be below worker repository work/: $value" >&2
        exit 2
    }
done
for value in "$MODEL" "$WEIGHTS" "$TENSOR"; do
    [[ -f "$value" ]] || { echo "probe input missing: $value" >&2; exit 2; }
done

command -v flock >/dev/null 2>&1 || {
    echo "flock is required for exclusive MYRIAD execution" >&2
    exit 2
}
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo '{"status":"blocked","failure_class":"worker_busy"}'
    exit 75
fi

container_path() {
    local value="$1"
    printf '/work/%s' "${value#"$ROOT/work/"}"
}

mkdir -p "$(dirname "$OUTPUT")"
"$ROOT/run.sh" --platform arm64 custom     --model "$(container_path "$MODEL")"     --weights "$(container_path "$WEIGHTS")"     --tensor "$(container_path "$TENSOR")"     --output "$(container_path "$OUTPUT")"     --iterations 1

test -s "$OUTPUT"
echo "physical compatibility probe: PASS"
