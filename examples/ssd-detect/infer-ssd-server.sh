#!/usr/bin/env bash
# Compatibility wrapper. Runtime/backend logic lives in scripts/run-inference-server.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for arg in "$@"; do
    [[ "$arg" == -h || "$arg" == --help ]] && exec "$ROOT/scripts/run-inference-server.sh" --help
done
BACKEND="${1:-auto}"
DEVICE="${2:-MYRIAD}"
MIN_CONF="${3:-0.5}"
exec "$ROOT/scripts/run-inference-server.sh" \
    --app ssd --backend "$BACKEND" --device "$DEVICE" --min-conf "$MIN_CONF"
