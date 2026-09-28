#!/usr/bin/env bash
# Compatibility wrapper. Runtime/backend logic lives in scripts/run-inference-server.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for arg in "$@"; do
    [[ "$arg" == -h || "$arg" == --help ]] && exec "$ROOT/scripts/run-inference-server.sh" --help
done
BACKEND="${1:-auto}"
IR="${2:-${IR:-fp16}}"
DEVICE="${3:-MYRIAD}"
exec "$ROOT/scripts/run-inference-server.sh" \
    --app mobilenet --backend "$BACKEND" --ir "$IR" --device "$DEVICE"
