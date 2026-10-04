#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/prepare-python-env.sh" apps >/dev/null
exec "$ROOT/work/venv-apps/bin/python" "$@"
