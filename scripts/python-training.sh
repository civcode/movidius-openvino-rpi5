#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/prepare-python-env.sh" training >/dev/null
exec "$ROOT/work/venv-training/bin/python" "$@"
