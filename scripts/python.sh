#!/usr/bin/env bash
# Run dependency-free repository Python tooling in a uv-managed environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lib/python-env.sh"
VENV="$ROOT/work/venv-tools"
python_env_ensure_venv "$VENV" 3.11
exec "$VENV/bin/python" "$@"
