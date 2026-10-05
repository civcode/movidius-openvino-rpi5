#!/usr/bin/env bash
# Shared uv-managed host Python environment helpers. Safe to source.

python_env_resolve_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi

    local candidate
    local -a candidates=()
    [[ -n "${UV_INSTALL_DIR:-}" ]] && candidates+=("${UV_INSTALL_DIR}/uv")
    [[ -n "${XDG_BIN_HOME:-}" ]] && candidates+=("${XDG_BIN_HOME}/uv")
    [[ -n "${HOME:-}" ]] && candidates+=(
        "${HOME}/.local/bin/uv"
        "${HOME}/.cargo/bin/uv"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            PATH="$(dirname "$candidate"):$PATH"
            export PATH
            command -v uv >/dev/null 2>&1 && return 0
        fi
    done
    return 1
}

python_env_require_uv() {
    if ! python_env_resolve_uv; then
        echo "uv is required for host Python environments." >&2
        echo "Install the uv binary, or ensure it is available in PATH," >&2
        echo "UV_INSTALL_DIR, XDG_BIN_HOME, ~/.local/bin, or ~/.cargo/bin." >&2
        echo "See docs/PYTHON-ENVIRONMENTS.md." >&2
        return 127
    fi
}

python_env_matches_version() {
    local venv="$1" request="$2"
    [[ -x "$venv/bin/python" ]] || return 1
    "$venv/bin/python" - "$request" <<PY >/dev/null 2>&1
import sys
req = sys.argv[1]
major, minor = (int(part) for part in req.split(".", 1))
raise SystemExit(0 if sys.version_info[:2] == (major, minor) else 1)
PY
}

python_env_ensure_venv() {
    local venv="$1" python_request="$2"
    python_env_require_uv || return
    if ! python_env_matches_version "$venv" "$python_request"; then
        rm -rf "$venv"
        mkdir -p "$(dirname "$venv")"
        uv venv --python "$python_request" "$venv"
    fi
}

python_env_ensure_requirements() {
    local venv="$1" python_request="$2" requirements="$3" imports="$4"
    python_env_ensure_venv "$venv" "$python_request" || return
    if ! "$venv/bin/python" -c "$imports" >/dev/null 2>&1; then
        echo "syncing Python dependencies into $venv with uv" >&2
        uv pip install --python "$venv/bin/python" -r "$requirements"
    fi
}
