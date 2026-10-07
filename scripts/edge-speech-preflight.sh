#!/usr/bin/env bash
# Hardware-free edge worker preflight for one declared benchmark manifest.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MANIFEST_REL=""
MANIFEST_SHA=""
RUNTIME_BACKEND="docker"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) [[ $# -ge 2 ]] || { echo "--manifest needs a path" >&2; exit 2; }; MANIFEST_REL="$2"; shift 2 ;;
        --manifest-sha256) [[ $# -ge 2 ]] || { echo "--manifest-sha256 needs a digest" >&2; exit 2; }; MANIFEST_SHA="$2"; shift 2 ;;
        --runtime-backend) [[ $# -ge 2 ]] || { echo "--runtime-backend needs auto|host|docker" >&2; exit 2; }; RUNTIME_BACKEND="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 --manifest repo-relative-path --manifest-sha256 digest [--runtime-backend auto|host|docker]"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$MANIFEST_REL" && -n "$MANIFEST_SHA" ]] || {
    echo "manifest path and sha256 are required" >&2
    exit 2
}
[[ "$MANIFEST_SHA" =~ ^[0-9a-f]{64}$ ]] || {
    echo "invalid manifest SHA-256" >&2
    exit 2
}
[[ "$MANIFEST_REL" != /* && "$MANIFEST_REL" != *".."* ]] || {
    echo "manifest must be a safe repository-relative path" >&2
    exit 2
}

case "$(uname -m)" in
    aarch64|arm64) ;;
    *) echo "edge speech worker requires native arm64/aarch64 host" >&2; exit 2 ;;
esac

command -v flock >/dev/null 2>&1 || {
    echo "flock is required for exclusive MYRIAD execution" >&2
    exit 2
}

# Non-interactive SSH sessions may not source the user's interactive PATH.
# Exercise the same resolver used by scripts/python.sh before expensive work.
# shellcheck source=scripts/lib/python-env.sh
source "$ROOT/scripts/lib/python-env.sh"
python_env_require_uv || exit $?

manifest="$ROOT/$MANIFEST_REL"
[[ -f "$manifest" ]] || {
    echo "declared dataset manifest missing: $manifest" >&2
    exit 2
}
actual="$(sha256sum "$manifest" | awk '{print $1}')"
[[ "$actual" == "$MANIFEST_SHA" ]] || {
    echo "dataset manifest hash mismatch: $actual != $MANIFEST_SHA" >&2
    exit 2
}

runtime_check="$("$ROOT/scripts/run-myriad-tensor.sh"     --platform arm64     --backend "$RUNTIME_BACKEND"     check)"
printf '%s\n' "$runtime_check"

echo "edge speech preflight: PASS"
echo "manifest_sha256=$actual"
