#!/usr/bin/env bash
# Hardware-free edge worker preflight for one declared benchmark manifest.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MANIFEST_REL=""
MANIFEST_SHA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) [[ $# -ge 2 ]] || { echo "--manifest needs a path" >&2; exit 2; }; MANIFEST_REL="$2"; shift 2 ;;
        --manifest-sha256) [[ $# -ge 2 ]] || { echo "--manifest-sha256 needs a digest" >&2; exit 2; }; MANIFEST_SHA="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 --manifest repo-relative-path --manifest-sha256 digest"
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

help_log="$(mktemp)"
trap 'rm -f "$help_log"' EXIT
set +e
"$ROOT/run.sh" --platform arm64 custom --help >"$help_log" 2>&1
status=$?
set -e
if (( status != 0 )) || ! grep -q -- '--tensor' "$help_log" || ! grep -q -- '--output' "$help_log"; then
    cat "$help_log" >&2
    echo "arm64 runtime image lacks required tensor I/O support" >&2
    exit 2
fi

echo "edge speech preflight: PASS"
echo "manifest_sha256=$actual"
