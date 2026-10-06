#!/usr/bin/env bash
# Provision the fresh model-quality-v4 validation boundary to the edge host.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="edge"
EDGE_BASE_REPO=""
EDGE_WORKER_REPO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --worker) [[ $# -ge 2 ]] || { echo "--worker needs a value" >&2; exit 2; }; WORKER="$2"; shift 2 ;;
        --edge-base-repo) [[ $# -ge 2 ]] || { echo "--edge-base-repo needs a path" >&2; exit 2; }; EDGE_BASE_REPO="$2"; shift 2 ;;
        --edge-worker-repo) [[ $# -ge 2 ]] || { echo "--edge-worker-repo needs a path" >&2; exit 2; }; EDGE_WORKER_REPO="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--worker edge] [--edge-base-repo ABS] [--edge-worker-repo ABS]"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ "$WORKER" =~ ^[A-Za-z0-9_.@-]+$ && "$WORKER" != -* ]] || {
    echo "invalid worker alias: $WORKER" >&2
    exit 2
}

"$ROOT/scripts/prepare-speech-model-data-v4.sh" --verify-only >/dev/null

AMI="$ROOT/work/speech-asr/ami"
LOCK_DIR="$AMI/model-quality-v4-source-lock"
VALIDATION_DIR="$AMI/ami-model-quality-v4-validation-v1"
QUALITY_DIR="$AMI/model-quality-v4"
QUALIFICATION="$QUALITY_DIR/qualification.json"

read -r TRAIN_SHA VALIDATION_SHA <<<"$(
    "$ROOT/scripts/python.sh" -c '
import json, pathlib, sys
q=json.loads(pathlib.Path(sys.argv[1]).read_text())
print(q["train"]["manifest_sha256"], q["validation"]["manifest_sha256"])
' "$QUALIFICATION"
)"

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
REMOTE_HOME="$("${SSH[@]}" "$WORKER" 'printf "%s\n" "$HOME"' | awk 'NF{last=$0} END{print last}')"
[[ "$REMOTE_HOME" == /* ]] || {
    echo "could not resolve absolute remote HOME for $WORKER" >&2
    exit 2
}

if [[ -z "$EDGE_BASE_REPO" ]]; then
    EDGE_BASE_REPO="$REMOTE_HOME/workspace/movidius-openvino-rpi5"
fi
if [[ -z "$EDGE_WORKER_REPO" ]]; then
    EDGE_WORKER_REPO="$REMOTE_HOME/workspace/movidius-openvino-rpi5-worker"
fi
for path in "$EDGE_BASE_REPO" "$EDGE_WORKER_REPO"; do
    [[ "$path" == /* && "$path" != *[[:space:]]* ]] || {
        echo "edge repository paths must be absolute and whitespace-free: $path" >&2
        exit 2
    }
done

COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
    echo "could not resolve local full Git commit" >&2
    exit 2
}

"${SSH[@]}" "$WORKER" mkdir -p "$EDGE_BASE_REPO/work/speech-asr/ami"
"${SSH[@]}" "$WORKER" bash -s --     "$EDGE_BASE_REPO" "$EDGE_WORKER_REPO" "$COMMIT"     < "$ROOT/scripts/edge-speech-bootstrap.sh"

rsync -a --delete --checksum -e "$RSYNC_RSH"     "$LOCK_DIR/"     "$WORKER:$EDGE_BASE_REPO/work/speech-asr/ami/model-quality-v4-source-lock/"
rsync -a --delete --checksum -e "$RSYNC_RSH"     "$VALIDATION_DIR/"     "$WORKER:$EDGE_BASE_REPO/work/speech-asr/ami/ami-model-quality-v4-validation-v1/"
rsync -a --delete --checksum -e "$RSYNC_RSH"     "$QUALITY_DIR/"     "$WORKER:$EDGE_BASE_REPO/work/speech-asr/ami/model-quality-v4/"

"${SSH[@]}" "$WORKER" bash -s --     "$EDGE_BASE_REPO" "$EDGE_WORKER_REPO" "$TRAIN_SHA" "$VALIDATION_SHA" <<'REMOTE'
set -euo pipefail
BASE_REPO="$1"
WORKER_REPO="$2"
TRAIN_SHA="$3"
VALIDATION_SHA="$4"

worker_ami="$WORKER_REPO/work/speech-asr/ami"
base_ami="$BASE_REPO/work/speech-asr/ami"
if [[ -L "$worker_ami" ]]; then
    [[ "$(readlink -f "$worker_ami")" == "$(readlink -f "$base_ami")" ]] || {
        echo "automation worktree AMI symlink points at unexpected dataset tree" >&2
        exit 2
    }
elif [[ ! -e "$worker_ami" ]]; then
    mkdir -p "$(dirname "$worker_ami")"
    ln -s "$base_ami" "$worker_ami"
else
    echo "automation worktree AMI path is not the expected symlink: $worker_ami" >&2
    exit 2
fi

"$WORKER_REPO/scripts/python.sh"     "$WORKER_REPO/examples/speech-asr/datasets/ami/prepare_ami.py"     --spec "$base_ami/model-quality-v4-source-lock/validation.split.json"     --output "$base_ami/ami-model-quality-v4-validation-v1"     --verify-only >/dev/null

validation="$base_ami/model-quality-v4/validation.manifest.jsonl"
[[ "$(sha256sum "$validation" | awk '{print $1}')" == "$VALIDATION_SHA" ]] || {
    echo "edge model-quality-v4 validation manifest hash mismatch" >&2
    exit 2
}

printf 'edge model-quality-v4 validation: PASS\n'
printf 'train_manifest_sha256=%s\n' "$TRAIN_SHA"
printf 'validation_manifest_sha256=%s\n' "$VALIDATION_SHA"
REMOTE

printf '{"status":"provisioned","worker":"%s","commit":"%s","train_manifest_sha256":"%s","validation_manifest_sha256":"%s"}\n'     "$WORKER" "$COMMIT" "$TRAIN_SHA" "$VALIDATION_SHA"
