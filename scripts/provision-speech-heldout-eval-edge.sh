#!/usr/bin/env bash
# Provision the frozen AMI held-out ASR evaluation corpus to edge.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="edge"
EDGE_BASE_REPO=""
EDGE_WORKER_REPO=""

SPLIT_SPEC="$ROOT/examples/speech-asr/datasets/ami/splits/eval-full-corpus-asr-sc-v1.json"
SOURCE_DIR="$ROOT/work/speech-asr/ami/ami-eval-full-corpus-asr-sc-v1"
HELDOUT_DIR="$ROOT/work/speech-asr/ami/heldout-eval-v1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --worker)
            [[ $# -ge 2 ]] || { echo "--worker needs a value" >&2; exit 2; }
            WORKER="$2"; shift 2 ;;
        --edge-base-repo)
            [[ $# -ge 2 ]] || { echo "--edge-base-repo needs a path" >&2; exit 2; }
            EDGE_BASE_REPO="$2"; shift 2 ;;
        --edge-worker-repo)
            [[ $# -ge 2 ]] || { echo "--edge-worker-repo needs a path" >&2; exit 2; }
            EDGE_WORKER_REPO="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--worker edge] [--edge-base-repo ABS] [--edge-worker-repo ABS]"
            exit 0 ;;
        *)
            echo "unknown option: $1" >&2
            exit 2 ;;
    esac
done

[[ "$WORKER" =~ ^[A-Za-z0-9_.@-]+$ && "$WORKER" != -* ]] || {
    echo "invalid worker alias: $WORKER" >&2
    exit 2
}

"$ROOT/scripts/python.sh"     "$ROOT/examples/speech-asr/datasets/ami/prepare_ami.py"     --spec "$SPLIT_SPEC"     --verify-only >/dev/null

"$ROOT/scripts/qualify-speech-heldout-eval.sh" --verify-only >/dev/null

for path in     "$SOURCE_DIR/manifest.jsonl"     "$SOURCE_DIR/provenance.json"     "$HELDOUT_DIR/manifest.jsonl"     "$HELDOUT_DIR/qualification.json"; do
    [[ -f "$path" ]] || {
        echo "required held-out artifact missing: $path" >&2
        exit 2
    }
done

HELDOUT_SHA="$(sha256sum "$HELDOUT_DIR/manifest.jsonl" | awk '{print $1}')"
SOURCE_SHA="$(sha256sum "$SOURCE_DIR/manifest.jsonl" | awk '{print $1}')"

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

rsync -a --delete --checksum -e "$RSYNC_RSH"     "$SOURCE_DIR/"     "$WORKER:$EDGE_BASE_REPO/work/speech-asr/ami/ami-eval-full-corpus-asr-sc-v1/"

rsync -a --delete --checksum -e "$RSYNC_RSH"     "$HELDOUT_DIR/"     "$WORKER:$EDGE_BASE_REPO/work/speech-asr/ami/heldout-eval-v1/"

"${SSH[@]}" "$WORKER" bash -s --     "$EDGE_BASE_REPO" "$EDGE_WORKER_REPO" "$SOURCE_SHA" "$HELDOUT_SHA" <<'REMOTE'
set -euo pipefail
BASE_REPO="$1"
WORKER_REPO="$2"
SOURCE_SHA="$3"
HELDOUT_SHA="$4"

base_ami="$BASE_REPO/work/speech-asr/ami"
worker_ami="$WORKER_REPO/work/speech-asr/ami"
if [[ -L "$worker_ami" ]]; then
    [[ "$(readlink -f "$worker_ami")" == "$(readlink -f "$base_ami")" ]] || {
        echo "worker AMI symlink points at an unexpected dataset tree" >&2
        exit 2
    }
elif [[ ! -e "$worker_ami" ]]; then
    mkdir -p "$(dirname "$worker_ami")"
    ln -s "$base_ami" "$worker_ami"
else
    echo "worker AMI path is not the expected symlink: $worker_ami" >&2
    exit 2
fi

"$WORKER_REPO/scripts/python.sh"     "$WORKER_REPO/examples/speech-asr/datasets/ami/prepare_ami.py"     --spec "$WORKER_REPO/examples/speech-asr/datasets/ami/splits/eval-full-corpus-asr-sc-v1.json"     --output "$base_ami/ami-eval-full-corpus-asr-sc-v1"     --verify-only >/dev/null

[[ "$(sha256sum "$base_ami/ami-eval-full-corpus-asr-sc-v1/manifest.jsonl" | awk '{print $1}')" == "$SOURCE_SHA" ]] || {
    echo "edge held-out source manifest hash mismatch" >&2
    exit 2
}
[[ "$(sha256sum "$base_ami/heldout-eval-v1/manifest.jsonl" | awk '{print $1}')" == "$HELDOUT_SHA" ]] || {
    echo "edge qualified held-out manifest hash mismatch" >&2
    exit 2
}

printf 'edge held-out evaluation data: PASS\n'
REMOTE

printf '{"status":"provisioned","worker":"%s","commit":"%s","source_manifest_sha256":"%s","heldout_manifest_sha256":"%s"}\n'     "$WORKER" "$COMMIT" "$SOURCE_SHA" "$HELDOUT_SHA"
