#!/usr/bin/env bash
# One-time provisioning of the reviewed model-quality AMI dataset to edge.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="edge"
EDGE_BASE_REPO=""
EDGE_WORKER_REPO=""

TRAIN_SHA="89a8624a5dc46ef28845f35591fc1e623dfbd3026d7a4729b7578153d03baf5a"
VALIDATION_SHA="07ebc41041238c1ec374ad64eefe7209fd6c11d1050e8f6f72f0226d358c8923"
BENCHMARK_SHA="9f4444c6c0e54cf25d92a69723727bb46a73632e34ec33402c71beca1b64cd3e"

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

BENCHMARK_DIR="$ROOT/work/speech-asr/ami/ami-benchmark-v1"
QUALITY_DIR="$ROOT/work/speech-asr/ami/model-quality-v1"
for path in     "$BENCHMARK_DIR/manifest.jsonl"     "$BENCHMARK_DIR/provenance.json"     "$QUALITY_DIR/train.manifest.jsonl"     "$QUALITY_DIR/validation.manifest.jsonl"     "$QUALITY_DIR/qualification.json"; do
    [[ -f "$path" ]] || {
        echo "required local dataset artifact missing: $path" >&2
        exit 2
    }
done

[[ "$(sha256sum "$BENCHMARK_DIR/manifest.jsonl" | awk '{print $1}')" == "$BENCHMARK_SHA" ]] || {
    echo "local benchmark manifest hash differs from reviewed identity" >&2
    exit 2
}
[[ "$(sha256sum "$QUALITY_DIR/train.manifest.jsonl" | awk '{print $1}')" == "$TRAIN_SHA" ]] || {
    echo "local training manifest hash differs from reviewed identity" >&2
    exit 2
}
[[ "$(sha256sum "$QUALITY_DIR/validation.manifest.jsonl" | awk '{print $1}')" == "$VALIDATION_SHA" ]] || {
    echo "local validation manifest hash differs from reviewed identity" >&2
    exit 2
}

"$ROOT/scripts/python.sh"     "$ROOT/examples/speech-asr/datasets/ami/prepare_ami.py"     --subset benchmark --verify-only >/dev/null

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

"${SSH[@]}" "$WORKER" mkdir -p     "$EDGE_BASE_REPO/work/speech-asr/ami/ami-benchmark-v1"

"${SSH[@]}" "$WORKER" bash -s --     "$EDGE_BASE_REPO" "$EDGE_WORKER_REPO" "$COMMIT"     < "$ROOT/scripts/edge-speech-bootstrap.sh"

rsync -a --delete --checksum -e "$RSYNC_RSH"     "$BENCHMARK_DIR/"     "$WORKER:$EDGE_BASE_REPO/work/speech-asr/ami/ami-benchmark-v1/"

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

"$WORKER_REPO/scripts/python.sh"     "$WORKER_REPO/examples/speech-asr/datasets/ami/prepare_ami.py"     --subset benchmark --verify-only >/dev/null

"$WORKER_REPO/scripts/qualify-speech-model-data.sh" >/dev/null

train="$base_ami/model-quality-v1/train.manifest.jsonl"
validation="$base_ami/model-quality-v1/validation.manifest.jsonl"
[[ "$(sha256sum "$train" | awk '{print $1}')" == "$TRAIN_SHA" ]] || {
    echo "edge training manifest hash differs from reviewed identity" >&2
    exit 2
}
[[ "$(sha256sum "$validation" | awk '{print $1}')" == "$VALIDATION_SHA" ]] || {
    echo "edge validation manifest hash differs from reviewed identity" >&2
    exit 2
}

printf 'edge model-quality data: PASS\n'
printf 'train_manifest_sha256=%s\n' "$TRAIN_SHA"
printf 'validation_manifest_sha256=%s\n' "$VALIDATION_SHA"
REMOTE

printf '{"status":"provisioned","worker":"%s","commit":"%s","train_manifest_sha256":"%s","validation_manifest_sha256":"%s"}\n'     "$WORKER" "$COMMIT" "$TRAIN_SHA" "$VALIDATION_SHA"
