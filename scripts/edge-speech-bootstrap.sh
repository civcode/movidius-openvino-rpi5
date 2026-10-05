#!/usr/bin/env bash
# Prepare the dedicated edge automation worktree at one exact repository commit.
set -euo pipefail

[[ $# -eq 3 ]] || {
    echo "usage: edge-speech-bootstrap.sh BASE_REPO WORKER_REPO COMMIT" >&2
    exit 2
}
BASE_REPO="$1"
WORKER_REPO="$2"
COMMIT="$3"

[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
    echo "invalid full Git commit: $COMMIT" >&2
    exit 2
}
git -C "$BASE_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "base repository is not a Git checkout: $BASE_REPO" >&2
    exit 2
}

# Fetching updates repository refs but never changes the human checkout branch.
git -C "$BASE_REPO" fetch --quiet origin
git -C "$BASE_REPO" cat-file -e "$COMMIT^{commit}" 2>/dev/null || {
    echo "requested worker commit is unavailable after fetch: $COMMIT" >&2
    exit 2
}

if [[ ! -e "$WORKER_REPO" ]]; then
    mkdir -p "$(dirname "$WORKER_REPO")"
    git -C "$BASE_REPO" worktree add --detach "$WORKER_REPO" "$COMMIT" >/dev/null
else
    git -C "$WORKER_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        echo "worker path exists but is not a Git worktree: $WORKER_REPO" >&2
        exit 2
    }
    if [[ -n "$(git -C "$WORKER_REPO" status --porcelain --untracked-files=no)" ]]; then
        echo "dedicated worker checkout has tracked local changes: $WORKER_REPO" >&2
        exit 2
    fi
    git -C "$WORKER_REPO" checkout --detach "$COMMIT" >/dev/null
fi

actual="$(git -C "$WORKER_REPO" rev-parse HEAD)"
[[ "$actual" == "$COMMIT" ]] || {
    echo "worker checkout revision mismatch: $actual != $COMMIT" >&2
    exit 2
}

# Dataset provisioning is deliberately out-of-band. Reuse an already prepared
# AMI tree from the human checkout via a symlink when the automation worktree
# does not already have its own dataset tree.
base_ami="$BASE_REPO/work/speech-asr/ami"
worker_speech="$WORKER_REPO/work/speech-asr"
worker_ami="$worker_speech/ami"
mkdir -p "$worker_speech"
if [[ ! -e "$worker_ami" && -e "$base_ami" ]]; then
    ln -s "$base_ami" "$worker_ami"
fi

printf 'worker_repo=%s\nworker_commit=%s\n' "$WORKER_REPO" "$actual"
