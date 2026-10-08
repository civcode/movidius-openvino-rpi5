#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

backend="${SPEECH_TORCH_BACKEND:-cuda}"
case "$backend" in
    rocm)
        "$ROOT/scripts/prepare-python-env.sh" training-rocm >/dev/null
        cpuset="${SPEECH_ROCM_CPUSET:-0-15}"
        threads="${SPEECH_ROCM_THREADS:-16}"
        [[ "$cpuset" =~ ^[0-9,-]+$ ]] || {
            echo "invalid SPEECH_ROCM_CPUSET: $cpuset" >&2
            exit 2
        }
        [[ "$threads" =~ ^[1-9][0-9]*$ ]] || {
            echo "invalid SPEECH_ROCM_THREADS: $threads" >&2
            exit 2
        }
        exec env \
            OMP_NUM_THREADS="$threads" \
            MKL_NUM_THREADS="$threads" \
            OPENBLAS_NUM_THREADS="$threads" \
            HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 \
            SPEECH_ROCM_CPUSET="$cpuset" \
            SPEECH_ROCM_THREADS="$threads" \
            taskset -c "$cpuset" \
            "$ROOT/work/venv-training-rocm/bin/python" \
            "$ROOT/scripts/python-rocm-affinity.py" \
            "$@"
        ;;
    auto|cpu|cuda)
        "$ROOT/scripts/prepare-python-env.sh" training >/dev/null
        exec "$ROOT/work/venv-training/bin/python" "$@"
        ;;
    *)
        echo "invalid SPEECH_TORCH_BACKEND: $backend" >&2
        exit 2
        ;;
esac
