#!/usr/bin/env bash
# Run the frozen cnn_ctc_v19 MYRIAD + CPU decoder measurement on the Pi 5.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_MANIFEST_SHA256="fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"
EXPECTED_SAMPLES=1273
EXPECTED_WER="0.5497732671129346"
EXPECTED_CER="0.4634567759027818"

MANIFEST="$ROOT/work/speech-asr/ami/model-quality-v4/validation.manifest.jsonl"
IR_DIR=""
DECODER_ARTIFACT="$ROOT/work/speech-asr/v19-decoder-bench/decoder-artifact-v1.json"
WORK_DIR="$ROOT/work/speech-asr/v19-deployed-eval/cache/evaluation"
OUTPUT="$ROOT/work/speech-asr/v19-deployed-eval/evidence/result.json"
LOG="$ROOT/work/speech-asr/v19-deployed-eval/evidence/evaluator.log"
BENCHMARK_ID="ami-model-quality-v4-architecture-screen-v1-es2011-validation"
EXPERIMENT_ID="exp-f4adb44ab833e896-deployed-decoder-v1"
CPUSET="${SPEECH_DECODER_CPUSET:-0-3}"
FRESH=0
PREFLIGHT_ONLY=0
LOCK_PATH="${SPEECH_MYRIAD_LOCK:-/tmp/movidius-speech-asr-myriad.lock}"

usage() {
    cat <<'EOF'
usage: evaluate-cnn-ctc-v19-deployed.sh --ir-dir PATH [options]

Required:
  --ir-dir PATH             directory containing cnn_ctc_v19.xml/.bin

Options:
  --manifest PATH           frozen v4 ES2011 validation manifest
  --decoder-artifact PATH   frozen decoder-artifact-v1.json
  --work-dir PATH           evaluator cache/work directory
  --output PATH             result JSON
  --log PATH                combined evaluator stdout/stderr log
  --benchmark-id ID         result benchmark id
  --experiment-id ID        result experiment id
  --cpuset LIST             CPU affinity for evaluator/decoder (default 0-3)
  --fresh                    remove prior evaluator cache/result/log first
  --preflight-only           validate inputs without touching MYRIAD
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --ir-dir) IR_DIR="$2"; shift 2 ;;
        --decoder-artifact) DECODER_ARTIFACT="$2"; shift 2 ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --log) LOG="$2"; shift 2 ;;
        --benchmark-id) BENCHMARK_ID="$2"; shift 2 ;;
        --experiment-id) EXPERIMENT_ID="$2"; shift 2 ;;
        --cpuset) CPUSET="$2"; shift 2 ;;
        --fresh) FRESH=1; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$IR_DIR" ]] || { echo "--ir-dir is required" >&2; exit 2; }
[[ "$CPUSET" =~ ^[0-9,-]+$ ]] || { echo "invalid --cpuset: $CPUSET" >&2; exit 2; }
[[ "$(uname -m)" =~ ^(aarch64|arm64)$ ]] || {
    echo "deployed v19 measurement requires an arm64/aarch64 Pi host" >&2
    exit 2
}
for command in sha256sum flock taskset; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "required command missing: $command" >&2
        exit 2
    }
done
for path in     "$MANIFEST"     "$IR_DIR/cnn_ctc_v19.xml"     "$IR_DIR/cnn_ctc_v19.bin"     "$DECODER_ARTIFACT"; do
    [[ -s "$path" ]] || { echo "required file missing or empty: $path" >&2; exit 2; }
done

actual_manifest_sha="$(sha256sum "$MANIFEST" | awk '{print $1}')"
[[ "$actual_manifest_sha" == "$EXPECTED_MANIFEST_SHA256" ]] || {
    echo "validation manifest hash mismatch: $actual_manifest_sha" >&2
    exit 2
}

"$ROOT/scripts/python-apps.sh" -     "$ROOT" "$MANIFEST" "$DECODER_ARTIFACT"     "$EXPECTED_MANIFEST_SHA256" "$EXPECTED_SAMPLES"     "$EXPECTED_WER" "$EXPECTED_CER" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
manifest = pathlib.Path(sys.argv[2]).resolve()
artifact_path = pathlib.Path(sys.argv[3]).resolve()
expected_manifest_sha = sys.argv[4]
expected_samples = int(sys.argv[5])
expected_wer = float(sys.argv[6])
expected_cer = float(sys.argv[7])

sys.path.insert(0, str(root / "examples" / "speech-asr" / "python"))
from speech_asr.cnn_ctc import load_vocab
from speech_asr.contracts import validate_speech_sample
from speech_asr.ctc_beam import load_decoder_artifact

vocab = load_vocab(root / "examples" / "speech-asr" / "models" / "cnn_ctc_v19" / "vocab.json")
artifact = load_decoder_artifact(artifact_path, vocab=vocab)
provenance = artifact.get("provenance", {})
if provenance.get("validation_manifest_sha256") != expected_manifest_sha:
    raise SystemExit("decoder artifact validation-manifest provenance mismatch")
if abs(float(provenance.get("selected_wer", -1.0)) - expected_wer) > 1e-12:
    raise SystemExit("decoder artifact selected WER changed")
if abs(float(provenance.get("selected_cer", -1.0)) - expected_cer) > 1e-12:
    raise SystemExit("decoder artifact selected CER changed")

records = 0
missing = []
with manifest.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        record = validate_speech_sample(json.loads(line))
        records += 1
        audio = pathlib.Path(record["audio"]["path"])
        if not audio.is_absolute():
            audio = manifest.parent / audio
        if not audio.is_file():
            missing.append(str(audio))
            if len(missing) >= 10:
                break
if records != expected_samples and not missing:
    raise SystemExit(f"validation manifest record count changed: {records} != {expected_samples}")
if missing:
    raise SystemExit("validation audio missing; first paths: " + ", ".join(missing))
print(json.dumps({
    "status": "preflight-ok",
    "samples": records,
    "decoder": artifact["decoder"],
    "manifest": str(manifest),
    "decoder_artifact": str(artifact_path),
}, sort_keys=True))
PY

if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
    exit 0
fi

if [[ "$FRESH" -eq 1 ]]; then
    work_abs="$(realpath -m "$WORK_DIR")"
    case "$work_abs" in
        "$ROOT"/work/*) rm -rf -- "$work_abs" ;;
        *) echo "refusing --fresh outside repository work/: $work_abs" >&2; exit 2 ;;
    esac
    rm -f -- "$OUTPUT" "$LOG"
fi

mkdir -p "$WORK_DIR" "$(dirname "$OUTPUT")" "$(dirname "$LOG")"

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo '{"status":"blocked","failure_class":"worker_busy"}' >&2
    exit 75
fi

set +e
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c "$CPUSET"     "$ROOT/scripts/evaluate-cnn-ctc-v19.sh"         --platform arm64         --manifest "$MANIFEST"         --benchmark-id "$BENCHMARK_ID"         --ir-dir "$IR_DIR"         --decoder-artifact "$DECODER_ARTIFACT"         --experiment-id "$EXPERIMENT_ID"         --work-dir "$WORK_DIR"         --output "$OUTPUT"         2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
set -e
exit "$status"
