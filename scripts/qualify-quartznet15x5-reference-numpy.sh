#!/usr/bin/env bash
# Requalify the source QuartzNet model with the Pi-safe NumPy frontend.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE=auto
OUT="$ROOT/work/speech-asr/quartznet15x5-reference/numpy-frontend"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) [[ $# -ge 2 ]] || { echo "--device needs auto|cpu|cuda" >&2; exit 2; }; DEVICE="$2"; shift 2 ;;
        --output-dir) [[ $# -ge 2 ]] || { echo "--output-dir needs a path" >&2; exit 2; }; OUT="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--device auto|cpu|cuda] [--output-dir PATH]"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT"
"$ROOT/scripts/evaluate-quartznet15x5-reference.sh"     --device "$DEVICE"     --frontend numpy     --output "$OUT/result.json"     --hypotheses "$OUT/hypotheses.jsonl"

"$ROOT/scripts/python.sh" - "$OUT/result.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
expected_wer = 0.037939781625675524
if value.get("status") != "pass":
    raise SystemExit(f"NumPy frontend reference qualification did not pass: {value.get('status')!r}")
if value.get("samples") != 2703 or value.get("full_dev_clean") is not True:
    raise SystemExit("NumPy frontend qualification must cover all 2,703 dev-clean utterances")
if value.get("frontend", {}).get("implementation") != "numpy":
    raise SystemExit("NumPy frontend qualification used the wrong frontend implementation")
delta = abs(float(value["wer"]) - expected_wer)
if delta > 0.001:
    raise SystemExit(
        f"NumPy frontend WER drift {delta} exceeds 0.001 from qualified Torch WER"
    )
print(json.dumps({
    "status": "pass",
    "samples": value["samples"],
    "wer": value["wer"],
    "cer": value["cer"],
    "qualified_torch_wer": expected_wer,
    "absolute_wer_delta": delta,
}, sort_keys=True))
PY
