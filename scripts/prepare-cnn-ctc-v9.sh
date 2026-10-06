#!/usr/bin/env bash
# Convert cnn_ctc_v9 ONNX to pinned OpenVINO 2020.3 FP16 IR.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX="$ROOT/work/speech-asr/cnn_ctc_v9/export/cnn_ctc_v9.onnx"
OUT="$ROOT/work/speech-asr/cnn_ctc_v9/openvino/fp16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx) ONNX="$2"; shift 2 ;;
        --output-dir) OUT="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--onnx path] [--output-dir path]"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -s "$ONNX" ]] || { echo "missing ONNX model: $ONNX" >&2; exit 2; }
mkdir -p "$OUT"
rm -f "$OUT/cnn_ctc_v9.xml" "$OUT/cnn_ctc_v9.bin"

"$ROOT/scripts/run-mo.sh" --framework onnx --     --input_model "$ONNX"     --output_dir "$OUT"     --model_name cnn_ctc_v9     --data_type FP16

test -s "$OUT/cnn_ctc_v9.xml"
test -s "$OUT/cnn_ctc_v9.bin"

"$ROOT/scripts/python.sh"     "$ROOT/examples/speech-asr/tools/inspect_ir.py"     "$OUT/cnn_ctc_v9.xml"     --bin "$OUT/cnn_ctc_v9.bin"     > "$OUT/ir-contract.json"

"$ROOT/scripts/python.sh"     "$ROOT/examples/speech-asr/tools/validate_cnn_ctc_ir.py"     "$ROOT/examples/speech-asr/models/cnn_ctc_v9/model_spec.json"     "$OUT/ir-contract.json"     --output "$OUT/ir-validation.json"

"$ROOT/scripts/python.sh" - "$ONNX" "$OUT" <<'PY'
import hashlib, json, pathlib, sys
onnx = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
contract = json.loads((out / "ir-contract.json").read_text(encoding="utf-8"))
validation = json.loads((out / "ir-validation.json").read_text(encoding="utf-8"))
value = {
    "schema": "speech-asr/cnn-ctc-openvino-artifacts",
    "version": 1,
    "model_id": "cnn_ctc_v9",
    "precision": "FP16",
    "openvino_version": "2020.3.2",
    "artifacts": {
        "onnx_sha256": sha(onnx),
        "xml_sha256": sha(out / "cnn_ctc_v9.xml"),
        "bin_sha256": sha(out / "cnn_ctc_v9.bin"),
        "ir_contract_sha256": sha(out / "ir-contract.json"),
        "ir_validation_sha256": sha(out / "ir-validation.json"),
    },
    "ir": {
        "version": contract["ir_version"],
        "canonical_graph_sha256": contract["canonical_graph_sha256"],
        "input": validation["input"],
        "output": validation["output"],
    },
}
(out / "artifacts.json").write_text(
    json.dumps(value, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(value, sort_keys=True))
PY
