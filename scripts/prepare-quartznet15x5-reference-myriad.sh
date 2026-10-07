#!/usr/bin/env bash
# Export the qualified 29-class QuartzNet source model and convert it to OV 2020.3 FP16.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRETRAINED="$ROOT/work/speech-asr/pretrained/quartznet15x5-en-base-v2/QuartzNet15x5-En-Base.nemo"
BASE="$ROOT/work/speech-asr/quartznet15x5-reference/myriad"
FIXED_TIME=512

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained) [[ $# -ge 2 ]] || { echo "--pretrained needs a path" >&2; exit 2; }; PRETRAINED="$2"; shift 2 ;;
        --output-dir) [[ $# -ge 2 ]] || { echo "--output-dir needs a path" >&2; exit 2; }; BASE="$2"; shift 2 ;;
        --fixed-time-frames) [[ $# -ge 2 ]] || { echo "--fixed-time-frames needs an integer" >&2; exit 2; }; FIXED_TIME="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--pretrained PATH] [--output-dir PATH] [--fixed-time-frames N]"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -s "$PRETRAINED" ]] || {
    echo "pretrained QuartzNet archive missing: $PRETRAINED" >&2
    echo "run ./scripts/prepare-cnn-ctc-v18-pretrained.sh first" >&2
    exit 2
}
[[ "$FIXED_TIME" =~ ^[1-9][0-9]*$ ]] || { echo "invalid fixed time: $FIXED_TIME" >&2; exit 2; }
(( FIXED_TIME % 16 == 0 )) || { echo "fixed time must be a multiple of 16" >&2; exit 2; }

DYNAMIC="$BASE/dynamic"
EXPORT="$BASE/export"
IR="$BASE/openvino/fp16"
rm -rf "$DYNAMIC" "$EXPORT" "$IR"
mkdir -p "$DYNAMIC" "$EXPORT" "$IR"

echo "== export dynamic reference ONNX for edge parity =="
"$ROOT/scripts/python-training.sh"     "$ROOT/examples/speech-asr/training/export_quartznet15x5_reference.py"     --archive "$PRETRAINED"     --output-dir "$DYNAMIC"
"$ROOT/scripts/python-apps.sh"     "$ROOT/examples/speech-asr/evaluation/compare_quartznet15x5_reference_onnx.py"     --export-dir "$DYNAMIC"     --output "$DYNAMIC/onnx-parity.json"

echo "== export fixed carrier ONNX for OpenVINO 2020.3 =="
"$ROOT/scripts/python-training.sh"     "$ROOT/examples/speech-asr/training/export_quartznet15x5_reference.py"     --archive "$PRETRAINED"     --output-dir "$EXPORT"     --fixed-time-frames "$FIXED_TIME"
"$ROOT/scripts/python-apps.sh"     "$ROOT/examples/speech-asr/evaluation/compare_quartznet15x5_reference_onnx.py"     --export-dir "$EXPORT"     --output "$EXPORT/onnx-parity.json"

echo "== convert fixed carrier graph to OpenVINO 2020.3 FP16 =="
"$ROOT/scripts/run-mo.sh" --framework onnx --     --input_model "$EXPORT/quartznet15x5_nvidia_ref.onnx"     --output_dir "$IR"     --model_name quartznet15x5_nvidia_ref     --data_type FP16

test -s "$IR/quartznet15x5_nvidia_ref.xml"
test -s "$IR/quartznet15x5_nvidia_ref.bin"

"$ROOT/scripts/python.sh"     "$ROOT/examples/speech-asr/tools/inspect_ir.py"     "$IR/quartznet15x5_nvidia_ref.xml"     --bin "$IR/quartznet15x5_nvidia_ref.bin"     > "$IR/ir-contract.json"

"$ROOT/scripts/python.sh" - "$ROOT" "$PRETRAINED" "$DYNAMIC" "$EXPORT" "$IR" "$FIXED_TIME" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
pretrained = pathlib.Path(sys.argv[2])
dynamic = pathlib.Path(sys.argv[3])
export = pathlib.Path(sys.argv[4])
ir = pathlib.Path(sys.argv[5])
fixed_time = int(sys.argv[6])

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

contract = json.loads((ir / "ir-contract.json").read_text(encoding="utf-8"))
if len(contract.get("inputs", [])) != 1 or len(contract.get("outputs", [])) != 1:
    raise SystemExit("reference IR must have exactly one input and one output")
input_shape = contract["inputs"][0]["ports"][0]["shape"]
output_shape = contract["outputs"][0]["ports"][0]["shape"]
if input_shape != [1, 64, fixed_time]:
    raise SystemExit(f"reference IR input shape changed: {input_shape}")
expected_out = (fixed_time + 1) // 2
if output_shape != [1, expected_out, 29]:
    raise SystemExit(
        f"reference IR output shape changed: {output_shape} != {[1, expected_out, 29]}"
    )

dynamic_meta = json.loads((dynamic / "export.json").read_text(encoding="utf-8"))
fixed_meta = json.loads((export / "export.json").read_text(encoding="utf-8"))
if dynamic_meta["onnx"]["dynamic_time_axis"] is not True:
    raise SystemExit("edge parity ONNX must retain dynamic time")
if fixed_meta["onnx"]["dynamic_time_axis"] is not False:
    raise SystemExit("OpenVINO carrier ONNX must be fixed-time")
if fixed_meta["onnx"]["fixed_time_frames"] != fixed_time:
    raise SystemExit("fixed carrier metadata time mismatch")

value = {
    "schema": "speech-asr/quartznet-reference-myriad-artifacts",
    "version": 1,
    "model_id": "quartznet15x5_nvidia_ref",
    "precision": "FP16",
    "openvino_version": "2020.3.2",
    "runtime_shape_policy": "reshape exact source-frontend padded time axis before MYRIAD LoadNetwork",
    "carrier_time_frames": fixed_time,
    "input_shape": input_shape,
    "output_shape": output_shape,
    "artifacts": {
        "pretrained_sha256": sha(pretrained),
        "dynamic_onnx_sha256": sha(dynamic / "quartznet15x5_nvidia_ref.onnx"),
        "fixed_onnx_sha256": sha(export / "quartznet15x5_nvidia_ref.onnx"),
        "xml_sha256": sha(ir / "quartznet15x5_nvidia_ref.xml"),
        "bin_sha256": sha(ir / "quartznet15x5_nvidia_ref.bin"),
        "ir_contract_sha256": sha(ir / "ir-contract.json"),
    },
    "ir": {
        "version": contract["ir_version"],
        "canonical_graph_sha256": contract["canonical_graph_sha256"],
    },
}
(ir / "artifacts.json").write_text(
    json.dumps(value, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(value, sort_keys=True))
PY

echo "QuartzNet15x5 reference MYRIAD artifacts: prepared"
