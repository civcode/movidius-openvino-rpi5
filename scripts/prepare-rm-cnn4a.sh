#!/usr/bin/env bash
# Prepare the Intel rm_cnn4a_smbr acoustic regression fixture and FP16 IR.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERIFY_ONLY=0
if [[ "${1:-}" == --verify-only ]]; then
    VERIFY_ONLY=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "usage: $0 [--verify-only]" >&2
    exit 2
fi

SOURCE_SPEC="$ROOT/examples/speech-asr/models/rm_cnn4a/source-v1.json"
MODEL_ROOT="$ROOT/vendor/models/rm_cnn4a_smbr"
SOURCE_DIR="$MODEL_ROOT/source"
IR_DIR="$MODEL_ROOT/openvino/fp16"
LOCK="$SOURCE_DIR/SOURCE-LOCK.sha256"
BASE_URL='https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/rm_cnn4a_smbr'
LICENSE_URL='https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/LICENSE.txt'
FILES=(rm_cnn4a.nnet rm_cnn4a.counts rm_cnn4a.mapping rm_cnn4a.md feat1_10.ark score1_10.ark)

if (( VERIFY_ONLY )); then
    if [[ ! -f "$MODEL_ROOT/openvino/model-spec.json" ]]; then
        echo "rm_cnn4a is not fully prepared yet: missing $MODEL_ROOT/openvino/model-spec.json" >&2
        echo "run ./scripts/prepare-rm-cnn4a.sh first, then retry --verify-only" >&2
        exit 2
    fi
    exec "$ROOT/scripts/python.sh" \
        "$ROOT/examples/speech-asr/tools/verify_prepared_rm_cnn4a.py" \
        --model-root "$MODEL_ROOT" \
        --source-spec "$SOURCE_SPEC"
fi

mkdir -p "$SOURCE_DIR" "$IR_DIR"

fetch() {
    local url="$1" path="$2"
    if [[ ! -f "$path" ]]; then
        echo "fetching $url"
        curl -fL --retry 4 --retry-delay 2 --connect-timeout 30 --max-time 1800 -o "$path.part" "$url"
        mv "$path.part" "$path"
    fi
}

for name in "${FILES[@]}"; do
    fetch "$BASE_URL/$name" "$SOURCE_DIR/$name"
done
fetch "$LICENSE_URL" "$SOURCE_DIR/LICENSE.txt"

if [[ -f "$LOCK" ]]; then
    echo "verifying existing rm_cnn4a source lock"
    (cd "$SOURCE_DIR" && sha256sum -c "$(basename "$LOCK")")
else
    echo "creating first-fetch SHA-256 lock at $LOCK"
    (
        cd "$SOURCE_DIR"
        sha256sum "${FILES[@]}" LICENSE.txt | LC_ALL=C sort -k2
    ) > "$LOCK"
fi

"$ROOT/scripts/prepare-model-optimizer.sh"
rm -f "$IR_DIR/rm_cnn4a_fp16.xml" "$IR_DIR/rm_cnn4a_fp16.bin"
"$ROOT/scripts/run-mo.sh" --framework kaldi --     --framework kaldi     --input_model "$SOURCE_DIR/rm_cnn4a.nnet"     --counts "$SOURCE_DIR/rm_cnn4a.counts"     --remove_output_softmax     --data_type FP16     --model_name rm_cnn4a_fp16     --output_dir "$IR_DIR"

test -s "$IR_DIR/rm_cnn4a_fp16.xml"
test -s "$IR_DIR/rm_cnn4a_fp16.bin"

"$ROOT/scripts/python.sh" "$ROOT/examples/speech-asr/tools/inspect_ir.py" \
    "$IR_DIR/rm_cnn4a_fp16.xml" \
    --bin "$IR_DIR/rm_cnn4a_fp16.bin" \
    --output "$IR_DIR/ir-contract.json"

"$ROOT/scripts/python.sh" - "$SOURCE_SPEC" "$SOURCE_DIR" "$IR_DIR" <<'PY'
import hashlib, json, pathlib, sys

spec_path = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2])
ir = pathlib.Path(sys.argv[3])
spec = json.loads(spec_path.read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

result = {
    "schema": "speech-asr/prepared-model",
    "version": 1,
    "id": "rm_cnn4a-fp16",
    "family": "rm_cnn4a",
    "source_spec_sha256": sha(spec_path),
    "source_release": spec["source_release"],
    "source_artifact_sha256": {
        entry["name"]: sha(source / entry["name"]) for entry in spec["files"]
    },
    "source_lock_sha256": sha(source / "SOURCE-LOCK.sha256"),
    "license_sha256": sha(source / "LICENSE.txt"),
    "openvino": {
        "version": "2020.3.2",
        "precision": "FP16",
        "xml_sha256": sha(ir / "rm_cnn4a_fp16.xml"),
        "bin_sha256": sha(ir / "rm_cnn4a_fp16.bin"),
        "ir_contract": json.loads((ir / "ir-contract.json").read_text(encoding="utf-8")),
    },
    "reference_fixture": {
        "features": "source/feat1_10.ark",
        "scores": "source/score1_10.ark"
    }
}
(ir.parent / "model-spec.json").write_text(
    json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(result["openvino"], sort_keys=True))
PY

"$ROOT/scripts/python.sh" \
    "$ROOT/examples/speech-asr/tools/verify_prepared_rm_cnn4a.py" \
    --model-root "$MODEL_ROOT" \
    --source-spec "$SOURCE_SPEC"

echo
echo "prepared:"
echo "  $IR_DIR/rm_cnn4a_fp16.xml"
echo "  $IR_DIR/rm_cnn4a_fp16.bin"
echo "  $MODEL_ROOT/openvino/model-spec.json"
echo "  $SOURCE_DIR/feat1_10.ark"
echo "  $SOURCE_DIR/score1_10.ark"
