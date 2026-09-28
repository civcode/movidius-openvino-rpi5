#!/usr/bin/env bash
#
# Prepare a real network for the MYRIAD runtime built by this project:
#
#   mobilenetv2-7.onnx  ->  OpenVINO 2020.3.2 IR (FP32 and FP16 weights)
#
# Everything is done with the pinned tools already inside the project:
#
#   * the ONNX model and its reference tensors come from the ONNX model zoo
#     (commit-pinned URLs + sha256),
#   * conversion uses the shared staged OpenVINO 2020.3.2 Model Optimizer in
#     work/model-optimizer and a dedicated work/venv-mo-onnx environment,
#   * MO runs natively; the IR it emits is architecture-independent. Inference
#     still runs in the selected OpenVINO runtime target.
#
# Usage:
#   ./scripts/prepare-mobilenet.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=platform.sh
source "$ROOT/scripts/platform.sh"

TARGET_REQUEST="$(platform_default_request)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) TARGET_REQUEST="$2"; shift 2 ;;
        -h|--help) echo "usage: $0 [--platform armv7|arm64|amd64]"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
platform_load "$TARGET_REQUEST"

MO_STAGE=work/model-optimizer
MODELS=vendor/models
MODEL_NAME=mobilenet-v2-ov203

# The legacy S3 bucket (download.onnx) now answers 403; the same artifact is
# hosted in the onnx/models repo (Git LFS, fetched via the media endpoint).
# The sha256 below is unchanged - it is the LFS oid of the file.
ZOO_TGZ_URL='https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.tar.gz'
ZOO_TGZ_URL_FALLBACK='https://s3.amazonaws.com/download.onnx/models/opset_7/mobilenetv2-7.tar.gz'
ZOO_TGZ_SHA256='b463ad62dae99f13afd88549ca7d43e9bda6876614f3592ebb41177e1db0fcc5'
ZOO_ONNX_SHA256='c1c513582d56afceff8516c73804e484c81c6a830712ab6d682253f4a3cd042f'
LABELS_URL='https://raw.githubusercontent.com/onnx/models/main/validated/vision/classification/synset.txt'

# name<TAB>url<TAB>sha256 of the downloaded source image
IMAGES=(
	"dog	https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg	5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866"
	"cat	https://raw.githubusercontent.com/BVLC/caffe/master/examples/images/cat.jpg	59b67bce343d435cb8f79b6abefc6eecc79924abcda6e7c7e39f8173bb932bc1"
	"eagle	https://raw.githubusercontent.com/pjreddie/darknet/master/data/eagle.jpg	f95f57297fd51a6c5ddceb5ed6916fa759d1d65f402fee34c05b31a9608755c0"
	"banana	https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Bananavarieties.jpg/330px-Bananavarieties.jpg	32c4d98303ff9c03613c39e47c0bedd9a4950fda949cca2c47087c369a15636a"
	"cup	https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Coffee_at_Christmas_%28Unsplash%29.jpg/330px-Coffee_at_Christmas_%28Unsplash%29.jpg	030c516d2c53b909e0edef63734c11f8fa5e2cc8916f111c4efb8d6df2d8e2a5"
)
IMAGE_NAMES=(dog cat eagle banana cup)

mkdir -p "$MODELS/onnx" "$MODELS/labels" "$MODELS/images/src" "$MODELS/test_data"

fetch() {  # $1 = url(s, space separated), $2 = path, $3 = optional sha256
	if [[ ! -f "$2" ]]; then
		for url in $1; do
			echo "    fetching ${url##*/} -> $2"
			if curl -sSfL --max-time 900 -o "$2" "$url"; then
				break
			fi
			rm -f "$2"
			if [[ "${url}" == "${1##* }" ]]; then
				echo "    all download URLs failed for ${url##*/}" >&2
				return 1
			fi
		done
	fi
	if [[ -n "${3:-}" ]]; then
		echo "$3  $2" | sha256sum -c -
	fi
}

# --------------------------------------------------------- 1. model + test tensors
echo "== ONNX model zoo: mobilenetv2-7 =="
fetch "${ZOO_TGZ_URL} ${ZOO_TGZ_URL_FALLBACK}" "$MODELS/onnx/mobilenetv2-7.tar.gz" "$ZOO_TGZ_SHA256"
tar xzf "$MODELS/onnx/mobilenetv2-7.tar.gz" -C "$MODELS/onnx" \
	--strip-components=1 mobilenetv2-7/mobilenetv2-7.onnx
tar xzf "$MODELS/onnx/mobilenetv2-7.tar.gz" -C "$MODELS/test_data" \
	--strip-components=1 mobilenetv2-7/test_data_set_0/input_0.pb \
	mobilenetv2-7/test_data_set_0/output_0.pb
echo "$ZOO_ONNX_SHA256  $MODELS/onnx/mobilenetv2-7.onnx" | sha256sum -c -
printf '    %s bytes  %s\n' "$(stat -c%s "$MODELS/onnx/mobilenetv2-7.onnx")" "$MODELS/onnx/mobilenetv2-7.onnx"

# -------------------------------------------------------------- 2. label list
echo
echo "== labels =="
fetch "$LABELS_URL" "$MODELS/labels/synset.txt"
printf '    %s lines\n' "$(wc -l < "$MODELS/labels/synset.txt")"

# ------------------------------------------------ 3. Model Optimizer 2020.3.2
echo
echo "== Model Optimizer 2020.3.2 (shared staged runner) =="
"$ROOT/scripts/prepare-model-optimizer.sh"

mo_convert() {  # $1 = --data_type, $2 = output subdirectory
    mkdir -p "$MODELS/$MODEL_NAME/$2"
    echo "    converting --data_type $1 -> $MODELS/$MODEL_NAME/$2"
    "$ROOT/scripts/run-mo.sh" --framework onnx -- \
        --input_model "$MODELS/onnx/mobilenetv2-7.onnx" \
        --output_dir "$MODELS/$MODEL_NAME/$2" \
        --model_name "$MODEL_NAME" \
        --data_type "$1" 2>&1 \
        | grep -E '\\[ (SUCCESS|ERROR) \\]|Elapsed|execution time'
    ls -l "$MODELS/$MODEL_NAME/$2/$MODEL_NAME.xml" \
          "$MODELS/$MODEL_NAME/$2/$MODEL_NAME.bin"
}
mo_convert FP32 fp32
mo_convert FP16 fp16

# ------------------------- 4. official test tensors -> raw little-endian float32
echo
echo "== reference tensors from the model zoo (test_data_set_0) =="
MO_PY="$ROOT/work/venv-mo-onnx/bin/python"
"$MO_PY" - "$MODELS/test_data" <<'PYEOF'
import os, sys
import onnx
import numpy as np
from onnx import numpy_helper

root = sys.argv[1]
for nm in ('input_0', 'output_0'):
    tensor = onnx.load_tensor(os.path.join(root, 'test_data_set_0', nm + '.pb'))
    arr = numpy_helper.to_array(tensor)
    print('   ', nm, tensor.name, arr.shape, arr.dtype,
          'min', arr.min(), 'max', arr.max())
    arr.astype('<f4').tofile(os.path.join(root, nm + '.f32'))
PYEOF

# --------------------------------------------------- 5. photos -> 224x224 PPMs
echo
echo "== test photos =="
for entry in "${IMAGES[@]}"; do
	IFS=$'\t' read -r name url sha <<< "$entry"
	fetch "$url" "$MODELS/images/src/$name.jpg" "$sha"
done

# ImageNet evaluation preprocessing, which is what the model zoo documents for this
# model: resize the short side to 256, center-crop 224x224.  A squashed full-frame
# variant is written next to it because two of these are detection photos where a
# center crop can cut the subject out; logs/MOBILENET.md reports both variants.
if python3 -c 'import PIL' 2>/dev/null; then
	echo "    preprocessing on the host (python3 + Pillow)"
	python3 - "$MODELS/images" "${IMAGE_NAMES[@]}" <<'PYEOF'
import os, sys
from PIL import Image

out, names = sys.argv[1], sys.argv[2:]
os.makedirs(os.path.join(out, 'src'), exist_ok=True)
os.makedirs(os.path.join(out, 'full'), exist_ok=True)
for name in names:
    src = os.path.join(out, 'src', name + '.jpg')
    im = Image.open(src).convert('RGB')
    w, h = im.size
    s = 256.0 / min(w, h)
    big = im.resize((int(round(w * s)), int(round(h * s))), Image.BILINEAR)
    left, top = (big.width - 224) // 2, (big.height - 224) // 2
    big.crop((left, top, left + 224, top + 224)).save(os.path.join(out, name + '.ppm'), 'PPM')
    im.resize((224, 224), Image.BILINEAR).save(os.path.join(out, 'full', name + '.ppm'), 'PPM')
    print(f'    {name:7} {w}x{h} -> centre crop at ({left},{top}) and full-frame squash')
PYEOF
else
	echo "    python3 + Pillow not available on the host - PPMs skipped"
	echo "    (pip install pillow, or provide your own 224x224 binary PPMs)"
fi

# ------------------------------------------------------ 6. ownership + inventory
if [[ -d "$MODELS/$MODEL_NAME/fp32" && ! -O "$MODELS/$MODEL_NAME/fp32" ]]; then
	sudo chown -R "$(id -u):$(id -g)" "$MODELS"
fi
echo
echo "== corpus =="
find "$MODELS" -type f -printf '%10s  %P\n' | sort -k2

cat <<EOF

next:
  ./run.sh mobilenet                                   numerical check against the
                                                       model zoo reference output
  ./run.sh mobilenet --image /models/images/cat.ppm    classify a photo
  IR=fp32 ./run.sh mobilenet                           the FP32 weights instead
EOF
