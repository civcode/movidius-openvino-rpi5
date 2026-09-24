#!/usr/bin/env bash
#
# Prepare the SSDLite-MobileNetV2 COCO object detector for the MYRIAD runtime
# built by this project:
#
#   ssdlite_mobilenet_v2_coco_2018_05_09  ->  OpenVINO 2020.3.2 IR (FP16)
#
# The source model is the TensorFlow 1.x Object Detection API frozen graph
# from the official model zoo.  Only two files of the release are needed:
#
#   frozen_inference_graph.pb   the graph (preprocessing included)
#   pipeline.config             SSD pipeline config (input size, class count)
#
# The conversion runs the vendored OpenVINO 2020.3.2 Model Optimizer
# (mo_tf.py + extensions/front/tf/ssd_v2_support.json) in a native
# python:3.7-slim container with a TensorFlow 1.x wheel installed - the MO 2020.3
# TF front-end imports tensorflow<2 to parse the frozen GraphDef.  TF 1.15.0 is
# the last 1.x release and ships wheels only up to python 3.7 (1.x was later
# removed from PyPI; the cp37 wheel remains), hence 3.7.  MO is pure Python and
# the IR it emits is architecture-independent, so no emulation is needed;
# inference still runs in the OpenVINO container built by ./build.sh.
#
# Layout after running:
#   vendor/models/ssdlite_mobilenet_v2/
#   ├── source/            frozen_inference_graph.pb, pipeline.config, tarball
#   ├── openvino/          ssdlite_mobilenet_v2.{xml,bin,mapping}  (FP16, MYRIAD)
#   └── openvino_fp32/     ssdlite_mobilenet_v2.{xml,bin,mapping}  (FP32, CPU)
#   vendor/models/labels/coco.txt   "id<TAB>name" lines (0=background, 1..90)
#
# Usage:
#   ./scripts/prepare-ssdlite.sh
#   MO_IMAGE=python:3.8-slim ./scripts/prepare-ssdlite.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MO_SRC=vendor/openvino-2020.3.2/model-optimizer
MO_STAGE=work/mo-2020.3
MODELS=vendor/models
MODEL_DIR="$MODELS/ssdlite_mobilenet_v2"
SRC_DIR="$MODEL_DIR/source"
OUT_DIR="$MODEL_DIR/openvino"
MODEL_NAME=ssdlite_mobilenet_v2
# python 3.7: TF 1.15.0 (the last 1.x) has no python 3.8 wheels.
MO_IMAGE="${MO_IMAGE:-python:3.7-slim}"
# The converter runs natively on the host (override with MO_PLATFORM=... if you
# want to convert under emulation instead).
if [[ -z "${MO_PLATFORM:-}" ]]; then
	case "$(uname -m)" in
		x86_64)  MO_PLATFORM=linux/amd64 ;;
		aarch64) MO_PLATFORM=linux/arm64 ;;
		armv7l)  MO_PLATFORM=linux/arm/v7 ;;
		*)       MO_PLATFORM="linux/$(uname -m)" ;;
	esac
fi
# numpy must stay <1.19 for TF 1.15; protobuf must stay <=3.20 for its
# generated protos.  protobuf 3.19.6 is what the rest of this project pins.
TF_PIP_PINS='tensorflow==1.15.0 numpy==1.18.5 networkx==2.6.3 protobuf==3.19.6 defusedxml==0.7.1'

# Official TensorFlow model zoo release (the model page's tarball link).
TARBALL_URL='http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz'
TARBALL_SHA256='542445cce834dbfbb7df1991425d475e85a2d7ec68c60a4f262bb18aac10c8b2'
# The model's pipeline.config references mscoco_label_map.pbtxt.  Pinned to the
# revision last modified 2017-09-21 (before the May 2018 training run): the
# current master copy carries the same 80 class ids, but pinning removes the
# doubt.  COCO class ids are sparse in 1..90 (80 classes); 0 is background.
LABELS_URL='https://raw.githubusercontent.com/tensorflow/models/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/object_detection/data/mscoco_label_map.pbtxt'
LABELS_SHA256='bac43afbd919df8cb70a357227f3dbc77e12283cac8494254ed99ec7d5371316'

mkdir -p "$SRC_DIR" "$OUT_DIR" "$MODELS/labels"

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

# --------------------------------------------- 1. model zoo release
echo "== TF model zoo: ssdlite_mobilenet_v2_coco_2018_05_09 =="
fetch "$TARBALL_URL" "$SRC_DIR/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz" "$TARBALL_SHA256"
tar xzf "$SRC_DIR/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz" -C "$SRC_DIR" \
	--strip-components=1 \
	ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb \
	ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config
printf '    %s  %s\n' "$(stat -c%s "$SRC_DIR/frozen_inference_graph.pb")" "frozen_inference_graph.pb"
printf '    %s  %s\n' "$(stat -c%s "$SRC_DIR/pipeline.config")" "pipeline.config"

# -------------------------- 2. COCO label map -> "id<TAB>name" lines
echo
echo "== COCO label map (mscoco, pinned revision) =="
fetch "$LABELS_URL" "$SRC_DIR/mscoco_label_map.pbtxt" "$LABELS_SHA256"
python3 - "$SRC_DIR/mscoco_label_map.pbtxt" "$MODELS/labels/coco.txt" <<'PYEOF'
import re, sys
src, out = sys.argv[1], sys.argv[2]
text = open(src).read()
items = re.findall(r'name:\s*"([^"]*)"\s*id:\s*(\d+)\s*display_name:\s*"([^"]*)"', text)
if not items:
    raise SystemExit('no label items found in %s' % src)
with open(out, 'w') as f:
    f.write('0\tbackground\n')
    for _name, cid, label in items:
        f.write('%s\t%s\n' % (cid, label))
print('    %d classes (+background) -> %s' % (len(items), out))
PYEOF

# ------------------- 3. Model Optimizer 2020.3.2 (vendored, staged)
echo
echo "== Model Optimizer 2020.3.2 (vendored, staged without unit tests) =="
if [[ -f "$MO_STAGE/.staged" && "$(cat "$MO_STAGE/.staged")" == "$(find "$MO_SRC" -name '*.py' | sort | sha256sum | cut -d' ' -f1)" ]]; then
	echo "    already staged (work/mo-2020.3)"
else
	rm -rf "$MO_STAGE"; mkdir -p "$MO_STAGE"
	rsync -a --exclude='*_test.py' --exclude='automation/' --exclude='install_prerequisites/' \
		"$MO_SRC/" "$MO_STAGE/"
	find "$MO_SRC" -name '*.py' | sort | sha256sum | cut -d' ' -f1 > "$MO_STAGE/.staged"
fi
printf '    staged %s python files\n' "$(find "$MO_STAGE" -name '*.py' | wc -l)"

# --------------------------------------- 4. convert with mo_tf.py (FP16)
echo
echo "== converting to OpenVINO 2020.3.2 IR (FP16, ssd_v2_support.json) =="
# set -e inside the container so a pip failure is not masked by the log filter.
docker run --rm --platform "$MO_PLATFORM" \
	-v "$PWD/$MO_STAGE:/mo:ro" -v "$PWD/$MODELS:/models" \
	-e PYTHONDONTWRITEBYTECODE=1 "$MO_IMAGE" bash -lc \
	"set -e
	 pip install --no-cache-dir -q $TF_PIP_PINS
	 python /mo/mo_tf.py \
		--input_model /models/$MODEL_NAME/source/frozen_inference_graph.pb \
		--tensorflow_object_detection_api_pipeline_config /models/$MODEL_NAME/source/pipeline.config \
		--tensorflow_use_custom_operations_config /mo/extensions/front/tf/ssd_v2_support.json \
		--data_type FP16 \
		--output_dir /models/$MODEL_NAME/openvino \
		--model_name $MODEL_NAME > /tmp/mo.log 2>&1
	 grep -E '\[ (SUCCESS|ERROR) \]|Elapsed time' /tmp/mo.log || true"
ls -l "$OUT_DIR/${MODEL_NAME}.xml" "$OUT_DIR/${MODEL_NAME}.bin"

# ------------------------------- 4b. convert the same graph to FP32 IR (CPU)
# The amd64 runtime's CPU plugin (OpenVINO 2020.3) does not accept FP16 input
# tensors ("Input image format FP16 is not supported yet"), so --device CPU
# runs against this FP32 variant; the launchers select it automatically.
echo
echo "== converting to OpenVINO 2020.3.2 IR (FP32, for the CPU plugin) =="
OUT_DIR_FP32="$MODEL_DIR/openvino_fp32"
mkdir -p "$OUT_DIR_FP32"
docker run --rm --platform "$MO_PLATFORM" \
	-v "$PWD/$MO_STAGE:/mo:ro" -v "$PWD/$MODELS:/models" \
	-e PYTHONDONTWRITEBYTECODE=1 "$MO_IMAGE" bash -lc \
	"set -e
	 pip install --no-cache-dir -q $TF_PIP_PINS
	 python /mo/mo_tf.py \
		--input_model /models/$MODEL_NAME/source/frozen_inference_graph.pb \
		--tensorflow_object_detection_api_pipeline_config /models/$MODEL_NAME/source/pipeline.config \
		--tensorflow_use_custom_operations_config /mo/extensions/front/tf/ssd_v2_support.json \
		--data_type FP32 \
		--output_dir /models/$MODEL_NAME/openvino_fp32 \
		--model_name $MODEL_NAME > /tmp/mo.log 2>&1
	 grep -E '\[ (SUCCESS|ERROR) \]|Elapsed time' /tmp/mo.log || true"
ls -l "$OUT_DIR_FP32/${MODEL_NAME}.xml" "$OUT_DIR_FP32/${MODEL_NAME}.bin"

# ------------------------------------ 5. report the IR's input/output contract
echo
echo "== IR input/output contract =="
python3 - "$OUT_DIR/${MODEL_NAME}.xml" <<'PYEOF'
import struct
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
for layer in root.iter('layer'):
    kind = layer.get('type')
    if kind not in ('Parameter', 'Result'):
        continue
    for tag, label in (('output', 'input '), ('input', 'output')):
        for port in layer.findall(tag + '/port'):
            dims = [d.text for d in port.findall('dim')]
            print('  %s %-12s %-32s dims %-14s precision %s'
                  % (label, kind, layer.get('name'), ','.join(dims),
                     port.get('precision')))

# the TF Preprocessor nodes are baked into the graph: report the scale/offset
data = {}
for layer in root.iter('layer'):
    if layer.get('type') == 'Const':
        content = layer.find('data')
        if content is None:
            continue
        raw = bytes.fromhex(content.get('value', ''))
        if len(raw) == 4:  # single fp32 const
            data[layer.get('name')] = struct.unpack('>f', raw)[0]
for name, v in sorted(data.items()):
    if name.startswith('Preprocessor'):
        print('  %-40s const %r' % (name, v))
PYEOF

# ------------------------- 5b. known-content test photo (person + dog, P6 PPM)
echo
echo "== test photo =="
DOG_JPG="$MODELS/images/src/dog.jpg"
DOG_PPM="$MODELS/images/dog_ssd.ppm"
DOG_URL='https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg'
DOG_SHA256='5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866'
if [[ -f "$DOG_PPM" ]]; then
	echo "    $DOG_PPM already present"
else
	mkdir -p "$MODELS/images/src"
	fetch "$DOG_URL" "$DOG_JPG" "$DOG_SHA256"
	if python3 -c 'import PIL' 2>/dev/null; then
		python3 - "$DOG_JPG" "$DOG_PPM" <<'PYEOF'
import sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert('RGB')
im.save(dst, 'PPM')  # native resolution, RGB channel order
print('    %s %dx%d -> %s (native resolution; the app resizes to 300x300)' % (src, im.width, im.height, dst))
PYEOF
	else
		echo "    python3 + Pillow not available on the host - $DOG_PPM skipped" >&2
	fi
fi

# --------------------- 5c. known-content test video clip (Big Buck Bunny, ~13 s)
# Used by the --video mode of ssd_stream.py / seg_stream.py / webcam_mobilenet.py.
VIDEO_MP4="$MODELS/images/sample_640x360.mp4"
VIDEO_URL='https://filesamples.com/samples/video/mp4/sample_640x360.mp4'
VIDEO_SHA256='5e66e01296a4984841baaf0b9542aed07a5d5eb84958135a8d612b9ff1ec9419'
if [[ -f "$VIDEO_MP4" ]]; then
	echo "    $VIDEO_MP4 already present"
else
	mkdir -p "$MODELS/images/src"
	fetch "$VIDEO_URL" "$VIDEO_MP4" "$VIDEO_SHA256"
fi

# ----------------------------------------------- 6. ownership + inventory
if [[ -d "$OUT_DIR" && ! -O "$OUT_DIR" ]]; then
	sudo chown -R "$(id -u):$(id -g)" "$MODEL_DIR" "$MODELS/labels"
fi
echo
echo "== corpus =="
find "$MODEL_DIR" "$MODELS/labels/coco.txt" -type f -printf '%10s  %P\n' | sort -k2

cat <<EOF

next:
  ./examples/ssd-detect/README.md                    build + run the detector app
  ./run.sh ssd --image /models/images/dog_ssd.ppm    detect objects on the MA2450
  ./scripts/verify.sh                                full device verification
EOF
