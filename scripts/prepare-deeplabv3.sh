#!/usr/bin/env bash
#
# Prepare the Open Model Zoo DeepLabV3 semantic segmentation model for the
# MYRIAD runtime built by this project:
#
#   deeplabv3  (OMZ public model, Pascal VOC 2012, 21 classes)
#     ->  OpenVINO 2020.3.2 IR (FP16)
#
# The source model is the TensorFlow 1.x frozen graph
# deeplabv3_mnv2_pascal_train_aug_2018_01_29 from the official model zoo
# (the same release the Open Model Zoo distributes).  Only one file of the
# release is needed:
#
#   frozen_inference_graph.pb   the graph (resize-independent; the
#                               513x513 resize and the (x-128)/128
#                               normalization are baked into the graph)
#
# The Open Model Zoo model.yml specifies the conversion arguments
# --reverse_input_channels, --output=ArgMax plus --input=ImageTensor and
# --input_shape=[1,513,513,3] (the TF placeholder shape is dynamic).  The
# generated IR is the runtime contract, and step 4 below logs its actual
# input/output metadata before anything is built on top of it.  What the
# 2020.3.2 MO actually produced (inspected after conversion):
#
#   input  ImageTensor      [1,3,513,513] FP16  NCHW, BGR, raw 0..255
#   baked in:  BGR->RGB channel swap, in-graph bilinear resize to 513x513
#              (identity at the fixed input size), (x/127.5) - 1
#   output ArgMax/sink_port_0  [1,513,513]  per-pixel class ids 0..20
#
# (MO also rewrites the TF NHWC input to NCHW via Transpose; the .bin stores
# f16 constants little-endian, so constant inspection must account for that.)
#
# The conversion runs the vendored OpenVINO 2020.3.2 Model Optimizer
# (mo_tf.py) in a native python:3.7-slim container with a TensorFlow 1.x
# wheel - the same environment the SSDLite conversion uses (prepare-ssdlite.sh),
# so no new compiler stack is introduced.
#
# Layout after running:
#   vendor/models/deeplabv3/
#   ├── source/            frozen_inference_graph.pb, tarball
#   ├── openvino/          deeplabv3.{xml,bin,mapping}  (FP16, MYRIAD)
#   └── openvino_fp32/     deeplabv3.{xml,bin,mapping}  (FP32, CPU)
#   vendor/models/labels/pascal_voc.txt   "id<TAB>name" lines (0=background)
#
# Usage:
#   ./scripts/prepare-deeplabv3.sh
#   MO_IMAGE=python:3.7-slim ./scripts/prepare-deeplabv3.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MO_SRC=vendor/openvino-2020.3.2/model-optimizer
MO_STAGE=work/mo-2020.3
MODELS=vendor/models
MODEL_DIR="$MODELS/deeplabv3"
SRC_DIR="$MODEL_DIR/source"
OUT_DIR="$MODEL_DIR/openvino"
MODEL_NAME=deeplabv3
MO_IMAGE="${MO_IMAGE:-python:3.7-slim}"
if [[ -z "${MO_PLATFORM:-}" ]]; then
	case "$(uname -m)" in
		x86_64)  MO_PLATFORM=linux/amd64 ;;
		aarch64) MO_PLATFORM=linux/arm64 ;;
		armv7l)  MO_PLATFORM=linux/arm/v7 ;;
		*)       MO_PLATFORM="linux/$(uname -m)" ;;
	esac
fi
# Same pins as prepare-ssdlite.sh (shared MO 2020.3 TF front-end env).
TF_PIP_PINS='tensorflow==1.15.0 numpy==1.18.5 networkx==2.6.3 protobuf==3.19.6 defusedxml==0.7.1'

# The TF model zoo release (the Open Model Zoo mirrors the identical file;
# both URLs are tried).  The OMZ model.yml documents the integrity check as a
# 96-hex-digit SHA-384 (not sha256): checksum and size (23882985) below are
# from models/public/deeplabv3/model.yml.
TARBALL_URLS='https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/deeplabv3/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'
TARBALL_SHA384='d3f2b27bb00c485ca45d68731f7198a23926cb865dd010fc444bcad9158e33413cd60da2588a9be3ff1e8c9b4557362b'

mkdir -p "$SRC_DIR" "$OUT_DIR" "$MODELS/labels"

fetch() {  # $1 = url(s, space separated), $2 = path, $3 = optional sha384
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
		echo "$3  $2" | sha384sum -c -
	fi
}

# --------------------------------------------- 1. model zoo release
echo "== TF model zoo: deeplabv3_mnv2_pascal_train_aug_2018_01_29 =="
fetch "$TARBALL_URLS" "$SRC_DIR/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz" "$TARBALL_SHA384"
tar xzf "$SRC_DIR/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz" -C "$SRC_DIR" \
	--strip-components=1 \
	deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb
printf '    %s  %s\n' "$(stat -c%s "$SRC_DIR/frozen_inference_graph.pb")" "frozen_inference_graph.pb"

# ---------------------- 2. Pascal VOC 2012 label map -> "id<TAB>name" lines
echo
echo "== Pascal VOC 2012 label map (21 classes) =="
# The canonical VOC 2012 21-class list (index order used by the model,
# 0=background + 20 classes):
cat > "$MODELS/labels/pascal_voc.txt" <<'EOF'
0	background
1	aeroplane
2	bicycle
3	bird
4	boat
5	bottle
6	bus
7	car
8	cat
9	chair
10	cow
11	diningtable
12	dog
13	horse
14	motorbike
15	person
16	pottedplant
17	sheep
18	sofa
19	train
20	tvmonitor
EOF
echo "    21 classes -> $MODELS/labels/pascal_voc.txt"

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
echo "== converting to OpenVINO 2020.3.2 IR (FP16, per the OMZ model.yml) =="
# The OMZ model.yml pins --reverse_input_channels (RGB->BGR) and
# --output=ArgMax (the class-id head).
# set -e inside the container so a pip failure is not masked by the log filter.
docker run --rm --platform "$MO_PLATFORM" \
	-v "$PWD/$MO_STAGE:/mo:ro" -v "$PWD/$MODELS:/models" \
	-e PYTHONDONTWRITEBYTECODE=1 "$MO_IMAGE" bash -lc \
	"set -e
	 pip install --no-cache-dir -q $TF_PIP_PINS
	 python /mo/mo_tf.py \
		--input_model /models/$MODEL_NAME/source/frozen_inference_graph.pb \
		--reverse_input_channels \
		--input=ImageTensor \
		--input_shape=[1,513,513,3] \
		--output=ArgMax \
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
		--reverse_input_channels \
		--input=ImageTensor \
		--input_shape=[1,513,513,3] \
		--output=ArgMax \
		--data_type FP32 \
		--output_dir /models/$MODEL_NAME/openvino_fp32 \
		--model_name $MODEL_NAME > /tmp/mo.log 2>&1
	 grep -E '\[ (SUCCESS|ERROR) \]|Elapsed time' /tmp/mo.log || true"
ls -l "$OUT_DIR_FP32/${MODEL_NAME}.xml" "$OUT_DIR_FP32/${MODEL_NAME}.bin"

# ------------------------- 5. report the IR's input/output contract
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

# the TF graph's normalization is baked in: report the f16/f32 constants
# that give it away (the source graph does x*(1/127.5) - 1) - inline XML
# consts are big-endian, .bin-stored consts are little-endian
xml_consts = {}
bin_consts = []
for layer in root.iter('layer'):
    if layer.get('type') != 'Const':
        continue
    name = layer.get('name')
    content = layer.find('data')
    if content is None:
        continue
    if content.get('value') is not None:
        raw = bytes.fromhex(content.get('value'))
        if len(raw) == 4:
            xml_consts[name] = struct.unpack('>f', raw)[0]
    elif content.get('offset') is not None and len(bin_consts) < 1 << 26:
        offset = int(content.get('offset'))
        bin_consts.append((name, offset, int(content.get('size', 0)),
                           content.get('element_type')))
if bin_consts:
    blob = open(sys.argv[1].replace('.xml', '.bin'), 'rb').read()
    for name, offset, size, et in bin_consts:
        if size >= 2 and size <= 8 and et in ('f16', 'f32'):
            fmt = {'f16': ('<e', 2), 'f32': ('<f', 4)}[et]
            vals = [struct.unpack_from(fmt[0], blob, offset + i * fmt[1])[0]
                    for i in range(min(size // fmt[1], 4))]
            if any(abs(v - 1 / 127.5) < 1e-4 or abs(v + 1.0) < 1e-4 for v in vals):
                print('  %-48s const %s' % (name, vals))
for name, v in sorted(xml_consts.items()):
    if v in (128.0, 0.0078125, 1.0, -1.0) or abs(v - 128.0) < 1e-6:
        print('  %-48s const %r' % (name, v))
PYEOF

# ----------------------------------------------- 6. ownership + inventory
if [[ -d "$OUT_DIR" && ! -O "$OUT_DIR" ]]; then
	sudo chown -R "$(id -u):$(id -g)" "$MODEL_DIR" "$MODELS/labels"
fi
echo
echo "== corpus =="
find "$MODEL_DIR" "$MODELS/labels/pascal_voc.txt" -type f -printf '%10s  %P\n' | sort -k2

cat <<EOF

test image (already in the repo, VOC classes present: dog/cat/car/bicycle):
  vendor/models/images/dog_ssd.ppm

next:
  ./examples/deeplab-seg/README.md                   build + run the segmenter
  ./run.sh seg --image /models/images/dog_ssd.ppm    segment on the MA2450
  ./scripts/verify.sh                                full device verification
EOF
