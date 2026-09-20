#!/usr/bin/env python3
"""Generate a tiny FP16 OpenVINO IR (model.xml + model.bin) for the smoke test.

Why this exists
---------------
OpenVINO 2020.3 Model Optimizer needs Python 3.6/3.7 plus numpy and a frontend
framework (TF/ONNX/Caffe), which is a heavy, armv7-hostile install for what is
supposed to be a smoke test. Instead we emit the IR directly.

OpenVINO 2020.3 can only read IR **version 10** (inference-engine/src/
inference_engine/ie_ir_parser.cpp handles `case 10` only), so this script
emits exactly that layout, with the same element names/attributes that Model
Optimizer 2020.3 emits:

  * Parameter      -> created from opset1 attributes:  <data shape= element_type=>
  * every layer    -> version="opset1" is mandatory
  * Const          -> <data element_type= shape= offset= size=> (the parser demands a
                       <data> node; the shape it uses comes from the OUTPUT port)
  * Convolution    -> <data strides= pads_begin= pads_end= dilations= auto_pad=>
  * Result         -> network output

Everything is FP16, which is what the MA2450 VPU wants: no cast nodes are
inserted and the graph compiles as-is.
"""

import argparse
import hashlib
import os
import struct
import sys

NET_NAME = "tiny_conv_fp16"

N, C, H, W = 1, 3, 32, 32       # input tensor
K, C2, KH, KW = 8, 3, 3, 3      # output channels, input channels, kernel


def pack_f16(values):
    """Pack a list of floats as IEEE-754 binary16 (struct 'e' is available since 3.6)."""
    return b"".join(struct.pack("<e", v) for v in values)


def dims_xml(dims):
    return "".join("<dim>{}</dim>".format(d) for d in dims)


def build_xml():
    weights_off, weights_size = 0, K * C2 * KH * KW * 2  # 216 fp16 values -> 432 B

    xml = """<?xml version="1.0"?>
<net name="{name}" version="10">
\t<layers>
\t\t<layer id="0" name="input" type="Parameter" version="opset1">
\t\t\t<data shape="{shape}" element_type="f16"/>
\t\t\t<output>
\t\t\t\t<port id="0" precision="FP16">
{dims}
\t\t\t\t</port>
\t\t\t</output>
\t\t</layer>
\t\t<layer id="1" name="conv_weights" type="Const" version="opset1">
\t\t\t<data element_type="f16" shape="{kshape}" offset="{off}" size="{size}"/>
\t\t\t<output>
\t\t\t\t<port id="0" precision="FP16">
{kdims}
\t\t\t\t</port>
\t\t\t</output>
\t\t\t<blobs>
\t\t\t\t<weights offset="{off}" size="{size}"/>
\t\t\t</blobs>
\t\t</layer>
\t\t<layer id="2" name="conv" type="Convolution" version="opset1">
\t\t\t<data strides="1,1" pads_begin="1,1" pads_end="1,1" dilations="1,1" auto_pad="explicit"/>
\t\t\t<input>
\t\t\t\t<port id="0" precision="FP16">
{dims}
\t\t\t\t</port>
\t\t\t\t<port id="1" precision="FP16">
{kdims}
\t\t\t\t</port>
\t\t\t</input>
\t\t\t<output>
\t\t\t\t<port id="2" precision="FP16">
{odims}
\t\t\t\t</port>
\t\t\t</output>
\t\t</layer>
\t\t<layer id="3" name="output" type="Result" version="opset1">
\t\t\t<input>
\t\t\t\t<port id="0" precision="FP16">
{odims}
\t\t\t\t</port>
\t\t\t</input>
\t\t</layer>
\t</layers>
\t<edges>
\t\t<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
\t\t<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
\t\t<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
\t</edges>
\t<meta_data>
\t\t<MO_version />
\t\t<cli_parameters>
\t\t\t<data_type value="FP16" />
\t\t\t<output name="output" />
\t\t</cli_parameters>
\t</meta_data>
</net>
""".format(
        name=NET_NAME,
        shape="{},{}".format(N, ",".join(str(d) for d in (C, H, W))),
        dims=dims_xml([N, C, H, W]),
        kdims=dims_xml([K, C2, KH, KW]),
        kshape="{}".format(",".join(str(d) for d in (K, C2, KH, KW))),
        odims=dims_xml([N, K, H, W]),
        off=weights_off,
        size=weights_size,
    )
    return xml


def build_bin():
    """Deterministic 3x3x3x8 kernel: a mild Laplacian-ish filter per output channel."""
    vals = []
    for k in range(K):
        for c in range(C2):
            for i in range(KH):
                for j in range(KW):
                    v = ((k + c + i + j) % 5 - 2) / 4.0 + (0.25 if i == 1 and j == 1 else 0.0)
                    vals.append(v)
    assert len(vals) == K * C2 * KH * KW
    return pack_f16(vals)


def main():
    ap = argparse.ArgumentParser(description="emit a tiny IR v10 model for the MYRIAD smoke test")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="directory for model.xml / model.bin")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    xml = build_xml()
    bin_blob = build_bin()

    xml_path = os.path.join(args.outdir, "model.xml")
    bin_path = os.path.join(args.outdir, "model.bin")
    with open(xml_path, "w") as f:
        f.write(xml)
    with open(bin_path, "wb") as f:
        f.write(bin_blob)

    print("wrote {}".format(xml_path))
    print("wrote {} ({} bytes, input {}, weights {})".format(
        bin_path, len(bin_blob),
        "{}x{}x{}x{}".format(N, C, H, W),
        "{}x{}x{}x{}".format(K, C2, KH, KW)))
    print("sha256 xml : {}".format(hashlib.sha256(xml.encode()).hexdigest()))
    print("sha256 bin : {}".format(hashlib.sha256(bin_blob).hexdigest()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
