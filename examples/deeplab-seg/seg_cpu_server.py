#!/usr/bin/env python3
"""
seg_cpu_server.py - CPU inference server for the DeepLabV3 segmentation
example (seg_stream.py).

Implements exactly the same stdin/stdout protocol as the C++
`seg_detect --stdin`:

  in : repeated frames, each = uint32 width + uint32 height (little endian)
       + width*height*3 RGB bytes
  out: per frame:  "FRAME <w> <h> <total_ms> <infer_ms>"
               "CLASSES <n>"
               "CLASS <id> <name> <pixels>"        (n lines)
               "MASK <w> <h>" + w*h uint16 LE class ids
               "END"

Diagnostics go to stderr; a stream error prints "ERROR <message>" on
stdout (as the C++ server does) and exits 1.  Exits 0 on stdin EOF.

Runs the TF 1.x frozen graph produced by scripts/prepare-deeplabv3.sh
(vendor/models/deeplabv3/source/frozen_inference_graph.pb) with full
TensorFlow - no conversion (the graph is a pure CNN, 970 nodes, no control
flow; see docs/CPU-BACKENDS.md).  Its `ImageTensor` input is uint8 NHWC
RGB (verified by parity spike: feeding BGR mis-segments ~5% of pixels);
the 513x513 bilinear resize is done here, as in `seg_detect`.

Post-processing mirrors examples/deeplab-seg/seg_postprocess.hpp:
nearest-neighbour mask resize to the frame size (masks must not be
averaged), class histogram sorted by descending pixel count (ties by id),
label names from the label file.

Examples:
  python3 seg_cpu_server.py
  python3 seg_cpu_server.py --model ... --labels ...
"""

import argparse
import os
import sys
import time

FRAME_MAX_DIM = 8192


def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return bytes(buf)
        buf += chunk
    return bytes(buf)


def load_labels(path: str) -> "dict[int, str]":
    labels = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            try:
                labels[int(parts[0])] = parts[1].strip()
            except ValueError:
                continue
    if not labels:
        raise RuntimeError(f"no labels parsed from {path}")
    return labels


def label_for(labels, cid: int) -> str:
    return labels.get(cid, f"class_{cid}")


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(
        description="CPU (full TensorFlow) segmentation server for seg_stream.py")
    ap.add_argument("--model",
                    default=os.path.join(root, "vendor/models/deeplabv3/"
                                          "source/frozen_inference_graph.pb"),
                    help="TF frozen graph (default: %(default)s)")
    ap.add_argument("--labels",
                    default=os.path.join(root, "vendor/models/labels/pascal_voc.txt"),
                    help="label file, 'id<ws>name' lines (default: %(default)s)")
    ap.add_argument("--model-size", type=int, default=513,
                    help="square model input (default: 513)")
    args = ap.parse_args()

    for path, what in ((args.model, "frozen graph"), (args.labels, "labels")):
        if not os.path.isfile(path):
            print(f"{what} not found: {path} - run ./scripts/prepare-deeplabv3.sh",
                  file=sys.stderr)
            return 1
    labels = load_labels(args.labels)
    msz = args.model_size

    try:
        import cv2
        import numpy as np
        import tensorflow as tf
    except ImportError as e:
        print(f"missing dependency: {e.name} - run:  pip install {e.name}",
              file=sys.stderr)
        return 1

    gdef = tf.compat.v1.GraphDef()
    with open(args.model, "rb") as fh:
        gdef.ParseFromString(fh.read())
    g = tf.compat.v1.Graph()
    with g.as_default():
        tf.compat.v1.import_graph_def(gdef, name="import")
    feed_name = "import/ImageTensor:0"
    out_name = "import/SemanticPredictions:0"
    t0 = time.perf_counter()
    sess = tf.compat.v1.Session(graph=g)
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"seg_cpu_server: TensorFlow {tf.__version__}, model {args.model}, "
          f"input 1x{msz}x{msz}x3 uint8 RGB (raw 0-255), load {load_ms:.1f} ms, "
          f"ready (device CPU)", file=sys.stderr, flush=True)

    while True:
        header = read_exact(sys.stdin.buffer, 8)
        if len(header) < 8:
            if header:
                print("seg_cpu_server: truncated frame header, stopping",
                      file=sys.stderr)
            print("seg_cpu_server: stdin EOF", file=sys.stderr)
            return 0
        fw, fh = int.from_bytes(header[:4], "little"), int.from_bytes(header[4:], "little")
        if fw == 0 or fh == 0 or fw > FRAME_MAX_DIM or fh > FRAME_MAX_DIM:
            msg = f"frame header out of range: {fw}x{fh}"
            print(f"ERROR {msg}", flush=True)
            print(f"error: {msg}", file=sys.stderr)
            return 1
        frame = read_exact(sys.stdin.buffer, fw * fh * 3)
        if len(frame) != fw * fh * 3:
            msg = f"truncated frame body ({len(frame)} < {fw * fh * 3} bytes)"
            print(f"ERROR {msg}", flush=True)
            print(f"error: {msg}", file=sys.stderr)
            return 1

        tp0 = time.perf_counter()
        # same preprocessing as seg_detect.cpp: bilinear resize to the model
        # (the frame bytes are RGB; ImageTensor is NHWC RGB uint8)
        rgb = np.frombuffer(frame, dtype="uint8").reshape(fh, fw, 3)
        resized = cv2.resize(rgb, (msz, msz), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype("uint8").reshape(1, msz, msz, 3)

        t0 = time.perf_counter()
        pred = sess.run(g.get_tensor_by_name(out_name), {feed_name: tensor})
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # nearest-neighbour resize back to the frame size (crisp masks);
        # mirrors seg_postprocess.hpp nearestResize(): integer sampling with
        # clamping to the last source row/column
        m513 = np.ascontiguousarray(pred[0], dtype="uint8")
        ys = np.minimum((np.arange(fh) * msz) // fh, msz - 1)
        xs = np.minimum((np.arange(fw) * msz) // fw, msz - 1)
        big = m513[np.ix_(ys, xs)].astype("<u2")
        t1 = time.perf_counter()
        total_ms = (t1 - tp0) * 1000.0

        # class histogram: descending pixel count, ties by id
        counts = {}
        for v in big.ravel().tolist():
            counts[v] = counts.get(v, 0) + 1
        hist = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

        out = [f"FRAME {fw} {fh} {total_ms:.1f} {infer_ms:.1f}",
               f"CLASSES {len(hist)}"]
        for cid, px in hist:
            out.append(f"CLASS {cid} {label_for(labels, cid)} {px}")
        out.append(f"MASK {fw} {fh}")
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()
        sys.stdout.buffer.write(big.tobytes())
        sys.stdout.buffer.flush()
        sys.stdout.write("END\n")
        sys.stdout.flush()
        print(f"seg_cpu_server: frame done (total {total_ms:.1f} ms, "
              f"infer {infer_ms:.1f} ms, {len(hist)} class(es))",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
