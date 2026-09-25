#!/usr/bin/env python3
"""
ssd_cpu_server.py - CPU inference server for the SSDLite webcam detection
example (ssd_stream.py).

Implements exactly the same stdin/stdout protocol as the C++
`ssd_detect --stdin`:

  in : repeated frames, each = uint32 width + uint32 height (little endian)
       + width*height*3 RGB bytes
  out: per frame:  "FRAME <w> <h> <infer_ms>"
               "DET <label> <score> <x1> <y1> <x2> <y2>"   (0..N lines)
               "END"

Diagnostics go to stderr; a stream error prints "ERROR <message>" on
stdout (as the C++ server does) and exits 1.  Exits 0 on stdin EOF.

Runs the TF 1.x frozen graph produced by scripts/prepare-ssdlite.sh
(vendor/models/ssdlite_mobilenet_v2/source/frozen_inference_graph.pb)
with full TensorFlow - no conversion.  The graph's detection NMS is
inlined TF While/TensorArray control flow, which tf2onnx cannot convert,
so full TF is the arm64 CPU backend (docs/CPU-BACKENDS.md).

Post-processing mirrors examples/ssd-detect/ssd_postprocess.hpp exactly:
keep score >= min-conf, round normalized corners to pixels
(int(norm * dim + 0.5), clamped to [0, dim]), reject degenerate boxes,
stable-sort by descending score, cut to max-detections.

Examples:
  python3 ssd_cpu_server.py
  python3 ssd_cpu_server.py --model ... --labels ... --min-conf 0.5
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


def to_pixel(norm: float, dim: int) -> int:
    """Mirror ssd_postprocess.hpp toPixel(): round, then clamp to [0, dim]."""
    v = int(norm * dim + 0.5)
    return 0 if v < 0 else (dim if v > dim else v)


def load_labels(path: str) -> "dict[int, str]":
    """Mirror ssd_postprocess.hpp loadLabels(): 'id<ws>name' lines."""
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


def parse_detections(boxes, scores, classes, num_dets, min_conf, max_dets, fw, fh):
    """Mirror ssd_postprocess.hpp parseDetections() on the TF outputs.

    boxes:    [N, 4] normalized [ymin, xmin, ymax, xmax]
    scores:   [N] sorted descending by the graph
    classes:  [N] float class ids
    """
    out = []
    try:
        nd = int(num_dets)
    except (TypeError, ValueError):
        nd = int(num_dets[0])
    n = min(len(scores), nd)
    for i in range(n):
        score = float(scores[i])
        if score < min_conf:
            continue
        ymin, xmin, ymax, xmax = (float(v) for v in boxes[i])
        d = {
            "class_id": int(round(float(classes[i]))),
            "score": score,
            "x1": to_pixel(xmin, fw),
            "y1": to_pixel(ymin, fh),
            "x2": to_pixel(xmax, fw),
            "y2": to_pixel(ymax, fh),
        }
        if d["x2"] <= d["x1"] or d["y2"] <= d["y1"]:
            continue
        out.append(d)
    out.sort(key=lambda d: -d["score"])  # stable sort, descending score
    return out[:max_dets]


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(
        description="CPU (full TensorFlow) detection server for ssd_stream.py")
    ap.add_argument("--model",
                    default=os.path.join(root, "vendor/models/ssdlite_mobilenet_v2/"
                                          "source/frozen_inference_graph.pb"),
                    help="TF frozen graph (default: %(default)s)")
    ap.add_argument("--labels",
                    default=os.path.join(root, "vendor/models/labels/coco.txt"),
                    help="label file, 'id<ws>name' lines (default: %(default)s)")
    ap.add_argument("--model-size", type=int, default=300,
                    help="square model input (default: 300)")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="keep detections with score >= this (default: 0.5)")
    ap.add_argument("--max-detections", type=int, default=10,
                    help="max DET lines per frame (default: 10)")
    args = ap.parse_args()

    for path, what in ((args.model, "frozen graph"), (args.labels, "labels")):
        if not os.path.isfile(path):
            print(f"{what} not found: {path} - run ./scripts/prepare-ssdlite.sh",
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

    # load the frozen graph once (TF1 session under TF2's compat namespace)
    t0 = time.perf_counter()
    gdef = tf.compat.v1.GraphDef()
    with open(args.model, "rb") as fh:
        gdef.ParseFromString(fh.read())
    g = tf.compat.v1.Graph()
    with g.as_default():
        tf.compat.v1.import_graph_def(gdef, name="import")
    feed_name = "import/image_tensor:0"
    fetches = {
        "boxes": g.get_tensor_by_name("import/detection_boxes:0"),
        "scores": g.get_tensor_by_name("import/detection_scores:0"),
        "classes": g.get_tensor_by_name("import/detection_classes:0"),
        "num_dets": g.get_tensor_by_name("import/num_detections:0"),
    }
    sess = tf.compat.v1.Session(graph=g)
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"ssd_cpu_server: TensorFlow {tf.__version__}, model {args.model}, "
          f"input 1x{msz}x{msz}x3 uint8 RGB (raw 0-255), load {load_ms:.1f} ms, "
          f"ready (device CPU)", file=sys.stderr, flush=True)

    frame = bytearray()

    while True:
        header = read_exact(sys.stdin.buffer, 8)
        if len(header) < 8:
            if header:
                print("ssd_cpu_server: truncated frame header, stopping",
                      file=sys.stderr)
            print("ssd_cpu_server: stdin EOF", file=sys.stderr)
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

        # same preprocessing as ssd_detect.cpp: bilinear resize to the model
        rgb = np.frombuffer(frame, dtype="uint8").reshape(fh, fw, 3)
        resized = cv2.resize(rgb, (msz, msz), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype("uint8").reshape(1, msz, msz, 3)

        t0 = time.perf_counter()
        boxes, scores, classes, num_dets = sess.run(
            list(fetches.values()), {feed_name: tensor})
        ms = (time.perf_counter() - t0) * 1000.0

        dets = parse_detections(boxes[0], scores[0], classes[0], num_dets,
                                args.min_conf, args.max_detections, fw, fh)
        out = [f"FRAME {fw} {fh} {ms:.1f}"]
        for d in dets:
            out.append(f"DET {label_for(labels, d['class_id'])} {d['score']:.2f} "
                       f"{d['x1']} {d['y1']} {d['x2']} {d['y2']}")
        out.append("END")
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()
        print(f"ssd_cpu_server: frame done ({ms:.1f} ms, {len(dets)} detection(s))",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
