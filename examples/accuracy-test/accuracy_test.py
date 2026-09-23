#!/usr/bin/env python3
"""
accuracy_test.py - MobileNet v2 top-1/top-5 accuracy check on the Movidius
MA2450 (OpenVINO 2020.3 MYRIAD) using images with known content.

Dataset: one representative image per ImageNet-1k class from
EliSchwartz/imagenet-sample-images
(https://github.com/EliSchwartz/imagenet-sample-images).  Each file name is
'<synset>_<label>.JPEG' (e.g. 'n02085620_Chihuahua.JPEG'), so the ground
truth class is the synset encoded in the file name, matched against
vendor/models/labels/synset.txt.

The images are preprocessed exactly like the webcam example (resize to
224x224, BGR->RGB, x/255, mean/std normalization, NCHW float32) and
classified on the stick through the long-running mobilenet_server backend
(shared with examples/webcam, see examples/mobilenet_client.py).

Usage:
  # fetch the dataset once (into examples/accuracy-test/images/)
  ./examples/accuracy-test/fetch-sample-images.sh

  # run the accuracy check (all 1000 images, ~1 min on the stick)
  python3 examples/accuracy-test/accuracy_test.py

  # quick subset / other model
  python3 examples/accuracy-test/accuracy_test.py --limit 100
  python3 examples/accuracy-test/accuracy_test.py --ir fp32 --backend docker

A MobileNetV2 top-1 accuracy around 81% on this set is expected (the
README's verified number on this dataset is 81.4%; the official fp32
ImageNet-val top-1 is 71.9%, a harder benchmark).  A score far below ~80%
means the pipeline is broken (preprocessing, tensor layout or device
output), not the model.
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mobilenet_client import (            # noqa: E402
    DEFAULT_LABELS,
    MyriadClient,
    load_labels,
    note,
    preprocess,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGES = os.path.join(HERE, "images")


def die(msg, code=1):
    print("accuracy_test: " + msg, file=sys.stderr, flush=True)
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser(
        description="MobileNet v2 accuracy check with known-content images on "
                    "the Movidius MA2450",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--images-dir", default=DEFAULT_IMAGES,
                    help="directory with '<synset>_<label>.JPEG' images")
    ap.add_argument("--labels", default=DEFAULT_LABELS, help="synset label file")
    ap.add_argument("--backend", choices=["auto", "host", "docker"], default="auto",
                    help="where mobilenet_server runs")
    ap.add_argument("--ir", choices=["fp16", "fp32"], default="fp16",
                    help="model precision (vendor/models/mobilenet-v2-ov203/<ir>)")
    ap.add_argument("--device", default="MYRIAD", help="OpenVINO device name")
    ap.add_argument("--limit", type=int, default=0,
                    help="classify at most N images (0 = all)")
    ap.add_argument("--show-errors", type=int, default=25,
                    help="how many misclassified images to list (0 = none)")
    ap.add_argument("--request-timeout", type=float, default=60.0,
                    help="max seconds to wait for one inference")
    ap.add_argument("--fail-under", type=float, default=0.0,
                    help="exit 1 if top-1 accuracy (in percent) is below this")
    args = ap.parse_args()

    if cv2 is None:
        die("OpenCV is required: pip install opencv-python numpy")

    # case-insensitive discovery (the dataset is .JPEG but users add others)
    files = sorted(
        os.path.join(args.images_dir, name)
        for name in os.listdir(args.images_dir)
        if name.lower().endswith((".jpeg", ".jpg", ".png")))
    if not files:
        die("no images in %s - run ./examples/accuracy-test/fetch-sample-images.sh"
            % args.images_dir)
    if args.limit > 0:
        files = files[:args.limit]

    labels = load_labels(args.labels)
    if len(labels) != 1000:
        die("label file %s has %d lines, expected 1000" % (args.labels, len(labels)))
    synset_to_idx = {}
    for idx, (cid, _name) in enumerate(labels):
        synset_to_idx[cid] = idx

    client = MyriadClient(args.backend, args.ir, args.device, args.request_timeout)
    total = len(files)
    ok = skip = top1_hits = top5_hits = 0
    bad_rows = []   # every top-1 miss (up to args.show_errors)
    top5_misses = 0
    t0 = time.monotonic()
    t_first = None
    try:
        for path in files:
            base = os.path.basename(path)
            synset = base.split("_")[0]
            if len(synset) != 9 or not synset[1:9].isdigit() or synset not in synset_to_idx:
                skip += 1
                continue
            frame = cv2.imread(path)
            if frame is None:
                skip += 1
                continue
            truth = synset_to_idx[synset]
            tensor = preprocess(frame)
            logits = client.infer(tensor)
            order = np.argsort(logits)[::-1][:5]
            top1, top5 = int(order[0]), [int(i) for i in order]
            ok += 1
            if top1 == truth:
                top1_hits += 1
            else:
                # a top-1 miss: record it (with the top-5 status) up to the cap
                if len(bad_rows) < args.show_errors:
                    bad_rows.append((base, labels[truth][1], top1, labels[top1][1],
                                     float(logits[top1]), truth in top5))
            if truth in top5:
                top5_hits += 1
            else:
                top5_misses += 1
            if t_first is None:
                t_first = time.monotonic() - t0
    except RuntimeError as ex:
        die("inference failure: %s" % ex)
    finally:
        client.close()

    dt = time.monotonic() - t0
    acc1 = 100.0 * top1_hits / ok if ok else 0.0
    acc5 = 100.0 * top5_hits / ok if ok else 0.0
    rate = ok / (dt - (t_first or 0.0)) if ok and dt > (t_first or 0.0) else 0.0

    print()
    print("images          : %d (%d classified, %d skipped)" % (total, ok, skip))
    print("top-1 accuracy  : %d/%d = %.2f%%" % (top1_hits, ok, acc1))
    print("top-5 accuracy  : %d/%d = %.2f%%" % (top5_hits, ok, acc5))
    print("throughput      : %.2f img/s (server start %.2f s, total %.1f s)"
          % (rate, t_first or 0.0, dt))
    if args.show_errors > 0 and bad_rows:
        print("misclassified (top-1) - up to %d of %d shown:" %
              (args.show_errors, ok - top1_hits))
        for base, truth_name, top1, pred_name, p, in_top5 in bad_rows:
            print("  %s: expected %-38s got %s (%s) p=%.3f%s"
                  % (base, truth_name, labels[top1][0], pred_name, p,
                     "" if in_top5 else "  [also a top-5 miss]"))
    print()
    sys.exit(1 if (args.fail_under > 0 and acc1 < args.fail_under) else 0)


if __name__ == "__main__":
    main()
