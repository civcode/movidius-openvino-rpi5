#!/usr/bin/env python3
"""
mobilenet_cpu_server.py - CPU inference server for the webcam example
(webcam_mobilenet.py).

Implements exactly the same stdin/stdout protocol as the C++
`mobilenet_server`: repeated 1x3x224x224 float32 (little-endian) input
tensors on stdin (602112 bytes per request), one 1x1000 float32 logits
vector (4000 bytes) on stdout per request, diagnostics on stderr.  The
client does the top-k; this server does no post-processing.  Exits
cleanly with status 0 when stdin reaches EOF.

Runs the model produced by scripts/prepare-mobilenet.sh
(vendor/models/onnx/mobilenetv2-7.onnx, input tensor name "data") with
ONNX Runtime's CPUExecutionProvider - the arm64 CPU backend described in
docs/CPU-BACKENDS.md.  The launcher (examples/webcam/infer-server.sh)
starts this script on arm64 when --device CPU is selected.

Examples:
  python3 mobilenet_cpu_server.py
  python3 mobilenet_cpu_server.py --model /path/to/mobilenetv2-7.onnx
"""

import argparse
import os
import sys
import time

REQUEST_BYTES = 1 * 3 * 224 * 224 * 4   # 602112 - one preprocessed frame
RESPONSE_BYTES = 1000 * 4               # 4000   - raw float32 logits


def write_all(fd, data: bytes) -> None:
    off = 0
    while off < len(data):
        off += os.write(fd, data[off:off + 65536])


def read_request(stdin, need: int) -> bytes:
    """Read exactly `need` bytes from stdin; b'' on clean EOF."""
    buf = bytearray()
    while len(buf) < need:
        chunk = stdin.read(need - len(buf))
        if not chunk:
            return bytes(buf)
        buf += chunk
    return bytes(buf)


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(
        description="CPU (ONNX Runtime) inference server for webcam_mobilenet.py")
    ap.add_argument("--model",
                    default=os.path.join(root, "vendor/models/onnx/mobilenetv2-7.onnx"),
                    help="ONNX model path (default: %(default)s)")
    args = ap.parse_args()

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        print(f"missing dependency: {e.name} - run:  pip install "
              f"{'numpy' if e.name == 'numpy' else 'onnxruntime'}", file=sys.stderr)
        return 1

    if not os.path.isfile(args.model):
        print(f"model not found: {args.model} - run ./scripts/prepare-mobilenet.sh",
              file=sys.stderr)
        return 1

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    print(f"mobilenet_cpu_server: ONNX Runtime {ort.__version__}, model {args.model}, "
          f"input '{in_name}', provider CPUExecutionProvider", file=sys.stderr, flush=True)

    req = 0
    while True:
        buf = read_request(sys.stdin.buffer, REQUEST_BYTES)
        if len(buf) < REQUEST_BYTES:
            if buf:
                print(f"mobilenet_cpu_server: truncated request "
                      f"({len(buf)} < {REQUEST_BYTES} bytes), stopping",
                      file=sys.stderr)
            print(f"mobilenet_cpu_server: stdin EOF after {req} request(s)",
                  file=sys.stderr)
            return 0
        tensor = np.frombuffer(buf, dtype="<f4").reshape(1, 3, 224, 224)
        t0 = time.perf_counter()
        logits = sess.run(None, {in_name: tensor})[0].ravel()
        ms = (time.perf_counter() - t0) * 1000.0
        if logits.dtype != "<f4" or len(logits) != RESPONSE_BYTES // 4:
            print(f"mobilenet_cpu_server: unexpected output {logits.dtype} x {len(logits)}",
                  file=sys.stderr)
            return 1
        print(f"mobilenet_cpu_server: request {req}: {ms:.1f} ms",
              file=sys.stderr, flush=True)
        write_all(1, logits.tobytes())
        req += 1


if __name__ == "__main__":
    sys.exit(main())
