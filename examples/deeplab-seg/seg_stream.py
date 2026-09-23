#!/usr/bin/env python3
"""
seg_stream.py -- DeepLabV3 (Pascal VOC) segmentation client.

Sends camera frames (or a single image file) to the C++ seg_detect
server in --stdin mode and renders a per-pixel class map over the frame.

Usage:
  python3 seg_stream.py --server-cmd "bash examples/deeplab-seg/infer-seg-server.sh"
  python3 seg_stream.py --camera 0 --width 640 --height 480
  python3 seg_stream.py --headless            # no GUI: print class stats
  python3 seg_stream.py --file image.ppm      # single image, no camera
  python3 seg_stream.py --mask-out mask.ppm   # also write the class map

The server protocol (see seg_detect.cpp):

  in : uint32 w  uint32 h  (little-endian) + w*h*3 RGB bytes
  out: FRAME  <w> <h> <total_ms> <infer_ms>
       CLASSES <n>
       CLASS  <id> <name> <pixels>
       MASK   <w> <h>  + w*h uint16 LE class ids
       END

Requires numpy + opencv:  pip3 install numpy opencv-python(-headless)
"""

import argparse
import os
import struct
import subprocess
import sys

# Pascal VOC palette (index = class id)
PALETTE = [
    (50, 50, 50),        # 0  background
    (135, 206, 250),     # 1  aeroplane
    (255, 140, 0),       # 2  bicycle
    (255, 99, 71),       # 3  bird
    (100, 149, 237),     # 4  boat
    (153, 50, 204),      # 5  bottle
    (255, 215, 0),       # 6  bus
    (220, 20, 120),      # 7  car
    (255, 184, 100),     # 8  cat
    (144, 238, 144),     # 9  chair
    (101, 67, 33),       # 10 cow
    (210, 180, 140),     # 11 diningtable
    (160, 120, 80),      # 12 dog
    (139, 69, 19),       # 13 horse
    (127, 127, 127),     # 14 motorbike
    (255, 0, 0),         # 15 person
    (0, 150, 60),        # 16 pottedplant
    (255, 255, 240),     # 17 sheep
    (200, 120, 100),     # 18 sofa
    (80, 80, 160),       # 19 train
    (0, 191, 255),       # 20 tvmonitor
]


class SegClient:
    def __init__(self, proc):
        self.proc = proc
        self.stdin = proc.stdin
        self.fd = os.dup(proc.stdout.fileno())
        self._line_buf = bytearray()

    def _readline(self):
        while b"\n" not in self._line_buf:
            chunk = os.read(self.fd, 65536)
            if not chunk:
                return None
            self._line_buf += chunk
        line, self._line_buf = self._line_buf.split(b"\n", 1)
        return line.decode("utf-8", "replace")

    def _read_bytes(self, n):
        buf = bytearray()
        if self._line_buf:
            buf += self._line_buf[:n]
            self._line_buf = self._line_buf[n:]
        while len(buf) < n:
            chunk = os.read(self.fd, max(n - len(buf), 4096))
            if not chunk:
                raise RuntimeError("server closed while reading MASK payload")
            buf += chunk
        return bytes(buf)

    def segment(self, rgb, w, h):
        """Send one RGB frame; returns (classes, mask_bytes, total_ms,
        infer_ms) where classes is a list of (id, name, pixels)."""
        hdr = struct.pack("<II", w, h)
        try:
            self.stdin.write(hdr + rgb.tobytes())
            self.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"server pipe closed: {e}")

        classes = []
        mask_bytes = None
        total_ms = infer_ms = 0.0
        for _ in range(64):
            line = self._readline()
            if line is None:
                raise RuntimeError("server closed stdout mid-frame")
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "FRAME":
                total_ms = float(tok[3])
                infer_ms = float(tok[4])
            elif tok[0] == "CLASSES":
                for _ in range(int(tok[1])):
                    l = self._readline()
                    if l is None:
                        raise RuntimeError("server closed mid-CLASS")
                    p = l.split()
                    classes.append((int(p[1]), p[2], int(p[3])))
            elif tok[0] == "MASK":
                fw, fh = int(tok[1]), int(tok[2])
                mask_bytes = self._read_bytes(fw * fh * 2)
            elif tok[0] == "END":
                return classes, mask_bytes, total_ms, infer_ms
            elif tok[0] == "ERROR":
                raise RuntimeError(line[6:])
        raise RuntimeError("no END line from server")

    def close(self):
        try:
            self.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def overlay(frame_bgr, mask_u16, w, h, alpha=0.4):
    import numpy as np
    m = np.frombuffer(mask_u16, dtype="uint16").reshape(h, w)
    pal = np.array(PALETTE, dtype="uint8")
    colored = pal[m]              # (h, w, 3) BGR-ish RGB from palette
    rgb_frame = frame_bgr[:, :, ::-1]
    out = (alpha * colored + (1 - alpha) * rgb_frame).astype("uint8")
    return out[:, :, ::-1]        # back to BGR for cv2


def print_classes(classes, total_ms, infer_ms, w, h):
    total = w * h
    parts = []
    for cid, name, px in classes:
        parts.append(f"{name} {100.0 * px / total:.1f}%")
    print(f"[frame {w}x{h}] total {total_ms:.0f} ms (infer {infer_ms:.0f} ms): "
          + ", ".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-cmd",
                    default="bash examples/deeplab-seg/infer-seg-server.sh")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--file", help="process a single image file and exit")
    ap.add_argument("--mask-out", help="write the class map of the last frame to this PPM")
    args = ap.parse_args()

    proc = subprocess.Popen(
        args.server_cmd.split(),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=None,  # server diagnostics go to our stderr
        text=False)
    client = SegClient(proc)

    try:
        if args.file:
            import cv2
            import numpy as np
            img = cv2.imread(args.file)
            if img is None:
                sys.exit(f"cannot read {args.file}")
            if img.shape[0] != args.height or img.shape[1] != args.width:
                img = cv2.resize(img, (args.width, args.height))
            rgb = img[:, :, ::-1]
            classes, mask, total_ms, infer_ms = client.segment(
                np.ascontiguousarray(rgb), args.width, args.height)
            print_classes(classes, total_ms, infer_ms, args.width, args.height)
            if args.mask_out:
                m = np.frombuffer(mask, dtype="uint16")
                with open(args.mask_out, "wb") as f:
                    f.write(b"P6\n%d %d\n255\n" % (args.width, args.height))
                    f.write(m.tobytes())
                print(f"wrote class map to {args.mask_out}")
            if not args.headless:
                overlaid = overlay(img, mask, args.width, args.height)
                cv2.imshow("DeepLabV3 segmentation", overlaid)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            import cv2
            import numpy as np
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                sys.exit(f"cannot open camera {args.camera}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

            if args.headless:
                import time
                t_last = time.time()
                n = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    rgb = frame[:, :, ::-1]
                    classes, mask, total_ms, infer_ms = client.segment(
                        np.ascontiguousarray(rgb), args.width, args.height)
                    t_now = time.time()
                    fps = 1.0 / max(t_now - t_last, 1e-6)
                    t_last = t_now
                    n += 1
                    print(f"[{n}] {fps:.1f} fps server {total_ms:.0f} ms "
                          f"(infer {infer_ms:.0f} ms) "
                          + " ".join(f"{name} {100.0*px/(args.width*args.height):.1f}%"
                                    for cid, name, px in classes))
            else:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    rgb = frame[:, :, ::-1]
                    classes, mask, total_ms, infer_ms = client.segment(
                        np.ascontiguousarray(rgb), args.width, args.height)
                    overlaid = overlay(frame, mask, args.width, args.height)
                    txt = " ".join(f"{name} {100.0*px/(args.width*args.height):.0f}%"
                                  for cid, name, px in classes[:5])
                    cv2.putText(overlaid, f"{total_ms:.0f} ms  {txt}",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow("DeepLabV3 segmentation", overlaid)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                cv2.destroyAllWindows()
            cap.release()
    except Exception as e:
        print(f"client error: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
