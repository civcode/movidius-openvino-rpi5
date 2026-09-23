#!/usr/bin/env bash
# Device-free protocol/unit tests for the Python example clients.
#
# Drives ssd_stream.py and seg_stream.py against fake servers (plain python
# scripts that speak the FRAME/DET/END and FRAME/CLASSES/MASK/END protocols,
# including the text/binary interleave that exposed the LinePipe.read_exact
# over-read) plus pure unit checks (LinePipe, windowed_fps semantics,
# frame_utils.hpp resize bounds) and default-launcher sanity (the broken
# seg_stream DEFAULT_SERVER_CMD of the first fix pass).
#
# No MYRIAD device, no Docker.  Skips (exit 0) when python3+numpy or g++ are
# unavailable, so it can run anywhere verify.sh runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "SKIP: python3 not found"; exit 0; }
"$PY" -c 'import numpy' 2>/dev/null || { echo "SKIP: python3 numpy not installed"; exit 0; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pass() { echo "ok - $*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

# ------------------------------------------------------------------ fixtures
"$PY" - "$WORK" <<'PYEOF'
import os, sys
work = sys.argv[1]

with open(os.path.join(work, "tiny.ppm"), "wb") as f:
    f.write(b"P6\n2 2\n255\n" + bytes(range(12)))

# ssd fake: multi-word label, multi-frame capable
open(os.path.join(work, "fake_ssd.py"), "w").write('''#!/usr/bin/env python3
import sys, struct
while True:
    hdr = sys.stdin.buffer.read(8)
    if len(hdr) < 8:
        break
    w, h = struct.unpack("<II", hdr)
    sys.stdin.buffer.read(w * h * 3)
    sys.stdout.write("FRAME %d %d 91.5\\n" % (w, h))
    sys.stdout.write("DET fire hydrant 0.71 10 20 30 40\\n")
    sys.stdout.write("DET cat 0.63 1 2 3 4\\n")
    sys.stdout.write("END\\n")
    sys.stdout.flush()
''')

# ssd fake that emits a garbage protocol line after FRAME
open(os.path.join(work, "fake_ssd_bad.py"), "w").write('''#!/usr/bin/env python3
import sys, struct
hdr = sys.stdin.buffer.read(8)
sys.stdout.write("FRAME 2 2 5.0\\nGARBAGE not the protocol\\n")
sys.stdout.flush()
''')

# ssd fake that exits with code 2 without answering
open(os.path.join(work, "fake_ssd_exit.py"), "w").write('''#!/usr/bin/env python3
import sys
sys.stdin.buffer.read(8)
sys.exit(2)
''')

# seg fake: header, CLASSES, MASK header, body and END are each SEPARATE
# pipe writes (the interleave that must not let read_exact swallow END)
open(os.path.join(work, "fake_seg.py"), "w").write('''#!/usr/bin/env python3
import sys, struct, time
while True:
    hdr = sys.stdin.buffer.read(8)
    if len(hdr) < 8:
        break
    w, h = struct.unpack("<II", hdr)
    sys.stdin.buffer.read(w * h * 3)
    sys.stdout.write("FRAME %d %d 10.0 5.0\\n" % (w, h))
    sys.stdout.write("CLASSES 2\\n")
    sys.stdout.write("CLASS 12 dog %d\\n" % (w * h // 2))
    sys.stdout.write("CLASS 11 dining table %d\\n" % (w * h // 4))
    sys.stdout.write("MASK 2 2\\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(struct.pack("<4H", 1, 2, 3, 4))
    sys.stdout.flush()
    sys.stdout.write("END\\n")
    sys.stdout.flush()
''')

# seg fake that announces MASK and then never sends the body (timeout path)
open(os.path.join(work, "fake_seg_hang.py"), "w").write('''#!/usr/bin/env python3
import sys, time
sys.stdin.buffer.read(8)
sys.stdout.write("MASK 2 2\\n")
sys.stdout.flush()
time.sleep(30)
''')
PYEOF
chmod +x "$WORK"/fake_*.py

# ------------------------------------------------------------------ unit: LinePipe + fps
"$PY" - <<'PYEOF'
import os, struct, sys, time
sys.path.insert(0, "examples")   # cwd is the repo root
from mobilenet_client import LinePipe, server_exit_meaning, windowed_fps

# 1) read_exact must stop at n even when the payload and the following
#    protocol line arrive in the SAME later write (over-read regression).
r, w = os.pipe()
try:
    os.write(w, b"MASK 2 2\n")
    p = LinePipe(r)
    assert p.readline() == "MASK 2 2"
    os.write(w, struct.pack("<4H", 1, 2, 3, 4) + b"END\n")   # body+END in one write
    body = p.read_exact(8, timeout=5)
    assert body == struct.pack("<4H", 1, 2, 3, 4), body
    assert p.readline(timeout=5) == "END"
finally:
    os.close(r); os.close(w)

# 2) buffered-remainder path: payload already in buf, END after it
r, w = os.pipe()
try:
    os.write(w, b"MASK 1 1\n\x07\x00END\n")
    p = LinePipe(r)
    assert p.readline() == "MASK 1 1"
    assert p.read_exact(2, timeout=5) == b"\x07\x00"
    assert p.readline(timeout=5) == "END"
finally:
    os.close(r); os.close(w)

# 3) read_exact honors a timeout (announced payload never arrives)
r, w = os.pipe()
try:
    os.write(w, b"M")            # keeps the pipe open, no payload
    p = LinePipe(r)
    t0 = time.monotonic()
    try:
        p.read_exact(1000, timeout=1)
        raise AssertionError("expected TimeoutError")
    except TimeoutError:
        assert time.monotonic() - t0 < 3
finally:
    os.close(r); os.close(w)

# 4) windowed_fps contract: PER-FRAME DURATIONS (seconds), most recent last
assert windowed_fps([]) == 0.0
assert abs(windowed_fps([0.3] * 10) - 10 / 3.0) < 1e-9
assert abs(windowed_fps([0.5, 0.5, 0.1, 0.1, 0.1]) - 5 / 1.3) < 1e-9
# (absolute timestamps instead of durations would yield ~1e-5 here - the
#  webcam_mobilenet regression this pins down)
stamps = [100000.0 + 0.3 * i for i in range(10)]
assert windowed_fps(stamps) < 1e-3   # proof: stamps instead of durations collapse fps (the webcam bug)

# 5) exit-code table
assert "device" in server_exit_meaning(2)
assert "model" in server_exit_meaning(3)
print("unit checks passed")
PYEOF
pass "LinePipe/read_exact/windowed_fps unit tests"

# ------------------------------------------------------------------ unit: frame_utils resize
if command -v g++ >/dev/null 2>&1; then
    cat > "$WORK/resize_probe.cpp" <<'CPPEOF'
#include "frame_utils.hpp"
#include <cassert>
#include <cstdio>
int main() {
    using namespace frameutils;
    // 1x1 source -> 2x2 upscale: the OOB regression read past the single row
    std::vector<uint8_t> one = {10, 20, 30};
    std::vector<uint8_t> dst;
    bilinearResize(one.data(), 1, 1, dst, 2, 2);
    assert(dst.size() == 12);
    for (int i = 0; i < 12; ++i)
        assert(dst[i] == one[i % 3]);

    // 1-pixel-tall strip -> 4x2 upscale: single row duplicated (in-range),
    // x samples half-pixel positions -0.25,0,0.25,0.75 between the two cols
    // (documented border semantics: index pair clamped, weights extrapolate,
    // value clamped).  Pinned so the semantics cannot drift silently.
    std::vector<uint8_t> strip = {1, 2, 3, 4, 5, 6};   // 2x1
    bilinearResize(strip.data(), 2, 1, dst, 4, 2);
    assert(dst.size() == 24);
    {
        const uint8_t row[12] = {0, 1, 2,  2, 3, 4,  3, 4, 5,  5, 6, 7};
        for (int y = 0; y < 2; ++y)
            for (int i = 0; i < 12; ++i)
                assert(dst[y * 12 + i] == row[i]);
    }

    // 1-pixel-wide strip -> 2x4 upscale: single col duplicated; y weights
    // -0.25, 0.25, 0.75, 1.25 between the two rows
    std::vector<uint8_t> col = {7, 8, 9, 10, 11, 12};  // 1x2
    bilinearResize(col.data(), 1, 2, dst, 2, 4);
    assert(dst.size() == 24);
    {
        const uint8_t rows[4][6] = {
            {6, 7, 8,   6, 7, 8},
            {8, 9, 10,  8, 9, 10},
            {9, 10, 11,  9, 10, 11},
            {11, 12, 13, 11, 12, 13},
        };
        for (int y = 0; y < 4; ++y)
            for (int i = 0; i < 6; ++i)
                assert(dst[y * 6 + i] == rows[y][i]);
    }

    // downsampling semantics: 2x2 -> 1x1 is the plain average, and a
    // saturated white upscale must not wrap (clamp holds)
    std::vector<uint8_t> quad = {0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255};
    bilinearResize(quad.data(), 2, 2, dst, 1, 1);
    assert(dst.size() == 3 && dst[0] == 128);
    std::vector<uint8_t> white(100 * 100 * 3, 255);
    bilinearResize(white.data(), 100, 100, dst, 137, 137);
    for (auto v : dst) assert(v == 255);
    std::printf("resize checks passed\n");
    return 0;
}
CPPEOF
    if printf 'int main(){return 0;}' | g++ -x c++ -fsanitize=address -o /dev/null - 2>/dev/null; then
        g++ -std=c++14 -O1 -fsanitize=address -I "$ROOT" -o "$WORK/resize_probe" "$WORK/resize_probe.cpp"
        ASAN_OPTIONS=detect_leaks=0 "$WORK/resize_probe"
    else
        g++ -std=c++14 -O1 -I "$ROOT" -o "$WORK/resize_probe" "$WORK/resize_probe.cpp"
    fi
    "$WORK/resize_probe"
    pass "frame_utils.hpp resize bounds (1px sources, clamped upscale)"
else
    echo "note: g++ not found - resize bounds check skipped"
fi

# ------------------------------------------------------------------ default launchers
"$PY" - <<'PYEOF'
import os
import sys
sys.path.insert(0, "examples/deeplab-seg")
sys.path.insert(0, "examples/ssd-detect")
import seg_stream, ssd_stream
for name, cmd in (("seg_stream.DEFAULT_SERVER_CMD", seg_stream.DEFAULT_SERVER_CMD),
                  ("ssd_stream.INFER_SERVER", ssd_stream.INFER_SERVER)):
    assert " " not in cmd, "%s is not a single path: %r" % (name, cmd)
    assert os.path.isfile(cmd), "%s missing: %s" % (name, cmd)
    assert os.access(cmd, os.X_OK), "%s not executable: %s" % (name, cmd)
    assert not cmd.endswith(cmd.split("/")[-2]), name  # no path duplication
print("default launcher checks passed")
PYEOF
pass "default launcher scripts (single path, present, mode 755)"

# ------------------------------------------------------------------ ssd_stream e2e
out="$("$PY" examples/ssd-detect/ssd_stream.py "$WORK/fake_ssd.py" \
        --file "$WORK/tiny.ppm" --frames 2 --headless --request-timeout 5 2>/dev/null)"
nframes="$(grep -c '^\[t=' <<<"$out")"
[[ "$nframes" == 2 ]] || fail "ssd_stream: expected 2 frame lines, got $nframes"
grep -q 'fire hydrant 0.71' <<<"$out" || fail "ssd_stream: multi-word label not parsed"
grep -q 'cat 0.63' <<<"$out" || fail "ssd_stream: label list incomplete"
# fps window must exclude the warm-up frame: frame 1 labeled `warm`,
# frame 2 already carries a numeric steady-state fps (no ramp)
grep -q 'fps= warm' <<<"$out" || fail "ssd_stream: first frame not labeled warm"
sed -n '2p' <<<"$out" | grep -Eq 'fps= *[0-9]' || fail "ssd_stream: frame 2 fps not numeric: $(sed -n '2p' <<<"$out")"
pass "ssd_stream fake-server end-to-end (multi-word labels, 2 frames, warm-up excluded)"

# garbage protocol line -> clean inference failure, no traceback
set +e
err="$("$PY" examples/ssd-detect/ssd_stream.py "$WORK/fake_ssd_bad.py" \
        --file "$WORK/tiny.ppm" --frames 1 --headless --request-timeout 5 2>&1 >/dev/null)"
rc=$?
set -e
[[ "$rc" != 0 ]] || fail "ssd_stream: garbage line should fail"
grep -q 'unexpected server line' <<<"$err" || fail "ssd_stream: no clean error for garbage line"
if grep -q 'Traceback' <<<"$err"; then fail "ssd_stream: traceback instead of clean error"; fi
pass "ssd_stream malformed-line handling (clean RuntimeError)"

# server exit -> exit-code explanation
set +e
err="$("$PY" examples/ssd-detect/ssd_stream.py "$WORK/fake_ssd_exit.py" \
        --file "$WORK/tiny.ppm" --frames 1 --headless --request-timeout 5 2>&1 >/dev/null)"
rc=$?
set -e
[[ "$rc" != 0 ]] || fail "ssd_stream: server exit should fail the client"
grep -q 'exited with code 2' <<<"$err" || fail "ssd_stream: exit code not reported: $err"
pass "ssd_stream server-exit diagnostics"

# ------------------------------------------------------------------ seg_stream e2e
out="$("$PY" examples/deeplab-seg/seg_stream.py "$WORK/fake_seg.py" \
        --file "$WORK/tiny.ppm" --frames 2 --headless --request-timeout 5 \
        --mask-out "$WORK/mask.ppm" 2>/dev/null)"
nframes="$(grep -c '^\[frame ' <<<"$out")"
[[ "$nframes" == 2 ]] || fail "seg_stream: expected 2 frame lines, got $nframes"
grep -q 'dog 50.0%' <<<"$out" || fail "seg_stream: CLASS line not parsed"
grep -q 'dining table 25.0%' <<<"$out" || fail "seg_stream: multi-word CLASS name not parsed"
grep -q ' warm fps' <<<"$out" || fail "seg_stream: first frame not labeled warm"
sed -n '2p' <<<"$out" | grep -Eq '[0-9.]+ fps' || fail "seg_stream: frame 2 fps not numeric: $(sed -n '2p' <<<"$out")"
"$PY" - "$WORK/mask.ppm" <<'PYEOF'
import sys
data = open(sys.argv[1], "rb").read()
assert data[:11] == b"P6\n2 2\n255\n", data[:16]
assert len(data) == 11 + 4 and data[11:] == b"\x01\x02\x03\x04", data[11:]
PYEOF
pass "seg_stream fake-server end-to-end (interleaved MASK/END, 2 frames, mask-out)"

# MASK announced, body never arrives -> request-timeout, not a hang
set +e
err="$("$PY" examples/deeplab-seg/seg_stream.py "$WORK/fake_seg_hang.py" \
        --file "$WORK/tiny.ppm" --frames 1 --headless --request-timeout 2 2>&1 >/dev/null)"
rc=$?
set -e
[[ "$rc" != 0 ]] || fail "seg_stream: hung MASK should fail"
grep -q 'payload within 2 s' <<<"$err" || fail "seg_stream: no timeout on hung MASK: $err"
pass "seg_stream MASK timeout honored"

echo
echo "PY_CLIENTS_RESULT=PASS"
