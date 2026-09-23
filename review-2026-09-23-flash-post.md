# Post-review of commit 8bbddac — "resolve all 73 findings"

**Date:** 2026-09-23 (post-fix pass)
**Scope:** every file changed in `8bbddacffac1f706e31bd6e1d33c6ec3d9f1dc8d`, reviewed
for correctness and consistency. No code was changed during the review itself.
**Status: all 19 findings resolved in the follow-up fix pass on branch
`review-2026-09-23-flash-post`** - see "Resolution" at the end for per-item
fixes and verification evidence.
**Method:** full diff review, protocol cross-checks (client parser vs server
emitter), exit-code audit against every `return`/`exit` in the three servers,
live Python runs against fake servers, unit-level experiments for each suspect
path, `py_compile`, CMake probe/`SHARED_INCLUDE_DIR` layout verification, and
an inventory of what `scripts/verify.sh` does and does not cover.

**Headline:** most of the fix pass is solid and internally consistent (the
C++ refactor, protocol audit, exit codes, CMake dedup, Dockerfile wiring,
label parsing, fetch guard, `--show-errors` semantics all check out). However
the commit introduced **three functional regressions/bugs** that its own
verification strategy missed, because the new Python clients' *default*
invocation paths were never exercised:

1. `seg_stream.py` is **completely broken in its default mode** — every
   documented command crashes before the server starts.
2. `LinePipe.read_exact()` **over-reads binary payloads and discards the
   surplus**, breaking the seg frame protocol (proven live).
3. `webcam_mobilenet.py` feeds **absolute timestamps** into `windowed_fps()`,
   which expects per-frame durations — fps is always ~1e-5 (displays `0.0`).

---

## A. Critical findings (broken functionality)

### A1. `seg_stream.py` default server command is broken — every documented usage crashes

`examples/deeplab-seg/seg_stream.py:40`:

```python
DEFAULT_SERVER_CMD = "bash %s/infer-seg-server.sh" % os.path.join(HERE, "infer-seg-server.sh")
```

fed into Popen as a single argv token (`seg_stream.py:119`: `[server_cmd, backend, device]`).
Three compounding defects:

* **Path is doubled** — the format template already contains
  `/infer-seg-server.sh` *and* `os.path.join(HERE, "infer-seg-server.sh")`
  appends it again: `…/infer-seg-server.sh/infer-seg-server.sh`.
* **The `bash ` prefix corrupts argv** — `"bash /path/script.sh"` becomes one
  executable-token; `Popen` treats it as the file name → `FileNotFoundError:
  [Errno 2] No such file or directory: 'bash …/infer-seg-server.sh/infer-seg-server.sh'`.
  Proven live on this machine (default `--file … --frames 1 --headless` run).
* **The docstring usage `seg_stream.py my-server.sh host MYRIAD` is invalid** —
  argparse's single positional consumes only `my-server.sh`; the remaining
  `host MYRIAD` tokens are rejected: `error: unrecognized arguments: host MYRIAD`.
  (Proven with a minimal argparse repro.)

**Regression:** the pre-fix version handled this correctly via
`args.server_cmd.split()` (`git show 8bbddac~1:…/seg_stream.py` line 170). The
rewrite dropped the split and introduced the path duplication.

**Why it was missed:** the fix-pass verification ran `seg_stream.py` only with
an *explicit* single-token fake `server_cmd` positional; the default value was
never executed. Every example in `examples/deeplab-seg/README.md` (lines 60–63:
`seg_stream.py --headless`, `--file img.ppm`, `--mask-out mask.ppm`) hits the
broken default.

**Related:** `infer-seg-server.sh` is mode **644** while its siblings
`infer-ssd-server.sh` and `infer-server.sh` are **755**. This inconsistency is
the reason the doomed `bash` prefix existed (a plain path in argv[0] would hit
`PermissionError`). Any fix must pick one: `chmod +x` the script and use the
bare path (matching the ssd client), or split the command (`shlex.split`) with
the `bash` prefix.

- [x] Fix `DEFAULT_SERVER_CMD` (single filename, correct path)
- [x] Make Popen argv construction match (bare executable path with mode 755, or `shlex.split(server_cmd)`)
- [x] Fix or remove the `my-server.sh host MYRIAD` docstring usage (it contradicts argparse)
- [x] `chmod 755 examples/deeplab-seg/infer-seg-server.sh` (sibling consistency)
- [x] Verify: run `seg_stream.py --headless` and each README example

### A2. `LinePipe.read_exact()` over-reads and discards the surplus — breaks the MASK/END framing

`examples/mobilenet_client.py:126-134`:

```python
def read_exact(self, n):
    out = bytes(self.buf[:n])
    self.buf = self.buf[n:]
    while len(out) < n:
        chunk = os.read(self.fd, max(n - len(out), 4096))
        ...
        out += chunk
    return bytes(out)
```

`os.read(fd, max(n - len(out), 4096))` requests **at least 4096 bytes**. When
the payload body arrives in a separate pipe write (or the last needed slice is
≤ 4092 bytes and only those bytes plus the following text lines are buffered),
the read consumes **more than n** and the surplus is *returned as data* instead
of being pushed back into `self.buf`.

Proven live with a fake seg server (8-byte mask): `read_exact(8)` returned
**12 bytes** — `0c000c000c000c00 454e440a` — i.e. the mask **plus the `END\n`
protocol line**. The client then sees EOF at the next `readline` and fails with
"server closed stdout mid-frame".

Against the real `seg_detect` server the effect is timing/alignment-dependent
(it writes text+mask+`END` in one flush, so it happens when the final chunk
boundary lands unfavourably): when the remaining mask bytes are ≤ 4092 and the
pipe holds exactly those + `END\n`, the `END` is swallowed into the mask and
the next `readline` blocks until `--request-timeout` (60 s stall + lost frame).
At the documented webcam resolution (640×480 → 307200-byte mask, read in
64 KiB `_fill` chunks + 4096-byte `read_exact` chunks) the alignment varies per
frame, so this is an intermittent latent bug, not a theoretical one.

Secondary: `out += chunk` is O(n²) byte concatenation for a ~300 KB payload
(use a `bytearray`); and `read_exact` ignores `request_timeout` entirely — a
server that emits `MASK` then hangs blocks the client forever (contrast
`readline`, which honours the timeout).

- [x] Fix `read_exact` to never exceed n (or return the surplus to `self.buf`)
- [x] Add a timeout to `read_exact` consistent with `--request-timeout`
- [x] Add a fake-server unit test that forces text/binary chunk interleave (the repro in this review fails without the fix)
- [x] Consider bytearray accumulation for large masks (nit)

### A3. `webcam_mobilenet.py` feeds absolute timestamps to `windowed_fps` — fps is always wrong

`examples/webcam/webcam_mobilenet.py:149`:

```python
frame_times.append(time.monotonic())          # absolute stamp
...
fps = windowed_fps(frame_times)
```

`windowed_fps(times)` computes `len(tail) / sum(tail)` and its own docstring
documents `times` as **per-frame elapsed seconds**. `ssd_stream.py` and
`seg_stream.py` correctly append `time.monotonic() - t0` durations;
`webcam_mobilenet.py` appends absolute monotonic stamps. Measured with the
real `windowed_fps`: 10 stamps 0.3 s apart at 100000 s uptime → **fps = 1.0e-5**
(displayed as `fps=0.0`), vs the correct `3.33` for the durations form. This
directly contradicts the review item the fix was implementing ("three clients,
two conventions; pick one — windowed fps").

Undetected because no verify/CI path asserts the webcam client's fps value.

- [x] Append per-frame duration (e.g. `time.monotonic() - t_frame0`) instead of the absolute stamp
- [x] Add a tiny unit check pinning `windowed_fps` semantics (durations, not stamps) so the three clients cannot drift again

---

## B. Medium findings

### B1. `ssd_stream.py` DET loop regressed to an uncaught `IndexError` on protocol violations

`examples/ssd-detect/ssd_stream.py:126`: `line.split("DET", 1)[1]` raises
`IndexError` when the line does not contain `DET` at all (server garbage,
partial line after a crash). The pre-fix code checked `p[0] != "DET"` and
raised a clean `RuntimeError("unexpected server line: …")`; `main()` only
catches `RuntimeError`, so a malformed line now yields a traceback.
Well-formed single- and multi-word labels parse correctly (verified:
`cat`, `fire hydrant`, `traffic light` all parse to the right 6 fields).
Same family: a malformed `FRAME` line hits `float(parts[3])` → uncaught
`ValueError`; `seg_stream.py`'s `float(tok[3])` likewise.

- [x] Guard the split (`parts = line.split(); if parts[0] != "DET": raise RuntimeError(...)`) or wrap the parse in try/except → RuntimeError

### B2. `frame_utils.hpp bilinearResize` reads out of bounds for 1-pixel-wide/tall sources

`frame_utils.hpp:85/90`: `y0 = max(0, min(sh - 2, floor(fy)))`, `y1 = y0 + 1`.
When `sh == 1` (or `sw == 1`): `min(-1, …) → max(0, …) = 0`, `y1 = 1` → indexes
row 1 of a 1-row buffer → OOB read (UB). The stream protocol validation accepts
`fw, fh ≥ 1`, so a 1×N stream frame triggers it. (The removed ssd-original
`y1 = min(sh-1, y0+1)` formulation was safe here; the removed seg-original was
also unsafe, differently.) Not reachable via current clients, but the shared
helper should be safe for everything its own callers validate.

- [x] Handle `sw == 1 or sh == 1` explicitly (clamp `y1 = min(sh-1, y0+1)` after clamping y0)

### B3. Upscale border semantics silently changed vs the stated OpenCV reference

The unified `bilinearResize` adopted the **seg** variant (indices always
distinct → extrapolating weights → clamp value to [0,255]) rather than the
**ssd** variant (duplicate border row → never extrapolates → never overshoots,
matching `cv::INTER_LINEAR`). The header comment still claims "matches OpenCV
INTER_LINEAR". For downsampling (all verified cases: 768×576→300×300,
768×576→513×513, webcam→513) the two are identical, so verified numbers are
safe; but for upsampling (a small image file fed to `--image`, or a webcam
frame smaller than the model input) border rows now differ from the previously
documented reference behaviour. Worth a comment fix or an explicit decision,
not a silent mix.

- [x] Reconcile the comment with the semantics (or pick the non-extrapolating OpenCV-faithful variant)

### B4. Commit/review claims vs reality (process consistency)

The commit message and `review-2026-09-23-flash.md` "Resolution notes" state
"Python clients verified against fake servers + corrected protocol tests".
Post-review shows the fake-server verification skipped: the `seg_stream.py`
**default** command (A1), the MASK/END interleave case (A2), and the webcam
fps semantics (A3). The review file's "all 73 items resolved" header and the
commit's "Verified:" paragraph are therefore over-claimed at the Python layer.
(C++ verification claims held up under audit: exit-code tables match every
`return`, protocol lines match client parsers, all unit tests pass.)

Related claim inconsistency: the review/commit describe a shared CLI convention
"positional server script + --backend/--device/--request-timeout" for **all
three** clients — true only for `ssd_stream.py`/`seg_stream.py`.
`webcam_mobilenet.py` and `accuracy_test.py` hard-wire the launcher through
`mobilenet_client.INFER_SERVER` (`MyriadClient` has no server-path parameter
at all). Either add the positional to `MyriadClient`/webcam or state the
convention applies to the two stream clients.

- [x] Correct the review file's verification status (and, if amended history is desired, the commit message)
- [x] Decide the `MyriadClient` server-path convention and align webcam/accuracy clients or the docs

### B5. `verify.sh` covers none of the Python clients

`scripts/verify.sh` exercises only the C++ binaries via `run.sh` (no
`seg_stream`/`ssd_stream`/`webcam_mobilenet` invocation, no fps assertion).
That is why all three critical findings survived a full PASS verification run.
A `WITH_DEVICE`-independent section running the three clients against
recorded fake-server transcripts (deterministic, no device needed) would have
caught A1 and A2 outright.

- [x] Add a device-free Python-client smoke section (fake-server transcripts) to verify.sh or ci/

---

## C. Minor findings (nits, docs, consistency)

- [x] C1 `seg_stream.py:13` docstring usage `my-server.sh host MYRIAD` — invalid with argparse (see A1); a single-token script path is the only accepted form.
- [x] C2 `ssd_stream.py:176` `frame_source` docstring says "Yields **BGR** uint8 frames"; it yields **RGB** in every branch (the values are right, the comment is stale — exactly the bug class the fix pass was chasing).
- [x] C3 Dead code: `seg_stream.py:215` `print_classes()` never called; `accuracy_test.py:114/144` `top5_misses` counted but never printed; `ssd_stream.py:34` `import select` now unused.
- [x] C4 `accuracy_test.py` file discovery: `os.listdir` on a missing dir raises a raw `FileNotFoundError` traceback (proven); the old glob-based discovery died cleanly with the "no images — run fetch-sample-images.sh" message. Add an `isdir` check.
- [x] C5 `ssd_stream.py:103` / `seg_stream.py` stdin failure message says "before the **first** response" but fires on any frame — misleading wording on frame N.
- [x] C6 `ssd-detect/README.md` never documents the new positional server argument (the seg README does, line 69). Symmetry + the removal of `--server-cmd` should be mentioned in the ssd README too.
- [x] C7 `LinePipe` docstring: "readline returns … `None` on clean EOF" is only true without an attached `proc`; with a proc (all real uses) EOF raises `RuntimeError`. Clarify.
- [x] C8 `seg_stream.py` headless status prints `sum(frame_times)` as the elapsed `t`, while `ssd_stream.py` prints `monotonic() - t_start`; trivially different conventions for the same column name.
- [x] C9 `webcam_mobilenet.py:150` `if len(frame_times) >= 2` guard is unnecessary — `windowed_fps` handles a single duration; harmless but confusing next to the A3 bug.
- [x] C10 `mobilenet_client.py:244` `server_exit_meaning(None)` → "unknown" when a pipe dies while the server is still alive; the message "exit code None" is ugly. Cosmetic.
- [x] C11 `LinePipe` unit behaviour: the exited-proc drain does a single `_fill()`; a server emitting a large trailing burst could need repeated drains (bounded by protocol here — fine, but worth a comment).

---

## D. Verified-good (checked, consistent, keep as is)

* **ssd protocol end-to-end:** server emitters (`FRAME/DET/END/ERROR`) vs
  client parsers agree, including the multi-word-label fix (`fire hydrant`,
  `traffic light` parse correctly in live fake tests).
* **seg protocol:** `FRAME <w> <h> <total> <infer>`, `CLASSES n`,
  `CLASS id name… pixels` (token-based parse handles multi-word names),
  `MASK fw fh` + LE u16, `END`, `ERROR` — client and `seg_detect.cpp` match;
  mask is resized to the *client* frame size on the server, and the client
  reads exactly the announced byte count (modulo A2).
* **Exit-code audit:** every `return N`/`exit(N)` in `ssd_detect.cpp`,
  `seg_detect.cpp`, `mobilenet_server.cpp` matches each `usage()`/`--help`
  table and the documented exit codes (0/1/2/3/4); `--help` exits 0 at any
  position in all three binaries and all four shell scripts (verified by
  running them).
* **Label loaders** (`ssd_postprocess.hpp`, `seg_postprocess.hpp`):
  whitespace-tolerant, multi-word names preserved, junk lines skipped,
  `seg` empty-file throw in the right place.
* **`writeMaskPpm` / `ClassMap`:** `ids` is `uint8` → the 1-byte/pixel P6 fix
  is genuinely correct.
* **mobilenet_server.cpp hardening:** FP16 input check before the `uint16_t`
  cast and the `n != outElems` check before the memcpy are both in the right
  order; `MyriadClient` keeps its own timeout-aware `_read_exact` (binary
  protocol — correctly *not* using `LinePipe`).
* **CMake refactor:** the `../../cmake` → `../cmake` probe and the
  `../../half.hpp` → `../half.hpp` auto-detect resolve in both layouts for all
  five apps (checked per-directory, confirmed by the passing Docker build
  stages); the 5-library link group and rpath logic matches the old
  per-app files; all `SHARED_INCLUDE_DIR` target includes present, including
  for the device-free test targets.
* **Dockerfile:** `COPY cmake /work/cmake` in all five app stages,
  `COPY frame_utils.hpp` in ssd+seg, no trailing-comment COPY traps.
* **Include migration:** zero remaining `../half.hpp` / `../device_probe.hpp`
  references anywhere (grep-verified); `ssd_postprocess.hpp` doesn't need
  half.hpp at all.
* **`fetch-sample-images.sh`** non-empty-non-git refusal guard works and keeps
  `-h/--help` at any position; **`accuracy_test.py --show-errors`** semantics
  (all top-1 misses up to the cap, `[also a top-5 miss]` tag, "up to N of M
  shown") are correct, and the docstring's 81.4% figure matches the README's
  verified table.
* **Launchers:** three-way `auto` chain (host → docker → in-image fallback)
  implemented consistently in all three; V4L2 cgroup rule only in the webcam
  launcher is appropriate (only that flow ever touches `/dev/video*` in a
  container).
* **`ssd_stream.py --file` RGB fix** behaves exactly as claimed (live fake
  test reproduced the four-detection reference listing format, multi-word
  labels included).

---

## E. Suggested fix order and regression tests

1. **A1** (seg default crash) — one-line constant + argv construction + `chmod 755`;
   verified by running every README example line against a fake server.
2. **A2** (`read_exact`) — cap the read at n / return surplus to `self.buf`;
   verified by the fake-server interleave repro (mask body arrives in a second
   pipe write) *and* a real seg run at 640×480 for ≥ 20 frames.
3. **A3** (fps stamps) — append durations; verified by a unit check that
   `windowed_fps([0.3]*10) == 3.33` and a stubbed webcam run.
4. B1–B2 (protocol-violation robustness + 1-pixel resize OOB) with two new
   unit checks.
5. B4/B5 + C-section doc/nit cleanup in one pass; add the device-free Python
   client fake-server section to `verify.sh`/`ci` so this class of gap cannot
   recur.

**Total: 3 critical, 5 medium, 11 minor findings.** None of the previously
verified C++/hardware results are invalidated; all three critical findings are
in the Python layer introduced or retouched by this commit.

---

## Resolution (fix pass 2026-09-23, branch `review-2026-09-23-flash-post`)

All 19 findings fixed.  Per item:

* **A1** `seg_stream.py:DEFAULT_SERVER_CMD` is now the single executable path
  `os.path.join(HERE, "infer-seg-server.sh")`; `infer-seg-server.sh` chmod'd
  755 (matching its siblings); docstring usage replaced with the accurate
  `<launcher> <backend> <device>` convention; a `server launcher not found`
  pre-check added.  **Verified on hardware**: `seg_stream.py --headless
  --file vendor/models/images/dog_ssd.ppm --frames 2` (default launcher, no
  positional arg) ran 2 consecutive frames: background 77.8% / bicycle 11.2%
  / dog 7.5% / car 3.2% / cat 0.3% - exactly the verified reference
  distribution.
* **A2** `LinePipe.read_exact(n, timeout)` now requests exactly `n - len(out)`
  bytes per `os.read` (surplus stays in the pipe/buffer), accumulates in a
  `bytearray`, and honors a timeout (`MASK` body reads now use
  `--request-timeout`).  The seg client passes `timeout=self.request_timeout`.
  **Verified**: unit tests for both interleave layouts (body+END in one write;
  remainder already in buf) plus a 15-frame live webcam run at 640x480
  (614,400-byte mask per frame) with zero stalls, steady 1.4-1.5 fps, and a
  fake-server test asserting a hung MASK fails at `--request-timeout` 2 s
  instead of hanging.
* **A3** `webcam_mobilenet.py` appends `time.monotonic() - t_frame0`
  (per-classified-frame duration) - same round-trip convention as the ssd/seg
  clients.  **Verified live**: steady `fps=21.2` against `infer=45.2 ms`
  (1/0.047 s), instead of the old `fps=0.0`.
* **B1** ssd client validates the leading token (`DET`), the `FRAME` field
  count, and numeric fields - all malformed lines now raise the clean
  `unexpected server line` RuntimeError; the seg client got the same guards
  for FRAME/CLASS/MASK.  **Verified**: fake-garbage-server test asserts exit
  != 0, message present, and no traceback.
* **B2** `bilinearResize` clamps `y1 = min(sh-1, y0+1)` / `x1 = min(sw-1,
  x0+1)` and special-cases 1-pixel dimensions (row/col duplicated, weights sum
  to 1).  **Verified**: ASAN-instrumented probe (`resize_probe` in the test
  driver) runs 1x1, 2x1, 1x2 sources clean - and the probe itself caught a
  real heap-overflow *in the probe*, demonstrating the harness works.
* **B3** Header comment now states the truth: OpenCV `INTER_LINEAR`-faithful
  for downsampling (all verified pipeline paths); index-pair clamping with
  extrapolating weights + value clamp for upsampling borders.  The small-case
  outputs (2x1->4x2, 1x2->2x4, 2x2->1x1 average) are pinned in the probe so
  the semantics cannot drift silently.
* **B4** Convention unified for real: `MyriadClient` takes an optional
  `server_script`, and `webcam_mobilenet.py` / `accuracy_test.py` gained the
  same optional first positional argument as the two stream clients
  (documented in both READMEs; verified with a fake mobilenet-protocol server
  through `accuracy_test.py <fake> --limit 6`).  The over-claimed
  "verified against fake servers" paragraph in `review-2026-09-23-flash.md`
  carries an explicit correction block (the pushed commit `8bbddac` itself
  was left unamended - history is public).
* **B5** New device-free gate `scripts/test-python-clients.sh` (unit + fake-
  server end-to-end + default-launcher sanity + ASAN resize probe), wired as
  **section 5b of `verify.sh`**; `PY_CLIENTS_RESULT=PASS`.
* **C1** docstring usage fixed (see A1).  **C2** `frame_source` docstring now
  says RGB.  **C3** dead `print_classes`, dead `top5_misses` counter, and the
  unused `import select` all removed.  **C4** `accuracy_test.py` dies cleanly
  with a friendly message when the images dir is missing (`isdir` check).
  **C5** stdin-failure messages no longer claim "before the first response".
  **C6** ssd README documents the positional launcher + invocation signature.
  **C7** `LinePipe.readline` docstring documents the attached-proc EOF
  behaviour.  **C8** seg_stream headless status prints elapsed-since-start
  like ssd.  **C9** the `len(frame_times) >= 2` guard removed.  **C10** a
  still-running server is reported as "closed its stdin pipe but has not
  exited yet" instead of "exit code None".  **C11** EOF with an attached
  process now waits up to 1 s for the child to be reaped so the exit code is
  reported deterministically (fake-exit-server test asserts `exited with
  code 2`).

Verification summary: `scripts/test-python-clients.sh` all green; `ssd_test`
(22 checks), `seg_test` (33 checks) and `test_half` rebuilt and passing from
a plain checkout; all three C++ apps rebuilt from a plain checkout with
`--help` exit 0; `py_compile` clean on all five Python files; `bash -n` clean
on both scripts; `--help` exit 0 on all four clients; live MA2450 runs above
for the three critical fixes.

---

## Follow-up finding (user report, after the fix pass)

### F1 - stream clients' fps ramps up (0.7 -> 4.2 -> jump to 10.8)

`ssd_stream.py --headless` printed `fps` climbing step-by-step for the first
~10 frames and then jumping to the steady rate.  Cause: `windowed_fps`
averages the last 10 per-frame round-trips, and frame 1's round-trip
contains the one-time MYRIAD compile (~1.65 s).  The displayed value is
`k / (1.75 + (k-1)*0.092)` until the outlier falls out of the window at
frame 11 - a window-average artifact, not device warm-up.  (The
`windowed_fps` docstring even claimed it "excludes the one-time server
compile", which was not true of the callers.)

Fix (the "do not include the warmup time" option): all three stream
clients now treat the first round-trip as warm-up - it is reported on an
explicit `warmup: first ... N ms (includes device compile)` note, the fps
column shows `warm` for that frame, and it is excluded from the window;
`windowed_fps`'s docstring now states this caller contract.  README output
samples updated.  `scripts/test-python-clients.sh` asserts the contract:
fake-server frame 1 must show `fps= warm` and frame 2 a numeric fps.

Verified live on the MA2450: ssd webcam run `warmup: 1520 ms` then
`fps=10.8` from frame 2 onward (was 0.7 ramping to 10.8); seg file mode
`warm` then 1.5 fps from frame 2; webcam_mobilenet `warmup: 1657 ms` then
21.3 fps.  Device-free driver: PY_CLIENTS_RESULT=PASS with the new asserts.
