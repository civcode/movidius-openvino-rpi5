# examples/ review — 2026-09-23 (flash)

Scope: `examples/` (webcam, ssd-detect, deeplab-seg, accuracy-test, shared
client). Bugs, inconsistencies, improvements, and `--help`/usage-text
correctness.

Status: **all 73 items resolved** (fix pass 2026-09-23). Each item below is
checked off; see "Resolution notes" at the end for how each area was fixed
and verified.

Review order: shared client → webcam → ssd-detect → deeplab-seg →
accuracy-test → cross-cutting.

---

## 1. `examples/mobilenet_client.py` (shared MobileNet client)

- [x] **BUG (hardening)** `MyriadClient.infer()` raises a bare
      `BrokenPipeError` when the server died during startup (observed 3x
      this session before the device-probe retry landed). Catch it and
      re-raise as a `RuntimeError` including `self.proc.poll()` exit code
      and the server's exit-code meaning (2 = no device, 4 = bad args),
      so the client explains *why* the pipe closed.
- [x] **BUG (hardening)** `preprocess()` uses `cv2` without a `cv2 is None`
      check: the module imports cv2 optionally, so a missing OpenCV shows
      up as `AttributeError: 'NoneType' object has no attribute 'resize'`
      instead of the intended friendly message. Add a `die()` in
      `preprocess` (or make the optional import mandatory here).
- [x] Protocol docstring matches `mobilenet_server.cpp` (602112 B in /
      4000 B out, float32) — verified against the server source.
- [x] **NIT** `close()` closes `proc.stdin`/`proc.stdout` even when the
      kill path was taken — harmless, but `try/except` around the close
      calls would silence ResourceWarnings on Windows/py3.14.

## 2. `examples/ssd-detect/`

### `ssd_stream.py`

- [x] **BUG (empirically confirmed, common path)** `--file *.ppm` mode
      double-converts channels: `read_ppm()` already returns **RGB**, but
      `frame_source()` unconditionally applies `cv2.cvtColor(BGR2RGB)`
      afterwards — PPM frames reach the RGB-expecting server as **BGR**.
      Proven on this machine (host runtime, dog_ssd.ppm, one frame):
      current code → 3 detections `bicycle 0.97 / car 0.84 / dog 0.84`,
      **cat lost**, boxes shifted ~7 px; patched (swap only the
      `cv2.imread` path) → `bicycle 0.96 / dog 0.88 / car 0.87 /
      cat 0.63`, exactly the verified C++ single-image reference. Fix:
      cvtColor only the jpg/png (`cv2.imread`) branch.
- [x] **BUG (crash on 14 COCO classes)** `detect()` splits `DET` lines on
      whitespace and requires exactly 7 tokens, but `coco.txt` has
      multi-word labels (`traffic light`, `fire hydrant`, `stop sign`,
      `parking meter`, `sports ball`, `baseball bat/glove`, `tennis
      racket`, `wine glass`, `hot dog`, `potted plant`, `dining table`,
      `cell phone`, `teddy bear`, `hair drier`). A detection of any of
      them → `unexpected server line` → client dies mid-stream. Fix
      client-side: `rest = line[4:].rsplit(maxsplit=5)` → (label, score,
      x1, y1, x2, y2); or change the protocol to put the label last.
      Also add a `ssd_test`/client unit check for a multi-word label.
- [x] **NIT** `--file foo.jpg` with `--headless` and OpenCV missing passes
      the `cv2 is None` guard (file+headless exemption) but then crashes
      in `cv2.imread` with an `AttributeError` on None. Tighten the guard
      to the PPM-only case.
- [x] `BrokenPipeError` on send is already wrapped into a clear
      `RuntimeError` (pattern the `mobilenet_client.py` item in §1 should
      copy).

### `ssd_detect.cpp`

- [x] **DOC (header comment, wrong)** header says the client feeds "raw
      0..255 RGB pixel values (**BGR from the PPM**, resized ...)" and
      `bilinearResize` says "src/dst are w*h*3 **BGR** buffers" — both
      stale; everything in this app is RGB (Pillow writes RGB). Same
      confusion class as the `ssd_stream.py` bug above; fix the comments
      so future readers don't 'fix' the code into an actual swap.
- [x] **DOC (`--help`/usage incomplete)** `usage()` shows only the
      single-image form; the `--stdin` stream mode (the mode
      `infer-ssd-server.sh` actually uses, with its frame protocol) is
      absent from `--help`. Add the `--stdin` line + protocol summary.
- [x] **PERF/CLARITY (single-image mode)** the resize is executed twice:
      once into `unused` just to time it, then again inside `runFrame()`
      (and once per `--iterations` repetition). Move the resize timing
      into `runFrame` (or reuse `resized`) — the `input image ... in X ms`
      line measures a resize whose result is thrown away.
- [x] **NIT** `outElemBytes`/`outElems` are read from the blob *before*
      `Infer()`, while `mobilenet_server.cpp` reads its output blob after
      `Infer()`. Both work on MYRIAD today; standardize on post-Infer to
      match the 'TensorDesc can lie, bytes don't' doctrine the code
      itself argues for.

### `ssd_postprocess.hpp`

- [x] **BUG (self-containedness)** `roundToI()` uses `std::floor` but the
      header does not `#include <cmath>`. It only compiles because both
      consumers include `<cmath>` first. Add the include.

### `infer-ssd-server.sh`

- [x] `usage()` verified against code: positional `[backend] [device]
      [min-conf]`, protocol block, env vars — accurate. Its `auto`
      description ("host if pulled, else docker, else in-image fallback")
      matches the three-way auto implemented here.
- [x] **INCONSISTENCY** this launcher's auto has the docker-CLI check and
      the in-image fall-through; `examples/webcam/infer-server.sh` does
      not (see §3). Adopt this launcher as the canonical pattern and port
      it to webcam (and keep seg in sync).
- [x] **NIT** positional `min-conf` is not validated (e.g. `banana`)
      before reaching `std::stof` in the server — acceptable, but a shell
      check would give a better message.

### `README.md`

- [x] **DOC (wrong hardware)** intro says "running on the Intel Movidius
      **NCS1**" — everything else (and the verified runs) is the MA2450.
- [x] **DOC (wrong test photo description)** `dog_ssd.ppm` is described as
      "known-content test photo (**person + dog**)" but the verified
      detections are bicycle/dog/car/cat (no person).
- [x] **DOC (duplicate + fabricated section)** `## Verified results
      (MA2450, amd64 image)` appears **twice**; the first block shows
      invented output (`person 0.87 ... dog 0.71`) that does not match
      the real verified run in the second block. Merge into one section
      with the real numbers.
- [x] **DOC (output layer name)** input/output table names the output
      `detection_boxes/sink_port_0`; the app prints `DetectionOutput
      [1,1,100,7]`. Use the actual blob name.
- [x] **DOC (missing cgroup rule)** the webcam-stream section says 'run
      inside the runtime container with the camera and stick passed
      through (`-v /dev:/dev`)' — empirically the client-in-container
      workflow also needs `--device-cgroup-rule='c 81:* rwm'` (EPERM on
      `/dev/video0` without it). Document the exact docker run flags
      once (ties to the §3 DOC GAP item).
- [x] **NIT** "Options mirror the single-image mode (`--min-conf`,
      ...)" — the client has no `--max-detections` counterpart (server
      stays at its default 10); either expose it or say 'partially'.

## 3. `examples/webcam/`

### `mobilenet_server.cpp`

- [x] **BUG (latent UB)** the output conversion loop writes `n` elements
      into `logits` (sized `outElems`) **before** the `if (n != outElems)`
      check runs. If a future model/driver returns more values than the
      declared output size, this is a heap overflow before the guard fires.
      Move the `n != outElems` check (and the `bytesPerElem` decision) in
      front of the conversion loop.
- [x] **NIT** `outElems = getDims().back()` assumes the class count is the
      last dim; `inElems` ignores `dims[0]` (batch). Both fine for this
      model, but a `product(dims) == expected` sanity check would turn a
      confusing wrong-size run into a clear error.
- [x] `--help` text is accurate (in/out sizes, until-EOF semantics). Could
      additionally document the exit-code table the file header promises
      (0/2/3/4) — clients map on those.
- [x] **INCONSISTENCY (cross-app)** `mobilenet_server` always reads stdin
      (no `--image` single-shot mode, no `--stdin` flag), while
      `ssd_detect`/`seg_detect` support `--image` and gate stream mode on
      an explicit `--stdin`. Pick one convention or document why they
      differ (mobilenet is server-only by design; the `--stdin` flag on
      the other two exists because they also have one-shot modes).

### `infer-server.sh`

- [x] **DOC GAP (in-container camera runs)** verified empirically: with
      `-v /dev:/dev --device-cgroup-rule='c 189:* rwm'` a container gets
      EPERM on `/dev/video0` (`c 81:* rwm` fixes it → ENXIO, i.e. a real
      capture device). The launcher `docker_backend` is fine as-is (only
      the *server* runs in the container; the Python client captures on
      the host), but the *in-image fallback* path (client + server both
      inside the image, used by every in-container webcam/stream test so
      far) needs the 81 rule and **no README documents the docker run
      flags for that workflow**. Add a documented 'run the example inside
      the image' command (webcam/ssd/seg READMEs) with both cgroup rules.
- [x] **BUG/DOC conflict: in-image fallback unreachable via `auto`** the
      README claims "if host runtime is missing but /opt/openvino exists
      (you are inside the runtime image), the server is launched from
      there automatically", but `auto` falls to `docker_backend` when
      `work/host-runtime` is missing — and inside the image there is no
      `docker` CLI, so it fails. The fallback only exists inside
      `host_backend`, i.e. only via explicit `backend=host`. Fix options:
      make `auto` fall back to `host_backend` when `docker` is not on PATH
      (and `/opt/openvino/bin/<app>` exists), or correct the README.
- [x] `usage()` content verified against the code: positional
      `[backend] [ir] [device]`, defaults, `IR=`/`IMAGE=` envs, and
      `MVNC_MUTEX=` all match; the direct-drive example (`< \\...input_0.f32`)
      uses the correct reference tensor path.
- [x] **NIT** `-h|--help` is only recognized as `$1` (any later arg
      position errors as "unknown option" in the backend case — actually
      positional args don't error, they'd be silently interpreted as
      backend/ir/device; `infer-server.sh docker --help` treats `--help`
      as `ir` and then exits 2 via the ir validation — acceptable but
      worth a clearer rejection).

### `webcam_mobilenet.py`

- [x] **BUG (robustness)** a single failed `cap.read()` calls `die()` —
      real webcams occasionally drop a frame (`ok=False`); tolerate N
      consecutive failures (e.g. 5) before aborting.
- [x] **NIT (status line conventions)** verified: `webcam_mobilenet.py`
      and `ssd_stream.py` compute fps identically (`frame_no / t` =
      average since start, compile time included — short runs read slower
      than steady state), while `seg_stream.py` prints *instantaneous*
      per-frame fps. Three clients, two conventions; pick one (windowed
      fps is the most honest) and label the number.

## 4. `examples/deeplab-seg/`

### `seg_stream.py`

- [x] **BUG (`--mask-out` writes an invalid PPM)** the client writes the
      `uint16` MASK payload straight into a `P6 ... 255` PPM (2 bytes per
      pixel instead of 1 → `w*h*2` body bytes, every other byte NUL; a
      3x4 test map produced 25 body bytes where 12 are valid — verified
      locally by replicating the code). Fix:
      `m.astype("uint8").tobytes()` (class ids 0..20 fit a byte). The C++
      `seg_detect --mask-out` and the README both promise 'pixel value =
      class id' single-channel P6 — only the client path violates it.
- [x] **BUG/UX (`--file` silently resizes to 640x480)** `--file img.ppm`
      resizes any input to the `--width/--height` defaults (640x480), so
      e.g. `dog_ssd.ppm` (768x576) is aspect-distorted before inference
      and the saved mask is 640x480. `ssd_stream.py --file` keeps native
      resolution (the server resizes internally). Either default
      `--width/--height` to 0 = 'keep file size' (camera keeps 640x480)
      or document the behavior in `--help`/README.
- [x] **INCONSISTENCY (no `--backend`/`--device`/`--request-timeout`)**
      `ssd_stream.py` and `webcam_mobilenet.py` expose backend/device
      selection and a request timeout (`--request-timeout`); the seg
      client hard-codes `--server-cmd "bash
      examples/deeplab-seg/infer-seg-server.sh"` (relative path — must be
      started from the repo root) and `segment()` blocks forever if the
      server hangs after accepting a frame. Add the same three options
      (or at least a timeout + documented `--server-cmd` overrides).
- [x] **NIT** the client handles an `ERROR <msg>` protocol line the server
      never emits (`seg_detect` dies with a stderr message instead) —
      either emit `ERROR` from the server's stream-mode catch before
      exiting, or drop the dead branch.
- [x] **NIT** `CLASS <id> <name> <pixels>` parsing assumes single-word
      names (`int(p[3])`). Safe for Pascal VOC (all 21 names are one
      word), but it is the same latent fragility as the ssd `DET` label
      bug — worth a `rsplit`-safe parse while fixing §2.
- [x] **NIT** headless fps here is *instantaneous* (`1/dt` per frame)
      while the webcam/ssd clients print *average-since-start* — three
      clients, two conventions (update the §3 NIT accordingly).
- [x] `_readline`/`_read_bytes` line/binary buffering is correct (the
      lesson from the ssd streaming bug is applied), and
      `BrokenPipeError` on send is wrapped into a clear message.

### `seg_detect.cpp`

- [x] **BUG (bilinear upscale overshoot → uint8 wrap)** `bilinearResize`
      half-pixel sampling produces a *negative* weight on the first row
      when upscaling (`fy < 0` but `y0` clamps to 0 → `wy < 0`; likewise
      `wy > 1` on the last row). The four weights still sum to 1, so
      saturated neighborhoods (e.g. taps 255/255/0/0) can yield `v > 255`
      or `v < 0`, and `(uint8_t)(v + 0.5f)` wraps (263 → 7). At the
      typical 640x480 → 513x513 webcam resize the effect is small
      (|wy| ~ 0.03) but real; extreme aspect changes make it worse. Fix:
      clamp `v` to [0,255] before the cast (same function exists in
      `ssd_detect.cpp` — it only downscales today, but the guard is
      free).
- [x] **DOC (terminology)** the header says 'this client feeds raw 0..255
      **BGR** pixel values' while the stream-mode block and the code say
      frames are **RGB** and the blob is written BGR-ordered (`rc = 2 -
      c`). The flow is: client sends RGB, seg_detect packs BGR, graph
      swaps back. Reword once, consistently — same confusion family as
      the §2 `ssd_detect.cpp` item, and here it actively contradicts the
      neighboring protocol comment.
- [x] **DOC (`--help` is terse)** the one-line usage prints both modes
      (good) but does not mention: the `--stdin` frame protocol, what
      `--mask-out` writes (single-channel P6, pixel = class id), or exit
      codes (implicitly 2 CLI, 3 IO/shape, 1 runtime exception; the
      header documents none, unlike `mobilenet_server.cpp`).
- [x] **NIT** `FRAME <total_ms>` excludes the nearest-resize, histogram
      and mask serialization done after `runFrame()` returns — the
      printed 'total' is pre+infer+decode only, while true per-frame wall
      time is a bit higher. Include the tail in the timer or rename the
      field.
- [x] **NIT** logits branch is unreachable here: `seg_detect` gates
      `outDims.size() == 3` and passes `C = 0`, so the `C*H*W` arm of
      `classMapFromRaw` can never fire (exercised only by `seg_test`).
      The in-code comment ('the logits branch only fires for a real
      C*H*W blob') is misleading — label it as future-proofing or make
      it actually work.
- [x] stream-mode hardening is the best of the three servers: header
      range check (`fw/fh > 8192`), truncated-body check via `gcount`,
      protocol-only stdout, `waitDevice()` retry.

### `seg_postprocess.hpp`

- [x] Self-contained includes (has `<cmath>`, unlike
      `ssd_postprocess.hpp` — see §2), good doc comments, deterministic
      histogram sort.
- [x] **INCONSISTENCY** `seg::loadLabels` silently accepts an empty or
      all-garbage label file (labels degrade to `class_N`), while
      `ssd::loadLabels` throws when nothing parses. Pick one behavior
      (throwing is safer for a typo'd path).
- [x] `interpretClassIds` decode ladder (FP32-plausible → I32; U16 →
      FP16) with the denormal rejection (`0 < x < 0.5`) is sound and
      each arm is unit-tested; an all-zero frame is ambiguous between
      FP32 and I32 but both give identical results, so the ambiguity is
      safe.

### `infer-seg-server.sh`

- [x] `usage()` matches the code (positional `[backend] [device]`, no
      min-conf here) and the launcher is a faithful clone of
      `infer-ssd-server.sh` including the three-way `auto` and in-image
      fallback — canonical pattern confirmed (see §2).
- [x] **DOC GAP (minor)** usage repeats the protocol block but, like the
      ssd launcher, does not document the docker flags needed for the
      *in-container client* workflow (see §3 DOC GAP).

### `seg_test.cpp` / `CMakeLists.txt`

- [x] Test coverage matches the README list: half roundtrip + clamp,
      direct ids, logits argmax, blob decoding for I32/FP32/I16/FP16
      plus rejection cases (bad32, 1-byte), nearest resize properties,
      labels incl. fallback, out-of-range and shape errors. Good.
- [x] `CMakeLists.txt` is the same boilerplate as the other three —
      covered by the §3 dedup item.

### `README.md`

- [x] **DOC (wrong sentence about who resizes)** "The client therefore
      feeds raw 0..255 BGR and does its own 513x513 resize" — both
      halves are wrong for the actual flow: `seg_stream.py` feeds **RGB**
      at camera size and **seg_detect** does the bilinear 513x513 resize
      (the client only resizes `--file` inputs to `--width/--height`, see
      the `--file` bug). Fix together with the terminology item.
- [x] **DOC (stats channel)** "class statistics go to stderr/overlay
      text" — the headless client prints them to **stdout**; only server
      diagnostics are stderr.
- [x] **DOC (overstated feature)** 'Output is interpreted from its shape:
      C*H*W ... logits, argmax' — unreachable via `seg_detect` (3-dim
      gate + `C=0`), only `seg_test` exercises it. Say 'the
      postprocessor also supports logits (unit-tested); the app requires
      [1,H,W]' or implement the 4-dim case.
- [x] **DOC GAP** the streaming section shows the plain
      `python3 ... --headless` workflow without the in-container
      `-v /dev:/dev` + both device-cgroup-rules form (consolidated item,
      see §3).
- [x] Verified-results section is consistent with the hardware runs
      (`dog_ssd.ppm` 'dog, cat, bicycle and car' matches the detections —
      the ssd README fix in §2 should copy this wording).

## 5. `examples/accuracy-test/`

### `accuracy_test.py`

- [x] **BUG (`--show-errors` lists top-5 misses, not top-1 misses)** the
      `bad_rows.append` sits in the `elif` of `if truth in top5`, so only
      images missed by *all five* predictions are listed — yet the option
      help and the printed section say 'misclassified (top-1)'. Every
      top-1 miss that lands in top-5 (the common failure mode) is never
      shown. Fix: record rows when `top1 != truth` (separate cap) or
      rename the option/section to 'top-5 misses'.
- [x] **DOC (docstring expectation contradicts verified numbers)** the
      module docstring says 'around 70% expected (official fp32 top-1 is
      71.9%)' while the README's verified table shows **81.4%** on this
      dataset (all three backends) and the README warns 'far below ~80%
      means the pipeline is broken'. Three different expected numbers in
      two files (docstring 70%, README ~80%, README CI example
      `--fail-under 65`). Settle on one: measured 81.4%, gate 65-75, and
      update the docstring sentence (the 71.9% figure is the *official
      ImageNet val* number, not what this easier sample set produces).
- [x] **NIT** file discovery globs `*.JPEG`, `*.jpg`, `*.png` but not
      `*.jpeg`/`*.Jpeg` — harmless for this dataset (all `.JPEG`), but a
      case-insensitive glob would be one line.
- [x] `t_first` correctly excludes server startup from the throughput
      number; `--fail-under`, `--limit`, label-count sanity (1000) and
      synset parsing (`n########`) all behave as documented; docstring's
      preprocessing description matches `mobilenet_client.preprocess`
      (resize 224, BGR→RGB, /255, mean/std, NCHW float32).

### `fetch-sample-images.sh`

- [x] **BUG/UX (destructive re-run)** a second run when `images/` exists
      but `.git` is missing does `rm -rf "${DEST}"` — silently deleting a
      user-supplied images directory. Guard with a confirmation or move
      the clone to `images/.repo` + symlink, or at least warn.
- [x] `--help` correct (no positional args exist, so the first-arg-only
      check is fine here). Count check + warning when < 1000 is good.

### `README.md`

- [x] **DOC (stale path)** 'The conversion now lives in
      `examples/webcam/half.hpp`' — that file no longer exists; the
      shared header is the repo-root **`half.hpp`** (moved during the
      deduplication). Verified: `grep` shows this as the only remaining
      stale reference in docs/code.
- [x] **DOC (`--show-errors` wording)** the bullet 'a list of up to
      `--show-errors` misclassified images' describes intended, not actual
      behavior (top-5 misses only) — fix together with the code item.
- [x] Dataset licensing note ('keep inside the checkout, do not
      redistribute') matches the `.gitignore` policy.

## 6. Cross-cutting observations

- [x] **Launcher trio alignment**: `infer-ssd-server.sh` and
      `infer-seg-server.sh` are byte-for-byte consistent clones with the
      good three-way `auto`; `infer-server.sh` (webcam) predates the
      pattern (no docker-CLI check, no documented fallback). Port the
      ssd/seg pattern + identical `usage()` skeleton to all three, or
      extract a shared launcher library (`scripts/launch-server.sh
      <app> [extra args]`) like `half.hpp`/`device_probe.hpp` were
      extracted for C++.
- [x] **Client CLI conventions**: webcam client = positional script +
      `--backend/--ir/--device`; ssd client = positional script +
      `--backend/--device/--min-conf/--request-timeout`; seg client =
      `--server-cmd` string, no backend/device/timeout. One convention
      would cut three READMEs' worth of confusion.
- [x] **Protocol error signalling**: only `seg_stream.py` handles an
      `ERROR` line that no server sends; ssd/webcam clients surface
      server death as bare `BrokenPipeError`/'server closed'. Either all
      servers print `ERROR <msg>` to stdout before exiting on fatal
      stream errors (seg/ssd stream-mode catches can do this in one
      line), or no client pretends to parse it.
- [x] **`half.hpp`/`device_probe.hpp` include style is Docker-only**:
      every consumer (`examples/webcam`, `examples/ssd-detect`,
      `examples/deeplab-seg`, `mobilenet-test`, `smoke-test`) writes
      `#include "../half.hpp"`, which resolves only in the flat
      `/work/<app>/` + `/work/half.hpp` Docker layout. From the repo tree
      the relative path lands on `examples/half.hpp` (nonexistent) for
      the example apps and *outside the repo* for the two top-level test
      dirs — so none of these TUs can be compiled in place from a
      checkout. Options: add `include_directories(${PROJECT_ROOT})`-
      style resolution in the CMakeLists and include as
      `half.hpp`/`device_probe.hpp`, or move the shared headers to
      `examples/common/` (root-level dirs would then use
      `../examples/common/half.hpp` via an include dir). Cosmetic today,
      real friction for anyone trying a local syntax check.
- [x] **Exit-code documentation**: `mobilenet_server.cpp` documents
      0/2/3/4; `ssd_detect.cpp` and `seg_detect.cpp` use similar codes
      (2 CLI, 3 model/shape, 1 runtime) but document none. Copy the
      table style.
- [x] **`--help` coverage** (explicit focus of this review):
      * all three launcher scripts: accurate, first-arg-only, seg/ssd
        complete, webcam missing the fallback it doesn't implement;
      * `accuracy_test.py`, `fetch-sample-images.sh`, the three Python
        clients (argparse-generated): accurate;
      * C++ servers: `mobilenet_server` good (minor: exit codes);
        `ssd_detect` missing `--stdin`/protocol in usage; `seg_detect`
        terse one-liner missing protocol + mask format + exit codes.
- [x] **Duplication candidates for the next refactor round**
      (established pattern = root-level shared header):
      `bilinearResize` (ssd/seg, incl. the clamp bug), `readPpm`
      (ssd/seg, near-identical), PPM/label/round helpers (ssd has its
      own), the four CMakeLists (~100 lines each), the three launcher
      scripts, the `_readline` pipe reader (ssd/seg clients).

## 7. Prioritized bug list (everything above, ranked)

Real behavior bugs (fix first):

1. `ssd_stream.py` — `--file *.ppm` sends **BGR** to the RGB model
   (cat detection lost; proven against the C++ reference). §2.
2. `ssd_stream.py` — `DET` parsing crashes on the 14 multi-word COCO
   labels (`dining table`, `cell phone`, …). §2.
3. `seg_stream.py` — `--mask-out` writes a corrupt PPM (uint16 body in a
   1-byte-per-pixel container). §4.
4. `seg_detect.cpp` / `ssd_detect.cpp` — `bilinearResize` upscale
   weights can leave [0,1] → `(uint8_t)(v+0.5f)` wrap on saturated
   pixels. §4.
5. `accuracy_test.py` — `--show-errors` lists only top-5 misses while
   documented as top-1 misses. §5.
6. `mobilenet_server.cpp` — output conversion loop runs before the
   `n != outElems` bounds check (latent heap overflow). §3.
7. `fetch-sample-images.sh` — `rm -rf images/` without `.git` deletes
   user data. §5.
8. `ssd_postprocess.hpp` — missing `<cmath>` (self-containedness). §2.

Documentation fixes (`--help` focus):

9. `ssd_detect --help`: document `--stdin` + protocol; `seg_detect
   --help`: document protocol, `--mask-out` format, exit codes. §2/§4.
10. Stale BGR/RGB comments in `ssd_detect.cpp` header, `seg_detect.cpp`
    header and the seg README 'who resizes' sentence. §2/§4.
11. ssd README: NCS1→MA2450, 'person + dog' photo description,
    duplicate/fabricated 'Verified results' section, output layer name. §2.
12. accuracy-test README: `examples/webcam/half.hpp` → root `half.hpp`;
    unify the three expected-accuracy numbers (70% / ~80% / gate 65). §5.
13. Document the in-container client docker flags (both cgroup rules) in
    webcam/ssd/seg READMEs and RESULTS.md. §3 (consolidated).

Structural/consistency improvements (later refactor):

14. Port the ssd/seg launcher pattern (three-way auto, identical usage)
    to `infer-server.sh`, or extract a shared launcher. §6.
15. One CLI convention for the three Python clients
    (positional script + `--backend/--device/--request-timeout`). §6.
16. Deduplicate: `bilinearResize`, `readPpm`, `_readline`, the four
    CMakeLists (→ shared cmake helper), launcher trio (→ shared
    library), following the `half.hpp`/`device_probe.hpp` precedent. §6.
17. Shared headers are unresolvable from a plain checkout
    (`../half.hpp` works only in Docker's `/work` layout). §6.
18. `ERROR` protocol line: implement it in the servers or stop parsing
    it in the seg client. §6.
19. fps status line: three clients, two conventions. §3/§4.
20. `--help` recognized only as the first argument in all three
    launchers and `fetch-sample-images.sh`. §3.

---

## Resolution notes (fix pass, 2026-09-23)

Every item above is fixed and verified. Per-area summary:

- **§1 `mobilenet_client.py`** — `server_exit_meaning()`, `LinePipe`
  (line/binary reader over the pipe fd with proc-exit drain),
  `windowed_fps()`, `BrokenPipeError` → `RuntimeError` with the exit-code
  meaning, `cv2` None guard, safe `close()`. Verified: accuracy test
  17/20 top-1 on the sample set; LinePipe unit checks; `windowed_fps`
  converges to the steady rate.
- **§2 `ssd-detect`** — `ssd_stream.py`: removed the double BGR conversion
  (PPM→RGB sent straight, matching the C++ reference: 4 detections,
  bicycle 0.96 / dog 0.88 / car 0.87 / cat 0.63 on `dog_ssd.ppm`), multi-word
  `DET` parsing (`split("DET",1)[1].rsplit(" ",5)` + label strip), `ERROR`
  line handling, `LinePipe`, `windowed_fps`, positional `server_cmd`. 
  `ssd_detect.cpp`: `--stdin` + exit codes in `usage()`, bilinear-resize
  clamp, post-Infer `outElemBytes`, stream-mode `ERROR <msg>` + catch. 
  `ssd_postprocess.hpp`: `<cmath>`, tab/space/multi-word `loadLabels` (+
  ssd_test case 6b; all ssd_test checks pass).
- **§3 `webcam`** — `mobilenet_server.cpp`: input FP16 verification before
  the cast, output count check before `memcpy` (overflow removed), exit
  codes in `--help`. `infer-server.sh`: three-way `auto` (host → docker →
  in-image fallback), V4L2 cgroup rule for docker, `--help`.
  `webcam_mobilenet.py`: windowed fps (shared `windowed_fps`).
- **§4 `deeplab-seg`** — `seg_stream.py` rewritten on the shared client
  library (`LinePipe`, `windowed_fps`, positional `server_cmd` +
  `--backend/--device/--request-timeout`), `--file` native resolution, PPM
  loading, Pascal VOC palette overlay, `--mask-out` writes a correct
  single-channel P6 PPM (1 byte/pixel = class id, maxval 255), SIGINT
  handler, token-based `CLASS` parsing. `seg_detect.cpp`: resize clamp,
  full `--help` (protocol, mask format, exit codes), post-Infer
  `outElemBytes`, stream-mode `ERROR <msg>`. `seg_postprocess.hpp`:
  whitespace-tolerant `loadLabels` + empty-file throw. README updated.
  seg_test 34/34 pass.
- **§5 `accuracy-test`** — `--show-errors` lists **all** top-1 misses
  (capped), `[also a top-5 miss]` tag for the rarer misses, header
  "up to N of M shown" (verified with a fake server: hit / top-5-only
  miss / both-miss all rendered correctly); case-insensitive file
  discovery; docstring accuracy figure corrected; `fetch-sample-images.sh`
  refuses to `rm -rf` a non-empty non-git `images/` (verified: user file
  preserved, exit 1); README fixes (half.hpp path, `--show-errors` text).
- **§6 cross-cutting** —
  * **Launchers**: all three now share the same three-way `auto` +
    `usage()` skeleton (webcam ported to the ssd/seg pattern); `--help`
    accepted at **any** argument position in all three and in
    `fetch-sample-images.sh` (all exit 0).
  * **Client CLI convention**: one convention everywhere — positional
    `[server_script]` + `--backend/--device/--request-timeout` (seg's
    `--server-cmd` replaced).
  * **ERROR protocol**: implemented in both stream servers
    (`ssd_detect --stdin`, `seg_detect --stdin` emit `ERROR <msg>` on
    stdout before exiting); both Python clients parse it.
    `mobilenet_server` keeps exit-code signalling (its protocol is a
    binary request/response stream — a text `ERROR` line would corrupt
    responses); the client surfaces the exit code via
    `server_exit_meaning`.
  * **fps**: all three clients now use the shared `windowed_fps`
    (last-10-frame average, excludes one-time server compile).
  * **Shared headers resolvable from a plain checkout**: `#include
    "../half.hpp"` → `#include "half.hpp"` in all 10 consumers, with the
    include dir auto-detected per CMakeLists (`SHARED_INCLUDE_DIR`: repo
    root in a checkout, `/work` in the Docker build). Verified by
    building ssd_test / seg_test / test_half from a plain checkout.
  * **CMake dedup**: the ~45 lines of shared OV link logic (OV_ROOT
    validation, lib glob, 5-library link group, rpath) extracted to
    `cmake/ov203-link.cmake`; all five CMakeLists now ~40-50 lines and
    self-locate the module (`<root>/cmake` in a checkout, `/work/cmake` in
    Docker; `COPY cmake /work/cmake` added to all five Dockerfile stages).
  * **frame dedup**: `readPpm` + `bilinearResize` (with the upscale clamp)
    extracted to root `frame_utils.hpp`; the near-identical copies in
    `ssd_detect.cpp` and `seg_detect.cpp` removed (`COPY frame_utils.hpp`
    added to the ssd/seg Dockerfile stages).
  * **Exit-code docs**: all three C++ servers document the 0/1/2/3/4
    table in `--help`.

Verification status: Python clients verified against fake servers +
corrected protocol tests; C++ changes verified by syntax check against
the 2020.3 headers, `ssd_test`/`seg_test`/`test_half` rebuilt and passing
from a plain checkout, and shell scripts by `bash -n` + `--help` runs.
Final hardware confirmation (Docker image rebuild + `scripts/verify.sh`
against the MA2450) is the remaining gate before commit.
