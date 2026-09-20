# How to run MobileNet v2 on the Movidius stick with this project

A step-by-step guide for someone who has never used this repo.  Every command below was
run on the Raspberry Pi 5 (`edge`) with an Intel NCS2 (MA2450) plugged in.

---

## 0. The idea in one picture

```text
mobilenetv2-7.onnx  --(Model Optimizer 2020.3.2, arm64 container)-->  IR (.xml + .bin)
                                                                    |
   your photo (.jpg -> .ppm)  -->  mobilenet_classify (arm32v7 container)  -->  top-5 classes
                                            |
                                     MYRIAD plugin boots the stick over USB and runs the
                                     compiled graph on the VPU: 44.5 ms per image
```

Two containers, both built for this project:

* **arm64 `python:3.8-slim`** - only for the one-time model conversion (Model Optimizer is
  pure Python; armhf has no `onnx` wheel).
* **arm32v7 `openvino-2020.3-rpi5`** - the OpenVINO 2020.3.2 runtime you compiled yourself;
  this is what talks to the stick.

## 1. What you need

| Requirement | Why | Check |
| --- | --- | --- |
| Raspberry Pi 5 (8 GB is comfortable), 64-bit OS | the arm32v7 container runs natively on it | `uname -m` -> `aarch64` |
| Intel NCS2 / Neural Compute Stick 2 in a USB port | the device | `lsusb \| grep 03e7` |
| Docker (with BuildKit) | build + run everything | `docker version` |
| ~5 GB free disk | build tree + image | `df -h /` |
| Internet for the first run only | OpenVINO source, apt snapshot, model + photos | - |
| `sudo` (passwordless here) | only to reset the USB stick | `sudo -n true && echo ok` |

```bash
cd /home/chris/workspace/flash-movidius-openvino/openvino-2020.3-rpi5
lsusb | grep -i 03e7        # Bus 00X Device 00Y: ID 03e7:2150 Intel Myriad VPU
```

`03e7:2150` = the stick in ROM mode (unbooted).  That is the state you want **before** a
run; OpenVINO boots its own firmware and the ID changes to `03e7:f63b` while it works.

## 2. Step 1 - fetch the pinned vendor payload (once, ~10 min)

```bash
./scripts/clone-openvino.sh     # git clone --recursive --branch 2020.3.2  (tag 2020.3.2)
./scripts/fetch-runtime.sh      # Intel's official 2020.3.2 raspbian runtime (headers + .mvcmd)
./scripts/prepare-deps.sh       # local dependency mirror: firmware_<dev>_1656.zip
```

You can skip these - `build.sh` calls them automatically if `vendor/` is incomplete - but
running them first makes it obvious what is being downloaded.  The build itself then
performs **zero** network fetches.

## 3. Step 2 - build the runtime image (once, ~13 min of compiling)

```bash
./build.sh                      # full image: builder -> smoke -> mobilenet -> runtime
```

Measured on this Pi 5: **748 s** of compiling with `-j2`, image **264 MB on disk / 75.6 MB
of content**.  Watch it with `df -h /` and `docker system df` in another terminal.  If the
Pi runs out of memory, use `BUILD_JOBS=1 ./build.sh`.

Useful variants:

```bash
./build.sh --target configure   # CMake configure only - a few minutes, best for iterating
./build.sh --target builder     # stop before the demo binaries are built
./build.sh --no-cache           # from scratch
```

`build.sh` also prints the stage list at the end and the next command to run.

## 4. Step 3 - prove the runtime works before touching models (1 min)

```bash
./run.sh                        # device enumeration + the tiny hand-written model
```

Expected tail of the output:

```text
available devices : MYRIAD
  plugin MYRIAD   api 2.1 build custom__ (myriadPlugin)
...
mean inference  : 1.78 ms (560.91 fps)
RESULT: PASS
```

If this fails, fix it here first - the MobileNet step uses exactly the same plumbing.  The
most common cause is Docker not being able to see the stick after it re-enumerates; see
[USB access on a Pi 5](README.md#usb-access-on-a-pi-5) in the README.

## 5. Step 4 - prepare the MobileNet corpus (once, ~2 min)

```bash
./scripts/prepare-mobilenet.sh
```

What it does, in order:

1. downloads `mobilenetv2-7.tar.gz` from the ONNX model zoo and checks its sha256
   (`b463ad62…`), then checks the extracted `mobilenetv2-7.onnx` (`c1c51358…`);
2. downloads `synset.txt` (1000 ImageNet class names, in the order this model expects);
3. stages the vendored Model Optimizer into `work/mo-2020.3` **without its unit tests**
   (897 of 1175 `.py` files) because `mo/utils/import_extensions.py` imports every file it
   finds and the vendored tests import a test-only helper;
4. converts the model twice in a native arm64 `python:3.8-slim` container:

   ```bash
   python /mo/mo.py --input_model /models/onnx/mobilenetv2-7.onnx \
       --output_dir /models/mobilenet-v2-ov203/fp16 --model_name mobilenet-v2-ov203 \
       --data_type FP16                       # 2020.3 has no --compress_to_fp16
   ```

   look for:

   ```text
   [ SUCCESS ] Generated IR version 10 model.
   [ SUCCESS ] Total execution time: 14.35 seconds.
   [ SUCCESS ] Memory consumed: 148 MB.
   ```

5. dumps the zoo's own `test_data_set_0` tensors as raw float32 files (the numerical
   reference used in step 6);
6. downloads 5 photos (pinned by sha256) and writes them as **binary PPM** `P6` files,
   224×224: `vendor/models/images/*.ppm` (ImageNet centre crop) and
   `vendor/models/images/full/*.ppm` (full frame squashed).

Result (`vendor/models/`, ~51 MB, excluded from the Docker image and mounted at run time):

```text
vendor/models/onnx/mobilenetv2-7.onnx
vendor/models/mobilenet-v2-ov203/{fp32,fp16}/mobilenet-v2-ov203.{xml,bin,mapping}
vendor/models/labels/synset.txt
vendor/models/images/{dog,cat,eagle,banana,cup}.ppm  + images/full/*.ppm + images/src/*.jpg
vendor/models/test_data/{input_0.f32,output_0.f32}
```

## 6. Step 5 - run MobileNet on the stick (the moment you wanted)

```bash
./run.sh mobilenet
```

This is the built-in self-test: the model zoo's random 224×224 tensor is pushed through the
stick and compared with the zoo's own ONNX output.  Expected output:

```text
model           : /models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml + …/mobilenet-v2-ov203.bin
network         : mobilenet-v2-ov203
input           : data 1,3,224,224 FP16 NCHW
output          : mobilenetv20_output_flatten0_reshape0 1,1000
load+compile    : 1681.47 ms on MYRIAD
input source    : /models/test_data/input_0.f32 (150528 float32)
inference       : 44.42 ms mean over 10 run(s) (22.5 fps)

top 5:
   1.    556   0.9999  n03347037  fire screen, fireguard  logit 26.219
   2.    818   0.0000  n04286575  spotlight, spot  logit 15.211
   ...
against reference /models/test_data/output_0.f32:
  max |diff|    : 0.0525  (reference logits up to 26.2702)
  mean |diff|   : 0.0122
RESULT: PASS (top-1 matches, max |diff| 0.0525 vs tolerance 1.0000)
```

`RESULT: PASS` means: this OpenVINO build + this Model Optimizer + the VPU agree with the
official ONNX numbers.

## 7. Step 6 - classify a photo

```bash
./run.sh mobilenet --image /models/images/cat.ppm --topk 3
```

```text
input source    : /models/images/cat.ppm 224x224, mean 0.49,0.46,0.41 std 0.23,0.22,0.22
inference       : 44.60 ms mean over 1 run(s) (22.4 fps)

top 3:
   1.    281   0.2463  n02123045  tabby, tabby cat  logit 9.852
   2.    282   0.1830  n02123159  tiger cat  logit 9.555
   3.    277   0.1182  n02119022  red fox, Vulpes vulpes  logit 9.117
```

Other ready-made inputs, and the crop comparison:

```bash
for n in dog cat eagle banana cup; do ./run.sh mobilenet --image "/models/images/$n.ppm" --topk 2; done
for n in dog cat eagle banana cup; do ./run.sh mobilenet --image "/models/images/full/$n.ppm" --topk 2; done
IR=fp32 ./run.sh mobilenet --image /models/images/banana.ppm      # FP32 weights instead
```

### Your own image

The demo binary takes a **binary PPM (P6), exactly 224×224, RGB**.  Make one on the host:

```bash
# with Python + Pillow
python3 - myphoto.jpg <<'PY'
import sys
from PIL import Image
im = Image.open(sys.argv[1]).convert('RGB')
w, h = im.size
s = 256 / min(w, h)                      # documented preprocessing of this model
big = im.resize((int(round(w*s)), int(round(h*s))), Image.BILINEAR)
l, t = (big.width-224)//2, (big.height-224)//2
big.crop((l, t, l+224, t+224)).save('vendor/models/images/mine.ppm', 'PPM')
PY

# or in one line with ImageMagick, if you have it
convert myphoto.jpg -resize 224x224! PPM:vendor/models/images/mine.ppm

./run.sh mobilenet --image /models/images/mine.ppm --topk 5
```

Put the file anywhere under `vendor/models/` (that whole directory is mounted read-only at
`/models` inside the container), then refer to it with the **container** path
(`/models/images/mine.ppm`).  No OpenCV, no Python at run time.

If the model and your numbers disagree, the preprocessing is the usual suspect: this export
wants RGB, NCHW, `x/255`, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.  Those
are the defaults; you can override them with `--mean r,g,b --std r,g,b`.

## 8. Optional - benchmark and repeat

```bash
./run.sh mobilenet --tensor /models/test_data/input_0.f32 \
    --reference /models/test_data/output_0.f32 --iterations 50 --tol 0.2
```

`--tol` is the maximum allowed absolute logit difference for the PASS/FAIL line (default
`1.0`).  Expect ~44.5 ms per frame regardless of FP32 or FP16 weights: the VPU computes in
FP16 internally, only the `.bin` file size differs (13.95 MB vs 6.98 MB).

## 9. If the stick stops responding

```bash
sudo python3 scripts/reset-stick.py            # USBDEVFS_RESET
sudo python3 scripts/reset-stick.py --power    # plus a VBUS power cycle
lsusb | grep -i 03e7                           # want: 03e7:2150
```

Then run `./run.sh` again.  A script that ends without clean teardown can leave a graph
resident on the stick; the reset clears it.

## 10. Full report / evidence

```bash
./scripts/verify.sh > logs/VERIFICATION.md      # add --no-device to skip the stick steps
```

`logs/VERIFICATION.md` section 10 is this same MobileNet flow.  Other artifacts:
`logs/MOBILENET.md` (the write-up), `logs/mobilenet-run.log` (every command with full
output), `logs/mo-fp16-transcript.txt` (Model Optimizer), `logs/mobilenet-build.log`
(the Docker build).

## 11. Cheat sheet

| Goal | Command |
| --- | --- |
| build the runtime | `./build.sh` |
| is the stick reachable? | `./run.sh list` |
| does the runtime work? | `./run.sh` |
| get + convert MobileNet | `./scripts/prepare-mobilenet.sh` |
| self-test on the stick | `./run.sh mobilenet` |
| classify a photo | `./run.sh mobilenet --image /models/images/banana.ppm --topk 3` |
| FP32 weights | `IR=fp32 ./run.sh mobilenet` |
| 50-frame benchmark | `./run.sh mobilenet --tensor /models/test_data/input_0.f32 --reference /models/test_data/output_0.f32 --iterations 50` |
| shell inside the runtime | `./run.sh shell` then `/opt/openvino/bin/mobilenet_classify --help` |
| run it without Docker (host process) | `./scripts/pull-runtime.sh && ./scripts/host-run.sh mobilenet` |
| compile the graph to a device blob | `./scripts/host-run.sh compile --model vendor/models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml --blob vendor/models/blobs/mobilenet-v2.blob` |
| reset the stick | `sudo python3 scripts/reset-stick.py` |
| regenerate the report | `./scripts/verify.sh > logs/VERIFICATION.md` |

## 11b. Running without Docker (same binaries, host process)

The image is only needed for the build and the conversion.  Copy the runtime and its armhf
libraries out once, then run the same binaries as a normal host process:

```bash
./scripts/pull-runtime.sh                      # 37 MB into work/host-runtime/
./scripts/host-run.sh mobilenet                # numerical self-test on the stick
./scripts/host-run.sh mobilenet --image vendor/models/images/banana.ppm --topk 3
```

Identical results (44.54 ms, `RESULT: PASS`, max `|diff|` 0.0525).  Two host-only
caveats - USB node permissions (`sudo cp scripts/99-movidius-ncs2.rules /etc/udev/rules.d/`
+ `sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=usb
--action=change`, or just use `sudo`) and the `/tmp/mvnc.mutex` ownership rule: everything is
written up in [`logs/HOST-RUN.md`](logs/HOST-RUN.md).

## 12. Troubleshooting (MobileNet path only)

| Symptom | Meaning / fix |
| --- | --- |
| `no IR in …/vendor/models/mobilenet-v2-ov203/fp16 - run ./scripts/prepare-mobilenet.sh first` | you skipped step 4 (or used `IR=something-wrong`) |
| `ERROR: cannot open /models/images/x.ppm`, exit 3 | wrong **container** path - it must start with `/models/`, and the file must be inside `vendor/models/` |
| `image is WxH but the model wants 224x224`, exit 3 | resize your PPM; the demo does not resample |
| `give exactly one of --tensor or --image`, exit 4 | you passed both (or neither) |
| `device MYRIAD not available (have: )` | the plugin found no stick: is it plugged in, is another process holding it, did `./run.sh` get edited (it must keep `--network=host -v /dev:/dev --device-cgroup-rule='c 189:* rwm'`) |
| `ncDeviceOpen:1009 Failed to find booted device after boot` | the classic Pi 5 USB/uevent problem - see the README USB section; do not add `--privileged`, it does not help |
| `Stopped shape/value propagation at "Gemm_104/WithoutBiases"` during conversion | you used `mobilenetv2-10.onnx` (dynamic batch); use the opset 7 model |
| MO dies importing a `*_test.py` | the staged copy must exclude unit tests - `prepare-mobilenet.sh` already does this; don't point MO at `vendor/openvino-2020.3.2/model-optimizer` directly |
| `RESULT: FAIL … max |diff| …` | preprocessing or precision mismatch; re-run the default self-test (`./run.sh mobilenet`) which needs no customization |

## 13. FAQ

**Why arm32v7?**  OpenVINO 2020.3.2 is the last release with MYRIAD support, and Intel's
supported target for a Pi-class board was the 32-bit `armv7l` raspbian build; this project
forces that target on a native compiler (`toolchain/armv7-native.toolchain.cmake`).

**Why `--network=host`?**  libusb only learns about the stick's second (post-boot)
identity through kernel uevents, and a private network namespace receives none.  Without
it the plugin waits 15 s and fails.  `--privileged` is not the fix; see the README.

**Why two containers?**  Conversion needs `onnx`/`numpy` Python wheels that do not exist
for armhf; inference needs the arm32v7 OpenVINO libraries.  Model Optimizer is pure
Python, so it runs in a native arm64 container and never touches the stick.

**Do I need internet every time?**  No.  Everything is cached in `vendor/`.  Only the very
first `clone-openvino.sh` / `fetch-runtime.sh` / `prepare-mobilenet.sh` run needs it.

**Can I use my own ONNX model?**  Yes, same idea: convert it with the staged Model
Optimizer (`--data_type FP16` or `FP32`, static input shapes recommended), put the `.xml`
+ `.bin` under `vendor/models/`, then run
`./run.sh mobilenet --model /models/<dir>/<name>.xml --weights /models/<dir>/<name>.bin …`.
If the graph uses ops this 2020.3 MYRIAD compiler does not support, compilation fails with
a list of the offending op - that is a model problem, not a plumbing problem.
