# Throwaway diagnostics used to find the USB root cause

These are the small programs and shell drivers that were used while diagnosing the
"the stick boots but the MYRIAD plugin cannot find it again" failure.  They are kept
as evidence and as a way to re-verify the container flag matrix; they are **not**
part of the build and are not referenced by `build.sh` or `run.sh`.

| file | what it does |
|---|---|
| `xlink_probe.c` | links directly against the vendored XLink sources, enumerates devices, calls `XLinkBootRemote()` on `1-ma2450`, then enumerates again before/after with both `X_LINK_ANY_STATE` and `X_LINK_BOOTED`. This is what proved the libusb device-cache diagnosis. |
| `mvnc_probe.c` | same experiment one level up, through `ncDeviceOpen()` with `NC_RW_LOG_LEVEL = NC_LOG_DEBUG`, so the exact `mvnc_api.c` log lines (`XLinkBootArcObtained`, `Failed to find booted device after boot`, `Device (1-ma2450) doesn't disappear`) are captured. |
| `usb_list.c` | plain libusb enumeration (bus/device numbers, ids) used to map the Pi 5 USB topology. |
| `probe-run.sh` | builds `mvnc_probe` inside the `ov203-reference-check:latest` arm32v7 image and runs it. |
| `xlink-probe-run.sh` | builds and runs `xlink_probe`. |
| `xlink-probe-flags.sh` | runs `xlink_probe` under each candidate Docker flag combination. |
| `docker-flags-test.sh` | the full flag matrix of §7 in `logs/RESULTS.md` (each row resets the stick first). |
| `diag-usb.sh` | host-side USB snapshot helper (lsusb, sysfs nodes, `/dev/bus/usb` listing). |

Requirements: the vendored tree (`vendor/openvino-2020.3.2/inference-engine/thirdparty/movidius`),
the `ov203-reference-check:latest` image (built by `scripts/reference-check.Dockerfile`), a stick
in ROM mode (`sudo ./scripts/reset-stick.py`), and `sudo` for the reset step.  Outputs land in
`work/reference-check/`.

Example — reproduce the diagnosis in one command:

```bash
sudo ./docs/diagnostics/xlink-probe-run.sh            # private netns  -> AFTER n=1 name='1-ma2450'
# then add --network=host inside the script (or use docker-flags-test.sh) -> AFTER n=1 name='1-'
```
