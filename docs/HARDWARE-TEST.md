# Hardware validation matrix

The source/build refactor supports three targets. `arm64` is the preferred native Raspberry Pi 5 target, `armv7` remains the known-good compatibility fallback, and `amd64` is the native x86_64 target. Each release should be exercised on real hardware before being called fully validated.

| Check | Raspberry Pi 5 / `arm64` | Raspberry Pi / `armv7` fallback | Ubuntu x86_64 / `amd64` |
|---|---|---|---|
| `./build.sh --platform TARGET` | Required | Required | Required |
| `./scripts/verify.sh --platform TARGET --no-device` | Required | Required | Required |
| NCS/MA2450 visible as USB `03e7:*` | Required | Required | Required |
| MYRIAD device enumeration | Required | Required | Required |
| Firmware boot + USB re-enumeration | Required | Required | Required |
| Tiny-network inference | Required | Required | Required |
| MobileNet reference comparison | Required | Required | Required |
| 10 cold plug/boot/inference cycles | Required | Required | Required |
| At least 20 inference iterations in one run | Required | Required | Required |
| Stop/start container without replug | Required | Required | Required |
| Host-native extracted runtime | Required | Required | Required |
| Logs retained under target directory | Required | Required | Required |

## Recommended procedure

Install the udev rule described in the main README, unplug/replug the stick, then build and run the strict verification script.

Preferred Pi 5 native ARM64 path:

```bash
./build.sh --platform arm64
./scripts/verify.sh --platform arm64 2>&1 | tee logs/rpi5-arm64/verify.log
```

Legacy/known-good Pi ARMv7 fallback:

```bash
./build.sh --platform armv7
./scripts/verify.sh --platform armv7 2>&1 | tee logs/rpi5-armv7/verify.log
```

Ubuntu x86_64:

```bash
./build.sh --platform amd64
./scripts/verify.sh --platform amd64 2>&1 | tee logs/ubuntu-amd64/verify.log
```

If the MobileNet corpus is absent, prepare it first with `./scripts/prepare-mobilenet.sh`.

For repeated cold-cycle testing, physically unplug/replug the NCS between cycles. Confirm that each cycle starts from the ROM-mode USB device, firmware boots, the device re-enumerates, and inference succeeds. Also run at least 20 consecutive inference iterations and separately stop/start the container without replugging the stick to exercise the warm-device path. A warm rerun alone does not validate firmware boot behavior.

## Host-native runtime

After the container test succeeds, validate the extracted runtime independently:

```bash
./scripts/pull-runtime.sh --platform TARGET
./scripts/host-run.sh --platform TARGET list
./scripts/host-run.sh --platform TARGET demo
```

On Pi 5/`arm64`, this executes native AArch64 binaries directly. On the `armv7` fallback it intentionally exercises the ARMHF compatibility loader/sysroot. On `amd64`, it executes native x86_64 binaries directly.

## ARM64-specific acceptance checks

For `arm64`, retain evidence that both the application and plugin are native AArch64:

```text
hello_myriad:       ELF64, e_machine 183 (AArch64)
libmyriadPlugin.so: ELF64, e_machine 183 (AArch64)
```

Also confirm that `host-run.sh --platform arm64` does not create or use an ARMHF sysroot and that the installed OpenVINO library directory is discovered dynamically rather than assumed to have a particular name.

## Pass criteria

A target passes when the strict verifier ends with `VERIFY_RESULT=PASS`, the repeated cold cycles complete without a manual recovery other than the planned unplug/replug between cycles, and the retained log contains no unresolved loader, firmware, USB-permission, or reference-comparison errors.
