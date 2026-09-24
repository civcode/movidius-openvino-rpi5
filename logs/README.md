# Validation logs

The files already present at the root of `logs/` are historical Raspberry Pi / ARMv7 development records retained unchanged for traceability.

New validation output should be placed by target:

- `rpi5-armv7/` — Raspberry Pi 5 host running the ARMv7 OpenVINO runtime.
- `rpi5-arm64/` — Raspberry Pi 5 host running the native arm64 (AArch64) OpenVINO runtime.
- `ubuntu-amd64/` — Ubuntu x86_64 host running the native amd64 runtime.
- `ci/` — CI build and hardware-free verification logs.

Recommended hardware-validation capture:

```bash
./scripts/verify.sh --platform armv7 2>&1 | tee logs/rpi5-armv7/verify-$(date +%Y%m%d-%H%M%S).log
./scripts/verify.sh --platform amd64 2>&1 | tee logs/ubuntu-amd64/verify-$(date +%Y%m%d-%H%M%S).log
```

Do not treat a hardware-free CI log (`--no-device`) as proof that firmware boot/re-enumeration or inference on an MA2450 was tested.
