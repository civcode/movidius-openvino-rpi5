# Movidius diagnostics (armv7, arm64 and amd64)

These low-level diagnostics use the same canonical target mapping as the main project. On a Raspberry Pi 5, prefer `--platform arm64`; use `--platform armv7` when reproducing the legacy ARMHF path. On x86_64 use `--platform amd64`.

The active probes compile the vendored MVNC/XLink sources natively inside a target-matching container, so the same diagnostic source can be exercised as ARMv7, AArch64, or x86_64 host code.

Typical commands:

```bash
./docs/diagnostics/runtime-usb.sh --platform arm64
./docs/diagnostics/probe-run.sh --platform arm64
./docs/diagnostics/xlink-probe-run.sh --platform arm64
```

Replace `arm64` with `armv7` or `amd64` as needed. Outputs are kept under `work/diagnostics/<target>/`.

The direct MVNC/XLink probes bypass `libmyriadPlugin.so`; they are diagnostic tools only. Normal inference should continue to use the OpenVINO Inference Engine -> MYRIAD plugin -> MVNC/XLink path.

## USB tests

`runtime-usb.sh` checks the self-built OpenVINO runtime and firmware packaging. `docker-flags-test.sh` and `xlink-probe-flags.sh` help isolate Docker device/cgroup/network behavior during firmware re-enumeration. `diag-usb.sh` can capture host `strace` output when ptrace permissions allow it.

For ARM64, the diagnostic image is also native `linux/arm64`; no ARMv7 toolchain is involved.
