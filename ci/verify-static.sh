#!/usr/bin/env bash
# Hardware-free regression checks for the multi-platform refactor.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

fail() { echo "ERROR: $*" >&2; exit 1; }

# All shell entry points must parse.
while IFS= read -r f; do
    bash -n "$f" || fail "bash syntax: $f"
done < <(find . -path './vendor' -prune -o -path './logs' -prune -o -type f -name '*.sh' -print | sort)

echo 'platform helper:'
armv7="$(./build.sh --platform armv7 --print-platform)"
arm64="$(./build.sh --platform arm64 --print-platform)"
amd64="$(./build.sh --platform amd64 --print-platform)"
grep -q '^target=armv7$' <<<"$armv7"
grep -q '^docker_platform=linux/arm/v7$' <<<"$armv7"
grep -q '^use_cmake_toolchain=1$' <<<"$armv7"
grep -q '^expected_elf_class=ELF32$' <<<"$armv7"
grep -q '^expected_elf_machine_id=40$' <<<"$armv7"
grep -q '^host_runtime_kind=armhf-sysroot$' <<<"$armv7"
grep -q '^target=arm64$' <<<"$arm64"
grep -q '^docker_platform=linux/arm64$' <<<"$arm64"
grep -q '^use_cmake_toolchain=0$' <<<"$arm64"
grep -q '^expected_elf_class=ELF64$' <<<"$arm64"
grep -q '^expected_elf_machine_id=183$' <<<"$arm64"
grep -q '^host_runtime_kind=native$' <<<"$arm64"
grep -q '^target=amd64$' <<<"$amd64"
grep -q '^docker_platform=linux/amd64$' <<<"$amd64"
grep -q '^use_cmake_toolchain=0$' <<<"$amd64"
grep -q '^expected_elf_class=ELF64$' <<<"$amd64"
grep -q '^expected_elf_machine_id=62$' <<<"$amd64"
grep -q '^host_runtime_kind=native$' <<<"$amd64"
printf '%s\n%s\n%s\n' "$armv7" "$arm64" "$amd64"

# Common build/runtime files must not contain an unconditional ARMv7 runtime path.
for f in Dockerfile build.sh run.sh container-entry.sh scripts/prepare-deps.sh scripts/prepare-mobilenet.sh; do
    if grep -nE 'lib/armv7l|ld-linux-armhf\.so\.3|--platform[ =]+linux/arm/v7' "$f"; then
        fail "unconditional ARMv7 literal remains in common file: $f"
    fi
done

grep -q 'USE_CMAKE_TOOLCHAIN' Dockerfile || fail 'Dockerfile lacks conditional toolchain selection'
grep -q 'ov203-build-${TARGET}' Dockerfile || fail 'Docker build cache is not target-qualified'
grep -q 'libmyriadPlugin.so' Dockerfile || fail 'MYRIAD plugin validation missing'
grep -q 'usb-ma2450.mvcmd' Dockerfile || fail 'MA2450 firmware validation missing'
grep -q 'readelf -h "${IE_PLUGIN}"' Dockerfile || fail 'MYRIAD plugin ELF architecture validation missing'
grep -q 'EXPECTED_ELF_MACHINE_ID' Dockerfile || fail 'runtime manifest lacks ELF machine metadata'

# OV linking is centralised in cmake/ov203-link.cmake.  The shared module
# must discover lib/<arch> dynamically, every CMakeLists must use it, and
# no CMake file may pin or re-implement an architecture lib dir.
grep -q 'lib/\*' cmake/ov203-link.cmake || fail 'cmake/ov203-link.cmake does not discover architecture lib dirs'
if grep -nE 'lib/(armv7l|aarch64|arm64|intel64|x86_64)' cmake/ov203-link.cmake; then
    fail 'cmake/ov203-link.cmake hard-codes an architecture lib dir'
fi
while IFS= read -r f; do
    grep -q 'ov203-link' "$f" || fail "$f does not include the shared OV link module"
    if grep -nE 'lib/(armv7l|aarch64|arm64|intel64|x86_64)' "$f"; then fail "$f hard-codes an architecture lib dir"; fi
done < <(find . -path './vendor' -prune -o -path './work' -prune -o -type f -name CMakeLists.txt -print | sort)

# ARM64 must be treated as a native target, not as ARMv7 with a renamed image.
grep -q 'arm64)' scripts/platform.sh || fail 'platform helper lacks arm64 target'
grep -q 'DOCKER_PLATFORM="linux/arm64"' scripts/platform.sh || fail 'arm64 Docker platform missing'
grep -q 'EXPECTED_ELF_MACHINE_ID=183' scripts/platform.sh || fail 'arm64 ELF machine ID missing'
grep -q 'arm64:aarch64' scripts/host-run.sh || fail 'host-run lacks native arm64 host acceptance'
grep -q 'arm64:arm64' scripts/verify.sh || fail 'runtime verifier lacks arm64 image validation'

python3 - <<'PY_CHECK'
from pathlib import Path
for name in ["smoke-test/model/make_tiny_ir.py", "scripts/reset-stick.py"]:
    src = Path(name).read_text()
    compile(src, name, "exec")
    print(f"python syntax: {name}: OK")
PY_CHECK

echo 'static multi-platform checks: PASS'
