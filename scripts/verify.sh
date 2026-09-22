#!/usr/bin/env bash
# Produce a strict, platform-aware verification report for the self-built runtime.
# Hardware-dependent checks can be skipped with --no-device.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck source=platform.sh
source "${ROOT}/scripts/platform.sh"

TARGET_REQUEST="$(platform_default_request)"
IMAGE_OVERRIDE="${IMAGE:-}"
WITH_DEVICE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) TARGET_REQUEST="$2"; shift 2 ;;
        --image) IMAGE_OVERRIDE="$2"; shift 2 ;;
        --no-device) WITH_DEVICE=0; shift ;;
        -h|--help) echo "usage: $0 [--platform armv7|arm64|amd64] [--image TAG] [--no-device]"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
platform_load "${TARGET_REQUEST}" || exit 1
IMAGE="${IMAGE_OVERRIDE:-${DEFAULT_IMAGE}}"

sec() { printf '\n## %s\n\n```text\n' "$*"; }
end() { printf '```\n'; }

printf '# Verification report - OpenVINO 2020.3.2 / Movidius MYRIAD (%s)\n\n' "${TARGET}"
printf 'Generated: %s   host: %s   image: %s\n' "$(date -Is)" "$(hostname)" "${IMAGE}"

sec '1. Selected platform and host'
platform_print
printf 'host_uname_m=%s\n' "$(uname -m)"
uname -a
if command -v dpkg >/dev/null 2>&1; then dpkg --print-architecture; fi
if command -v lsusb >/dev/null 2>&1; then
    printf 'Movidius USB: '
    lsusb | grep -i -E 'myriad|03e7' || echo 'not visible'
fi
if [[ "${TARGET}" == armv7 ]]; then
    printf 'CONFIG_COMPAT: '
    if [[ -r "/boot/config-$(uname -r)" ]]; then grep -m1 CONFIG_COMPAT "/boot/config-$(uname -r)" || true; else echo unavailable; fi
fi
end

sec '2. Docker image metadata'
command -v docker >/dev/null 2>&1 || { echo 'FAIL: docker is required for runtime verification'; end; exit 1; }
docker --version
docker image inspect "${IMAGE}" --format 'image={{.Id}} os={{.Os}} arch={{.Architecture}} size={{.Size}}'
IMAGE_ARCH="$(docker image inspect "${IMAGE}" --format '{{.Architecture}}')"
case "${TARGET}:${IMAGE_ARCH}" in
    armv7:arm|arm64:arm64|amd64:amd64) ;;
    *) echo "FAIL: image architecture ${IMAGE_ARCH} does not match target ${TARGET}"; end; exit 1 ;;
esac
end

sec '3. Runtime manifest and native ELF architecture'
docker run --rm --platform "${DOCKER_PLATFORM}" --entrypoint bash "${IMAGE}" -lc '
set -euo pipefail
cat /opt/openvino/runtime-manifest.env
cat /opt/openvino/BUILD-INFO.txt
IE_LIB="$(dirname "$(find /opt/openvino/inference_engine/lib -mindepth 2 -maxdepth 2 -type f -name libmyriadPlugin.so -print -quit)")"
test -n "$IE_LIB"
echo "IE_LIB=$IE_LIB"
python3 - "$OV_TARGET" /opt/openvino/bin/hello_myriad "$IE_LIB/libmyriadPlugin.so" <<"PY"
import struct, sys

target = sys.argv[1]
expected = {"armv7": (1, 40), "arm64": (2, 183), "amd64": (2, 62)}[target]
for path in sys.argv[2:]:
    with open(path, "rb") as f:
        h = f.read(20)
    if h[:4] != b"\\x7fELF" or len(h) < 20:
        raise SystemExit(f"not an ELF file: {path}")
    elf_class = h[4]
    byte_order = "little" if h[5] == 1 else "big"
    machine = int.from_bytes(h[18:20], byte_order)
    print(f"{path}: ELFCLASS={elf_class} e_machine={machine}")
    if (elf_class, machine) != expected:
        raise SystemExit(
            f"ELF architecture mismatch for {path}: got {(elf_class, machine)}, expected {expected}"
        )
PY
'
end

sec '4. MYRIAD plugin, registry, firmware and dynamic libraries'
docker run --rm --platform "${DOCKER_PLATFORM}" --entrypoint bash "${IMAGE}" -lc '
set -euo pipefail
IE_LIB="$(dirname "$(find /opt/openvino/inference_engine/lib -mindepth 2 -maxdepth 2 -type f -name libmyriadPlugin.so -print -quit)")"
test -f "$IE_LIB/libmyriadPlugin.so"
test -f "$IE_LIB/plugins.xml"
test -f "$IE_LIB/usb-ma2450.mvcmd"
grep -qi "MYRIAD" "$IE_LIB/plugins.xml"
ls -l "$IE_LIB/libmyriadPlugin.so" "$IE_LIB/plugins.xml" "$IE_LIB"/*.mvcmd
cat /opt/openvino/firmware.sha256
(cd /opt/openvino && sha256sum -c firmware.sha256)
export LD_LIBRARY_PATH="$IE_LIB:/opt/openvino/ngraph/lib"
echo "hello_myriad ldd:"
ldd /opt/openvino/bin/hello_myriad
echo "mobilenet_classify ldd:"
ldd /opt/openvino/bin/mobilenet_classify
echo "myriad plugin ldd:"
ldd "$IE_LIB/libmyriadPlugin.so"
for f in /opt/openvino/bin/hello_myriad /opt/openvino/bin/mobilenet_classify "$IE_LIB/libmyriadPlugin.so"; do
    ! ldd "$f" | grep -q "not found"
done
if [[ -x "$IE_LIB/myriad_compile" ]]; then
    echo "myriad_compile startup:"
    "$IE_LIB/myriad_compile" --help 2>&1 | head -12 || true
fi
'
end

sec '5. Plugin/application startup without requiring hardware'
docker run --rm --platform "${DOCKER_PLATFORM}" --entrypoint bash "${IMAGE}" -lc '
set -euo pipefail
IE_LIB="$(dirname "$(find /opt/openvino/inference_engine/lib -mindepth 2 -maxdepth 2 -type f -name libmyriadPlugin.so -print -quit)")"
export LD_LIBRARY_PATH="$IE_LIB:/opt/openvino/ngraph/lib"
/opt/openvino/bin/hello_myriad --help
/opt/openvino/bin/mobilenet_classify --help
'
end

sec '6. Device enumeration and tiny inference'
if (( WITH_DEVICE )); then
    timeout 300 ./run.sh --platform "${TARGET}" --image "${IMAGE}" list
    echo
    timeout 600 ./run.sh --platform "${TARGET}" --image "${IMAGE}"
else
    echo 'skipped: --no-device'
fi
end

sec '7. MobileNet functional check'
if [[ -f vendor/models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml && -f vendor/models/test_data/output_0.f32 ]]; then
    if (( WITH_DEVICE )); then
        timeout 900 ./run.sh --platform "${TARGET}" --image "${IMAGE}" mobilenet \
            --tensor /models/test_data/input_0.f32 --reference /models/test_data/output_0.f32 --iterations 3
    else
        echo 'corpus present; inference skipped: --no-device'
    fi
else
    echo 'MobileNet corpus not present; run ./scripts/prepare-mobilenet.sh'
fi
end

sec '8. Source pin and patches'
git -C vendor/openvino-2020.3.2 rev-parse HEAD 2>/dev/null | sed 's/^/openvino_commit=/' || echo 'OpenVINO source not cloned'
for p in patches/*.patch; do [[ -f "$p" ]] && sha256sum "$p"; done
end

sec '9. Static architecture-literal audit'
echo 'Hard-coded armv7 runtime literals outside explicitly ARM/reference-scoped files:'
grep -RIn --include='*.sh' --include='Dockerfile' 'linux/arm/v7\|lib/armv7l\|ld-linux-armhf' . \
    --exclude-dir=vendor --exclude-dir=logs --exclude-dir=.git \
    | grep -vE 'scripts/platform\.sh|scripts/host-run\.sh|scripts/pull-runtime\.sh|scripts/validate-reference\.sh|scripts/reference-check\.Dockerfile|docs/diagnostics' \
    || echo 'none'
end

printf '\nVERIFY_RESULT=PASS target=%s device_checks=%s image=%s\n' \
    "${TARGET}" "$([[ ${WITH_DEVICE} -eq 1 ]] && echo enabled || echo skipped)" "${IMAGE}"
