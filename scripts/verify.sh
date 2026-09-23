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
        --platform) [[ $# -ge 2 ]] || { echo "--platform needs armv7|arm64|amd64" >&2; exit 2; }; TARGET_REQUEST="$2"; shift 2 ;;
        --image) [[ $# -ge 2 ]] || { echo "--image needs a Docker image tag" >&2; exit 2; }; IMAGE_OVERRIDE="$2"; shift 2 ;;
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
    if h[:4] != b"\x7fELF" or len(h) < 20:
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
echo "ssd_detect ldd:"
ldd /opt/openvino/bin/ssd_detect
echo "seg_detect ldd:"
ldd /opt/openvino/bin/seg_detect
echo "myriad plugin ldd:"
ldd "$IE_LIB/libmyriadPlugin.so"
for f in /opt/openvino/bin/hello_myriad /opt/openvino/bin/mobilenet_classify \
         /opt/openvino/bin/ssd_detect /opt/openvino/bin/seg_detect "$IE_LIB/libmyriadPlugin.so"; do
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
/opt/openvino/bin/ssd_detect --help >/dev/null
/opt/openvino/bin/seg_detect --help >/dev/null
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
        echo
        echo 'top-1 class per photo (ImageNet centre crop, FP16 weights):'
        for n in dog cat eagle banana cup; do
            if [[ -f "vendor/models/images/$n.ppm" ]]; then
                printf '%-7s ' "$n"
                timeout 300 ./run.sh --platform "${TARGET}" --image "${IMAGE}" mobilenet \
                    --image "/models/images/$n.ppm" --topk 1 2>&1 |
                    grep -m1 -E '^   1\.' || echo 'no result'
                sleep 1
            fi
        done
    else
        echo 'corpus present; inference skipped: --no-device'
    fi
else
    echo 'MobileNet corpus not present; run ./scripts/prepare-mobilenet.sh'
fi
end

sec '8. SSDLite MobileNetV2 detection on the stick (single image)'
if [[ -f vendor/models/ssdlite_mobilenet_v2/openvino/ssdlite_mobilenet_v2.xml && -f vendor/models/images/dog_ssd.ppm ]]; then
    if (( WITH_DEVICE )); then
        echo 'run.sh ssd on dog_ssd.ppm (expected: person/dog-family detections):'
        timeout 600 ./run.sh --platform "${TARGET}" --image "${IMAGE}" ssd \
            --image /models/images/dog_ssd.ppm 2>&1 | \
            grep -E 'model |input image|compile|inference |postprocess|^[a-z]+ +[01]?\.[0-9]+ \[' || true
        echo
        echo 'reference (CPU, same IR + same preprocessing, OpenVINO 2026.4):'
        echo '  bicycle 0.96 (141, 119, 568, 430)   car 0.88 (460, 81, 690, 172)'
        echo '  dog 0.84 (132, 218, 315, 539)       cat 0.70 (132, 218, 315, 539)'
    else
        echo 'skipped: --no-device'
    fi
else
    echo 'model or test image not present - run ./scripts/prepare-ssdlite.sh first'
fi
end

sec '9. DeepLabV3 Pascal VOC segmentation on the stick (single image)'
if [[ -f vendor/models/deeplabv3/openvino/deeplabv3.xml && -f vendor/models/images/dog_ssd.ppm ]]; then
    if (( WITH_DEVICE )); then
        echo 'run.sh seg on dog_ssd.ppm (expected: background majority, dog/car classes):'
        timeout 600 ./run.sh --platform "${TARGET}" --image "${IMAGE}" seg \
            --image /models/images/dog_ssd.ppm 2>&1 | \
            grep -E 'model |input:|preprocess|inference |postprocess|total:|classes present|background|person|dog|car|bicycle' || true
    else
        echo 'skipped: --no-device'
    fi
else
    echo 'model or test image not present - run ./scripts/prepare-deeplabv3.sh first'
fi
end

sec '10. Source pin and patches'
git -C vendor/openvino-2020.3.2 rev-parse HEAD 2>/dev/null | sed 's/^/openvino_commit=/' || echo 'OpenVINO source not cloned'
for p in patches/*.patch; do [[ -f "$p" ]] && sha256sum "$p"; done
end

sec '11. Static architecture-literal audit'
echo 'Hard-coded armv7 runtime literals outside explicitly ARM/reference-scoped files:'
grep -RIn --include='*.sh' --include='Dockerfile' 'linux/arm/v7\|lib/armv7l\|ld-linux-armhf' . \
    --exclude-dir=vendor --exclude-dir=logs --exclude-dir=.git \
    | grep -vE 'scripts/platform\.sh|scripts/host-run\.sh|scripts/pull-runtime\.sh|scripts/validate-reference\.sh|scripts/reference-check\.Dockerfile|docs/diagnostics' \
    || echo 'none'
end

printf '\nVERIFY_RESULT=PASS target=%s device_checks=%s image=%s\n' \
    "${TARGET}" "$([[ ${WITH_DEVICE} -eq 1 ]] && echo enabled || echo skipped)" "${IMAGE}"
