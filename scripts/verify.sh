#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify.sh - one report that shows the whole stack works: host, target
# userspace, the built runtime image, and a real inference on the Movidius stick.
#
#   ./scripts/verify.sh > logs/VERIFICATION.md
#   ./scripts/verify.sh --no-device     # skip the two steps that need the stick
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# Load platform variables without inheriting platform.sh's `set -e` (this
# script is designed to print every section even when a step fails).
platform_dump=$(bash -c "source scripts/platform.sh; platform_load \"\$(platform_default_request)\"; \
    printf 'TARGET=%s\\nDOCKER_PLATFORM=%s\\nDEFAULT_IMAGE=%s\\nEXPECTED_ELF_CLASS=%s\\nEXPECTED_ELF_MACHINE_ID=%s\\nTARGET_LIB_DIR=%s\\n' \
    \"\${TARGET}\" \"\${DOCKER_PLATFORM}\" \"\${DEFAULT_IMAGE}\" \"\${EXPECTED_ELF_CLASS}\" \"\${EXPECTED_ELF_MACHINE_ID}\" \"\${TARGET_LIB_DIR}\"" )
eval "${platform_dump}"

# docker image .Architecture per target: armv7:arm, arm64:arm64, amd64:amd64
IMAGE_ARCH="armv7:arm arm64:arm64 amd64:amd64"
EXPECT_IMAGE_ARCH=$(for p in ${IMAGE_ARCH}; do [[ "${p%%:*}" == "${TARGET}" ]] && printf '%s' "${p#*:}"; done)

IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"
WITH_DEVICE=1
[[ "${1:-}" == "--no-device" ]] && WITH_DEVICE=0

have_image() { docker image inspect "${IMAGE}" >/dev/null 2>&1; }

sec() { printf '\n## %s\n\n```text\n' "$*"; }
end() { printf '```\n'; }

echo '# Verification report - OpenVINO 2020.3.2 / Movidius MYRIAD on Raspberry Pi 5'
echo
echo "Generated: $(date -Is)   host: $(hostname)   image: ${IMAGE}"

# ---------------------------------------------------------------------------
sec '1. Host'
uname -a
uname -m
dpkg --print-architecture
free -h | head -2
df -h / | tail -1
printf 'CONFIG_COMPAT: '
if [[ -r "/boot/config-$(uname -r)" ]]; then
    grep -m1 CONFIG_COMPAT "/boot/config-$(uname -r)"
else
    echo 'not readable (kernel config unavailable)'
fi
printf 'binfmt qemu-arm (little-endian arm32 emulation) present: '
if ls /proc/sys/fs/binfmt_misc/ 2>/dev/null | grep -qx 'qemu-arm'; then
    echo 'yes - arm32 code could be emulated'
else
    echo 'no - 32-bit armv7 code must run natively on the aarch64 kernel'
fi
printf 'Movidius stick: '
lsusb | grep -i -E 'myriad|03e7' || echo 'not visible (unplugged or held by another process)'
end

# ---------------------------------------------------------------------------
sec '2. Docker / BuildKit'
docker --version
docker version --format 'server {{.Server.Version}} {{.Server.Os}}/{{.Server.Arch}}'
docker buildx version | head -1
docker info --format 'docker info: server={{.ServerVersion}} os={{.OSType}} arch={{.Architecture}} kernel={{.KernelVersion}} cpus={{.NCPU}} containers={{.Containers}} images={{.Images}} storage={{.Driver}} cgroup={{.CgroupDriver}}'
docker image inspect "${IMAGE}" \
    --format 'image {{.Id}} {{.Os}}/{{.Architecture}} size={{.Size}' 2>/dev/null \
    || echo "image ${IMAGE} not built yet (run ./build.sh --platform ${TARGET})"
docker image inspect --format '{{.Architecture}}' "${IMAGE}" 2>/dev/null | \
    grep -qx "${EXPECT_IMAGE_ARCH}" || \
    echo "ERROR: image architecture does not match target ${TARGET} (expected '${EXPECT_IMAGE_ARCH}')"
end

# ---------------------------------------------------------------------------
sec "3. The container userspace matches target ${TARGET} (${EXPECTED_ELF_CLASS})"
docker run --rm --platform "${DOCKER_PLATFORM}" "${IMAGE}" bash -c '
    echo "kernel   : $(uname -m)          (the host kernel)"
    echo "userspace: $(getconf LONG_BIT)-bit"
    echo "uname -P : $(uname -p 2>/dev/null || echo unknown)"
    echo "hello_myriad ELF header:"
    od -An -tx1 -N20 /opt/openvino/bin/hello_myriad | sed "s/^/  /"
    echo "  bytes: EI_CLASS = offset 4, e_machine = offset 18-19 (little endian)"
    echo "linked libraries (rpath resolved):"
    ldd /opt/openvino/bin/hello_myriad | sed "s/^/  /"
' 2>&1
case "${EXPECTED_ELF_CLASS}" in
    ELF32) EI_BYTE=1 ;;
    ELF64) EI_BYTE=2 ;;
esac
printf '  expected for %s: EI_CLASS=%d (%s), e_machine=%02x %02x (machine id %d)\n' \
    "${TARGET}" "${EI_BYTE}" "${EXPECTED_ELF_CLASS}" \
    $(( EXPECTED_ELF_MACHINE_ID & 0xff )) $(( EXPECTED_ELF_MACHINE_ID >> 8 )) \
    "${EXPECTED_ELF_MACHINE_ID}"
end

# ---------------------------------------------------------------------------
sec '4. Runtime image contents'
docker run --rm --platform "${DOCKER_PLATFORM}" --entrypoint bash "${IMAGE}" -c \
    "echo \"plugins shipped in plugins.xml:\"
    grep -o \"plugin name=\\\"[A-Z]*\\\"\" /opt/openvino/inference_engine/${TARGET_LIB_DIR}/plugins.xml
    echo \"firmware next to libmyriadPlugin.so (getFirmwarePath uses dladdr):\"
    ls -1 /opt/openvino/inference_engine/${TARGET_LIB_DIR}/*.mvcmd
    echo \"ngraph part of the install tree:\"
    ls -1 /opt/openvino/ngraph/lib" 2>&1
end

if [[ ${WITH_DEVICE} == 1 ]]; then
    # -----------------------------------------------------------------------
    sec '5. Device enumeration through the Inference Engine'
    timeout 300 ./run.sh list 2>&1
    end

    # -----------------------------------------------------------------------
    sec '6. Compile + inference on the Movidius stick'
    sudo python3 scripts/reset-stick.py 2>&1 | tail -2
    timeout 600 ./run.sh bench --model /opt/openvino-demo/model/model.xml \
        --weights /opt/openvino-demo/model/model.bin --iterations 10 2>&1
    printf '\nafter teardown the stick is back in ROM mode: '
    lsusb | grep -i -E 'myriad|03e7' || echo 'not visible'
    end

    # -----------------------------------------------------------------------
    sec '7. compile_tool from the same build (blob produced on the VPU)'
    timeout 600 docker run --rm --platform "${DOCKER_PLATFORM}" --network=host \
        -v /dev:/dev --device-cgroup-rule='c 189:* rwm' --entrypoint bash \
        -v "${ROOT}/smoke-test/model:/model:ro" -v "${ROOT}/work:/out" "${IMAGE}" -c \
        "set -eu
            export LD_LIBRARY_PATH=/opt/openvino/inference_engine/${TARGET_LIB_DIR}:/opt/openvino/ngraph/lib
            rm -f /out/verify_blob.bin
            /opt/openvino/inference_engine/${TARGET_LIB_DIR}/compile_tool -m /model/model.xml -d MYRIAD -ip FP16 -o /out/verify_blob.bin 2>&1 | tail -3
            md5sum /out/verify_blob.bin
            ls -l /out/verify_blob.bin" 2>&1 | tail -6
    if [[ -f work/reference-check/blob/blob.bin ]]; then
        echo "blob produced by the official Intel raspbian runtime (same IR, same -d MYRIAD):"
        md5sum work/reference-check/blob/blob.bin
    fi
    rm -f work/verify_blob.bin
    sudo python3 scripts/reset-stick.py 2>&1 | tail -1
    end
fi

# ---------------------------------------------------------------------------
sec '8. Python policy (no Python libraries from APT)'
echo 'APT package lines in the Dockerfile (no python3-* library packages):'
awk '/apt-get .*install/ { grab=1; next }
     grab {
       line = $0
       sub(/[[:space:]]+$/, "", line)
       while (sub(/[[:space:]]*[;\\\\][[:space:]]*$/, "", line)) { }
       if (line ~ /^[[:space:]]+[a-z0-9][a-z0-9+.-]+$/) print line; else grab=0
     }' Dockerfile
echo
echo 'python lines in the Dockerfile:'
grep -n 'python' Dockerfile | sed 's/^/  /'
echo 'pinned pip libraries (requirements-build.txt):'
grep -vE '^\s*#|^\s*$' requirements-build.txt | sed 's/^/  /'
end

sec '9. Vendor pin'
git -C vendor/openvino-2020.3.2 rev-parse HEAD 2>/dev/null | sed 's/^/  openvino commit: /'
git -C vendor/openvino-2020.3.2 describe --tags 2>/dev/null | sed 's/^/  tag: /'
git -C vendor/openvino-2020.3.2 submodule status 2>/dev/null | sed 's/^/  /'
end

sec '10. MobileNet v2 converted by the vendored Model Optimizer, run on the stick'
if [[ -d vendor/models/mobilenet-v2-ov203/fp16 && -f vendor/models/test_data/output_0.f32 ]]; then
    echo 'corpus produced by ./scripts/prepare-mobilenet.sh:'
    find vendor/models -maxdepth 3 -type f -not -name '*.tar.gz' -not -name '*.pb' \
        -not -path '*/images/*' -printf '%10s  %P\n' | sort -k2
    echo '    plus vendor/models/images/*.ppm (224x224) and vendor/models/images/full/*.ppm'
    echo
    if [[ "$WITH_DEVICE" == 1 ]]; then
        echo 'numerical check against the ONNX model zoo reference output:'
        timeout 900 ./run.sh mobilenet --tensor /models/test_data/input_0.f32 \
            --reference /models/test_data/output_0.f32 --iterations 10 2>&1 |
            grep -E 'model |network  |input  |output |load\+compile|inference |max \|diff|mean \|diff|top-1  |^RESULT'
        echo
        echo 'top-1 class per photo (ImageNet centre crop, FP16 weights):'
        for n in dog cat eagle banana cup; do
            if [[ -f "vendor/models/images/$n.ppm" ]]; then
                printf '%-7s ' "$n"
                timeout 300 ./run.sh mobilenet --image "/models/images/$n.ppm" --topk 1 2>&1 |
                    grep -m1 -E '^   1\.' || echo 'no result'
                sleep 1
            fi
        done
    else
        echo 'skipped: --no-device'
    fi
else
    echo 'corpus not present - run ./scripts/prepare-mobilenet.sh first'
fi
end
