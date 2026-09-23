#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/host-run.sh - run the extracted OpenVINO 2020.3.2 runtime directly on
# the host: no Docker, no container, no QEMU at inference time.
#
#   ./scripts/host-run.sh list                    device enumeration
#   ./scripts/host-run.sh demo                    tiny model on the stick
#   ./scripts/host-run.sh mobilenet               numerical self-test on the stick
#   ./scripts/host-run.sh mobilenet --image vendor/models/images/banana.ppm --topk 3
#   ./scripts/host-run.sh compile --model vendor/models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml \
#                                 --blob  vendor/models/blobs/mobilenet-v2.blob
#   ./scripts/host-run.sh shell                   bash with ov_* wrappers on PATH
#
# The extracted runtime lives in work/host-runtime/<target>/ (one directory per
# target, with a host-runtime.env manifest), populated by pull-runtime.sh.
#
# Target selection follows scripts/platform.sh (OV_PLATFORM / TARGET env or
# host-CPU auto-detection):
#   armv7  the extracted binaries are ARMHF; they run through the armhf loader
#          (sysroot/lib/ld-linux-armhf.so.3) pulled by pull-runtime.sh, either
#          on an armv7l host or an aarch64 host in compat mode.
#   arm64  native AArch64 host; the binaries run directly, no loader.
#   amd64  native x86_64 host; the binaries run directly, no loader.
#
# See logs/HOST-RUN.md for the two host-side requirements (USB node
# permissions, /tmp/mvnc.mutex ownership).
# ---------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/platform.sh
source "${ROOT}/scripts/platform.sh"

TARGET_REQUEST="$(platform_default_request)"
IMAGE_OVERRIDE="${IMAGE:-}"
VERBOSE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) [[ $# -ge 2 ]] || { echo "--platform needs armv7|arm64|amd64" >&2; exit 2; }; TARGET_REQUEST="$2"; shift 2 ;;
        --image) [[ $# -ge 2 ]] || { echo "--image needs a Docker image tag" >&2; exit 2; }; IMAGE_OVERRIDE="$2"; shift 2 ;;
        --verbose|-v) VERBOSE=1; shift ;;
        --print-platform) platform_load "${TARGET_REQUEST}"; IMAGE="${IMAGE_OVERRIDE:-${DEFAULT_IMAGE}}"; platform_print; exit 0 ;;
        -h|--help)
            awk '/^set -euo/{exit} {print}' "$0"
            exit 0
            ;;
        *) break ;;
    esac
done
platform_load "${TARGET_REQUEST}"
IMAGE="${IMAGE_OVERRIDE:-${DEFAULT_IMAGE}}"

case "${TARGET}:$(uname -m)" in
    armv7:aarch64|armv7:arm64|armv7:armv7l|armv7:armv7*) ;;
    arm64:aarch64|arm64:arm64) ;;
    amd64:x86_64|amd64:amd64) ;;
    *)
        echo "host-native target mismatch: selected ${TARGET}, host is $(uname -m)" >&2
        echo "use Docker via ./run.sh for cross-architecture execution" >&2
        exit 1
        ;;
esac

RT="${ROOT}/work/host-runtime/${TARGET}"
OV="${RT}/openvino"
if [[ ! -d "${OV}" ]]; then
    echo "host runtime ${TARGET} is not populated - extracting it from ${IMAGE}" >&2
    "${ROOT}/scripts/pull-runtime.sh" --platform "${TARGET}" --image "${IMAGE}"
fi

manifest="${RT}/host-runtime.env"
manifest_target="$(platform_read_manifest_target "${manifest}" 2>/dev/null || true)"
if [[ "${manifest_target}" != "${TARGET}" ]]; then
    echo "host runtime manifest mismatch: ${manifest} says '${manifest_target:-unknown}', expected ${TARGET}" >&2
    echo "rerun ./scripts/pull-runtime.sh --platform ${TARGET}" >&2
    exit 1
fi

IE_LIB="$(platform_find_ie_libdir "${OV}/inference_engine")"
if (( VERBOSE )); then
    echo '--- host runtime manifest ---'
    cat "${manifest}"
    [[ -f "${OV}/BUILD-INFO.txt" ]] && { echo '--- build info ---'; cat "${OV}/BUILD-INFO.txt"; }
fi
NGRAPH_LIB="${OV}/ngraph/lib"
LP="${IE_LIB}:${NGRAPH_LIB}"
BIN="${RT}/bin"
IR="${IR:-fp16}"
MODELS="${ROOT}/vendor/models"
COMPILE_TOOL="${IE_LIB}/compile_tool"
[[ -x "${COMPILE_TOOL}" ]] || COMPILE_TOOL="${IE_LIB}/myriad_compile"

if [[ ! -f "${IE_LIB}/libmyriadPlugin.so" || ! -f "${IE_LIB}/usb-ma2450.mvcmd" ]]; then
    echo "incomplete MYRIAD runtime in ${IE_LIB}; expected libmyriadPlugin.so and usb-ma2450.mvcmd" >&2
    exit 1
fi

mkdir -p "${BIN}"
if [[ "${TARGET}" == armv7 ]]; then
    SYSROOT="${RT}/sysroot"
    LD="${SYSROOT}/lib/ld-linux-armhf.so.3"
    [[ -x "${LD}" ]] || { echo "ARMHF loader missing: ${LD}; rerun pull-runtime.sh" >&2; exit 1; }
    LP="${SYSROOT}/lib/arm-linux-gnueabihf:${SYSROOT}/usr/lib/arm-linux-gnueabihf:${LP}"
    ov_run() { "${LD}" --library-path "${LP}" "$@"; }
else
    ov_run() { env LD_LIBRARY_PATH="${LP}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "$@"; }
fi

# Regenerate convenient wrappers for interactive shell mode.
rm -f "${BIN}"/* 2>/dev/null || true
for b in "${OV}/bin/"* "${COMPILE_TOOL}"; do
    [[ -x "${b}" && ! -d "${b}" ]] || continue
    name="$(basename "${b}")"
    if [[ "${TARGET}" == armv7 ]]; then
        cat > "${BIN}/${name}" <<EOF_WRAP
#!/usr/bin/env bash
exec "${LD}" --library-path "${LP}" "${b}" "\$@"
EOF_WRAP
    else
        cat > "${BIN}/${name}" <<EOF_WRAP
#!/usr/bin/env bash
export LD_LIBRARY_PATH="${LP}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
exec "${b}" "\$@"
EOF_WRAP
    fi
    chmod +x "${BIN}/${name}"
done

# mvnc's global lock is a fixed path, open()ed with O_CREAT by whoever gets
# there first.  With fs.protected_regular != 0 (Debian: 2) a file created by
# another user in world-writable /tmp is not openable - even by root - and
# mvnc aborts with "global mutex initialization failed".  Warn instead of
# exiting(1).
mutex="${MVNC_MUTEX:-/tmp/mvnc.mutex}"
if [[ -e "${mutex}" ]]; then
    owner="$(stat -c '%u %a' "${mutex}")"
    mode="${owner##* }"
    other_bit=$(( 8#${mode: -1} & 4 ))
    if [[ "${owner%% *}" != "$(id -u)" && ${other_bit} -eq 0 ]]; then
        echo "WARNING: ${mutex} is owned by uid ${owner%% *} mode ${mode}; mvnc may fail to initialize its global mutex." >&2
        echo "fix: remove ${mutex}, use one user consistently, or chmod 0666 ${mutex}" >&2
    fi
fi

MODE="${1:-list}"; shift || true
case "${MODE}" in
    list)
        ov_run "${OV}/bin/hello_myriad" --list-only "$@" || {
            if command -v lsusb >/dev/null 2>&1; then
                echo "hint: host stick state -> $(lsusb | grep -i 03e7 || echo 'no 03e7 device visible')" >&2
            fi
            echo "hint: verify permissions on /dev/bus/usb/*/* and that ${IE_LIB}/usb-ma2450.mvcmd is readable" >&2
            exit 1
        }
        ;;
    demo)
        ov_run "${OV}/bin/hello_myriad" \
            --model "${RT}/openvino-demo/model/model.xml" \
            --weights "${RT}/openvino-demo/model/model.bin" "$@"
        ;;
    mobilenet)
        if [[ ! -f "${MODELS}/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.xml" ]]; then
            echo "no IR in ${MODELS}/mobilenet-v2-ov203/${IR} - run ./scripts/prepare-mobilenet.sh first" >&2
            exit 1
        fi
        if [[ $# -eq 0 ]]; then
            set -- --tensor "${MODELS}/test_data/input_0.f32" --reference "${MODELS}/test_data/output_0.f32"
        fi
        ov_run "${OV}/bin/mobilenet_classify" \
            --model "${MODELS}/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.xml" \
            --weights "${MODELS}/mobilenet-v2-ov203/${IR}/mobilenet-v2-ov203.bin" \
            --labels "${MODELS}/labels/synset.txt" "$@"
        ;;
    compile)
        model=""; blob=""
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --model) model="$2"; shift 2 ;;
                --blob) blob="$2"; shift 2 ;;
                *) echo "unknown option: $1" >&2; exit 2 ;;
            esac
        done
        [[ -n "${model}" && -n "${blob}" ]] || { echo "compile needs --model <xml> --blob <path>" >&2; exit 2; }
        mkdir -p "$(dirname "${blob}")"
        ov_run "${COMPILE_TOOL}" -m "${model}" -d MYRIAD -o "${blob}"
        ls -l "${blob}"
        ;;
    shell)
        echo "host-native OpenVINO 2020.3 shell (${TARGET}); wrappers are on PATH"
        if [[ "${TARGET}" == armv7 ]]; then
            PATH="${BIN}:${PATH}" PS1="hostov-${TARGET}# " exec bash
        else
            LD_LIBRARY_PATH="${LP}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" PATH="${BIN}:${PATH}" PS1="hostov-${TARGET}# " exec bash
        fi
        ;;
    *) echo "unknown mode: ${MODE}" >&2; exit 2 ;;
esac
