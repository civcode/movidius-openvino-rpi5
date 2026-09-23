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
platform_load "$(platform_default_request)"

RT="${ROOT}/work/host-runtime"
OV="${RT}/openvino"
LIBDIR="${OV}/inference_engine/lib/${TARGET_LIB_DIR}"
IR="${IR:-fp16}"
MODELS="${ROOT}/vendor/models"

# Host acceptance: target:host-CPU pairs that are allowed to run this target.
# armv7 binaries run on an aarch64 host in compat mode (CONFIG_COMPAT=y).
HOST_ACCEPT="armv7:armv7l armv7:aarch64 arm64:aarch64 amd64:x86_64"
host_cpu="$(uname -m)"
accepted=0
for pair in ${HOST_ACCEPT}; do
    [[ "${pair%%:*}" == "${TARGET}" && "${pair#*:}" == "${host_cpu}" ]] && accepted=1
done
if (( ! accepted )); then
    echo "target '${TARGET}' cannot run on host CPU '${host_cpu}' (accepted: ${HOST_ACCEPT});" >&2
    echo "try OV_PLATFORM=armv7|arm64|amd64 and rebuild with ./build.sh --platform <target>" >&2
    exit 1
fi

if [[ ! -d "${OV}" || ! -x "${OV}/bin/hello_myriad" ]]; then
    echo "work/host-runtime is not populated yet - running ./scripts/pull-runtime.sh" >&2
    "${ROOT}/scripts/pull-runtime.sh"
fi

if [[ "${TARGET}" == armv7 ]]; then
    # armhf glibc resolves a SONAME only from a directory it can search; the
    # PT_INTERP /lib/ld-linux-armhf.so.3 does not exist on an aarch64 host, so
    # every binary needs the loader wrapper below.  They are cheap and
    # regenerated on every call.
    SYSROOT="${RT}/sysroot"
    LD="${SYSROOT}/lib/ld-linux-armhf.so.3"
    if [[ ! -x "${LD}" ]]; then
        echo "armhf sysroot missing - re-running ./scripts/pull-runtime.sh" >&2
        "${ROOT}/scripts/pull-runtime.sh"
    fi
    LP="${SYSROOT}/lib/arm-linux-gnueabihf:${SYSROOT}/usr/lib/arm-linux-gnueabihf:${LIBDIR}:${OV}/ngraph/lib"
    BIN="${RT}/bin"
    mkdir -p "${BIN}"
    for b in "${OV}/bin/"* "${LIBDIR}/compile_tool"; do
        [[ -x "${b}" && ! -d "${b}" ]] || continue
        name="$(basename "${b}")"
        cat > "${BIN}/${name}" <<EOF
#!/usr/bin/env bash
exec "${LD}" --library-path "${LP}" "${b}" "\$@"
EOF
        chmod +x "${BIN}/${name}"
    done
    ov_run() { "${LD}" --library-path "${LP}" "$@"; }
else
    # Native target: binaries match the host, so no loader is needed.
    LP="${LIBDIR}:${OV}/ngraph/lib"
    BIN="${OV}/bin"
    ov_run() { LD_LIBRARY_PATH="${LP}" "$@"; }
fi

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
        echo "WARNING: ${mutex} is owned by uid ${owner%% *} with mode ${owner##* };" >&2
        echo "         mvnc will abort with 'global mutex initialization failed'." >&2
        echo "         fix: rm ${mutex} (or run everything as the same user)," >&2
        echo "         or make it world readable: chmod 0666 ${mutex}" >&2
    fi
fi

MODE="${1:-list}"; shift || true

case "${MODE}" in
    list)
        ov_run "${OV}/bin/hello_myriad" || {
            echo "hint: host-side stick state -> $(lsusb | grep -i 03e7 || echo 'no 03e7 device visible')" >&2
            echo "hint: node permissions      -> stat -c '%A %U:%G %n' /dev/bus/usb/*/*  (logs/HOST-RUN.md)" >&2
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
                --blob)  blob="$2";  shift 2 ;;
                *) echo "unknown option: $1" >&2; exit 2 ;;
            esac
        done
        [[ -n "${model}" && -n "${blob}" ]] || { echo "compile needs --model <xml> --blob <path>" >&2; exit 2; }
        mkdir -p "$(dirname "${blob}")"
        ov_run "${LIBDIR}/compile_tool" \
            -m "${model}" -d MYRIAD -o "${blob}"
        ls -l "${blob}"
        ;;
    shell)
        echo "host-native OpenVINO 2020.3 shell (${TARGET}): hello_myriad, mobilenet_classify,"
        echo "compile_tool reachable via PATH.  LD_LIBRARY_PATH is exported too."
        LD_LIBRARY_PATH="${LP}" PATH="${BIN}:${PATH}" PS1='hostov# ' exec bash
        ;;
    *)
        sed -n '2,24p' "$0"; exit 2 ;;
esac
