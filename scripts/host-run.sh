#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/host-run.sh - run the extracted OpenVINO 2020.3.2 runtime directly on
# the aarch64 host: no Docker, no container, no QEMU at inference time.
#
#   ./scripts/host-run.sh list                    device enumeration
#   ./scripts/host-run.sh demo                    tiny model on the stick
#   ./scripts/host-run.sh mobilenet               numerical self-test on the stick
#   ./scripts/host-run.sh mobilenet --image vendor/models/images/banana.ppm --topk 3
#   ./scripts/host-run.sh compile --model vendor/models/mobilenet-v2-ov203/fp16/mobilenet-v2-ov203.xml \
#                                 --blob  vendor/models/blobs/mobilenet-v2.blob
#   ./scripts/host-run.sh shell                   bash with ov_* wrappers on PATH
#
# Everything is a host path here (no /models mount): the binaries are the arm32v7
# ones from the image, executed by the armhf loader extracted with
# ./scripts/pull-runtime.sh.  See logs/HOST-RUN.md for the two host-side
# requirements (USB node permissions, /tmp/mvnc.mutex ownership).
# ---------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="${ROOT}/work/host-runtime"
SYSROOT="${RT}/sysroot"
OV="${RT}/openvino"
LD="${SYSROOT}/lib/ld-linux-armhf.so.3"
LP="${SYSROOT}/lib/arm-linux-gnueabihf:${SYSROOT}/usr/lib/arm-linux-gnueabihf:${OV}/inference_engine/lib/armv7l:${OV}/ngraph/lib"
BIN="${RT}/bin"
IR="${IR:-fp16}"
MODELS="${ROOT}/vendor/models"

if [[ ! -x "${LD}" || ! -d "${OV}" ]]; then
    echo "work/host-runtime is not populated yet - running ./scripts/pull-runtime.sh" >&2
    "${ROOT}/scripts/pull-runtime.sh"
fi

# armhf glibc resolves a SONAME only from a directory it can search; the PT_INTERP
# /lib/ld-linux-armhf.so.3 does not exist on the host, so every binary needs the
# loader wrapper below.  They are cheap and regenerated on every call.
mkdir -p "${BIN}"
for b in "${OV}/bin/"* "${OV}/inference_engine/lib/armv7l/compile_tool"; do
    [[ -x "${b}" && ! -d "${b}" ]] || continue
    name="$(basename "${b}")"
    cat > "${BIN}/${name}" <<EOF
#!/usr/bin/env bash
exec "${LD}" --library-path "${LP}" "${b}" "\$@"
EOF
    chmod +x "${BIN}/${name}"
done
ov_run() { "${LD}" --library-path "${LP}" "$@"; }

# mvnc's global lock is a fixed path, open()ed with O_CREAT by whoever gets there
# first.  With fs.protected_regular != 0 (Debian: 2) a file created by another
# user in world-writable /tmp is not openable - even by root - and mvnc aborts
# with "global mutex initialization failed".  Warn instead of exiting(1).
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
        ov_run "${OV}/inference_engine/lib/armv7l/compile_tool" \
            -m "${model}" -d MYRIAD -o "${blob}"
        ls -l "${blob}"
        ;;
    shell)
        echo "host-native OpenVINO 2020.3 shell: ov wrappers on PATH (hello_myriad,"
        echo "mobilenet_classify, compile_tool).  LD_LIBRARY_PATH is exported too."
        LD_LIBRARY_PATH="${LP}" PATH="${BIN}:${PATH}" PS1='hostov# ' exec bash
        ;;
    *)
        sed -n '2,22p' "$0"; exit 2 ;;
esac
