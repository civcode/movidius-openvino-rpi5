#!/usr/bin/env bash
# Live regression matrix for the three examples.
#
#   examples : webcam (mobilenet), ssd (ssdlite), seg (deeplab)
#   backend  : docker, host
#   device   : MYRIAD, CPU
#   mode     : headless, gui
#
# 24 cells in total.  Each cell ends PASS, SKIP (reason) or FAIL; the
# matrix as a whole succeeds when no cell FAILs.  A cell is SKIP when a
# prerequisite is missing (no docker, no DISPLAY/xdotool, host runtime
# absent, ...) - the skip is detected from the launcher's own diagnostics
# so the same script works on hosts where more cells are feasible.
#
# GUI cells open the example window, wait for it to appear, close it with
# xdotool and require the "bye" line within the cell timeout (this covers
# the whole close path: window watcher, client shutdown, server stop).
#
# MYRIAD cells can take ~20 s each (stick boot).  When any MYRIAD cell
# ran, the stick is returned to ROM mode at the end (best effort).
#
# Usage:
#   scripts/test-examples.sh [--no-gui] [--only webcam,ssd,seg]
#                            [--backend docker,host] [--device MYRIAD,CPU]
#                            [--timeout 300]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

VENV_PY="examples/webcam/venv/bin/python"
[ -x "${VENV_PY}" ] || VENV_PY="python3"
VIDEO="examples/webcam/sample_640x360.mp4"

TIMEOUT=300
ONLY="webcam,ssd,seg"
BACKENDS="docker,host"
DEVICES="MYRIAD,CPU"
MODES="headless,gui"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-gui)    MODES="headless" ;;
        --only)      ONLY="${2:?}"; shift ;;
        --backend)   BACKENDS="${2:?}"; shift ;;
        --device)    DEVICES="${2:?}"; shift ;;
        --timeout)   TIMEOUT="${2:?}"; shift ;;
        *) echo "unknown option: $1 (see header)" >&2; exit 2 ;;
    esac
    shift
done

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
LOG="${WORK}/matrix.log"

# ------------------------------------------------------------------ prereqs
HAVE_DOCKER=0
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    HAVE_DOCKER=1
fi

HAVE_GUI=0
if [[ -n "${DISPLAY:-}" ]] && command -v xdotool >/dev/null 2>&1; then
    HAVE_GUI=1
fi

# Test image for ssd/seg: the first frame of the sample video, as PPM.
IMG="${WORK}/test.ppm"
HAVE_IMG=0
if [[ -f "${VIDEO}" ]] && "${VENV_PY}" - "${VIDEO}" "${IMG}" >/dev/null 2>&1 <<'PYEOF'
import sys
import cv2
src, dst = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(src)
ok, frame = cap.read()
cap.release()
if ok:
    cv2.imwrite(dst, frame)
PYEOF
then
    if [[ -s "${IMG}" ]]; then
        HAVE_IMG=1
    fi
fi

result=()   # "name|STATUS|detail"

record() {   # record name status detail
    result+=("$1|$2|$3")
    printf '%-38s %-6s %s\n' "$1" "$2" "$3"
}

in_list() {  # in_list VALUE LIST(comma)
    local v
    for v in ${2//,/ }; do
        [[ "$v" == "$1" ]] && return 0
    done
    return 1
}

# Skip markers: launcher/client diagnostics that mean "not feasible here",
# not a regression.
SKIP_RE='host runtime missing|cannot run on host CPU|tensorflow is not installed|onnxruntime is not installed|python3 not found|MYRIAD.*(not available|not found)|No devices found|File was not found|Error loading (XML|BIN) file'

client_for() {   # client_for example -> sets CLIENT, TITLE, FRAME_ARGS
    case "$1" in
        webcam)
            CLIENT="examples/webcam/webcam_mobilenet.py"
            TITLE="webcam mobilenet"
            ;;
        ssd)
            CLIENT="examples/ssd-detect/ssd_stream.py"
            TITLE="ssdlite"
            ;;
        seg)
            CLIENT="examples/deeplab-seg/seg_stream.py"
            TITLE="deeplab"
            ;;
    esac
}

window_wait() {   # window_wait title seconds client_pid -> 0 if the window appeared
    local deadline=$(( $(date +%s) + $2 ))
    while [[ "$(date +%s)" -lt "$deadline" ]]; do
        # a dead client will never show a window - don't burn the timeout
        if [[ -n "$3" ]] && ! kill -0 "$3" 2>/dev/null; then
            return 1
        fi
        if DISPLAY="${DISPLAY}" xdotool search --name "$1" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

run_cell() {   # run_cell example backend device mode
    local ex="$1" backend="$2" device="$3" mode="$4"
    local name="${ex}-${backend}-${device}-${mode}"
    local celllog="${WORK}/${ex}_${backend}_${device}_${mode}.log"

    client_for "${ex}"
    local frame_args=()
    case "${ex}" in
        webcam) frame_args=(--video "${VIDEO}") ;;
        ssd|seg) frame_args=(--file "${IMG}" --frames 1) ;;
    esac
    if [[ "${mode}" == headless ]]; then
        frame_args+=(--headless)
    fi

    # feasibility pre-checks (cheap, no launch)
    if [[ "${backend}" == docker && "${HAVE_DOCKER}" -eq 0 ]]; then
        record "${name}" SKIP "no docker"; return 0
    fi
    if [[ "${mode}" == gui && "${HAVE_GUI}" -eq 0 ]]; then
        record "${name}" SKIP "no DISPLAY/xdotool"; return 0
    fi
    if [[ "${ex}" != webcam && "${HAVE_IMG}" -eq 0 ]]; then
        record "${name}" SKIP "no test image (cv2 or sample video missing)"; return 0
    fi

    local rc
    if [[ "${mode}" == gui ]]; then
        # launch detached, then drive the window from here
        DISPLAY="${DISPLAY}" nohup "${VENV_PY}" "${CLIENT}" \
            --backend "${backend}" --device "${device}" "${frame_args[@]}" \
            >"${celllog}" 2>&1 < /dev/null &
        local cpid=$!
        local wdeadline=$(( $(date +%s) + TIMEOUT ))
        local t0
        if window_wait "${TITLE}" "$(( TIMEOUT - 15 ))" "${cpid}"; then
            t0=$(date +%s)
            DISPLAY="${DISPLAY}" xdotool search --name "${TITLE}" windowclose
            local ok=0
            while [[ "$(date +%s)" -lt "${wdeadline}" ]]; do
                if grep -q "^bye" "${celllog}" 2>/dev/null; then ok=1; break; fi
                sleep 0.3
            done
            if [[ "${ok}" -eq 1 ]]; then
                local dt=$(( $(date +%s) - t0 ))
                record "${name}" PASS "bye after window close (+${dt}s)"
            elif grep -qE "${SKIP_RE}" "${celllog}" 2>/dev/null; then
                record "${name}" SKIP "$(grep -m1 -oE "${SKIP_RE}" "${celllog}")"
            else
                record "${name}" FAIL "no bye within ${TIMEOUT}s (see log)"
            fi
        else
            # no window in time: skip if the log says infeasible, else fail
            if grep -qE "${SKIP_RE}" "${celllog}" 2>/dev/null; then
                record "${name}" SKIP "$(grep -m1 -oE "${SKIP_RE}" "${celllog}")"
            else
                record "${name}" FAIL "no window within ${TIMEOUT}s (see log)"
            fi
        fi
        wait "${cpid}" 2>/dev/null
    else
        timeout "${TIMEOUT}" "${VENV_PY}" "${CLIENT}" \
            --backend "${backend}" --device "${device}" "${frame_args[@]}" \
            >"${celllog}" 2>&1
        rc=$?
        if [[ ${rc} -eq 0 ]] && grep -q "^bye" "${celllog}" 2>/dev/null; then
            record "${name}" PASS "clean bye"
        elif grep -qE "${SKIP_RE}" "${celllog}" 2>/dev/null; then
            record "${name}" SKIP "$(grep -m1 -oE "${SKIP_RE}" "${celllog}")"
        else
            local tail
            tail="$(tail -1 "${celllog}" 2>/dev/null)"
            record "${name}" FAIL "rc=${rc}: ${tail}"
        fi
    fi
}

# ------------------------------------------------------------------ run
echo "example matrix: examples={${ONLY}} backends={${BACKENDS}} devices={${DEVICES}} modes={${MODES}} (timeout ${TIMEOUT}s)"
echo "prereqs: docker=${HAVE_DOCKER} gui=${HAVE_GUI} test-image=${HAVE_IMG} python=${VENV_PY}"
echo

MYRIAD_RAN=0
for ex in ${ONLY//,/ }; do
    for backend in ${BACKENDS//,/ }; do
        for device in ${DEVICES//,/ }; do
            for mode in ${MODES//,/ }; do
                run_cell "${ex}" "${backend}" "${device}" "${mode}"
                [[ "${device}" == MYRIAD ]] && MYRIAD_RAN=1
            done
        done
    done
done
echo

# MYRIAD stick back to ROM mode (best effort)
if [[ "${MYRIAD_RAN}" -eq 1 ]] && command -v python3 >/dev/null 2>&1; then
    if sudo -n python3 scripts/reset-stick.py >/dev/null 2>&1; then
        echo "note: MYRIAD stick returned to ROM mode"
    else
        echo "note: MYRIAD stick NOT reset (needs sudo or stick absent)"
    fi
fi

# ------------------------------------------------------------------ summary
fails=0
skips=0
for r in "${result[@]}"; do
    status="${r#*|}"; status="${status%%|*}"
    [[ "${status}" == FAIL ]] && fails=$((fails + 1))
    [[ "${status}" == SKIP ]] && skips=$((skips + 1))
done
echo "summary: ${#result[@]} cells, ${skips} skipped, ${fails} failed"
if [[ ${fails} -gt 0 ]]; then
    echo "MATRIX_RESULT=FAIL"
    exit 1
fi
echo "MATRIX_RESULT=PASS"
