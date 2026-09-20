#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# prepare-deps.sh - materialise OpenVINO 2020.3.2's "FTP" dependencies locally
#
# OpenVINO 2020.3 downloads its prebuilt dependencies from
#   https://download.01.org/opencv/2020/openvinotoolkit/2020.3/inference_engine/
# which no longer serves any file (404).  The build system has a supported
# escape hatch: if the environment variable IE_PATH_TO_DEPS is defined, every
# dependency URL becomes "$IE_PATH_TO_DEPS/<archive name>" and, because that is
# not an http/https/ftp URL, cmake/download/download_and_check.cmake simply does
# file(COPY) from the local directory.  So we only have to reproduce the exact
# archive names + internal layout that inference-engine/cmake/vpu_dependencies.cmake
# expects:
#
#   firmware_<device>_<FIRMWARE_PACKAGE_VERSION>.zip
#       mvnc/<device>.mvcmd
#
# FIRMWARE_PACKAGE_VERSION is read from the source tree so this stays in sync
# with the pinned OpenVINO revision.  The firmware payloads themselves come from
# the official 2020.3.2 raspbian runtime package (vendor/reference-runtime).
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/vendor/openvino-2020.3.2"
FW_SRC="${ROOT}/vendor/firmware"
DEPS="${ROOT}/vendor/deps"

[[ -d "${SRC}" ]] || { echo "missing OpenVINO source tree: ${SRC}" >&2; exit 1; }
[[ -d "${FW_SRC}" ]] || { echo "missing ${FW_SRC} (see scripts/fetch-runtime.sh)" >&2; exit 1; }

VER="$(sed -n 's/^set(FIRMWARE_PACKAGE_VERSION  *\([0-9][0-9]*\)).*$/\1/p' "${SRC}/inference-engine/cmake/vpu_dependencies.cmake")"
DEVICES="$(sed -n 's/^set(VPU_SUPPORTED_FIRMWARES  *\(.*\)).*$/\1/p' "${SRC}/inference-engine/cmake/vpu_dependencies.cmake")"
echo "   firmware package version: ${VER:-<none>}"
echo "   supported firmwares     : ${DEVICES:-<none>}"

if [[ -z "${VER}" || -z "${DEVICES}" ]]; then
    echo "could not read VPU_SUPPORTED_FIRMWARES / FIRMWARE_PACKAGE_VERSION from ${SRC}" >&2
    exit 1
fi

mkdir -p "${DEPS}"
python3 - "$DEPS" "$FW_SRC" "$VER" $DEVICES <<'PY'
import pathlib, sys, zipfile
deps, fw_src, ver, devices = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
for dev in devices:
    fw = pathlib.Path(fw_src) / f"{dev}.mvcmd"
    if not fw.is_file():
        raise SystemExit(f"missing firmware payload: {fw}")
    out = pathlib.Path(deps) / f"firmware_{dev}_{ver}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(fw, arcname=f"mvnc/{dev}.mvcmd")
    print(f"wrote {out.name} (mvnc/{dev}.mvcmd, {fw.stat().st_size} B)")
PY

echo
echo "local dependency mirror (${DEPS}):"
ls -l "${DEPS}"
