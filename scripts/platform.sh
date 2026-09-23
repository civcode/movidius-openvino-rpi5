#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# platform.sh - canonical platform mapping for the multi-platform build.
#
# Project targets (the only valid values of TARGET):
#   armv7  legacy Raspberry Pi ARMHF userspace (known-good fallback)
#   arm64  native AArch64 (preferred Raspberry Pi 5 path)
#   amd64  native x86_64 Linux
#
# Sourced by build.sh, the docs/diagnostics scripts, and (indirectly) the
# Dockerfile build args.  Exposes:
#
#   platform_normalize <req>     canonicalise an alias (armhf, x86_64,
#                                linux/amd64, ...) to armv7|arm64|amd64
#   platform_detect              echo the target matching this host CPU
#   platform_default_request     echo auto-detected / env-selected request
#   platform_load <request>      populate TARGET and all derived settings
#   platform_print               dump the resolved configuration
#   platform_lib_dir             print lib/<arch> dir for the loaded target
#   platform_find_ie_libdir R    find the single installed IE lib dir under R
#   platform_read_manifest_target M  print TARGET= of a runtime manifest
#
# Environment overrides:
#   OV_PLATFORM=armv7|arm64|amd64   preferred explicit selection
#   TARGET=<valid target>           legacy, accepted only when it normalizes
# Anything else falls back to host-CPU auto-detection.
# ---------------------------------------------------------------------------

# Must be safe to source (no set -e here; callers may already use it).

platform_normalize() {
    case "${1:-}" in
        armv7|armhf|arm32|linux/arm/v7) printf '%s\n' armv7 ;;
        arm64|aarch64|arm64v8|linux/arm64|linux/arm64/v8) printf '%s\n' arm64 ;;
        amd64|x86_64|x64|linux/amd64)   printf '%s\n' amd64 ;;
        *) return 1 ;;
    esac
}

platform_detect() {
    case "$(uname -m)" in
        x86_64|amd64) printf '%s\n' amd64 ;;
        aarch64|arm64) printf '%s\n' arm64 ;;
        armv7l|armv7*) printf '%s\n' armv7 ;;
        *)
            echo "unsupported host architecture: $(uname -m); use --platform armv7|arm64|amd64 explicitly" >&2
            return 1
            ;;
    esac
}

# Resolve an optional environment-selected project platform without trusting a
# generic TARGET variable blindly. Some CI/container environments define TARGET
# for unrelated purposes. OV_PLATFORM is the preferred explicit override; a
# legacy TARGET value is accepted only when it normalizes to a supported target.
platform_default_request() {
    local requested="${OV_PLATFORM:-}"
    if [[ -n "${requested}" ]]; then
        if platform_normalize "${requested}" >/dev/null 2>&1; then
            printf '%s\n' "${requested}"
            return 0
        fi
        echo "invalid OV_PLATFORM='${requested}'; expected armv7, arm64 or amd64" >&2
        return 2
    fi
    if [[ -n "${TARGET:-}" ]] && platform_normalize "${TARGET}" >/dev/null 2>&1; then
        printf '%s\n' "${TARGET}"
    else
        printf '%s\n' auto
    fi
}

platform_load() {
    local requested="${1:-auto}"
    if [[ -z "${requested}" || "${requested}" == "auto" ]]; then
        TARGET="$(platform_detect)" || return 1
    else
        TARGET="$(platform_normalize "${requested}")" || {
            echo "unsupported platform '${requested}'; expected armv7, arm64 or amd64" >&2
            return 2
        }
    fi

    case "${TARGET}" in
        armv7)
            DOCKER_PLATFORM="linux/arm/v7"
            # arm32v7/debian is the known-good ARMHF userspace the original
            # single-target build validated against.
            DEFAULT_BASE_IMAGE="arm32v7/debian:bullseye"
            DEFAULT_IMAGE="openvino-2020.3-movidius-armv7:latest"
            USE_CMAKE_TOOLCHAIN=1
            CMAKE_TOOLCHAIN_FILE="/work/toolchain-ro/armv7-native.toolchain.cmake"
            EXPECTED_ELF_CLASS="ELF32"
            EXPECTED_ELF_MACHINE_REGEX='ARM'
            EXPECTED_ELF_MACHINE_ID=40
            HOST_RUNTIME_KIND="armhf-sysroot"
            TARGET_LIB_DIR="lib/armv7l"
            ;;
        arm64)
            DOCKER_PLATFORM="linux/arm64"
            DEFAULT_BASE_IMAGE="arm64v8/debian:bullseye"
            DEFAULT_IMAGE="openvino-2020.3-movidius-arm64:latest"
            USE_CMAKE_TOOLCHAIN=0
            CMAKE_TOOLCHAIN_FILE=""
            EXPECTED_ELF_CLASS="ELF64"
            EXPECTED_ELF_MACHINE_REGEX='AArch64|ARM aarch64'
            EXPECTED_ELF_MACHINE_ID=183
            HOST_RUNTIME_KIND="native"
            TARGET_LIB_DIR="lib/aarch64"
            ;;
        amd64)
            DOCKER_PLATFORM="linux/amd64"
            DEFAULT_BASE_IMAGE="debian:bullseye"
            DEFAULT_IMAGE="openvino-2020.3-movidius-amd64:latest"
            USE_CMAKE_TOOLCHAIN=0
            CMAKE_TOOLCHAIN_FILE=""
            EXPECTED_ELF_CLASS="ELF64"
            EXPECTED_ELF_MACHINE_REGEX='Advanced Micro Devices X86-64|X86-64|x86-64'
            EXPECTED_ELF_MACHINE_ID=62
            HOST_RUNTIME_KIND="native"
            TARGET_LIB_DIR="lib/intel64"
            ;;
    esac

    export TARGET DOCKER_PLATFORM DEFAULT_BASE_IMAGE DEFAULT_IMAGE
    export USE_CMAKE_TOOLCHAIN CMAKE_TOOLCHAIN_FILE
    export EXPECTED_ELF_CLASS EXPECTED_ELF_MACHINE_REGEX EXPECTED_ELF_MACHINE_ID
    export HOST_RUNTIME_KIND TARGET_LIB_DIR
    return 0
}

platform_print() {
    printf 'target=%s\n'                 "${TARGET}"
    printf 'docker_platform=%s\n'        "${DOCKER_PLATFORM}"
    printf 'base_image=%s\n'             "${DEFAULT_BASE_IMAGE}"
    printf 'image=%s\n'                  "${DEFAULT_IMAGE}"
    printf 'use_cmake_toolchain=%s\n'    "${USE_CMAKE_TOOLCHAIN}"
    printf 'cmake_toolchain_file=%s\n'   "${CMAKE_TOOLCHAIN_FILE:-<none>}"
    printf 'expected_elf_class=%s\n'     "${EXPECTED_ELF_CLASS}"
    printf 'expected_elf_machine_regex=%s\n' "${EXPECTED_ELF_MACHINE_REGEX}"
    printf 'expected_elf_machine_id=%s\n'   "${EXPECTED_ELF_MACHINE_ID}"
    printf 'host_runtime_kind=%s\n'      "${HOST_RUNTIME_KIND}"
    printf 'lib_dir=%s\n'                "${TARGET_LIB_DIR}"
}

# Print the installed library directory (e.g. lib/aarch64) for the loaded
# target; the examples use this instead of hard-coding one architecture.
platform_lib_dir() {
    printf '%s\n' "${TARGET_LIB_DIR}"
}

# Discover the one installed Inference Engine architecture directory by finding
# libinference_engine.so. This intentionally does not assume armv7l/aarch64/intel64.
platform_find_ie_libdir() {
    local root="$1"
    local -a matches=()
    while IFS= read -r p; do
        matches+=("$(dirname "$p")")
    done < <(find "${root}/lib" -mindepth 2 -maxdepth 2 -name 'libinference_engine.so' -print 2>/dev/null | sort -u)

    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "expected exactly one OpenVINO library directory under ${root}/lib; found ${#matches[@]}" >&2
        printf '  %s\n' "${matches[@]:-}" >&2
        return 1
    fi
    local libdir="${matches[0]}"
    [[ -f "${libdir}/libmyriadPlugin.so" ]] || { echo "MYRIAD plugin missing from ${libdir}" >&2; return 1; }
    [[ -f "${libdir}/plugins.xml" ]] || { echo "plugins.xml missing from ${libdir}" >&2; return 1; }
    [[ -f "${libdir}/usb-ma2450.mvcmd" ]] || { echo "MA2450 firmware missing from ${libdir}" >&2; return 1; }
    printf '%s\n' "${libdir}"
}

platform_read_manifest_target() {
    local manifest="$1"
    [[ -f "${manifest}" ]] || return 1
    sed -n 's/^TARGET=//p' "${manifest}" | head -1
}
