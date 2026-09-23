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
#   platform_default_request   echo auto-detected request
#   platform_load <request>    populate TARGET and all derived settings
#   platform_print             dump the resolved configuration
#   platform_lib_dir           print lib/<arch> dir for the loaded target
#
# Environment overrides:
#   OV_PLATFORM=armv7|arm64|amd64   preferred explicit selection
#   TARGET=<valid target>           legacy, still accepted
# Anything else falls back to host-CPU auto-detection.
# ---------------------------------------------------------------------------

# Must be safe to source (no set -e here; callers may already use it).

# Valid project targets, in display order.
_platform_is_valid() {
    case "$1" in
        armv7|arm64|amd64) return 0 ;;
        *) return 1 ;;
    esac
}

platform_default_request() {
    local request=""
    # Explicit environment selection wins.
    if _platform_is_valid "${OV_PLATFORM:-}"; then
        request="${OV_PLATFORM}"
    elif _platform_is_valid "${TARGET:-}"; then
        # Legacy: an explicit TARGET that names a project target is honored.
        # Unrelated TARGET values (e.g. from CI containers) are ignored so
        # auto-detection stays reliable.
        request="${TARGET}"
    fi
    if [[ -n "${request}" ]]; then
        printf '%s\n' "${request}"
        return 0
    fi
    local cpu
    cpu="$(uname -m)"
    case "${cpu}" in
        x86_64|amd64)          printf 'amd64\n' ;;
        aarch64|arm64)         printf 'arm64\n' ;;
        armv7l|armv7|arm)      printf 'armv7\n' ;;
        *)
            echo "cannot map host CPU '${cpu}' to a project target; use --platform armv7|arm64|amd64" >&2
            return 1
            ;;
    esac
}

platform_load() {
    local request="${1:-auto}"
    if [[ "${request}" == "auto" || -z "${request}" ]]; then
        request="$(platform_default_request)" || return 1
    fi

    case "${request}" in
        armv7)
            TARGET="armv7"
            DOCKER_PLATFORM="linux/arm/v7"
            # arm32v7/debian is the known-good ARMHF userspace the original
            # single-target build validated against.
            DEFAULT_BASE_IMAGE="arm32v7/debian:bullseye"
            DEFAULT_IMAGE="openvino-2020.3-movidius-armv7:latest"
            USE_CMAKE_TOOLCHAIN=1
            EXPECTED_ELF_CLASS="ELF32"
            EXPECTED_ELF_MACHINE_REGEX="ARM"
            EXPECTED_ELF_MACHINE_ID=40
            HOST_RUNTIME_KIND="armhf-sysroot"
            TARGET_LIB_DIR="lib/armv7l"
            ;;
        arm64)
            TARGET="arm64"
            DOCKER_PLATFORM="linux/arm64"
            DEFAULT_BASE_IMAGE="debian:bullseye"
            DEFAULT_IMAGE="openvino-2020.3-movidius-arm64:latest"
            USE_CMAKE_TOOLCHAIN=0
            EXPECTED_ELF_CLASS="ELF64"
            EXPECTED_ELF_MACHINE_REGEX="AArch64"
            EXPECTED_ELF_MACHINE_ID=183
            HOST_RUNTIME_KIND="native"
            TARGET_LIB_DIR="lib/aarch64"
            ;;
        amd64)
            TARGET="amd64"
            DOCKER_PLATFORM="linux/amd64"
            DEFAULT_BASE_IMAGE="debian:bullseye"
            DEFAULT_IMAGE="openvino-2020.3-movidius-amd64:latest"
            USE_CMAKE_TOOLCHAIN=0
            EXPECTED_ELF_CLASS="ELF64"
            EXPECTED_ELF_MACHINE_REGEX="X86-64"
            EXPECTED_ELF_MACHINE_ID=62
            HOST_RUNTIME_KIND="native"
            TARGET_LIB_DIR="lib/intel64"
            ;;
        *)
            echo "unknown platform '${request}': expected armv7, arm64 or amd64" >&2
            return 1
            ;;
    esac
    return 0
}

platform_print() {
    printf 'target=%s\n'                 "${TARGET}"
    printf 'docker_platform=%s\n'        "${DOCKER_PLATFORM}"
    printf 'base_image=%s\n'             "${DEFAULT_BASE_IMAGE}"
    printf 'image=%s\n'                  "${DEFAULT_IMAGE}"
    printf 'use_cmake_toolchain=%s\n'    "${USE_CMAKE_TOOLCHAIN}"
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
