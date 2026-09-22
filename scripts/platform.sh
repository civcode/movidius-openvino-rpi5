#!/usr/bin/env bash
# Shared platform selection for build/run/host tooling.
# Canonical project targets:
#   armv7  -> linux/arm/v7  (legacy/known-good Pi ARMHF userspace)
#   arm64  -> linux/arm64   (native AArch64; preferred Raspberry Pi 5 path)
#   amd64  -> linux/amd64   (native x86_64 Linux)

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
            printf '%s
' "${requested}"
            return 0
        fi
        echo "invalid OV_PLATFORM='${requested}'; expected armv7, arm64 or amd64" >&2
        return 2
    fi
    if [[ -n "${TARGET:-}" ]] && platform_normalize "${TARGET}" >/dev/null 2>&1; then
        printf '%s
' "${TARGET}"
    else
        printf '%s
' auto
    fi
}

platform_load() {
    local requested="${1:-auto}"
    if [[ -z "${requested}" || "${requested}" == auto ]]; then
        TARGET="$(platform_detect)"
    else
        TARGET="$(platform_normalize "${requested}")" || {
            echo "unsupported platform '${requested}'; expected armv7, arm64 or amd64" >&2
            return 2
        }
    fi

    case "${TARGET}" in
        armv7)
            DOCKER_PLATFORM="linux/arm/v7"
            DEFAULT_BASE_IMAGE="arm32v7/debian:bullseye"
            DEFAULT_IMAGE="openvino-2020.3-movidius-armv7:latest"
            USE_CMAKE_TOOLCHAIN=1
            CMAKE_TOOLCHAIN_FILE="/work/toolchain-ro/armv7-native.toolchain.cmake"
            EXPECTED_ELF_CLASS="ELF32"
            EXPECTED_ELF_MACHINE_REGEX='ARM'
            EXPECTED_ELF_MACHINE_ID=40
            HOST_RUNTIME_KIND="armhf-sysroot"
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
            ;;
    esac

    export TARGET DOCKER_PLATFORM DEFAULT_BASE_IMAGE DEFAULT_IMAGE
    export USE_CMAKE_TOOLCHAIN CMAKE_TOOLCHAIN_FILE
    export EXPECTED_ELF_CLASS EXPECTED_ELF_MACHINE_REGEX EXPECTED_ELF_MACHINE_ID HOST_RUNTIME_KIND
}

platform_print() {
    cat <<EOF_PRINT
target=${TARGET}
docker_platform=${DOCKER_PLATFORM}
base_image=${BASE_IMAGE:-${DEFAULT_BASE_IMAGE}}
image=${IMAGE:-${DEFAULT_IMAGE}}
use_cmake_toolchain=${USE_CMAKE_TOOLCHAIN}
cmake_toolchain_file=${CMAKE_TOOLCHAIN_FILE:-<none>}
expected_elf_class=${EXPECTED_ELF_CLASS}
expected_elf_machine_id=${EXPECTED_ELF_MACHINE_ID}
host_runtime_kind=${HOST_RUNTIME_KIND}
EOF_PRINT
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
