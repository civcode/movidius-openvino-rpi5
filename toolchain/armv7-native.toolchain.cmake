# ---------------------------------------------------------------------------
# armv7-native.toolchain.cmake
#
# OpenVINO 2020.3.2 toolchain descriptor for a build that runs an *armhf*
# (32-bit ARMv7) userspace inside an arm/v7 container on the aarch64
# Raspberry Pi 5 kernel (CONFIG_COMPAT=y, so the code runs natively, no QEMU).
#
# Why a toolchain file is needed at all:
#   * CMake detects the machine with `uname -m`, which reports the *kernel*
#     architecture ("aarch64") even inside an arm32v7 container.
#   * OpenVINO keys several decisions off CMAKE_SYSTEM_PROCESSOR
#     (cmake/target_flags.cmake -> ARM vs AARCH64 flags, ARCH/ARCH_FOLDER used
#     for the install directory name, THREADING default, ...).  Upstream
#     OpenVINO uses "armv7l" for this target (see cmake/arm.toolchain.cmake),
#     and the official raspbian runtime package installs into
#     deployment_tools/inference_engine/lib/armv7l.
#   * Declaring CMAKE_SYSTEM_NAME/PROCESSOR makes CMake treat the build as a
#     cross build, which is also what disables LTO/ITT and selects SEQ
#     threading - i.e. exactly the conservative configuration we want.
#
# Unlike upstream cmake/arm.toolchain.cmake this file deliberately does NOT
# name cross compilers: the container's own gcc/g++ already target
# armv7-a / hard-float (Debian armhf), so they *are* the cross compilers of
# the aarch64 host.  Side effect: CMAKE_FIND_ROOT_PATH must be re-rooted at "/"
# so that find_library()/find_path() still see the container's system
# libraries (e.g. libusb-1.0).
# ---------------------------------------------------------------------------

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)

# Native Debian armhf compilers (already -march=armv7-a -mfpu=neon-vfpv4
# -mfloat-abi=hard); no arm-linux-gnueabihf-* prefix is required.
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

# Re-root the system search paths: with *_MODE_* ONLY and no root path a
# cross-building CMake would refuse to find the container's own libusb.
set(CMAKE_FIND_ROOT_PATH /)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

macro(__cmake_find_root_save_and_reset)
    foreach(v
            CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
            CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
            CMAKE_FIND_ROOT_PATH_MODE_PACKAGE
            CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
            )
        set(__save_${v} ${${v}})
        set(${v} NEVER)
    endforeach()
endmacro()

macro(__cmake_find_root_restore)
    foreach(v
            CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
            CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
            CMAKE_FIND_ROOT_PATH_MODE_PACKAGE
            CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
            )
        set(${v} ${__save_${v}})
        unset(__save_${v})
    endforeach()
endmacro()

# macros to find programs/packages on the host OS (copied from upstream
# cmake/arm.toolchain.cmake; OpenVINO uses them for host-only build tools)
macro(find_host_program)
    __cmake_find_root_save_and_reset()
    if(CMAKE_HOST_WIN32)
        set(WIN32 1)
        set(UNIX)
    elseif(CMAKE_HOST_APPLE)
        set(APPLE 1)
        set(UNIX)
    endif()
    find_program(${ARGN})
    set(WIN32)
    set(APPLE)
    set(UNIX 1)
    __cmake_find_root_restore()
endmacro()

macro(find_host_package)
    __cmake_find_root_save_and_reset()
    if(CMAKE_HOST_WIN32)
        set(WIN32 1)
        set(UNIX)
    elseif(CMAKE_HOST_APPLE)
        set(APPLE 1)
        set(UNIX)
    endif()
    find_package(${ARGN})
    set(WIN32)
    set(APPLE)
    set(UNIX 1)
    __cmake_find_root_restore()
endmacro()
