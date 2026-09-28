# ---------------------------------------------------------------------------
# hello_myriad: smoke test for the self-built OpenVINO 2020.3.2 runtime.
#
# 1. enumerate the devices the Inference Engine reports  -> must contain MYRIAD
# 2. report the plugin version
# 3. optionally load an FP16 IR and run inference on the Movidius stick
#
# Built against the install tree produced by the builder stage. The Docker build
# supplies the ARM toolchain only for the armv7 target; arm64 and amd64 build natively.
# The OV link setup is shared in cmake/ov203-link.cmake.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)
project(hello_myriad CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# shared OV link setup + shared-header include dir (self-locating probe:
# <root>/cmake in a checkout, /work/cmake in the Docker build)
foreach(_cand "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake"
              "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
    if(EXISTS "${_cand}/ov203-link.cmake")
        include("${_cand}/ov203-link.cmake")
        break()
    endif()
endforeach()

ov203_add_inference_executable(hello_myriad main.cpp)

message(STATUS "hello_myriad: linking ${ov_link_libs}")
message(STATUS "hello_myriad: rpath ${ov_rpath_use}")
