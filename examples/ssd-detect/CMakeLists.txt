# ---------------------------------------------------------------------------
# ssd_detect: SSDLite-MobileNetV2 object detection demo for the self-built
# OpenVINO 2020.3.2 runtime (examples/ssd-detect/ssd_detect.cpp, README.md).
#
# The OV link setup (library resolution, link group, rpath) is shared in
# cmake/ov203-link.cmake.  A second target, ssd_test, exercises the
# post-processing logic without any hardware and is run in the Docker build
# stage before the runtime image is built.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)
project(ssd_detect CXX)

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

ov203_add_inference_executable(ssd_detect ssd_detect.cpp)

# device-free post-processing unit test (runs in the build stage)
add_executable(ssd_test ssd_test.cpp)
target_include_directories(ssd_test PRIVATE "${SHARED_INCLUDE_DIR}")

message(STATUS "ssd_detect: linking ${ov_link_libs}")
message(STATUS "ssd_detect: rpath ${ov_rpath_use}")
