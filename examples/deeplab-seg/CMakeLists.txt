# ---------------------------------------------------------------------------
# seg_detect: DeepLabV3 (Pascal VOC) segmentation demo for the self-built
# OpenVINO 2020.3.2 runtime (examples/deeplab-seg/seg_detect.cpp, README.md).
#
# The OV link setup is shared in cmake/ov203-link.cmake.  A second target,
# seg_test, exercises the post-processing logic without any hardware and is
# run in the Docker build stage before the runtime image is built.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)
project(seg_detect CXX)

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

add_executable(seg_detect seg_detect.cpp)
target_include_directories(seg_detect PRIVATE
    "${OV_ROOT}/include"
    "${SHARED_INCLUDE_DIR}")
if(EXISTS "${NGRAPH_ROOT}/include")
    target_include_directories(seg_detect PRIVATE "${NGRAPH_ROOT}/include")
endif()
target_link_libraries(seg_detect PRIVATE
    "-Wl,--start-group" ${ov_link_libs} "-Wl,--end-group" dl)

if(OV_RUNTIME_PREFIX)
    set_target_properties(seg_detect PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "${ov_rpath_use}")
else()
    set_target_properties(seg_detect PROPERTIES
        BUILD_RPATH "${ov_rpath_use}"
        INSTALL_RPATH "${ov_rpath_use}")
endif()

# device-free post-processing unit test (runs in the build stage)
add_executable(seg_test seg_test.cpp)
target_include_directories(seg_test PRIVATE "${SHARED_INCLUDE_DIR}")

message(STATUS "seg_detect: linking ${ov_link_libs}")
message(STATUS "seg_detect: rpath ${ov_rpath_use}")
