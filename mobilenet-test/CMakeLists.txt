# ---------------------------------------------------------------------------
# mobilenet_classify: real image classification demo for the self-built
# OpenVINO 2020.3.2 runtime (see mobilenet-test/main.cpp and README.md).
#
# The OV link setup is shared in cmake/ov203-link.cmake.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)
project(mobilenet_classify CXX)

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

add_executable(mobilenet_classify main.cpp)
target_include_directories(mobilenet_classify PRIVATE
    "${OV_ROOT}/include"
    "${SHARED_INCLUDE_DIR}")
if(EXISTS "${NGRAPH_ROOT}/include")
    target_include_directories(mobilenet_classify PRIVATE "${NGRAPH_ROOT}/include")
endif()
target_link_libraries(mobilenet_classify PRIVATE
    "-Wl,--start-group" ${ov_link_libs} "-Wl,--end-group" dl)

if(OV_RUNTIME_PREFIX)
    set_target_properties(mobilenet_classify PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "${ov_rpath_use}")
else()
    set_target_properties(mobilenet_classify PROPERTIES
        BUILD_RPATH "${ov_rpath_use}"
        INSTALL_RPATH "${ov_rpath_use}")
endif()

message(STATUS "mobilenet_classify: linking ${ov_link_libs}")
message(STATUS "mobilenet_classify: rpath ${ov_rpath_use}")
