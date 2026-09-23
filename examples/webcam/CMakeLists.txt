# ---------------------------------------------------------------------------
# mobilenet_server: long-running MYRIAD inference server for the webcam
# example (see examples/webcam/mobilenet_server.cpp).
#
# The OV link setup is shared in cmake/ov203-link.cmake.  test_half is a
# device-free unit test for the shared half.hpp conversion, run in the
# Docker build stage.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)
project(mobilenet_server CXX)

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

# device-free unit test for the shared half.hpp conversion
add_executable(test_half test_half.cpp)
target_include_directories(test_half PRIVATE "${SHARED_INCLUDE_DIR}")

add_executable(mobilenet_server mobilenet_server.cpp)
target_include_directories(mobilenet_server PRIVATE
    "${OV_ROOT}/include"
    "${SHARED_INCLUDE_DIR}")
if(EXISTS "${NGRAPH_ROOT}/include")
    target_include_directories(mobilenet_server PRIVATE "${NGRAPH_ROOT}/include")
endif()
target_link_libraries(mobilenet_server PRIVATE
    "-Wl,--start-group" ${ov_link_libs} "-Wl,--end-group" dl)

if(OV_RUNTIME_PREFIX)
    set_target_properties(mobilenet_server PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "${ov_rpath_use}")
else()
    set_target_properties(mobilenet_server PROPERTIES
        BUILD_RPATH "${ov_rpath_use}"
        INSTALL_RPATH "${ov_rpath_use}")
endif()

message(STATUS "mobilenet_server: linking ${ov_link_libs}")
message(STATUS "mobilenet_server: rpath ${ov_rpath_use}")
