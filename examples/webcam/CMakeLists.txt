# ---------------------------------------------------------------------------
# mobilenet_server: long-running MYRIAD inference server for the webcam
# example (see examples/webcam/mobilenet_server.cpp).
#
# Same link setup as mobilenet-test/CMakeLists.txt: the Inference Engine
# shared objects only resolve as a link group.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)
project(mobilenet_server CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED OV_ROOT)
    message(FATAL_ERROR
            "give the Inference Engine install tree: -DOV_ROOT=<...>/deployment_tools/inference_engine")
endif()

if(NOT EXISTS "${OV_ROOT}/include/inference_engine.hpp")
    message(FATAL_ERROR "no inference_engine.hpp under ${OV_ROOT}/include")
endif()

# the install tree places the shared objects in lib/<arch>/ (armv7l upstream,
# aarch64 if the toolchain file is dropped) - glob it so this file is not
# architecture-specific
file(GLOB ov_lib_dirs LIST_DIRECTORIES true "${OV_ROOT}/lib/*")
list(FILTER ov_lib_dirs INCLUDE REGEX ".*/lib/[^/]+$")
if(NOT ov_lib_dirs)
    message(FATAL_ERROR "no library directory found under ${OV_ROOT}/lib")
endif()

find_library(IE_LIBRARY
    NAMES inference_engine
    PATHS ${ov_lib_dirs}
    REQUIRED NO_DEFAULT_PATH)

get_filename_component(IE_LIBDIR "${IE_LIBRARY}" DIRECTORY)

# ngraph is installed next to the inference engine (deployment_tools/ngraph).
set(NGRAPH_ROOT "${OV_ROOT}/../ngraph" CACHE PATH "ngraph part of the install tree")

# libinference_engine.so does not resolve on its own: it references
# libinference_engine_legacy (the CNNNetwork / layer structs), the two
# transformations libraries and libngraph, and some of those reference back into
# libinference_engine.  Resolve the whole set as a link group.
set(ov_link_libs "")
foreach(ov_name IN ITEMS
        inference_engine
        inference_engine_legacy
        inference_engine_transformations
        inference_engine_lp_transformations
        ngraph)
    find_library(ov_found_${ov_name}
        NAMES ${ov_name}
        PATHS ${ov_lib_dirs} "${NGRAPH_ROOT}/lib"
        REQUIRED NO_DEFAULT_PATH)
    list(APPEND ov_link_libs "${ov_found_${ov_name}}")
endforeach()

# rpath: the build-time library directories
set(ov_rpath_build "${ov_lib_dirs}" "${NGRAPH_ROOT}/lib")
# When the executable is copied into the runtime image the build-time library
# directories are gone, so the rpath is rewritten to the runtime install layout.
set(OV_RUNTIME_PREFIX "" CACHE STRING
    "install prefix in the runtime image, e.g. /opt/openvino/inference_engine")
if(OV_RUNTIME_PREFIX)
    # keep the architecture directory the build actually produced (armv7l here)
    list(GET ov_lib_dirs 0 ov_first_libdir)
    get_filename_component(ov_arch "${ov_first_libdir}" NAME)
    set(ov_rpath_use "${OV_RUNTIME_PREFIX}/lib/${ov_arch};${OV_RUNTIME_PREFIX}/../ngraph/lib")
else()
    set(ov_rpath_use "${ov_rpath_build}")
endif()

add_executable(mobilenet_server mobilenet_server.cpp)
target_include_directories(mobilenet_server PRIVATE "${OV_ROOT}/include")
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
