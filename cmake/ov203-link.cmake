# ---------------------------------------------------------------------------
# ov203-link.cmake - shared CMake logic for every app that links the
# self-built OpenVINO 2020.3.2 Inference Engine.
#
# Include from a per-app CMakeLists.txt (after project() and the
# CMAKE_CXX_STANDARD settings) via a self-locating probe so it resolves
# in both layouts:
#
#   - a plain checkout:   <root>/cmake/ov203-link.cmake
#   - the Docker build:    /work/cmake/ov203-link.cmake  (COPY cmake /work/cmake)
#
#   foreach(_cand "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake"
#                 "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
#       if(EXISTS "${_cand}/ov203-link.cmake")
#           include("${_cand}/ov203-link.cmake")
#           break()
#       endif()
#   endforeach()
#
# Requires:  -DOV_ROOT=<install>/deployment_tools/inference_engine
# Produces:
#   IE_LIBDIR          directory holding libinference_engine.so
#   ov_link_libs       the full 5-library link group
#   ov_rpath_use       rpath (build dirs, or the OV_RUNTIME_PREFIX layout)
#   SHARED_INCLUDE_DIR the include root containing ov203/*.hpp.
# ---------------------------------------------------------------------------

if(NOT DEFINED OV_ROOT)
    message(FATAL_ERROR
            "give the Inference Engine install tree: -DOV_ROOT=<...>/deployment_tools/inference_engine")
endif()

if(NOT EXISTS "${OV_ROOT}/include/inference_engine.hpp")
    message(FATAL_ERROR "no inference_engine.hpp under ${OV_ROOT}/include")
endif()

# the install tree places the shared objects in lib/<arch>/ - glob it so this
# file is not architecture-specific
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
    # keep the architecture directory the build actually produced
    list(GET ov_lib_dirs 0 ov_first_libdir)
    get_filename_component(ov_arch "${ov_first_libdir}" NAME)
    set(ov_rpath_use "${OV_RUNTIME_PREFIX}/lib/${ov_arch};${OV_RUNTIME_PREFIX}/../ngraph/lib")
else()
    set(ov_rpath_use "${ov_rpath_build}")
endif()

# Shared project headers live under include/ov203.  Resolve the include root
# in both source-checkout layouts (examples/<app>) and Docker build layouts
# (/work/<app> with /work/include copied alongside it).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../include/ov203/half.hpp")
    set(SHARED_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../include/ov203/half.hpp")
    set(SHARED_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
else()
    message(FATAL_ERROR
            "shared headers not found (expected include/ov203 below repository/build root)")
endif()


function(ov203_add_inference_executable target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE
        "${OV_ROOT}/include"
        "${SHARED_INCLUDE_DIR}")
    if(EXISTS "${NGRAPH_ROOT}/include")
        target_include_directories(${target} PRIVATE "${NGRAPH_ROOT}/include")
    endif()
    target_link_libraries(${target} PRIVATE
        "-Wl,--start-group" ${ov_link_libs} "-Wl,--end-group" dl)
    if(OV_RUNTIME_PREFIX)
        set_target_properties(${target} PROPERTIES
            BUILD_WITH_INSTALL_RPATH ON
            INSTALL_RPATH "${ov_rpath_use}")
    else()
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "${ov_rpath_use}"
            INSTALL_RPATH "${ov_rpath_use}")
    endif()
endfunction()
