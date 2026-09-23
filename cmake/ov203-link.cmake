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
#   SHARED_INCLUDE_DIR the directory holding the repo-root shared headers
#                       (half.hpp, device_probe.hpp, frame_utils.hpp) -
#                       add it to each target so #include "half.hpp" etc.
#                       resolve in both the checkout and the Docker layout.
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

# Shared headers (half.hpp, device_probe.hpp, frame_utils.hpp) live next to
# the app directory in the Docker build (/work/<app> + /work/half.hpp) and in
# the repository root in a checkout (examples/<app> or <app> + <root>/half.hpp).
# Auto-detect so #include "half.hpp" resolves in both layouts.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../half.hpp")
    set(SHARED_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../half.hpp")
    set(SHARED_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
else()
    message(FATAL_ERROR
            "shared header half.hpp not found (expected in the repository root)")
endif()
