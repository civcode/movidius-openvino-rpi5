# syntax=docker/dockerfile:1.4
# =============================================================================
# OpenVINO 2020.3.2 with Intel Movidius (MA2450 / MYRIAD) support, built on a
# Raspberry Pi 5 inside an ARMv7 (armhf) container.
#
# Why arm32v7: OpenVINO 2020.3 is the last release whose MYRIAD/VPU plugin
# supports the MA2450 stick, and its prebuilt arm artifacts (and the whole
# mvNC/fathom toolchain it wraps) are 32-bit armv7l.  The Pi 5 kernel has
# CONFIG_COMPAT=y, so arm/v7 containers execute natively (no QEMU).
#
# Stages
#   build-deps : apt + pip layer cache (everything that rarely changes)
#   builder    : configure/build/install inference_engine + MYRIAD plugin for
#                armv7l; source and build trees live in cache mounts
#   smoke      : build smoke-test/hello_myriad against that install tree and
#                generate the tiny FP16 IR used by the demo
#   runtime    : small image with the install tree + hello_myriad + demo entry
#
# Build it with ./build.sh (it resets the vendor tree to pristine, applies
# patches/*.patch and computes the SYNC_STAMP used below).
# =============================================================================

ARG BASE_IMAGE=arm32v7/debian:bullseye
# Debian 11 left LTS in June 2026: deb.debian.org still advertises
# libc6-dev/linux-libc-dev versions in bullseye-security that its pool no
# longer serves (404).  Pin the snapshot the base image itself was cut from.
ARG SNAPSHOT_DATE=20260824T000000Z

# -----------------------------------------------------------------------------
# Stage 1: toolchain / dependency layers (cached aggressively)
# -----------------------------------------------------------------------------
FROM --platform=linux/arm/v7 ${BASE_IMAGE} AS build-deps

ARG SNAPSHOT_DATE
ARG REQUIREMENTS=requirements-build.txt

RUN set -eux; \
    printf 'deb http://snapshot.debian.org/archive/debian/%s bullseye main\ndeb http://snapshot.debian.org/archive/debian-security/%s bullseye-security main\ndeb http://snapshot.debian.org/archive/debian/%s bullseye-updates main\n' \
        "${SNAPSHOT_DATE}" "${SNAPSHOT_DATE}" "${SNAPSHOT_DATE}" > /etc/apt/sources.list; \
    apt -o Acquire::Retries=5 -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update; \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        patch \
        git \
        ca-certificates \
        libusb-1.0-0-dev \
        zlib1g-dev \
        python3 \
        python3-venv; \
    rm -rf /var/lib/apt/lists/*

# Every Python *library* lives in this venv: no python3-* APT packages, no
# system-wide pip installs.  python3-venv above is only the venv enabler.
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY ${REQUIREMENTS} /tmp/requirements-build.txt
RUN pip install --no-cache-dir -r /tmp/requirements-build.txt; \
    rm -rf /root/.cache/pip; \
    python -c 'import sys; print("venv python:", sys.executable, sys.version)'

# -----------------------------------------------------------------------------
# Stage 2: build + install OpenVINO 2020.3.2 for armv7l
# -----------------------------------------------------------------------------
FROM build-deps AS configure

ARG OPENVINO_SOURCE=vendor/openvino-2020.3.2
ARG SYNC_STAMP=unpinned
ARG BUILD_JOBS=2
# set to 0 (./build.sh --no-patches) to reproduce the upstream failures that the
# patches fix; build.sh changes SYNC_STAMP in that case so the cache re-syncs
ARG APPLY_PATCHES=1

# inference-engine/cmake/dependencies.cmake sets CMAKE_STAGING_PREFIX to
# $DL_SDK_TEMP when cross-compiling, so every install() lands here.
ENV DL_SDK_TEMP=/work/stage
# cmake/download/download_and_check.cmake: with IE_PATH_TO_DEPS set, every
# dependency "URL" becomes a local path and is file(COPY)'ed instead of fetched.
ENV IE_PATH_TO_DEPS=/work/deps
ENV OV_BUILD_JOBS=${BUILD_JOBS}
ENV MAKEFLAGS=

RUN mkdir -p /work/src /work/build /work/stage

# 2a. copy the pristine vendor tree + apply patches/NNNN-*.patch into a cache
#     mount.  Bind mounts are read-only, so the patched copy has to live in the
#     cache mount; SYNC_STAMP (pinned commit + patch hashes) decides whether the
#     copy is reused from a previous build.
RUN --mount=type=bind,source=${OPENVINO_SOURCE},target=/work/src-ro \
    --mount=type=bind,source=patches,target=/work/patches-ro \
    --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    --mount=type=cache,target=/work/src \
    set -eux; \
    if [ -f /work/src/.sync_stamp ] && [ "$(cat /work/src/.sync_stamp)" = "${SYNC_STAMP}" ]; then \
        echo "reusing patched source tree (stamp: ${SYNC_STAMP})"; \
    else \
        echo "re-syncing source tree (stamp: ${SYNC_STAMP})"; \
        find /work/src -mindepth 1 -delete; \
        cp -a /work/src-ro/. /work/src/; \
        if [ "${APPLY_PATCHES}" = "1" ]; then \
            for p in $(ls /work/patches-ro/*.patch 2>/dev/null | sort); do \
                echo "applying ${p}"; \
                patch -p1 -d /work/src --quiet < "${p}"; \
            done; \
        else \
            echo "APPLY_PATCHES=${APPLY_PATCHES}: leaving the vendor tree unpatched"; \
        fi; \
        printf '%s' "${SYNC_STAMP}" > /work/src/.sync_stamp; \
    fi; \
    cmake --version; gcc --version | head -1

# 2b. configure.  The toolchain file reports CMAKE_SYSTEM_PROCESSOR=armv7l even
#     though the kernel says aarch64 - that is what selects the armv7l code
#     paths, the armv7l install directory and THREADING=SEQ.
RUN --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    --mount=type=bind,source=vendor/deps,target=/work/deps \
    --mount=type=cache,target=/work/src \
    --mount=type=cache,target=/work/build \
    set -eux; \
    test -f /work/src/CMakeLists.txt; \
    test "$(cat /work/src/.sync_stamp)" = "${SYNC_STAMP}" \
        || { echo "source cache does not match SYNC_STAMP - rebuild with --no-cache"; exit 1; }; \
    cmake -S /work/src -B /work/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=/work/toolchain-ro/armv7-native.toolchain.cmake \
        -DCMAKE_INSTALL_PREFIX=/work/stage \
        -DTHREADS_PTHREAD_ARG=-pthread \
        -DCMAKE_EXE_LINKER_FLAGS=-pthread \
        -DCMAKE_SHARED_LINKER_FLAGS=-pthread \
        -DENABLE_VPU=ON \
        -DENABLE_MYRIAD=ON \
        -DENABLE_MYRIAD_NO_BOOT=OFF \
        -DENABLE_INTEL_MYRIAD_COMMON_ENABLE=ON \
        -DENABLE_IR_READER=ON \
        -DENABLE_GNA=OFF \
        -DENABLE_MKL_DNN=OFF \
        -DENABLE_CLDNN=OFF \
        -DENABLE_OPENCV=OFF \
        -DENABLE_SAMPLES=OFF \
        -DENABLE_TESTS=OFF \
        -DENABLE_FUNCTIONAL_TESTS=OFF \
        -DENABLE_BEH_TESTS=OFF \
        -DENABLE_PYTHON=OFF \
        -DENABLE_PROFILING_ITT=OFF \
        -DENABLE_DEBUG_SYMBOLS=OFF \
        -DENABLE_LTO=OFF \
        -DTHREADING=SEQ \
        -DNGRAPH_UNIT_TEST_ENABLE=FALSE \
        -DNGRAPH_ONNX_IMPORT_ENABLE=FALSE \
        -DNGRAPH_INTERPRETER_ENABLE=TRUE; \
    grep -E 'TARGET_ARCH|CMAKE_SYSTEM_PROCESSOR|CMAKE_CROSSCOMPILING|Enabling VPU|VPU firmware|Myriad|THREADING|ARCH' \
        /work/build/CMakeCache.txt /work/build/CMakeFiles/CMakeOutput.log 2>/dev/null || true; \
    ls -l /work/build

# 3. compile + install (jobs capped: 8 GB RAM, the fathom/MI translation units
#    are heavy).  `--target configure` stops before this stage.
FROM configure AS builder

ARG SYNC_STAMP=unpinned

# The toolchain file has to be visible here too: make re-runs CMake's
# cmake_check_build_system, which re-includes CMAKE_TOOLCHAIN_FILE from the
# cache.
# SYNC_STAMP appears in this instruction on purpose: BuildKit keys a layer on the
# resolved instruction text, not on the *contents* of a cache mount, so without it
# `make` could be skipped from cache after the source tree was re-synced.
RUN --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    --mount=type=bind,source=vendor/deps,target=/work/deps \
    --mount=type=cache,target=/work/src \
    --mount=type=cache,target=/work/build \
    set -eux; \
    test "$(cat /work/src/.sync_stamp)" = "${SYNC_STAMP}" \
        || { echo "source cache does not match SYNC_STAMP - rebuild with --no-cache"; exit 1; }; \
    make -C /work/build -j"${OV_BUILD_JOBS}"; \
    make -C /work/build install; \
    # ngraph installs into the staging prefix top level (lib/, include/, cmake/)
    # while the Inference Engine lands in deployment_tools/inference_engine/.
    # Rearrange it into deployment_tools/ngraph/{lib,include,cmake} so the tree
    # has the same shape as Intel's raspbian package - plugins.xml, the .mvcmd
    # firmware and libmyriadPlugin.so stay in one directory, which is what the
    # MYRIAD plugin requires (it locates the firmware with dladdr()).
    mkdir -p /work/stage/deployment_tools/ngraph/lib \
             /work/stage/deployment_tools/ngraph/include \
             /work/stage/deployment_tools/ngraph/cmake; \
    mv /work/stage/lib/libngraph.so /work/stage/lib/libinterpreter_backend.so \
       /work/stage/deployment_tools/ngraph/lib/; \
    cp -a /work/stage/include/ngraph /work/stage/deployment_tools/ngraph/include/; \
    mv /work/stage/cmake/ngraph*.cmake /work/stage/deployment_tools/ngraph/cmake/; \
    find /work/stage -maxdepth 4 -name '*.so' -o -maxdepth 4 -name 'plugins.xml' | sort; \
    du -sh /work/stage; \
    LD_LIBRARY_PATH=/work/stage/deployment_tools/inference_engine/lib/armv7l:/work/stage/deployment_tools/ngraph/lib \
        /work/stage/deployment_tools/inference_engine/lib/*/myriad_compile --help 2>&1 | head -6 || true

# -----------------------------------------------------------------------------
# Stage 3: smoke test app + tiny model
# -----------------------------------------------------------------------------
FROM builder AS smoke

COPY smoke-test /work/smoke

RUN --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    set -eux; \
    python3 /work/smoke/model/make_tiny_ir.py --outdir /work/smoke/model; \
    cmake -S /work/smoke -B /work/smoke-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=/work/toolchain-ro/armv7-native.toolchain.cmake \
        -DOV_ROOT=/work/stage/deployment_tools/inference_engine \
        -DOV_RUNTIME_PREFIX=/opt/openvino/inference_engine; \
    cmake --build /work/smoke-build -- -j"${OV_BUILD_JOBS}"; \
    readelf -h /work/smoke-build/hello_myriad | head -12; \
    LD_LIBRARY_PATH=/work/stage/deployment_tools/inference_engine/lib/armv7l:/work/stage/deployment_tools/ngraph/lib \
        ldd /work/smoke-build/hello_myriad

# -----------------------------------------------------------------------------
# Stage 3b: the MobileNet classification demo (mobilenet-test/)
# -----------------------------------------------------------------------------
FROM builder AS mobilenet

COPY mobilenet-test /work/mnb

RUN --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    set -eux; \
    cmake -S /work/mnb -B /work/mnb-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=/work/toolchain-ro/armv7-native.toolchain.cmake \
        -DOV_ROOT=/work/stage/deployment_tools/inference_engine \
        -DOV_RUNTIME_PREFIX=/opt/openvino/inference_engine; \
    cmake --build /work/mnb-build -- -j"${OV_BUILD_JOBS}"; \
    readelf -h /work/mnb-build/mobilenet_classify | head -12; \
    LD_LIBRARY_PATH=/work/stage/deployment_tools/inference_engine/lib/armv7l:/work/stage/deployment_tools/ngraph/lib \
        ldd /work/mnb-build/mobilenet_classify

# -----------------------------------------------------------------------------
# Stage 4: runtime image
# -----------------------------------------------------------------------------
FROM --platform=linux/arm/v7 ${BASE_IMAGE} AS runtime

ARG SNAPSHOT_DATE
ENV OV_ROOT=/opt/openvino

RUN set -eux; \
    printf 'deb http://snapshot.debian.org/archive/debian/%s bullseye main\ndeb http://snapshot.debian.org/archive/debian-security/%s bullseye-security main\ndeb http://snapshot.debian.org/archive/debian/%s bullseye-updates main\n' \
        "${SNAPSHOT_DATE}" "${SNAPSHOT_DATE}" "${SNAPSHOT_DATE}" > /etc/apt/sources.list; \
    apt -o Acquire::Retries=5 -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update; \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        libusb-1.0-0 \
        ca-certificates \
        python3; \
    rm -rf /var/lib/apt/lists/*

# the install tree keeps lib/<arch>/*.so, plugins.xml and the .mvcmd firmware
# files in one directory, which is what the MYRIAD plugin expects (it locates
# the firmware next to libmyriadPlugin.so via dladdr)
COPY --from=builder /work/stage/deployment_tools ${OV_ROOT}/
COPY --from=smoke /work/smoke-build/hello_myriad ${OV_ROOT}/bin/hello_myriad
COPY --from=smoke /work/smoke/model /opt/openvino-demo/model
COPY --from=mobilenet /work/mnb-build/mobilenet_classify ${OV_ROOT}/bin/mobilenet_classify
COPY container-entry.sh /opt/openvino-demo/run.sh

# the MobileNet corpus (ONNX source, IR, labels, photos) stays on the host and is
# mounted read-only by ./run.sh mobilenet, so the image does not carry 20 MB of model
RUN chmod +x /opt/openvino-demo/run.sh ${OV_ROOT}/bin/hello_myriad ${OV_ROOT}/bin/mobilenet_classify \
    && ls -l ${OV_ROOT} ${OV_ROOT}/bin ${OV_ROOT}/inference_engine/lib/*/ \
    && LD_LIBRARY_PATH="$(ls -d ${OV_ROOT}/inference_engine/lib/*/):${OV_ROOT}/ngraph/lib" \
       ${OV_ROOT}/bin/hello_myriad --help \
    && LD_LIBRARY_PATH="$(ls -d ${OV_ROOT}/inference_engine/lib/*/):${OV_ROOT}/ngraph/lib" \
       ${OV_ROOT}/bin/mobilenet_classify --help

WORKDIR /opt/openvino-demo
ENTRYPOINT ["/opt/openvino-demo/run.sh"]
CMD ["--demo"]
