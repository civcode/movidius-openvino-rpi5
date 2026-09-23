# syntax=docker/dockerfile:1.4
# =============================================================================
# OpenVINO 2020.3.2 + Intel Movidius MA2450/MYRIAD, built from the same pinned
# source tree for three project targets:
#   armv7 -> linux/arm/v7 (legacy/known-good Raspberry Pi ARMHF userspace)
#   arm64 -> linux/arm64  (native AArch64; preferred Raspberry Pi 5 path)
#   amd64 -> linux/amd64  (native x86_64 Linux)
#
# The application path is identical on all targets:
# application -> Inference Engine -> libmyriadPlugin.so -> MVNC/XLink/libusb ->
# usb-ma2450.mvcmd -> MA2450. MA2Host is not part of the normal runtime.
# =============================================================================

ARG TARGET=armv7
ARG DOCKER_PLATFORM=linux/arm/v7
ARG BASE_IMAGE=arm32v7/debian:bullseye
# Debian 11 package indices are pinned to a known snapshot for reproducibility.
ARG SNAPSHOT_DATE=20260824T000000Z

# -----------------------------------------------------------------------------
# Stage 1: build dependencies
# -----------------------------------------------------------------------------
FROM --platform=${DOCKER_PLATFORM} ${BASE_IMAGE} AS build-deps

ARG TARGET
ARG DOCKER_PLATFORM
ARG SNAPSHOT_DATE
ARG REQUIREMENTS=requirements-build.txt

RUN set -eux; \
    printf 'target=%s docker_platform=%s\n' "${TARGET}" "${DOCKER_PLATFORM}"; \
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

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY ${REQUIREMENTS} /tmp/requirements-build.txt
RUN pip install --no-cache-dir -r /tmp/requirements-build.txt; \
    rm -rf /root/.cache/pip; \
    python -c 'import sys; print("venv python:", sys.executable, sys.version)'

# -----------------------------------------------------------------------------
# Stage 2: configure OpenVINO. ARMv7 uses the project toolchain; ARM64/amd64 are native.
# -----------------------------------------------------------------------------
FROM build-deps AS configure

ARG TARGET
ARG OPENVINO_SOURCE=vendor/openvino-2020.3.2
ARG SYNC_STAMP=unpinned
ARG BUILD_JOBS=2
ARG APPLY_PATCHES=1
ARG USE_CMAKE_TOOLCHAIN=1

ENV DL_SDK_TEMP=/work/stage
ENV IE_PATH_TO_DEPS=/work/deps
ENV OV_BUILD_JOBS=${BUILD_JOBS}
ENV MAKEFLAGS=

RUN mkdir -p /work/src /work/build /work/stage

# Target-qualified source/build caches prevent generated CMake state from one
# architecture leaking into another target. The source tree itself is copied pristine
# before patches are applied.
RUN --mount=type=bind,source=${OPENVINO_SOURCE},target=/work/src-ro \
    --mount=type=bind,source=patches,target=/work/patches-ro \
    --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    --mount=type=cache,id=ov203-src-${TARGET},target=/work/src \
    set -eux; \
    if [ -f /work/src/.sync_stamp ] && [ "$(cat /work/src/.sync_stamp)" = "${SYNC_STAMP}" ]; then \
        echo "reusing patched source tree (stamp: ${SYNC_STAMP})"; \
    else \
        echo "re-syncing source tree (stamp: ${SYNC_STAMP})"; \
        find /work/src -mindepth 1 -delete; \
        cp -a /work/src-ro/. /work/src/; \
        if [ "${APPLY_PATCHES}" = 1 ]; then \
            for p in $(find /work/patches-ro -maxdepth 1 -type f -name '*.patch' | sort); do \
                echo "applying ${p}"; \
                patch -p1 -d /work/src --quiet < "${p}"; \
            done; \
        else \
            echo "APPLY_PATCHES=${APPLY_PATCHES}: leaving vendor tree unpatched"; \
        fi; \
        printf '%s' "${SYNC_STAMP}" > /work/src/.sync_stamp; \
    fi; \
    cmake --version; gcc --version | head -1

RUN --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    --mount=type=bind,source=vendor/deps,target=/work/deps \
    --mount=type=cache,id=ov203-src-${TARGET},target=/work/src \
    --mount=type=cache,id=ov203-build-${TARGET},target=/work/build \
    set -eux; \
    test -f /work/src/CMakeLists.txt; \
    test "$(cat /work/src/.sync_stamp)" = "${SYNC_STAMP}" \
        || { echo "source cache does not match SYNC_STAMP"; exit 1; }; \
    set --; \
    if [ "${USE_CMAKE_TOOLCHAIN}" = 1 ]; then \
        set -- "$@" -DCMAKE_TOOLCHAIN_FILE=/work/toolchain-ro/armv7-native.toolchain.cmake; \
    fi; \
    cmake -S /work/src -B /work/build \
        "$@" \
        -DCMAKE_BUILD_TYPE=Release \
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
    echo "target=${TARGET} dpkg_arch=$(dpkg --print-architecture) toolchain=${USE_CMAKE_TOOLCHAIN}"; \
    grep -E 'TARGET_ARCH|CMAKE_SYSTEM_PROCESSOR|CMAKE_CROSSCOMPILING|Enabling VPU|VPU firmware|Myriad|THREADING|ARCH' \
        /work/build/CMakeCache.txt /work/build/CMakeFiles/CMakeOutput.log 2>/dev/null || true

# -----------------------------------------------------------------------------
# Stage 3: compile and install the runtime/plugin stack
# -----------------------------------------------------------------------------
FROM configure AS builder

ARG TARGET
ARG DOCKER_PLATFORM
ARG BASE_IMAGE
ARG PROJECT_REVISION=unversioned
ARG OPENVINO_COMMIT=unknown
ARG SYNC_STAMP=unpinned
ARG EXPECTED_ELF_CLASS=ELF32
ARG EXPECTED_ELF_MACHINE_REGEX=ARM
ARG EXPECTED_ELF_MACHINE_ID=40

RUN --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    --mount=type=bind,source=vendor/deps,target=/work/deps \
    --mount=type=cache,id=ov203-src-${TARGET},target=/work/src \
    --mount=type=cache,id=ov203-build-${TARGET},target=/work/build \
    set -eux; \
    test "$(cat /work/src/.sync_stamp)" = "${SYNC_STAMP}"; \
    make -C /work/build -j"${OV_BUILD_JOBS}"; \
    make -C /work/build install; \
    mkdir -p /work/stage/deployment_tools/ngraph/lib \
             /work/stage/deployment_tools/ngraph/include \
             /work/stage/deployment_tools/ngraph/cmake; \
    NGRAPH_SO="$(find /work/stage -name libngraph.so -print -quit)"; \
    test -n "${NGRAPH_SO}"; \
    NGRAPH_SRC="$(dirname "${NGRAPH_SO}")"; \
    if [ "${NGRAPH_SRC}" != /work/stage/deployment_tools/ngraph/lib ]; then \
        cp -a "${NGRAPH_SRC}"/libngraph.so* /work/stage/deployment_tools/ngraph/lib/; \
        if ls "${NGRAPH_SRC}"/libinterpreter_backend.so* >/dev/null 2>&1; then \
            cp -a "${NGRAPH_SRC}"/libinterpreter_backend.so* /work/stage/deployment_tools/ngraph/lib/; \
        fi; \
    fi; \
    if [ -d /work/stage/include/ngraph ]; then \
        cp -a /work/stage/include/ngraph /work/stage/deployment_tools/ngraph/include/; \
    fi; \
    if ls /work/stage/cmake/ngraph*.cmake >/dev/null 2>&1; then \
        cp -a /work/stage/cmake/ngraph*.cmake /work/stage/deployment_tools/ngraph/cmake/; \
    fi; \
    IE_PLUGIN="$(find /work/stage/deployment_tools/inference_engine/lib -mindepth 2 -maxdepth 2 -type f -name libmyriadPlugin.so -print -quit)"; \
    test -n "${IE_PLUGIN}"; \
    IE_LIB="$(dirname "${IE_PLUGIN}")"; \
    echo "installed MYRIAD plugin: ${IE_PLUGIN}"; \
    readelf -h "${IE_PLUGIN}" | tee /tmp/myriad-plugin.elf; \
    grep -q "Class:.*${EXPECTED_ELF_CLASS}" /tmp/myriad-plugin.elf; \
    grep -Eq "Machine:.*(${EXPECTED_ELF_MACHINE_REGEX})" /tmp/myriad-plugin.elf; \
    test -f "${IE_LIB}/usb-ma2450.mvcmd"; \
    test -f "${IE_LIB}/plugins.xml"; \
    sha256sum "${IE_LIB}"/*.mvcmd; \
    CMAKE_PROCESSOR="$(sed -n 's/^CMAKE_SYSTEM_PROCESSOR:.*=//p' /work/build/CMakeCache.txt | head -1)"; \
    { \
        printf 'PROJECT_REVISION=%s\n' "${PROJECT_REVISION}"; \
        printf 'TARGET=%s\n' "${TARGET}"; \
        printf 'EXPECTED_ELF_CLASS=%s\n' "${EXPECTED_ELF_CLASS}"; \
        printf 'EXPECTED_ELF_MACHINE_ID=%s\n' "${EXPECTED_ELF_MACHINE_ID}"; \
        printf 'DOCKER_PLATFORM=%s\n' "${DOCKER_PLATFORM}"; \
        printf 'BASE_IMAGE=%s\n' "${BASE_IMAGE}"; \
        printf 'OPENVINO_COMMIT=%s\n' "${OPENVINO_COMMIT}"; \
        printf 'CMAKE_SYSTEM_PROCESSOR=%s\n' "${CMAKE_PROCESSOR}"; \
        printf 'IE_LIB_BASENAME=%s\n' "$(basename "${IE_LIB}")"; \
        printf 'CC_VERSION=%s\n' "$(gcc --version | head -1)"; \
        for fw in "${IE_LIB}"/*.mvcmd; do printf 'FIRMWARE_SHA256_%s=%s\n' "$(basename "${fw}")" "$(sha256sum "${fw}" | cut -d' ' -f1)"; done; \
    } > /work/stage/deployment_tools/BUILD-INFO.txt; \
    cat /work/stage/deployment_tools/BUILD-INFO.txt; \
    find /work/stage -maxdepth 5 \( -name '*.so' -o -name 'plugins.xml' -o -name '*.mvcmd' \) -print | sort; \
    du -sh /work/stage; \
    LD_LIBRARY_PATH="${IE_LIB}:/work/stage/deployment_tools/ngraph/lib" \
        "${IE_LIB}/myriad_compile" --help 2>&1 | head -6 || true

# -----------------------------------------------------------------------------
# Stage 4: build the smoke app natively for the selected target
# -----------------------------------------------------------------------------
FROM builder AS smoke

ARG TARGET
ARG USE_CMAKE_TOOLCHAIN=1
ARG EXPECTED_ELF_CLASS=ELF32
ARG EXPECTED_ELF_MACHINE_REGEX=ARM
COPY smoke-test /work/smoke
COPY half.hpp /work/half.hpp
COPY cmake /work/cmake

RUN --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    set -eux; \
    python3 /work/smoke/model/make_tiny_ir.py --outdir /work/smoke/model; \
    set --; \
    if [ "${USE_CMAKE_TOOLCHAIN}" = 1 ]; then \
        set -- "$@" -DCMAKE_TOOLCHAIN_FILE=/work/toolchain-ro/armv7-native.toolchain.cmake; \
    fi; \
    cmake -S /work/smoke -B /work/smoke-build \
        "$@" \
        -DCMAKE_BUILD_TYPE=Release \
        -DOV_ROOT=/work/stage/deployment_tools/inference_engine \
        -DOV_RUNTIME_PREFIX=/opt/openvino/inference_engine; \
    cmake --build /work/smoke-build -- -j"${OV_BUILD_JOBS}"; \
    readelf -h /work/smoke-build/hello_myriad | tee /tmp/hello.elf; \
    grep -q "Class:.*${EXPECTED_ELF_CLASS}" /tmp/hello.elf; \
    grep -Eq "Machine:.*(${EXPECTED_ELF_MACHINE_REGEX})" /tmp/hello.elf; \
    IE_LIB="$(dirname "$(find /work/stage/deployment_tools/inference_engine/lib -mindepth 2 -maxdepth 2 -name libinference_engine.so -print -quit)")"; \
    LD_LIBRARY_PATH="${IE_LIB}:/work/stage/deployment_tools/ngraph/lib" ldd /work/smoke-build/hello_myriad; \
    ! LD_LIBRARY_PATH="${IE_LIB}:/work/stage/deployment_tools/ngraph/lib" ldd /work/smoke-build/hello_myriad | grep -q 'not found'

# -----------------------------------------------------------------------------
# Stage 5: build MobileNet demo natively for the selected target
# -----------------------------------------------------------------------------
FROM builder AS mobilenet

ARG TARGET
ARG USE_CMAKE_TOOLCHAIN=1
ARG EXPECTED_ELF_CLASS=ELF32
ARG EXPECTED_ELF_MACHINE_REGEX=ARM
COPY mobilenet-test /work/mnb
COPY half.hpp /work/half.hpp
COPY cmake /work/cmake

RUN --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    set -eux; \
    set --; \
    if [ "${USE_CMAKE_TOOLCHAIN}" = 1 ]; then \
        set -- "$@" -DCMAKE_TOOLCHAIN_FILE=/work/toolchain-ro/armv7-native.toolchain.cmake; \
    fi; \
    cmake -S /work/mnb -B /work/mnb-build \
        "$@" \
        -DCMAKE_BUILD_TYPE=Release \
        -DOV_ROOT=/work/stage/deployment_tools/inference_engine \
        -DOV_RUNTIME_PREFIX=/opt/openvino/inference_engine; \
    cmake --build /work/mnb-build -- -j"${OV_BUILD_JOBS}"; \
    readelf -h /work/mnb-build/mobilenet_classify | tee /tmp/mobilenet.elf; \
    grep -q "Class:.*${EXPECTED_ELF_CLASS}" /tmp/mobilenet.elf; \
    grep -Eq "Machine:.*(${EXPECTED_ELF_MACHINE_REGEX})" /tmp/mobilenet.elf; \
    IE_LIB="$(dirname "$(find /work/stage/deployment_tools/inference_engine/lib -mindepth 2 -maxdepth 2 -name libinference_engine.so -print -quit)")"; \
    LD_LIBRARY_PATH="${IE_LIB}:/work/stage/deployment_tools/ngraph/lib" ldd /work/mnb-build/mobilenet_classify; \
    ! LD_LIBRARY_PATH="${IE_LIB}:/work/stage/deployment_tools/ngraph/lib" ldd /work/mnb-build/mobilenet_classify | grep -q 'not found'

# -----------------------------------------------------------------------------
# Stage 5b: build the webcam example inference server for the selected target
# -----------------------------------------------------------------------------
FROM builder AS webcam

ARG TARGET
ARG USE_CMAKE_TOOLCHAIN=1
ARG EXPECTED_ELF_CLASS=ELF32
ARG EXPECTED_ELF_MACHINE_REGEX=ARM
COPY examples/webcam /work/webcam
COPY half.hpp /work/half.hpp
COPY device_probe.hpp /work/device_probe.hpp
COPY cmake /work/cmake

RUN --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    set -eux; \
    set --; \
    if [ "${USE_CMAKE_TOOLCHAIN}" = 1 ]; then \
        set -- "$@" -DCMAKE_TOOLCHAIN_FILE=/work/toolchain-ro/armv7-native.toolchain.cmake; \
    fi; \
    cmake -S /work/webcam -B /work/webcam-build \
        "$@" \
        -DCMAKE_BUILD_TYPE=Release \
        -DOV_ROOT=/work/stage/deployment_tools/inference_engine \
        -DOV_RUNTIME_PREFIX=/opt/openvino/inference_engine; \
    cmake --build /work/webcam-build -- -j"${OV_BUILD_JOBS}"; \
    /work/webcam-build/test_half; \
    readelf -h /work/webcam-build/mobilenet_server | tee /tmp/webcam.elf; \
    grep -q "Class:.*${EXPECTED_ELF_CLASS}" /tmp/webcam.elf; \
    grep -Eq "Machine:.*(${EXPECTED_ELF_MACHINE_REGEX})" /tmp/webcam.elf; \
    IE_LIB="$(dirname "$(find /work/stage/deployment_tools/inference_engine/lib -mindepth 2 -maxdepth 2 -name libinference_engine.so -print -quit)")"; \
    LD_LIBRARY_PATH="${IE_LIB}:/work/stage/deployment_tools/ngraph/lib" ldd /work/webcam-build/mobilenet_server; \
    ! LD_LIBRARY_PATH="${IE_LIB}:/work/stage/deployment_tools/ngraph/lib" ldd /work/webcam-build/mobilenet_server | grep -q 'not found'

# -----------------------------------------------------------------------------
# Stage 5c: build the SSDLite detector for the selected target
# -----------------------------------------------------------------------------
FROM builder AS ssd

ARG TARGET
ARG USE_CMAKE_TOOLCHAIN=1
ARG EXPECTED_ELF_CLASS=ELF32
ARG EXPECTED_ELF_MACHINE_REGEX=ARM
COPY examples/ssd-detect /work/ssd
COPY half.hpp /work/half.hpp
COPY device_probe.hpp /work/device_probe.hpp
COPY frame_utils.hpp /work/frame_utils.hpp
COPY cmake /work/cmake

RUN --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    set -eux; \
    set --; \
    if [ "${USE_CMAKE_TOOLCHAIN}" = 1 ]; then \
        set -- "$@" -DCMAKE_TOOLCHAIN_FILE=/work/toolchain-ro/armv7-native.toolchain.cmake; \
    fi; \
    cmake -S /work/ssd -B /work/ssd-build \
        "$@" \
        -DCMAKE_BUILD_TYPE=Release \
        -DOV_ROOT=/work/stage/deployment_tools/inference_engine \
        -DOV_RUNTIME_PREFIX=/opt/openvino/inference_engine; \
    cmake --build /work/ssd-build -- -j"${OV_BUILD_JOBS}"; \
    /work/ssd-build/ssd_test; \
    readelf -h /work/ssd-build/ssd_detect | tee /tmp/ssd.elf; \
    grep -q "Class:.*${EXPECTED_ELF_CLASS}" /tmp/ssd.elf; \
    grep -Eq "Machine:.*(${EXPECTED_ELF_MACHINE_REGEX})" /tmp/ssd.elf; \
    IE_LIB="$(dirname "$(find /work/stage/deployment_tools/inference_engine/lib -mindepth 2 -maxdepth 2 -name libinference_engine.so -print -quit)")"; \
    LD_LIBRARY_PATH="${IE_LIB}:/work/stage/deployment_tools/ngraph/lib" ldd /work/ssd-build/ssd_detect; \
    ! LD_LIBRARY_PATH="${IE_LIB}:/work/stage/deployment_tools/ngraph/lib" ldd /work/ssd-build/ssd_detect | grep -q 'not found'

# -----------------------------------------------------------------------------
# Stage 5d: build the DeepLabV3 segmenter for the selected target
# -----------------------------------------------------------------------------
FROM builder AS seg

ARG TARGET
ARG USE_CMAKE_TOOLCHAIN=1
ARG EXPECTED_ELF_CLASS=ELF32
ARG EXPECTED_ELF_MACHINE_REGEX=ARM
COPY examples/deeplab-seg /work/seg
COPY half.hpp /work/half.hpp
COPY device_probe.hpp /work/device_probe.hpp
COPY frame_utils.hpp /work/frame_utils.hpp
COPY cmake /work/cmake

RUN --mount=type=bind,source=toolchain,target=/work/toolchain-ro \
    set -eux; \
    set --; \
    if [ "${USE_CMAKE_TOOLCHAIN}" = 1 ]; then \
        set -- "$@" -DCMAKE_TOOLCHAIN_FILE=/work/toolchain-ro/armv7-native.toolchain.cmake; \
    fi; \
    cmake -S /work/seg -B /work/seg-build \
        "$@" \
        -DCMAKE_BUILD_TYPE=Release \
        -DOV_ROOT=/work/stage/deployment_tools/inference_engine \
        -DOV_RUNTIME_PREFIX=/opt/openvino/inference_engine; \
    cmake --build /work/seg-build -- -j"${OV_BUILD_JOBS}"; \
    /work/seg-build/seg_test; \
    readelf -h /work/seg-build/seg_detect | tee /tmp/seg.elf; \
    grep -q "Class:.*${EXPECTED_ELF_CLASS}" /tmp/seg.elf; \
    grep -Eq "Machine:.*(${EXPECTED_ELF_MACHINE_REGEX})" /tmp/seg.elf; \
    IE_LIB="$(dirname "$(find /work/stage/deployment_tools/inference_engine/lib -mindepth 2 -maxdepth 2 -name libinference_engine.so -print -quit)")"; \
    LD_LIBRARY_PATH="${IE_LIB}:/work/stage/deployment_tools/ngraph/lib" ldd /work/seg-build/seg_detect; \
    ! LD_LIBRARY_PATH="${IE_LIB}:/work/stage/deployment_tools/ngraph/lib" ldd /work/seg-build/seg_detect | grep -q 'not found'

# -----------------------------------------------------------------------------
# Stage 6: runtime image
# -----------------------------------------------------------------------------
FROM --platform=${DOCKER_PLATFORM} ${BASE_IMAGE} AS runtime

ARG TARGET
ARG DOCKER_PLATFORM
ARG BASE_IMAGE
ARG PROJECT_REVISION=unversioned
ARG OPENVINO_COMMIT=unknown
ARG SNAPSHOT_DATE
ARG EXPECTED_ELF_CLASS
ARG EXPECTED_ELF_MACHINE_ID
ENV OV_ROOT=/opt/openvino
ENV OV_TARGET=${TARGET}

RUN set -eux; \
    printf 'deb http://snapshot.debian.org/archive/debian/%s bullseye main\ndeb http://snapshot.debian.org/archive/debian-security/%s bullseye-security main\ndeb http://snapshot.debian.org/archive/debian/%s bullseye-updates main\n' \
        "${SNAPSHOT_DATE}" "${SNAPSHOT_DATE}" "${SNAPSHOT_DATE}" > /etc/apt/sources.list; \
    apt -o Acquire::Retries=5 -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update; \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        libusb-1.0-0 ca-certificates python3; \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /work/stage/deployment_tools ${OV_ROOT}/
COPY --from=smoke /work/smoke-build/hello_myriad ${OV_ROOT}/bin/hello_myriad
COPY --from=smoke /work/smoke/model /opt/openvino-demo/model
COPY --from=mobilenet /work/mnb-build/mobilenet_classify ${OV_ROOT}/bin/mobilenet_classify
COPY --from=webcam /work/webcam-build/mobilenet_server ${OV_ROOT}/bin/mobilenet_server
COPY --from=ssd /work/ssd-build/ssd_detect ${OV_ROOT}/bin/ssd_detect
COPY --from=seg /work/seg-build/seg_detect ${OV_ROOT}/bin/seg_detect
COPY container-entry.sh /opt/openvino-demo/run.sh

RUN set -eux; \
    chmod +x /opt/openvino-demo/run.sh ${OV_ROOT}/bin/hello_myriad ${OV_ROOT}/bin/mobilenet_classify ${OV_ROOT}/bin/mobilenet_server ${OV_ROOT}/bin/ssd_detect ${OV_ROOT}/bin/seg_detect; \
    IE_PLUGIN="$(find ${OV_ROOT}/inference_engine/lib -mindepth 2 -maxdepth 2 -type f -name libmyriadPlugin.so -print -quit)"; \
    test -n "${IE_PLUGIN}"; \
    IE_LIB="$(dirname "${IE_PLUGIN}")"; \
    test -f "${IE_LIB}/usb-ma2450.mvcmd"; \
    test -f "${IE_LIB}/plugins.xml"; \
    printf 'PROJECT_REVISION=%s\nTARGET=%s\nDOCKER_PLATFORM=%s\nBASE_IMAGE=%s\nOPENVINO_COMMIT=%s\nIE_LIB_BASENAME=%s\nEXPECTED_ELF_CLASS=%s\nEXPECTED_ELF_MACHINE_ID=%s\n' \
        "${PROJECT_REVISION}" "${TARGET}" "${DOCKER_PLATFORM}" "${BASE_IMAGE}" "${OPENVINO_COMMIT}" "$(basename "${IE_LIB}")" "${EXPECTED_ELF_CLASS}" "${EXPECTED_ELF_MACHINE_ID}" \
        > ${OV_ROOT}/runtime-manifest.env; \
    (cd ${OV_ROOT} && sha256sum "inference_engine/lib/$(basename "${IE_LIB}")"/*.mvcmd) > ${OV_ROOT}/firmware.sha256; \
    cat ${OV_ROOT}/runtime-manifest.env; \
    cat ${OV_ROOT}/firmware.sha256; \
    LD_LIBRARY_PATH="${IE_LIB}:${OV_ROOT}/ngraph/lib" ${OV_ROOT}/bin/hello_myriad --help; \
    LD_LIBRARY_PATH="${IE_LIB}:${OV_ROOT}/ngraph/lib" ${OV_ROOT}/bin/mobilenet_classify --help; \
    ! LD_LIBRARY_PATH="${IE_LIB}:${OV_ROOT}/ngraph/lib" ldd ${OV_ROOT}/bin/hello_myriad | grep -q 'not found'; \
    ! LD_LIBRARY_PATH="${IE_LIB}:${OV_ROOT}/ngraph/lib" ldd ${OV_ROOT}/bin/mobilenet_classify | grep -q 'not found'; \
    LD_LIBRARY_PATH="${IE_LIB}:${OV_ROOT}/ngraph/lib" ${OV_ROOT}/bin/mobilenet_server --help; \
    ! LD_LIBRARY_PATH="${IE_LIB}:${OV_ROOT}/ngraph/lib" ldd ${OV_ROOT}/bin/mobilenet_server | grep -q 'not found'; \
    LD_LIBRARY_PATH="${IE_LIB}:${OV_ROOT}/ngraph/lib" ${OV_ROOT}/bin/ssd_detect --help; \
    ! LD_LIBRARY_PATH="${IE_LIB}:${OV_ROOT}/ngraph/lib" ldd ${OV_ROOT}/bin/ssd_detect | grep -q 'not found'; \
    LD_LIBRARY_PATH="${IE_LIB}:${OV_ROOT}/ngraph/lib" ${OV_ROOT}/bin/seg_detect --help; \
    ! LD_LIBRARY_PATH="${IE_LIB}:${OV_ROOT}/ngraph/lib" ldd ${OV_ROOT}/bin/seg_detect | grep -q 'not found'

WORKDIR /opt/openvino-demo
ENTRYPOINT ["/opt/openvino-demo/run.sh"]
CMD ["--demo"]
