# Native compiler image for standalone MVNC/XLink probes on any project target.
ARG DOCKER_PLATFORM=linux/amd64
ARG BASE_IMAGE=debian:bullseye
FROM --platform=${DOCKER_PLATFORM} ${BASE_IMAGE}
ARG SNAPSHOT_DATE=20260824T000000Z
RUN set -eux; \
    printf 'deb http://snapshot.debian.org/archive/debian/%s bullseye main\ndeb http://snapshot.debian.org/archive/debian-security/%s bullseye-security main\ndeb http://snapshot.debian.org/archive/debian/%s bullseye-updates main\n' \
      "$SNAPSHOT_DATE" "$SNAPSHOT_DATE" "$SNAPSHOT_DATE" >/etc/apt/sources.list; \
    apt -o Acquire::Retries=5 -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update; \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
      build-essential libusb-1.0-0-dev libusb-1.0-0 ca-certificates; \
    rm -rf /var/lib/apt/lists/*
