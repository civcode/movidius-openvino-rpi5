# Reference-runtime sanity check image (arm32v7 = armhf on the Pi 5).
# Used only by scripts/validate-reference.sh to prove, on this host and with
# this stick, that: an arm32v7 container can boot the MA2450 from ROM, the
# official OpenVINO 2020.3.2 MYRIAD plugin compiles our IR, and smoke-test/main.cpp
# compiles against the 2020.3.2 API.  It is NOT part of the deliverable image.
#
# APT note: Debian 11 (bullseye) left LTS support in June 2026 and the
# bullseye-security pool on deb.debian.org no longer serves the .debs its own
# Packages index advertises (404 for libc6-dev / linux-libc-dev).  We pin the
# same snapshot.debian.org date the arm32v7/debian:bullseye base image was built
# from, which is both reproducible and reachable.
FROM --platform=linux/arm/v7 arm32v7/debian:bullseye

ARG SNAPSHOT_DATE=20260824T000000Z

RUN set -eux; \
    printf 'deb http://snapshot.debian.org/archive/debian/%s bullseye main\ndeb http://snapshot.debian.org/archive/debian-security/%s bullseye-security main\ndeb http://snapshot.debian.org/archive/debian/%s bullseye-updates main\n' \
        "${SNAPSHOT_DATE}" "${SNAPSHOT_DATE}" "${SNAPSHOT_DATE}" > /etc/apt/sources.list; \
    apt -o Acquire::Retries=5 -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update; \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        g++ \
        libusb-1.0-0-dev \
        libusb-1.0-0; \
    rm -rf /var/lib/apt/lists/*
