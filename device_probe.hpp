// ---------------------------------------------------------------------------
// device_probe.hpp - shared OpenVINO device probe with retry.
//
// The MA2450's USB state can be in transition when a server starts (e.g.
// right after a previous process released mvnc/libusb, or right after a
// clean shutdown that powers the stick back to ROM mode and re-enumerates
// the USB interface), in which case Core::GetAvailableDevices() may return
// a list that does not contain MYRIAD for several seconds.  waitDevice()
// re-queries repeatedly - by default for about 12 s - before declaring the
// device missing, so transition states do not abort a run with "device not
// available".  When the device is present the first probe returns
// immediately, so the window costs nothing in the normal case.
//
// Device-free to build: it only needs the InferenceEngine core headers,
// which every server already includes.
// ---------------------------------------------------------------------------
#pragma once

#include <ie_core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

// Wait up to (retries + 1) attempts for `device` to appear in the
// available-devices list.  Returns true on first sight; on failure,
// `have` holds the last observed list (for the error message).  The default
// window (12 attempts, ~12 s) covers the USB re-enumeration transition
// after a previous server's shutdown.
inline bool waitDevice(const InferenceEngine::Core& ie,
                       const std::string& device,
                       std::vector<std::string>& have,
                       int retries = 11, int delayMs = 1000) {
    for (int attempt = 0; attempt <= retries; ++attempt) {
        have = ie.GetAvailableDevices();
        if (std::find(have.begin(), have.end(), device) != have.end())
            return true;
        if (attempt < retries) {
            std::fprintf(stderr,
                         "device %s not visible yet (attempt %d/%d), retrying in %d ms\n",
                         device.c_str(), attempt + 1, retries + 1, delayMs);
            std::fflush(stderr);
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
        }
    }
    return false;
}
