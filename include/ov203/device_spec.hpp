#pragma once

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov203 {

struct DeviceSpec {
    std::string requested;
    bool usesMyriad = false;
    bool usesCpu = false;
    std::vector<std::string> physicalDevices;
};

inline DeviceSpec parseDeviceSpec(const std::string& requested) {
    if (requested.empty())
        throw std::invalid_argument("device must not be empty");
    DeviceSpec out;
    out.requested = requested;
    std::string token;
    auto flush = [&]() {
        if (token.empty()) return;
        if (token == "MYRIAD") {
            out.usesMyriad = true;
            out.physicalDevices.push_back("MYRIAD");
        } else if (token == "CPU") {
            out.usesCpu = true;
            out.physicalDevices.push_back("CPU");
        } else if (token != "HETERO" && token != "MULTI") {
            out.physicalDevices.push_back(token);
        }
        token.clear();
    };
    for (char ch : requested) {
        if (ch == ':' || ch == ',' || std::isspace(static_cast<unsigned char>(ch))) {
            flush();
        } else {
            token.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
        }
    }
    flush();
    std::sort(out.physicalDevices.begin(), out.physicalDevices.end());
    out.physicalDevices.erase(std::unique(out.physicalDevices.begin(), out.physicalDevices.end()),
                              out.physicalDevices.end());
    if (out.physicalDevices.empty())
        throw std::invalid_argument("device specification has no physical plugin");
    return out;
}

}  // namespace ov203
