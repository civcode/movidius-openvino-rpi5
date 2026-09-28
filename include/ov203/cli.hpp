#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace ov203 {

inline int parsePositiveInt(const std::string& name, const std::string& text) {
    std::size_t pos = 0;
    int value = 0;
    try { value = std::stoi(text, &pos); }
    catch (const std::exception&) { throw std::runtime_error(name + " must be an integer >= 1"); }
    if (pos != text.size() || value < 1)
        throw std::runtime_error(name + " must be an integer >= 1");
    return value;
}

inline float parseProbability(const std::string& name, const std::string& text) {
    std::size_t pos = 0;
    float value = 0.0f;
    try { value = std::stof(text, &pos); }
    catch (const std::exception&) { throw std::runtime_error(name + " must be between 0 and 1"); }
    if (pos != text.size() || !std::isfinite(value) || value < 0.0f || value > 1.0f)
        throw std::runtime_error(name + " must be between 0 and 1");
    return value;
}

}  // namespace ov203
