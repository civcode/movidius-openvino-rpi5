#pragma once

#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace ov203 {

inline uint16_t readLe16(std::istream& in) {
    unsigned char b[2];
    in.read(reinterpret_cast<char*>(b), 2);
    if (!in) throw std::runtime_error("unexpected EOF reading uint16");
    return static_cast<uint16_t>(b[0]) |
           (static_cast<uint16_t>(b[1]) << 8);
}

inline uint32_t readLe32(std::istream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("unexpected EOF reading uint32");
    return static_cast<uint32_t>(b[0]) |
           (static_cast<uint32_t>(b[1]) << 8) |
           (static_cast<uint32_t>(b[2]) << 16) |
           (static_cast<uint32_t>(b[3]) << 24);
}

inline void writeLe16(std::ostream& out, uint16_t value) {
    const unsigned char b[2] = {
        static_cast<unsigned char>(value & 0xffu),
        static_cast<unsigned char>((value >> 8) & 0xffu),
    };
    out.write(reinterpret_cast<const char*>(b), 2);
}

inline void writeLe32(std::ostream& out, uint32_t value) {
    const unsigned char b[4] = {
        static_cast<unsigned char>(value & 0xffu),
        static_cast<unsigned char>((value >> 8) & 0xffu),
        static_cast<unsigned char>((value >> 16) & 0xffu),
        static_cast<unsigned char>((value >> 24) & 0xffu),
    };
    out.write(reinterpret_cast<const char*>(b), 4);
}

inline float readLeFloat32(std::istream& in) {
    const uint32_t bits = readLe32(in);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline void writeLeFloat32(std::ostream& out, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    writeLe32(out, bits);
}

}  // namespace ov203
