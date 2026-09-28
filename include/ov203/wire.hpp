#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace ov203 {

inline bool hostIsLittleEndian() {
    const uint16_t value = 1;
    return *reinterpret_cast<const unsigned char*>(&value) == 1;
}

inline uint32_t byteSwap32(uint32_t value) {
    return ((value & 0x000000ffu) << 24) |
           ((value & 0x0000ff00u) << 8) |
           ((value & 0x00ff0000u) >> 8) |
           ((value & 0xff000000u) >> 24);
}

// A bulk MobileNet request is read directly into float storage. On little-endian
// hosts (all currently supported targets) this is a no-op; on a big-endian host
// it converts the protocol's little-endian IEEE-754 bytes in place.
inline void littleEndianFloat32ToNativeInPlace(float* values, std::size_t count) {
    if (hostIsLittleEndian()) return;
    for (std::size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, values + i, sizeof(bits));
        bits = byteSwap32(bits);
        std::memcpy(values + i, &bits, sizeof(bits));
    }
}

inline void nativeFloat32ToLittleEndianBytes(const float* values,
                                              unsigned char* out,
                                              std::size_t count) {
    if (hostIsLittleEndian()) {
        std::memcpy(out, values, count * sizeof(float));
        return;
    }
    for (std::size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, values + i, sizeof(bits));
        out[4 * i + 0] = static_cast<unsigned char>(bits & 0xffu);
        out[4 * i + 1] = static_cast<unsigned char>((bits >> 8) & 0xffu);
        out[4 * i + 2] = static_cast<unsigned char>((bits >> 16) & 0xffu);
        out[4 * i + 3] = static_cast<unsigned char>((bits >> 24) & 0xffu);
    }
}

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
