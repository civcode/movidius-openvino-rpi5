// half.hpp - IEEE-754 binary16 <-> binary32 conversion, shared by
// examples/webcam/mobilenet_server.cpp, mobilenet-test/main.cpp,
// smoke-test/main.cpp and examples/webcam/test_half.cpp.
//
// History: mobilenet_server.cpp originally carried its own copy of these
// functions with the sign bit missing from two return paths, which silently
// flipped every negative input value to positive (halving tensor accuracy).
// test_half.cpp guards against a regression.
#pragma once

#include <cstdint>
#include <cstring>

// IEEE-754 binary32 -> binary16, round-to-nearest-even, inf on overflow.
inline uint16_t floatToHalf(float value) {
    uint32_t f = 0;
    std::memcpy(&f, &value, sizeof(f));

    const uint32_t sign = (f >> 16) & 0x8000u;
    const uint32_t fexp = (f >> 23) & 0xFFu;
    uint32_t mant = f & 0x7FFFFFu;

    if (fexp == 0u && mant == 0u) return static_cast<uint16_t>(sign);            // +-0
    if (fexp == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));

    int32_t exp = static_cast<int32_t>(fexp) - 127 + 15;                          // half biased
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);                 // overflow

    if (exp <= 0) {                                                               // subnormal half
        if (exp < -10) return static_cast<uint16_t>(sign);                        // rounds to 0
        mant |= 0x800000u;                                                        // implicit 1
        const int shift = 14 - exp;
        uint32_t halfMant = mant >> shift;
        const uint32_t rest = mant & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rest > half || (rest == half && (halfMant & 1u))) ++halfMant;
        return static_cast<uint16_t>(sign | halfMant);
    }

    uint32_t halfMant = mant >> 13;
    const uint32_t rest = mant & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (halfMant & 1u))) {                 // tie: to even
        ++halfMant;
        if (halfMant == 0x400u) {
            halfMant = 0;
            ++exp;
            if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | halfMant);
}

// IEEE-754 binary16 -> binary32 (exact; binary16 is a subset of binary32).
inline float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t frac = h & 0x3FFu;

    uint32_t bits = 0;
    if (exp == 0u) {
        if (frac == 0u) {
            bits = sign;
        } else {                                                                   // subnormal half
            uint32_t f = frac;
            int shifts = 0;
            while ((f & 0x400u) == 0u) {
                f <<= 1;
                ++shifts;
            }
            bits = sign | (static_cast<uint32_t>(113 - shifts) << 23) | ((f & 0x3FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (frac << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (frac << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}
