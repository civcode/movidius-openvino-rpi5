// test_half - unit test for the float<->half conversion used by
// mobilenet_server.cpp (half.hpp).
//
// No Inference Engine or device required; built and run in the Docker build
// stage as a guard. It exists because a past version of floatToHalf dropped
// the sign bit in two return paths, silently flipping every negative input
// value to positive and cutting top-1 accuracy roughly in half.
#include "half.hpp"  // shared IEEE-754 f16<->f32 conversion (unit-tested in examples/webcam/test_half.cpp)

#include <cstdio>
#include <cstring>

namespace {

struct Case {
    float in;
    uint16_t out;
};

int fails = 0;

void expectHalf(float in, uint16_t want) {
    const uint16_t got = floatToHalf(in);
    if (got != want) {
        std::fprintf(stderr, "FAIL floatToHalf(%g) = 0x%04x, want 0x%04x\n",
                     (double)in, got, want);
        ++fails;
    }
}

// halfToFloat is exact for every half value, so converting back must be a
// fixed point: floatToHalf(halfToFloat(h)) == h.
void expectRoundTrip(float in) {
    const uint16_t h = floatToHalf(in);
    const float back = halfToFloat(h);
    if (floatToHalf(back) != h) {
        std::fprintf(stderr, "FAIL round-trip %g -> 0x%04x -> %g\n", (double)in, h,
                     (double)back);
        ++fails;
    }
}

void expectHalfToFloat(uint16_t h, float want) {
    const float got = halfToFloat(h);
    uint32_t wb, gb;
    std::memcpy(&wb, &want, sizeof wb);
    std::memcpy(&gb, &got, sizeof gb);
    if (wb != gb) {
        std::fprintf(stderr, "FAIL halfToFloat(0x%04x) = %g, want %g\n", h, (double)got,
                     (double)want);
        ++fails;
    }
}

}  // namespace

int main() {
    // expected bits computed with an independent implementation (Python struct
    // '<e', round-to-nearest-even)
    const Case cases[] = {
        {-5.307483673f, 0xC54Fu},  // negative normal (regression: was 0x454F)
        {-0.0f, 0x8000u},
        {0.0f, 0x0000u},
        {1.0f, 0x3C00u},
        {-1.0f, 0xBC00u},
        {65504.0f, 0x7BFFu},
        {-65504.0f, 0xFBFFu},
        {-1e-8f, 0x8000u},        // subnormal rounding to -0
        {5.9604644775e-8f, 0x0001u},  // smallest positive subnormal
        {0.5f, 0x3800u},
        {-6.103515625e-5f, 0x8400u},  // smallest negative normal
        {3.14159265358979f, 0x4248u},
    };
    for (const Case& c : cases) expectHalf(c.in, c.out);

    for (const Case& c : cases) expectRoundTrip(c.in);
    expectRoundTrip(12345.0f);
    expectRoundTrip(-0.001f);

    expectHalfToFloat(0xC54Fu, -5.30859375f);
    expectHalfToFloat(0x0001u, 5.9604644775e-8f);
    expectHalfToFloat(0x8000u, -0.0f);
    expectHalfToFloat(0x7BFFu, 65504.0f);

    if (fails) {
        std::fprintf(stderr, "test_half: %d failure(s)\n", fails);
        return 1;
    }
    std::printf("test_half: all checks passed\n");
    return 0;
}
