// seg_test.cpp
//
// Device-free tests for the DeepLabV3 segmentation postprocessor and the
// shared float<->half header.  Runs in the Docker build stage (no OpenVINO
// inference, no MYRIAD access).
//
//   1. half.hpp roundtrip
//   2. direct class map [1,H,W]  -> HxW mask
//   3. logits [1,C,H,W]          -> argmax HxW mask
//   4. nearest-neighbour resize preserves only present classes
//   5. Pascal VOC labels (0 background, 7 car, 15 person)
//   6. unknown class id falls back to "class_<id>"
//   7. out-of-range class id is a hard error
//   8. unsupported output shape is a hard error
//
// Exit 0 on success, 1 on failure.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../half.hpp"
#include "seg_postprocess.hpp"

static int failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) {                                                             \
            printf("ok   %s\n", msg);                                          \
        } else {                                                                \
            printf("FAIL %s\n", msg);                                          \
            failures++;                                                         \
        }                                                                       \
    } while (0)

static bool near(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

// 1. half.hpp roundtrip -----------------------------------------------------
static void testHalf() {
    struct {
        float in;
        float want;
    } cases[] = {
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {-1.0f, -1.0f},
        {0.5f, 0.5f},
        {-0.25f, -0.25f},
        {1.0f / 127.5f, 1.0f / 127.5f},  // the DeepLabV3 scale (exact in f16)
        {127.5f, 127.5f},
        {255.0f, 255.0f},
        {-0.0f, 0.0f},
    };
    for (const auto& c : cases) {
        uint16_t h = floatToHalf(c.in);
        float back = halfToFloat(h);
        CHECK(near(back, c.want),
              (std::string("half roundtrip ") + std::to_string(c.in)).c_str());
    }
    // f16 has no infinities; large values clamp, not NaN.
    uint16_t big = floatToHalf(1e9f);
    CHECK(halfToFloat(big) > 60000.0f, "half large value clamps");
}

// 2. direct class map [1,H,W] -----------------------------------------------
static void testDirectClassMap() {
    // 2x3 mask: row0 = 0 7 20 ; row1 = 15 0 7  (H=2, W=3)
    std::vector<float> rows = {0, 7, 20, 15, 0, 7};
    seg::ClassMap m = seg::classMapFromRaw(rows.data(), rows.size(), 1, 2, 3, 21,
                                            "test", "out");
    CHECK(m.w == 3 && m.h == 2 && m.ids.size() == 6, "direct shape");
    CHECK(m.ids[0] == 0 && m.ids[1] == 7 && m.ids[2] == 20 && m.ids[3] == 15 &&
              m.ids[4] == 0 && m.ids[5] == 7,
          "direct values");
}

// 3. logits [1,C,H,W] -------------------------------------------------------
static void testLogits() {
    // C=3, H=2, W=2.  Row-major: class c plane first.
    // plane0 = { 5, 1,  2, 9 }
    // plane1 = { 1, 4,  8, 2 }
    // plane2 = { 3, 9,  1, 1 }
    std::vector<float> rows = {5, 1, 2, 9, 1, 4, 8, 2, 3, 9, 1, 1};
    seg::ClassMap m = seg::classMapFromRaw(rows.data(), rows.size(), 3, 2, 2,
                                            3, "test", "out");
    CHECK(m.w == 2 && m.h == 2, "logits shape");
    // argmax per pixel:
    // (0,0): max(5,1,3)=5 -> c0
    // (0,1): max(1,4,9)=9 -> c2
    // (1,0): max(2,8,1)=8 -> c1
    // (1,1): max(9,2,1)=9 -> c0
    CHECK(m.ids[0] == 0 && m.ids[1] == 2 && m.ids[2] == 1 && m.ids[3] == 0,
          "logits argmax values");
}

// 4. nearest-neighbour resize ------------------------------------------------
static void testNearestResize() {
    // source 2x2:  0 1
    //              2 3
    std::vector<uint8_t> src = {0, 1, 2, 3};
    std::vector<uint8_t> dst(4 * 4);
    seg::nearestResize(src.data(), 2, 2, dst.data(), 4, 4);
    // expected:
    // 0 0 1 1
    // 0 0 1 1
    // 2 2 3 3
    // 2 2 3 3
    const uint8_t want[16] = {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3};
    bool all = true;
    for (int i = 0; i < 16; ++i)
        if (dst[i] != want[i])
            all = false;
    CHECK(all, "nearest resize 2x2 -> 4x4");

    // upscaling a 3x3 mask to 4x4: only values {0,1,2} may appear
    std::vector<uint8_t> src3 = {0, 1, 2, 1, 2, 0, 2, 0, 1};
    std::vector<uint8_t> dst4(16);
    seg::nearestResize(src3.data(), 3, 3, dst4.data(), 4, 4);
    bool valid = true;
    for (uint8_t v : dst4)
        if (v > 2)
            valid = false;
    CHECK(valid, "nearest resize only present classes");
}

// 5-6. labels ------------------------------------------------------------------
static void testLabels() {
    std::string path = "/tmp/seg_test_pascal.txt";
    {
        FILE* f = fopen(path.c_str(), "w");
        fprintf(f, "0\tbackground\n7\tcar\n15\tperson\n20\ttvmonitor\n");
        fclose(f);
    }
    seg::Labels l = seg::loadLabels(path);
    CHECK(l.names.size() >= 21, "labels size");
    CHECK(l.names[0] == "background", "label 0 background");
    CHECK(l.names[7] == "car", "label 7 car");
    CHECK(l.names[15] == "person", "label 15 person");
    CHECK(l.names[20] == "tvmonitor", "label 20 tvmonitor");
    CHECK(seg::labelFor(l, 42) == "class_42", "unknown id fallback");
}

// 7. out-of-range class id -----------------------------------------------------
static void testOutOfRange() {
    bool threw = false;
    try {
        std::vector<float> rows(6, 0);
        rows[2] = 25;  // >= numClasses (21)
        seg::classMapFromRaw(rows.data(), rows.size(), 1, 2, 3, 21, "test",
                             "out");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "out-of-range class id throws");
}

// 8. unsupported shape -----------------------------------------------------------
static void testBadShape() {
    bool threw = false;
    try {
        // 34 elements: neither H*W=6 nor C*H*W=18
        std::vector<float> rows(34, 1);
        seg::classMapFromRaw(rows.data(), rows.size(), 3, 2, 3, 21, "test",
                             "out");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "unsupported shape throws");
}

// 9. raw blob decoding (I32/FP32/I16/FP16) -------------------------------------
static void testInterpretClassIds() {
    std::vector<float> ids;

    // FP32 class ids (a backend that copies the I32 output to float)
    const float f32[3] = {0.0f, 12.0f, 3.0f};
    CHECK(seg::interpretClassIds((const uint8_t*)f32, 3, 4, 21, ids),
          "fp32 ids decode");
    CHECK(ids[0] == 0.0f && ids[1] == 12.0f && ids[2] == 3.0f,
          "fp32 ids values");

    // I32 class ids (the IR's declared output precision) - the case that
    // used to be silently read as denormal floats and rounded to 0
    const int32_t i32[3] = {0, 12, 3};
    CHECK(seg::interpretClassIds((const uint8_t*)i32, 3, 4, 21, ids),
          "i32 ids decode");
    CHECK(ids[0] == 0.0f && ids[1] == 12.0f && ids[2] == 3.0f,
          "i32 ids values");

    // I32 with an out-of-range value: neither interpretation works
    const int32_t bad32[2] = {12, 99};
    CHECK(!seg::interpretClassIds((const uint8_t*)bad32, 2, 4, 21, ids),
          "i32 out-of-range rejected");

    // I16 class ids
    const int16_t i16[2] = {0, 20};
    CHECK(seg::interpretClassIds((const uint8_t*)i16, 2, 2, 21, ids),
          "i16 ids decode");
    CHECK(ids[0] == 0.0f && ids[1] == 20.0f, "i16 ids values");

    // FP16 class ids (half bit patterns would be huge uint16 values, so the
    // uint16 pass must fail and the half pass must succeed)
    const uint16_t f16[2] = {floatToHalf(0.0f), floatToHalf(12.0f)};
    CHECK(seg::interpretClassIds((const uint8_t*)f16, 2, 2, 21, ids),
          "fp16 ids decode");
    CHECK(ids[0] == 0.0f && ids[1] == 12.0f, "fp16 ids values");

    // unsupported element size
    const uint8_t one[2] = {0, 0};
    CHECK(!seg::interpretClassIds(one, 1, 1, 21, ids),
          "1-byte elements rejected");
}

int main() {
    testHalf();
    testDirectClassMap();
    testLogits();
    testNearestResize();
    testLabels();
    testOutOfRange();
    testBadShape();
    testInterpretClassIds();
    if (failures) {
        printf("%d test(s) FAILED\n", failures);
        return 1;
    }
    printf("all seg tests passed\n");
    return 0;
}
