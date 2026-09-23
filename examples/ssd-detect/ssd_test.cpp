// ssd_test: device-free unit tests for the SSD post-processing logic
// (ssd_postprocess.hpp) plus the shared half.hpp round-trip.  Built and run in
// the Docker build stage (no MYRIAD needed).  Exit code 0 = all pass.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../half.hpp"
#include "ssd_postprocess.hpp"

static int failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) {                                                             \
            std::printf("PASS  %s\n", msg);                                    \
        } else {                                                                \
            std::printf("FAIL  %s\n", msg);                                    \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

static bool nearEq(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

// Build one DetectionOutput row: (image_id, class, score, x1, y1, x2, y2)
static void row(float* p, int imageId, int cls, float score,
                float x1, float y1, float x2, float y2) {
    p[0] = static_cast<float>(imageId);
    p[1] = static_cast<float>(cls);
    p[2] = score;
    p[3] = x1;
    p[4] = y1;
    p[5] = x2;
    p[6] = y2;
}

int main() {
    using namespace ssd;

    // 1. box conversion to original-image pixels, 640x480
    {
        std::vector<float> buf(7);
        row(&buf[0], 0, 1, 0.9f, 0.1f, 0.2f, 0.8f, 0.9f);
        std::vector<Detection> d = parseDetections(&buf[0], 1, 0.5f, 10, 640, 480);
        CHECK(d.size() == 1, "single row parses");
        if (d.size() == 1) {
            CHECK(d[0].class_id == 1 && d[0].x1 == 64 && d[0].y1 == 96 &&
                      d[0].x2 == 512 && d[0].y2 == 432,
                  "normalized (0.1,0.2,0.8,0.9) @640x480 -> (64,96,512,432)");
        }
    }

    // 2. confidence threshold uses >= (keep at exactly minConf)
    {
        std::vector<float> buf(4 * 7);
        row(&buf[0], 0, 1, 0.2f, 0.0f, 0.0f, 0.5f, 0.5f);   // below
        row(&buf[7], 0, 2, 0.49f, 0.0f, 0.0f, 0.5f, 0.5f);  // below
        row(&buf[14], 0, 3, 0.5f, 0.0f, 0.0f, 0.5f, 0.5f);  // exactly at threshold: kept
        row(&buf[21], 0, 4, 0.91f, 0.0f, 0.0f, 0.9f, 0.9f); // above
        std::vector<Detection> d = parseDetections(&buf[0], 4, 0.5f, 10, 640, 480);
        CHECK(d.size() == 2, "scores {0.2,0.49,0.50,0.91} at min 0.5 -> 2 kept (>= convention)");
        if (d.size() == 2) {
            CHECK(d[0].confidence > d[1].confidence, "results sorted by descending confidence");
            CHECK(d[0].class_id == 4 && d[1].class_id == 3, "order is 0.91 (class 4) then 0.50 (class 3)");
        }
    }

    // 3. padded rows (image_id < 0) and degenerate boxes are dropped
    {
        std::vector<float> buf(4 * 7);
        row(&buf[0], -1, 1, 0.99f, 0.1f, 0.1f, 0.5f, 0.5f);  // padded
        row(&buf[7], 0, 2, 0.9f, 0.5f, 0.2f, 0.5f, 0.9f);    // x2 == x1: degenerate
        row(&buf[14], 0, 3, 0.8f, 0.0f, 0.5f, 0.7f, 0.5f);   // y2 == y1: degenerate
        row(&buf[21], 0, 4, 0.7f, 0.0f, 0.0f, 1.0f, 1.0f);   // valid full frame
        std::vector<Detection> d = parseDetections(&buf[0], 4, 0.5f, 10, 100, 100);
        CHECK(d.size() == 1, "padded + 2 degenerate rows dropped");
        if (d.size() == 1)
            CHECK(d[0].x1 == 0 && d[0].y1 == 0 && d[0].x2 == 100 && d[0].y2 == 100,
                  "full-frame box clamps to (0,0,100,100)");
    }

    // 4. clamping of out-of-range normalized coords
    {
        std::vector<float> buf(7);
        row(&buf[0], 0, 1, 0.9f, -0.2f, -0.2f, 1.2f, 1.2f);
        std::vector<Detection> d = parseDetections(&buf[0], 1, 0.5f, 10, 100, 100);
        CHECK(d.size() == 1 && d[0].x1 == 0 && d[0].y1 == 0 && d[0].x2 == 100 && d[0].y2 == 100,
              "out-of-range coords clamp into [0, dim]");
    }

    // 5. maxDetections cuts the tail (highest confidence survives)
    {
        std::vector<float> buf(5 * 7);
        for (int i = 0; i < 5; ++i)
            row(&buf[i * 7], 0, 10 + i, 0.6f + i * 0.05f, 0.0f, 0.0f, 0.5f, 0.5f);
        std::vector<Detection> d = parseDetections(&buf[0], 5, 0.5f, 2, 100, 100);
        CHECK(d.size() == 2 && d[0].class_id == 14 && d[1].class_id == 13,
              "max_detections=2 keeps the two highest-confidence rows");
    }

    // 6. COCO label mapping, including sparse ids and the unknown-id fallback
    {
        std::map<int, std::string> labels;
        labels[1] = "person";
        labels[3] = "car";
        labels[13] = "bear";
        labels[90] = "toothbrush";
        CHECK(labelFor(labels, 1) == "person", "label 1 -> person");
        CHECK(labelFor(labels, 3) == "car", "label 3 -> car");
        CHECK(labelFor(labels, 13) == "bear", "label 13 -> bear");
        CHECK(labelFor(labels, 90) == "toothbrush", "label 90 -> toothbrush");
        CHECK(labelFor(labels, 12) == "class_12", "unknown id 12 -> class_12 fallback");
    }

    // 7. shared half.hpp: sign bit survives the round trip (the 2024 bug)
    {
        const float neg = -0.125f;
        CHECK(halfToFloat(floatToHalf(neg)) == neg, "negative value survives float->half->float");
        CHECK(halfToFloat(floatToHalf(-100.0f)) == -100.0f, "negative 100 round-trips exactly");
        CHECK(floatToHalf(-0.125f) == 0xB000, "floatToHalf(-0.125) == 0xB000");
    }

    if (failures) {
        std::printf("\n%d check(s) FAILED\n", failures);
        return 1;
    }
    std::printf("\nssd_test: all checks passed\n");
    return 0;
}
