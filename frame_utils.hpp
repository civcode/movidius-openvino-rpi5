// frame_utils.hpp - shared PPM I/O and bilinear resize for the C++ example
// apps (ssd_detect, seg_detect).
//
// Both detection and segmentation read P6 PPM frames and bilinear-resize the
// interleaved RGB buffer to the model's input size; this is the single
// implementation of both (the per-app copies have been removed).
//
// Header-only, no OpenVINO or device dependencies.
#ifndef FRAME_UTILS_HPP
#define FRAME_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace frameutils {

// P6 PPM in RGB channel order (our PPMs are written by Pillow / PIL, whose
// PPM writer uses RGB).
struct Ppm {
    int w = 0;
    int h = 0;
    std::vector<uint8_t> rgb;  // w*h*3, row-major, RGB
};

// Read a P6 PPM (maxval 255).  Throws std::runtime_error on any malformed
// header or truncated body.
inline Ppm readPpm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::string magic;
    if (!(in >> magic))
        throw std::runtime_error("not a P6 PPM: " + path);
    if (magic != "P6")
        throw std::runtime_error("expected P6, got " + magic + ": " + path);
    auto skipComments = [&]() {
        int c;
        while ((c = in.peek()) == '#') {
            std::string line;
            std::getline(in, line);
        }
    };
    int w, h, maxv;
    skipComments();
    if (!(in >> w)) throw std::runtime_error("bad PPM header in " + path);
    skipComments();
    if (!(in >> h)) throw std::runtime_error("bad PPM header in " + path);
    skipComments();
    if (!(in >> maxv)) throw std::runtime_error("bad PPM header in " + path);
    if (w <= 0 || h <= 0 || maxv != 255)
        throw std::runtime_error("bad PPM dimensions: " + std::to_string(w) +
                                 "x" + std::to_string(h) + " maxval " +
                                 std::to_string(maxv));
    in.get();  // single whitespace after maxval
    Ppm p;
    p.w = w;
    p.h = h;
    p.rgb.assign(static_cast<std::size_t>(w) * h * 3, 0);
    in.read(reinterpret_cast<char*>(p.rgb.data()),
            static_cast<std::streamsize>(p.rgb.size()));
    if (in.gcount() != static_cast<std::streamsize>(p.rgb.size()))
        throw std::runtime_error("truncated PPM body: " + path);
    return p;
}

// Bilinear resize of an interleaved RGB buffer to dw x dh.
// Half-pixel sampling (matches OpenCV INTER_LINEAR, the reference for the
// model's in-graph resize).  When upscaling (dst larger than src in a
// dimension) the first/last rows sample partly outside the source, so
// individual weights can exceed 1 and the result can overshoot 255 on
// saturated pixels; the value is clamped before the uint8 conversion so the
// cast cannot wrap.
inline void bilinearResize(const uint8_t* src, int sw, int sh,
                           std::vector<uint8_t>& dst, int dw, int dh) {
    dst.assign(static_cast<std::size_t>(dw) * dh * 3, 0);
    const float xRatio = static_cast<float>(sw) / dw;
    const float yRatio = static_cast<float>(sh) / dh;
    for (int y = 0; y < dh; ++y) {
        const float fy = (y + 0.5f) * yRatio - 0.5f;
        const int y0 = std::max(0, std::min(sh - 2, static_cast<int>(std::floor(fy))));
        const float wy = fy - y0;
        const int y1 = y0 + 1;
        for (int x = 0; x < dw; ++x) {
            const float fx = (x + 0.5f) * xRatio - 0.5f;
            const int x0 = std::max(0, std::min(sw - 2, static_cast<int>(std::floor(fx))));
            const float wx = fx - x0;
            const int x1 = x0 + 1;
            for (int c = 0; c < 3; ++c) {
                const float v00 = src[static_cast<std::size_t>(y0) * sw * 3 + x0 * 3 + c];
                const float v01 = src[static_cast<std::size_t>(y0) * sw * 3 + x1 * 3 + c];
                const float v10 = src[static_cast<std::size_t>(y1) * sw * 3 + x0 * 3 + c];
                const float v11 = src[static_cast<std::size_t>(y1) * sw * 3 + x1 * 3 + c];
                float v = (v00 * (1 - wx) + v01 * wx) * (1 - wy) +
                          (v10 * (1 - wx) + v11 * wx) * wy;
                v = std::min(255.0f, std::max(0.0f, v));
                dst[static_cast<std::size_t>(y) * dw * 3 + x * 3 + c] =
                    static_cast<uint8_t>(v + 0.5f);
            }
        }
    }
}

}  // namespace frameutils

#endif  // FRAME_UTILS_HPP
