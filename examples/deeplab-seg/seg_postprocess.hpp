// seg_postprocess.hpp
//
// Device-free postprocessing for the DeepLabV3 (Pascal VOC) segmentation
// IR.  Two output layouts are supported so the postprocessor does not
// assume a layout it has not checked:
//
//   * direct class ids : rows of length  H*W  (the ArgMax'd IR)
//   * logits           : rows of length  C*H*W (C == numClasses)
//
// and the per-pixel mask is resized to the client's original image size
// with plain nearest-neighbour sampling (a segmentation mask must not be
// blurred by bilinear averaging).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "half.hpp"

namespace seg {

// ---------------------------------------------------------------------------
// class map (one class id per pixel)
// ---------------------------------------------------------------------------

struct ClassMap {
    int w = 0;
    int h = 0;
    std::vector<uint8_t> ids;   // ids.size() == w * h
};

// Interpret a raw output blob as a class map.  `elems` is the total number
// of output elements; the layout is decided from it:
//
//   elems == H*W        -> direct class ids (the ArgMax'd IR)
//   elems == C*H*W      -> per-pixel class logits, C == numClasses,
//                          argmax over the class axis
//
// Any other shape, or any id outside [0, numClasses), is a hard error.

inline ClassMap classMapFromRaw(const float* rows, size_t elems,
                                int C, int H, int W, int numClasses,
                                const std::string& model,
                                const std::string& outName) {
    ClassMap m;
    m.w = W;
    m.h = H;
    m.ids.assign((size_t)W * (size_t)H, 0);

    if (elems == (size_t)H * (size_t)W) {
        for (size_t i = 0; i < elems; ++i) {
            int v = (int)rows[i];
            if (v < 0 || v >= numClasses)
                throw std::runtime_error(
                    "class id out of range in " + model + " output '" +
                    outName + "': " + std::to_string(v));
            m.ids[i] = (uint8_t)v;
        }
    } else if (elems == (size_t)C * (size_t)H * (size_t)W) {
        const size_t plane = (size_t)H * (size_t)W;
        for (size_t y = 0; y < (size_t)H; ++y) {
            for (size_t x = 0; x < (size_t)W; ++x) {
                int best = -1;
                float bestV = -1e30f;
                for (int c = 0; c < C; ++c) {
                    float v = rows[(size_t)c * plane + y * (size_t)W + x];
                    if (v > bestV) {
                        bestV = v;
                        best = c;
                    }
                }
                if (best < 0 || best >= numClasses)
                    throw std::runtime_error(
                        "argmax class out of range in " + model + " output '" +
                        outName + "': " + std::to_string(best));
                m.ids[y * (size_t)W + x] = (uint8_t)best;
            }
        }
    } else {
        throw std::runtime_error(
            "unsupported output shape for " + model + " output '" + outName +
            "' (" + std::to_string(C) + ", " + std::to_string(H) + ", " +
            std::to_string(W) + ", " + std::to_string(elems) + " elements): "
            "expected H*W direct ids or C*H*W logits");
    }
    return m;
}

// ---------------------------------------------------------------------------
// nearest-neighbour mask resize (no averaging: masks stay crisp)
// ---------------------------------------------------------------------------

// Resize `src` (sw x sh) into `dst` (dw x dh) with nearest sampling.
inline void nearestResize(const uint8_t* src, int sw, int sh,
                          uint8_t* dst, int dw, int dh) {
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0)
        throw std::runtime_error("nearestResize: bad dimensions");
    for (int y = 0; y < dh; ++y) {
        int sy = (y * sh) / dh;
        if (sy >= sh)
            sy = sh - 1;
        for (int x = 0; x < dw; ++x) {
            int sx = (x * sw) / dw;
            if (sx >= sw)
                sx = sw - 1;
            dst[(size_t)y * (size_t)dw + x] = src[(size_t)sy * (size_t)sw + sx];
        }
    }
}

// ---------------------------------------------------------------------------
// raw output blob decoding
// ---------------------------------------------------------------------------

// The DeepLabV3 IR declares its ArgMax output as I32 (per-pixel class ids),
// but the blob a backend hands back may be I16/I32 or an FP16/FP32 copy of
// the same values.  The byte size alone cannot distinguish I32 from FP32
// (int 12 re-read as a float is a ~8.8e-44 denormal, i.e. silently rounds to
// class 0), so the decoders are tried in order and the first interpretation
// that yields plausible class ids wins:
//
//   4 bytes/element : float values (integer-ish, 0..numClasses-1)
//                     -> int32 values (0..numClasses-1)
//   2 bytes/element : uint16 values (0..numClasses-1)
//                     -> half values (integer-ish, 0..numClasses-1)
//
// Returns false when no interpretation works.
inline bool interpretClassIds(const uint8_t* data, size_t elems, int elemBytes,
                              int numClasses, std::vector<float>& ids) {
    ids.assign(elems, 0.0f);
    auto plausible = [&](const std::vector<float>& v) {
        for (float x : v) {
            if (x < 0.0f || x >= (float)numClasses)
                return false;
            if (std::fabs(x - std::round(x)) > 1e-3f)
                return false;
            // a re-read int32 id 1..20 is a denormal ~1e-43 float: round() is
            // 0 for it, so reject anything between 0 and 1 explicitly
            if (x > 0.0f && x < 0.5f)
                return false;
        }
        return true;
    };
    if (elemBytes == 4) {
        std::memcpy(ids.data(), data, elems * sizeof(float));
        if (plausible(ids))
            return true;
        std::vector<int32_t> raw(elems);
        std::memcpy(raw.data(), data, elems * sizeof(int32_t));
        for (size_t i = 0; i < elems; ++i) {
            if (raw[i] < 0 || raw[i] >= numClasses)
                return false;
            ids[i] = (float)raw[i];
        }
        return true;
    }
    if (elemBytes == 2) {
        bool ok = true;
        for (size_t i = 0; i < elems; ++i) {
            uint16_t v = (uint16_t)data[2 * i] | ((uint16_t)data[2 * i + 1] << 8);
            if (v >= (uint16_t)numClasses) {
                ok = false;
                break;
            }
            ids[i] = (float)v;
        }
        if (ok)
            return true;
        const uint16_t* halves = (const uint16_t*)data;
        for (size_t i = 0; i < elems; ++i)
            ids[i] = halfToFloat(halves[i]);
        return plausible(ids);
    }
    return false;
}

// ---------------------------------------------------------------------------
// Pascal VOC labels
// ---------------------------------------------------------------------------

struct Labels {
    std::vector<std::string> names;  // index = class id
};

// Label files are "id<whitespace>name" lines (0 = background .. 20); the
// first field is the id and the rest of the line is the name, so tab- and
// space-separated files (and multi-word names) all work.
inline Labels loadLabels(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open labels file: " + path);
    std::string line;
    std::vector<std::string> names;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty() || line[0] == '#')
            continue;
        size_t sep = line.find_first_of(" \t");
        if (sep == std::string::npos)
            continue;
        // skip runs of whitespace so the name starts clean
        size_t nameStart = sep;
        while (nameStart < line.size() &&
               (line[nameStart] == ' ' || line[nameStart] == '\t'))
            ++nameStart;
        if (nameStart >= line.size())
            continue;
        int id = 0;
        try {
            id = std::stoi(line.substr(0, sep));
        } catch (...) {
            continue;
        }
        if (id < 0 || id >= 1024)
            continue;
        if ((int)names.size() <= id)
            names.resize((size_t)id + 1);
        names[(size_t)id] = line.substr(nameStart);
    }
    if (names.empty())
        throw std::runtime_error("no labels parsed from " + path);
    return Labels{names};
}

inline std::string labelFor(const Labels& labels, int id) {
    if (id >= 0 && id < (int)labels.names.size() && !labels.names[(size_t)id].empty())
        return labels.names[(size_t)id];
    return "class_" + std::to_string(id);
}

// ---------------------------------------------------------------------------
// class histogram (console output, sorted by pixel count desc)
// ---------------------------------------------------------------------------

struct ClassCount {
    int id = 0;
    uint64_t pixels = 0;
};

inline std::vector<ClassCount> classHistogram(const ClassMap& m) {
    std::vector<uint64_t> counts(256, 0);
    for (uint8_t v : m.ids)
        counts[v]++;
    std::vector<ClassCount> out;
    for (size_t i = 0; i < counts.size(); ++i)
        if (counts[i])
            out.push_back(ClassCount{(int)i, counts[i]});
    std::sort(out.begin(), out.end(),
              [](const ClassCount& a, const ClassCount& b) {
                  if (a.pixels != b.pixels)
                      return a.pixels > b.pixels;
                  return a.id < b.id;
              });
    return out;
}

}  // namespace seg
