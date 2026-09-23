// Post-processing for SSDLite (and other 2020.3 opset1 `DetectionOutput` IRs).
//
// The OpenVINO 2020.3 `DetectionOutput` op runs NMS internally and writes
// keep_top_k rows, each laid out as
//
//     [image_id, class, score, xmin, ymin, xmax, ymax]      (normalized 0..1)
//
// with padded (unused) rows carrying image_id == -1.  Everything the
// detection client does after reading those rows lives here so it can be
// exercised by ssd_test.cpp without hardware:
//
//   * padded rows are skipped (image_id < 0)
//   * rows below the confidence threshold are dropped (kept at score >= minConf)
//   * normalized corners are converted to original-image pixels and clamped
//   * degenerate boxes (x2 <= x1 or y2 <= y1) are rejected
//   * the result is sorted by descending confidence and cut to maxDetections
#ifndef SSD_POSTPROCESS_HPP
#define SSD_POSTPROCESS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ssd {

struct Detection {
    int class_id = -1;
    float confidence = 0.0f;
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;  // original-image pixels, clamped, x2 > x1, y2 > y1
};

inline int clampI(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// round half to the nearest int, correct for negatives (plain truncation of
// -0.5f is 0, which would accept padded rows with image_id -1)
inline int roundToI(float v) { return static_cast<int>(std::floor(v + 0.5f)); }

inline int toPixel(float norm, int dim) {
    // round to the nearest pixel, then clamp into [0, dim]
    int v = static_cast<int>(norm * static_cast<float>(dim) + 0.5f);
    return clampI(v, 0, dim);
}

inline std::vector<Detection> parseDetections(const float* rows, std::size_t rowCount,
                                              float minConf, int maxDetections,
                                              int origW, int origH) {
    std::vector<Detection> out;
    out.reserve(64);
    for (std::size_t i = 0; i < rowCount; ++i) {
        const float* r = rows + i * 7;
        if (roundToI(r[0]) < 0) continue;  // padded row
        if (r[2] < minConf) continue;                     // below threshold (kept at >=)
        Detection d;
        d.class_id = roundToI(r[1]);
        d.confidence = r[2];
        d.x1 = toPixel(r[3], origW);
        d.y1 = toPixel(r[4], origH);
        d.x2 = toPixel(r[5], origW);
        d.y2 = toPixel(r[6], origH);
        if (d.x2 <= d.x1 || d.y2 <= d.y1) continue;       // degenerate box
        out.push_back(d);
    }
    std::stable_sort(out.begin(), out.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    if (static_cast<std::size_t>(maxDetections) < out.size())
        out.resize(static_cast<std::size_t>(maxDetections));
    return out;
}

// Label files are "id<whitespace>name" lines (0 = background, sparse COCO
// ids for 1..90); the first field is the id and the rest of the line is the
// name, so tab- and space-separated files (and multi-word names) all work.
inline std::map<int, std::string> loadLabels(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open label file " + path);
    std::map<int, std::string> labels;
    std::string line;
    while (std::getline(in, line)) {
        std::size_t space = line.find_first_of(" \t");
        if (space == std::string::npos) continue;
        // skip runs of whitespace so '9   car' names start clean
        std::size_t nameStart = space;
        while (nameStart < line.size() &&
               (line[nameStart] == ' ' || line[nameStart] == '\t'))
            ++nameStart;
        if (nameStart >= line.size()) continue;
        try {
            labels[std::stoi(line.substr(0, space))] = line.substr(nameStart);
        } catch (const std::exception&) {
            // not an id line - skip
        }
    }
    if (labels.empty()) throw std::runtime_error("no labels parsed from " + path);
    return labels;
}

inline std::string labelFor(const std::map<int, std::string>& labels, int id) {
    std::map<int, std::string>::const_iterator it = labels.find(id);
    if (it != labels.end()) return it->second;
    return "class_" + std::to_string(id);  // never crash on an unexpected id
}

}  // namespace ssd

#endif
