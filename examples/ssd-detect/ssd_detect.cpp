// ssd_detect: single-image object detection with SSDLite-MobileNetV2 on the
// self-built OpenVINO 2020.3.2 MYRIAD runtime (Milestone 1 of the SSDLite
// integration).
//
//   build/.../ssd_detect --model ssdlite_mobilenet_v2.xml --weights ssdlite_mobilenet_v2.bin \
//       --image dog.ppm --labels coco.txt --device MYRIAD [--min-conf 0.5]
//
// The IR (vendor/models/ssdlite_mobilenet_v2/openvino/, produced by
// scripts/prepare-ssdlite.sh) has its preprocessing baked in: the `image_tensor`
// input is FP16 NCHW [1,3,300,300] and the first two ops are
// x * (2/255) - 1.  So this client feeds raw 0..255 RGB pixel values (BGR from
// the PPM, resized bilinearly to the model's input size) and the graph scales.
// The single `DetectionOutput` op runs NMS internally and yields keep_top_k
// rows of [image_id, class, score, xmin, ymin, xmax, ymax] normalized 0..1;
// ssd_postprocess.hpp turns those into clamped original-image pixel boxes.
//
// The output blob's element width is taken from its real byte size (the MYRIAD
// TensorDesc precision can lie, the same trap as in mobilenet-test).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../half.hpp"
#include "ssd_postprocess.hpp"

#include "inference_engine.hpp"

using namespace InferenceEngine;

namespace {

void usage() {
    std::cout
        << "ssd_detect --model <ir.xml> --weights <ir.bin> --labels <coco.txt>\n"
        << "            --image <photo.ppm> [--device MYRIAD] [--min-conf 0.5]\n"
        << "            [--max-detections 10] [--iterations 1] [--debug]\n";
}

struct Options {
    std::string model, weights, image, labels;
    std::string device = "MYRIAD";
    float minConf = 0.5f;
    int maxDetections = 10;
    int iterations = 1;
    bool debug = false;
};

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) throw std::runtime_error(std::string(what) + " needs a value");
            return argv[++i];
        };
        if (a == "--model") o.model = next("--model");
        else if (a == "--weights") o.weights = next("--weights");
        else if (a == "--image") o.image = next("--image");
        else if (a == "--labels") o.labels = next("--labels");
        else if (a == "--device") o.device = next("--device");
        else if (a == "--min-conf") o.minConf = std::stof(next("--min-conf"));
        else if (a == "--max-detections") o.maxDetections = std::stoi(next("--max-detections"));
        else if (a == "--iterations") o.iterations = std::stoi(next("--iterations"));
        else if (a == "--debug") o.debug = true;
        else if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown argument: " + a);
    }
    if (o.model.empty() || o.weights.empty() || o.image.empty() || o.labels.empty()) {
        usage();
        throw std::runtime_error("--model, --weights, --image and --labels are required");
    }
    return o;
}

struct PpmImage {
    std::string path;
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgb;  // w*h*3, RGB order (our PPMs are written by Pillow)
};

PpmImage readPpm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string magic;
    if (!(in >> magic)) throw std::runtime_error("not a P6 PPM: " + path);
    if (magic != "P6") throw std::runtime_error("expected P6, got " + magic + ": " + path);
    int w, h, maxv;
    if (!(in >> w >> h >> maxv)) throw std::runtime_error("bad PPM header in " + path);
    in.get();  // single whitespace after maxval
    PpmImage img;
    img.path = path;
    img.width = w;
    img.height = h;
    img.rgb.resize(static_cast<std::size_t>(w) * h * 3);
    in.read(reinterpret_cast<char*>(img.rgb.data()),
            static_cast<std::streamsize>(img.rgb.size()));
    if (in.gcount() != static_cast<std::streamsize>(img.rgb.size()))
        throw std::runtime_error("truncated PPM body: " + path);
    return img;
}

// Bilinear resize to tw x th; src/dst are w*h*3 BGR buffers.  Half-pixel sample
// (matches OpenCV::resize INTER_LINEAR, the reference for the model's input).
void bilinearResize(const std::vector<unsigned char>& src, int sw, int sh,
                    std::vector<unsigned char>& dst, int tw, int th) {
    dst.assign(static_cast<std::size_t>(tw) * th * 3, 0);
    const float xRatio = static_cast<float>(sw) / tw;
    const float yRatio = static_cast<float>(sh) / th;
    for (int y = 0; y < th; ++y) {
        const float fy = (y + 0.5f) * yRatio - 0.5f;
        const int y0 = std::max(0, static_cast<int>(std::floor(fy)));
        const int y1 = std::min(sh - 1, y0 + 1);
        const float wy = fy - y0;
        for (int x = 0; x < tw; ++x) {
            const float fx = (x + 0.5f) * xRatio - 0.5f;
            const int x0 = std::max(0, static_cast<int>(std::floor(fx)));
            const int x1 = std::min(sw - 1, x0 + 1);
            const float wx = fx - x0;
            for (int c = 0; c < 3; ++c) {
                const float v00 = src[(y0 * sw + x0) * 3 + c];
                const float v01 = src[(y0 * sw + x1) * 3 + c];
                const float v10 = src[(y1 * sw + x0) * 3 + c];
                const float v11 = src[(y1 * sw + x1) * 3 + c];
                const float v = (v00 * (1 - wx) + v01 * wx) * (1 - wy) +
                                (v10 * (1 - wx) + v11 * wx) * wy;
                dst[(y * tw + x) * 3 + c] = static_cast<unsigned char>(v + 0.5f);
            }
        }
    }
}

std::string dimsToString(const SizeVector& dims) {
    std::ostringstream s;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i) s << ",";
        s << dims[i];
    }
    return s.str();
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    try {
        opt = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }

    try {
        Core ie;
        const auto devices = ie.GetAvailableDevices();
        if (std::find(devices.begin(), devices.end(), opt.device) == devices.end()) {
            std::cerr << "device " << opt.device << " not available (have: ";
            for (size_t i = 0; i < devices.size(); ++i)
                std::cout << (i ? ", " : "") << devices[i];
            std::cout << ")\n";
            return 2;
        }

        const auto t0 = std::chrono::steady_clock::now();
        CNNNetwork network = ie.ReadNetwork(opt.model.c_str(), opt.weights.c_str());
        const auto t1 = std::chrono::steady_clock::now();
        std::cout << "load+read IR    : " << std::fixed << std::setprecision(2)
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

        InputsDataMap inputs = network.getInputsInfo();
        if (inputs.size() != 1) { std::cerr << "expected exactly one input\n"; return 3; }
        const std::string inputName = inputs.begin()->first;
        const SizeVector inputDims = inputs.begin()->second->getInputData()->getDims();
        const auto inPrec = inputs.begin()->second->getPrecision();
        inputs.begin()->second->setPrecision(Precision::FP16);
        inputs.begin()->second->setLayout(Layout::NCHW);
        const char* inPrecStr = inPrec == Precision::FP16 ? "FP16" : inPrec == Precision::FP32 ? "FP32" : "other";
        std::cout << "model input     : " << inputName << " [" << dimsToString(inputDims)
                  << "] " << inPrecStr
                  << " (raw 0-255 RGB; the 2/255 scale and -1 offset are inside the graph)\n";

        if (inputDims.size() != 4 || inputDims[1] != 3) {
            std::cerr << "expected NCHW [1,3,H,W] input, got " << dimsToString(inputDims) << "\n";
            return 3;
        }
        const int modelH = static_cast<int>(inputDims[2]);
        const int modelW = static_cast<int>(inputDims[3]);

        OutputsDataMap outputs = network.getOutputsInfo();
        if (outputs.size() != 1) { std::cerr << "expected exactly one output\n"; return 3; }
        const std::string outputName = outputs.begin()->first;
        const SizeVector outDims = outputs.begin()->second->getDims();
        std::cout << "model output    : " << outputName << " [" << dimsToString(outDims)
                  << "] DetectionOutput rows = (image_id, class, score, xmin, ymin, xmax, ymax)\n";
        if (outDims.size() != 4 || outDims[3] != 7) {
            std::cerr << "expected [1,1,N,7] output, got " << dimsToString(outDims) << "\n";
            return 3;
        }
        const std::size_t rowCount = outDims[2];

        // image: P6 PPM (RGB order, as written by Pillow)
        PpmImage img = readPpm(opt.image);
        std::vector<unsigned char> resized;
        const auto t2 = std::chrono::steady_clock::now();
        bilinearResize(img.rgb, img.width, img.height, resized, modelW, modelH);
        const auto t3 = std::chrono::steady_clock::now();
        std::cout << "input image     : " << img.path << " " << img.width << "x" << img.height
                  << " -> " << modelW << "x" << modelH << " in "
                  << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms\n";

        const auto t4 = std::chrono::steady_clock::now();
        ExecutableNetwork executable = ie.LoadNetwork(network, opt.device);
        const auto t5 = std::chrono::steady_clock::now();
        std::cout << "compile to " << opt.device << " : "
                  << std::chrono::duration<double, std::milli>(t5 - t4).count() << " ms\n";
        InferRequest request = executable.CreateInferRequest();

        // fill the input blob: raw 0..255 values, RGB channels (the PPMs are RGB)
        Blob::Ptr inBlob = request.GetBlob(inputName);
        const std::size_t inElems = inBlob->size();
        const std::size_t inBytes = inBlob->byteSize();
        const int inElemBytes = static_cast<int>(inBytes / inElems);
        if (inElems != static_cast<std::size_t>(3) * modelH * modelW) {
            std::cerr << "blob has " << inElems << " elements, expected "
                      << 3 * modelH * modelW << "\n";
            return 3;
        }
        const std::size_t plane = static_cast<std::size_t>(modelW) * modelH;
        if (opt.debug)
            std::cout << "in blob bytes   : " << inBytes << " (" << inElemBytes << "/elem)\n";
        if (inElemBytes == 2) {
            uint16_t* p = inBlob->buffer().as<uint16_t*>();
            for (int c = 0; c < 3; ++c)
                for (std::size_t px = 0; px < plane; ++px)
                    p[c * plane + px] = floatToHalf(static_cast<float>(resized[3 * px + c]));
        } else if (inElemBytes == 4) {
            float* p = inBlob->buffer().as<float*>();
            for (int c = 0; c < 3; ++c)
                for (std::size_t px = 0; px < plane; ++px)
                    p[c * plane + px] = static_cast<float>(resized[3 * px + c]);
        } else {
            std::cerr << "unsupported input element size " << inElemBytes << " bytes\n";
            return 3;
        }

        // inference loop
        double inferMs = 0.0;
        for (int it = 0; it < opt.iterations; ++it) {
            const auto a = std::chrono::steady_clock::now();
            request.Infer();
            const auto b = std::chrono::steady_clock::now();
            inferMs += std::chrono::duration<double, std::milli>(b - a).count();
        }
        const double meanInferMs = inferMs / opt.iterations;
        std::cout << "inference       : " << meanInferMs << " ms mean over " << opt.iterations
                  << " run(s) (" << std::setprecision(1)
                  << 1000.0 * opt.iterations / inferMs << " fps)\n";

        // output: the MYRIAD TensorDesc can claim FP32 while handing back FP16,
        // so decide the element width from the real byte size
        Blob::Ptr outBlob = request.GetBlob(outputName);
        const std::size_t outElems = outBlob->size();
        const int outElemBytes = static_cast<int>(outBlob->byteSize() / outElems);
        if (outElems != rowCount * 7) {
            std::cerr << "output blob has " << outElems << " elements, expected "
                      << rowCount * 7 << "\n";
            return 3;
        }
        const auto t6 = std::chrono::steady_clock::now();
        std::vector<float> rows(outElems);
        if (outElemBytes == 2) {
            const uint16_t* p = outBlob->buffer().as<uint16_t*>();
            for (std::size_t i = 0; i < outElems; ++i) rows[i] = halfToFloat(p[i]);
        } else if (outElemBytes == 4) {
            std::memcpy(&rows[0], outBlob->buffer().as<float*>(), outElems * sizeof(float));
        } else {
            std::cerr << "unsupported output element size " << outElemBytes << " bytes\n";
            return 3;
        }

        std::map<int, std::string> labels = ssd::loadLabels(opt.labels);
        const std::vector<ssd::Detection> dets =
            ssd::parseDetections(&rows[0], rowCount, opt.minConf, opt.maxDetections,
                                 img.width, img.height);
        const auto t7 = std::chrono::steady_clock::now();

        if (dets.empty()) {
            std::cout << "no detections at confidence >= " << opt.minConf << "\n";
        } else {
            for (const ssd::Detection& d : dets) {
                std::cout << std::setw(14) << std::left
                          << ssd::labelFor(labels, d.class_id)
                          << " " << std::fixed << std::setprecision(2) << d.confidence
                          << " [" << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << "]\n";
            }
        }
        std::cout << "postprocess     : " << std::chrono::duration<double, std::milli>(t7 - t6).count()
                  << " ms, " << dets.size() << " detection(s)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
