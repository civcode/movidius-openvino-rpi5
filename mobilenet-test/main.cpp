// ---------------------------------------------------------------------------
// mobilenet_classify - real image classification on the Intel Movidius MA2450
// with the OpenVINO 2020.3.2 runtime built by this project.
//
// Two ways to feed the network:
//   --tensor <file>   raw little-endian float32 tensor exactly as shipped in the
//                     ONNX model zoo test_data_set_0 (1x3x224x224).  Pairs with
//                     --reference <file>, the framework's own float32 output, so
//                     the whole chain (IR -> VPU compile -> inference) is checked
//                     numerically against a known-good result.
//   --image <ppm>     224x224 binary PPM (P6), normalised here with the
//                     preprocessing documented by the model zoo:
//                     x/255, mean [0.485,0.456,0.406], std [0.229,0.224,0.225].
//
// Exit codes: 0 ok   2 no MYRIAD   3 model/IO failure   4 bad command line
//             5 numerical comparison against --reference failed
// ---------------------------------------------------------------------------

#include <inference_engine.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "half.hpp"  // shared IEEE-754 f16<->f32 conversion (unit-tested in examples/webcam/test_half.cpp)

using namespace InferenceEngine;

namespace {


std::string dimsToString(const SizeVector& dims) {
    std::ostringstream oss;
    for (size_t i = 0; i < dims.size(); ++i) oss << (i ? "," : "") << dims[i];
    return oss.str();
}

// --------------------------------------------------------------- file helpers

std::vector<float> readFloats(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    in.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(in.tellg());
    if (bytes % 4) throw std::runtime_error(path + ": size is not a multiple of 4");
    std::vector<float> v(bytes / 4);
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error(path + ": read failed");
    return v;
}

// binary PPM (P6), 8 bit per channel, comments and \r\n tolerated
struct PpmImage {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
};

PpmImage readPpm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    if (in.get() != 'P' || in.get() != '6') {
        throw std::runtime_error(path + ": only binary PPM (P6) is supported");
    }

    // one whitespace/comment separated token; the whitespace that terminates a
    // token is consumed, which is exactly what the spec allows after maxval
    auto nextToken = [&in]() {
        std::string tok;
        int c = in.get();
        while (in) {
            if (c == '#') {
                while (in && in.get() != '\n') {}
                c = in.get();
                continue;
            }
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                if (!tok.empty()) break;
                c = in.get();
                continue;
            }
            tok.push_back(static_cast<char>(c));
            c = in.get();
        }
        return tok;
    };

    PpmImage img;
    const std::string w = nextToken(), h = nextToken(), m = nextToken();
    try {
        img.width = std::stoi(w);
        img.height = std::stoi(h);
        const int maxval = std::stoi(m);
        if (maxval != 255) throw std::runtime_error(path + ": PPM maxval must be 255");
    } catch (const std::invalid_argument&) {
        throw std::runtime_error(path + ": bad PPM header (width height maxval)");
    }
    if (img.width <= 0 || img.height <= 0) throw std::runtime_error(path + ": empty PPM");

    img.rgb.resize(static_cast<size_t>(img.width) * img.height * 3);
    in.read(reinterpret_cast<char*>(img.rgb.data()), static_cast<std::streamsize>(img.rgb.size()));
    if (!in) throw std::runtime_error(path + ": truncated PPM pixel data");
    return img;
}

std::vector<std::string> readLabels(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty()) labels.push_back(line);
    }
    return labels;
}

// "n03347037 fire screen, fireguard" -> "fire screen, fireguard"
std::string labelName(const std::string& line) {
    if (line.size() > 9 && line[0] == 'n') {
        size_t space = line.find(' ');
        if (space != std::string::npos && space + 1 < line.size()) return line.substr(space + 1);
    }
    return line;
}

std::string labelId(const std::string& line) {
    if (line.size() > 9 && line[0] == 'n') {
        size_t space = line.find(' ');
        return line.substr(0, space == std::string::npos ? line.size() : space);
    }
    return "-";
}

std::vector<float> parseTriple(const std::string& text, const char* opt) {
    std::vector<float> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stof(item));
    if (out.size() != 3) {
        std::cerr << opt << " needs 3 comma separated values (got '" << text << "')\n";
        std::exit(4);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    // flush stdout before any diagnostic on stderr, so an error message does not
    // appear in the middle of the report
    std::cerr.tie(&std::cout);

    std::string modelXml, modelBin, device = "MYRIAD";
    std::string tensorPath, imagePath, labelsPath, referencePath;
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> stddev = {0.229f, 0.224f, 0.225f};
    int topk = 5, iterations = 1;
    double tol = 1.0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* opt) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << opt << "\n"; std::exit(4); }
            return argv[++i];
        };
        if (arg == "--model") { modelXml = next("--model"); }
        else if (arg == "--weights") { modelBin = next("--weights"); }
        else if (arg == "--device") { device = next("--device"); }
        else if (arg == "--tensor") { tensorPath = next("--tensor"); }
        else if (arg == "--image") { imagePath = next("--image"); }
        else if (arg == "--labels") { labelsPath = next("--labels"); }
        else if (arg == "--reference") { referencePath = next("--reference"); }
        else if (arg == "--mean") { mean = parseTriple(next("--mean"), "--mean"); }
        else if (arg == "--std") { stddev = parseTriple(next("--std"), "--std"); }
        else if (arg == "--topk") { topk = std::max(1, std::stoi(next("--topk"))); }
        else if (arg == "--iterations") { iterations = std::max(1, std::stoi(next("--iterations"))); }
        else if (arg == "--tol") { tol = std::stod(next("--tol")); }
        else if (arg == "-h" || arg == "--help") {
            std::cout << "usage: mobilenet_classify --model <IR.xml> [--weights <IR.bin>]"
                      << " [--device MYRIAD]\n"
                      << "       (--tensor <raw f32 1x3x224x224> | --image <224x224 P6 PPM>)\n"
                      << "       [--labels <synset.txt>] [--reference <raw f32 1x1000>] [--tol 1.0]\n"
                      << "       [--mean r,g,b] [--std r,g,b] [--topk 5] [--iterations N]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return 4;
        }
    }

    if (modelXml.empty()) { std::cerr << "--model is required\n"; return 4; }
    if (tensorPath.empty() == imagePath.empty()) {
        std::cerr << "give exactly one of --tensor or --image\n";
        return 4;
    }

    try {
        Core ie;
        const auto devices = ie.GetAvailableDevices();
        if (std::find(devices.begin(), devices.end(), device) == devices.end()) {
            std::cerr << "device " << device << " not available (have: ";
            for (size_t i = 0; i < devices.size(); ++i) std::cout << (i ? ", " : "") << devices[i];
            std::cout << ")\n";
            return 2;
        }

        std::cout << "model           : " << modelXml;
        if (!modelBin.empty()) std::cout << " + " << modelBin;
        std::cout << "\n";

        CNNNetwork network = ie.ReadNetwork(modelXml, modelBin);
        std::cout << "network         : " << network.getName() << "\n";

        InputsDataMap inputs = network.getInputsInfo();
        if (inputs.size() != 1) { std::cerr << "expected exactly one input\n"; return 3; }
        const std::string inputName = inputs.begin()->first;
        SizeVector inputDims = inputs.begin()->second->getInputData()->getDims();
        inputs.begin()->second->setPrecision(Precision::FP16);
        inputs.begin()->second->setLayout(Layout::NCHW);
        std::cout << "input           : " << inputName << " " << dimsToString(inputDims)
                  << " FP16 NCHW\n";

        OutputsDataMap outputs = network.getOutputsInfo();
        if (outputs.size() != 1) { std::cerr << "expected exactly one output\n"; return 3; }
        const std::string outputName = outputs.begin()->first;
        std::cout << "output          : " << outputName << " "
                  << dimsToString(outputs.begin()->second->getDims()) << "\n";

        const auto t0 = std::chrono::steady_clock::now();
        ExecutableNetwork executable = ie.LoadNetwork(network, device);
        const auto t1 = std::chrono::steady_clock::now();
        std::cout << "load+compile    : " << std::fixed << std::setprecision(2)
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms on "
                  << device << "\n";

        InferRequest request = executable.CreateInferRequest();
        Blob::Ptr inBlob = request.GetBlob(inputName);
        const size_t inElems = inBlob->size();
        uint16_t* inHalf = inBlob->buffer().as<uint16_t*>();

        // build the activation tensor: either the shipped float32 tensor or the photo
        std::vector<float> input;
        if (!tensorPath.empty()) {
            input = readFloats(tensorPath);
            std::cout << "input source    : " << tensorPath << " (" << input.size() << " float32)\n";
        } else {
            PpmImage img = readPpm(imagePath);
            if (inputDims.size() != 4 || inputDims[1] != 3 ||
                inputDims[2] != static_cast<size_t>(img.height) ||
                inputDims[3] != static_cast<size_t>(img.width)) {
                std::cerr << "image is " << img.width << "x" << img.height << " but the model wants "
                          << dimsToString(inputDims) << "\n";
                return 3;
            }
            input.resize(inElems);
            const size_t plane = static_cast<size_t>(img.width) * img.height;
            for (int c = 0; c < 3; ++c) {
                for (size_t p = 0; p < plane; ++p) {
                    const float v = static_cast<float>(img.rgb[3 * p + c]) / 255.0f;
                    input[static_cast<size_t>(c) * plane + p] = (v - mean[c]) / stddev[c];
                }
            }
            std::cout << "input source    : " << imagePath << " " << img.width << "x" << img.height
                      << ", mean " << mean[0] << "," << mean[1] << "," << mean[2] << " std "
                      << stddev[0] << "," << stddev[1] << "," << stddev[2] << "\n";
        }
        if (input.size() != inElems) {
            std::cerr << "input has " << input.size() << " elements, blob needs " << inElems << "\n";
            return 3;
        }
        for (size_t i = 0; i < inElems; ++i) inHalf[i] = floatToHalf(input[i]);

        double totalMs = 0.0;
        for (int it = 0; it < iterations; ++it) {
            const auto a = std::chrono::steady_clock::now();
            request.Infer();
            const auto b = std::chrono::steady_clock::now();
            totalMs += std::chrono::duration<double, std::milli>(b - a).count();
        }
        std::cout << "inference       : " << std::fixed << std::setprecision(2)
                  << totalMs / iterations << " ms mean over " << iterations << " run(s) ("
                  << std::setprecision(1) << 1000.0 * iterations / totalMs << " fps)\n";

        // logits: the MYRIAD output TensorDesc can claim a precision that does not
                // match the buffer handed back, so decide from the real bytes per element
        Blob::Ptr outBlob = request.GetBlob(outputName);
        const size_t n = outBlob->size();
        const uint8_t* raw = outBlob->cbuffer().as<uint8_t*>();
        const size_t bytesPerElem = n ? outBlob->byteSize() / n : 0;
        std::vector<float> logits(n);
        if (bytesPerElem == 2) {
            for (size_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, raw + 2 * i, sizeof(h));
                logits[i] = halfToFloat(h);
            }
        } else if (bytesPerElem == 4) {
            for (size_t i = 0; i < n; ++i) {
                float f;
                std::memcpy(&f, raw + 4 * i, sizeof(f));
                logits[i] = f;
            }
        } else {
            std::cerr << "unsupported output element size " << bytesPerElem << "\n";
            return 3;
        }
        std::cout << "logits          : " << n << " values, " << bytesPerElem
                  << " B/elem, desc precision " << outBlob->getTensorDesc().getPrecision().name()
                  << "\n";

        // softmax + top-k
        double maxLogit = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probs(n);
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) { probs[i] = static_cast<float>(std::exp(logits[i] - maxLogit)); sum += probs[i]; }
        for (float& p : probs) p /= static_cast<float>(sum);

        std::vector<size_t> order(n);
        for (size_t i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return probs[a] > probs[b]; });

        const std::vector<std::string> labels = labelsPath.empty() ? std::vector<std::string>{}
                                                                   : readLabels(labelsPath);
        std::cout << "\ntop " << topk << ":\n";
        for (int k = 0; k < topk && k < static_cast<int>(n); ++k) {
            const size_t idx = order[k];
            std::cout << "  " << std::setw(2) << (k + 1) << ". " << std::setw(6) << idx << " "
                      << std::fixed << std::setprecision(4) << std::setw(8) << probs[idx] << "  ";
            if (idx < labels.size()) {
                std::cout << labelId(labels[idx]) << "  " << labelName(labels[idx]);
            } else if (!labelsPath.empty()) {
                std::cout << "(label file has " << labels.size() << " lines)";
            }
            std::cout << "  logit " << std::setprecision(3) << logits[idx] << "\n";
        }

        int rc = 0;
        if (!referencePath.empty()) {
            const std::vector<float> ref = readFloats(referencePath);
            if (ref.size() != n) {
                std::cerr << "reference has " << ref.size() << " values, output has " << n << "\n";
                return 3;
            }
            double maxAbs = 0.0, sumAbs = 0.0, refMaxAbs = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double d = std::fabs(static_cast<double>(logits[i]) - ref[i]);
                maxAbs = std::max(maxAbs, d);
                sumAbs += d;
                refMaxAbs = std::max(refMaxAbs, std::fabs(static_cast<double>(ref[i])));
            }
            std::vector<size_t> refOrder(n);
            for (size_t i = 0; i < n; ++i) refOrder[i] = i;
            std::sort(refOrder.begin(), refOrder.end(),
                      [&](size_t a, size_t b) { return ref[a] > ref[b]; });

            std::cout << "\nagainst reference " << referencePath << ":\n"
                      << "  max |diff|    : " << std::fixed << std::setprecision(4) << maxAbs
                      << "  (reference logits up to " << refMaxAbs << ")\n"
                      << "  mean |diff|   : " << sumAbs / n << "\n"
                      << "  top-1         : device " << order[0] << " vs reference " << refOrder[0]
                      << "\n";
            for (int k = 0; k < 5; ++k) {
                std::cout << "  top-" << (k + 1) << "        : device " << order[k] << "  reference "
                          << refOrder[k] << (order[k] == refOrder[k] ? "  = match" : "") << "\n";
            }
            const bool top1ok = order[0] == refOrder[0];
            const bool close = maxAbs <= tol;
            std::cout << "\nRESULT: " << (top1ok && close ? "PASS" : "FAIL")
                      << " (top-1 " << (top1ok ? "matches" : "differs") << ", max |diff| "
                      << maxAbs << " vs tolerance " << tol << ")\n";
            if (!top1ok || !close) rc = 5;
        }
        return rc;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 3;
    }
}
