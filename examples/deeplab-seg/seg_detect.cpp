// seg_detect.cpp
//
// DeepLabV3 (Pascal VOC 21-class) semantic segmentation for the Movidius
// MYRIAD (MA2450) on OpenVINO 2020.3.2.
//
//   seg_detect --model M --weights W --labels L [--image img.ppm]
//               [--device MYRIAD] [--mask-out mask.ppm] [--debug]
//   seg_detect --model M --weights W --labels L --stdin
//
// The IR (vendor/models/deeplabv3/openvino/, produced by
// scripts/prepare-deeplabv3.sh) has its preprocessing baked in: the
// `ImageTensor` input is FP16 NCHW [1,3,513,513] BGR raw 0..255; the graph
// itself does the BGR->RGB channel swap, a bilinear resize to 513x513
// (identity at the fixed input size) and (x/127.5) - 1.  So this client
// feeds raw 0..255 BGR pixel values (resized bilinearly to the model's
// input size) and the graph scales.  The `ArgMax` output is [1,513,513]
// per-pixel class ids 0..20; the 513x513 mask is resized back to the
// original frame size with nearest-neighbour sampling only (no averaging:
// segmentation boundaries must not be blurred).
//
// Milestone 2 (frame stream, driven by seg_stream.py over stdio):
//   seg_detect --model ... --weights ... --labels ... --stdin
//   stdin : repeated frames, each = uint32 w + uint32 h (little endian) + w*h*3 RGB bytes
//   stdout: per frame:  "FRAME <w> <h> <total_ms> <infer_ms>"
//                        "CLASSES <n>"
//                        "CLASS <id> <name> <pixels>"        (n lines)
//                        "MASK <w> <h>" + w*h uint16 LE class ids
//                        "END"
//   Startup diagnostics go to stderr.
//
// The output blob's element width is taken from its real byte size (the
// MYRIAD TensorDesc precision can lie, the same trap as in mobilenet-test).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../half.hpp"
#include "seg_postprocess.hpp"

#include "inference_engine.hpp"

using namespace InferenceEngine;

namespace {

struct Args {
    std::string model, weights, labels, image, maskOut;
    std::string device = "MYRIAD";
    bool stdinMode = false;
    bool debug = false;
    bool help = false;
};

int parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << s << std::endl;
                return nullptr;
            }
            return argv[++i];
        };
        auto need = [&](std::string& dst) -> bool {
            const char* v = next();
            if (v == nullptr)
                return false;
            dst = v;
            return true;
        };
        if (s == "--model") {
            if (!need(a.model))
                return 2;
        } else if (s == "--weights") {
            if (!need(a.weights))
                return 2;
        } else if (s == "--labels") {
            if (!need(a.labels))
                return 2;
        } else if (s == "--image") {
            if (!need(a.image))
                return 2;
        } else if (s == "--mask-out") {
            if (!need(a.maskOut))
                return 2;
        } else if (s == "--device") {
            if (!need(a.device))
                return 2;
        } else if (s == "--stdin") {
            a.stdinMode = true;
        } else if (s == "--debug") {
            a.debug = true;
        } else if (s == "-h" || s == "--help") {
            std::cerr << "usage: seg_detect --model M --weights W --labels L "
                         "[--image img.ppm | --stdin] [--device MYRIAD] "
                         "[--mask-out m.ppm] [--debug]\n";
            a.help = true;
            return 0;
        } else {
            std::cerr << "unknown arg: " << s << std::endl;
            return 2;
        }
    }
    if (a.model.empty() || a.weights.empty() || a.labels.empty()) {
        std::cerr << "--model, --weights and --labels are required\n";
        return 2;
    }
    if (a.image.empty() && !a.stdinMode) {
        std::cerr << "one of --image or --stdin is required\n";
        return 2;
    }
    return 0;
}

struct Ppm {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;  // w*h*3, row-major
};

Ppm readPpm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    Ppm p;
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::string magic;
    in >> magic;
    if (magic != "P6")
        throw std::runtime_error("unsupported PPM magic: " + magic);
    auto skipComments = [&]() {
        int c;
        while ((c = in.peek()) == '#') {
            std::string line;
            std::getline(in, line);
        }
    };
    int w, h, maxv;
    skipComments();
    in >> w;
    skipComments();
    in >> h;
    skipComments();
    in >> maxv;
    if (w <= 0 || h <= 0 || maxv != 255)
        throw std::runtime_error("bad PPM dimensions: " + std::to_string(w) +
                                 "x" + std::to_string(h) + " maxval " +
                                 std::to_string(maxv));
    in.get();  // single whitespace after maxval
    p.w = w;
    p.h = h;
    p.rgb.assign((size_t)w * (size_t)h * 3, 0);
    in.read((char*)p.rgb.data(), (std::streamsize)p.rgb.size());
    if (!in)
        throw std::runtime_error("truncated PPM data in " + path);
    return p;
}

// Bilinear resize of an interleaved RGB buffer (used to get the client's
// image to the model's input size).
void bilinearResize(const uint8_t* src, int sw, int sh,
                    std::vector<uint8_t>& dst, int dw, int dh) {
    dst.assign((size_t)dw * (size_t)dh * 3, 0);
    for (int y = 0; y < dh; ++y) {
        float fy = (y + 0.5f) * sh / dh - 0.5f;
        int y0 = (int)fy;
        if (y0 < 0)
            y0 = 0;
        if (y0 > sh - 2)
            y0 = sh - 2;
        float wy = fy - y0;
        int y1 = y0 + 1;
        for (int x = 0; x < dw; ++x) {
            float fx = (x + 0.5f) * sw / dw - 0.5f;
            int x0 = (int)fx;
            if (x0 < 0)
                x0 = 0;
            if (x0 > sw - 2)
                x0 = sw - 2;
            float wx = fx - x0;
            int x1 = x0 + 1;
            for (int c = 0; c < 3; ++c) {
                float v00 = src[((size_t)y0 * sw + x0) * 3 + c];
                float v01 = src[((size_t)y0 * sw + x1) * 3 + c];
                float v10 = src[((size_t)y1 * sw + x0) * 3 + c];
                float v11 = src[((size_t)y1 * sw + x1) * 3 + c];
                float v = v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy) +
                          v10 * (1 - wx) * wy + v11 * wx * wy;
                dst[((size_t)y * dw + x) * 3 + c] =
                    (uint8_t)(v + 0.5f);
            }
        }
    }
}

std::string dimsToString(const SizeVector& dims) {
    std::ostringstream oss;
    for (size_t i = 0; i < dims.size(); ++i)
        oss << (i ? "," : "") << dims[i];
    return oss.str();
}

// Write a class map as a single-channel P6 PPM (class id = pixel value).
int writeMaskPpm(const std::string& path, const seg::ClassMap& m) {
    std::ofstream out(path, std::ios::binary);
    if (!out)
        return -1;
    out << "P6\n" << m.w << " " << m.h << "\n255\n";
    out.write((const char*)m.ids.data(), (std::streamsize)m.ids.size());
    return out ? 0 : -1;
}

struct FrameResult {
    seg::ClassMap mask513;  // at the model's output size
    double preprocessMs = 0;
    double inferMs = 0;
    double postprocessMs = 0;
};

}  // namespace

int main(int argc, char** argv) {
    Args a;
    int rc = parseArgs(argc, argv, a);
    if (rc != 0)
        return rc;
    if (a.help)
        return 0;
    if (a.model.empty() || a.weights.empty() || a.labels.empty() ||
        (a.image.empty() && !a.stdinMode)) {
        std::cerr << "seg_detect: --model, --weights, --labels and --image "
                      "(or --stdin) are required\n";
        return 2;
    }

    try {
        Core ie;
        const auto devices = ie.GetAvailableDevices();
        if (std::find(devices.begin(), devices.end(), a.device) == devices.end()) {
            std::cerr << "device " << a.device << " not available (have: ";
            for (size_t i = 0; i < devices.size(); ++i)
                std::cout << (i ? ", " : "") << devices[i];
            std::cout << ")\n";
            return 2;
        }

        // in stream mode stdout is the frame protocol; diagnostics go to stderr
        std::ostream& info = a.stdinMode ? std::cerr : std::cout;

        const auto t0 = std::chrono::steady_clock::now();
        CNNNetwork network = ie.ReadNetwork(a.model.c_str(), a.weights.c_str());
        const auto t1 = std::chrono::steady_clock::now();
        info << "load+read IR    : " << std::fixed << std::setprecision(2)
             << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

        InputsDataMap inputs = network.getInputsInfo();
        if (inputs.size() != 1) {
            std::cerr << "expected exactly one input\n";
            return 3;
        }
        const std::string inputName = inputs.begin()->first;
        const SizeVector inputDims = inputs.begin()->second->getInputData()->getDims();
        const auto inPrec = inputs.begin()->second->getPrecision();
        inputs.begin()->second->setPrecision(Precision::FP16);
        inputs.begin()->second->setLayout(Layout::NCHW);
        const char* inPrecStr = inPrec == Precision::FP16 ? "FP16" : inPrec == Precision::FP32 ? "FP32" : "other";
        info << "model input     : " << inputName << " [" << dimsToString(inputDims)
             << "] " << inPrecStr
             << " (raw 0-255 BGR; the channel swap, resize and (x/127.5)-1 are inside the graph)\n";

        if (inputDims.size() != 4 || inputDims[1] != 3) {
            std::cerr << "expected NCHW [1,3,H,W] input, got " << dimsToString(inputDims) << "\n";
            return 3;
        }
        const int modelH = (int)inputDims[2];
        const int modelW = (int)inputDims[3];

        OutputsDataMap outputs = network.getOutputsInfo();
        if (outputs.size() != 1) {
            std::cerr << "expected exactly one output\n";
            return 3;
        }
        const std::string outputName = outputs.begin()->first;
        const SizeVector outDims = outputs.begin()->second->getDims();
        info << "model output    : " << outputName << " [" << dimsToString(outDims)
             << "] per-pixel class ids 0..20\n";
        if (outDims.size() != 3) {
            std::cerr << "expected [1,H,W] output, got " << dimsToString(outDims) << "\n";
            return 3;
        }

        const auto t4 = std::chrono::steady_clock::now();
        ExecutableNetwork executable = ie.LoadNetwork(network, a.device);
        const auto t5 = std::chrono::steady_clock::now();
        info << "compile to " << a.device << " : "
             << std::chrono::duration<double, std::milli>(t5 - t4).count() << " ms\n";
        InferRequest request = executable.CreateInferRequest();

        // input blob: raw 0..255 values, BGR channels (the graph swaps to RGB)
        Blob::Ptr inBlob = request.GetBlob(inputName);
        const std::size_t inElems = inBlob->size();
        const std::size_t inBytes = inBlob->byteSize();
        const int inElemBytes = (int)(inBytes / inElems);
        if (inElems != (std::size_t)3 * modelH * modelW) {
            std::cerr << "blob has " << inElems << " elements, expected "
                      << 3 * modelH * modelW << "\n";
            return 3;
        }
        const std::size_t plane = (std::size_t)modelW * modelH;

        // output: the MYRIAD TensorDesc can claim FP32 while handing back FP16,
        // so decide the element width from the real byte size
        Blob::Ptr outBlob = request.GetBlob(outputName);
        const std::size_t outElems = outBlob->size();
        const int outElemBytes = (int)(outBlob->byteSize() / outElems);
        const int outH = (int)outDims[1];
        const int outW = (int)outDims[2];

        const int numClasses = 21;
        auto labels = seg::loadLabels(a.labels);

        std::vector<uint8_t> resized;

        // run one frame: raw 0..255 RGB (w*h*3) in, 513x513 class map out
        auto runFrame = [&](const std::vector<uint8_t>& rgb, int fw, int fh) -> FrameResult {
            const auto pp0 = std::chrono::steady_clock::now();
            bilinearResize(rgb.data(), fw, fh, resized, modelW, modelH);
            if (inElemBytes == 2) {
                uint16_t* p = inBlob->buffer().as<uint16_t*>();
                for (int c = 0; c < 3; ++c) {
                    const int rc = 2 - c;  // 0->B, 1->G, 2->R
                    for (std::size_t px = 0; px < plane; ++px)
                        p[(std::size_t)c * plane + px] =
                            floatToHalf((float)resized[3 * px + rc]);
                }
            } else if (inElemBytes == 4) {
                float* p = inBlob->buffer().as<float*>();
                for (int c = 0; c < 3; ++c) {
                    const int rc = 2 - c;
                    for (std::size_t px = 0; px < plane; ++px)
                        p[(std::size_t)c * plane + px] =
                            (float)resized[3 * px + rc];
                }
            } else {
                throw std::runtime_error("unsupported input element size " +
                                         std::to_string(inElemBytes) + " bytes");
            }
            const auto pp1 = std::chrono::steady_clock::now();

            const auto ia = std::chrono::steady_clock::now();
            request.Infer();
            const auto ib = std::chrono::steady_clock::now();

            const auto post0 = std::chrono::steady_clock::now();
            // the IR declares the ArgMax output as I32 class ids, but the
            // backend may hand back I16/I32 or an FP16/FP32 copy: decode the
            // raw bytes and verify the values are plausible class ids
            std::vector<float> outRows;
            if (!seg::interpretClassIds(
                    outBlob->buffer().as<uint8_t*>(), outElems, outElemBytes,
                    numClasses, outRows))
                throw std::runtime_error(
                    "cannot interpret the " + outputName + " output blob ("
                    + std::to_string(outElemBytes) + " bytes/element) as "
                    "class ids 0.." + std::to_string(numClasses - 1));
            // C=0: the output is [1,H,W] (no class axis); the logits branch
            // only fires for a real C*H*W blob with C == numClasses.
            seg::ClassMap m = seg::classMapFromRaw(outRows.data(), outElems,
                                                   0, outH, outW, numClasses,
                                                   a.model, outputName);
            const auto post1 = std::chrono::steady_clock::now();

            FrameResult r;
            r.mask513 = m;
            r.preprocessMs =
                std::chrono::duration<double, std::milli>(pp1 - pp0).count();
            r.inferMs = std::chrono::duration<double, std::milli>(ib - ia).count();
            r.postprocessMs =
                std::chrono::duration<double, std::milli>(post1 - post0).count();
            return r;
        };

        std::ios::sync_with_stdio(false);

        if (a.stdinMode) {
            // stream mode: repeated frames on stdin (see the header comment)
            uint32_t fw = 0, fh = 0;
            std::vector<uint8_t> frame;
            std::size_t frameNo = 0;
            while (true) {
                uint8_t hdr[8];
                if (!std::cin.read((char*)hdr, 8))
                    break;
                fw = (uint32_t)hdr[0] | ((uint32_t)hdr[1] << 8) |
                     ((uint32_t)hdr[2] << 16) | ((uint32_t)hdr[3] << 24);
                fh = (uint32_t)hdr[4] | ((uint32_t)hdr[5] << 8) |
                     ((uint32_t)hdr[6] << 16) | ((uint32_t)hdr[7] << 24);
                if (fw == 0 || fh == 0 || fw > 8192 || fh > 8192)
                    throw std::runtime_error("frame header out of range: " +
                                             std::to_string(fw) + "x" +
                                             std::to_string(fh));
                frame.resize((std::size_t)fw * fh * 3);
                std::cin.read((char*)frame.data(), frame.size());
                if ((std::size_t)std::cin.gcount() != frame.size())
                    throw std::runtime_error("truncated frame body (" +
                                             std::to_string(frame.size()) +
                                             " bytes expected)");
                FrameResult res = runFrame(frame, (int)fw, (int)fh);
                ++frameNo;

                // nearest-neighbour resize back to the original frame size
                seg::ClassMap big;
                big.w = (int)fw;
                big.h = (int)fh;
                big.ids.assign((std::size_t)fw * fh, 0);
                seg::nearestResize(res.mask513.ids.data(), outW, outH,
                                   big.ids.data(), (int)fw, (int)fh);

                std::cout << "FRAME " << fw << " " << fh << " "
                          << std::fixed << std::setprecision(1)
                          << (res.preprocessMs + res.inferMs + res.postprocessMs)
                          << " " << res.inferMs << "\n";
                auto hist = seg::classHistogram(big);
                std::cout << "CLASSES " << hist.size() << "\n";
                for (const auto& cc : hist) {
                    std::cout << "CLASS " << cc.id << " "
                              << seg::labelFor(labels, cc.id) << " "
                              << cc.pixels << "\n";
                }
                std::vector<uint16_t> mask16(big.ids.begin(), big.ids.end());
                std::cout << "MASK " << fw << " " << fh << "\n";
                std::cout.write((const char*)mask16.data(),
                                (std::streamsize)mask16.size() * 2);
                std::cout << "END\n";
                std::cout.flush();
            }
            std::cerr << "seg_detect: stream done (" << frameNo << " frame(s))\n";
            return 0;
        }

        // single-image mode
        Ppm img = readPpm(a.image);
        FrameResult res = runFrame(img.rgb, img.w, img.h);

        seg::ClassMap big;
        big.w = img.w;
        big.h = img.h;
        big.ids.assign((std::size_t)img.w * (std::size_t)img.h, 0);
        seg::nearestResize(res.mask513.ids.data(), outW, outH, big.ids.data(),
                           img.w, img.h);

        const double totalMs =
            res.preprocessMs + res.inferMs + res.postprocessMs;
        info << "preprocess    : " << std::fixed << std::setprecision(2)
             << res.preprocessMs << " ms\n";
        info << "inference     : " << res.inferMs << " ms\n";
        info << "postprocess   : " << res.postprocessMs << " ms\n";
        info << "total         : " << totalMs << " ms\n";
        info << "input         : " << img.w << "x" << img.h
             << " (resized to " << modelW << "x" << modelH << ")\n";
        info << "classes present (" << img.w << "x" << img.h << " mask):\n";
        auto hist = seg::classHistogram(big);
        const uint64_t totalPx = (uint64_t)img.w * img.h;
        for (const auto& cc : hist) {
            info << "  " << std::setw(4) << cc.id << "  "
                 << seg::labelFor(labels, cc.id) << "  " << cc.pixels << " ("
                 << std::fixed << std::setprecision(1)
                 << (100.0 * (double)cc.pixels / (double)totalPx) << "%)\n";
        }
        if (a.debug)
            info << "mask: " << hist.size() << " classes present\n";

        if (!a.maskOut.empty() && writeMaskPpm(a.maskOut, big) != 0) {
            std::cerr << "failed to write mask to " << a.maskOut << "\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "seg_detect: " << e.what() << "\n";
        return 1;
    }
}
