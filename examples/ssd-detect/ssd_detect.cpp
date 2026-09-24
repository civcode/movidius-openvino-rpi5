// ssd_detect: object detection with SSDLite-MobileNetV2 on the self-built
// OpenVINO 2020.3.2 MYRIAD runtime.
//
// Milestone 1 (single image):
//   build/.../ssd_detect --model ssdlite_mobilenet_v2.xml --weights ssdlite_mobilenet_v2.bin \
//       --image dog.ppm --labels coco.txt --device MYRIAD [--min-conf 0.5]
//
// Milestone 2 (frame stream, driven by ssd_stream.py over stdio):
//   ssd_detect --model ... --weights ... --labels ... --stdin
//   stdin : repeated frames, each = uint32 w + uint32 h (little endian) + w*h*3 RGB bytes
//   stdout: per frame:  "FRAME <w> <h> <infer_ms>"
//                        "DET <label> <score> <x1> <y1> <x2> <y2>"   (0..N lines)
//                        "END"
//
// The IR (vendor/models/ssdlite_mobilenet_v2/openvino/, produced by
// scripts/prepare-ssdlite.sh) has its preprocessing baked in: the `image_tensor`
// input is FP16 NCHW [1,3,300,300] and the first two ops are
// x * (2/255) - 1.  So this client feeds raw 0..255 RGB pixel values (straight
// from the RGB PPM / stream frames, resized bilinearly to the model's input
// size) and the graph scales.  No channel conversion happens anywhere.
// The single `DetectionOutput` op runs NMS internally and yields keep_top_k
// rows of [image_id, class, score, xmin, ymin, xmax, ymax] normalized 0..1;
// ssd_postprocess.hpp turns those into clamped original-image pixel boxes.
//
// The output blob's element width is taken from its real byte size (the MYRIAD
// TensorDesc precision can lie, the same trap as in mobilenet-test).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "half.hpp"
#include "device_probe.hpp"  // shared MYRIAD device probe with retry
#include "frame_utils.hpp"   // shared PPM I/O + bilinear resize
#include "ssd_postprocess.hpp"

#include "inference_engine.hpp"

using namespace InferenceEngine;

namespace {

void usage() {
    std::cout
        << "ssd_detect --model <ir.xml> --weights <ir.bin> --labels <coco.txt>\n"
        << "            --image <photo.ppm> [--device MYRIAD] [--min-conf 0.5]\n"
        << "            [--max-detections 10] [--iterations 1] [--debug]\n"
        << "            --stdin   (stream mode: frames on stdin, see header)\n\n"
        << "Exit codes: 0 ok, 1 runtime failure, 2 device/CLI error, 3 model/shape error.\n";
}

struct Options {
    std::string model, weights, image, labels;
    std::string device = "MYRIAD";
    float minConf = 0.5f;
    int maxDetections = 10;
    int iterations = 1;
    bool debug = false;
    bool stdinStream = false;
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
        else if (a == "--stdin") o.stdinStream = true;
        else if (a == "--debug") o.debug = true;
        else if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown argument: " + a);
    }
    if (o.model.empty() || o.weights.empty() || o.labels.empty()) {
        usage();
        throw std::runtime_error("--model, --weights and --labels are required");
    }
    if (!o.stdinStream && o.image.empty()) {
        usage();
        throw std::runtime_error("--image is required unless --stdin is given");
    }
    return o;
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
        std::vector<std::string> devices;
        if (!waitDevice(ie, opt.device, devices)) {
            // whole message on stderr: in stream mode stdout is the frame protocol
            std::cerr << "device " << opt.device << " not available (have: ";
            for (size_t i = 0; i < devices.size(); ++i)
                std::cerr << (i ? ", " : "") << devices[i];
            std::cerr << ")\n";
            return 2;
        }

        // in stream mode stdout is the frame protocol; diagnostics go to stderr
        std::ostream& info = opt.stdinStream ? std::cerr : std::cout;

        const auto t0 = std::chrono::steady_clock::now();
        CNNNetwork network = ie.ReadNetwork(opt.model.c_str(), opt.weights.c_str());
        const auto t1 = std::chrono::steady_clock::now();
        info << "load+read IR    : " << std::fixed << std::setprecision(2)
             << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

        InputsDataMap inputs = network.getInputsInfo();
        if (inputs.size() != 1) { std::cerr << "expected exactly one input\n"; return 3; }
        const std::string inputName = inputs.begin()->first;
        const SizeVector inputDims = inputs.begin()->second->getInputData()->getDims();
        const auto inPrec = inputs.begin()->second->getPrecision();
        // The MYRIAD VPU only accepts FP16 inputs; the CPU plugin of this
        // OpenVINO release rejects FP16 ("Input image format FP16 is not
        // supported yet"), so keep the IR's native precision elsewhere - the
        // launchers feed the FP32 IRs when the device is CPU.
        if (opt.device == "MYRIAD")
            inputs.begin()->second->setPrecision(Precision::FP16);
        inputs.begin()->second->setLayout(Layout::NCHW);
        const char* inPrecStr = inPrec == Precision::FP16 ? "FP16" : inPrec == Precision::FP32 ? "FP32" : "other";
        info << "model input     : " << inputName << " [" << dimsToString(inputDims)
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
        info << "model output    : " << outputName << " [" << dimsToString(outDims)
             << "] DetectionOutput rows = (image_id, class, score, xmin, ymin, xmax, ymax)\n";
        if (outDims.size() != 4 || outDims[3] != 7) {
            std::cerr << "expected [1,1,N,7] output, got " << dimsToString(outDims) << "\n";
            return 3;
        }
        const std::size_t rowCount = outDims[2];

        const auto t4 = std::chrono::steady_clock::now();
        ExecutableNetwork executable = ie.LoadNetwork(network, opt.device);
        const auto t5 = std::chrono::steady_clock::now();
        info << "compile to " << opt.device << " : "
             << std::chrono::duration<double, std::milli>(t5 - t4).count() << " ms\n";
        InferRequest request = executable.CreateInferRequest();

        // input blob: raw 0..255 values, RGB channels (our PPMs/frames are RGB)
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

        // output: the MYRIAD TensorDesc can claim FP32 while handing back FP16,
        // so the element width is decided from the real byte size, read after
        // Infer (the blob's storage is only final after the first inference)
        Blob::Ptr outBlob = request.GetBlob(outputName);
        const std::size_t outElems = outBlob->size();
        if (outElems != rowCount * 7) {
            std::cerr << "output blob has " << outElems << " elements, expected "
                      << rowCount * 7 << "\n";
            return 3;
        }

        std::vector<float> rows(outElems);
        std::vector<unsigned char> resized;
        std::map<int, std::string> labels = ssd::loadLabels(opt.labels);

        // run one frame: raw 0..255 RGB (w*h*3) in, (detections, infer ms) out
        auto runFrame = [&](const std::vector<unsigned char>& rgb, int fw, int fh) -> std::pair<std::vector<ssd::Detection>, double> {
            frameutils::bilinearResize(rgb.data(), fw, fh, resized, modelW, modelH);
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
                throw std::runtime_error("unsupported input element size " +
                                         std::to_string(inElemBytes) + " bytes");
            }
            const auto a = std::chrono::steady_clock::now();
            request.Infer();
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - a).count();
            const int outElemBytes =
                static_cast<int>(outBlob->byteSize() / outElems);  // post-Infer
            if (outElemBytes == 2) {
                const uint16_t* p = outBlob->buffer().as<uint16_t*>();
                for (std::size_t i = 0; i < outElems; ++i) rows[i] = halfToFloat(p[i]);
            } else if (outElemBytes == 4) {
                std::memcpy(&rows[0], outBlob->buffer().as<float*>(), outElems * sizeof(float));
            } else {
                throw std::runtime_error("unsupported output element size " +
                                         std::to_string(outElemBytes) + " bytes");
            }
            const std::vector<ssd::Detection> dets =
                ssd::parseDetections(&rows[0], rowCount, opt.minConf, opt.maxDetections,
                                     fw, fh);
            return std::make_pair(std::move(dets), ms);
        };

        std::ios::sync_with_stdio(false);

        if (opt.stdinStream) {
            // stream mode: repeated frames on stdin (see the header comment).
            // stdout carries the frame protocol, so failures are reported as
            // "ERROR <message>" on stdout (plus a copy on stderr) - the stream
            // clients turn that into a readable error instead of a bare
            // "server closed the pipe".
            int rc = 0;
            try {
                uint32_t fw = 0, fh = 0;
                std::vector<unsigned char> frame;
                std::size_t frameNo = 0;
                while (std::cin.read(reinterpret_cast<char*>(&fw), sizeof(fw)) &&
                       std::cin.read(reinterpret_cast<char*>(&fh), sizeof(fh))) {
                    if (fw == 0 || fh == 0 || fw > 8192 || fh > 8192)
                        throw std::runtime_error("frame header out of range: " +
                                                 std::to_string(fw) + "x" + std::to_string(fh));
                    frame.resize(static_cast<std::size_t>(fw) * fh * 3);
                    std::cin.read(reinterpret_cast<char*>(frame.data()), frame.size());
                    if (static_cast<std::size_t>(std::cin.gcount()) != frame.size())
                        throw std::runtime_error("truncated frame body (" +
                                                 std::to_string(frame.size()) + " bytes expected)");
                    auto res = runFrame(frame, fw, fh);
                    ++frameNo;
                    std::cout << "FRAME " << fw << " " << fh << " "
                              << std::fixed << std::setprecision(1) << res.second << "\n";
                    for (const ssd::Detection& d : res.first) {
                        std::cout << "DET " << ssd::labelFor(labels, d.class_id) << " "
                                  << std::fixed << std::setprecision(2) << d.confidence
                                  << " " << d.x1 << " " << d.y1 << " " << d.x2 << " " << d.y2 << "\n";
                    }
                    std::cout << "END\n";
                    std::cout.flush();
                }
                std::cerr << "ssd_detect: stream done (" << frameNo << " frame(s))\n";
            } catch (const std::exception& e) {
                std::cout << "ERROR " << e.what() << "\n";
                std::cout.flush();
                std::cerr << "error: " << e.what() << "\n";
                rc = 1;
            }
            return rc;
        }

        // single-image mode
        frameutils::Ppm img = frameutils::readPpm(opt.image);
        std::vector<unsigned char> unused;
        const auto t2 = std::chrono::steady_clock::now();
        frameutils::bilinearResize(img.rgb.data(), img.w, img.h, unused, modelW, modelH);
        const auto t3 = std::chrono::steady_clock::now();
        std::cout << "input image     : " << opt.image << " " << img.w << "x" << img.h
                  << " -> " << modelW << "x" << modelH << " in "
                  << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms\n";

        double inferMs = 0.0;
        std::vector<ssd::Detection> dets;
        for (int it = 0; it < opt.iterations; ++it) {
            auto res = runFrame(img.rgb, img.w, img.h);
            inferMs += res.second;
            dets = std::move(res.first);
        }
        const double meanInferMs = inferMs / opt.iterations;
        std::cout << "inference       : " << meanInferMs << " ms mean over " << opt.iterations
                  << " run(s) (" << std::setprecision(1)
                  << 1000.0 * opt.iterations / inferMs << " fps)\n";

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
        std::cout << "postprocess     : " << dets.size() << " detection(s)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
