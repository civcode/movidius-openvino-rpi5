// ---------------------------------------------------------------------------
// hello_myriad - smoke test for the OpenVINO 2020.3.2 runtime built by this
// project for the Intel Movidius MA2450 Neural Compute Stick.
//
//   1. enumerate plugins  -> the MYRIAD plugin must be present
//   2. report its API version
//   3. optionally: read an FP16 IR, load it on MYRIAD, run N inferences and
//      report latency + basic output statistics
//
// Exit codes: 0 = all requested checks passed
//             2 = MYRIAD plugin not available
//             3 = model load / inference failure
//             4 = bad command line
// ---------------------------------------------------------------------------

#include <inference_engine.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../half.hpp"  // shared IEEE-754 f16<->f32 conversion (unit-tested in examples/webcam/test_half.cpp)

using namespace InferenceEngine;

namespace {

const char* layoutToString(Layout l) {
    switch (l) {
        case Layout::ANY: return "ANY";
        case Layout::NCHW: return "NCHW";
        case Layout::NHWC: return "NHWC";
        case Layout::NCDHW: return "NCDHW";
        case Layout::NDHWC: return "NDHWC";
        case Layout::OIHW: return "OIHW";
        case Layout::SCALAR: return "SCALAR";
        case Layout::C: return "C";
        default: return "other";
    }
}

std::string dimsToString(const SizeVector& dims) {
    std::ostringstream oss;
    for (size_t i = 0; i < dims.size(); ++i) {
        oss << (i ? "," : "") << dims[i];
    }
    return oss.str();
}

void printDevices(Core& ie) {
    const auto devices = ie.GetAvailableDevices();
    std::cout << "available devices : ";
    for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << (i ? ", " : "") << devices[i];
    }
    std::cout << "\n";

    for (const auto& dev : devices) {
        try {
            for (const auto& kv : ie.GetVersions(dev)) {
                const Version& v = kv.second;
                std::cout << "  plugin " << std::left << std::setw(8) << dev
                          << " api " << v.apiVersion.major << "." << v.apiVersion.minor
                          << " build " << (v.buildNumber ? v.buildNumber : "?")
                          << " (" << (v.description ? v.description : "?") << ")\n";
            }
        } catch (const std::exception& ex) {
            std::cout << "  plugin " << dev << ": version query failed: " << ex.what() << "\n";
        }
    }
}

int runInference(Core& ie, const std::string& modelXml, const std::string& modelBin,
                 const std::string& device, int iterations) {
    std::cout << "\nreading model   : " << modelXml;
    if (!modelBin.empty()) std::cout << " + " << modelBin;
    std::cout << "\n";

    CNNNetwork network = ie.ReadNetwork(modelXml, modelBin);
    std::cout << "network         : " << network.getName() << "\n";

    InputsDataMap inputs = network.getInputsInfo();
    if (inputs.empty()) {
        std::cout << "model has no inputs\n";
        return 3;
    }
    for (auto& item : inputs) {
        // the VPU wants FP16 activations; asking for anything else makes the
        // MYRIAD compiler fail or inserts a conversion node
        item.second->setPrecision(Precision::FP16);
        item.second->setLayout(Layout::NCHW);
        // InputInfo exposes precision/layout only; the shape comes from its Data.
        std::cout << "  input  " << item.first << " dims "
                  << dimsToString(item.second->getInputData()->getDims())
                  << " -> " << item.second->getPrecision().name() << " "
                  << layoutToString(item.second->getLayout()) << "\n";
    }

    OutputsDataMap outputs = network.getOutputsInfo();
    for (const auto& item : outputs) {
        std::cout << "  output " << item.first << " dims " << dimsToString(item.second->getDims())
                  << " " << item.second->getPrecision().name() << "\n";
    }

    std::cout << "loading on      : " << device << " (first load compiles the graph on the VPU)\n";
    const auto t0 = std::chrono::steady_clock::now();
    ExecutableNetwork executable = ie.LoadNetwork(network, device);
    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "load+compile    : "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    InferRequest request = executable.CreateInferRequest();

    // deterministic, cheap input pattern: every activation = 0x3c3c (FP16 ~0.0093)
    for (const auto& item : inputs) {
        Blob::Ptr blob = request.GetBlob(item.first);
        std::memset(blob->buffer().as<uint8_t*>(), 0x3c, blob->byteSize());
    }

    double totalMs = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const auto a = std::chrono::steady_clock::now();
        request.Infer();
        const auto b = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(b - a).count();
        totalMs += ms;
        std::cout << "  inference " << (i + 1) << " : " << std::fixed << std::setprecision(2)
                  << ms << " ms\n";
    }
    std::cout << "mean inference  : " << std::fixed << std::setprecision(2)
              << (totalMs / iterations) << " ms (" << std::setprecision(2)
              << (1000.0 * iterations / totalMs) << " fps)\n";

    for (const auto& item : outputs) {
        Blob::Ptr blob = request.GetBlob(item.first);
        const size_t n = blob->size();
        const uint8_t* raw = blob->cbuffer().as<uint8_t*>();
        // The TensorDesc of a MYRIAD output can claim FP32 while the buffer the
        // plugin hands back is the FP16 activation buffer, so decide from the
        // actual bytes per element instead of from the declared precision.
        const size_t bytesPerElem = n ? blob->byteSize() / n : 0;

        std::cout << "  output " << item.first << " ("
                  << dimsToString(blob->getTensorDesc().getDims())
                  << ", desc precision " << blob->getTensorDesc().getPrecision().name()
                  << ", " << bytesPerElem << " B/elem)";

        double mn = 1e300, mx = -1e300, sum = 0.0;
        if (bytesPerElem == 2) {
            for (size_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, raw + 2 * i, sizeof(h));
                const float f = halfToFloat(h);
                mn = std::min(mn, static_cast<double>(f));
                mx = std::max(mx, static_cast<double>(f));
                sum += f;
            }
        } else if (bytesPerElem == 4) {
            for (size_t i = 0; i < n; ++i) {
                float f;
                std::memcpy(&f, raw + 4 * i, sizeof(f));
                mn = std::min(mn, static_cast<double>(f));
                mx = std::max(mx, static_cast<double>(f));
                sum += f;
            }
        }
        if (n && bytesPerElem) {
            std::cout << " min " << std::fixed << std::setprecision(5) << mn
                      << " max " << mx << " mean " << sum / static_cast<double>(n);
        }
        std::cout << "\n";
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string modelXml, modelBin, device = "MYRIAD";
    int iterations = 1;
    bool listOnly = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* opt) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << opt << "\n";
                std::exit(4);
            }
            return argv[++i];
        };
        if (arg == "--model") {
            modelXml = next("--model");
        } else if (arg == "--weights") {
            modelBin = next("--weights");
        } else if (arg == "--device") {
            device = next("--device");
        } else if (arg == "--iterations") {
            iterations = std::max(1, std::stoi(next("--iterations")));
        } else if (arg == "--list-only") {
            listOnly = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "usage: hello_myriad [--model <IR.xml>] [--weights <IR.bin>]"
                      << " [--device MYRIAD] [--iterations N] [--list-only]\n";
            return 0;
        } else {
            std::cerr << "usage: hello_myriad [--model <IR.xml>] [--weights <IR.bin>]"
                      << " [--device MYRIAD] [--iterations N] [--list-only]\n";
            return 4;
        }
    }

    try {
        Core ie;
        std::cout << "OpenVINO InferenceEngine smoke test\n";
        printDevices(ie);

        const auto devices = ie.GetAvailableDevices();
        const bool myriadAvailable =
            std::find(devices.begin(), devices.end(), "MYRIAD") != devices.end();

        if (listOnly) {
            std::cout << "RESULT: " << (myriadAvailable ? "MYRIAD present" : "MYRIAD absent") << "\n";
            return 0;
        }

        if (!myriadAvailable) {
            std::cout << "RESULT: FAIL - no MYRIAD plugin (is the MA2450 stick plugged in,"
                         " and does the container have --device=/dev/bus/usb?)\n";
            return 2;
        }

        if (!modelXml.empty()) {
            const int rc = runInference(ie, modelXml, modelBin, device, iterations);
            std::cout << "RESULT: " << (rc == 0 ? "PASS" : "FAIL") << "\n";
            return rc;
        }

        std::cout << "RESULT: PASS (no --model given: plugin enumeration only)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cout << "RESULT: FAIL - " << ex.what() << "\n";
        return 3;
    }
}
