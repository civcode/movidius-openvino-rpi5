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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <ov203/half.hpp>  // shared IEEE-754 f16<->f32 conversion (unit-tested in examples/webcam/test_half.cpp)
#include <ov203/device_probe.hpp>
#include <ov203/device_spec.hpp>
#include <ov203/wire.hpp>

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

std::vector<float> readF32(const std::string& path, size_t expected) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open tensor: " + path);
    const std::streamsize bytes = in.tellg();
    if (bytes < 0 || static_cast<size_t>(bytes) != expected * sizeof(float)) {
        std::ostringstream oss;
        oss << "tensor byte size " << bytes << " does not match expected "
            << (expected * sizeof(float));
        throw std::runtime_error(oss.str());
    }
    in.seekg(0);
    std::vector<float> values(expected);
    in.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!in) throw std::runtime_error("cannot read tensor: " + path);
    return values;
}

void writeF32(const std::string& path, const std::vector<float>& values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open output tensor: " + path);
    out.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) throw std::runtime_error("cannot write output tensor: " + path);
}


int runStreamInference(const std::string& modelXml, const std::string& modelBin,
                       const std::string& device, int warmupIterations) {
    Core ie;
    const ov203::DeviceSpec deviceSpec = ov203::parseDeviceSpec(device);
    std::vector<std::string> devices;
    if (!waitPhysicalDevices(ie, deviceSpec.physicalDevices, devices)) {
        std::fprintf(stderr, "tensor_server: device %s not available (have:", device.c_str());
        for (const auto& item : devices) std::fprintf(stderr, " %s", item.c_str());
        std::fprintf(stderr, ")\n");
        return 2;
    }

    CNNNetwork network = ie.ReadNetwork(modelXml, modelBin);
    InputsDataMap inputs = network.getInputsInfo();
    OutputsDataMap outputs = network.getOutputsInfo();
    if (inputs.size() != 1 || outputs.size() != 1) {
        std::fprintf(stderr, "tensor_server: expected exactly one input and one output\n");
        return 3;
    }

    const std::string inputName = inputs.begin()->first;
    const SizeVector inputDims = inputs.begin()->second->getInputData()->getDims();
    if (deviceSpec.usesMyriad) inputs.begin()->second->setPrecision(Precision::FP16);
    inputs.begin()->second->setLayout(TensorDesc::getLayoutByDims(inputDims));

    size_t inElems = 1;
    for (size_t value : inputDims) inElems *= value;

    const std::string outputName = outputs.begin()->first;
    const SizeVector outputDims = outputs.begin()->second->getDims();
    size_t outElems = 1;
    for (size_t value : outputDims) outElems *= value;
    if (inElems == 0 || outElems == 0) {
        std::fprintf(stderr, "tensor_server: zero-sized model tensor\n");
        return 3;
    }

    const auto loadStart = std::chrono::steady_clock::now();
    ExecutableNetwork executable = ie.LoadNetwork(network, device);
    const auto loadEnd = std::chrono::steady_clock::now();
    const double loadMs =
        std::chrono::duration<double, std::milli>(loadEnd - loadStart).count();
    InferRequest request = executable.CreateInferRequest();

    Blob::Ptr inBlob = request.GetBlob(inputName);
    const size_t inElemBytes = inBlob->byteSize() / inElems;
    if (inElemBytes != 2 && inElemBytes != 4) {
        std::fprintf(stderr,
                     "tensor_server: unsupported input width %zu B/element\n",
                     inElemBytes);
        return 3;
    }
    uint16_t* inHalf = inBlob->buffer().as<uint16_t*>();
    float* inFloat = inBlob->buffer().as<float*>();

    // Warm the compiled graph once per server session. This is deliberately
    // outside the measured request stream so every recorded latency is
    // steady-state Infer() time.
    std::memset(inBlob->buffer().as<uint8_t*>(), 0, inBlob->byteSize());
    for (int i = 0; i < warmupIterations; ++i) request.Infer();

    const size_t inBytes = inElems * sizeof(float);
    const size_t outBytes = outElems * sizeof(float);
    std::fprintf(
        stderr,
        "READY protocol=tensor-stream-v1 input_elements=%zu output_elements=%zu "
        "load_ms=%.6f warmup=%d\n",
        inElems, outElems, loadMs, warmupIterations);
    std::fflush(stderr);

    std::vector<float> inputValues(inElems);
    std::vector<float> outputValues(outElems);
    std::vector<unsigned char> outputWire(outBytes);
    size_t requestIndex = 0;

    while (true) {
        size_t got = 0;
        auto* inputBytes = reinterpret_cast<unsigned char*>(inputValues.data());
        while (got < inBytes) {
            const size_t count = std::fread(inputBytes + got, 1, inBytes - got, stdin);
            if (count == 0) {
                if (got == 0 && std::feof(stdin)) {
                    std::fprintf(stderr,
                                 "tensor_server: stdin EOF after %zu request(s)\n",
                                 requestIndex);
                    std::fflush(stderr);
                    return 0;
                }
                std::fprintf(stderr,
                             "tensor_server: truncated request (%zu of %zu bytes)\n",
                             got, inBytes);
                return 3;
            }
            got += count;
        }

        ov203::littleEndianFloat32ToNativeInPlace(inputValues.data(), inElems);
        if (inElemBytes == 2) {
            for (size_t i = 0; i < inElems; ++i) {
                inHalf[i] = floatToHalf(inputValues[i]);
            }
        } else {
            std::memcpy(inFloat, inputValues.data(), inBytes);
        }

        const auto inferStart = std::chrono::steady_clock::now();
        request.Infer();
        const auto inferEnd = std::chrono::steady_clock::now();
        const double inferMs =
            std::chrono::duration<double, std::milli>(inferEnd - inferStart).count();

        Blob::Ptr outBlob = request.GetBlob(outputName);
        if (outBlob->size() != outElems) {
            std::fprintf(stderr,
                         "tensor_server: output has %zu values, expected %zu\n",
                         outBlob->size(), outElems);
            return 3;
        }
        const uint8_t* raw = outBlob->cbuffer().as<uint8_t*>();
        const size_t outElemBytes = outBlob->byteSize() / outElems;
        if (outElemBytes == 2) {
            for (size_t i = 0; i < outElems; ++i) {
                uint16_t value;
                std::memcpy(&value, raw + 2 * i, sizeof(value));
                outputValues[i] = halfToFloat(value);
            }
        } else if (outElemBytes == 4) {
            std::memcpy(outputValues.data(), raw, outBytes);
        } else {
            std::fprintf(stderr,
                         "tensor_server: unsupported output width %zu B/element\n",
                         outElemBytes);
            return 3;
        }

        ov203::nativeFloat32ToLittleEndianBytes(
            outputValues.data(), outputWire.data(), outElems);
        size_t sent = 0;
        while (sent < outBytes) {
            const size_t count =
                std::fwrite(outputWire.data() + sent, 1, outBytes - sent, stdout);
            if (count == 0) {
                std::fprintf(stderr, "tensor_server: client output pipe closed\n");
                return 3;
            }
            sent += count;
        }
        std::fflush(stdout);

        ++requestIndex;
        std::fprintf(stderr, "TIMING request=%zu inference_ms=%.6f\n",
                     requestIndex, inferMs);
        std::fflush(stderr);
    }
}

int runInference(Core& ie, const std::string& modelXml, const std::string& modelBin,
                 const std::string& device, int iterations,
                 const std::string& tensorPath, const std::string& outputPath) {
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
        item.second->setLayout(
            TensorDesc::getLayoutByDims(item.second->getInputData()->getDims()));
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

    if (!tensorPath.empty()) {
        if (inputs.size() != 1) {
            throw std::runtime_error("--tensor requires a model with exactly one input");
        }
        const auto& item = *inputs.begin();
        Blob::Ptr blob = request.GetBlob(item.first);
        const auto values = readF32(tensorPath, blob->size());
        if (blob->byteSize() != values.size() * sizeof(uint16_t)) {
            throw std::runtime_error("MYRIAD input blob is not FP16-sized");
        }
        uint8_t* raw = blob->buffer().as<uint8_t*>();
        for (size_t i = 0; i < values.size(); ++i) {
            const uint16_t h = floatToHalf(values[i]);
            std::memcpy(raw + 2 * i, &h, sizeof(h));
        }
        std::cout << "input tensor    : " << tensorPath << " (" << values.size()
                  << " f32 values -> FP16)\n";
    } else {
        // deterministic, cheap input pattern: every activation = 0x3c3c
        for (const auto& item : inputs) {
            Blob::Ptr blob = request.GetBlob(item.first);
            std::memset(blob->buffer().as<uint8_t*>(), 0x3c, blob->byteSize());
        }
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
        std::vector<float> decoded;
        decoded.reserve(n);
        if (bytesPerElem == 2) {
            for (size_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, raw + 2 * i, sizeof(h));
                const float f = halfToFloat(h);
                decoded.push_back(f);
                mn = std::min(mn, static_cast<double>(f));
                mx = std::max(mx, static_cast<double>(f));
                sum += f;
            }
        } else if (bytesPerElem == 4) {
            for (size_t i = 0; i < n; ++i) {
                float f;
                std::memcpy(&f, raw + 4 * i, sizeof(f));
                decoded.push_back(f);
                mn = std::min(mn, static_cast<double>(f));
                mx = std::max(mx, static_cast<double>(f));
                sum += f;
            }
        } else {
            throw std::runtime_error("unsupported output bytes per element");
        }
        if (!outputPath.empty()) {
            if (outputs.size() != 1) {
                throw std::runtime_error("--output requires a model with exactly one output");
            }
            writeF32(outputPath, decoded);
            std::cout << " -> " << outputPath;
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
    std::string modelXml, modelBin, device = "MYRIAD", tensorPath, outputPath;
    int iterations = 1;
    int warmupIterations = 1;
    bool listOnly = false;
    bool streamMode = false;

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
        } else if (arg == "--tensor") {
            tensorPath = next("--tensor");
        } else if (arg == "--output") {
            outputPath = next("--output");
        } else if (arg == "--list-only") {
            listOnly = true;
        } else if (arg == "--stdin") {
            streamMode = true;
        } else if (arg == "--warmup") {
            warmupIterations = std::max(0, std::stoi(next("--warmup")));
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "usage: hello_myriad [--model <IR.xml>] [--weights <IR.bin>]"
                      << " [--device MYRIAD] [--iterations N] [--tensor input.f32]"
                      << " [--output output.f32] [--list-only]\n"
                      << "       hello_myriad --model <IR.xml> [--weights <IR.bin>]"
                      << " [--device MYRIAD] --stdin [--warmup N]\n"
                      << "stream protocol: repeated little-endian float32 input tensors"
                      << " on stdin; one float32 output tensor per request on stdout\n";
            return 0;
        } else {
            std::cerr << "usage: hello_myriad [--model <IR.xml>] [--weights <IR.bin>]"
                      << " [--device MYRIAD] [--iterations N] [--tensor input.f32]"
                      << " [--output output.f32] [--list-only|--stdin]\n";
            return 4;
        }
    }

    try {
        if (streamMode) {
            if (modelXml.empty()) {
                std::cerr << "--stdin requires --model\n";
                return 4;
            }
            if (!tensorPath.empty() || !outputPath.empty()) {
                std::cerr << "--stdin cannot be combined with --tensor/--output\n";
                return 4;
            }
            return runStreamInference(
                modelXml, modelBin, device, warmupIterations);
        }

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
            const int rc = runInference(
                ie, modelXml, modelBin, device, iterations, tensorPath, outputPath);
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
