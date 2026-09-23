// ---------------------------------------------------------------------------
// mobilenet_server - long-running MYRIAD inference server for the webcam
// example (examples/webcam/webcam_mobilenet.py).
//
// Protocol (fixed sizes, little-endian, no header):
//   stdin : repeated 1x3x224x224 float32 tensors   = 602112 bytes each
//   stdout: one float32 logits vector per request   = 1000 * 4 = 4000 bytes
//   stderr: human-readable startup diagnostics; a "ready" line on success
//
// The network is loaded and compiled once (the stick boot takes ~1.6 s),
// then the server loops until stdin reaches EOF and exits 0.  This is what
// keeps per-frame latency at the device rate (~44 ms) instead of paying the
// load/compile cost on every frame.
//
// Exit codes: 0 clean   2 no MYRIAD   3 model/IO failure   4 bad command line
// ---------------------------------------------------------------------------

#include <inference_engine.hpp>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "half.hpp"  // shared IEEE-754 f16<->f32 conversion (unit-tested in examples/webcam/test_half.cpp)
#include "device_probe.hpp"  // shared MYRIAD device probe with retry
using namespace InferenceEngine;

namespace {


std::string dimsToString(const SizeVector& dims) {
    std::string out;
    for (size_t i = 0; i < dims.size(); ++i) out += (i ? "," : "") + std::to_string(dims[i]);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string modelXml, modelBin, device = "MYRIAD";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* opt) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", opt); std::exit(4); }
            return argv[++i];
        };
        if (arg == "--model") { modelXml = next("--model"); }
        else if (arg == "--weights") { modelBin = next("--weights"); }
        else if (arg == "--device") { device = next("--device"); }
        else if (arg == "-h" || arg == "--help") {
            std::printf("usage: mobilenet_server --model <IR.xml> [--weights <IR.bin>]"
                        " [--device MYRIAD]\n"
                        "stdin : 1x3x224x224 float32 tensors (602112 B each)\n"
                        "stdout: float32 logits (1000 * 4 B) per request, until EOF\n"
                        "exit codes: 0 ok, 2 no device, 3 model/IO failure, 4 bad command line\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 4;
        }
    }

    if (modelXml.empty()) { std::fprintf(stderr, "--model is required\n"); return 4; }

    try {
        Core ie;
        std::vector<std::string> devices;
        if (!waitDevice(ie, device, devices)) {
            std::fprintf(stderr, "device %s not available (have: ", device.c_str());
            for (size_t i = 0; i < devices.size(); ++i) {
                std::fprintf(stderr, "%s%s", i ? ", " : "", devices[i].c_str());
            }
            std::fprintf(stderr, ")\n");
            return 2;
        }

        CNNNetwork network = ie.ReadNetwork(modelXml, modelBin);
        std::fprintf(stderr, "mobilenet_server: model %s%s loaded as %s\n", modelXml.c_str(),
                     modelBin.empty() ? "" : (" + " + modelBin).c_str(), network.getName().c_str());

        InputsDataMap inputs = network.getInputsInfo();
        if (inputs.size() != 1) { std::fprintf(stderr, "expected exactly one input\n"); return 3; }
        const std::string inputName = inputs.begin()->first;
        const SizeVector inputDims = inputs.begin()->second->getInputData()->getDims();
        inputs.begin()->second->setPrecision(Precision::FP16);
        inputs.begin()->second->setLayout(Layout::NCHW);
        const size_t inElems = static_cast<size_t>(inputDims[1]) * inputDims[2] * inputDims[3];

        OutputsDataMap outputs = network.getOutputsInfo();
        if (outputs.size() != 1) { std::fprintf(stderr, "expected exactly one output\n"); return 3; }
        const std::string outputName = outputs.begin()->first;
        const size_t outElems = outputs.begin()->second->getDims().back();

        ExecutableNetwork executable = ie.LoadNetwork(network, device);
        InferRequest request = executable.CreateInferRequest();
        Blob::Ptr inBlob = request.GetBlob(inputName);
        // verify the input blob really is FP16 before casting: the TensorDesc
        // and the handed-back buffer can disagree, and writing uint16_t into a
        // float32 blob would be undefined behaviour
        const size_t inElemBytes = inElems ? inBlob->byteSize() / inElems : 0;
        if (inElemBytes != 2) {
            std::fprintf(stderr, "expected a FP16 input blob (2 B/elem), got %zu B/elem\n",
                         inElemBytes);
            return 3;
        }
        uint16_t* inHalf = inBlob->buffer().as<uint16_t*>();

        const size_t inBytes = inElems * 4;   // request tensor arrives as float32
        const size_t outBytes = outElems * 4; // logits leave as float32
        std::fprintf(stderr,
                     "mobilenet_server: ready (device=%s input=%s FP16 NCHW %zu elems / %zu B per"
                     " request, output %zu elems / %zu B per response)\n",
                     device.c_str(), dimsToString(inputDims).c_str(), inElems, inBytes, outElems,
                     outBytes);
        std::fflush(stderr);

        std::vector<float> requestTensor(inElems);
        std::vector<float> logits(outElems);
        bool eof = false;

        while (!eof) {
            // gather one full request tensor from stdin (pipe writes may arrive
            // in pieces); a short read at 0 bytes is EOF
            size_t got = 0;
            while (got < inBytes) {
                const size_t n = std::fread(requestTensor.data() + got / 4, 4,
                                            (inBytes - got) / 4, stdin);
                if (n == 0) {
                    if (got == 0) { eof = true; break; }   // clean EOF between frames
                    std::fprintf(stderr, "mobilenet_server: truncated request (%zu of %zu B), "
                                         "closing\n", got, inBytes);
                    return 3;
                }
                got += n * 4;
            }
            if (eof) break;

            for (size_t i = 0; i < inElems; ++i) inHalf[i] = floatToHalf(requestTensor[i]);

            request.Infer();

            // the MYRIAD output TensorDesc can claim a precision that does not match
            // the buffer handed back, so decide from the real bytes per element.
            // Check the element count BEFORE copying so an oversized blob cannot
            // overflow the logits vector.
            Blob::Ptr outBlob = request.GetBlob(outputName);
            const size_t n = outBlob->size();
            if (n != outElems) {
                std::fprintf(stderr, "output has %zu values, expected %zu\n", n, outElems);
                return 3;
            }
            const uint8_t* raw = outBlob->cbuffer().as<uint8_t*>();
            const size_t bytesPerElem = n ? outBlob->byteSize() / n : 0;
            if (bytesPerElem == 2) {
                for (size_t i = 0; i < n; ++i) {
                    uint16_t h;
                    std::memcpy(&h, raw + 2 * i, sizeof(h));
                    logits[i] = halfToFloat(h);
                }
            } else if (bytesPerElem == 4) {
                std::memcpy(logits.data(), raw, n * 4);
            } else {
                std::fprintf(stderr, "unsupported output element size %zu\n", bytesPerElem);
                return 3;
            }

            size_t sent = 0;
            while (sent < outBytes) {
                const size_t w = std::fwrite(logits.data() + sent / 4, 4, (outBytes - sent) / 4,
                                             stdout);
                if (w == 0) { std::fprintf(stderr, "client gone (write failed), exiting\n");
                             return 3; }
                sent += w * 4;
            }
            std::fflush(stdout);
        }

        std::fprintf(stderr, "mobilenet_server: stdin closed, bye\n");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "ERROR: %s\n", ex.what());
        return 3;
    }
}
