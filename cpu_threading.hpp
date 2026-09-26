// ---------------------------------------------------------------------------
// cpu_threading.hpp - keep the OpenVINO CPU plugin from pinning every server
// process's inference thread onto the same core.
//
// CPU_BIND_THREAD defaults to YES in this release: the plugin binds its worker
// threads to particular cores.  For a single process that is a small win, but
// every process binds to the *same* core, so running several servers - which is
// how a small network is scaled across a many-core machine - serialises them all
// on one CPU.  The machine then looks mostly idle while throughput sits at one
// core's worth.  Measured on a 32-core amd64 host, mobilenet_server on CPU,
// identical FP32 IR:
//
//     servers   pinned (plugin default)   unpinned (this header)
//         1        ~180 inferences/s          ~150 inferences/s
//         2        ~190 inferences/s          ~296 inferences/s
//         4        ~190 inferences/s          ~565 inferences/s
//        12          (no gain available)     ~1324 inferences/s
//
// So the pinning costs a few percent on a lone server and costs nearly everything
// above that.  Servers therefore ask for NO by default and print what they set.
// MYRIAD has no such option and is left untouched.
//
// Override per server with --bind-thread yes|no|numa (mobilenet_server), or by
// editing the call site.  Note this is about *thread placement*: it does not make
// one inference use several cores, which is what CPU_THROUGHPUT_STREAMS and
// multiple InferRequests would be for.
// ---------------------------------------------------------------------------
#ifndef OPENVINO_DEMO_CPU_THREADING_HPP
#define OPENVINO_DEMO_CPU_THREADING_HPP

#include <inference_engine.hpp>

#include <cstdio>
#include <map>
#include <string>

// Apply the CPU plugin threading config for `device` and return a short,
// printable summary of what was set ("" when nothing applied, i.e. non-CPU).
// A refused key is reported on stderr and swallowed: a server that cannot tune
// its threads should still serve frames.
inline std::string configureCpuThreading(InferenceEngine::Core& ie,
                                         const std::string& device,
                                         std::string bindThread,
                                         const std::string& streams,
                                         const std::string& threadsNum) {
    using InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD;
    using InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM;
    using InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS;

    if (device.find("CPU") == std::string::npos) {
        if (!bindThread.empty() || !streams.empty() || !threadsNum.empty()) {
            std::fprintf(stderr,
                         "cpu_threading: --bind-thread/--streams/--threads apply to a "
                         "CPU device only, ignored for %s\n", device.c_str());
        }
        return std::string();
    }
    if (bindThread.empty()) {
        bindThread = InferenceEngine::PluginConfigParams::NO;   // un-pin: see the note
    }
    std::map<std::string, std::string> cfg;
    cfg[KEY_CPU_BIND_THREAD] = bindThread;
    if (!streams.empty()) cfg[KEY_CPU_THROUGHPUT_STREAMS] = streams;
    if (!threadsNum.empty()) cfg[KEY_CPU_THREADS_NUM] = threadsNum;
    try {
        ie.SetConfig(cfg, device);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cpu_threading: %s refused the config (%s); "
                     "continuing with plugin defaults\n", device.c_str(), e.what());
        return std::string();
    }
    std::string shown = std::string(KEY_CPU_BIND_THREAD) + "=" + bindThread;
    if (!streams.empty())
        shown += std::string(" ") + KEY_CPU_THROUGHPUT_STREAMS + "=" + streams;
    if (!threadsNum.empty())
        shown += std::string(" ") + KEY_CPU_THREADS_NUM + "=" + threadsNum;
    return shown;
}

#endif  // OPENVINO_DEMO_CPU_THREADING_HPP
