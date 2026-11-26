// Phase 3.3 per-call submission overhead micro-benchmark.
//
//   test_percall_p3 <native|shim|lv1|lv3> [iters]
//
// native : run against the system libcuda (no XSched on LD_LIBRARY_PATH)
// shim   : XSched shim loaded, no XQueue on the stream (pass-through)
// lv1/lv3: an XQueue at Level-1 / Level-3 on the stream
//
// Patterns (all on one non-blocking stream, tiny 1-CTA kernel, P50/P90 over `iters`):
//   sync1 : launch 1 kernel + cudaStreamSynchronize     (decode-style single call)
//   sync4 : launch 4 kernels + cudaStreamSynchronize
//   async : 2000 back-to-back launches, then one sync     (launch throughput, us/launch)
// The last line is machine-readable: [P3-PERCALL] mode=.. sync1_p50=.. sync4_p50=.. async_us=..
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef XSCHED_PERCALL_WITH_XSCHED
#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "cuda error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

__global__ void k_tiny(int *out, int v)
{
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

// Busy kernel of ~`cycles` SM clocks (models a 0.1-2 ms decode/prefill operator).
__global__ void k_busy(int *out, long long cycles, int v)
{
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
    if (threadIdx.x == 0) out[1024 + blockIdx.x] = v;
}

static void host_work_us(double us)
{
    const auto a = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count() < us) { }
}

using clk = std::chrono::steady_clock;

static double pct(std::vector<double> v, double q)
{
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1)))];
}

int main(int argc, char **argv)
{
    const std::string mode = argc > 1 ? argv[1] : "native";
    const int iters = argc > 2 ? std::atoi(argv[2]) : 3000;

    CK(cudaSetDevice(0));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    int *d = nullptr;
    CK(cudaMalloc(&d, 8192));   // [0,1024): k_tiny, [1024,2048): k_busy
    CK(cudaMemset(d, 0, 8192));
    CK(cudaDeviceSynchronize());

#ifdef XSCHED_PERCALL_WITH_XSCHED
    HwQueueHandle hwq = 0;
    XQueueHandle xq = 0;
    if (mode == "lv1" || mode == "lv3") {
        const XPreemptLevel lvl = mode == "lv1" ? kPreemptLevelBlock : kPreemptLevelInterrupt;
        if (CudaQueueCreate(&hwq, s) != kXSchedSuccess ||
            XQueueCreate(&xq, hwq, lvl, kQueueCreateFlagNone) != kXSchedSuccess) {
            std::fprintf(stderr, "XQueueCreate failed\n");
            return 2;
        }
    }
#else
    if (mode != "native") {
        std::fprintf(stderr, "built without XSched; only 'native' is supported\n");
        return 2;
    }
#endif

    // Warm-up (Lv3: first launch instruments the kernel).
    for (int i = 0; i < 200; ++i) k_tiny<<<1, 32, 0, s>>>(d, i);
    CK(cudaStreamSynchronize(s));

    std::vector<double> t1, t4;
    t1.reserve(iters);
    t4.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto a = clk::now();
        k_tiny<<<1, 32, 0, s>>>(d, i);
        CK(cudaStreamSynchronize(s));
        t1.push_back(std::chrono::duration<double, std::micro>(clk::now() - a).count());
    }
    for (int i = 0; i < iters; ++i) {
        auto a = clk::now();
        for (int k = 0; k < 4; ++k) k_tiny<<<1, 32, 0, s>>>(d, i + k);
        CK(cudaStreamSynchronize(s));
        t4.push_back(std::chrono::duration<double, std::micro>(clk::now() - a).count());
    }
    std::vector<double> ta;
    for (int r = 0; r < 5; ++r) {
        const int n = 2000;
        auto a = clk::now();
        for (int i = 0; i < n; ++i) k_tiny<<<1, 32, 0, s>>>(d, i);
        CK(cudaStreamSynchronize(s));
        ta.push_back(std::chrono::duration<double, std::micro>(clk::now() - a).count() / n);
    }

    // long: ~LONG_US busy kernel + GAP_US host work before each call (models a PyTorch operator
    // followed by stream.synchronize(), with Python overhead between calls).
    const double long_us = std::getenv("LONG_US") ? std::atof(std::getenv("LONG_US")) : 300.0;
    const double gap_us = std::getenv("GAP_US") ? std::atof(std::getenv("GAP_US")) : 50.0;
    int clk_khz = 0;
    CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
    const long long cycles = static_cast<long long>(long_us * clk_khz / 1000.0);
    std::vector<double> tl;
    const int long_iters = std::max(200, iters / 10);
    for (int i = 0; i < long_iters; ++i) {
        host_work_us(gap_us);
        auto a = clk::now();
        k_busy<<<1, 32, 0, s>>>(d, cycles, i);
        CK(cudaStreamSynchronize(s));
        tl.push_back(std::chrono::duration<double, std::micro>(clk::now() - a).count());
    }

    int h = -1, hl = -1;
    CK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hl, d + 1024, sizeof(int), cudaMemcpyDeviceToHost));
    // Measure the busy kernel alone with events (detects a kernel body that did not run its loop).
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, s));
    k_busy<<<1, 32, 0, s>>>(d, cycles, 7);
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float busy_ms = 0.f;
    CK(cudaEventElapsedTime(&busy_ms, e0, e1));
    const bool ok = (h == 1999) && (hl == long_iters - 1) && busy_ms * 1000.0 > 0.8 * long_us;
    std::printf("[P3-PERCALL-CHECK] last_tiny=%d last_busy=%d (expect %d) busy_kernel_us=%.1f (expect ~%.0f) clk_khz=%d cycles=%lld\n",
                h, hl, long_iters - 1, busy_ms * 1000.0, long_us, clk_khz, cycles);

#ifdef XSCHED_PERCALL_WITH_XSCHED
    if (xq) {
        XQueueDestroy(xq);
        HwQueueDestroy(hwq);
    }
#endif
    std::printf("[P3-PERCALL] %s mode=%s sync1_p50=%.2f sync1_p90=%.2f sync4_p50=%.2f sync4_p90=%.2f async_us=%.2f "
                "long_p50=%.2f long_p90=%.2f long_us=%.0f gap_us=%.0f last=%d\n",
                ok ? "PASS" : "FAIL", mode.c_str(), pct(t1, 0.5), pct(t1, 0.9), pct(t4, 0.5), pct(t4, 0.9),
                pct(ta, 0.5), pct(tl, 0.5), pct(tl, 0.9), long_us, gap_us, h);
    return ok ? 0 : 1;
}
