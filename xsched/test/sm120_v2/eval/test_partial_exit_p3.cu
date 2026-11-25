// Phase-3 targeted SM120 Level-3 (Mode 5) trap-frame test: warps trapped while PARTIALLY EXITED
// or intra-warp DIVERGENT.
//
// Motivation: Suite 7 (bench_prod_ops_p3.py) found deterministic Mode 5 restore faults
// (cuda error 700/717) on PyTorch's unrolled_elementwise_kernel<direct_copy_kernel_cuda>
// (23 EXIT, nested BSSY B0/B1/B6/B7), while vectorized elementwise kernels pass.
// This test isolates the warp-state patterns with synthetic long-running single-wave kernels:
//
//   K0 uniform          : all lanes run the same trip count (control)
//   K1 half_exit        : odd lanes EXIT at entry; even lanes run the long loop
//   K2 staggered_exit   : lane i runs (base + i*step) iterations then EXITs -> warps are
//                         partially exited at trap time
//   K3 staggered_reconv : lane i runs (base + i*step) iterations, then the warp reconverges
//                         (__syncwarp) and every lane stores -> divergent but no early exit
//
// Each kernel is a non-idempotent in-place LCG accumulation (acc = acc*A + C, repeated), so a
// lost / duplicated / resurrected lane is detected bit-exactly against a native golden run.
//
// Build:  nvcc -arch=sm_120 -O2 -I$XSCHED/include test_partial_exit_p3.cu -o test_partial_exit_p3 \
//              -L$XSCHED/lib -lpreempt -lhalcuda -lcuda
// Run  :  sudo env LD_LIBRARY_PATH=$XSCHED/lib:... ./test_partial_exit_p3 [K0|K1|K2|K3|all] [runs] [np]
#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", (int)e_, cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(2); } } while (0)

// Host->device upload that is complete before returning. A synchronous cudaMemcpy from PAGEABLE
// memory may return once the data is staged, before the DMA reaches the destination; kernels on
// NON-BLOCKING streams are not ordered after it. (This race produced false "contiguous tail of
// wrong CTAs" failures in the first version of these tests.)
#define H2D(dst, src, bytes) do { CK(cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice)); \
                                  CK(cudaDeviceSynchronize()); } while (0)

__device__ __forceinline__ uint32_t lcg_iter(uint32_t a, int n)
{
    for (int i = 0; i < n; ++i) {
        a = a * 1664525u + 1013904223u;
        a ^= (a >> 13);
    }
    return a;
}

// mode: 0 uniform, 1 half_exit, 2 staggered_exit, 3 staggered_reconv
__global__ void k_warp_patterns(uint32_t *acc, int base, int step, int mode)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    uint32_t a = acc[gid];
    if (mode == 1) {
        if (lane & 1) return;               // odd lanes exit immediately (partial warp)
        a = lcg_iter(a, base);
        acc[gid] = a;
        return;
    }
    if (mode == 2) {
        a = lcg_iter(a, base + lane * step); // lanes finish at different times...
        acc[gid] = a;
        return;                              // ...and EXIT one by one (partially exited warp)
    }
    if (mode == 3) {
        a = lcg_iter(a, base + lane * step); // divergent trip counts
        __syncwarp();                        // reconverge before the store
        acc[gid] = a;
        return;
    }
    a = lcg_iter(a, base);
    acc[gid] = a;
}

struct Case { const char *name; int mode; };

int main(int argc, char **argv)
{
    const std::string which = argc > 1 ? argv[1] : "all";
    const int runs = argc > 2 ? std::atoi(argv[2]) : 5;
    const int np = argc > 3 ? std::atoi(argv[3]) : 3;

    int dev = 0, sms = 0;
    CK(cudaSetDevice(dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const int threads = 256;
    const int blocks = sms * 2;              // single wave -> preemption catches State-B CTAs
    const size_t n = static_cast<size_t>(blocks) * threads;

    cudaStream_t gold_s, lp;
    CK(cudaStreamCreateWithFlags(&gold_s, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&lp, cudaStreamNonBlocking));

    // Calibrate a ~40 ms kernel on the golden stream.
    uint32_t *d_acc = nullptr, *d_gold = nullptr;
    CK(cudaMalloc(&d_acc, n * 4));
    CK(cudaMalloc(&d_gold, n * 4));
    std::vector<uint32_t> h_init(n), h_out(n), h_gold(n);
    for (size_t i = 0; i < n; ++i) h_init[i] = static_cast<uint32_t>(i * 2654435761u + 12345u);
    int base = 20000;
    {
        H2D(d_acc, h_init.data(), n * 4);
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0, gold_s));
        k_warp_patterns<<<blocks, threads, 0, gold_s>>>(d_acc, base, 0, 0);
        CK(cudaEventRecord(e1, gold_s)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        base = static_cast<int>(base * (40.0f / std::max(ms, 0.05f)));
    }
    const int step = base / 64;              // lane 31 runs ~1.5x lane 0's trip count

    HwQueueHandle hwq; XQueueHandle xq;
    if (CudaQueueCreate(&hwq, lp) != kXSchedSuccess ||
        XQueueCreate(&xq, hwq, kPreemptLevelInterrupt, kQueueCreateFlagNone) != kXSchedSuccess) {
        std::fprintf(stderr, "XQueueCreate(Level-3) failed\n");
        return 2;
    }

    const Case cases[] = {{"K0 uniform", 0}, {"K1 half_exit", 1}, {"K2 staggered_exit", 2},
                          {"K3 staggered_reconv", 3}};
    int fails = 0;
    for (const auto &c : cases) {
        if (which != "all" && which != std::string(c.name).substr(0, 2)) continue;
        // Golden (native stream, no XQueue).
        H2D(d_gold, h_init.data(), n * 4);
        k_warp_patterns<<<blocks, threads, 0, gold_s>>>(d_gold, base, step, c.mode);
        CK(cudaStreamSynchronize(gold_s));
        CK(cudaMemcpy(h_gold.data(), d_gold, n * 4, cudaMemcpyDeviceToHost));

        int ok_runs = 0;
        double susp_sum = 0; int susp_n = 0;
        for (int r = 0; r < runs; ++r) {
            H2D(d_acc, h_init.data(), n * 4);
            k_warp_patterns<<<blocks, threads, 0, lp>>>(d_acc, base, step, c.mode);
            for (int p = 0; p < np; ++p) {
                std::this_thread::sleep_for(std::chrono::microseconds(6000 + 3000 * p));
                auto t0 = std::chrono::steady_clock::now();
                XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue);
                auto t1 = std::chrono::steady_clock::now();
                susp_sum += std::chrono::duration<double, std::milli>(t1 - t0).count(); ++susp_n;
                XQueueResume(xq, kQueueResumeFlagNone);
            }
            cudaError_t se = cudaStreamSynchronize(lp);
            if (se != cudaSuccess) {
                std::printf("[P3-PARTIAL %s] run %d: CUDA error %d (%s)\n", c.name, r, (int)se,
                            cudaGetErrorString(se));
                std::printf("[P3-PARTIAL %s] FAIL (context lost)\n", c.name);
                return 1;
            }
            CK(cudaMemcpy(h_out.data(), d_acc, n * 4, cudaMemcpyDeviceToHost));
            size_t bad = 0, first = n;
            for (size_t i = 0; i < n; ++i) if (h_out[i] != h_gold[i]) { if (first == n) first = i; ++bad; }
            if (bad == 0) ++ok_runs;
            else std::printf("  run %d: mismatched=%zu/%zu first_bad=%zu (lane %zu)\n", r, bad, n, first, first % 32);
        }
        const bool pass = ok_runs == runs;
        fails += pass ? 0 : 1;
        std::printf("[P3-PARTIAL %s] %s | runs=%d ok=%d np=%d susp_avg=%.3f ms | blocks=%d thr=%d base=%d step=%d\n",
                    c.name, pass ? "PASS" : "FAIL", runs, ok_runs, np, susp_n ? susp_sum / susp_n : 0.0,
                    blocks, threads, base, step);
    }
    XQueueDestroy(xq);
    HwQueueDestroy(hwq);
    return fails ? 1 : 0;
}
