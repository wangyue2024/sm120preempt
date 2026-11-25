// Phase-3 targeted SM120 Level-3 (Mode 5) test: deeply nested divergent loops with `break`,
// which make the compiler allocate HIGH-INDEX convergence barriers (BSSY/BREAK/BSYNC B4..B7,
// including the .RELIABLE forms) -- the pattern seen at the trap PCs of DEF-14
// (PyTorch unrolled_elementwise_kernel<direct_copy>, trapped at `@P0 BREAK.RELIABLE B6`).
//
//   C0 shallow : one divergent loop with break            (low barrier indices, control)
//   C1 deep    : 6 nested data-dependent loops, each with a divergent break
//   D1..D6     : depth sweep (template nest<D>) to locate the barrier-index threshold
//   R0..R3     : register-pressure sweep (k_regs<EXTRA>) -> exercises the R14+ save/restore cascade
//   Q3         : k_regs_dump<48>: per-accumulator mismatch attribution (which live values are lost)
//
// Non-idempotent in-place accumulation; bit-exact comparison against a native golden run.
// Build/run like test_partial_exit_p3.cu (cases: C0 C1 D1..D6 R0..R3 or all):
//   nvcc -arch=sm_120 -O3 -I$XSCHED/include test_convbar_p3.cu -o test_convbar_p3 -L$XSCHED/lib -lpreempt -lhalcuda -lcuda
//   sudo env LD_LIBRARY_PATH=$XSCHED/lib:... ./test_convbar_p3 [C0|C1|all] [runs] [np]
#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", (int)e_, cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(2); } } while (0)

__device__ __forceinline__ uint32_t mix(uint32_t a) { a = a * 1664525u + 1013904223u; return a ^ (a >> 13); }

// Depth-D nest of data-dependent loops, each with a divergent break (compiler allocates ~B0..B(D-1)).
template <int D>
__device__ __forceinline__ void nest(uint32_t &a, uint32_t key)
{
    for (int i = 0; i < 4; ++i) {
        a = mix(a);
        if ((a & 7u) == key) break;
        if constexpr (D > 1) nest<D - 1>(a, key + 1);
        else a ^= static_cast<uint32_t>(i * 977);
    }
}

template <int D>
__global__ void k_depth(uint32_t *acc, int outer)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t a = acc[gid];
    for (int o = 0; o < outer; ++o) nest<D>(a, 1u);
    acc[gid] = a;
}

// Register-pressure variant: one divergent loop (B0 only) plus EXTRA independent accumulators that
// stay live across the whole loop, pushing the allocation past R13 (R14+ are saved by the trap
// engine's Cave 2/5 register cascade, R0..R13 by the 112-byte hardware frame).
template <int EXTRA>
__global__ void k_regs(uint32_t *acc, int outer)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t a = acc[gid];
    uint32_t x[EXTRA];
#pragma unroll
    for (int k = 0; k < EXTRA; ++k) x[k] = a + 0x9e3779b9u * (k + 1);
    for (int o = 0; o < outer; ++o) {
        for (int i = 0; i < 64; ++i) {
            a = mix(a);
#pragma unroll
            for (int k = 0; k < EXTRA; ++k) x[k] = x[k] * 747796405u + a;
            if ((a & 63u) == (uint32_t)(threadIdx.x & 63)) break;
        }
    }
#pragma unroll
    for (int k = 0; k < EXTRA; ++k) a ^= x[k];
    acc[gid] = a;
}

// Same as k_regs but stores every accumulator separately (acc2[gid*EXTRA + k]) so a mismatch can be
// attributed to individual live values / registers.
template <int EXTRA>
__global__ void k_regs_dump(uint32_t *acc, uint32_t *acc2, int outer)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t a = acc[gid];
    uint32_t x[EXTRA];
#pragma unroll
    for (int k = 0; k < EXTRA; ++k) x[k] = a + 0x9e3779b9u * (k + 1);
    for (int o = 0; o < outer; ++o) {
        for (int i = 0; i < 64; ++i) {
            a = mix(a);
#pragma unroll
            for (int k = 0; k < EXTRA; ++k) x[k] = x[k] * 747796405u + a;
            if ((a & 63u) == (uint32_t)(threadIdx.x & 63)) break;
        }
    }
    acc[gid] = a;
#pragma unroll
    for (int k = 0; k < EXTRA; ++k) acc2[(size_t)gid * EXTRA + k] = x[k];
}

__global__ void k_shallow(uint32_t *acc, int outer)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t a = acc[gid];
    for (int o = 0; o < outer; ++o) {
        for (int i = 0; i < 64; ++i) {
            a = mix(a);
            if ((a & 63u) == (uint32_t)(threadIdx.x & 63)) break;   // divergent break
        }
    }
    acc[gid] = a;
}

__global__ void k_deep(uint32_t *acc, int outer)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t a = acc[gid];
    for (int o = 0; o < outer; ++o) {
        for (int i0 = 0; i0 < 4; ++i0) {
            a = mix(a);
            if ((a & 7u) == 1u) break;
            for (int i1 = 0; i1 < 4; ++i1) {
                a = mix(a);
                if ((a & 7u) == 2u) break;
                for (int i2 = 0; i2 < 4; ++i2) {
                    a = mix(a);
                    if ((a & 7u) == 3u) break;
                    for (int i3 = 0; i3 < 4; ++i3) {
                        a = mix(a);
                        if ((a & 7u) == 4u) break;
                        for (int i4 = 0; i4 < 4; ++i4) {
                            a = mix(a);
                            if ((a & 7u) == 5u) break;
                            for (int i5 = 0; i5 < 4; ++i5) {
                                a = mix(a);
                                if ((a & 7u) == 6u) break;
                                a ^= (uint32_t)(i0 * 4096 + i1 * 512 + i2 * 64 + i3 * 8 + i4 + i5);
                            }
                        }
                    }
                }
            }
        }
    }
    acc[gid] = a;
}

int main(int argc, char **argv)
{
    const std::string which = argc > 1 ? argv[1] : "all";
    const int runs = argc > 2 ? std::atoi(argv[2]) : 5;
    const int np = argc > 3 ? std::atoi(argv[3]) : 3;
    int sms = 0;
    CK(cudaSetDevice(0));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int threads = 256, blocks = sms * 2;
    const size_t n = static_cast<size_t>(blocks) * threads;
    cudaStream_t gold_s, lp;
    CK(cudaStreamCreateWithFlags(&gold_s, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&lp, cudaStreamNonBlocking));
    uint32_t *d_acc, *d_gold;
    CK(cudaMalloc(&d_acc, n * 4)); CK(cudaMalloc(&d_gold, n * 4));
    std::vector<uint32_t> h_init(n), h_out(n), h_gold(n);
    for (size_t i = 0; i < n; ++i) h_init[i] = static_cast<uint32_t>(i * 2654435761u + 7u);

    if (which == "Q3") {   // per-accumulator attribution for the 55-register kernel
        constexpr int E = 48;
        uint32_t *d2 = nullptr, *g2 = nullptr; CK(cudaMalloc(&d2, n * E * 4)); CK(cudaMalloc(&g2, n * E * 4));
        std::vector<uint32_t> h2(n * E), hg2(n * E);
        HwQueueHandle qh; XQueueHandle qx;
        if (CudaQueueCreate(&qh, lp) != kXSchedSuccess ||
            XQueueCreate(&qx, qh, kPreemptLevelInterrupt, kQueueCreateFlagNone) != kXSchedSuccess) return 2;
        const int outer = 3000;
        CK(cudaMemcpy(d_gold, h_init.data(), n * 4, cudaMemcpyHostToDevice));
        k_regs_dump<E><<<blocks, threads, 0, gold_s>>>(d_gold, g2, outer);
        CK(cudaStreamSynchronize(gold_s)); CK(cudaMemcpy(hg2.data(), g2, n * E * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_gold.data(), d_gold, n * 4, cudaMemcpyDeviceToHost));
        for (int r = 0; r < runs; ++r) {
            CK(cudaMemcpy(d_acc, h_init.data(), n * 4, cudaMemcpyHostToDevice));
            k_regs_dump<E><<<blocks, threads, 0, lp>>>(d_acc, d2, outer);
            for (int p = 0; p < np; ++p) {
                std::this_thread::sleep_for(std::chrono::microseconds(6000 + 3000 * p));
                XQueueSuspend(qx, kQueueSuspendFlagSyncHwQueue); XQueueResume(qx, kQueueResumeFlagNone);
            }
            CK(cudaStreamSynchronize(lp));
            CK(cudaMemcpy(h2.data(), d2, n * E * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h_out.data(), d_acc, n * 4, cudaMemcpyDeviceToHost));
            size_t bad_a = 0; for (size_t i = 0; i < n; ++i) bad_a += h_out[i] != h_gold[i];
            std::printf("[P3-CONVBAR Q3 run %d] a(final) wrong=%zu/%zu | per-accumulator wrong threads:", r, bad_a, n);
            for (int k = 0; k < E; ++k) { size_t b = 0; for (size_t i = 0; i < n; ++i) b += h2[i * E + k] != hg2[i * E + k]; std::printf(" x%d=%zu", k, b); }
            std::printf("\n");
        }
        XQueueDestroy(qx); HwQueueDestroy(qh);
        return 0;
    }
    HwQueueHandle hwq; XQueueHandle xq;
    if (CudaQueueCreate(&hwq, lp) != kXSchedSuccess ||
        XQueueCreate(&xq, hwq, kPreemptLevelInterrupt, kQueueCreateFlagNone) != kXSchedSuccess) {
        std::fprintf(stderr, "XQueueCreate(Level-3) failed\n"); return 2;
    }
    int fails = 0;
    const char *names[] = {"C0 shallow", "C1 deep", "D1 depth1", "D2 depth2", "D3 depth3", "D4 depth4",
                           "D5 depth5", "D6 depth6", "R0 regs+0", "R1 regs+4", "R2 regs+16", "R3 regs+48"};
    for (int c = 0; c < 12; ++c) {
        const char *name = names[c];
        if (which != "all" && which != std::string(name).substr(0, 2)) continue;
        auto launch = [&](uint32_t *p, cudaStream_t s, int outer) {
            switch (c) {
            case 0: k_shallow<<<blocks, threads, 0, s>>>(p, outer); break;
            case 1: k_deep<<<blocks, threads, 0, s>>>(p, outer); break;
            case 2: k_depth<1><<<blocks, threads, 0, s>>>(p, outer); break;
            case 3: k_depth<2><<<blocks, threads, 0, s>>>(p, outer); break;
            case 4: k_depth<3><<<blocks, threads, 0, s>>>(p, outer); break;
            case 5: k_depth<4><<<blocks, threads, 0, s>>>(p, outer); break;
            case 6: k_depth<5><<<blocks, threads, 0, s>>>(p, outer); break;
            case 7: k_depth<6><<<blocks, threads, 0, s>>>(p, outer); break;
            case 8: k_regs<1><<<blocks, threads, 0, s>>>(p, outer); break;
            case 9: k_regs<4><<<blocks, threads, 0, s>>>(p, outer); break;
            case 10: k_regs<16><<<blocks, threads, 0, s>>>(p, outer); break;
            default: k_regs<48><<<blocks, threads, 0, s>>>(p, outer); break;
            }
        };
        // calibrate ~40 ms on the golden stream
        int outer = 64;
        {
            CK(cudaMemcpy(d_gold, h_init.data(), n * 4, cudaMemcpyHostToDevice));
            cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            CK(cudaEventRecord(e0, gold_s)); launch(d_gold, gold_s, outer); CK(cudaEventRecord(e1, gold_s));
            CK(cudaEventSynchronize(e1)); float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            outer = std::max(1, static_cast<int>(outer * 40.0f / std::max(ms, 0.01f)));
        }
        CK(cudaMemcpy(d_gold, h_init.data(), n * 4, cudaMemcpyHostToDevice));
        launch(d_gold, gold_s, outer);
        CK(cudaStreamSynchronize(gold_s));
        CK(cudaMemcpy(h_gold.data(), d_gold, n * 4, cudaMemcpyDeviceToHost));
        int ok = 0;
        for (int r = 0; r < runs; ++r) {
            CK(cudaMemcpy(d_acc, h_init.data(), n * 4, cudaMemcpyHostToDevice));
            launch(d_acc, lp, outer);
            for (int p = 0; p < np; ++p) {
                std::this_thread::sleep_for(std::chrono::microseconds(6000 + 3000 * p));
                XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue);
                XQueueResume(xq, kQueueResumeFlagNone);
            }
            cudaError_t se = cudaStreamSynchronize(lp);
            if (se != cudaSuccess) {
                std::printf("[P3-CONVBAR %s] run %d: CUDA error %d (%s) -> FAIL (context lost)\n", name, r, (int)se,
                            cudaGetErrorString(se));
                return 1;
            }
            CK(cudaMemcpy(h_out.data(), d_acc, n * 4, cudaMemcpyDeviceToHost));
            size_t bad = 0; for (size_t i = 0; i < n; ++i) bad += (h_out[i] != h_gold[i]);
            if (bad == 0) ++ok; else std::printf("  run %d: mismatched=%zu/%zu\n", r, bad, n);
            if (bad != 0 && std::getenv("MISMATCH_DETAIL")) {
                // Histogram of wrong threads by lane, by warp-in-CTA, and number of CTAs affected.
                int lane_h[32] = {}, warp_h[32] = {}; std::vector<int> cta_h(blocks, 0);
                for (size_t i = 0; i < n; ++i) if (h_out[i] != h_gold[i]) {
                    lane_h[i % 32]++; warp_h[(i % threads) / 32]++; cta_h[i / threads]++;
                }
                int ctas = 0, full_warps = 0; for (int b = 0; b < blocks; ++b) ctas += cta_h[b] > 0;
                for (size_t w = 0; w < n / 32; ++w) { int c = 0; for (int l = 0; l < 32; ++l) c += h_out[w*32+l] != h_gold[w*32+l]; full_warps += (c == 32); }
                std::printf("    lanes:"); for (int l = 0; l < 32; ++l) std::printf(" %d", lane_h[l]);
                std::printf("\n    warp-in-cta:"); for (int w = 0; w < threads / 32; ++w) std::printf(" %d", warp_h[w]);
                std::printf("\n    ctas_affected=%d/%d fully_wrong_warps=%d\n", ctas, blocks, full_warps);
            }
        }
        fails += (ok != runs);
        std::printf("[P3-CONVBAR %s] %s | runs=%d ok=%d np=%d outer=%d blocks=%d thr=%d\n", name,
                    ok == runs ? "PASS" : "FAIL", runs, ok, np, outer, blocks, threads);
    }
    XQueueDestroy(xq); HwQueueDestroy(hwq);
    return fails ? 1 : 0;
}
