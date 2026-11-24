#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "xsched/cuda/hal/level3/sm120_trap_engine.h"

using namespace xsched::cuda::sm120;

namespace
{

__global__ void k_bench_null() {}

__global__ void k_bench_wait(uint32_t *d_out,
                             uint32_t *d_started,
                             volatile uint32_t *d_allow_exit,
                             uint32_t touch_smem)
{
    extern __shared__ uint32_t s_dyn[];
    const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) {
        if (touch_smem != 0u) s_dyn[0] = blockIdx.x;
        atomicAdd(d_started, 1u);
        __threadfence_system();
    }
    uint32_t acc = gtid ^ 0x9e3779b9u;
    while (*d_allow_exit == 0u) {
        acc = acc * 1664525u + 1013904223u;
    }
    d_out[gtid] = (acc ^ acc) + gtid + (touch_smem != 0u ? (s_dyn[0] & 0u) : 0u);
}

struct Stats
{
    double min = 0, p50 = 0, p95 = 0, p99 = 0, max = 0;
};

Stats CalcStats(std::vector<double> v)
{
    Stats s{};
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    s.min = v.front();
    s.p50 = v[n / 2];
    s.p95 = v[(n * 95) / 100];
    s.p99 = v[(n * 99) / 100];
    s.max = v.back();
    return s;
}

CUfunction ResolveFn(const void *sym)
{
    cudaFunction_t f = nullptr;
    if (cudaGetFuncBySymbol(&f, sym) != cudaSuccess) return nullptr;
    return reinterpret_cast<CUfunction>(f);
}

bool PollStarted(volatile uint32_t *flag, uint32_t target)
{
    const auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) < target) {
        if (std::chrono::steady_clock::now() >= dl) return false;
    }
    return true;
}

void BenchEvictRestore(CUcontext ctx,
                       CUdevice dev,
                       CUstream stream,
                       const std::shared_ptr<Sm120TrapEngine> &eng,
                       Sm120PreemptMode mode,
                       uint32_t num_ctas,
                       uint32_t dyn_smem,
                       int iters,
                       const char *label)
{
    Sm120StreamContext sctx(ctx, dev, eng);
    CUfunction fn = ResolveFn(reinterpret_cast<const void *>(k_bench_wait));
    if (dyn_smem > 0) {
        cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(dyn_smem));
    }

    uint32_t *d_out = nullptr;
    uint32_t *h_started = nullptr;
    CUdeviceptr d_started = 0, d_allow = 0;
    cudaMalloc(&d_out, static_cast<size_t>(num_ctas) * 128u * sizeof(uint32_t));
    cuMemAlloc(&d_allow, sizeof(uint32_t));
    cudaHostAlloc(&h_started, sizeof(uint32_t), cudaHostAllocMapped);
    void *p1 = nullptr;
    cudaHostGetDevicePointer(&p1, h_started, 0);
    d_started = reinterpret_cast<CUdeviceptr>(p1);

    uint32_t touch_smem = (dyn_smem > 0u) ? 1u : 0u;
    void *args[4] = {&d_out, &d_started, &d_allow, &touch_smem};
    std::vector<double> evict_us, restore_us;
    evict_us.reserve(iters);
    restore_us.reserve(iters);
    size_t arena_kb = 0;

    for (int it = 0; it < iters; ++it) {
        sctx.ResetControlSync();
        __atomic_store_n(h_started, 0u, __ATOMIC_RELEASE);
        cuMemsetD32(d_allow, 0u, 1u);

        sctx.PrepareLaunch(fn, num_ctas, 128, dyn_smem, static_cast<uint64_t>(50 + it), mode, stream);
        cuLaunchKernel(fn, num_ctas, 1, 1, 128, 1, 1, dyn_smem, stream, args, nullptr);
        if (!PollStarted(h_started, std::min(num_ctas, 42u))) {
            cuMemsetD32Async(d_allow, 1u, 1u, stream);
            cuStreamSynchronize(stream);
            std::printf("  [%s] timeout!\n", label);
            break;
        }

        const auto te0 = std::chrono::steady_clock::now();
        sctx.PreemptStream(mode);
        cuStreamSynchronize(stream);
        const auto te1 = std::chrono::steady_clock::now();
        evict_us.push_back(std::chrono::duration<double, std::micro>(te1 - te0).count());

        cuMemsetD32(d_allow, 1u, 1u);
        const auto tr0 = std::chrono::steady_clock::now();
        sctx.PrepareRestore(mode);
        sctx.PrepareLaunch(fn, num_ctas, 128, dyn_smem, static_cast<uint64_t>(50 + it), mode, stream);
        cuLaunchKernel(fn, num_ctas, 1, 1, 128, 1, 1, dyn_smem, stream, args, nullptr);
        cuStreamSynchronize(stream);
        const auto tr1 = std::chrono::steady_clock::now();
        restore_us.push_back(std::chrono::duration<double, std::micro>(tr1 - tr0).count());
        arena_kb = sctx.Telemetry().arena_bytes >> 10;
    }

    const auto es = CalcStats(evict_us);
    const auto rs = CalcStats(restore_us);
    std::printf("  %-34s iters=%2d | Evict P50=%6.1f P95=%6.1f Max=%6.1f us | "
                "Restore(0-pulse) P50=%6.1f P95=%6.1f Max=%6.1f us | u_arena=%zu KiB\n",
                label, iters, es.p50, es.p95, es.max, rs.p50, rs.p95, rs.max, arena_kb);

    cudaFreeHost(h_started);
    cuMemFree(d_allow);
    cudaFree(d_out);
}

} // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    cuInit(0);
    CUdevice dev = 0;
    cuDeviceGet(&dev, 0);
    char name[128] = {};
    cuDeviceGetName(name, sizeof(name), dev);

    cudaSetDevice(0);
    cudaFree(0);
    CUcontext ctx = nullptr;
    cuDevicePrimaryCtxRetain(&ctx, dev);
    cuCtxSetCurrent(ctx);
    CUstream stream = nullptr;
    cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);

    auto eng = Sm120TrapEngine::Instance(ctx);
    eng->Init();

    std::printf("[SM120_V2_BENCH] GPU=%s SMs=%d\n", name, eng->SmCount());

    // 1. Entry Guardian cmd==0 overhead vs uninstrumented baseline.
    CUfunction fnull = ResolveFn(reinterpret_cast<const void *>(k_bench_null));
    constexpr int kLaunchIters = 100;
    // Warmup uninstrumented
    for (int i = 0; i < 10; ++i) cuLaunchKernel(fnull, 84, 1, 1, 128, 1, 1, 0, stream, nullptr, nullptr);
    cuStreamSynchronize(stream);
    std::vector<double> base_us, guard_us;
    for (int i = 0; i < kLaunchIters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        cuLaunchKernel(fnull, 84, 1, 1, 128, 1, 1, 0, stream, nullptr, nullptr);
        cuStreamSynchronize(stream);
        const auto t1 = std::chrono::steady_clock::now();
        base_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    Sm120StreamContext sctx_g(ctx, dev, eng);
    sctx_g.ResetControlSync();
    sctx_g.PrepareLaunch(fnull, 84, 128, 0, 1, Sm120PreemptMode::kEarlyExit, stream);
    cuStreamSynchronize(stream);
    for (int i = 0; i < kLaunchIters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        cuLaunchKernel(fnull, 84, 1, 1, 128, 1, 1, 0, stream, nullptr, nullptr);
        cuStreamSynchronize(stream);
        const auto t1 = std::chrono::steady_clock::now();
        guard_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    const auto bs = CalcStats(base_us);
    const auto gs = CalcStats(guard_us);
    std::printf("\n1) Entry Guardian Launch-Sync Overhead (84x128 null kernel, N=%d):\n"
                "   Baseline P50=%.2f us (P95=%.2f) | Guarded cmd=0 P50=%.2f us (P95=%.2f) | Delta=%.2f us\n",
                kLaunchIters, bs.p50, bs.p95, gs.p50, gs.p95, gs.p50 - bs.p50);

    // 2. Mode 1 vs Mode 5 Evict + 0-Pulse Restore latency (SMEM=0 vs SMEM=32 KiB)
    std::printf("\n2) Mode 1 vs Mode 5 Preemption & 0-Pulse Restore Latency:\n");
    BenchEvictRestore(ctx, dev, stream, eng, Sm120PreemptMode::kEarlyExit, 512, 49152, 20, "Mode 1 (512 CTAs, 48K SMEM)");
    BenchEvictRestore(ctx, dev, stream, eng, Sm120PreemptMode::kTrapFrame, 84,  0,     20, "Mode 5 (84 CTAs, 0K SMEM)");
    BenchEvictRestore(ctx, dev, stream, eng, Sm120PreemptMode::kTrapFrame, 84,  32768, 20, "Mode 5 (84 CTAs, 32K SMEM)");
    BenchEvictRestore(ctx, dev, stream, eng, Sm120PreemptMode::kTrapFrame, 512, 49152, 20, "Mode 5 (512 CTAs, 48K SMEM)");

    // 3. Compact-Slot Pooled u_arena vs Naive v1 Per-Grid Fixed-SMEM Arena Scaling
    std::printf("\n3) Compact-Slot u_arena Memory Scaling (128 threads/CTA, ~32 regs, 0 SMEM vs 64 KiB fixed v1):\n");
    CUfunction fwait = ResolveFn(reinterpret_cast<const void *>(k_bench_wait));
    for (uint32_t g : {256u, 1024u, 4096u, 16384u}) {
        Sm120StreamContext tmp_ctx(ctx, dev, eng);
        tmp_ctx.PrepareLaunch(fwait, g, 128, 0, 1, Sm120PreemptMode::kTrapFrame, stream);
        const size_t v2_bytes = tmp_ctx.Telemetry().arena_bytes;
        const size_t v1_naive_bytes = static_cast<size_t>(g) * (65536ULL + 128ULL * 1536ULL);
        std::printf("   Grid=%5u CTAs | v1 Naive=%7.2f MiB | v2 Compact Pooled=%6.2f MiB (%.1fx smaller)\n",
                    g,
                    static_cast<double>(v1_naive_bytes) / (1024.0 * 1024.0),
                    static_cast<double>(v2_bytes) / (1024.0 * 1024.0),
                    static_cast<double>(v1_naive_bytes) / static_cast<double>(std::max<size_t>(v2_bytes, 1)));
    }

    cuStreamDestroy(stream);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
