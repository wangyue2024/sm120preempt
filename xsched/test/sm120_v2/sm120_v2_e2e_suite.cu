#include "xsched/cuda/hal/common/cuda_command.h"
#include "xsched/cuda/hal/level3/cuda_queue.h"
#include "xsched/cuda/hal/level3/sm120_trap_engine.h"
#include "xsched/cuda/hal/level3/sm120_guardian.h"
#include "xsched/preempt/xqueue/async_xqueue.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

using namespace xsched::cuda;
using namespace xsched::cuda::sm120;
using namespace xsched::preempt;

namespace xsched::cuda
{
CUresult DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream);
}

namespace
{

#define CHK_CU(call)                                                                 \
    do {                                                                             \
        CUresult _r = (call);                                                        \
        if (_r != CUDA_SUCCESS) {                                                    \
            const char *en = nullptr;                                                \
            cuGetErrorName(_r, &en);                                                 \
            std::fprintf(stderr, "CUDA error %d (%s) at %s:%d\n",                   \
                         (int)_r, en ? en : "?", __FILE__, __LINE__);                \
            return false;                                                            \
        }                                                                            \
    } while (0)

#define CHK_RT(call)                                                                 \
    do {                                                                             \
        cudaError_t _r = (call);                                                     \
        if (_r != cudaSuccess) {                                                     \
            std::fprintf(stderr, "Runtime error %d (%s) at %s:%d\n",                \
                         (int)_r, cudaGetErrorString(_r), __FILE__, __LINE__);       \
            return false;                                                            \
        }                                                                            \
    } while (0)

CUfunction ResolveCuFunc(const void *symbol)
{
    cudaFunction_t fn = nullptr;
    if (cudaGetFuncBySymbol(&fn, symbol) != cudaSuccess || !fn) return nullptr;
    return reinterpret_cast<CUfunction>(fn);
}

bool PollHostFlag(volatile uint32_t *flag, uint32_t target, int timeout_ms)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) < target) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
    return true;
}

template <int BAR_ID>
__device__ __forceinline__ void NamedBarSync(int thread_cnt)
{
    asm volatile("bar.sync %0, %1;" : : "n"(BAR_ID), "r"(thread_cnt) : "memory");
}

template <int BAR_ID>
__device__ __forceinline__ void NamedBarArrive(int thread_cnt)
{
    asm volatile("bar.arrive %0, %1;" : : "n"(BAR_ID), "r"(thread_cnt) : "memory");
}

// --------------------------------------------------------------------------
// Test Kernels
// --------------------------------------------------------------------------

// Case 1: Idempotent multi-wave kernel (Mode 1)
__global__ void k_mode1_idempotent(uint32_t *d_entry_cnt,
                                   uint32_t *d_out,
                                   uint32_t *d_started,
                                   volatile uint32_t *d_allow_exit,
                                   uint32_t *d_fast_ctas,
                                   uint32_t fast_threshold)
{
    extern __shared__ uint32_t s_pad[];
    if (threadIdx.x == 0) {
        s_pad[0] = blockIdx.x;
        atomicAdd(&d_entry_cnt[blockIdx.x], 1u);
        atomicAdd(d_started, 1u);
        __threadfence_system();
    }
    __syncthreads();

    if (blockIdx.x >= fast_threshold) {
        while (atomicAdd(const_cast<uint32_t *>(d_allow_exit), 0u) == 0u) {
            __nanosleep(200);
        }
    } else if (threadIdx.x == 0 && d_fast_ctas != nullptr) {
        atomicAdd(d_fast_ctas, 1u);
        __threadfence_system();
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = blockIdx.x * 13u + 7u + (s_pad[0] & 0u);
    }
}

// Case 2, 3, 11: Non-idempotent atomicAdd kernel with configurable fast State A subset
__global__ void k_nonidemp_multiwave(unsigned long long *d_total,
                                     uint32_t *d_pre_cta,
                                     uint32_t *d_post_cta,
                                     uint32_t *d_fast_done,
                                     uint32_t *d_active_started,
                                     volatile uint32_t *d_allow_exit,
                                     uint32_t fast_ctas,
                                     uint32_t steps_per_thread,
                                     uint32_t touch_smem)
{
    extern __shared__ uint32_t s_pad[];
    const uint32_t b = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (tid == 0 && touch_smem != 0u) s_pad[0] = b;
    __syncthreads();

    if (b < fast_ctas) {
        for (uint32_t s = 0; s < steps_per_thread; ++s) {
            atomicAdd(d_total, 1ULL);
        }
        __syncthreads();
        if (tid == 0) {
            atomicAdd(&d_pre_cta[b], 1u);
            atomicAdd(&d_post_cta[b], 1u);
            __threadfence_system();
            atomicAdd(d_fast_done, 1u);
        }
        return;
    }

    if (tid == 0) {
        atomicAdd(&d_pre_cta[b], 1u);
        atomicAdd(d_active_started, 1u);
        __threadfence_system();
    }

    const uint32_t half = steps_per_thread / 2u;
    for (uint32_t s = 0; s < half; ++s) {
        atomicAdd(d_total, 1ULL);
    }
    __syncthreads();

    while (atomicAdd(const_cast<uint32_t *>(d_allow_exit), 0u) == 0u) {
        __nanosleep(200);
    }

    for (uint32_t s = half; s < steps_per_thread; ++s) {
        atomicAdd(d_total, 1ULL);
    }
    if (tid == 0) {
        atomicAdd(&d_post_cta[b], 1u + (touch_smem != 0u ? (s_pad[0] & 0u) : 0u));
    }
}

// Case 4: Full 3D Grid x 3D Block non-idempotent check
__global__ void k_3d_grid_block(uint32_t *d_out,
                                uint32_t *d_pre_cnt,
                                uint32_t *d_started,
                                volatile uint32_t *d_allow_exit)
{
    const uint32_t cta_id = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const uint32_t tpc = blockDim.x * blockDim.y * blockDim.z;
    const uint32_t gtid = cta_id * tpc + tid;

    uint32_t acc = atomicAdd(&d_pre_cnt[gtid], 1u) * 1000000u;
    acc += cta_id * 131u + tid * 17u + 3u;

    if (tid == 0) {
        atomicAdd(d_started, 1u);
        __threadfence_system();
    }
    __syncthreads();

    while (atomicAdd(const_cast<uint32_t *>(d_allow_exit), 0u) == 0u) {
        __nanosleep(200);
    }

    acc ^= (blockIdx.x * 7u + blockIdx.y * 19u + blockIdx.z * 53u);
    acc += (threadIdx.x * 3u + threadIdx.y * 11u + threadIdx.z * 29u);
    d_out[gtid] = acc;
}

// Case 5: Templated High Register Pressure + Uniform Registers / Uniform Predicates
template <int N_REGS>
__global__ void __launch_bounds__(64, 1)
k_high_regs_and_ur(uint32_t *d_out,
                   const uint32_t *d_seeds,
                   uint32_t *d_started,
                   volatile uint32_t *d_allow_exit,
                   uint32_t iters)
{
    const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t v[N_REGS];
    const uint32_t base = d_seeds[gtid];
#pragma unroll
    for (int i = 0; i < N_REGS; ++i) {
        v[i] = base ^ static_cast<uint32_t>(i * 2654435761u + gtid);
    }

    // Encourage ptxas to retain live uniform predicates / uniform registers across loop.
    uint32_t u_acc = iters * 17u + 5u;
    for (uint32_t it = 0; it < iters; ++it) {
        if ((it & 1u) == 0u) u_acc += 3u;
        else u_acc ^= 0x5a5a5a5au;
#pragma unroll
        for (int i = 0; i < N_REGS; ++i) {
            v[i] = v[i] * 1664525u + static_cast<uint32_t>(i + it + 1u);
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_started, 1u);
        __threadfence_system();
    }
    __syncthreads();

    while (atomicAdd(const_cast<uint32_t *>(d_allow_exit), 0u) == 0u) {
        __nanosleep(200);
    }

    uint32_t fold = u_acc;
#pragma unroll
    for (int i = 0; i < N_REGS; ++i) {
        fold = (fold * 33u) ^ v[i];
    }
    d_out[gtid] = fold;
}

// Case 6: Static + Dynamic SMEM (36 KiB)
__global__ void k_static_dyn_smem(uint32_t *d_out,
                                  uint32_t dyn_words,
                                  uint32_t *d_started,
                                  volatile uint32_t *d_allow_exit)
{
    __shared__ uint32_t s_static[1024]; // 4 KiB
    extern __shared__ uint32_t s_dyn[]; // 32 KiB+
    const uint32_t tid = threadIdx.x;

    for (uint32_t i = tid; i < 1024u; i += blockDim.x) {
        s_static[i] = (blockIdx.x << 16) ^ (i * 29u + 11u);
    }
    for (uint32_t i = tid; i < dyn_words; i += blockDim.x) {
        s_dyn[i] = (blockIdx.x << 20) ^ (i * 97u + 0x1234u);
    }
    __syncthreads();

    if (tid == 0) {
        atomicAdd(d_started, 1u);
        __threadfence_system();
    }

    while (atomicAdd(const_cast<uint32_t *>(d_allow_exit), 0u) == 0u) {
        __nanosleep(200);
    }
    __syncthreads();

    uint32_t sum = 0;
    for (uint32_t i = tid; i < 1024u; i += blockDim.x) {
        sum ^= s_static[(i + 17u) & 1023u];
    }
    for (uint32_t i = tid; i < dyn_words; i += blockDim.x) {
        sum += s_dyn[(i + 31u) % dyn_words];
    }
    d_out[blockIdx.x * blockDim.x + tid] = sum;
}

// Case 7: Stack Spill LMEM + __noinline__ Recursive Call
__device__ __noinline__ uint32_t RecurseLmemWorker(int depth,
                                                   uint32_t seed,
                                                   uint32_t *d_started,
                                                   volatile uint32_t *d_allow_exit)
{
    uint32_t local_arr[16];
    for (int i = 0; i < 16; ++i) {
        local_arr[i] = seed + static_cast<uint32_t>(depth * 101 + i * 7);
    }

    if (depth == 0) {
        if (threadIdx.x == 0) {
            atomicAdd(d_started, 1u);
            __threadfence_system();
        }
        while (atomicAdd(const_cast<uint32_t *>(d_allow_exit), 0u) == 0u) {
            __nanosleep(200);
        }
        uint32_t s = 0;
        for (int i = 0; i < 16; ++i) s += local_arr[(i + (seed & 3u)) & 15u];
        return s ^ 0x9e3779b9u;
    }

    uint32_t child = RecurseLmemWorker(depth - 1, seed * 3u + 1u, d_started, d_allow_exit);
    uint32_t mix = child;
    for (int i = 0; i < 16; ++i) {
        mix ^= local_arr[(i + (child & 7u)) & 15u] + static_cast<uint32_t>(i * 13);
    }
    return mix;
}

__global__ void k_lmem_recursion(uint32_t *d_out,
                                 uint32_t *d_started,
                                 volatile uint32_t *d_allow_exit)
{
    uint32_t buf[32];
    const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 32; ++i) {
        buf[i] = gtid * 19u + static_cast<uint32_t>(i * 5 + 3);
    }
    uint32_t rec = RecurseLmemWorker(4, gtid + 17u, d_started, d_allow_exit);
    uint32_t acc = rec;
    for (int i = 0; i < 32; ++i) {
        acc += buf[(i + (rec & 7u)) & 31u];
    }
    d_out[gtid] = acc;
}

// Case 8: All 16 Named Barriers (bar.sync 0,1,5,7,11,15 + bar.arrive 11)
__global__ void k_named_barriers_all(uint32_t *d_out,
                                     uint32_t *d_ready_stage,
                                     volatile uint32_t *d_release_bars)
{
    __shared__ uint32_t s_data[256];
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5;
    s_data[tid] = (blockIdx.x + 1u) * 1000u + tid;
    __syncthreads(); // bar.sync 0

    // Pair 0 (warps 0,1): Barrier 5 (warp 0 enters blocking bar.sync 5 before Evict; warp 1 releases after Restore)
    if (warp == 0) {
        s_data[tid] += 55u;
        NamedBarSync<5>(64);
    } else if (warp == 1) {
        if ((tid & 31u) == 0u) {
            atomicAdd(d_ready_stage, 1u);
            __threadfence_system();
        }
        while (atomicAdd(const_cast<uint32_t *>(d_release_bars), 0u) == 0u) {
            __nanosleep(200);
        }
        s_data[tid] += 66u;
        NamedBarSync<5>(64);
    } else if (warp == 2) {
        // Pair 1 (warps 2,3): Warp 2 arrives non-blocking on Barrier 11 BEFORE Evict!
        s_data[tid] += 111u;
        NamedBarArrive<11>(64);
        if ((tid & 31u) == 0u) {
            atomicAdd(d_ready_stage, 1u);
            __threadfence_system();
        }
        while (atomicAdd(const_cast<uint32_t *>(d_release_bars), 0u) == 0u) {
            __nanosleep(200);
        }
    } else if (warp == 3) {
        while (atomicAdd(const_cast<uint32_t *>(d_release_bars), 0u) == 0u) {
            __nanosleep(200);
        }
        s_data[tid] += 222u;
        NamedBarSync<11>(64);
    } else if (warp == 4 || warp == 5) {
        // Pair 2: Warp 4 parks in Barrier 7 while Warp 5 waits for release
        if (warp == 4) {
            s_data[tid] += 77u;
            NamedBarSync<7>(64);
        } else {
            if ((tid & 31u) == 0u) {
                atomicAdd(d_ready_stage, 1u);
                __threadfence_system();
            }
            while (atomicAdd(const_cast<uint32_t *>(d_release_bars), 0u) == 0u) {
                __nanosleep(200);
            }
            s_data[tid] += 88u;
            NamedBarSync<7>(64);
        }
    } else {
        // Pair 3 (warps 6,7): Warp 6 parks in Barrier 15 while Warp 7 waits for release
        if (warp == 6) {
            s_data[tid] += 155u;
            NamedBarSync<15>(64);
        } else {
            if ((tid & 31u) == 0u) {
                atomicAdd(d_ready_stage, 1u);
                __threadfence_system();
            }
            while (atomicAdd(const_cast<uint32_t *>(d_release_bars), 0u) == 0u) {
                __nanosleep(200);
            }
            s_data[tid] += 166u;
            NamedBarSync<15>(64);
        }
    }

    // Also exercise Barriers 1..15 post-restore across full CTA.
    NamedBarSync<1>(256);
    NamedBarSync<15>(256);
    __syncthreads();

    const uint32_t partner = (tid ^ 32u) & 255u;
    d_out[blockIdx.x * blockDim.x + tid] = s_data[tid] + s_data[partner] * 3u;
}

// Case 9: Deeply Nested Divergence Convergence Barriers B1..B15
__global__ void k_nested_divergence(uint32_t *d_out,
                                    uint32_t *d_started,
                                    volatile uint32_t *d_allow_exit)
{
    const uint32_t tid = threadIdx.x;
    uint32_t acc = tid * 17u + blockIdx.x * 5u;

    if ((tid & 1u) == 0u) {
        acc += 10u;
        if ((tid & 2u) == 0u) {
            acc += 20u;
            if ((tid & 4u) == 0u) {
                acc += 30u;
                if ((tid & 8u) == 0u) {
                    acc += 40u;
                    if ((tid & 16u) == 0u) {
                        if (tid == 0u) {
                            atomicAdd(d_started, 1u);
                            __threadfence_system();
                        }
                        while (atomicAdd(const_cast<uint32_t *>(d_allow_exit), 0u) == 0u) {
                            __nanosleep(200);
                        }
                        acc += 500u;
                    } else {
                        acc += 51u;
                    }
                    acc ^= 0x1111u;
                } else {
                    acc += 41u;
                }
                acc ^= 0x2222u;
            } else {
                acc += 31u;
            }
            acc ^= 0x3333u;
        } else {
            acc += 21u;
        }
        acc ^= 0x4444u;
    } else {
        acc += 11u;
    }

    __syncthreads();
    d_out[blockIdx.x * blockDim.x + tid] = acc;
}

// Case 10: K1 + K2 FIFO stream kernels (K1 reused on HP DirectLaunch!)
__global__ void k_fifo_k1(uint32_t *d_k1_out,
                          uint32_t *d_started,
                          volatile uint32_t *d_allow_exit,
                          uint32_t add_val)
{
    const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pre = atomicAdd(&d_k1_out[gtid], add_val);
    if (threadIdx.x == 0) {
        atomicAdd(d_started, 1u);
        __threadfence_system();
    }
    while (atomicAdd(const_cast<uint32_t *>(d_allow_exit), 0u) == 0u) {
        __nanosleep(200);
    }
    d_k1_out[gtid] = pre + add_val + 1000u + gtid;
}

__global__ void k_fifo_k2(uint32_t *d_k2_out,
                          const uint32_t *d_k1_out,
                          uint32_t *d_k2_started)
{
    const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid == 0) {
        atomicAdd(d_k2_started, 1u);
        __threadfence_system();
    }
    d_k2_out[gtid] = d_k1_out[gtid] * 2u + 7u;
}

// --------------------------------------------------------------------------
// Helpers for Mapped Flags & Case Execution
// --------------------------------------------------------------------------

struct MappedU32
{
    uint32_t *host = nullptr;
    CUdeviceptr dev = 0;

    bool Alloc(uint32_t init_val = 0)
    {
        if (cudaHostAlloc(&host, sizeof(uint32_t),
                          cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            return false;
        }
        *host = init_val;
        void *dptr = nullptr;
        if (cudaHostGetDevicePointer(&dptr, host, 0) != cudaSuccess) return false;
        dev = reinterpret_cast<CUdeviceptr>(dptr);
        return true;
    }

    ~MappedU32()
    {
        if (host) cudaFreeHost(host);
    }

    void Set(uint32_t v) { __atomic_store_n(host, v, __ATOMIC_RELEASE); }
    uint32_t Get() const { return __atomic_load_n(host, __ATOMIC_ACQUIRE); }
};

bool RunHpVerify(CUstream hp_stream, uint32_t expected_base)
{
    const uint32_t n = 16 * 128;
    uint32_t *d_out = nullptr;
    MappedU32 started, allow;
    CHK_RT(cudaMalloc(&d_out, n * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_out, 0, n * sizeof(uint32_t)));
    if (!started.Alloc(0) || !allow.Alloc(1)) return false;

    void *args[4] = {&d_out, &started.dev, &allow.dev, &expected_base};
    CUfunction fn = ResolveCuFunc(reinterpret_cast<const void *>(k_fifo_k1));
    Sm120StreamContext::ClearDebuggerParams(fn);
    CHK_CU(cuLaunchKernel(fn, 16, 1, 1, 128, 1, 1, 0, hp_stream, args, nullptr));
    CHK_CU(cuStreamSynchronize(hp_stream));

    std::vector<uint32_t> h(n);
    CHK_RT(cudaMemcpy(h.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    for (uint32_t i = 0; i < n; ++i) {
        if (h[i] != expected_base + 1000u + i) {
            std::fprintf(stderr, "HP mismatch at %u: got %u exp %u\n",
                         i, h[i], expected_base + 1000u + i);
            return false;
        }
    }
    return true;
}

bool CheckKernelExecGeneric(CUcontext ctx,
                            CUstream lp_stream,
                            CUstream hp_stream,
                            const char *name,
                            CUfunction fn,
                            dim3 grid,
                            dim3 block,
                            uint32_t smem_bytes,
                            void **args,
                            MappedU32 &started_flag,
                            uint32_t min_started,
                            MappedU32 &allow_flag,
                            bool check_exact_regs = false)
{
    (void)check_exact_regs;
    auto eng = Sm120TrapEngine::Instance(ctx);
    if (!eng->Init()) {
        std::fprintf(stderr, "[%s] engine->Init() failed\n", name);
        return false;
    }
    CUdevice dev = 0;
    CHK_CU(cuCtxGetDevice(&dev));
    Sm120StreamContext sctx(ctx, dev, eng);
    sctx.ResetControlSync();

    const uint32_t num_ctas = grid.x * grid.y * grid.z;
    const uint32_t tpc = block.x * block.y * block.z;

    started_flag.Set(0);
    allow_flag.Set(0);

    {
        std::lock_guard<std::mutex> lock(Sm120TrapEngine::LaunchMutex(ctx));
        if (!sctx.PrepareLaunch(fn, num_ctas, tpc, smem_bytes, 1001, Sm120PreemptMode::kTrapFrame, lp_stream)) {
            std::fprintf(stderr, "[%s] PrepareLaunch failed\n", name);
            return false;
        }
        CHK_CU(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                              smem_bytes, lp_stream, args, nullptr));
    }

    if (!PollHostFlag(started_flag.host, min_started, 3000)) {
        allow_flag.Set(1);
        cuStreamSynchronize(lp_stream);
        std::fprintf(stderr, "[%s] timeout waiting for started >= %u (got %u)\n",
                     name, min_started, started_flag.Get());
        return false;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(150));

    if (!sctx.PreemptStream(Sm120PreemptMode::kTrapFrame)) {
        allow_flag.Set(1);
        std::fprintf(stderr, "[%s] PreemptStream failed (need BAR0 perms?)\n", name);
        return false;
    }
    CHK_CU(cuStreamSynchronize(lp_stream));

    const auto tev = sctx.Telemetry();
    if (tev.overflow_err != 0 || tev.evicted_slots == 0) {
        allow_flag.Set(1);
        std::fprintf(stderr, "[%s] bad evict telemetry: evicted=%u overflow=%u\n",
                     name, tev.evicted_slots, tev.overflow_err);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(Sm120TrapEngine::LaunchMutex(ctx));
        if (!RunHpVerify(hp_stream, 55)) return false;
    }

    allow_flag.Set(1);
    sctx.PrepareRestore(Sm120PreemptMode::kTrapFrame);
    {
        std::lock_guard<std::mutex> lock(Sm120TrapEngine::LaunchMutex(ctx));
        if (!sctx.PrepareLaunch(fn, num_ctas, tpc, smem_bytes, 1001, Sm120PreemptMode::kTrapFrame, lp_stream)) {
            return false;
        }
        CHK_CU(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                              smem_bytes, lp_stream, args, nullptr));
    }
    CHK_CU(cuStreamSynchronize(lp_stream));
    const auto tre = sctx.Telemetry();
    sctx.EnqueuePostRestoreClear(lp_stream);
    CHK_CU(cuStreamSynchronize(lp_stream));
    std::printf("  [%s] evicted=%u restored=%u drained=%u arena=%zu KiB\n",
                name, tev.evicted_slots, tre.restored_slots, tev.drained_cnt,
                tre.arena_bytes >> 10);
    if (tre.restored_slots != tev.evicted_slots) {
        std::fprintf(stderr, "[%s] restored (%u) != evicted (%u)\n",
                     name, tre.restored_slots, tev.evicted_slots);
        return false;
    }
    return true;
}

// --------------------------------------------------------------------------
// 11 Automated E2E Cases
// --------------------------------------------------------------------------

bool RunCase1Mode1MultiWave(CUcontext ctx, CUstream lp_stream, CUstream hp_stream)
{
    const uint32_t num_ctas = 512, tpc = 128, smem = 49152;
    uint32_t *d_entry_cnt = nullptr, *d_out = nullptr;
    MappedU32 started, allow, fast_done;
    CHK_RT(cudaMalloc(&d_entry_cnt, num_ctas * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&d_out, num_ctas * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_entry_cnt, 0, num_ctas * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_out, 0, num_ctas * sizeof(uint32_t)));
    if (!started.Alloc(0) || !allow.Alloc(0) || !fast_done.Alloc(0)) return false;

    CUfunction fn = ResolveCuFunc(reinterpret_cast<const void *>(k_mode1_idempotent));
    CHK_CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(smem)));
    auto eng = Sm120TrapEngine::Instance(ctx);
    if (!eng->Init()) return false;
    CUdevice dev = 0;
    CHK_CU(cuCtxGetDevice(&dev));
    Sm120StreamContext sctx(ctx, dev, eng);
    sctx.ResetControlSync();

    uint32_t fast_thresh = 0;
    void *args[6] = {&d_entry_cnt, &d_out, &started.dev, &allow.dev, &fast_done.dev, &fast_thresh};
    {
        std::lock_guard<std::mutex> lock(Sm120TrapEngine::LaunchMutex(ctx));
        if (!sctx.PrepareLaunch(fn, num_ctas, tpc, smem, 101, Sm120PreemptMode::kEarlyExit, lp_stream)) return false;
        CHK_CU(cuLaunchKernel(fn, num_ctas, 1, 1, tpc, 1, 1, smem, lp_stream, args, nullptr));
    }

    int sm_cnt_c1 = 30;
    cuDeviceGetAttribute(&sm_cnt_c1, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    const uint32_t target_c1 = std::min<uint32_t>(40u, static_cast<uint32_t>(std::max(1, sm_cnt_c1 - 2)));
    if (!PollHostFlag(started.host, target_c1, 3000)) {
        allow.Set(1);
        cuStreamSynchronize(lp_stream);
        return false;
    }
    if (!sctx.PreemptStream(Sm120PreemptMode::kEarlyExit)) return false;
    CHK_CU(cuStreamSynchronize(lp_stream));

    const auto tev = sctx.Telemetry();
    std::vector<uint32_t> h_entry_mid(num_ctas);
    CHK_RT(cudaMemcpy(h_entry_mid.data(), d_entry_cnt, num_ctas * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<Sm120CtaEntry> entries;
    if (!sctx.ReadCtaEntries(entries, num_ctas)) return false;
    uint32_t state_c_leaks = 0, replay_marked = 0;
    for (uint32_t b = 0; b < num_ctas; ++b) {
        if (entries[b].state == kCtaStateReplay) ++replay_marked;
        if (entries[b].state == kCtaStateQueued && h_entry_mid[b] != 0) ++state_c_leaks;
    }

    {
        std::lock_guard<std::mutex> lock(Sm120TrapEngine::LaunchMutex(ctx));
        if (!RunHpVerify(hp_stream, 11)) return false;
    }

    allow.Set(1);
    sctx.PrepareRestore(Sm120PreemptMode::kEarlyExit);
    {
        std::lock_guard<std::mutex> lock(Sm120TrapEngine::LaunchMutex(ctx));
        if (!sctx.PrepareLaunch(fn, num_ctas, tpc, smem, 101, Sm120PreemptMode::kEarlyExit, lp_stream)) return false;
        CHK_CU(cuLaunchKernel(fn, num_ctas, 1, 1, tpc, 1, 1, smem, lp_stream, args, nullptr));
    }
    sctx.EnqueuePostRestoreClear(lp_stream);
    CHK_CU(cuStreamSynchronize(lp_stream));

    std::vector<uint32_t> h_out(num_ctas);
    CHK_RT(cudaMemcpy(h_out.data(), d_out, num_ctas * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_entry_cnt);
    cudaFree(d_out);

    uint32_t errs = 0;
    for (uint32_t b = 0; b < num_ctas; ++b) {
        if (h_out[b] != b * 13u + 7u) ++errs;
    }
    std::printf("[CASE 01] Mode1 MultiWave: mode1_exit=%u replay_marked=%u leaks=%u errs=%u -> %s\n",
                tev.mode1_exit_cnt, replay_marked, state_c_leaks, errs,
                (tev.mode1_exit_cnt > 0 && state_c_leaks == 0 && errs == 0) ? "PASS" : "FAIL");
    return (tev.mode1_exit_cnt > 0 && state_c_leaks == 0 && errs == 0);
}

bool RunCase2Mode5SingleWave(CUcontext ctx, CUstream lp_stream, CUstream hp_stream)
{
    const uint32_t num_ctas = 84, tpc = 128, fast_ctas = 0, steps = 10;
    unsigned long long *d_total = nullptr;
    uint32_t *d_pre = nullptr, *d_post = nullptr;
    MappedU32 fast_done, started, allow;
    CHK_RT(cudaMalloc(&d_total, sizeof(unsigned long long)));
    CHK_RT(cudaMalloc(&d_pre, num_ctas * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&d_post, num_ctas * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_total, 0, sizeof(unsigned long long)));
    CHK_RT(cudaMemset(d_pre, 0, num_ctas * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_post, 0, num_ctas * sizeof(uint32_t)));
    if (!fast_done.Alloc(0) || !started.Alloc(0) || !allow.Alloc(0)) return false;

    uint32_t smem = 0, touch_smem = 0;
    void *args[9] = {&d_total, &d_pre, &d_post, &fast_done.dev, &started.dev, &allow.dev,
                     const_cast<uint32_t *>(&fast_ctas), const_cast<uint32_t *>(&steps), &touch_smem};
    CUfunction fn = ResolveCuFunc(reinterpret_cast<const void *>(k_nonidemp_multiwave));
    const bool ok = CheckKernelExecGeneric(
        ctx, lp_stream, hp_stream, "CASE 02", fn, dim3(num_ctas), dim3(tpc), smem, args,
        started, num_ctas, allow);

    unsigned long long h_total = 0;
    CHK_RT(cudaMemcpy(&h_total, d_total, sizeof(h_total), cudaMemcpyDeviceToHost));
    cudaFree(d_total);
    cudaFree(d_pre);
    cudaFree(d_post);

    const unsigned long long exp = static_cast<unsigned long long>(num_ctas) * tpc * steps;
    const bool pass = ok && (h_total == exp);
    std::printf("[CASE 02] Mode5 SingleWave atomicAdd: got=%llu exp=%llu -> %s\n",
                h_total, exp, pass ? "PASS" : "FAIL");
    return pass;
}

bool RunCase3Mode5MultiWaveZeroLeak(CUcontext ctx, CUstream lp_stream, CUstream hp_stream)
{
    const uint32_t num_ctas = 512, tpc = 128, smem = 49152, fast_ctas = 40, steps = 8;
    unsigned long long *d_total = nullptr;
    uint32_t *d_pre = nullptr, *d_post = nullptr;
    MappedU32 fast_done, started, allow;
    CHK_RT(cudaMalloc(&d_total, sizeof(unsigned long long)));
    CHK_RT(cudaMalloc(&d_pre, num_ctas * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&d_post, num_ctas * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_total, 0, sizeof(unsigned long long)));
    CHK_RT(cudaMemset(d_pre, 0, num_ctas * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_post, 0, num_ctas * sizeof(uint32_t)));
    if (!fast_done.Alloc(0) || !started.Alloc(0) || !allow.Alloc(0)) return false;

    CUfunction fn = ResolveCuFunc(reinterpret_cast<const void *>(k_nonidemp_multiwave));
    CHK_CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(smem)));

    auto eng = Sm120TrapEngine::Instance(ctx);
    if (!eng->Init()) return false;
    CUdevice dev = 0;
    CHK_CU(cuCtxGetDevice(&dev));
    Sm120StreamContext sctx(ctx, dev, eng);
    sctx.ResetControlSync();

    uint32_t touch_smem = 1;
    void *args[9] = {&d_total, &d_pre, &d_post, &fast_done.dev, &started.dev, &allow.dev,
                     const_cast<uint32_t *>(&fast_ctas), const_cast<uint32_t *>(&steps), &touch_smem};
    {
        std::lock_guard<std::mutex> lock(Sm120TrapEngine::LaunchMutex(ctx));
        if (!sctx.PrepareLaunch(fn, num_ctas, tpc, smem, 303, Sm120PreemptMode::kTrapFrame, lp_stream)) return false;
        CHK_CU(cuLaunchKernel(fn, num_ctas, 1, 1, tpc, 1, 1, smem, lp_stream, args, nullptr));
    }

    int sm_cnt_c3 = 30;
    cuDeviceGetAttribute(&sm_cnt_c3, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    const uint32_t target_c3 = std::min<uint32_t>(60u, static_cast<uint32_t>(std::max(1, sm_cnt_c3 - 2)));
    if (!PollHostFlag(fast_done.host, fast_ctas, 3000) ||
        !PollHostFlag(started.host, target_c3, 3000)) {
        allow.Set(1);
        cuStreamSynchronize(lp_stream);
        return false;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(2000));

    if (!sctx.PreemptStream(Sm120PreemptMode::kTrapFrame)) return false;
    CHK_CU(cuStreamSynchronize(lp_stream));

    const auto tev = sctx.Telemetry();
    std::vector<uint32_t> h_pre_mid(num_ctas), h_post_mid(num_ctas);
    CHK_RT(cudaMemcpy(h_pre_mid.data(), d_pre, num_ctas * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHK_RT(cudaMemcpy(h_post_mid.data(), d_post, num_ctas * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::vector<Sm120CtaEntry> entries;
    if (!sctx.ReadCtaEntries(entries, num_ctas)) return false;
    uint32_t st_a = 0, st_b = 0, st_c = 0, leaks = 0;
    for (uint32_t b = 0; b < num_ctas; ++b) {
        if (entries[b].state == kCtaStateDone) ++st_a;
        else if (entries[b].state == kCtaStateSaved) ++st_b;
        else if (entries[b].state == kCtaStateQueued) {
            ++st_c;
            if (h_pre_mid[b] != 0 || h_post_mid[b] != 0) ++leaks;
        }
    }

    {
        std::lock_guard<std::mutex> lock(Sm120TrapEngine::LaunchMutex(ctx));
        if (!RunHpVerify(hp_stream, 33)) return false;
    }

    allow.Set(1);
    sctx.PrepareRestore(Sm120PreemptMode::kTrapFrame);
    {
        std::lock_guard<std::mutex> lock(Sm120TrapEngine::LaunchMutex(ctx));
        if (!sctx.PrepareLaunch(fn, num_ctas, tpc, smem, 303, Sm120PreemptMode::kTrapFrame, lp_stream)) return false;
        CHK_CU(cuLaunchKernel(fn, num_ctas, 1, 1, tpc, 1, 1, smem, lp_stream, args, nullptr));
    }
    CHK_CU(cuStreamSynchronize(lp_stream));
    const auto tre = sctx.Telemetry();
    sctx.EnqueuePostRestoreClear(lp_stream);
    CHK_CU(cuStreamSynchronize(lp_stream));
    unsigned long long h_total = 0;
    std::vector<uint32_t> h_pre(num_ctas), h_post(num_ctas);
    CHK_RT(cudaMemcpy(&h_total, d_total, sizeof(h_total), cudaMemcpyDeviceToHost));
    CHK_RT(cudaMemcpy(h_pre.data(), d_pre, num_ctas * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHK_RT(cudaMemcpy(h_post.data(), d_post, num_ctas * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_total);
    cudaFree(d_pre);
    cudaFree(d_post);

    uint32_t cta_errs = 0;
    for (uint32_t b = 0; b < num_ctas; ++b) {
        if (h_pre[b] != 1u || h_post[b] != 1u) ++cta_errs;
    }
    const unsigned long long exp = static_cast<unsigned long long>(num_ctas) * tpc * steps;
    const bool pass = (st_a >= fast_ctas) && (st_b > 0) && (st_c > 0) &&
                      (st_a + st_b + st_c == num_ctas) && (leaks == 0) &&
                      (tev.overflow_err == 0) && (tre.restored_slots == tev.evicted_slots) &&
                      (cta_errs == 0) && (h_total == exp);
    std::printf("[CASE 03] Mode5 MultiWave A+B+C: A=%u B=%u C=%u (sum=%u/512) ev=%u rest=%u cta_errs=%u ovf=%u leaks=%u total=%llu/%llu -> %s\n",
                st_a, st_b, st_c, st_a + st_b + st_c,
                tev.evicted_slots, tre.restored_slots, cta_errs, tev.overflow_err,
                leaks, h_total, exp, pass ? "PASS" : "FAIL");
    return pass;
}

bool RunCase4GridBlock3D(CUcontext ctx, CUstream lp_stream, CUstream hp_stream)
{
    const dim3 grid(4, 3, 2);
    const dim3 block(8, 4, 2);
    const uint32_t n = 24 * 64;
    uint32_t *d_out = nullptr, *d_pre = nullptr;
    MappedU32 started, allow;
    CHK_RT(cudaMalloc(&d_out, n * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&d_pre, n * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_out, 0, n * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_pre, 0, n * sizeof(uint32_t)));
    if (!started.Alloc(0) || !allow.Alloc(0)) return false;

    void *args[4] = {&d_out, &d_pre, &started.dev, &allow.dev};
    CUfunction fn = ResolveCuFunc(reinterpret_cast<const void *>(k_3d_grid_block));
    const bool ok = CheckKernelExecGeneric(
        ctx, lp_stream, hp_stream, "CASE 04", fn, grid, block, 0, args, started, 24, allow);

    std::vector<uint32_t> h(n);
    CHK_RT(cudaMemcpy(h.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    cudaFree(d_pre);

    uint32_t errs = 0;
    for (uint32_t cz = 0; cz < grid.z; ++cz) {
        for (uint32_t cy = 0; cy < grid.y; ++cy) {
            for (uint32_t cx = 0; cx < grid.x; ++cx) {
                const uint32_t cta_id = (cz * grid.y + cy) * grid.x + cx;
                for (uint32_t tz = 0; tz < block.z; ++tz) {
                    for (uint32_t ty = 0; ty < block.y; ++ty) {
                        for (uint32_t tx = 0; tx < block.x; ++tx) {
                            const uint32_t tid = (tz * block.y + ty) * block.x + tx;
                            const uint32_t gtid = cta_id * 64u + tid;
                            uint32_t exp = (cta_id * 131u + tid * 17u + 3u) ^
                                           (cx * 7u + cy * 19u + cz * 53u);
                            exp += (tx * 3u + ty * 11u + tz * 29u);
                            if (h[gtid] != exp) ++errs;
                        }
                    }
                }
            }
        }
    }
    const bool pass = ok && (errs == 0);
    std::printf("[CASE 04] 3D Grid(4,3,2) x Block(8,4,2): errs=%u/%u -> %s\n",
                errs, n, pass ? "PASS" : "FAIL");
    return pass;
}

template <int N_REGS>
bool RunHighRegSubCase(CUcontext ctx, CUstream lp_stream, CUstream hp_stream, const char *tag)
{
    const uint32_t grid = 16, block = 64, n = grid * block, iters = 6;
    uint32_t *d_out = nullptr, *d_seeds = nullptr;
    MappedU32 started, allow;
    CHK_RT(cudaMalloc(&d_out, n * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&d_seeds, n * sizeof(uint32_t)));
    if (!started.Alloc(0) || !allow.Alloc(0)) return false;

    std::vector<uint32_t> h_seeds(n);
    for (uint32_t i = 0; i < n; ++i) h_seeds[i] = i * 977u + 13u;
    CHK_RT(cudaMemcpy(d_seeds, h_seeds.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUfunction fn = ResolveCuFunc(reinterpret_cast<const void *>(k_high_regs_and_ur<N_REGS>));
    int num_regs = 0;
    CHK_CU(cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));

    uint32_t it_arg = iters;
    void *args[5] = {&d_out, &d_seeds, &started.dev, &allow.dev, &it_arg};
    const bool ok = CheckKernelExecGeneric(
        ctx, lp_stream, hp_stream, tag, fn, dim3(grid), dim3(block), 0, args, started, grid, allow);

    std::vector<uint32_t> h(n);
    CHK_RT(cudaMemcpy(h.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    cudaFree(d_seeds);

    uint32_t errs = 0;
    for (uint32_t gtid = 0; gtid < n; ++gtid) {
        uint32_t v[N_REGS];
        for (int i = 0; i < N_REGS; ++i) v[i] = h_seeds[gtid] ^ static_cast<uint32_t>(i * 2654435761u + gtid);
        uint32_t u_acc = iters * 17u + 5u;
        for (uint32_t it = 0; it < iters; ++it) {
            if ((it & 1u) == 0u) u_acc += 3u;
            else u_acc ^= 0x5a5a5a5au;
            for (int i = 0; i < N_REGS; ++i) v[i] = v[i] * 1664525u + static_cast<uint32_t>(i + it + 1u);
        }
        uint32_t fold = u_acc;
        for (int i = 0; i < N_REGS; ++i) fold = (fold * 33u) ^ v[i];
        if (h[gtid] != fold) ++errs;
    }
    const bool pass = ok && (errs == 0);
    std::printf("  [%s] num_regs=%d errs=%u/%u -> %s\n", tag, num_regs, errs, n, pass ? "PASS" : "FAIL");
    return pass;
}

bool RunCase5HighRegs(CUcontext ctx, CUstream lp_stream, CUstream hp_stream)
{
    const bool a = RunHighRegSubCase<96>(ctx, lp_stream, hp_stream, "CASE05-A");
    const bool b = RunHighRegSubCase<180>(ctx, lp_stream, hp_stream, "CASE05-B");
    std::printf("[CASE 05] High GPR Cascade + UR/UPR -> %s\n", (a && b) ? "PASS" : "FAIL");
    return a && b;
}

bool RunCase6DynamicSmem(CUcontext ctx, CUstream lp_stream, CUstream hp_stream)
{
    const uint32_t grid = 32, block = 128, dyn_words = 8192, dyn_bytes = dyn_words * 4u;
    uint32_t *d_out = nullptr;
    MappedU32 started, allow;
    CHK_RT(cudaMalloc(&d_out, grid * block * sizeof(uint32_t)));
    if (!started.Alloc(0) || !allow.Alloc(0)) return false;

    uint32_t dw = dyn_words;
    void *args[4] = {&d_out, &dw, &started.dev, &allow.dev};
    CUfunction fn = ResolveCuFunc(reinterpret_cast<const void *>(k_static_dyn_smem));
    CHK_CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(dyn_bytes)));

    const bool ok = CheckKernelExecGeneric(
        ctx, lp_stream, hp_stream, "CASE 06", fn, dim3(grid), dim3(block), dyn_bytes, args, started, grid, allow);

    std::vector<uint32_t> h(grid * block);
    CHK_RT(cudaMemcpy(h.data(), d_out, grid * block * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    uint32_t errs = 0;
    for (uint32_t b = 0; b < grid; ++b) {
        std::vector<uint32_t> st(1024), dy(dyn_words);
        for (uint32_t i = 0; i < 1024u; ++i) st[i] = (b << 16) ^ (i * 29u + 11u);
        for (uint32_t i = 0; i < dyn_words; ++i) dy[i] = (b << 20) ^ (i * 97u + 0x1234u);
        for (uint32_t tid = 0; tid < block; ++tid) {
            uint32_t sum = 0;
            for (uint32_t i = tid; i < 1024u; i += block) sum ^= st[(i + 17u) & 1023u];
            for (uint32_t i = tid; i < dyn_words; i += block) sum += dy[(i + 31u) % dyn_words];
            if (h[b * block + tid] != sum) ++errs;
        }
    }
    const bool pass = ok && (errs == 0);
    std::printf("[CASE 06] Static+Dyn SMEM (36 KiB): errs=%u/%u -> %s\n", errs, grid * block, pass ? "PASS" : "FAIL");
    return pass;
}

uint32_t CpuRecurseRef(int depth, uint32_t seed)
{
    uint32_t local_arr[16];
    for (int i = 0; i < 16; ++i) local_arr[i] = seed + static_cast<uint32_t>(depth * 101 + i * 7);
    if (depth == 0) {
        uint32_t s = 0;
        for (int i = 0; i < 16; ++i) s += local_arr[(i + (seed & 3u)) & 15u];
        return s ^ 0x9e3779b9u;
    }
    uint32_t child = CpuRecurseRef(depth - 1, seed * 3u + 1u);
    uint32_t mix = child;
    for (int i = 0; i < 16; ++i) mix ^= local_arr[(i + (child & 7u)) & 15u] + static_cast<uint32_t>(i * 13);
    return mix;
}

bool RunCase7LmemRecursion(CUcontext ctx, CUstream lp_stream, CUstream hp_stream)
{
    const uint32_t grid = 32, block = 64, n = grid * block;
    uint32_t *d_out = nullptr;
    MappedU32 started, allow;
    CHK_RT(cudaMalloc(&d_out, n * sizeof(uint32_t)));
    if (!started.Alloc(0) || !allow.Alloc(0)) return false;

    void *args[3] = {&d_out, &started.dev, &allow.dev};
    CUfunction fn = ResolveCuFunc(reinterpret_cast<const void *>(k_lmem_recursion));
    int local_sz = 0;
    CHK_CU(cuFuncGetAttribute(&local_sz, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn));
    const bool ok = CheckKernelExecGeneric(
        ctx, lp_stream, hp_stream, "CASE 07", fn, dim3(grid), dim3(block), 0, args, started, grid, allow);

    std::vector<uint32_t> h(n);
    CHK_RT(cudaMemcpy(h.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    uint32_t errs = 0;
    for (uint32_t gtid = 0; gtid < n; ++gtid) {
        uint32_t buf[32];
        for (int i = 0; i < 32; ++i) buf[i] = gtid * 19u + static_cast<uint32_t>(i * 5 + 3);
        uint32_t rec = CpuRecurseRef(4, gtid + 17u);
        uint32_t acc = rec;
        for (int i = 0; i < 32; ++i) acc += buf[(i + (rec & 7u)) & 31u];
        if (h[gtid] != acc) ++errs;
    }
    const bool pass = ok && (errs == 0);
    std::printf("[CASE 07] LMEM Stack-Spill + __noinline__ Recursion (static_lmem=%d B): errs=%u/%u -> %s\n",
                local_sz, errs, n, pass ? "PASS" : "FAIL");
    return pass;
}

bool RunCase8NamedBarriers(CUcontext ctx, CUstream lp_stream, CUstream hp_stream)
{
    const uint32_t grid = 8, block = 256, n = grid * block;
    uint32_t *d_out = nullptr;
    MappedU32 ready_stage, release_bars;
    CHK_RT(cudaMalloc(&d_out, n * sizeof(uint32_t)));
    if (!ready_stage.Alloc(0) || !release_bars.Alloc(0)) return false;

    void *args[3] = {&d_out, &ready_stage.dev, &release_bars.dev};
    CUfunction fn = ResolveCuFunc(reinterpret_cast<const void *>(k_named_barriers_all));
    // Each CTA signals 4 times across warps 1,2,5,7 -> grid * 4
    const bool ok = CheckKernelExecGeneric(
        ctx, lp_stream, hp_stream, "CASE 08", fn, dim3(grid), dim3(block), 0, args,
        ready_stage, grid * 4u, release_bars);

    std::vector<uint32_t> h(n);
    CHK_RT(cudaMemcpy(h.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    static const uint32_t kWarpAdd[8] = {55u, 66u, 111u, 222u, 77u, 88u, 155u, 166u};
    uint32_t errs = 0;
    for (uint32_t b = 0; b < grid; ++b) {
        std::vector<uint32_t> s(block);
        for (uint32_t tid = 0; tid < block; ++tid) {
            s[tid] = (b + 1u) * 1000u + tid + kWarpAdd[tid >> 5];
        }
        for (uint32_t tid = 0; tid < block; ++tid) {
            const uint32_t partner = (tid ^ 32u) & 255u;
            if (h[b * block + tid] != s[tid] + s[partner] * 3u) ++errs;
        }
    }
    const bool pass = ok && (errs == 0);
    std::printf("[CASE 08] Named Barriers 0..15 (bar.sync 0,1,5,7,11,15 + bar.arrive 11): errs=%u/%u -> %s\n",
                errs, n, pass ? "PASS" : "FAIL");
    return pass;
}

bool RunCase9NestedDivergence(CUcontext ctx, CUstream lp_stream, CUstream hp_stream)
{
    const uint32_t grid = 16, block = 128, n = grid * block;
    uint32_t *d_out = nullptr;
    MappedU32 started, allow;
    CHK_RT(cudaMalloc(&d_out, n * sizeof(uint32_t)));
    if (!started.Alloc(0) || !allow.Alloc(0)) return false;

    void *args[3] = {&d_out, &started.dev, &allow.dev};
    CUfunction fn = ResolveCuFunc(reinterpret_cast<const void *>(k_nested_divergence));
    const bool ok = CheckKernelExecGeneric(
        ctx, lp_stream, hp_stream, "CASE 09", fn, dim3(grid), dim3(block), 0, args,
        started, grid, allow);

    std::vector<uint32_t> h(n);
    CHK_RT(cudaMemcpy(h.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    uint32_t errs = 0;
    for (uint32_t b = 0; b < grid; ++b) {
        for (uint32_t tid = 0; tid < block; ++tid) {
            uint32_t acc = tid * 17u + b * 5u;
            if ((tid & 1u) == 0u) {
                acc += 10u;
                if ((tid & 2u) == 0u) {
                    acc += 20u;
                    if ((tid & 4u) == 0u) {
                        acc += 30u;
                        if ((tid & 8u) == 0u) {
                            acc += 40u;
                            acc += ((tid & 16u) == 0u) ? 500u : 51u;
                            acc ^= 0x1111u;
                        } else acc += 41u;
                        acc ^= 0x2222u;
                    } else acc += 31u;
                    acc ^= 0x3333u;
                } else acc += 21u;
                acc ^= 0x4444u;
            } else acc += 11u;
            if (h[b * block + tid] != acc) ++errs;
        }
    }
    const bool pass = ok && (errs == 0);
    std::printf("[CASE 09] Nested Divergence Convergence Barriers (B1..B15): errs=%u/%u -> %s\n",
                errs, n, pass ? "PASS" : "FAIL");
    return pass;
}

bool RunCase10XSchedFifoAndDirectLaunch(CUcontext ctx, CUstream lp_stream, CUstream hp_stream)
{
    const uint32_t grid = 16, block = 128, n = grid * block, add_val = 77u;
    uint32_t *d_k1 = nullptr, *d_k2 = nullptr;
    MappedU32 k1_started, k1_allow, k2_started;
    CHK_RT(cudaMalloc(&d_k1, n * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&d_k2, n * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_k1, 0, n * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_k2, 0, n * sizeof(uint32_t)));
    if (!k1_started.Alloc(0) || !k1_allow.Alloc(0) || !k2_started.Alloc(0)) return false;

    auto lv3q = std::make_shared<CudaQueueLv3Trap>(lp_stream);
    auto xq = std::make_shared<AsyncXQueue>(lv3q, kPreemptLevelInterrupt, 16, 4);

    CUfunction fn1 = ResolveCuFunc(reinterpret_cast<const void *>(k_fifo_k1));
    CUfunction fn2 = ResolveCuFunc(reinterpret_cast<const void *>(k_fifo_k2));

    uint32_t add_v = add_val;
    void *p1[4] = {&d_k1, &k1_started.dev, &k1_allow.dev, &add_v};
    void *p2[3] = {&d_k2, &d_k1, &k2_started.dev};

    auto cmd1 = std::make_shared<CudaKernelLaunchCommand>(fn1, grid, 1, 1, block, 1, 1, 0, p1, nullptr, false);
    auto cmd2 = std::make_shared<CudaKernelLaunchCommand>(fn2, grid, 1, 1, block, 1, 1, 0, p2, nullptr, false);

    xq->Submit(cmd1);
    xq->Submit(cmd2);

    if (!PollHostFlag(k1_started.host, grid, 3000)) {
        k1_allow.Set(1);
        xq->WaitAll();
        return false;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(150));

    xq->Suspend(kQueueSuspendFlagSyncHwQueue);

    const uint32_t k2_mid = k2_started.Get();
    // Test DirectLaunch on unmanaged HP stream using the EXACT SAME CUfunction fn1 while K1 is suspended!
    uint32_t *d_hp = nullptr;
    MappedU32 hp_started, hp_allow;
    CHK_RT(cudaMalloc(&d_hp, n * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_hp, 0, n * sizeof(uint32_t)));
    if (!hp_started.Alloc(0) || !hp_allow.Alloc(1)) return false;
    uint32_t hp_add = 500u;
    void *php[4] = {&d_hp, &hp_started.dev, &hp_allow.dev, &hp_add};
    auto hp_cmd = std::make_shared<CudaKernelLaunchCommand>(fn1, grid, 1, 1, block, 1, 1, 0, php, nullptr, false);
    CHK_CU(cuCtxSetCurrent(ctx));
    CHK_CU(DirectLaunch(hp_cmd, hp_stream));
    CHK_CU(cuStreamSynchronize(hp_stream));

    k1_allow.Set(1);
    xq->Resume(0);
    xq->WaitAll();

    std::vector<uint32_t> h1(n), h2(n), hhp(n);
    CHK_RT(cudaMemcpy(h1.data(), d_k1, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHK_RT(cudaMemcpy(h2.data(), d_k2, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHK_RT(cudaMemcpy(hhp.data(), d_hp, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_k1);
    cudaFree(d_k2);
    cudaFree(d_hp);

    uint32_t e1 = 0, e2 = 0, ehp = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t exp1 = add_val + 1000u + i;
        if (h1[i] != exp1) ++e1;
        if (h2[i] != exp1 * 2u + 7u) ++e2;
        if (hhp[i] != hp_add + 1000u + i) ++ehp;
    }
    const bool pass = (k2_mid == 0 && e1 == 0 && e2 == 0 && ehp == 0);
    std::printf("[CASE 10] XSched AsyncXQueue FIFO [K1->K2] + Same-Func DirectLaunch HP: "
                "k2_mid=%u errs(k1=%u,k2=%u,hp=%u) -> %s\n",
                k2_mid, e1, e2, ehp, pass ? "PASS" : "FAIL");
    return pass;
}

bool RunCase11Stress100Rounds(CUcontext ctx, CUstream lp_stream)
{
    const uint32_t num_ctas = 336, tpc = 128, smem = 49152, steps = 4;
    uint32_t round_fast_ctas = 16;
    unsigned long long *d_total = nullptr;
    uint32_t *d_pre = nullptr, *d_post = nullptr;
    MappedU32 fast_done, started, allow;
    CHK_RT(cudaMalloc(&d_total, sizeof(unsigned long long)));
    CHK_RT(cudaMalloc(&d_pre, num_ctas * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&d_post, num_ctas * sizeof(uint32_t)));
    if (!fast_done.Alloc(0) || !started.Alloc(0) || !allow.Alloc(0)) return false;

    CUfunction fn = ResolveCuFunc(reinterpret_cast<const void *>(k_nonidemp_multiwave));
    CHK_CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(smem)));
    auto eng = Sm120TrapEngine::Instance(ctx);
    if (!eng->Init()) return false;
    CUdevice dev = 0;
    CHK_CU(cuCtxGetDevice(&dev));
    Sm120StreamContext sctx(ctx, dev, eng);

    uint32_t touch_smem = 1;
    void *args[9] = {&d_total, &d_pre, &d_post, &fast_done.dev, &started.dev, &allow.dev,
                     &round_fast_ctas, const_cast<uint32_t *>(&steps), &touch_smem};

    uint32_t total_errs = 0;
    const int kRounds = 100;
    const unsigned long long exp_per_round = static_cast<unsigned long long>(num_ctas) * tpc * steps;

    for (int r = 0; r < kRounds; ++r) {
        round_fast_ctas = ((r % 10) == 0) ? 184u : 16u;
        sctx.ResetControlSync();
        CHK_RT(cudaMemsetAsync(d_total, 0, sizeof(unsigned long long), lp_stream));
        CHK_RT(cudaMemsetAsync(d_pre, 0, num_ctas * sizeof(uint32_t), lp_stream));
        CHK_RT(cudaMemsetAsync(d_post, 0, num_ctas * sizeof(uint32_t), lp_stream));
        CHK_CU(cuStreamSynchronize(lp_stream));

        fast_done.Set(0);
        started.Set(0);
        allow.Set(0);

        {
            std::lock_guard<std::mutex> lock(Sm120TrapEngine::LaunchMutex(ctx));
            if (!sctx.PrepareLaunch(fn, num_ctas, tpc, smem, static_cast<uint64_t>(2000 + r),
                                    Sm120PreemptMode::kTrapFrame, lp_stream)) {
                return false;
            }
            CHK_CU(cuLaunchKernel(fn, num_ctas, 1, 1, tpc, 1, 1, smem, lp_stream, args, nullptr));
        }
        if (!PollHostFlag(fast_done.host, round_fast_ctas, 2000) || !PollHostFlag(started.host, 40, 2000)) {
            allow.Set(1);
            cuStreamSynchronize(lp_stream);
            std::fprintf(stderr, "[CASE 11] round %d start timeout\n", r);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(60));
        if (!sctx.PreemptStream(Sm120PreemptMode::kTrapFrame)) return false;
        CHK_CU(cuStreamSynchronize(lp_stream));

        const auto tev = sctx.Telemetry();
        // Exercise double-round restore-wave State A preservation every 10th round!
        if ((r % 10) == 0) {
            // Re-launch restore BEFORE setting allow=1, sleep 80 us, then preempt Round 2!
            sctx.PrepareRestore(Sm120PreemptMode::kTrapFrame);
            {
                std::lock_guard<std::mutex> lock(Sm120TrapEngine::LaunchMutex(ctx));
                if (!sctx.PrepareLaunch(fn, num_ctas, tpc, smem, static_cast<uint64_t>(2000 + r),
                                        Sm120PreemptMode::kTrapFrame, lp_stream)) return false;
                CHK_CU(cuLaunchKernel(fn, num_ctas, 1, 1, tpc, 1, 1, smem, lp_stream, args, nullptr));
            }
            std::this_thread::sleep_for(std::chrono::microseconds(120));
            if (!sctx.PreemptStream(Sm120PreemptMode::kTrapFrame)) return false;
            CHK_CU(cuStreamSynchronize(lp_stream));
        }

        allow.Set(1);
        sctx.PrepareRestore(Sm120PreemptMode::kTrapFrame);
        {
            std::lock_guard<std::mutex> lock(Sm120TrapEngine::LaunchMutex(ctx));
            if (!sctx.PrepareLaunch(fn, num_ctas, tpc, smem, static_cast<uint64_t>(2000 + r),
                                    Sm120PreemptMode::kTrapFrame, lp_stream)) return false;
            CHK_CU(cuLaunchKernel(fn, num_ctas, 1, 1, tpc, 1, 1, smem, lp_stream, args, nullptr));
        }
        sctx.EnqueuePostRestoreClear(lp_stream);
        CHK_CU(cuStreamSynchronize(lp_stream));

        unsigned long long h_total = 0;
        CHK_RT(cudaMemcpy(&h_total, d_total, sizeof(h_total), cudaMemcpyDeviceToHost));
        if (tev.overflow_err != 0 || h_total != exp_per_round) {
            std::fprintf(stderr, "[CASE 11] round %d mismatch: total=%llu exp=%llu ovf=%u\n",
                         r, h_total, exp_per_round, tev.overflow_err);
            ++total_errs;
            break;
        }
    }

    cudaFree(d_total);
    cudaFree(d_pre);
    cudaFree(d_post);
    const bool pass = (total_errs == 0);
    std::printf("[CASE 11] 100-Round Multi-Wave Stress (incl. double-preempt State A): errs=%u -> %s\n",
                total_errs, pass ? "PASS" : "FAIL");
    return pass;
}

} // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuInit failed\n");
        return 1;
    }
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 1;

    char name[128] = {};
    cuDeviceGetName(name, sizeof(name), dev);
    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

    if (cudaSetDevice(0) != cudaSuccess || cudaFree(0) != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice/cudaFree failed\n");
        return 1;
    }
    CUcontext ctx = nullptr;
    if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS || cuCtxSetCurrent(ctx) != CUDA_SUCCESS) return 1;
    int lp = 0, hp = 0;
    cuCtxGetStreamPriorityRange(&lp, &hp);

    CUstream lp_stream = nullptr, hp_stream = nullptr;
    cuStreamCreateWithPriority(&lp_stream, CU_STREAM_NON_BLOCKING, lp);
    cuStreamCreateWithPriority(&hp_stream, CU_STREAM_NON_BLOCKING, hp);

    std::printf("[SM120_V2_E2E_SUITE] GPU=%s SM=%d.%d\n", name, major, minor);

    int passed = 0, total = 11;
    passed += RunCase1Mode1MultiWave(ctx, lp_stream, hp_stream) ? 1 : 0;
    passed += RunCase2Mode5SingleWave(ctx, lp_stream, hp_stream) ? 1 : 0;
    passed += RunCase3Mode5MultiWaveZeroLeak(ctx, lp_stream, hp_stream) ? 1 : 0;
    passed += RunCase4GridBlock3D(ctx, lp_stream, hp_stream) ? 1 : 0;
    passed += RunCase5HighRegs(ctx, lp_stream, hp_stream) ? 1 : 0;
    passed += RunCase6DynamicSmem(ctx, lp_stream, hp_stream) ? 1 : 0;
    passed += RunCase7LmemRecursion(ctx, lp_stream, hp_stream) ? 1 : 0;
    passed += RunCase8NamedBarriers(ctx, lp_stream, hp_stream) ? 1 : 0;
    passed += RunCase9NestedDivergence(ctx, lp_stream, hp_stream) ? 1 : 0;
    passed += RunCase10XSchedFifoAndDirectLaunch(ctx, lp_stream, hp_stream) ? 1 : 0;
    passed += RunCase11Stress100Rounds(ctx, lp_stream) ? 1 : 0;

    std::printf("\n=== SUMMARY: %d / %d CASES PASSED (%s) ===\n",
                passed, total, (passed == total) ? "ALL_PASS" : "FAIL");

    cuStreamDestroy(hp_stream);
    cuStreamDestroy(lp_stream);
    cuDevicePrimaryCtxRelease(dev);
    return (passed == total) ? 0 : 1;
}
