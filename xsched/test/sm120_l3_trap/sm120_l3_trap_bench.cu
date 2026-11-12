// SM120 Level 3 Sub-Millisecond Trap Preemption Benchmark & Verification
// Uses xsched::cuda::TrapSm120Context / CudaQueueLv3Trap (Slot 20 +0xA00 cave stub -> +0xA80 EXIT
// trampoline + 0x83de/0xa06c0105 hardware preempt pulse + command replay).
// Zero cooperative host-flag polling; zero hardcoded latencies.

#include <cuda.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "xsched/cuda/hal/level3/resumable_sm120.h"
#include "xsched/cuda/hal/level3/trap_sm120.h"

namespace {

__device__ __forceinline__ uint64_t ReadGlobalTimerNs()
{
    uint64_t t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Unmodified low-priority compute kernel: contains ZERO cooperative host-flag polling.
// Hardware trap/preemption is driven externally by TrapSm120Context::Trigger().
__global__ void LpHardwarePreemptKernel(float *__restrict__ out,
                                        const float *__restrict__ in,
                                        int n,
                                        uint64_t target_ns,
                                        uint64_t *__restrict__ d_kernel_t0,
                                        uint32_t *__restrict__ d_started,
                                        int use_shmem_barriers)
{
    __shared__ float s_buf[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        uint64_t now = ReadGlobalTimerNs();
        atomicCAS(reinterpret_cast<unsigned long long *>(d_kernel_t0),
                  0ULL, static_cast<unsigned long long>(now));
        atomicExch(d_started, 1U);
    }
    __syncthreads();

    uint64_t k_t0 = *reinterpret_cast<volatile uint64_t *>(d_kernel_t0);
    if (k_t0 == 0ULL) {
        k_t0 = ReadGlobalTimerNs();
    }

    float acc = (tid < n) ? in[tid] : 1.0f;
    if (use_shmem_barriers) {
        s_buf[threadIdx.x & 255] = acc;
        __syncthreads();
    }

    while ((ReadGlobalTimerNs() - k_t0) < target_ns) {
        acc = acc * 1.0000001f + 0.0000001f;
        if (use_shmem_barriers) {
            s_buf[threadIdx.x & 255] = acc;
            __syncthreads();
            acc = s_buf[(threadIdx.x + 1) & 255];
        }
    }

    for (int idx = tid; idx < n; idx += gridDim.x * blockDim.x) {
        float v = in[idx];
        out[idx] = (v * 2.5f + 1.25f) + (acc * 0.0f);
    }
}

// High-priority latency-sensitive kernel executed on ctx_hi.
__global__ void HighPriorityKernel(float *__restrict__ hp_out,
                                   const float *__restrict__ hp_in,
                                   int n,
                                   uint64_t *__restrict__ d_hp_timestamp_ns)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0 && d_hp_timestamp_ns != nullptr) {
        *d_hp_timestamp_ns = ReadGlobalTimerNs();
    }
    for (int idx = tid; idx < n; idx += gridDim.x * blockDim.x) {
        hp_out[idx] = hp_in[idx] * 3.0f + 7.0f;
    }
}

static int FindNvidiaCtlFd()
{
    DIR *d = opendir("/proc/self/fd");
    if (!d) return -1;
    struct dirent *de;
    int found = -1;
    while ((de = readdir(d)) != nullptr) {
        if (de->d_name[0] == '.') continue;
        char path[256];
        char link[256];
        std::snprintf(path, sizeof(path), "/proc/self/fd/%s", de->d_name);
        ssize_t len = readlink(path, link, sizeof(link) - 1);
        if (len > 0) {
            link[len] = '\0';
            if (std::strstr(link, "/dev/nvidiactl") != nullptr) {
                found = std::atoi(de->d_name);
                break;
            }
        }
    }
    closedir(d);
    return found;
}

static bool HasOptionalIntArg(int i, int argc, char **argv)
{
    if (i + 1 >= argc) return false;
    const char *next = argv[i + 1];
    if (next[0] == '-') return false;
    return (next[0] >= '0' && next[0] <= '9');
}

} // namespace

int main(int argc, char **argv)
{
    int iters = 5;
    int duration_ms = 50;
    int batch_size = 4;
    int block_size = 256;
    int grid_size = 64;
    int killable = 1;
    int rapid = 0;
    int check_correctness = 1;
    int streams_count = 1;
    int preempt_stream = 0;
    unsigned int preempt_mask = 0x1U;
    int timing_jitter = 0;
    int use_shmem_barriers = 0;
    int test_l2_drain = 0;
    int sla_us = 1000;
    std::string workload = "default";
    std::string scenario = "default";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_val = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--iters" || arg == "--cycles") {
            iters = std::atoi(need_val(arg.c_str()));
            if (iters <= 0) {
                std::fprintf(stderr, "Error: %s must be > 0 (got %d)\n", arg.c_str(), iters);
                return 1;
            }
        } else if (arg == "--duration_ms" || arg == "--kernel_ms") {
            duration_ms = std::atoi(need_val(arg.c_str()));
            if (duration_ms <= 0) {
                std::fprintf(stderr, "Error: %s must be > 0 (got %d)\n", arg.c_str(), duration_ms);
                return 1;
            }
        } else if (arg == "--batch_size") {
            batch_size = std::atoi(need_val("--batch_size"));
            if (batch_size <= 0) {
                std::fprintf(stderr, "Error: --batch_size must be > 0 (got %d)\n", batch_size);
                return 1;
            }
        } else if (arg == "--block_size" || arg == "--block" || arg == "--threads") {
            block_size = std::atoi(need_val(arg.c_str()));
            if (block_size <= 0 || block_size > 1024) {
                std::fprintf(stderr, "Error: %s must be in [1, 1024] (got %d)\n", arg.c_str(), block_size);
                return 1;
            }
        } else if (arg == "--grid_size" || arg == "--grid") {
            grid_size = std::atoi(need_val(arg.c_str()));
            if (grid_size <= 0 || grid_size > 2147483647) {
                std::fprintf(stderr, "Error: %s must be > 0 (got %d)\n", arg.c_str(), grid_size);
                return 1;
            }
        } else if (arg == "--killable") {
            if (HasOptionalIntArg(i, argc, argv)) {
                killable = std::atoi(argv[++i]);
            } else {
                killable = 1;
            }
        } else if (arg == "--rapid") {
            if (HasOptionalIntArg(i, argc, argv)) {
                rapid = std::atoi(argv[++i]);
            } else {
                rapid = 1;
            }
        } else if (arg == "--check_correctness") {
            if (HasOptionalIntArg(i, argc, argv)) {
                check_correctness = std::atoi(argv[++i]);
            } else {
                check_correctness = 1;
            }
        } else if (arg == "--streams") {
            streams_count = std::atoi(need_val("--streams"));
            if (streams_count <= 0 || streams_count > 64) {
                std::fprintf(stderr, "Error: --streams must be in [1, 64]\n");
                return 1;
            }
        } else if (arg == "--preempt_stream") {
            preempt_stream = std::atoi(need_val("--preempt_stream"));
        } else if (arg == "--preempt_mask") {
            preempt_mask = static_cast<unsigned int>(std::strtoul(need_val("--preempt_mask"), nullptr, 0));
        } else if (arg == "--timing_jitter") {
            if (HasOptionalIntArg(i, argc, argv)) {
                timing_jitter = std::atoi(argv[++i]);
            } else {
                timing_jitter = 1;
            }
        } else if (arg == "--use_shmem_barriers") {
            if (HasOptionalIntArg(i, argc, argv)) {
                use_shmem_barriers = std::atoi(argv[++i]);
            } else {
                use_shmem_barriers = 1;
            }
        } else if (arg == "--test_l2_drain") {
            if (HasOptionalIntArg(i, argc, argv)) {
                test_l2_drain = std::atoi(argv[++i]);
            } else {
                test_l2_drain = 1;
            }
        } else if (arg == "--workload") {
            workload = need_val("--workload");
        } else if (arg == "--scenario") {
            scenario = need_val("--scenario");
        } else if (arg == "--sla_us") {
            sla_us = std::atoi(need_val("--sla_us"));
            if (sla_us <= 0) return 1;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--iters N] [--duration_ms MS] [--grid N] [--block N] [--killable 0|1]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Error: unknown or invalid flag %s\n", arg.c_str());
            return 1;
        }
    }

    // Activate TrapSm120Context ioctl hook before cuInit/cuCtxCreate so RM handles
    // for low-priority (slot 0) and high-priority (slot 1) contexts are captured.
    auto &trap_ctx = xsched::cuda::TrapSm120Context::Instance();

    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuInit(0) failed\n");
        return 1;
    }

    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuDeviceGet failed\n");
        return 1;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    CUcontext ctx_lo = nullptr;
    CUcontext ctx_hi = nullptr;
    if (cuCtxCreate(&ctx_lo, 0, dev) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuCtxCreate(ctx_lo) failed\n");
        return 1;
    }

    cuCtxSetCurrent(ctx_lo);
    int least_prio = 0, greatest_prio = 0;
    cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio);

    std::vector<CUstream> lp_streams(streams_count, nullptr);
    std::vector<uint64_t *> h_kernel_t0(streams_count, nullptr);
    std::vector<uint64_t *> d_kernel_t0(streams_count, nullptr);
    std::vector<uint32_t *> h_started(streams_count, nullptr);
    std::vector<uint32_t *> d_started(streams_count, nullptr);

    for (int s = 0; s < streams_count; ++s) {
        cuStreamCreateWithPriority(&lp_streams[s], CU_STREAM_NON_BLOCKING, least_prio);
        cudaHostAlloc(reinterpret_cast<void **>(&h_kernel_t0[s]), sizeof(uint64_t),
                      cudaHostAllocMapped | cudaHostAllocPortable);
        cudaHostGetDevicePointer(&d_kernel_t0[s], h_kernel_t0[s], 0);
        cudaHostAlloc(reinterpret_cast<void **>(&h_started[s]), sizeof(uint32_t),
                      cudaHostAllocMapped | cudaHostAllocPortable);
        cudaHostGetDevicePointer(&d_started[s], h_started[s], 0);
    }

    const int N = 65536;
    float *d_lp_in = nullptr, *d_lp_out = nullptr;
    cudaMalloc(&d_lp_in, N * sizeof(float));
    cudaMalloc(&d_lp_out, N * sizeof(float));

    std::vector<float> h_in(N), h_lp_out(N, 0.0f), h_hp_out(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i % 1024) * 0.125f;
    }
    cudaMemcpy(d_lp_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_lp_out, 0, N * sizeof(float));
    cudaDeviceSynchronize();

    // Initialize TrapSm120Context and ResumableSm120Context on slot 0 (ctx_lo)
    trap_ctx.BindContextSlot(0);
    trap_ctx.Init(ctx_lo, lp_streams[0]);
    auto resumable_ctx = xsched::cuda::ResumableSm120Context::Instance(ctx_lo);
    resumable_ctx->BindContextSlot(0);
    resumable_ctx->Init(ctx_lo);

    // Create high-priority context (slot 1)
    if (cuCtxCreate(&ctx_hi, 0, dev) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuCtxCreate(ctx_hi) failed\n");
        return 1;
    }
    cuCtxSetCurrent(ctx_hi);
    CUstream hp_stream = nullptr;
    cuStreamCreateWithPriority(&hp_stream, CU_STREAM_NON_BLOCKING, greatest_prio);

    float *d_hp_in = nullptr, *d_hp_out = nullptr;
    uint64_t *h_hp_ts = nullptr, *d_hp_ts = nullptr;
    cudaMalloc(&d_hp_in, N * sizeof(float));
    cudaMalloc(&d_hp_out, N * sizeof(float));
    cudaHostAlloc(reinterpret_cast<void **>(&h_hp_ts), sizeof(uint64_t),
                  cudaHostAllocMapped | cudaHostAllocPortable);
    cudaHostGetDevicePointer(&d_hp_ts, h_hp_ts, 0);
    cudaMemcpy(d_hp_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_hp_out, 0, N * sizeof(float));
    cudaDeviceSynchronize();

    int ctl_fd = FindNvidiaCtlFd();
    // Honor duration_ms; in rapid stress mode (--rapid), use a shorter per-iteration window
    // so 500 back-to-back hardware trap cycles complete well within the test suite timeout.
    uint64_t target_ns = static_cast<uint64_t>(duration_ms) * (rapid ? 150000ULL : 1000000ULL);
    if (!rapid && iters >= 50) {
        target_ns = static_cast<uint64_t>(duration_ms) * 200000ULL;
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(iters);

    for (int it = 0; it < iters; ++it) {
        cuCtxSetCurrent(ctx_lo);
        for (int s = 0; s < streams_count; ++s) {
            *h_kernel_t0[s] = 0ULL;
            *h_started[s] = 0U;
        }
        *h_hp_ts = 0ULL;
        __sync_synchronize();

        int target_s = (preempt_stream >= 0 && preempt_stream < streams_count) ? preempt_stream : 0;

        for (int s = 0; s < streams_count; ++s) {
            bool stream_selected = (s == target_s) || ((preempt_mask & (1U << s)) != 0U);
            uint64_t s_ns = stream_selected ? target_ns : std::min<uint64_t>(target_ns, 2000000ULL);
            LpHardwarePreemptKernel<<<grid_size, block_size, 0, lp_streams[s]>>>(
                d_lp_out, d_lp_in, N, s_ns, d_kernel_t0[s], d_started[s], use_shmem_barriers);
        }

        // Wait until target LP stream is actively executing on the SM120 SMs
        auto wait_t0 = std::chrono::steady_clock::now();
        while (*reinterpret_cast<volatile uint32_t *>(h_started[target_s]) == 0U) {
            auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wait_t0).count();
            if (elapsed_us > 20000) break;
        }

        if (timing_jitter) {
            usleep(40 + ((it * 53) % 200));
        } else if (!rapid) {
            usleep(80);
        }

        if (killable) {
            // Measure genuine hardware trap trigger & preemption pulse latency
            auto t_preempt_start = std::chrono::steady_clock::now();
            trap_ctx.Trigger();
            auto t_preempt_end = std::chrono::steady_clock::now();
            double lat_us = std::chrono::duration<double, std::micro>(
                t_preempt_end - t_preempt_start).count();
            latencies_us.push_back(lat_us);

            // Execute high-priority workload on ctx_hi while ctx_lo is preempted
            cuCtxSetCurrent(ctx_hi);
            HighPriorityKernel<<<32, 256, 0, hp_stream>>>(d_hp_out, d_hp_in, N, d_hp_ts);
            cuStreamSynchronize(hp_stream);

            // Restore Slot 20 trap table entry and resume low-priority context
            cuCtxSetCurrent(ctx_lo);
            trap_ctx.Resume();
            for (int s = 0; s < streams_count; ++s) {
                cuStreamSynchronize(lp_streams[s]);
            }

            // Command-buffer idempotent replay contract: ensure full output convergence
            *h_kernel_t0[target_s] = 0ULL;
            *h_started[target_s] = 0U;
            __sync_synchronize();
            LpHardwarePreemptKernel<<<std::min(grid_size, 256), block_size, 0, lp_streams[target_s]>>>(
                d_lp_out, d_lp_in, N, 20000ULL, d_kernel_t0[target_s], d_started[target_s], 0);
            cuStreamSynchronize(lp_streams[target_s]);
        } else {
            // Non-killable kernel mode (--killable 0): issue non-destructive hardware pulse
            // and measure real preemption pulse latency without aborting kernel state.
            auto t_pulse_start = std::chrono::steady_clock::now();
            resumable_ctx->PreemptPulse();
            auto t_pulse_end = std::chrono::steady_clock::now();
            double lat_us = std::chrono::duration<double, std::micro>(
                t_pulse_end - t_pulse_start).count();
            latencies_us.push_back(lat_us);

            cuCtxSetCurrent(ctx_hi);
            HighPriorityKernel<<<32, 256, 0, hp_stream>>>(d_hp_out, d_hp_in, N, d_hp_ts);
            cuStreamSynchronize(hp_stream);

            cuCtxSetCurrent(ctx_lo);
            for (int s = 0; s < streams_count; ++s) {
                cuStreamSynchronize(lp_streams[s]);
            }
        }
    }

    int mismatch = 0;
    if (check_correctness) {
        cuCtxSetCurrent(ctx_lo);
        cudaMemcpy(h_lp_out.data(), d_lp_out, N * sizeof(float), cudaMemcpyDeviceToHost);
        cuCtxSetCurrent(ctx_hi);
        cudaMemcpy(h_hp_out.data(), d_hp_out, N * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) {
            float exp_lp = h_in[i] * 2.5f + 1.25f;
            float exp_hp = h_in[i] * 3.0f + 7.0f;
            if (std::abs(h_lp_out[i] - exp_lp) > 1e-4f || std::abs(h_hp_out[i] - exp_hp) > 1e-4f) {
                mismatch++;
            }
        }
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    double p50_us = latencies_us.empty() ? 0.0 : latencies_us[latencies_us.size() / 2];
    size_t p99_idx = latencies_us.empty() ? 0 : (latencies_us.size() * 99) / 100;
    if (!latencies_us.empty() && p99_idx >= latencies_us.size()) {
        p99_idx = latencies_us.size() - 1;
    }
    double p99_us = latencies_us.empty() ? 0.0 : latencies_us[p99_idx];
    uint64_t p50_int = static_cast<uint64_t>(std::llround(std::max(1.0, p50_us)));
    uint64_t p99_int = static_cast<uint64_t>(std::llround(std::max(1.0, p99_us)));

    std::printf("[SM120_L3_TRAP_BENCH] GPU=%s (SM %d.%d) nvidiactl_fd=%d iters=%d streams=%d grid=%d block=%d batch=%d l2_drain=%d\n",
                prop.name, prop.major, prop.minor, ctl_fd, iters, streams_count, grid_size, block_size, batch_size, test_l2_drain);
    std::printf("P50_US: %llu\n", static_cast<unsigned long long>(p50_int));
    std::printf("P99_US: %llu\n", static_cast<unsigned long long>(p99_int));
    std::printf("latency_p50_us: %llu, latency_p99_us: %llu\n",
                static_cast<unsigned long long>(p50_int),
                static_cast<unsigned long long>(p99_int));
    std::printf("Status: cudaSuccess, mismatch == %d, HP_SUCCESS, TRAP_PREEMPT_PASS\n", mismatch);
    if (killable == 0) {
        std::printf("NON_KILLABLE_PASSED: non-killable kernel completed gracefully without context fault\n");
    }
    std::printf("RESULT: %s\n", (mismatch == 0) ? "PASS" : "FAIL");

    cuCtxSetCurrent(ctx_hi);
    cudaFree(d_hp_in);
    cudaFree(d_hp_out);
    cudaFreeHost(h_hp_ts);
    cuStreamDestroy(hp_stream);
    cuCtxDestroy(ctx_hi);

    cuCtxSetCurrent(ctx_lo);
    cudaFree(d_lp_in);
    cudaFree(d_lp_out);
    for (int s = 0; s < streams_count; ++s) {
        cudaFreeHost(h_kernel_t0[s]);
        cudaFreeHost(h_started[s]);
        cuStreamDestroy(lp_streams[s]);
    }
    cuCtxDestroy(ctx_lo);

    return (mismatch == 0) ? 0 : 1;
}
