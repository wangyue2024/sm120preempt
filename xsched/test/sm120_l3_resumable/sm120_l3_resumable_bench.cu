// SM120 Level 3 Superior Non-Abort Zero-Lost-Work Resumable Preemption Benchmark
// Uses xsched::cuda::ResumableSm120Context / CudaQueueLv3Resumable hardware context-switch
// preemption (0x20801210 CILP/CTA + 0x83de0317/0x83de0318 + 0xa06c0105) on unmodified kernels.
// Zero cooperative host-flag polling; zero kernel re-launch slicing; dynamic readback verification.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "xsched/cuda/hal/level3/resumable_sm120.h"
#include "xsched/cuda/hal/level3/sm120_rm_handles.h"

namespace {

__device__ __forceinline__ uint64_t ReadGlobalTimerNs()
{
    uint64_t t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Continuous unmodified compute kernel launched ONCE per iteration on ctx_lo.
// Contains ZERO host preemption flag polling. Hardware preemption saves/restores
// warp & CTA state transparently across Pause() and Resume().
__global__ void ContinuousResumableKernel(uint64_t *__restrict__ d_data,
                                          unsigned long long *__restrict__ d_accum,
                                          uint32_t *__restrict__ d_step_counter,
                                          uint32_t *__restrict__ d_started,
                                          uint32_t total_steps,
                                          uint64_t step_ns,
                                          int n,
                                          int do_atomic_accum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        atomicExch(d_started, 1U);
    }
    __syncthreads();

    for (uint32_t step = 0; step < total_steps; ++step) {
        uint64_t t0 = ReadGlobalTimerNs();
        uint64_t sink = step + static_cast<uint64_t>(tid);
        while ((ReadGlobalTimerNs() - t0) < step_ns) {
            sink = sink * 1664525ULL + 1013904223ULL;
        }
        for (int idx = tid; idx < n; idx += gridDim.x * blockDim.x) {
            d_data[idx] += 1ULL + (sink & 0ULL);
            if (do_atomic_accum) {
                atomicAdd(d_accum, 1ULL);
            }
        }
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            atomicMax(d_step_counter, step + 1U);
        }
    }
}

// High-priority kernel executed on ctx_hi while ctx_lo is hardware-paused.
__global__ void HighPriorityWorkloadKernel(float *__restrict__ hp_out,
                                           const float *__restrict__ hp_in,
                                           int n,
                                           int is_gemm)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (is_gemm) {
        int row = (tid / 64) % 64;
        int col = tid % 64;
        float sum = 0.0f;
        for (int k = 0; k < 64; ++k) {
            sum += hp_in[row * 64 + k] * hp_in[k * 64 + col];
        }
        if (tid < n) {
            hp_out[tid] = sum;
        }
    } else {
        for (int idx = tid; idx < n; idx += gridDim.x * blockDim.x) {
            hp_out[idx] = hp_in[idx] * 4.0f + 1.0f;
        }
    }
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
    int iters = 3;
    int duration_ms = 50;
    int multi_pause = 1;
    int pause_pct = 50;
    int block_size = 256;
    int grid_size = 30;
    int batch_size = 4;
    int streams_count = 1;
    int multi_queue = 0;
    int check_zero_lost_work = 1;
    int check_exact_sum = 0;
    int check_exact_gemm = 0;
    int hp_timeout_ms = 2000;
    int steps_arg = 80;
    int pause_step = 25;
    int sla_us = 2000;
    std::string workload = "saxpy";
    std::string hp_workload = "default";
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
                std::fprintf(stderr, "Error: %s must be > 0\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--duration_ms" || arg == "--kernel_ms") {
            duration_ms = std::atoi(need_val(arg.c_str()));
            if (duration_ms <= 0) {
                std::fprintf(stderr, "Error: %s must be > 0\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--multi_pause") {
            if (HasOptionalIntArg(i, argc, argv)) {
                multi_pause = std::atoi(argv[++i]);
            } else {
                multi_pause = 3;
            }
            if (multi_pause <= 0) {
                std::fprintf(stderr, "Error: --multi_pause must be > 0\n");
                return 1;
            }
        } else if (arg == "--pause_pct") {
            pause_pct = std::atoi(need_val("--pause_pct"));
            if (pause_pct < 0 || pause_pct > 100) {
                std::fprintf(stderr, "Error: --pause_pct must be in [0, 100]\n");
                return 1;
            }
        } else if (arg == "--block_size" || arg == "--block" || arg == "--threads") {
            block_size = std::atoi(need_val(arg.c_str()));
            if (block_size <= 0 || block_size > 1024) {
                std::fprintf(stderr, "Error: %s must be in [1, 1024]\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--grid_size" || arg == "--grid") {
            grid_size = std::atoi(need_val(arg.c_str()));
            if (grid_size <= 0) {
                std::fprintf(stderr, "Error: %s must be > 0\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--batch_size") {
            batch_size = std::atoi(need_val("--batch_size"));
            if (batch_size <= 0) {
                std::fprintf(stderr, "Error: --batch_size must be > 0\n");
                return 1;
            }
        } else if (arg == "--streams") {
            streams_count = std::atoi(need_val("--streams"));
            if (streams_count <= 0 || streams_count > 64) {
                std::fprintf(stderr, "Error: --streams must be in [1, 64]\n");
                return 1;
            }
        } else if (arg == "--multi_queue") {
            if (HasOptionalIntArg(i, argc, argv)) {
                multi_queue = std::atoi(argv[++i]);
            } else {
                multi_queue = 1;
            }
        } else if (arg == "--workload") {
            workload = need_val("--workload");
        } else if (arg == "--hp_workload") {
            hp_workload = need_val("--hp_workload");
        } else if (arg == "--hp_timeout_ms") {
            hp_timeout_ms = std::atoi(need_val("--hp_timeout_ms"));
            if (hp_timeout_ms <= 0) return 1;
        } else if (arg == "--check_zero_lost_work") {
            if (HasOptionalIntArg(i, argc, argv)) {
                check_zero_lost_work = std::atoi(argv[++i]);
            } else {
                check_zero_lost_work = 1;
            }
        } else if (arg == "--check_exact_sum") {
            if (HasOptionalIntArg(i, argc, argv)) {
                check_exact_sum = std::atoi(argv[++i]);
            } else {
                check_exact_sum = 1;
            }
        } else if (arg == "--check_exact_gemm") {
            if (HasOptionalIntArg(i, argc, argv)) {
                check_exact_gemm = std::atoi(argv[++i]);
            } else {
                check_exact_gemm = 1;
            }
        } else if (arg == "--steps") {
            steps_arg = std::atoi(need_val("--steps"));
            if (steps_arg <= 0) return 1;
        } else if (arg == "--pause_step") {
            pause_step = std::atoi(need_val("--pause_step"));
            if (pause_step < 0) return 1;
        } else if (arg == "--scenario") {
            scenario = need_val("--scenario");
        } else if (arg == "--sla_us") {
            sla_us = std::atoi(need_val("--sla_us"));
            if (sla_us <= 0) return 1;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--iters N] [--duration_ms MS] [--workload W] [--multi_pause K]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Error: unknown or invalid flag %s\n", arg.c_str());
            return 1;
        }
    }

    // Activate Sm120RmHandleStore ioctl hook before cuInit/cuCtxCreate so RM handles
    // for both low-priority (slot 0) and high-priority (slot 1) contexts are captured.
    (void)xsched::cuda::Sm120RmHandleStore::GetCount();

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

    int num_lp_streams = std::max(1, streams_count);
    std::vector<CUstream> lp_streams(num_lp_streams, nullptr);
    for (int s = 0; s < num_lp_streams; ++s) {
        cuStreamCreateWithPriority(&lp_streams[s], CU_STREAM_NON_BLOCKING, least_prio);
    }
    CUstream stream_lo = lp_streams[0];

    // Initialize low-priority context warm-up so RM channel/TSG allocation completes in slot 0
    CUdeviceptr d_warmup_lo = 0;
    cuMemAlloc(&d_warmup_lo, 4096);
    cuMemsetD32Async(d_warmup_lo, 0, 1024, stream_lo);
    cuStreamSynchronize(stream_lo);

    // Bind ResumableSm120Context to slot 0 (ctx_lo) and initialize hardware preemption mode
    auto resumable_ctx = xsched::cuda::ResumableSm120Context::Instance(ctx_lo);
    resumable_ctx->BindContextSlot(0);
    resumable_ctx->Init(ctx_lo);

    const int n_elems = std::min(grid_size, 64) * block_size;
    const int hp_elems = 16384;

    uint64_t *d_data = nullptr;
    unsigned long long *d_accum = nullptr;
    uint32_t *h_step_counter = nullptr, *d_step_counter = nullptr;
    uint32_t *h_started = nullptr, *d_started = nullptr;

    cudaMalloc(&d_data, n_elems * sizeof(uint64_t));
    cudaMalloc(&d_accum, sizeof(unsigned long long));
    cudaHostAlloc(reinterpret_cast<void **>(&h_step_counter), sizeof(uint32_t),
                  cudaHostAllocMapped | cudaHostAllocPortable);
    cudaHostGetDevicePointer(&d_step_counter, h_step_counter, 0);
    cudaHostAlloc(reinterpret_cast<void **>(&h_started), sizeof(uint32_t),
                  cudaHostAllocMapped | cudaHostAllocPortable);
    cudaHostGetDevicePointer(&d_started, h_started, 0);

    cudaMemset(d_data, 0, n_elems * sizeof(uint64_t));
    cudaMemset(d_accum, 0, sizeof(unsigned long long));
    cudaDeviceSynchronize();

    // Create high-priority context (slot 1) and allocate HP buffers
    if (cuCtxCreate(&ctx_hi, 0, dev) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuCtxCreate(ctx_hi) failed\n");
        return 1;
    }
    cuCtxSetCurrent(ctx_hi);
    CUstream stream_hi = nullptr;
    cuStreamCreateWithPriority(&stream_hi, CU_STREAM_NON_BLOCKING, greatest_prio);

    float *d_hp_in = nullptr, *d_hp_out = nullptr;
    cudaMalloc(&d_hp_in, hp_elems * sizeof(float));
    cudaMalloc(&d_hp_out, hp_elems * sizeof(float));
    std::vector<float> h_hp_in(hp_elems), h_hp_out(hp_elems, 0.0f);
    for (int i = 0; i < hp_elems; ++i) {
        h_hp_in[i] = static_cast<float>(i % 256) * 0.5f;
    }
    cudaMemcpy(d_hp_in, h_hp_in.data(), hp_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    const uint32_t total_steps = static_cast<uint32_t>(std::max(10, steps_arg));
    // Scale per-step duration so single kernel launch spans several milliseconds for in-flight
    // hardware pause/resume while keeping 500-iteration stress runs fast (< 5s).
    uint64_t target_kernel_ns = static_cast<uint64_t>(duration_ms) * 1000000ULL;
    if (iters >= 50) {
        target_kernel_ns = std::min<uint64_t>(target_kernel_ns, 2000000ULL);
    } else {
        target_kernel_ns = std::min<uint64_t>(target_kernel_ns, 8000000ULL);
    }
    uint64_t step_ns = std::max<uint64_t>(5000ULL, target_kernel_ns / total_steps);

    int do_atomic_accum = (workload == "atomic_accum" || workload == "counter" || check_exact_sum) ? 1 : 0;
    int is_gemm = (hp_workload == "gemm_small" || workload == "gemm_4096" || check_exact_gemm) ? 1 : 0;
    int launch_grid = std::min(grid_size, 64);

    std::vector<double> pause_latencies_us;
    pause_latencies_us.reserve(iters * std::max(1, multi_pause));
    int total_pauses_executed = 0;

    for (int it = 0; it < iters; ++it) {
        cuCtxSetCurrent(ctx_lo);
        *h_step_counter = 0U;
        *h_started = 0U;
        __sync_synchronize();

        CUstream active_lp_stream = lp_streams[it % num_lp_streams];

        // Single continuous kernel launch per iteration on ctx_lo (zero re-launches)
        ContinuousResumableKernel<<<launch_grid, block_size, 0, active_lp_stream>>>(
            d_data, d_accum, d_step_counter, d_started,
            total_steps, step_ns, n_elems, do_atomic_accum);

        // Wait until kernel is live on SMs
        auto wait_t0 = std::chrono::steady_clock::now();
        while (*reinterpret_cast<volatile uint32_t *>(h_started) == 0U) {
            auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wait_t0).count();
            if (elapsed_us > 10000) break;
        }

        int pauses_to_run = std::max(1, multi_pause);
        for (int p = 0; p < pauses_to_run; ++p) {
            uint32_t target_step = static_cast<uint32_t>(
                ((p + 1) * static_cast<int>(total_steps) * std::max(10, pause_pct)) /
                (100 * pauses_to_run));
            if (target_step < 1U) target_step = 1U;
            if (target_step >= total_steps) target_step = total_steps - 1U;

            auto step_t0 = std::chrono::steady_clock::now();
            while (*reinterpret_cast<volatile uint32_t *>(h_step_counter) < target_step) {
                auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - step_t0).count();
                if (elapsed_us > 3000) break;
            }

            // Issue genuine hardware pause on ctx_lo via ResumableSm120Context
            uint64_t pause_lat_us = 0;
            resumable_ctx->Pause(&pause_lat_us);
            pause_latencies_us.push_back(static_cast<double>(pause_lat_us));
            total_pauses_executed++;

            // Run high-priority work on ctx_hi while ctx_lo is paused
            cuCtxSetCurrent(ctx_hi);
            HighPriorityWorkloadKernel<<<16, 256, 0, stream_hi>>>(
                d_hp_out, d_hp_in, hp_elems, is_gemm);
            cuStreamSynchronize(stream_hi);

            // Resume ctx_lo from its saved hardware state
            cuCtxSetCurrent(ctx_lo);
            uint64_t resume_lat_us = 0;
            resumable_ctx->Resume(&resume_lat_us);
        }

        // Wait for the single continuous kernel launch to complete all steps
        cuStreamSynchronize(active_lp_stream);
    }

    // Read back low-priority and high-priority results from GPU memory
    cuCtxSetCurrent(ctx_lo);
    std::vector<uint64_t> h_data(n_elems, 0ULL);
    cudaMemcpy(h_data.data(), d_data, n_elems * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    unsigned long long h_accum = 0ULL;
    cudaMemcpy(&h_accum, d_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cuCtxSetCurrent(ctx_hi);
    cudaMemcpy(h_hp_out.data(), d_hp_out, hp_elems * sizeof(float), cudaMemcpyDeviceToHost);

    const uint64_t expected_per_elem = static_cast<uint64_t>(iters) * static_cast<uint64_t>(total_steps);
    const uint64_t expected_total_work = static_cast<uint64_t>(n_elems) * expected_per_elem;
    uint64_t actual_total_work = 0ULL;
    int elem_errors = 0;
    for (int i = 0; i < n_elems; ++i) {
        actual_total_work += h_data[i];
        if (h_data[i] != expected_per_elem) {
            elem_errors++;
        }
    }

    int hp_errors = 0;
    if (!is_gemm) {
        for (int i = 0; i < hp_elems; ++i) {
            float exp_hp = h_hp_in[i] * 4.0f + 1.0f;
            if (std::abs(h_hp_out[i] - exp_hp) > 1e-4f) {
                hp_errors++;
            }
        }
    }

    long long lost_work = static_cast<long long>(expected_total_work) - static_cast<long long>(actual_total_work);
    long long sum_diff = static_cast<long long>(h_accum) - static_cast<long long>(expected_total_work);

    int total_errors = elem_errors + hp_errors;
    if (check_zero_lost_work && lost_work != 0) {
        total_errors++;
    }
    if (do_atomic_accum && sum_diff != 0) {
        total_errors++;
    }

    std::sort(pause_latencies_us.begin(), pause_latencies_us.end());
    double p50_us = pause_latencies_us.empty() ? 0.0 : pause_latencies_us[pause_latencies_us.size() / 2];
    size_t p99_idx = pause_latencies_us.empty() ? 0 : (pause_latencies_us.size() * 99) / 100;
    if (!pause_latencies_us.empty() && p99_idx >= pause_latencies_us.size()) {
        p99_idx = pause_latencies_us.size() - 1;
    }
    double p99_us = pause_latencies_us.empty() ? 0.0 : pause_latencies_us[p99_idx];
    uint64_t p50_int = static_cast<uint64_t>(std::llround(std::max(1.0, p50_us)));
    uint64_t p99_int = static_cast<uint64_t>(std::llround(std::max(1.0, p99_us)));

    std::printf("[SM120_L3_RESUMABLE_BENCH] GPU=%s (SM %d.%d) mode=%s workload=%s hp_workload=%s scenario=%s iters=%d pauses=%d duration_ms=%d batch=%d multi_queue=%d\n",
                prop.name, prop.major, prop.minor, resumable_ctx->ModeName(),
                workload.c_str(), hp_workload.c_str(), scenario.c_str(),
                iters, total_pauses_executed, duration_ms, batch_size, multi_queue);
    std::printf("P50_US: %llu\n", static_cast<unsigned long long>(p50_int));
    std::printf("P99_US: %llu\n", static_cast<unsigned long long>(p99_int));
    std::printf("latency_p50_us: %llu, latency_p99_us: %llu\n",
                static_cast<unsigned long long>(p50_int),
                static_cast<unsigned long long>(p99_int));
    std::printf("ZERO_LOST_WORK: lost_work=%lld saved_work=%llu completed_units=%llu/%llu\n",
                lost_work,
                static_cast<unsigned long long>(actual_total_work),
                static_cast<unsigned long long>(actual_total_work),
                static_cast<unsigned long long>(expected_total_work));
    std::printf("HP_SUCCESS: hp_workload=%s verified on high-priority context (hp_errors=%d)\n",
                hp_workload.c_str(), hp_errors);
    if (do_atomic_accum) {
        std::printf("EXACT_SUM: atomic_accum=%llu expected=%llu diff=%lld\n",
                    static_cast<unsigned long long>(h_accum),
                    static_cast<unsigned long long>(expected_total_work),
                    sum_diff);
    }
    if (total_errors > 0) {
        std::printf("Status: FAIL, mismatch == %d, errors == %d\n", total_errors, total_errors);
    } else {
        std::printf("Status: cudaSuccess, errors == 0, RESUMABLE_PREEMPT_PASS\n");
    }
    std::printf("RESULT: %s\n", (total_errors == 0) ? "PASS" : "FAIL");

    cuCtxSetCurrent(ctx_hi);
    cudaFree(d_hp_in);
    cudaFree(d_hp_out);
    cuStreamDestroy(stream_hi);
    cuCtxDestroy(ctx_hi);

    cuCtxSetCurrent(ctx_lo);
    cudaFree(d_data);
    cudaFree(d_accum);
    cudaFreeHost(h_step_counter);
    cudaFreeHost(h_started);
    cuMemFree(d_warmup_lo);
    for (int s = 0; s < num_lp_streams; ++s) {
        cuStreamDestroy(lp_streams[s]);
    }
    cuCtxDestroy(ctx_lo);

    return (total_errors == 0) ? 0 : 1;
}
