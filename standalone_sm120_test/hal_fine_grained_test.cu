#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include "xsched/cuda/hal/level3/trap_sm120.h"

__device__ unsigned long long get_globaltimer() {
    unsigned long long r;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r));
    return r;
}

__global__ void non_idempotent_k1(volatile unsigned long long *progress,
                                  volatile unsigned long long *t_last,
                                  unsigned long long *atomic_accum,
                                  unsigned int *out_checksums,
                                  unsigned int total_iters) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int state = tid * 0x9e3779b9u + 0x12345678u;
    for (unsigned int it = 1; it <= total_iters; it++) {
        state = state * 1664525u + 1013904223u + (it ^ tid);
        unsigned long long t0 = get_globaltimer();
        while (get_globaltimer() - t0 < 300ULL) {
            state ^= 0x55aa55aau;
            state ^= 0x55aa55aau;
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) *t_last = get_globaltimer();
        if ((it & 0xffu) == 0) {
            atomicAdd((unsigned long long *)atomic_accum, 1ULL);
            if (threadIdx.x == 0) progress[blockIdx.x] = it;
        }
    }
    out_checksums[tid] = state;
    if (threadIdx.x == 0) {
        progress[blockIdx.x] = total_iters | (1ULL << 62);
        if (blockIdx.x == 0) *t_last = get_globaltimer();
    }
}

__global__ void non_idempotent_k2(volatile unsigned long long *progress,
                                  volatile unsigned long long *t_last,
                                  unsigned long long *atomic_accum,
                                  unsigned int *out_checksums,
                                  unsigned int total_iters) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int state = tid * 0x9e3779b9u + 0x12345678u;
    for (unsigned int it = 1; it <= total_iters; it++) {
        state = state * 1664525u + 1013904223u + (it ^ tid);
        unsigned long long t0 = get_globaltimer();
        while (get_globaltimer() - t0 < 300ULL) {
            state ^= 0x55aa55aau;
            state ^= 0x55aa55aau;
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) *t_last = get_globaltimer();
        if ((it & 0xffu) == 0) {
            atomicAdd((unsigned long long *)atomic_accum, 1ULL);
            if (threadIdx.x == 0) progress[blockIdx.x] = it;
        }
    }
    out_checksums[tid] = state;
    if (threadIdx.x == 0) {
        progress[blockIdx.x] = total_iters | (1ULL << 62);
        if (blockIdx.x == 0) *t_last = get_globaltimer();
    }
}

__global__ void worker_low(volatile unsigned long long *progress,
                           volatile unsigned long long *t_last,
                           unsigned long long duration_ns) {
    unsigned long long t0 = get_globaltimer();
    unsigned long long local_cnt = 0;
    while (get_globaltimer() - t0 < duration_ns) {
        local_cnt++;
        if ((local_cnt & 0xff) == 0 && threadIdx.x == 0) {
            progress[blockIdx.x] = local_cnt;
            if (blockIdx.x == 0) *t_last = get_globaltimer();
        }
    }
    if (threadIdx.x == 0) {
        progress[blockIdx.x] = local_cnt | (1ULL << 62);
        if (blockIdx.x == 0) *t_last = get_globaltimer();
    }
}

__global__ void worker_high(volatile unsigned long long *progress,
                            volatile unsigned long long *t_last,
                            unsigned long long duration_ns) {
    unsigned long long t0 = get_globaltimer();
    unsigned long long local_cnt = 0;
    while (get_globaltimer() - t0 < duration_ns) {
        local_cnt++;
        if ((local_cnt & 0xff) == 0 && threadIdx.x == 0) {
            progress[blockIdx.x] = local_cnt;
            if (blockIdx.x == 0) *t_last = get_globaltimer();
        }
    }
    if (threadIdx.x == 0) {
        progress[blockIdx.x] = local_cnt | (1ULL << 62);
        if (blockIdx.x == 0) *t_last = get_globaltimer();
    }
}

__global__ void warmup_k() {}

int main() {
    cudaFree(0);
    warmup_k<<<1, 1>>>();
    cudaDeviceSynchronize();

    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);

    auto &trap = xsched::cuda::TrapSm120Context::Instance();
    if (!trap.InstallFineGrainedTrapHandler(ctx)) {
        printf("Failed to install fine-grained trap handler\n");
        return 1;
    }

    CUstream stream_low, stream_high;
    cuStreamCreate(&stream_low, CU_STREAM_NON_BLOCKING);
    cuStreamCreate(&stream_high, CU_STREAM_NON_BLOCKING);

    volatile uint32_t *preempt_buf_low, *preempt_buf_high;
    volatile unsigned long long *prog_low, *prog_high, *t_last_low, *t_last_high, *atomic_accum;
    unsigned int *out_checksums;
    cudaHostAlloc((void **)&preempt_buf_low, 256, cudaHostAllocMapped);
    cudaHostAlloc((void **)&preempt_buf_high, 256, cudaHostAllocMapped);
    cudaHostAlloc((void **)&prog_low, 15 * 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&prog_high, 15 * 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&t_last_low, 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&t_last_high, 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&atomic_accum, 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&out_checksums, 15 * 128 * sizeof(unsigned int), cudaHostAllocMapped);

    CUfunction f_k1 = nullptr, f_k2 = nullptr, f_low = nullptr, f_high = nullptr;
    cudaGetFuncBySymbol(&f_k1, (const void *)non_idempotent_k1);
    cudaGetFuncBySymbol(&f_k2, (const void *)non_idempotent_k2);
    cudaGetFuncBySymbol(&f_low, (const void *)worker_low);
    cudaGetFuncBySymbol(&f_high, (const void *)worker_high);

    // 1. Mode 2 via TrapSm120Context API
    memset((void *)preempt_buf_low, 0, 256);
    memset((void *)preempt_buf_high, 0, 256);
    memset((void *)prog_low, 0, 15 * 8);
    memset((void *)prog_high, 0, 15 * 8);
    *t_last_low = 0; *t_last_high = 0; *atomic_accum = 0;
    trap.BindKernelPreemptBuffer(f_k1, (uint64_t)(uintptr_t)preempt_buf_low, 101);
    trap.BindKernelPreemptBuffer(f_high, (uint64_t)(uintptr_t)preempt_buf_high, 102);

    unsigned int total_iters = 204800;
    void *args_k1[] = { &prog_low, &t_last_low, &atomic_accum, &out_checksums, &total_iters };
    unsigned long long dur_high = 40000000ULL;
    void *args_high[] = { &prog_high, &t_last_high, &dur_high };
    cuLaunchKernel(f_k1, 15, 1, 1, 128, 1, 1, 0, stream_low, args_k1, nullptr);
    cuLaunchKernel(f_high, 15, 1, 1, 128, 1, 1, 0, stream_high, args_high, nullptr);
    for (int w = 0; w < 10000 && (*t_last_low == 0 || *t_last_high == 0); w++) usleep(10);
    usleep(10000);

    trap.PauseStreamInPlace(preempt_buf_low);
    usleep(200);
    unsigned long long p1 = prog_low[0], hp1 = prog_high[0];
    usleep(10000);
    unsigned long long p2 = prog_low[0], hp2 = prog_high[0];
    trap.ResumeStreamInPlace(preempt_buf_low);
    cuStreamSynchronize(stream_high);
    cuStreamSynchronize(stream_low);
    int total_errors = 0;
    unsigned long long expected_atomic = (unsigned long long)(15 * 128) * (total_iters / 256);
    printf("[HAL Mode 2] low_delta=%llu (expected 0) | high_delta=+%llu | atomic=%llu/1536000\n",
           p2 - p1, hp2 - hp1, *atomic_accum);
    if (p2 != p1 || hp2 <= hp1 || *atomic_accum != expected_atomic) total_errors++;

    // 2. Mode 3 via TrapSm120Context API
    memset((void *)preempt_buf_low, 0, 256);
    memset((void *)prog_low, 0, 15 * 8);
    *t_last_low = 0; *atomic_accum = 0;
    trap.BindKernelPreemptBuffer(f_k2, (uint64_t)(uintptr_t)preempt_buf_low, 103);
    cuLaunchKernel(f_k2, 15, 1, 1, 128, 1, 1, 0, stream_low, args_k1, nullptr);
    for (int w = 0; w < 10000 && (*t_last_low == 0 || prog_low[8] == 0); w++) usleep(10);
    usleep(10000);

    uint32_t cta_modes[15];
    for (int i = 0; i < 8; i++) cta_modes[i] = 2;
    for (int i = 8; i < 15; i++) cta_modes[i] = 0;
    trap.ConfigureAndPauseCtasInPlace(preempt_buf_low, cta_modes, 15);
    usleep(200);
    unsigned long long c0_1 = prog_low[0], c8_1 = prog_low[8];
    usleep(10000);
    unsigned long long c0_2 = prog_low[0], c8_2 = prog_low[8];
    trap.ResumeStreamInPlace(preempt_buf_low);
    cuStreamSynchronize(stream_low);
    printf("[HAL Mode 3] cta0_delta=%llu (expected 0) | cta8_delta=+%llu | atomic=%llu/1536000\n",
           c0_2 - c0_1, (c8_2 & ~(1ULL << 62)) - (c8_1 & ~(1ULL << 62)), *atomic_accum);
    if (c0_2 != c0_1 || (c8_2 & ~(1ULL << 62)) <= (c8_1 & ~(1ULL << 62)) || *atomic_accum != expected_atomic) total_errors++;

    // 3. Mode 1 via TrapSm120Context API
    memset((void *)preempt_buf_low, 0, 256);
    memset((void *)prog_low, 0, 15 * 8);
    *t_last_low = 0;
    trap.BindKernelPreemptBuffer(f_low, (uint64_t)(uintptr_t)preempt_buf_low, 104);
    unsigned long long dur_low = 3000000000ULL;
    void *args_low[] = { &prog_low, &t_last_low, &dur_low };
    cuLaunchKernel(f_low, 15, 1, 1, 128, 1, 1, 0, stream_low, args_low, nullptr);
    for (int w = 0; w < 10000 && *t_last_low == 0; w++) usleep(10);
    usleep(10000);

    trap.EarlyExitStream(preempt_buf_low);
    cuStreamSynchronize(stream_low);
    int flags = 0;
    for (int i = 0; i < 15; i++) if (preempt_buf_low[5 + 2 * i] == 1) flags++;
    uint64_t recorded_idx = *(volatile uint64_t *)&preempt_buf_low[2];
    printf("[HAL Mode 1] preempt_idx=%llu (expected 104) | restore_flags=%d/15\n",
           (unsigned long long)recorded_idx, flags);
    if (recorded_idx != 104ULL || flags != 15) total_errors++;

    trap.RestoreOriginalTrapHandler();
    return (total_errors == 0) ? 0 : 1;
}
