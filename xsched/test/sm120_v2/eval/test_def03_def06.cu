#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <thread>
#include <vector>
#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"

__global__ void k_work(uint32_t *d_out, volatile uint32_t *d_started, int iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *d_started = 1;
        __threadfence_system();
    }
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x = static_cast<uint32_t>(idx + 7);
    for (int i = 0; i < iters; ++i) x = x * 1664525u + 1013904223u;
    d_out[idx] = x;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    cudaSetDevice(0); cudaFree(0);

    cudaStream_t lp = nullptr, hp = nullptr;
    cudaStreamCreateWithFlags(&lp, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&hp, cudaStreamNonBlocking);
    HwQueueHandle hwq = 0; XQueueHandle xq = 0;
    CudaQueueCreate(&hwq, lp);
    XQueueCreate(&xq, hwq, kPreemptLevelInterrupt, kQueueCreateFlagNone);

    const int N = 60 * 128;
    uint32_t *d_ref = nullptr, *d_lp = nullptr;
    cudaMalloc(&d_ref, N * sizeof(uint32_t));
    cudaMalloc(&d_lp, N * sizeof(uint32_t));
    uint32_t *h_started = nullptr, *d_started = nullptr;
    cudaHostAlloc(&h_started, 4, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_started, h_started, 0);

    // Reference
    k_work<<<60, 128, 0, hp>>>(d_ref, d_started, 8000000);
    cudaStreamSynchronize(hp);
    std::vector<uint32_t> href(N), hlp(N);
    cudaMemcpy(href.data(), d_ref, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Warmup LP instrumentation
    *h_started = 0;
    k_work<<<60, 128, 0, lp>>>(d_lp, d_started, 100);
    cudaStreamSynchronize(lp);

    // Test 1: DEF-06 — launch kernel & memcpy from a fresh std::thread with NO cuCtxSetCurrent!
    *h_started = 0;
    bool def06_ok = false;
    std::thread worker([&]() {
        CUcontext cur = nullptr;
        cuCtxGetCurrent(&cur);
        // cur MUST be nullptr on a fresh thread!
        CUfunction fn = nullptr;
        cudaFunction_t cf = nullptr;
        cudaGetFuncBySymbol(&cf, (const void*)k_work);
        fn = (CUfunction)cf;
        int iters = 8000000;
        void *args[3] = { &d_lp, &d_started, &iters };
        CUresult r = cuLaunchKernel(fn, 60, 1, 1, 128, 1, 1, 0, (CUstream)lp, args, nullptr);
        def06_ok = (cur == nullptr && r == CUDA_SUCCESS);
    });
    worker.join();

    // Test 2: DEF-03 — suspend LP mid-kernel, then call cudaMalloc + cudaFree (XMemFree_v2)!
    while (*h_started == 0) {}
    auto t0 = std::chrono::steady_clock::now();
    XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue);
    auto t1 = std::chrono::steady_clock::now();

    auto tf0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 8; ++i) {
        void *tmp = nullptr;
        cudaMalloc(&tmp, 4 * 1024 * 1024);
        cudaMemsetAsync(tmp, i, 4 * 1024 * 1024, hp);
        cudaStreamSynchronize(hp);
        cudaFree(tmp); // Calls cuMemFree_v2 -> XMemFree_v2 while LP is suspended!
    }
    auto tf1 = std::chrono::steady_clock::now();

    XQueueResume(xq, kQueueResumeFlagNone);
    cudaStreamSynchronize(lp);
    cudaMemcpy(hlp.data(), d_lp, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    size_t mismatch = 0;
    for (int i = 0; i < N; ++i) if (hlp[i] != href[i]) ++mismatch;

    double susp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double free_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();
    printf("[DEF-06] Context-Free Worker Thread (cuCtxGetCurrent==NULL -> StreamGetCtx): %s\n",
           def06_ok ? "PASS" : "FAIL");
    printf("[DEF-03] Mid-Suspend 8x(cudaMalloc+cudaFree 4MiB): susp=%.3fms free_loop=%.3fms mismatch=%zu/%d -> %s\n",
           susp_ms, free_ms, mismatch, N, (mismatch == 0) ? "PASS" : "FAIL");

    XQueueDestroy(xq);
    HwQueueDestroy(hwq);
    cudaFreeHost(h_started);
    cudaFree(d_ref);
    cudaFree(d_lp);
    cudaStreamDestroy(hp);
    cudaStreamDestroy(lp);
    return (def06_ok && mismatch == 0) ? 0 : 1;
}
