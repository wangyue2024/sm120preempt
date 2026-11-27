// Phase 3.5 diagnostic: which driver operations need free SM resources?
// Occupies every SM completely (2 x 768 threads, ~48 KB shared memory each) with spinning CTAs, then
// issues control-path operations on another stream and reports whether each completes while the
// SMs stay full. Native CUDA only (no XSched).  nvcc -o ce_vs_sm_probe ce_vs_sm_probe.cu -lcuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <thread>

__global__ void spin(volatile int *flag)
{
    extern __shared__ int s[];
    s[threadIdx.x] = threadIdx.x;
    while (*flag == 0) { __nanosleep(1000); }
}

static bool done_within(CUstream st, int ms)
{
    auto t0 = std::chrono::steady_clock::now();
    while (cuStreamQuery(st) == CUDA_ERROR_NOT_READY) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(ms)) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
}

int main()
{
    cudaSetDevice(0);
    cudaFree(0);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int *flag = nullptr;
    cudaHostAlloc(&flag, 4, cudaHostAllocMapped);
    *flag = 0;
    const int smem = 48 * 1024;
    cudaFuncSetAttribute(spin, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    CUstream busy, ctl;
    cuStreamCreate(&busy, CU_STREAM_NON_BLOCKING);
    int lo, hi;
    cuCtxGetStreamPriorityRange(&lo, &hi);
    cuStreamCreateWithPriority(&ctl, CU_STREAM_NON_BLOCKING, hi);
    CUdeviceptr d = 0;
    cuMemAlloc(&d, 1 << 20);
    void *h = nullptr, *big = nullptr;
    cuMemHostAlloc(&h, 4096, CU_MEMHOSTALLOC_PORTABLE);
    cuMemHostAlloc(&big, 1 << 20, CU_MEMHOSTALLOC_PORTABLE);
    // Baseline: each op alone completes.
    spin<<<sms * 2, 768, smem, (cudaStream_t)busy>>>(flag);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    printf("[CE-VS-SM] SMs %d full (spin kernel running: %s)\n", sms,
           cuStreamQuery(busy) == CUDA_ERROR_NOT_READY ? "yes" : "NO");
    struct Op { const char *name; int kind; } ops[] = {
        {"cuMemcpyHtoDAsync 4B", 0}, {"cuMemcpyDtoHAsync 80B", 1}, {"cuMemsetD8Async 32B", 2},
        {"cuMemsetD32Async 1 word", 3}, {"cuMemsetD2D32Async 256 x 1 (stride 128B)", 4},
        {"cuMemsetD8Async 32KB", 5}, {"cuMemcpyDtoDAsync 4KB", 6},
        {"cuMemcpy2DAsync HtoD 4B x 256 (dst pitch 128B)", 7}, {"cuMemcpyHtoDAsync 32KB", 8},
        {"cuMemcpyHtoDAsync 1MB", 9}, {"cuMemsetD32Async 64 words", 10}, {"cuMemsetD32Async 256 words", 11},
        {"cuMemsetD8Async 4KB", 12}, {"cuMemsetD8Async 1MB", 13}};
    for (auto &op : ops) {
        switch (op.kind) {
        case 0: cuMemcpyHtoDAsync(d, h, 4, ctl); break;
        case 1: cuMemcpyDtoHAsync(h, d, 80, ctl); break;
        case 2: cuMemsetD8Async(d, 0, 32, ctl); break;
        case 3: cuMemsetD32Async(d + 0x44, 7, 1, ctl); break;
        case 4: cuMemsetD2D32Async(d, 128, 5, 1, 256, ctl); break;
        case 5: cuMemsetD8Async(d, 0, 32768, ctl); break;
        case 6: cuMemcpyDtoDAsync(d + 65536, d, 4096, ctl); break;
        case 7: {
            CUDA_MEMCPY2D c = {};
            c.srcMemoryType = CU_MEMORYTYPE_HOST; c.srcHost = h; c.srcPitch = 4;
            c.dstMemoryType = CU_MEMORYTYPE_DEVICE; c.dstDevice = d; c.dstPitch = 128;
            c.WidthInBytes = 4; c.Height = 256;
            cuMemcpy2DAsync(&c, ctl);
            break;
        }
        case 8: cuMemcpyHtoDAsync(d, big, 32768, ctl); break;
        case 9: cuMemcpyHtoDAsync(d, big, 1 << 20, ctl); break;
        case 10: cuMemsetD32Async(d, 3, 64, ctl); break;
        case 11: cuMemsetD32Async(d, 3, 256, ctl); break;
        case 12: cuMemsetD8Async(d, 0, 4096, ctl); break;
        case 13: cuMemsetD8Async(d, 0, 1 << 20, ctl); break;
        }
        bool ok = done_within(ctl, 500);
        printf("[CE-VS-SM] %-48s %s\n", op.name, ok ? "completes while SMs are full" : "BLOCKED until SMs free");
        if (!ok) {   // drain it before the next probe
            *flag = 1;
            cuStreamSynchronize(busy);
            cuStreamSynchronize(ctl);
            *flag = 0;
            spin<<<sms * 2, 768, smem, (cudaStream_t)busy>>>(flag);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    *flag = 1;
    cuStreamSynchronize(busy);
    printf("[CE-VS-SM] done (%s)\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
