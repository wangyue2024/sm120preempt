#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s; cuGetErrorString(r_, &s); printf("ERR %s @%d\n", s, __LINE__); exit(1);} } while (0)
int main(int argc, char **argv) {
    const bool lv3 = argc > 1 && argv[1][0] == '3';
    cudaSetDevice(0); cudaFree(0);
    FILE *f = fopen("probe_patched.cubin", "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    std::vector<char> img(sz); if (fread(img.data(), 1, sz, f) != (size_t)sz) return 1; fclose(f);
    CUmodule m; CK(cuModuleLoadData(&m, img.data()));
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    HwQueueHandle hwq = 0; XQueueHandle xq = 0;
    if (lv3 && (CudaQueueCreate(&hwq, s) != kXSchedSuccess || XQueueCreate(&xq, hwq, kPreemptLevelInterrupt, kQueueCreateFlagNone) != kXSchedSuccess)) { printf("xq fail\n"); return 1; }
    unsigned *out, *src; cudaMalloc(&out, 8); cudaMalloc(&src, 4096); cudaMemset(src, 1, 4096);
    int Ns[] = {8, 12, 20, 40, 64, 120, 200};
    printf("%-6s %-9s %-11s %-14s %-12s %-15s %s\n", "N", "REG(attr)", "SR_REGALLOC", "cascade_top", "frame_regs", "engine_reg_ceil", "verdict");
    for (int N : Ns) {
        char name[64]; snprintf(name, sizeof(name), "_Z5probeILi%dEEvPjPKj", N);
        CUfunction fn; CK(cuModuleGetFunction(&fn, m, name));
        int regs = 0; cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
        void *args[] = {&out, &src};
        CK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, (CUstream)s, args, nullptr));
        cudaStreamSynchronize(s);
        unsigned h[2]; cudaMemcpy(h, out, 8, cudaMemcpyDeviceToHost);
        int r4 = (int)h[0] - 3;                               // engine: R4 = SR_REGALLOC - 3
        int top = r4 < 14 ? -1 : (r4 >= 252 ? 252 : ((r4 & ~1) + 1));   // highest register the cascade saves
        int ceil16 = ((regs + 16 + 15) / 16) * 16; if (ceil16 > 256) ceil16 = 256;   // PrepareLaunch reg_ceil
        int stride = ((0x78 + (ceil16 - 14) * 4) + 15) / 16 * 16;
        int frame_max_reg = 14 + (stride - 0x78) / 4 - 1;  // last register that fits in the thread frame
        const char *v = (top < regs - 1) ? "MISSES TOP REGS" : (top > frame_max_reg ? "OVERFLOWS FRAME" : "ok");
        printf("%-6d %-9d %-11u R%-13d R%-11d %-15d %s\n", N, regs, h[0], top, frame_max_reg, ceil16, v);
    }
    if (xq) { XQueueDestroy(xq); HwQueueDestroy(hwq); }
    return 0;
}
