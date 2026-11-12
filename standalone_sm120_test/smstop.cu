// Experiment: single-SM STOP_TRIGGER / RUN_TRIGGER via direct BAR0 MMIO.
#include "bar0.h"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <unistd.h>
#define NB 30
__device__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__global__ void work(volatile unsigned long long *ctr, volatile unsigned *sm, long long cycles) {
    extern __shared__ char pad[];
    if (threadIdx.x == 0) sm[blockIdx.x] = smid();
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {
        if (threadIdx.x == 0) ctr[blockIdx.x] += 1;
        pad[threadIdx.x] = 1;
    }
}
static double now_us() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
int main(int argc, char **argv) {
    uint32_t ctl = argc > 1 ? strtoul(argv[1], 0, 0) : 0x504704;   // DBGR_CONTROL0 of target SM
    uint32_t st = ctl - 4;                                           // DBGR_STATUS0
    int do_run = argc > 2 ? atoi(argv[2]) : 1;
    if (bar0_open()) return 1;
    unsigned long long *h, *d; unsigned *hs, *ds;
    cudaHostAlloc(&h, NB * 8, cudaHostAllocMapped); memset(h, 0, NB * 8); cudaHostGetDevicePointer(&d, h, 0);
    cudaHostAlloc(&hs, NB * 4, cudaHostAllocMapped); cudaHostGetDevicePointer(&ds, hs, 0);
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    cudaFuncSetAttribute(work, cudaFuncAttributeMaxDynamicSharedMemorySize, 80 * 1024);
    work<<<NB, 128, 80 * 1024>>>(d, ds, (long long)(2.0 * clk * 1000.0));  // ~2 s
    usleep(300000);
    printf("before: ctl=%08x st=%08x wvm=%08x\n", rd32(ctl), rd32(st), rd32(ctl + 4));
    unsigned long long s0[NB]; memcpy(s0, h, sizeof s0);
    uint32_t c = rd32(ctl);
    double t0 = now_us();
    wr32(ctl, c | 0x80000000u);                 // STOP_TRIGGER
    uint32_t s; int n = 0;
    do { s = rd32(st); n++; } while (!(s & 0x10) && now_us() - t0 < 100000);
    double t1 = now_us();
    { double tl = now_us(); uint32_t s2; uint32_t g;
      do { s2 = rd32(st); g = rd32(ctl + 0x30); } while ((s2 & 0x10) && now_us() - tl < 2000000);
      printf("lock window: %.1f us, then st=%08x ctl=%08x gesr=%08x\n", now_us() - tl, s2, rd32(ctl), g); }
    printf("stop: locked_down=%d after %.1f us (%d polls) st=%08x ctl=%08x\n", !!(s & 0x10), t1 - t0, n, s, rd32(ctl));
    printf("warp_esr=%08x global_esr=%08x esr_pc=%08x\n", rd32(ctl + 0x2c), rd32(ctl + 0x30), rd32(ctl + 0x34));
    unsigned long long s1[NB]; memcpy(s1, h, sizeof s1);
    usleep(100000);
    printf("delta over 100ms while stopped (block:smid:delta):\n");
    for (int i = 0; i < NB; i++) printf("%d:%u:%lld ", i, hs[i], (long long)(h[i] - s1[i])); printf("\n");
    printf("after 100ms: st=%08x ctl=%08x gesr=%08x wvm=%08x\n", rd32(st), rd32(ctl), rd32(ctl + 0x30), rd32(ctl + 4));
    if (do_run) {
        c = rd32(ctl);
        double t2 = now_us();
        wr32(ctl, (c & ~0x80000000u) | 0x40000000u);   // RUN_TRIGGER
        do { s = rd32(st); } while ((s & 0x10) && now_us() - t2 < 100000);
        printf("run: locked_down=%d after %.1f us st=%08x ctl=%08x\n", !!(s & 0x10), now_us() - t2, s, rd32(ctl));
    }
    cudaError_t e = cudaDeviceSynchronize();
    printf("kernel result: %s\n", cudaGetErrorString(e));
    for (int i = 0; i < NB; i++) printf("%d:%u:%llu ", i, hs[i], h[i]); printf("\n");
    return 0;
}
