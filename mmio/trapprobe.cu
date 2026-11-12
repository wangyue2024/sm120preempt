// Experiment: which trap reason does an MMIO SM STOP_TRIGGER produce, and how
// fast do warps enter the trap handler?
#include <cuda.h>
#include <cuxtra/cuxtra.h>
#include "bar0.h"
#include "livepatch.h"
#include <unistd.h>
#include <ctime>
#define NB 30
__device__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__global__ void work(volatile unsigned long long *ctr, volatile unsigned *sm, long long cycles) {
    if (threadIdx.x == 0) sm[blockIdx.x] = smid();
    long long t0 = clock64();
    while (clock64() - t0 < cycles) if (threadIdx.x == 0) ctr[blockIdx.x] += 1;
}
__global__ void nop_k() {}
static uint64_t ptimer() {
    uint32_t hi, lo, hi2;
    do { hi = rd32(0x9410); lo = rd32(0x9400); hi2 = rd32(0x9410); } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
}
struct Rec { unsigned long long t; unsigned reason, pad; };
int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "unicast";
    if (bar0_open()) return 1;
    cudaFree(0); nop_k<<<1, 1>>>(); cudaDeviceSynchronize();
    LivePatch P; if (!P.probe()) return 1;
    Rec *rec; cudaHostAlloc(&rec, 256 * sizeof(Rec), cudaHostAllocMapped); memset(rec, 0, 256 * sizeof(Rec));
    uint64_t rva = (uint64_t)(uintptr_t)rec;
    using namespace s120;
    std::vector<I> cave = {
        mov_imm(3, 0),                        // displaced: MOV R3, RZ (imm 0)
        s2r(4, 0x43, 0),                       // S2R R4, SR_VIRTUALSMID
        mov_imm(8, (uint32_t)rva), mov_imm(9, (uint32_t)(rva >> 32)),
        cs2r_globaltimer(6),
        imad_wide_u32_imm(8, 4, sizeof(Rec), 8),
        st_e(8, 0, 6, true), st_e(8, 8, 2),
        iadd_imm(12, 12, 0x10), ret_r12(),
    };
    disasm(cave, "/tmp/claude-1000/cave.bin");
    if (!P.install(cave)) return 1;
    unsigned long long *h; unsigned *hs; cudaHostAlloc(&h, NB * 8, cudaHostAllocMapped); cudaHostAlloc(&hs, NB * 4, cudaHostAllocMapped);
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    work<<<NB, 128>>>(h, hs, (long long)(1.5 * clk * 1000.0));
    usleep(300000);
    printf("ptimer=%llu (check vs globaltimer)\n", (unsigned long long)ptimer());
    uint32_t reg = strcmp(mode, "bcast") ? 0x504704 : 0x419e84;
    uint64_t t0 = ptimer();
    wr32(reg, 0x80001c01);
    uint64_t t1 = ptimer();
    usleep(50000);
    printf("stop write took %llu ns; status0(sm0)=%08x ctl=%08x\n", (unsigned long long)(t1 - t0), rd32(0x504700), rd32(0x504704));
    int n = 0; uint64_t first = ~0ull, last = 0;
    for (int i = 0; i < 256; i++) if (rec[i].t) {
        n++; uint64_t d = rec[i].t - t0; if (d < first) first = d; if (d > last) last = d;
        printf("  smid %3d reason=%#x trap_at=+%.1f us\n", i, rec[i].reason, (rec[i].t - (double)t0) / 1000.0);
    }
    printf("SMs trapped: %d  first=+%.1f us last=+%.1f us\n", n, first / 1000.0, last / 1000.0);
    cudaError_t e = cudaDeviceSynchronize();
    printf("kernel result: %s\n", cudaGetErrorString(e));
    P.restore();
    nop_k<<<1, 1>>>(); printf("post: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    return 0;
}
