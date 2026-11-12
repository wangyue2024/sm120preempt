#include <cuda.h>
#include <cuxtra/cuxtra.h>
#include "bar0.h"
#include "arena.h"
#include "sass120.h"
#include <unistd.h>
#include <cstdio>
#include <cstring>

#define NB 30

__device__ unsigned get_smid() {
    unsigned r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}

__device__ unsigned get_vsmid() {
    return get_smid();
}

__device__ unsigned long long get_globaltimer() {
    unsigned long long r;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r));
    return r;
}

__global__ void active_kernel(volatile unsigned long long *ctr,
                              volatile unsigned *sm,
                              volatile unsigned *vsm,
                              volatile unsigned long long *gpu_now,
                              volatile int *stop_flag) {
    unsigned my_sm = get_smid();
    if (threadIdx.x == 0) {
        sm[blockIdx.x] = my_sm;
        vsm[blockIdx.x] = get_vsmid();
    }
    while (!(*stop_flag)) {
        if (threadIdx.x == 0) {
            ctr[blockIdx.x] += 1;
            if (blockIdx.x == 0) *gpu_now = get_globaltimer();
        }
    }
}

__global__ void warmup_k() {}

static uint64_t read_ptimer() {
    uint32_t hi, lo, hi2;
    do {
        hi = rd32(0x9410);
        lo = rd32(0x9400);
        hi2 = rd32(0x9410);
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
}

struct TrapRecord {
    unsigned long long t_entry;
    unsigned reason;
    unsigned pad;
};

// All 18 TPC base addresses across GPC0, GPC1, GPC2
static const uint32_t kTpcBases[18] = {
    0x504000, 0x505000, 0x506000, 0x507000, 0x508000, 0x509000,
    0x514000, 0x515000, 0x516000, 0x517000, 0x518000, 0x519000,
    0x524000, 0x525000, 0x526000, 0x527000, 0x528000, 0x529000
};

int main(int argc, char **argv) {
    // mode 0: unicast 0x504704 (1 SM)
    // mode 1: unicast 0x504684 (1 TPC = 2 SMs via SM_SFE)
    // mode 2: unicast all active TPCs +0x684 (all 30 SMs via SM_SFE)
    // mode 3: broadcast GPCS_TPC0..5 +0x684 (0x424684..0x429684, all 30 SMs in 6 writes!)
    int trig_mode = (argc > 1) ? atoi(argv[1]) : 0;

    if (bar0_open()) return 1;
    cudaFree(0);
    warmup_k<<<1, 1>>>();
    cudaDeviceSynchronize();

    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    sm120arena::Info info = sm120arena::probe(ctx, 0);
    if (!info.ok || !info.handler_is_memobj) {
        printf("Arena info unavailable\n");
        return 1;
    }

    // Extract LOP3.LUT R2, R2, imm, RZ, 0xc0, !PT from handler + 0x220
    uint64_t lop_w0, lop_w1;
    memcpy(&lop_w0, &info.handler[0x220], 8);
    memcpy(&lop_w1, &info.handler[0x228], 8);
    // Replace immediate (bits 63:32 of w0) with 0x10 so R2 = R2 & 0x10
    // (clears 0x1, 0x8, 0x20 debugger pause bits on Pass 1 while preserving 0x10 context switch!)
    s120::I clear_bit8 = {(lop_w0 & 0xffffffffULL) | (0x10ULL << 32), lop_w1};

    uint64_t ext_va = info.tools_pc + 0xa00;
    uint64_t orig_390[2];
    memcpy(orig_390, &info.handler[0x390], 16);

    TrapRecord *rec;
    cudaHostAlloc(&rec, 256 * sizeof(TrapRecord), cudaHostAllocMapped);
    memset(rec, 0, 256 * sizeof(TrapRecord));
    uint64_t rva = (uint64_t)(uintptr_t)rec;

    using namespace s120;
    std::vector<I> ext_code = {
        mov_imm(3, 0),
        s2r(4, 0x43, 0),                       // S2R R4, SR_VIRTUALSMID
        mov_imm(8, (uint32_t)rva),
        mov_imm(9, (uint32_t)(rva >> 32)),
        cs2r_globaltimer(6),
        imad_wide_u32_imm(8, 4, sizeof(TrapRecord), 8),
        st_e(8, 0, 6, true),
        st_e(8, 8, 2),
        clear_bit8,                            // Clear bit 0x8 (STOP_TRIGGER) so do_pause is skipped!
        iadd_imm(12, 12, 0x10),
        ret_r12(),
    };

    auto flat_words = flat(ext_code);
    std::vector<uint64_t> ext_backup(flat_words.size());
    cuXtraMemcpyDtoH(ext_backup.data(), ext_va, flat_words.size() * 8, 0);

    uint64_t o0, o1;
    memcpy(&o0, &info.handler[0x380], 8);
    memcpy(&o1, &info.handler[0x388], 8);
    uint64_t enc = (ext_va - 0x100) >> 8, branch_inst[2];
    branch_inst[0] = ((enc & 0xffffffffULL) << 32) | (o0 & 0xffffffffULL);
    branch_inst[1] = (o1 & ~0xffULL) | ((enc >> 32) & 0xffULL);

    cuXtraMemcpyHtoD(ext_va, flat_words.data(), flat_words.size() * 8, 0);
    cuXtraMemcpyHtoD(info.handler_va + 0x390, branch_inst, 16, 0);
    cuXtraInvalInstrCache(ctx);

    // Find active TPCs (where +0x684 reads 0x1c00 or 0x1c01)
    std::vector<uint32_t> active_tpcs;
    for (int i = 0; i < 18; i++) {
        uint32_t v = rd32(kTpcBases[i] + 0x684);
        if ((v & 0xffff) == 0x1c00 || (v & 0xffff) == 0x1c01) {
            active_tpcs.push_back(kTpcBases[i]);
        }
    }
    printf("Active TPCs found in GPC0..2: %zu\n", active_tpcs.size());

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_blocks = (prop.multiProcessorCount > 0) ? prop.multiProcessorCount : NB;
    if (num_blocks > 256) num_blocks = 256;
    printf("Target GPU: %s (%d SMs) -> Launching %d blocks\n", prop.name, prop.multiProcessorCount, num_blocks);

    unsigned long long *ctr, *gpu_now;
    unsigned *sm, *vsm;
    int *stop_flag;
    cudaHostAlloc(&ctr, num_blocks * 8, cudaHostAllocMapped);
    cudaHostAlloc(&sm, num_blocks * 4, cudaHostAllocMapped);
    cudaHostAlloc(&vsm, num_blocks * 4, cudaHostAllocMapped);
    cudaHostAlloc(&gpu_now, 8, cudaHostAllocMapped);
    cudaHostAlloc(&stop_flag, 4, cudaHostAllocMapped);
    memset(ctr, 0, num_blocks * 8);
    *gpu_now = 0;
    *stop_flag = 0;

    active_kernel<<<num_blocks, 128>>>(ctr, sm, vsm, gpu_now, stop_flag);
    while (*gpu_now == 0) usleep(10);
    usleep(20000);
    memset(rec, 0, 256 * sizeof(TrapRecord));

    uint64_t pt0 = read_ptimer();
    if (trig_mode == 0) {
        wr32(0x504704, 0x80001c01);
        wr32(0x504704, 0x00001c01);
    } else if (trig_mode == 1) {
        wr32(0x504684, 0x80001c01);
        wr32(0x504684, 0x00001c01);
    } else if (trig_mode == 2) {
        for (uint32_t base : active_tpcs) wr32(base + 0x684, 0x80001c01);
        for (uint32_t base : active_tpcs) wr32(base + 0x684, 0x00001c01);
    } else if (trig_mode == 3) {
        for (int t = 0; t < 6; t++) wr32(0x424684 + t * 0x1000, 0x80001c01);
        for (int t = 0; t < 6; t++) wr32(0x424684 + t * 0x1000, 0x00001c01);
    } else if (trig_mode == 4) {
        for (int t = 0; t < 3; t++) wr32(0x504684 + t * 0x1000, 0x80001c01);
        for (int t = 0; t < 3; t++) wr32(0x504684 + t * 0x1000, 0x00001c01);
    } else if (trig_mode == 5) {
        for (int t = 0; t < 3; t++) wr32(0x424684 + t * 0x1000, 0x80001c01);
        for (int t = 0; t < 3; t++) wr32(0x424684 + t * 0x1000, 0x00001c01);
    } else if (trig_mode == 6) {
        wr32(0x424684, 0x80001c01);
        wr32(0x424684, 0x00001c01);
    }
    uint64_t pt1 = read_ptimer();

    usleep(300);
    uint32_t st_final = rd32(0x504700);
    uint32_t ctl_final = rd32(0x504704);

    int trapped_count = 0;
    double min_us = 1e9, max_us = 0;
    for (int i = 0; i < 256; i++) {
        if (rec[i].t_entry) {
            trapped_count++;
            double d_us = (double)((long long)(rec[i].t_entry - pt0)) / 1000.0;
            if (d_us < min_us) min_us = d_us;
            if (d_us > max_us) max_us = d_us;
            printf("  vsmid %2d: reason=%#x latency=+%.2f us\n", i, rec[i].reason, d_us);
        }
    }
    printf("[trig_mode=%d] MMIO write took %.2f us | Trapped SMs: %d/%d | min_lat=+%.2f us max_lat=+%.2f us | st_final=%08x ctl_final=%08x\n",
           trig_mode, (pt1 - pt0) / 1000.0, trapped_count, num_blocks,
           trapped_count ? min_us : 0.0, trapped_count ? max_us : 0.0,
           st_final, ctl_final);

    *stop_flag = 1;
    cudaError_t err = cudaDeviceSynchronize();
    printf("Kernel completion status: %s\n", cudaGetErrorString(err));

    cuXtraMemcpyHtoD(info.handler_va + 0x390, orig_390, 16, 0);
    cuXtraMemcpyHtoD(ext_va, ext_backup.data(), flat_words.size() * 8, 0);
    cuXtraInvalInstrCache(ctx);
    return 0;
}
