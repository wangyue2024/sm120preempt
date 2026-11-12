#include <cuda.h>
#include <cuxtra/cuxtra.h>
#include "bar0.h"
#include "arena.h"
#include "sass120.h"
#include <unistd.h>
#include <cstdio>
#include <cstring>

__device__ unsigned long long get_globaltimer() {
    unsigned long long r;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r));
    return r;
}

// Non-idempotent stateful kernel:
// Every thread updates a stateful recurrence and performs global atomicAdds.
// If any warp/CTA is aborted, replayed, or has any register clobbered, the
// final atomic counter and per-thread state checksums will mismatch!
__global__ void non_idempotent_kernel(volatile unsigned long long *progress,
                                      volatile unsigned long long *t_last,
                                      unsigned long long *atomic_accum,
                                      unsigned int *out_checksums,
                                      unsigned int total_iters) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int state = tid * 0x9e3779b9u + 0x12345678u;

    for (unsigned int it = 1; it <= total_iters; it++) {
        // Stateful non-idempotent recurrence using multiple registers & predicates
        state = state * 1664525u + 1013904223u + (it ^ tid);
        // Intentional small delay loop to make total runtime ~80 ms
        unsigned long long t0 = get_globaltimer();
        while (get_globaltimer() - t0 < 300ULL) {
            state ^= 0x55aa55aau;
            state ^= 0x55aa55aau;
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) *t_last = get_globaltimer();
        if ((it & 0xffu) == 0) {
            atomicAdd((unsigned long long *)atomic_accum, 1ULL);
            if (threadIdx.x == 0) {
                progress[blockIdx.x] = it;
            }
        }
    }
    out_checksums[tid] = state;
    if (threadIdx.x == 0) {
        progress[blockIdx.x] = total_iters | (1ULL << 62);
        if (blockIdx.x == 0) *t_last = get_globaltimer();
    }
}

__global__ void non_idempotent_kernel_2(volatile unsigned long long *progress,
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
            if (threadIdx.x == 0) {
                progress[blockIdx.x] = it;
            }
        }
    }
    out_checksums[tid] = state;
    if (threadIdx.x == 0) {
        progress[blockIdx.x] = total_iters | (1ULL << 62);
        if (blockIdx.x == 0) *t_last = get_globaltimer();
    }
}

__global__ void low_worker(volatile unsigned long long *progress,
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

__global__ void high_worker(volatile unsigned long long *progress,
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

static uint64_t read_ptimer() {
    uint32_t hi, lo, hi2;
    do {
        hi = rd32(0x9410);
        lo = rd32(0x9400);
        hi2 = rd32(0x9410);
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
}

namespace s120_ext {
using namespace s120;

// LDC Rd, c[0x0][off] with wb=0 (&wr=0x0)
inline I ldc_u32(int rd, uint32_t off) {
    uint64_t encoded_off = (uint64_t)(off >> 2) << 8;
    return {(encoded_off << 32) | 0xff000000ULL | ((uint64_t)rd << 16) | 0x7b82ULL,
            0x000e220000000800ULL};
}

// LD.E.STRONG.GPU R0, [R4] / [R6]
inline I ld_e_strong_gpu_r0_r4() {
    return {0x0000000004007980ULL, 0x001e24000010e900ULL};
}
inline I ld_e_strong_gpu_r0_r6() {
    return {0x0000000006007980ULL, 0x001e24000010e900ULL};
}
inline I ldl_32_r0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0x00000000ff007983ULL, 0x0032240000100800ULL};
}
inline I ldl_64_r0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0x00000000ff007983ULL, 0x0032240000100a00ULL};
}
inline I stl_64_r0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0x00000000ff007387ULL, 0x0033e40000100a00ULL};
}
inline I stl_64_r0_p0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0x00000000ff000387ULL, 0x0033e40000100a00ULL};
}
inline I mov_imm_w01(int rd, uint32_t imm) {
    return s120::mov_imm(rd, imm, 0x003fda0000000000ULL);
}

// ISETP.NE.AND P0, PT, Ra, RZ, PT with req={0} and WAIT13_END_GROUP
inline I isetp_ne_rz(int ra) {
    return {0x000000ff0000720cULL | ((uint64_t)ra << 24),
            0x001fda0003f05270ULL};
}

// ISETP.NE.AND P0, PT, Ra, imm, PT with req={0} and WAIT13_END_GROUP
inline I isetp_ne_imm(int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL,
            0x001fda0003f05270ULL};
}
inline I isetp_ne_imm_and_p0(int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL,
            0x001fda0000705270ULL};
}
inline I isetp_eq_imm(int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL,
            0x001fda0003f02270ULL};
}
inline I isetp_eq_imm_or_p0(int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL,
            0x001fda0000702470ULL};
}

// NANOSLEEP imm_ns
inline I nanosleep_imm(uint32_t imm_ns) {
    return {((uint64_t)imm_ns << 32) | 0x795dULL,
            0x000fea0003800000ULL};
}

inline uint64_t encode_bra_w0(int n_instrs, uint32_t low16) {
    int64_t off = (int64_t)n_instrs * 4;
    uint64_t lo8 = (uint64_t)(off & 0xff);
    uint64_t hi = (uint64_t)(off >> 8);
    return (hi << 34) | (lo8 << 16) | low16;
}

// @P0 BRA +n_instrs (relative to next instruction)
inline I bra_p0(int n_instrs) {
    return {encode_bra_w0(n_instrs, 0x0947),
            (n_instrs < 0) ? 0x000fea000383ffffULL : 0x000fea0003800000ULL};
}

// @!P0 BRA +n_instrs (relative to next instruction)
inline I bra_not_p0(int n_instrs) {
    return {encode_bra_w0(n_instrs, 0x8947),
            (n_instrs < 0) ? 0x000fea000383ffffULL : 0x000fea0003800000ULL};
}

// Unconditional BRA +n_instrs (relative to next instruction)
inline I bra_rel(int n_instrs) {
    return {encode_bra_w0(n_instrs, 0x7947),
            (n_instrs < 0) ? 0x000fea000383ffffULL : 0x000fea0003800000ULL};
}
} // namespace s120_ext

int main() {
    if (bar0_open()) return 1;
    cudaFree(0);
    warmup_k<<<1, 1>>>();
    cudaDeviceSynchronize();

    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    sm120arena::Info info = sm120arena::probe(ctx, 0);
    if (!info.ok || !info.handler_is_memobj) return 1;

    // Extract LOP3.LUT P0, RZ, R2, 0x11, RZ, 0xc0, !PT from handler + 0x3b0
    s120::I test_r2_ctxsw;
    memcpy(&test_r2_ctxsw.w0, &info.handler[0x3b0], 8);
    memcpy(&test_r2_ctxsw.w1, &info.handler[0x3b8], 8);
    test_r2_ctxsw.w0 = (test_r2_ctxsw.w0 & 0xffffffffULL) | (0x11ULL << 32);
    test_r2_ctxsw.w1 = (test_r2_ctxsw.w1 & ~0x000ffe0000000000ULL) | 0x000fda0000000000ULL;

    // Extract LOP3.LUT R2, R2, ~0x8, RZ, 0xc0, !PT from handler + 0x220
    uint64_t lop_w0, lop_w1;
    memcpy(&lop_w0, &info.handler[0x220], 8);
    memcpy(&lop_w1, &info.handler[0x228], 8);
    s120::I clear_r2_bit8 = {(lop_w0 & 0xffffffffULL) | (0xfffffff7ULL << 32), lop_w1};

    // Extract STL.64 [0xffffa8], R0 from handler + 0x110
    s120::I stl_64_rpc;
    memcpy(&stl_64_rpc.w0, &info.handler[0x110], 8);
    memcpy(&stl_64_rpc.w1, &info.handler[0x118], 8);

    uint64_t ext_va = info.tools_pc + 0xa00;
    uint64_t exit_stub_va = info.tools_pc + 0xe00; // Normal-mode EXIT stub (past end of ext_code)
    uint64_t pause_tramp_va = info.tools_pc + 0xe20; // Normal-mode 2-instr Pause Trampoline
    uint32_t tramp0_lo = (uint32_t)pause_tramp_va;
    uint32_t tramp1_lo = (uint32_t)(pause_tramp_va + 0x10);
    uint32_t tramp_hi  = (uint32_t)(pause_tramp_va >> 32);

    using namespace s120;
    using namespace s120_ext;
    // Unified 3-Granularity 52-Instruction Divergence-Free Normal-Mode Trampoline Trap Routine:
    std::vector<I> ext_code = {
        ldc_u32(4, 0x170),                                              // 0: R4 = preempt_buf lo
        ldc_u32(5, 0x174),                                              // 1: R5 = preempt_buf hi
        isetp_ne_rz(4),                                                 // 2: P0 = (R4 != 0)
        bra_not_p0(45),                                                 // 3: @!P0 BRA done (idx 49)
        test_r2_ctxsw,                                                  // 4: (R2 & 0x11) != 0?
        bra_p0(43),                                                     // 5: @P0 BRA done (idx 49)
        clear_r2_bit8,                                                  // 6: R2 &= ~0x8
        // Compute R6:R7 = &preempt_buf[4 + 2 * block_idx]
        {0x0000000000007919ULL, 0x000e620000002600ULL},                 // 7: S2R R0, SR_CTAID.Y
        {0x0000dd00ff037b82ULL, 0x000e620000000800ULL},                 // 8: LDC R3, c[0x0][0x374]
        {0x0000000000067919ULL, 0x001e620000002500ULL},                 // 9: S2R R6, SR_CTAID.X
        {0x0000000000087919ULL, 0x000e2c0000002700ULL},                 // 10: S2R R8, SR_CTAID.Z
        {0x0000dc00ff077b82ULL, 0x000e220000000800ULL},                 // 11: LDC R7, c[0x0][0x370] (gridDim.x)
        {0x0000000803007224ULL, 0x003fda00078e0200ULL},                 // 12: IMAD R0, R3, R8, R0 (R0 = gridDim.y * blockIdx.z + blockIdx.y)
        {0x0000000700007224ULL, 0x001fda00078e0206ULL},                 // 13: IMAD R0, R0, R7, R6 (R0 = R0 * gridDim.x + blockIdx.x)
        {0x0000000400037810ULL, 0x000fda0007ffe000ULL},                 // 14: IADD3 R3, PT, PT, R0, 0x4, R0
        {0x0000000403067825ULL, 0x000fda00078e0004ULL},                 // 15: IMAD.WIDE.U32 R6, R3, 0x4, R4
        ld_e_strong_gpu_r0_r4(),                                        // 16: LD.E.STRONG.GPU R0, [R4]
        isetp_ne_imm(0, 3),                                             // 17: P0 = (R0 != 3)
        bra_p0(1),                                                      // 18: @P0 BRA check_pause (idx 20)
        ld_e_strong_gpu_r0_r6(),                                        // 19: LD.E.STRONG.GPU R0, [R6] (per-CTA command!)
        // idx 20: check_pause
        isetp_ne_imm(0, 2),                                             // 20: P0 = (R0 != 2)
        bra_p0(10),                                                     // 21: @P0 BRA check_exit (idx 32)
        // idx 22..31: do_pause (100% divergence-free predicated redirect to pause_tramp_va!)
        ldl_32_r0(0xffff90),                                            // 22: LDL R0, [0xffff90] (THREAD_STATE_ENUM.0)
        isetp_ne_rz(0),                                                 // 23: P0 = (R0 != 0)
        ldl_64_r0(0xffffa8),                                            // 24: LDL.64 R0, [0xffffa8]
        isetp_ne_imm_and_p0(0, tramp0_lo),                              // 25: P0 = P0 && (R0 != tramp0_lo)
        isetp_ne_imm_and_p0(0, tramp1_lo),                              // 26: P0 = P0 && (R0 != tramp1_lo)
        stl_64_r0_p0(0xffff88),                                         // 27: @P0 STL.64 [0xffff88], R0 (save PC_orig)
        mov_imm_w01(0, tramp0_lo),                                      // 28: MOV R0, tramp0_lo
        mov_imm_w01(1, tramp_hi),                                       // 29: MOV R1, tramp_hi
        stl_64_r0_p0(0xffffa8),                                         // 30: @P0 STL.64 [0xffffa8], R0
        bra_rel(16),                                                    // 31: BRA done_r1 (idx 48)
        // idx 32..42: check_exit
        isetp_ne_imm(0, 1),                                             // 32: P0 = (R0 != 1)
        bra_p0(9),                                                      // 33: @P0 BRA check_resume (idx 43)
        mov_imm_w01(0, 1),                                              // 34: MOV R0, 1
        {0x0000000406007385ULL, 0x000fda0000100900ULL},                 // 35: ST.E [R6+0x4], R0 (restore_flag_of_block = 1)
        ldc_u32(8, 0x180),                                              // 36: LDC R8, c[0x0][0x180] (kernel_idx lo)
        ldc_u32(9, 0x184),                                              // 37: LDC R9, c[0x0][0x184] (kernel_idx hi)
        {0x0000000804007385ULL, 0x001fda0000100b08ULL},                 // 38: ST.E.64 [R4+0x8], R8 (preempt_idx = kernel_idx)
        mov_imm_w01(0, (uint32_t)exit_stub_va),                         // 39: MOV R0, lo32(exit_stub_va)
        mov_imm_w01(1, (uint32_t)(exit_stub_va >> 32)),                 // 40: MOV R1, hi32(exit_stub_va)
        stl_64_r0(0xffffa8),                                            // 41: STL.64 [0xffffa8], R0
        bra_rel(5),                                                     // 42: BRA done_r1 (idx 48)
        // idx 43..47: check_resume (R0 == 0, 100% divergence-free predicated restore!)
        ldl_64_r0(0xffffa8),                                            // 43: LDL.64 R0, [0xffffa8]
        isetp_eq_imm(0, tramp0_lo),                                     // 44: P0 = (R0 == tramp0_lo)
        isetp_eq_imm_or_p0(0, tramp1_lo),                               // 45: P0 = P0 || (R0 == tramp1_lo)
        ldl_64_r0(0xffff88),                                            // 46: LDL.64 R0, [0xffff88]
        stl_64_r0_p0(0xffffa8),                                         // 47: @P0 STL.64 [0xffffa8], R0
        // idx 48..51: done_r1 / done
        mov_imm_w01(1, 0xffff80),                                       // 48: MOV R1, 0xffff80
        mov_imm_w01(3, 0),                                              // 49: MOV R3, 0
        iadd_imm(12, 12, 0x10),                                         // 50: IADD R12, R12, 0x10
        ret_r12(),                                                      // 51: RET.ABS R12 0x20
    };

    std::vector<I> exit_stub = { exit_(), exit_() };
    std::vector<I> pause_tramp = { nop(), bra_rel(-2) };
    auto flat_ext = flat(ext_code);
    auto flat_exit = flat(exit_stub);
    auto flat_tramp = flat(pause_tramp);

    uint64_t orig_390[2];
    memcpy(orig_390, &info.handler[0x390], 16);
    std::vector<uint64_t> ext_backup(flat_ext.size());
    std::vector<uint64_t> exit_backup(flat_exit.size());
    std::vector<uint64_t> tramp_backup(flat_tramp.size());
    cuXtraMemcpyDtoH(ext_backup.data(), ext_va, flat_ext.size() * 8, 0);
    cuXtraMemcpyDtoH(exit_backup.data(), exit_stub_va, flat_exit.size() * 8, 0);
    cuXtraMemcpyDtoH(tramp_backup.data(), pause_tramp_va, flat_tramp.size() * 8, 0);

    uint64_t o0, o1;
    memcpy(&o0, &info.handler[0x380], 8);
    memcpy(&o1, &info.handler[0x388], 8);
    uint64_t enc = (ext_va - 0x100) >> 8, branch_inst[2];
    branch_inst[0] = ((enc & 0xffffffffULL) << 32) | (o0 & 0xffffffffULL);
    branch_inst[1] = (o1 & ~0xffULL) | ((enc >> 32) & 0xffULL);

    cuXtraMemcpyHtoD(exit_stub_va, flat_exit.data(), flat_exit.size() * 8, 0);
    cuXtraMemcpyHtoD(pause_tramp_va, flat_tramp.data(), flat_tramp.size() * 8, 0);
    cuXtraMemcpyHtoD(ext_va, flat_ext.data(), flat_ext.size() * 8, 0);
    cuXtraMemcpyHtoD(info.handler_va + 0x390, branch_inst, 16, 0);
    cuXtraInvalInstrCache(ctx);

    CUstream stream_low, stream_high;
    cuStreamCreate(&stream_low, CU_STREAM_NON_BLOCKING);
    cuStreamCreate(&stream_high, CU_STREAM_NON_BLOCKING);

    volatile uint32_t *preempt_buf_low, *preempt_buf_high;
    volatile unsigned long long *prog_low, *prog_high, *t_last_low, *t_last_high;
    volatile unsigned long long *atomic_accum;
    unsigned int *out_checksums;
    cudaHostAlloc((void **)&preempt_buf_low, 256, cudaHostAllocMapped);
    cudaHostAlloc((void **)&preempt_buf_high, 256, cudaHostAllocMapped);
    cudaHostAlloc((void **)&prog_low, 15 * 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&prog_high, 15 * 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&t_last_low, 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&t_last_high, 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&atomic_accum, 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&out_checksums, 15 * 128 * sizeof(unsigned int), cudaHostAllocMapped);

    struct {
        uint64_t preempt_buf;
        uint64_t guardian;
        int64_t kernel_idx;
        uint32_t sched_enabled;
    } __attribute__((packed)) dbg_params;

    // =========================================================================
    // TEST 1: Mode 2 — Fine-Grained Non-Abort Resumable Preemption (Pause & Resume)
    // =========================================================================
    printf("=== TEST 1: Mode 2 (Non-Abort Resumable Stream Pause & Resume, Non-Idempotent) ===\n");
    memset((void *)preempt_buf_low, 0, 256);
    memset((void *)preempt_buf_high, 0, 256);
    memset((void *)prog_low, 0, 15 * 8);
    memset((void *)prog_high, 0, 15 * 8);
    *t_last_low = 0;
    *t_last_high = 0;
    *atomic_accum = 0;
    memset(out_checksums, 0, 15 * 128 * sizeof(unsigned int));

    CUfunction func_non_idemp = nullptr, func_non_idemp_2 = nullptr, func_high = nullptr, func_low = nullptr;
    cudaGetFuncBySymbol(&func_non_idemp, (const void *)non_idempotent_kernel);
    cudaGetFuncBySymbol(&func_non_idemp_2, (const void *)non_idempotent_kernel_2);
    cudaGetFuncBySymbol(&func_high, (const void *)high_worker);
    cudaGetFuncBySymbol(&func_low, (const void *)low_worker);

    dbg_params.preempt_buf = (uint64_t)(uintptr_t)preempt_buf_low;
    dbg_params.guardian = 0;
    dbg_params.kernel_idx = 10;
    dbg_params.sched_enabled = 1;
    cuXtraSetDebuggerParams(func_non_idemp, &dbg_params, sizeof(dbg_params));

    dbg_params.preempt_buf = (uint64_t)(uintptr_t)preempt_buf_high;
    dbg_params.guardian = 0;
    dbg_params.kernel_idx = 11;
    dbg_params.sched_enabled = 1;
    cuXtraSetDebuggerParams(func_high, &dbg_params, sizeof(dbg_params));

    unsigned int total_iters = 204800; // 800 * 256 -> ~75 ms runtime
    void *args_non_idemp[] = { &prog_low, &t_last_low, &atomic_accum, &out_checksums, &total_iters };
    cuLaunchKernel(func_non_idemp, 15, 1, 1, 128, 1, 1, 0, stream_low, args_non_idemp, nullptr);

    unsigned long long dur_high = 40000000ULL; // 40 ms high-priority task in SAME CUcontext
    void *args_high[] = { &prog_high, &t_last_high, &dur_high };
    cuLaunchKernel(func_high, 15, 1, 1, 128, 1, 1, 0, stream_high, args_high, nullptr);

    while (*t_last_low == 0 || *t_last_high == 0) usleep(10);
    usleep(10000); // Let both streams run for 10 ms

    unsigned long long prog_before_pause = prog_low[0];

    // Pause ONLY stream_low (*preempt_buf_low = 2)!
    uint64_t t_pause_cmd = read_ptimer();
    *preempt_buf_low = 2;
    __sync_synchronize();
    wr32(0x424684, 0x80001c01);
    wr32(0x424684, 0x00001c01);
    uint64_t t_pause_mmio = read_ptimer();

    usleep(200); // Wait 200 us for warps to settle in NANOSLEEP loop
    uint64_t t_last_at_pause = *t_last_low;
    unsigned long long prog_paused_1 = prog_low[0];
    unsigned long long atomic_paused_1 = *atomic_accum;
    unsigned long long high_prog_1 = prog_high[0];

    // Hold stream_low PAUSED for 10 ms while stream_high runs freely in the SAME CUcontext!
    usleep(10000);

    unsigned long long prog_paused_2 = prog_low[0];
    unsigned long long atomic_paused_2 = *atomic_accum;
    unsigned long long high_prog_2 = prog_high[0];

    // Resume stream_low in place (*preempt_buf_low = 0 + MMIO Trap Pulse to restore PC_orig from [0xffff88])!
    uint64_t t_resume_cmd = read_ptimer();
    *preempt_buf_low = 0;
    __sync_synchronize();
    wr32(0x424684, 0x80001c01);
    wr32(0x424684, 0x00001c01);
    while (*t_last_low == t_last_at_pause) {}
    uint64_t t_resumed = *t_last_low;

    CUresult res_high_during_pause = cuStreamSynchronize(stream_high);
    CUresult res_non_idemp = cuStreamSynchronize(stream_low);

    printf("  [Paused State] low_prog: %llu -> %llu -> %llu (low_delta=%llu!) | high_prog: %llu -> %llu (high_delta=+%llu!)\n",
           prog_before_pause, prog_paused_1, prog_paused_2, prog_paused_2 - prog_paused_1,
           high_prog_1, high_prog_2, high_prog_2 - high_prog_1);
    printf("  [Paused State] atomic_paused_1=%llu -> atomic_paused_2=%llu | stream_high res=%d completed=%d\n",
           atomic_paused_1, atomic_paused_2, (int)res_high_during_pause,
           (int)((prog_high[0] & (1ULL << 62)) != 0));

    // Verify 100% mathematical non-idempotent correctness!
    unsigned long long expected_atomic = (unsigned long long)(15 * 128) * (total_iters / 256);
    int checksum_errors = 0;
    for (unsigned int tid = 0; tid < 15 * 128; tid++) {
        unsigned int st = tid * 0x9e3779b9u + 0x12345678u;
        for (unsigned int it = 1; it <= total_iters; it++) {
            st = st * 1664525u + 1013904223u + (it ^ tid);
        }
        if (out_checksums[tid] != st) checksum_errors++;
    }
    printf("  [Resumed & Completed] res=%d | MMIO=%.2f us | On-GPU Pause latency=%.2f us | Resume latency=%.2f us\n",
           (int)res_non_idemp, (t_pause_mmio - t_pause_cmd) / 1000.0,
           (double)((long long)(t_last_at_pause - t_pause_cmd)) / 1000.0,
           (double)((long long)(t_resumed - t_resume_cmd)) / 1000.0);
    printf("  [Non-Idempotent Verification] atomic_accum=%llu (expected=%llu) | thread_checksum_errors=%d / %d\n",
           *atomic_accum, expected_atomic, checksum_errors, 15 * 128);
    int total_errors = 0;
    if (res_high_during_pause != CUDA_SUCCESS || res_non_idemp != CUDA_SUCCESS ||
        prog_paused_2 != prog_paused_1 || high_prog_2 <= high_prog_1 ||
        *atomic_accum != expected_atomic || checksum_errors != 0) {
        total_errors++;
    }

    // =========================================================================
    // TEST 2: Mode 3 — Sub-Stream Per-CTA Selective Pause & Resume in a Single Kernel
    // =========================================================================
    printf("\n=== TEST 2: Mode 3 (Sub-Stream Per-CTA Selective Pause & Resume in Same Kernel) ===\n");
    memset((void *)preempt_buf_low, 0, 256);
    memset((void *)prog_low, 0, 15 * 8);
    *t_last_low = 0;
    *atomic_accum = 0;
    memset(out_checksums, 0, 15 * 128 * sizeof(unsigned int));

    dbg_params.preempt_buf = (uint64_t)(uintptr_t)preempt_buf_low;
    dbg_params.guardian = 0;
    dbg_params.kernel_idx = 12;
    dbg_params.sched_enabled = 1;
    cuXtraSetDebuggerParams(func_non_idemp_2, &dbg_params, sizeof(dbg_params));
    cuLaunchKernel(func_non_idemp_2, 15, 1, 1, 128, 1, 1, 0, stream_low, args_non_idemp, nullptr);
    for (int w = 0; w < 10000 && (*t_last_low == 0 || prog_low[8] == 0); w++) usleep(10);
    usleep(10000);

    // Configure Per-CTA selective mode (preempt_buf_low[0] = 3):
    // Pause ONLY CTAs 0..7 (exit_flag_of_block[0..7] = 2), while CTAs 8..14 keep running (= 0)!
    for (int cta = 0; cta < 8; cta++) preempt_buf_low[4 + 2 * cta] = 2;
    for (int cta = 8; cta < 15; cta++) preempt_buf_low[4 + 2 * cta] = 0;
    preempt_buf_low[0] = 3;
    __sync_synchronize();
    wr32(0x424684, 0x80001c01);
    wr32(0x424684, 0x00001c01);

    usleep(200);
    unsigned long long p1[15], p2[15];
    for (int cta = 0; cta < 15; cta++) p1[cta] = prog_low[cta];
    usleep(10000); // Wait 10 ms while CTAs 0..7 are paused and CTAs 8..14 run!
    for (int cta = 0; cta < 15; cta++) p2[cta] = prog_low[cta];

    int paused_ctas_ok = 0, running_ctas_ok = 0;
    for (int cta = 0; cta < 8; cta++) if (p2[cta] == p1[cta]) paused_ctas_ok++;
    for (int cta = 8; cta < 15; cta++) if ((p2[cta] & ~(1ULL << 62)) > (p1[cta] & ~(1ULL << 62))) running_ctas_ok++;

    // Resume CTAs 0..7 in place!
    preempt_buf_low[0] = 0;
    __sync_synchronize();
    wr32(0x424684, 0x80001c01);
    wr32(0x424684, 0x00001c01);
    CUresult res_cta_test = cuStreamSynchronize(stream_low);

    int cta_checksum_errors = 0;
    for (unsigned int tid = 0; tid < 15 * 128; tid++) {
        unsigned int st = tid * 0x9e3779b9u + 0x12345678u;
        for (unsigned int it = 1; it <= total_iters; it++) {
            st = st * 1664525u + 1013904223u + (it ^ tid);
        }
        if (out_checksums[tid] != st) cta_checksum_errors++;
    }
    printf("  [Per-CTA Selectivity] Paused CTAs 0..7 frozen=%d/8 (delta=%llu, %llu) | Running CTAs 8..14 active=%d/7 (delta=+%llu, +%llu)\n",
           paused_ctas_ok, p2[0] - p1[0], p2[7] - p1[7],
           running_ctas_ok,
           (p2[8] & ~(1ULL << 62)) - (p1[8] & ~(1ULL << 62)),
           (p2[14] & ~(1ULL << 62)) - (p1[14] & ~(1ULL << 62)));
    printf("  [Per-CTA Resumed & Completed] res=%d | atomic_accum=%llu (expected=%llu) | checksum_errors=%d / %d\n",
           (int)res_cta_test, *atomic_accum, expected_atomic, cta_checksum_errors, 15 * 128);
    if (res_cta_test != CUDA_SUCCESS || paused_ctas_ok != 8 || running_ctas_ok != 7 ||
        *atomic_accum != expected_atomic || cta_checksum_errors != 0) {
        total_errors++;
    }

    // =========================================================================
    // TEST 3: Mode 1 — Paper-Grade Microsecond Early-Exit + Per-CTA Restore Flag
    // =========================================================================
    printf("\n=== TEST 3: Mode 1 (Selective Per-Stream Early-Exit + XSched Per-CTA Restore Flags) ===\n");
    memset((void *)preempt_buf_low, 0, 256);
    memset((void *)preempt_buf_high, 0, 256);
    memset((void *)prog_low, 0, 15 * 8);
    memset((void *)prog_high, 0, 15 * 8);
    *t_last_low = 0;
    *t_last_high = 0;

    dbg_params.preempt_buf = (uint64_t)(uintptr_t)preempt_buf_low;
    dbg_params.kernel_idx = 20;
    dbg_params.sched_enabled = 1;
    cuXtraSetDebuggerParams(func_low, &dbg_params, sizeof(dbg_params));
    unsigned long long dur_low = 3000000000ULL; // 3.0 seconds
    void *args_low[] = { &prog_low, &t_last_low, &dur_low };
    cuLaunchKernel(func_low, 15, 1, 1, 128, 1, 1, 0, stream_low, args_low, nullptr);

    dbg_params.preempt_buf = (uint64_t)(uintptr_t)preempt_buf_high;
    dbg_params.kernel_idx = 21;
    dbg_params.sched_enabled = 1;
    cuXtraSetDebuggerParams(func_high, &dbg_params, sizeof(dbg_params));
    dur_high = 50000000ULL; // 50 ms
    cuLaunchKernel(func_high, 15, 1, 1, 128, 1, 1, 0, stream_high, args_high, nullptr);

    while (*t_last_low == 0 || *t_last_high == 0) usleep(10);
    usleep(10000);

    uint64_t t_preempt_start = read_ptimer();
    *preempt_buf_low = 1;
    __sync_synchronize();
    wr32(0x424684, 0x80001c01);
    wr32(0x424684, 0x00001c01);
    uint64_t t_mmio_done = read_ptimer();

    CUresult res_low = cuStreamSynchronize(stream_low);
    uint64_t t_low_synced = read_ptimer();
    CUresult res_high = cuStreamSynchronize(stream_high);

    uint64_t recorded_preempt_idx = *(volatile uint64_t *)&preempt_buf_low[2];
    int restore_flags_set = 0;
    for (int cta = 0; cta < 15; cta++) {
        if (preempt_buf_low[5 + 2 * cta] == 1) restore_flags_set++;
    }

    printf("  [Early-Exit] res_low=%d | MMIO=%.2f us | Host cuStreamSynchronize latency=%.2f us | res_high=%d high_completed=%d\n",
           (int)res_low, (t_mmio_done - t_preempt_start) / 1000.0,
           (t_low_synced - t_preempt_start) / 1000.0,
           (int)res_high, (int)((prog_high[0] & (1ULL << 62)) != 0));
    printf("  [XSched Metadata] preempt_idx=%llu (expected=20) | restore_flag_of_block count=%d / 15\n",
           (unsigned long long)recorded_preempt_idx, restore_flags_set);
    if (res_low != CUDA_SUCCESS || res_high != CUDA_SUCCESS ||
        recorded_preempt_idx != 20ULL || restore_flags_set != 15) {
        total_errors++;
    }

    cuXtraMemcpyHtoD(info.handler_va + 0x390, orig_390, 16, 0);
    cuXtraMemcpyHtoD(ext_va, ext_backup.data(), flat_ext.size() * 8, 0);
    cuXtraMemcpyHtoD(exit_stub_va, exit_backup.data(), flat_exit.size() * 8, 0);
    cuXtraMemcpyHtoD(pause_tramp_va, tramp_backup.data(), flat_tramp.size() * 8, 0);
    cuXtraInvalInstrCache(ctx);
    return (total_errors == 0) ? 0 : 1;
}
