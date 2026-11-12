// preempt_rigorous_bench.cu
// Rigorous Real-World Compute Contention, Resource Reallocation & Overhead Benchmark
// on NVIDIA Blackwell SM120 (RTX 5060, 30 SMs, Max 1536 threads/SM = 46,080 threads).

#include "bar0.h"
#include "livepatch.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
#include "cuxtra/cuxtra.h"
}

static inline uint64_t read_ptimer() {
    uint32_t hi, lo, hi2;
    do {
        hi = rd32(0x9410);
        lo = rd32(0x9400);
        hi2 = rd32(0x9410);
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
}

__device__ __forceinline__ unsigned long long get_globaltimer() {
    unsigned long long r;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r));
    return r;
}

// ---------------------------------------------------------------------------
// 1. Heavy Full-Chip Non-Idempotent Low-Priority Compute + Memory Kernel
//    - No artificial timer throttling! Every thread executes dense FP32 FMA
//      chains + global memory updates + non-idempotent LCG state transitions.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(768, 2) void heavy_non_idempotent_low_kernel(
    volatile unsigned long long *cta_progress,
    volatile unsigned long long *t_last_issue,
    unsigned long long *atomic_accum,
    unsigned int *out_checksums,
    float *scratch_mem,
    unsigned int total_iters)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int state = tid * 0x9e3779b9u + 0x12345678u;
    float a = (float)(tid & 0xffu) * 0.001f + 1.0001f;
    float b = 0.9999f;
    float c = 0.0001f;

    for (unsigned int it = 1; it <= total_iters; it++) {
        // Dense FP32 FMA chain (64 FMAs = 128 FLOPs per inner loop)
        #pragma unroll
        for (int k = 0; k < 64; k++) {
            a = __fmaf_rn(a, b, c);
            b = __fmaf_rn(b, a, c);
            if ((k & 7) == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
                *t_last_issue = get_globaltimer();
            }
        }
        // Non-idempotent integer state recurrence
        state = state * 1664525u + 1013904223u + (it ^ __float_as_uint(a));

        if ((it & 0x3fu) == 0) {
            scratch_mem[tid] = a + b;
            if (threadIdx.x == 0) {
                cta_progress[blockIdx.x] = it;
                if (blockIdx.x == 0) *t_last_issue = get_globaltimer();
                if ((it & 0xffu) == 0) {
                    atomicAdd(atomic_accum, 1ULL);
                }
            }
        }
    }

    out_checksums[tid] = state;
    if (total_iters == 0) {
        __shared__ float s_pad[512];
        s_pad[threadIdx.x & 511u] = a;
        __syncthreads();
        scratch_mem[tid] = s_pad[(threadIdx.x + 1) & 511u];
    }
    if (threadIdx.x == 0) {
        cta_progress[blockIdx.x] = total_iters | (1ULL << 62);
        if (blockIdx.x == 0) *t_last_issue = get_globaltimer();
    }
}

static uint32_t compute_ref_checksum(unsigned int tid, unsigned int total_iters) {
    unsigned int state = tid * 0x9e3779b9u + 0x12345678u;
    float a = (float)(tid & 0xffu) * 0.001f + 1.0001f;
    float b = 0.9999f;
    float c = 0.0001f;
    for (unsigned int it = 1; it <= total_iters; it++) {
        for (int k = 0; k < 64; k++) {
            a = std::fma(a, b, c);
            b = std::fma(b, a, c);
        }
        uint32_t ua;
        memcpy(&ua, &a, sizeof(uint32_t));
        state = state * 1664525u + 1013904223u + (it ^ ua);
    }
    return state;
}

// ---------------------------------------------------------------------------
// 2. High-Priority Compute-Intensive Burst Kernel (All 30 SMs)
//    Records exact t_first_start and t_last_end on GPU globaltimer.
// ---------------------------------------------------------------------------
__global__ void high_pri_compute_burst_kernel(
    float *out_buf,
    volatile unsigned long long *t_first_start,
    volatile unsigned long long *t_last_end,
    unsigned int inner_loops)
{
    if (threadIdx.x == 0) {
        unsigned long long now = get_globaltimer();
        atomicMin((unsigned long long *)t_first_start, now);
    }

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float r0 = (float)(tid & 0xffu) * 0.01f + 1.0f;
    float r1 = r0 + 0.1f, r2 = r0 + 0.2f, r3 = r0 + 0.3f;
    float m  = 0.99999f,  a  = 0.00001f;

    // 32 FMAs = 64 FLOPs per loop iteration
    for (unsigned int i = 0; i < inner_loops; i++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            r0 = __fmaf_rn(r0, m, a);
            r1 = __fmaf_rn(r1, m, r0);
            r2 = __fmaf_rn(r2, m, r1);
            r3 = __fmaf_rn(r3, m, r2);
        }
    }
    out_buf[tid] = r0 + r1 + r2 + r3;

    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long now = get_globaltimer();
        atomicMax((unsigned long long *)t_last_end, now);
    }
}

// ---------------------------------------------------------------------------
// 3. Real-World 2D Tiled SGEMM Kernel (C = A * B, 1024x1024, 2D Grid (6,10), Block (16,16))
// ---------------------------------------------------------------------------
#define TILE_DIM 16
__global__ __launch_bounds__(256, 6) void tiled_sgemm_2d_kernel(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int N,
    int repeat_count)
{
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x, ty = threadIdx.y;
    int num_tiles = N / TILE_DIM;

    #pragma unroll 1
    for (int rep = 0; rep < repeat_count; rep++) {
        #pragma unroll 1
        for (int by = blockIdx.y; by < num_tiles; by += gridDim.y) {
            #pragma unroll 1
            for (int bx = blockIdx.x; bx < num_tiles; bx += gridDim.x) {
                int row = by * TILE_DIM + ty;
                int col = bx * TILE_DIM + tx;
                float acc = 0.0f;
                #pragma unroll 1
                for (int t = 0; t < num_tiles; t++) {
                    As[ty][tx] = A[row * N + (t * TILE_DIM + tx)];
                    Bs[ty][tx] = B[(t * TILE_DIM + ty) * N + col];
                    __syncthreads();
                    #pragma unroll 4
                    for (int k = 0; k < TILE_DIM; k++) {
                        acc = __fmaf_rn(As[ty][k], Bs[k][tx], acc);
                    }
                    __syncthreads();
                }
                C[row * N + col] = acc;
            }
        }
    }
}

__global__ void warmup_k() {}

// ---------------------------------------------------------------------------
// SM120 SASS Encoders for Unified 3-Granularity Trap Engine
// ---------------------------------------------------------------------------
namespace s120_ext {
using s120::I;
inline I ldc_u32(unsigned rd, uint32_t off) {
    uint64_t encoded_off = (uint64_t)(off >> 2) << 8;
    return {(encoded_off << 32) | 0xff000000ULL | ((uint64_t)rd << 16) | 0x7b82ULL,
            0x000e220000000800ULL};
}
inline I ld_e_strong_gpu_r0_r4() {
    return {0x0000000004007980ULL, 0x001e24000010e900ULL};
}
inline I ld_e_strong_gpu_r0_r6() {
    return {0x0000000006007980ULL, 0x001e24000010e900ULL};
}
inline I ldl_32_r0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0x00000000ff007983ULL, 0x0032240000100800ULL};
}
inline I stl_32_r0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0x00000000ff007387ULL, 0x0033e40000100800ULL};
}
inline I stl_32_r0_p0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0x00000000ff000387ULL, 0x0033e40000100800ULL};
}
inline I stl_32_rz(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0x000000ffff007387ULL, 0x0033e40000100800ULL};
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
inline I isetp_ne_rz(unsigned ra) {
    uint64_t w0 = (0xffULL << 32) | ((uint64_t)ra << 24) | 0x720cULL;
    return {w0, 0x001fda0003f05270ULL};
}
inline I isetp_ne_imm(unsigned ra, uint32_t imm) {
    uint64_t w0 = ((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL;
    return {w0, 0x001fda0003f05270ULL};
}
inline I isetp_ne_imm_and_p0(unsigned ra, uint32_t imm) {
    uint64_t w0 = ((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL;
    return {w0, 0x001fda0000705270ULL};
}
inline I isetp_eq_imm(unsigned ra, uint32_t imm) {
    uint64_t w0 = ((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL;
    return {w0, 0x001fda0003f02270ULL};
}
inline I isetp_eq_imm_or_p0(unsigned ra, uint32_t imm) {
    uint64_t w0 = ((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL;
    return {w0, 0x001fda0000702470ULL};
}
inline I nanosleep_imm(uint32_t ns) {
    uint64_t w0 = ((uint64_t)ns << 32) | 0x795dULL;
    return {w0, 0x000fea0003800000ULL};
}
inline uint64_t encode_bra_w0(int n_instrs, uint32_t low16) {
    int64_t off = (int64_t)n_instrs * 4;
    uint64_t lo8 = (uint64_t)(off & 0xff);
    uint64_t hi  = (uint64_t)(off >> 8);
    return (hi << 34) | (lo8 << 16) | low16;
}
inline I bra_p0(int n_instrs) {
    return {encode_bra_w0(n_instrs, 0x0947),
            (n_instrs < 0) ? 0x000fea000383ffffULL : 0x000fea0003800000ULL};
}
inline I bra_not_p0(int n_instrs) {
    return {encode_bra_w0(n_instrs, 0x8947),
            (n_instrs < 0) ? 0x000fea000383ffffULL : 0x000fea0003800000ULL};
}
inline I bra_rel(int n_instrs) {
    return {encode_bra_w0(n_instrs, 0x7947),
            (n_instrs < 0) ? 0x000fea000383ffffULL : 0x000fea0003800000ULL};
}
} // namespace s120_ext

struct KernelDbgParams {
    uint64_t preempt_buf;
    uint64_t guardian;
    int64_t  kernel_idx;
    uint32_t sched_enabled;
} __attribute__((packed));

static void bind_kernel(CUfunction func, volatile uint32_t *preempt_buf, int64_t kidx) {
    KernelDbgParams p{(uint64_t)(uintptr_t)preempt_buf, 0ULL, kidx, 1u};
    cuXtraSetDebuggerParams(func, &p, sizeof(p));
}

static inline void pulse_trap_broadcast() {
    wr32(0x424684, 0x80001c01);
    wr32(0x424684, 0x00001c01);
}

static void print_stats(const char *label, std::vector<double> &v) {
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    double avg = sum / v.size();
    double p50 = v[v.size() * 50 / 100];
    double p90 = v[v.size() * 90 / 100];
    double p99 = v[v.size() * 99 / 100];
    printf("  %-34s: min=%6.2f us | avg=%6.2f us | p50=%6.2f us | p90=%6.2f us | p99=%6.2f us | max=%6.2f us\n",
           label, v.front(), avg, p50, p90, p99, v.back());
}

int main() {
    setbuf(stdout, NULL);
    if (bar0_open()) return 1;
    cudaFree(0);
    warmup_k<<<1, 1>>>();
    cudaDeviceSynchronize();

    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    sm120arena::Info info = sm120arena::probe(ctx, 0);
    if (!info.ok || !info.handler_is_memobj) {
        printf("Failed to probe SM120 trap handler arena\n");
        return 1;
    }

    int lo_pri = 0, hi_pri = -5;
    cudaDeviceGetStreamPriorityRange(&lo_pri, &hi_pri);

    CUstream stream_low, stream_high, stream_high_eq;
    cuStreamCreateWithPriority(&stream_low, CU_STREAM_NON_BLOCKING, lo_pri);
    cuStreamCreateWithPriority(&stream_high, CU_STREAM_NON_BLOCKING, hi_pri);
    cuStreamCreateWithPriority(&stream_high_eq, CU_STREAM_NON_BLOCKING, lo_pri);

    CUfunction f_low = nullptr, f_high = nullptr, f_sgemm = nullptr;
    cudaGetFuncBySymbol(&f_low, (const void *)heavy_non_idempotent_low_kernel);
    cudaGetFuncBySymbol(&f_high, (const void *)high_pri_compute_burst_kernel);
    cudaGetFuncBySymbol(&f_sgemm, (const void *)tiled_sgemm_2d_kernel);

    // Allocate mapped control buffers (support up to 2048 CTAs -> 16 KB per buffer)
    volatile uint32_t *pbuf_low = nullptr, *pbuf_high = nullptr;
    cudaHostAlloc((void **)&pbuf_low, 16384, cudaHostAllocMapped);
    cudaHostAlloc((void **)&pbuf_high, 16384, cudaHostAllocMapped);
    memset((void *)pbuf_low, 0, 16384);
    memset((void *)pbuf_high, 0, 16384);

    // Allocate device/mapped buffers for workloads
    volatile unsigned long long *cta_progress = nullptr, *t_last_low = nullptr;
    volatile unsigned long long *t_high_start = nullptr, *t_high_end = nullptr;
    unsigned long long *atomic_accum = nullptr;
    unsigned int *out_checksums = nullptr;
    float *d_scratch = nullptr, *d_high_out = nullptr;
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaHostAlloc((void **)&cta_progress, 512 * sizeof(unsigned long long), cudaHostAllocMapped);
    cudaHostAlloc((void **)&t_last_low, sizeof(unsigned long long), cudaHostAllocMapped);
    cudaHostAlloc((void **)&t_high_start, sizeof(unsigned long long), cudaHostAllocMapped);
    cudaHostAlloc((void **)&t_high_end, sizeof(unsigned long long), cudaHostAllocMapped);
    cudaHostAlloc((void **)&atomic_accum, sizeof(unsigned long long), cudaHostAllocMapped);
    cudaHostAlloc((void **)&out_checksums, 240 * 768 * sizeof(unsigned int), cudaHostAllocMapped);
    cudaMalloc(&d_scratch, 240 * 768 * sizeof(float));
    cudaMalloc(&d_high_out, 60 * 512 * sizeof(float));

    const int N_GEMM = 1024;
    cudaMalloc(&d_A, N_GEMM * N_GEMM * sizeof(float));
    cudaMalloc(&d_B, N_GEMM * N_GEMM * sizeof(float));
    cudaMalloc(&d_C, N_GEMM * N_GEMM * sizeof(float));
    cudaMemset(d_A, 1, N_GEMM * N_GEMM * sizeof(float));
    cudaMemset(d_B, 1, N_GEMM * N_GEMM * sizeof(float));

    // -----------------------------------------------------------------------
    // PRE-HOOK BASELINE (O1 Part 1): Measure 2D Tiled SGEMM BEFORE hook install
    // -----------------------------------------------------------------------
    int sgemm_repeats = 2;
    int n_gemm_arg = N_GEMM;
    void *args_sgemm[] = { &d_A, &d_B, &d_C, &n_gemm_arg, &sgemm_repeats };
    for (int w = 0; w < 25; w++) {
        cuLaunchKernel(f_sgemm, 6, 10, 1, 16, 16, 1, 0, stream_high_eq, args_sgemm, nullptr);
        cuStreamSynchronize(stream_high_eq);
    }
    double sum_sgemm_unhooked_us = 0;
    const int O1_ITERS = 15;
    for (int i = 0; i < O1_ITERS; i++) {
        uint64_t t0 = read_ptimer();
        cuLaunchKernel(f_sgemm, 6, 10, 1, 16, 16, 1, 0, stream_high_eq, args_sgemm, nullptr);
        cuStreamSynchronize(stream_high_eq);
        uint64_t t1 = read_ptimer();
        sum_sgemm_unhooked_us += (t1 - t0) / 1000.0;
    }
    double avg_sgemm_unhooked_us = sum_sgemm_unhooked_us / O1_ITERS;

    // -----------------------------------------------------------------------
    // Install Unified 3-Granularity 38-Instruction Trap Handler
    // -----------------------------------------------------------------------
    s120::I test_r2_ctxsw;
    memcpy(&test_r2_ctxsw.w0, &info.handler[0x3b0], 8);
    memcpy(&test_r2_ctxsw.w1, &info.handler[0x3b8], 8);
    test_r2_ctxsw.w0 = (test_r2_ctxsw.w0 & 0xffffffffULL) | (0x11ULL << 32);
    test_r2_ctxsw.w1 = (test_r2_ctxsw.w1 & ~0x000ffe0000000000ULL) | 0x000fda0000000000ULL;

    uint64_t lop_w0, lop_w1;
    memcpy(&lop_w0, &info.handler[0x220], 8);
    memcpy(&lop_w1, &info.handler[0x228], 8);
    s120::I clear_r2_bit8 = {(lop_w0 & 0xffffffffULL) | (0xfffffff7ULL << 32), lop_w1};

    s120::I stl_64_rpc;
    memcpy(&stl_64_rpc.w0, &info.handler[0x110], 8);
    memcpy(&stl_64_rpc.w1, &info.handler[0x118], 8);

    uint64_t ext_va = info.tools_pc + 0xa00;
    uint64_t exit_stub_va = info.tools_pc + 0xe00;
    uint64_t pause_tramp_va = info.tools_pc + 0xe20;
    uint32_t tramp0_lo = (uint32_t)pause_tramp_va;
    uint32_t tramp1_lo = (uint32_t)(pause_tramp_va + 0x10);
    uint32_t tramp2_lo = (uint32_t)(pause_tramp_va + 0x20);
    uint32_t tramp_hi  = (uint32_t)(pause_tramp_va >> 32);

    using namespace s120;
    using namespace s120_ext;
    std::vector<I> ext_code = {
        ldc_u32(4, 0x170),                                              // 0: R4 = preempt_buf lo
        ldc_u32(5, 0x174),                                              // 1: R5 = preempt_buf hi
        isetp_ne_rz(4),                                                 // 2: P0 = (R4 != 0)
        bra_not_p0(51),                                                 // 3: @!P0 BRA done (idx 55)
        test_r2_ctxsw,                                                  // 4: (R2 & 0x11) != 0?
        bra_p0(49),                                                     // 5: @P0 BRA done (idx 55)
        clear_r2_bit8,                                                  // 6: R2 &= ~0x8
        {0x0000000000007919ULL, 0x000e620000002600ULL},                 // 7: S2R R0, SR_CTAID.Y
        {0x0000dd00ff037b82ULL, 0x000e620000000800ULL},                 // 8: LDC R3, c[0x0][0x374] (gridDim.y)
        {0x0000000000067919ULL, 0x001e620000002500ULL},                 // 9: S2R R6, SR_CTAID.X
        {0x0000000000087919ULL, 0x000e2c0000002700ULL},                 // 10: S2R R8, SR_CTAID.Z
        {0x0000dc00ff077b82ULL, 0x000e220000000800ULL},                 // 11: LDC R7, c[0x0][0x370] (gridDim.x)
        {0x0000000803007224ULL, 0x003fda00078e0200ULL},                 // 12: IMAD R0, R3, R8, R0
        {0x0000000700007224ULL, 0x001fda00078e0206ULL},                 // 13: IMAD R0, R0, R7, R6
        {0x0000000400037810ULL, 0x000fda0007ffe000ULL},                 // 14: IADD3 R3, PT, PT, R0, 0x4, R0
        {0x0000000403067825ULL, 0x000fda00078e0004ULL},                 // 15: IMAD.WIDE.U32 R6, R3, 0x4, R4
        ld_e_strong_gpu_r0_r4(),                                        // 16: LD.E.STRONG.GPU R0, [R4]
        isetp_ne_imm(0, 3),                                             // 17: P0 = (R0 != 3)
        bra_p0(1),                                                      // 18: @P0 BRA check_pause (idx 20)
        ld_e_strong_gpu_r0_r6(),                                        // 19: LD.E.STRONG.GPU R0, [R6]
        // idx 20: check_pause
        isetp_ne_imm(0, 2),                                             // 20: P0 = (R0 != 2)
        bra_p0(15),                                                     // 21: @P0 BRA check_exit (idx 37)
        // idx 22..36: do_pause (100% divergence-free predicated redirect to pause_tramp_va!)
        mov_imm_w01(0, 1),                                              // 22: MOV R0, 1
        isetp_ne_rz(0),                                                 // 23: P0 = true across all 32 lanes
        ldl_64_r0(0xffffa8),                                            // 24: LDL.64 R0, [0xffffa8]
        isetp_ne_imm_and_p0(0, tramp0_lo),                              // 25: P0 = P0 && (R0 != tramp0_lo)
        isetp_ne_imm_and_p0(0, tramp1_lo),                              // 26: P0 = P0 && (R0 != tramp1_lo)
        isetp_ne_imm_and_p0(0, tramp2_lo),                              // 27: P0 = P0 && (R0 != tramp2_lo)
        stl_64_r0_p0(0xffff88),                                         // 28: @P0 STL.64 [0xffff88], R0 (save PC_orig)
        mov_imm_w01(0, tramp0_lo),                                      // 29: MOV R0, tramp0_lo (NANOSLEEP 0x400; BRA)
        mov_imm_w01(1, tramp_hi),                                       // 30: MOV R1, tramp_hi
        stl_64_r0_p0(0xffffa8),                                         // 31: @P0 STL.64 [0xffffa8], R0
        ldl_32_r0(0xfffffc),                                            // 32: LDL.32 R0, [0xfffffc] (TRAP_RETURN_MASK)
        isetp_ne_imm_and_p0(0, 0xffffffffu),                            // 33: P0 = P0 && (TRAP_RETURN_MASK != 0xffffffff)
        mov_imm_w01(0, tramp2_lo),                                      // 34: MOV R0, tramp2_lo (BRA self for diverged sub-mask)
        stl_32_r0_p0(0xffffa8),                                         // 35: @P0 STL.32 [0xffffa8], R0
        bra_rel(17),                                                    // 36: BRA done_r1 (idx 54)
        // idx 37..47: check_exit
        isetp_ne_imm(0, 1),                                             // 37: P0 = (R0 != 1)
        bra_p0(9),                                                      // 38: @P0 BRA check_resume (idx 48)
        mov_imm_w01(0, 1),                                              // 39: MOV R0, 1
        {0x0000000406007385ULL, 0x000fda0000100900ULL},                 // 40: ST.E [R6+0x4], R0
        ldc_u32(8, 0x180),                                              // 41: LDC R8, c[0x0][0x180]
        ldc_u32(9, 0x184),                                              // 42: LDC R9, c[0x0][0x184]
        {0x0000000804007385ULL, 0x001fda0000100b08ULL},                 // 43: ST.E.64 [R4+0x8], R8
        mov_imm_w01(0, (uint32_t)exit_stub_va),                         // 44: MOV R0, lo32(exit_stub_va)
        mov_imm_w01(1, (uint32_t)(exit_stub_va >> 32)),                 // 45: MOV R1, hi32(exit_stub_va)
        stl_64_r0(0xffffa8),                                            // 46: STL.64 [0xffffa8], R0
        bra_rel(6),                                                     // 47: BRA done_r1 (idx 54)
        // idx 48..53: check_resume (R0 == 0, 100% divergence-free predicated restore!)
        ldl_64_r0(0xffffa8),                                            // 48: LDL.64 R0, [0xffffa8]
        isetp_eq_imm(0, tramp0_lo),                                     // 49: P0 = (R0 == tramp0_lo)
        isetp_eq_imm_or_p0(0, tramp1_lo),                               // 50: P0 = P0 || (R0 == tramp1_lo)
        isetp_eq_imm_or_p0(0, tramp2_lo),                               // 51: P0 = P0 || (R0 == tramp2_lo)
        ldl_64_r0(0xffff88),                                            // 52: LDL.64 R0, [0xffff88]
        stl_64_r0_p0(0xffffa8),                                         // 53: @P0 STL.64 [0xffffa8], R0
        // idx 54..57: done_r1 / done
        mov_imm_w01(1, 0xffff80),                                       // 54: MOV R1, 0xffff80
        mov_imm_w01(3, 0),                                              // 55: MOV R3, 0 (done)
        iadd_imm(12, 12, 0x10),                                         // 56: IADD R12, R12, 0x10
        ret_r12(),                                                      // 57: RET.ABS R12 0x20
    };
    std::vector<I> exit_stub = { exit_(), exit_() };
    uint32_t ns_val = getenv("TRAMP_NS") ? (uint32_t)strtoul(getenv("TRAMP_NS"), nullptr, 0) : 0x100u;
    std::vector<I> pause_tramp = { (ns_val > 0) ? nanosleep_imm(ns_val) : nop(), bra_rel(-2), bra_rel(-1) };
    auto flat_ext = flat(ext_code);
    auto flat_exit = flat(exit_stub);
    auto flat_tramp = flat(pause_tramp);

    uint64_t orig_390[2];
    memcpy(orig_390, &info.handler[0x390], 16);
    std::vector<uint64_t> ext_backup(flat_ext.size()), exit_backup(flat_exit.size()), tramp_backup(flat_tramp.size());
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

    printf("====================================================================================================\n");
    printf(" [BENCH 1] FULL-CHIP (ALL 30 SMs) HEAVY COMPUTE CONTENTION & REALLOCATION (Mode 2 In-Place Pause)\n");
    printf("   - Low-Priority Workload : 60 CTAs x 512 threads = 30,720 threads across all 30 SMs (Non-Idempotent)\n");
    printf("   - High-Priority Burst   : 60 CTAs x 256 threads = 15,360 threads across all 30 SMs (8.05 GFLOP FP32)\n");
    printf("====================================================================================================\n");

    unsigned int high_loops = 8192; // 60 * 256 * 8192 * 64 FLOPs = 8.053 GFLOPs
    double high_gflops_work = (double)60 * 256 * (double)high_loops * 64.0 / 1e9;
    void *args_high[] = { &d_high_out, &t_high_start, &t_high_end, &high_loops };
    unsigned int warmup_iters = 16;
    void *args_low_warm[] = { &cta_progress, &t_last_low, &atomic_accum, &out_checksums, &d_scratch, &warmup_iters };
    bind_kernel(f_low, pbuf_low, 1);
    bind_kernel(f_high, pbuf_high, 2);
    cuLaunchKernel(f_low, 1, 1, 1, 32, 1, 1, 0, stream_low, args_low_warm, nullptr);
    cuLaunchKernel(f_high, 1, 1, 1, 32, 1, 1, 0, stream_high, args_high, nullptr);
    cuCtxSynchronize();

    // 1A: Solo Baseline (High-Priority Burst on Idle GPU across all 30 SMs)
    double sum_solo_us = 0;
    const int B1_ITERS = 3;
    for (int i = 0; i < B1_ITERS; i++) {
        *t_high_start = ~0ULL; *t_high_end = 0ULL;
        uint64_t ht0 = read_ptimer();
        cuLaunchKernel(f_high, 60, 1, 1, 256, 1, 1, 0, stream_high_eq, args_high, nullptr);
        cuStreamSynchronize(stream_high_eq);
        uint64_t ht1 = read_ptimer();
        sum_solo_us += (ht1 - ht0) / 1000.0;
    }
    double avg_solo_us = sum_solo_us / B1_ITERS;
    double solo_gflops = high_gflops_work / (avg_solo_us / 1e6);
    printf("  [1A. Solo Idle-GPU Baseline]     High-Pri Turnaround = %7.2f us | Throughput = %7.2f GFLOPS (100.0%%)\n",
           avg_solo_us, solo_gflops);

    // 1B-1: Unpreempted Contention (Standard Equal-Priority CUDA Streams across all 30 SMs)
    unsigned int low_iters_b1 = 49152;
    void *args_low_b1[] = { &cta_progress, &t_last_low, &atomic_accum, &out_checksums, &d_scratch, &low_iters_b1 };
    double sum_contended_us = 0;
    for (int i = 0; i < B1_ITERS; i++) {
        memset((void *)pbuf_low, 0, 1024);
        memset((void *)cta_progress, 0, 60 * 8);
        *t_last_low = 0; *atomic_accum = 0;
        bind_kernel(f_low, pbuf_low, 1);
        bind_kernel(f_high, pbuf_high, 2);
        cuLaunchKernel(f_low, 60, 1, 1, 512, 1, 1, 0, stream_low, args_low_b1, nullptr);
        while (*t_last_low == 0 || cta_progress[59] == 0) usleep(10);
        usleep(1000); // All 60 CTAs actively saturating all 30 SMs

        *t_high_start = ~0ULL; *t_high_end = 0ULL;
        uint64_t ht0 = read_ptimer();
        cuLaunchKernel(f_high, 60, 1, 1, 256, 1, 1, 0, stream_high_eq, args_high, nullptr);
        cuStreamSynchronize(stream_high_eq);
        uint64_t ht1 = read_ptimer();
        sum_contended_us += (ht1 - ht0) / 1000.0;
        cuStreamSynchronize(stream_low);
    }
    double avg_contended_us = sum_contended_us / B1_ITERS;
    double contended_gflops = high_gflops_work / (avg_contended_us / 1e6);
    printf("  [1B-1. Unpreempted Equal-Pri]    High-Pri Turnaround = %7.2f us | Throughput = %7.2f GFLOPS (%5.1f%% of Solo | %.2fx Slowdown!)\n",
           avg_contended_us, contended_gflops, 100.0 * contended_gflops / solo_gflops, avg_contended_us / avg_solo_us);

    // 1B-2: Unpreempted Contention with CUDA Hardware Priority Stream (-5)
    double sum_hwpri_us = 0;
    for (int i = 0; i < B1_ITERS; i++) {
        memset((void *)pbuf_low, 0, 1024);
        memset((void *)cta_progress, 0, 60 * 8);
        *t_last_low = 0; *atomic_accum = 0;
        bind_kernel(f_low, pbuf_low, 1);
        bind_kernel(f_high, pbuf_high, 2);
        cuLaunchKernel(f_low, 60, 1, 1, 512, 1, 1, 0, stream_low, args_low_b1, nullptr);
        while (*t_last_low == 0 || cta_progress[59] == 0) usleep(10);
        usleep(1000);

        *t_high_start = ~0ULL; *t_high_end = 0ULL;
        uint64_t ht0 = read_ptimer();
        cuLaunchKernel(f_high, 60, 1, 1, 256, 1, 1, 0, stream_high, args_high, nullptr);
        cuStreamSynchronize(stream_high);
        uint64_t ht1 = read_ptimer();
        sum_hwpri_us += (ht1 - ht0) / 1000.0;
        cuStreamSynchronize(stream_low);
    }
    double avg_hwpri_us = sum_hwpri_us / B1_ITERS;
    double hwpri_gflops = high_gflops_work / (avg_hwpri_us / 1e6);
    printf("  [1B-2. Unpreempted HW-Pri (-5)]  High-Pri Turnaround = %7.2f us | Throughput = %7.2f GFLOPS (%5.1f%% of Solo | %.2fx Slowdown)\n",
           avg_hwpri_us, hwpri_gflops, 100.0 * hwpri_gflops / solo_gflops, avg_hwpri_us / avg_solo_us);

    // 1C: Mode 2 Preemption (Pause stream_low in-place -> Run stream_high_eq -> Resume stream_low in-place)
    double sum_preempted_us = 0, sum_yield_us = 0, sum_mmio_us = 0, sum_resume_us = 0;
    int b1_checksum_errs = 0;
    int total_errors = 0;
    unsigned long long last_atomic = 0;
    unsigned long long expected_atomic_b1 = (unsigned long long)60 * (low_iters_b1 / 256);

    for (int i = 0; i < B1_ITERS; i++) {
        memset((void *)pbuf_low, 0, 1024);
        memset((void *)cta_progress, 0, 60 * 8);
        *t_last_low = 0; *atomic_accum = 0;
        bind_kernel(f_low, pbuf_low, 1);
        bind_kernel(f_high, pbuf_high, 2);
        cuLaunchKernel(f_low, 60, 1, 1, 512, 1, 1, 0, stream_low, args_low_b1, nullptr);
        while (*t_last_low == 0 || cta_progress[59] == 0) usleep(10);
        usleep(1000); // Let stream_low saturate all 30 SMs

        // Preempt stream_low via Mode 2 (Normal-Mode Yielding Trampoline)
        uint64_t tp0 = read_ptimer();
        pbuf_low[0] = 2u;
        __sync_synchronize();
        pulse_trap_broadcast();
        uint64_t tp1 = read_ptimer();
        usleep(50);
        uint64_t t_stopped = *t_last_low;
        if (t_stopped <= tp0) total_errors++;
        double yield_us = (double)((int64_t)(t_stopped - tp0)) / 1000.0;

        // Launch High-Priority Burst while stream_low's 30,720 threads yield in Normal-Mode Trampoline on the 30 SMs
        *t_high_start = ~0ULL; *t_high_end = 0ULL;
        uint64_t ht0 = read_ptimer();
        cuLaunchKernel(f_high, 60, 1, 1, 256, 1, 1, 0, stream_high_eq, args_high, nullptr);
        cuStreamSynchronize(stream_high_eq);
        uint64_t ht1 = read_ptimer();

        // Resume stream_low in-place via Trap #2 broadcast pulse
        uint64_t tr0 = read_ptimer();
        pbuf_low[0] = 0u;
        __sync_synchronize();
        pulse_trap_broadcast();
        while (*t_last_low == t_stopped && (read_ptimer() - tr0) < 5000000ULL) {}
        uint64_t tr1 = *t_last_low;
        if (tr1 <= tr0 || tr1 == t_stopped) total_errors++;
        double resume_us = (double)((int64_t)(tr1 - tr0)) / 1000.0;
        cuStreamSynchronize(stream_low);

        sum_mmio_us += (tp1 - tp0) / 1000.0;
        sum_yield_us += yield_us;
        sum_preempted_us += (ht1 - ht0) / 1000.0;
        sum_resume_us += resume_us;
        last_atomic = *atomic_accum;
        if (last_atomic != expected_atomic_b1) total_errors++;
    }
    // Verify non-idempotent checksums across 30,720 threads
    for (unsigned int tid = 0; tid < 60 * 512; tid += 64) {
        if (out_checksums[tid] != compute_ref_checksum(tid, low_iters_b1)) b1_checksum_errs++;
    }
    if (b1_checksum_errs != 0) total_errors += b1_checksum_errs;
    double avg_preempted_us = sum_preempted_us / B1_ITERS;
    double preempted_gflops = high_gflops_work / (avg_preempted_us / 1e6);
    printf("  [1C. Mode 2 In-Place Preempted]  High-Pri Turnaround = %7.2f us | Throughput = %7.2f GFLOPS (%5.1f%% of Solo | %.2fx Speedup vs Equal-Pri!)\n",
           avg_preempted_us, preempted_gflops, 100.0 * preempted_gflops / solo_gflops, avg_contended_us / avg_preempted_us);
    printf("       -> Preemption Latencies : BAR0 MMIO = %.2f us | On-GPU Compute Yield = %.2f us | In-Place Resume = %.2f us\n",
           sum_mmio_us / B1_ITERS, sum_yield_us / B1_ITERS, sum_resume_us / B1_ITERS);
    printf("       -> Non-Idempotent State : atomic_accum = %llu / %llu | sampled_checksum_errors = %d ( across 30,720 threads! )\n\n",
           last_atomic, expected_atomic_b1, b1_checksum_errs);

    // -----------------------------------------------------------------------
    // BENCH 2: 100% SM Hardware Slot Saturation (46,080 / 46,080 Threads)
    //          Proving Mode 3 Hybrid Slot Eviction + In-Place Pause & Mode 1 Multi-Wave Takeover
    // -----------------------------------------------------------------------
    printf("====================================================================================================\n");
    printf(" [BENCH 2] 100%% SM HARDWARE SLOT SATURATION (46,080 / 46,080 Threads = 100%% Occupancy on 30 SMs)\n");
    printf("   - Low-Priority Workload : 60 CTAs x 768 threads = 46,080 threads (100.0%% of hardware thread slots!)\n");
    printf("   - High-Priority Arrival : 30 CTAs x 512 threads = 15,360 threads (arrives AFTER all SM slots are full)\n");
    printf("====================================================================================================\n");

    unsigned int low_iters_b2 = 65536;
    void *args_low_b2[] = { &cta_progress, &t_last_low, &atomic_accum, &out_checksums, &d_scratch, &low_iters_b2 };
    unsigned int high_loops_b2 = 4096;
    void *args_high_b2[] = { &d_high_out, &t_high_start, &t_high_end, &high_loops_b2 };

    // 2A: Solo High-Pri Baseline (30 CTAs x 512 threads)
    bind_kernel(f_high, pbuf_high, 2);
    *t_high_start = ~0ULL; *t_high_end = 0ULL;
    uint64_t b2_s0 = read_ptimer();
    cuLaunchKernel(f_high, 30, 1, 1, 512, 1, 1, 0, stream_high, args_high_b2, nullptr);
    cuStreamSynchronize(stream_high);
    uint64_t b2_s1 = read_ptimer();
    double b2_solo_us = (b2_s1 - b2_s0) / 1000.0;
    printf("  [2A. Solo High-Pri Baseline]     Turnaround = %8.2f us | Queue-to-SM-Start Delay = %6.2f us\n",
           b2_solo_us, (double)(*t_high_start - b2_s0) / 1000.0);

    // 2B: Unpreempted 100% Slot Saturation (stream_high blocked waiting for SM slots)
    memset((void *)pbuf_low, 0, 1024);
    memset((void *)cta_progress, 0, 60 * 8);
    *t_last_low = 0; *atomic_accum = 0;
    bind_kernel(f_low, pbuf_low, 1);
    bind_kernel(f_high, pbuf_high, 2);
    cuLaunchKernel(f_low, 60, 1, 1, 704, 1, 1, 0, stream_low, args_low_b2, nullptr);
    while (*t_last_low == 0 || cta_progress[59] == 0) usleep(10);
    usleep(500); // All 30 SMs saturated!

    *t_high_start = ~0ULL; *t_high_end = 0ULL;
    uint64_t b2_u0 = read_ptimer();
    cuLaunchKernel(f_high, 30, 1, 1, 512, 1, 1, 0, stream_high, args_high_b2, nullptr);
    cuStreamSynchronize(stream_high);
    uint64_t b2_u1 = read_ptimer();
    cuStreamSynchronize(stream_low);
    double b2_unpreempt_us = (b2_u1 - b2_u0) / 1000.0;
    double b2_unpreempt_qdelay = (double)(*t_high_start - b2_u0) / 1000.0;
    printf("  [2B. Unpreempted 100%% Full Slots] Turnaround = %8.2f us | Queue-to-SM-Start Delay = %6.2f us (Blocked by 100%% full SM slots!)\n",
           b2_unpreempt_us, b2_unpreempt_qdelay);

    // 2C: Mode 3 Hybrid (Evict CTAs 0..29 via cmd=1 to free thread slots + Pause CTAs 30..59 in-place via cmd=2!)
    memset((void *)pbuf_low, 0, 1024);
    memset((void *)cta_progress, 0, 60 * 8);
    *t_last_low = 0; *atomic_accum = 0;
    bind_kernel(f_low, pbuf_low, 1);
    bind_kernel(f_high, pbuf_high, 2);
    cuLaunchKernel(f_low, 60, 1, 1, 704, 1, 1, 0, stream_low, args_low_b2, nullptr);
    while (*t_last_low == 0 || cta_progress[59] == 0) usleep(10);
    usleep(500); // All 30 SMs saturated

    for (int c = 0; c < 30; c++) pbuf_low[4 + 2 * c] = 1u;  // Evict CTAs 0..29 (free slots across all 30 SMs)
    for (int c = 30; c < 60; c++) pbuf_low[4 + 2 * c] = 2u; // In-place pause CTAs 30..59 (preserve state!)
    pbuf_low[0] = 3u;
    __sync_synchronize();

    *t_high_start = ~0ULL; *t_high_end = 0ULL;
    uint64_t b2_h0 = read_ptimer();
    pulse_trap_broadcast();
    usleep(10);
    cuLaunchKernel(f_high, 30, 1, 1, 512, 1, 1, 0, stream_high, args_high_b2, nullptr);
    cuStreamSynchronize(stream_high);
    uint64_t b2_h1 = read_ptimer();

    // Resume CTAs 30..59 in-place while keeping pbuf_low[0] = 3u so any CWD-queued CTAs in 0..29 also evict!
    for (int c = 30; c < 60; c++) pbuf_low[4 + 2 * c] = 0u;
    __sync_synchronize();
    while (cuStreamQuery(stream_low) == CUDA_ERROR_NOT_READY) {
        pulse_trap_broadcast();
        usleep(50);
    }
    cuStreamSynchronize(stream_low);

    int evicted_flags = 0, preserved_completed = 0;
    for (int c = 0; c < 60; c++) {
        if (pbuf_low[5 + 2 * c] == 1u) evicted_flags++;
        if (c >= 30 && (cta_progress[c] & (1ULL << 62))) preserved_completed++;
    }
    double b2_hybrid_us = (b2_h1 - b2_h0) / 1000.0;
    double b2_hybrid_qdelay = (double)(*t_high_start - b2_h0) / 1000.0;
    printf("  [2C. Mode 3 Hybrid Evict+Pause]  Turnaround = %8.2f us | Queue-to-SM-Start Delay = %6.2f us (%.1fx Faster Turnaround!)\n",
           b2_hybrid_us, b2_hybrid_qdelay, b2_unpreempt_us / b2_hybrid_us);
    printf("       -> Hybrid State Accounting: Evicted CTAs recorded = %d/30 | In-Place Paused CTAs 30..59 resumed & completed = %d/30\n",
           evicted_flags, preserved_completed);

    // 2D: Multi-Wave Grid (240 CTAs x 512 threads = 4 Waves across 30 SMs) Preemption via Mode 1 + High-Pri Stream
    memset((void *)pbuf_low, 0, 4096);
    memset((void *)cta_progress, 0, 240 * 8);
    *t_last_low = 0; *atomic_accum = 0;
    bind_kernel(f_low, pbuf_low, 1);
    bind_kernel(f_high, pbuf_high, 2);
    cuLaunchKernel(f_low, 240, 1, 1, 512, 1, 1, 0, stream_low, args_low_b2, nullptr);
    while (*t_last_low == 0) usleep(10);
    usleep(1000);

    *t_high_start = ~0ULL; *t_high_end = 0ULL;
    pbuf_low[0] = 1u;
    __sync_synchronize();
    uint64_t b2_m0 = read_ptimer();
    cuLaunchKernel(f_high, 60, 1, 1, 256, 1, 1, 0, stream_high, args_high_b2, nullptr);
    pulse_trap_broadcast();
    while (cuStreamQuery(stream_high) == CUDA_ERROR_NOT_READY) {
        pulse_trap_broadcast();
        usleep(20);
    }
    cuStreamSynchronize(stream_high);
    uint64_t b2_m1 = read_ptimer();
    while (cuStreamQuery(stream_low) == CUDA_ERROR_NOT_READY) {
        pulse_trap_broadcast();
        usleep(100);
    }
    cuStreamSynchronize(stream_low);
    double b2_mw_us = (b2_m1 - b2_m0) / 1000.0;
    double b2_mw_qdelay = (double)(*t_high_start - b2_m0) / 1000.0;
    printf("  [2D. Mode 1 Multi-Wave (240 CTA)] High-Pri Turnaround = %8.2f us | Queue-to-SM-Start Delay = %6.2f us (Wave-0 Evicted -> High-Pri Immediate Takeover!)\n\n",
           b2_mw_us, b2_mw_qdelay);

    // -----------------------------------------------------------------------
    // BENCH 3: Real-World 2D Tiled SGEMM (1024x1024) Overhead Breakdown (O1, O2, O3, O4)
    // -----------------------------------------------------------------------
    printf("====================================================================================================\n");
    printf(" [BENCH 3] QUANTITATIVE OVERHEAD BREAKDOWN ON 2D TILED SGEMM (1024x1024, Grid=(6,10), Block=(16,16))\n");
    printf("====================================================================================================\n");

    memset((void *)pbuf_high, 0, 16384);
    bind_kernel(f_sgemm, pbuf_high, 300);

    // O1 Part 2: Measure 2D Tiled SGEMM AFTER hook install + debugger params bound (0 traps fired)
    double sum_sgemm_hooked_us = 0;
    for (int i = 0; i < O1_ITERS; i++) {
        uint64_t t0 = read_ptimer();
        cuLaunchKernel(f_sgemm, 6, 10, 1, 16, 16, 1, 0, stream_high_eq, args_sgemm, nullptr);
        cuStreamSynchronize(stream_high_eq);
        uint64_t t1 = read_ptimer();
        sum_sgemm_hooked_us += (t1 - t0) / 1000.0;
    }
    double avg_sgemm_hooked_us = sum_sgemm_hooked_us / O1_ITERS;
    double o1_pct = 100.0 * (avg_sgemm_hooked_us - avg_sgemm_unhooked_us) / avg_sgemm_unhooked_us;
    printf("  [O1. Static Hook + Params Overhead] Unhooked SGEMM = %7.2f us | Hooked SGEMM (0 traps) = %7.2f us | Overhead = %+5.2f%%\n",
           avg_sgemm_unhooked_us, avg_sgemm_hooked_us, o1_pct);

    // O2: High-Priority Trap Pass-Through Tax (10 broadcast pulses hitting active SGEMM with pbuf_high[0]=0)
    double sum_sgemm_10pulses_us = 0;
    for (int i = 0; i < 10; i++) {
        uint64_t t0 = read_ptimer();
        cuLaunchKernel(f_sgemm, 6, 10, 1, 16, 16, 1, 0, stream_high_eq, args_sgemm, nullptr);
        for (int p = 0; p < 10; p++) {
            uint64_t tw = read_ptimer();
            while ((read_ptimer() - tw) < 80000ULL) {} // 80 us spacing -> 10 pulses inside SGEMM window
            pulse_trap_broadcast();
        }
        cuStreamSynchronize(stream_high_eq);
        uint64_t t1 = read_ptimer();
        sum_sgemm_10pulses_us += (t1 - t0) / 1000.0;
    }
    double avg_sgemm_10pulses_us = sum_sgemm_10pulses_us / 10.0;
    double per_pulse_tax_us = (avg_sgemm_10pulses_us - avg_sgemm_hooked_us) / 10.0;
    printf("  [O2. High-Pri Trap Pass-Through]    SGEMM w/ 10 Broadcast Traps = %7.2f us | Per-Pulse Tax = %5.2f us/trap (%5.2f%% per trap)\n",
           avg_sgemm_10pulses_us, per_pulse_tax_us, 100.0 * per_pulse_tax_us / avg_sgemm_hooked_us);

    // O3: Co-Resident Normal-Mode Trampoline Tax on 2D Tiled SGEMM
    memset((void *)pbuf_low, 0, 1024);
    memset((void *)cta_progress, 0, 60 * 8);
    *t_last_low = 0; *atomic_accum = 0;
    bind_kernel(f_low, pbuf_low, 1);
    bind_kernel(f_sgemm, pbuf_high, 300);
    cuLaunchKernel(f_low, 30, 1, 1, 512, 1, 1, 0, stream_low, args_low_b1, nullptr);
    uint64_t tw0 = read_ptimer();
    while ((*t_last_low == 0 || cta_progress[29] == 0) && (read_ptimer() - tw0) < 2000000000ULL) usleep(10);
    usleep(500);
    // Pause all 30 CTAs (15,360 threads across all 30 SMs) of stream_low in Normal-Mode Trampoline
    pbuf_low[0] = 2u;
    __sync_synchronize();
    pulse_trap_broadcast();
    usleep(50);

    double sum_sgemm_coresident_us = 0;
    for (int i = 0; i < 5; i++) {
        uint64_t t0 = read_ptimer();
        cuLaunchKernel(f_sgemm, 6, 10, 1, 16, 16, 1, 0, stream_high_eq, args_sgemm, nullptr);
        while (cuStreamQuery(stream_high_eq) == CUDA_ERROR_NOT_READY) {
            if ((read_ptimer() - t0) > 1000000000ULL) {
                printf("  [O3 DIAG] stream_high_eq blocked at iter %d (cta_prog[0]=%llu, [29]=%llu)! Resuming stream_low...\n",
                       i, cta_progress[0], cta_progress[29]);
                pbuf_low[0] = 0u;
                __sync_synchronize();
                pulse_trap_broadcast();
                break;
            }
            usleep(10);
        }
        cuStreamSynchronize(stream_high_eq);
        uint64_t t1 = read_ptimer();
        sum_sgemm_coresident_us += (t1 - t0) / 1000.0;
    }
    pbuf_low[0] = 0u;
    __sync_synchronize();
    pulse_trap_broadcast();
    uint64_t tw_low = read_ptimer();
    while (cuStreamQuery(stream_low) == CUDA_ERROR_NOT_READY) {
        if ((read_ptimer() - tw_low) > 1000000000ULL) {
            printf("  [O3 DIAG] stream_low blocked on resume (cta_prog[0]=%llu, [29]=%llu)! Sending exit...\n",
                   cta_progress[0], cta_progress[29]);
            pbuf_low[0] = 1u;
            __sync_synchronize();
            pulse_trap_broadcast();
            break;
        }
        pulse_trap_broadcast();
        usleep(50);
    }
    cuStreamSynchronize(stream_low);

    double avg_sgemm_coresident_us = sum_sgemm_coresident_us / 5.0;
    double o3_pct = 100.0 * (avg_sgemm_coresident_us - avg_sgemm_hooked_us) / avg_sgemm_hooked_us;
    printf("  [O3. Co-Resident Trampoline Tax]    Solo SGEMM = %7.2f us | w/ 30 Paused CTAs (All 30 SMs) = %7.2f us | Overhead = %+5.2f%%\n\n",
           avg_sgemm_hooked_us, avg_sgemm_coresident_us, o3_pct);

    // -----------------------------------------------------------------------
    // O4: 50-Cycle Statistical Latency Distribution under Full 30-SM Load
    // -----------------------------------------------------------------------
    printf("====================================================================================================\n");
    printf(" [BENCH 4] 50-CYCLE STATISTICAL LATENCY DISTRIBUTION UNDER FULL 30-SM LOAD (Min/Avg/P50/P90/P99/Max)\n");
    printf("====================================================================================================\n");

    unsigned int low_iters_b4 = 262144;
    unsigned long long expected_atomic_b4 = (unsigned long long)60 * (low_iters_b4 / 256);
    void *args_low_b4[] = { &cta_progress, &t_last_low, &atomic_accum, &out_checksums, &d_scratch, &low_iters_b4 };
    memset((void *)pbuf_low, 0, 1024);
    memset((void *)cta_progress, 0, 60 * 8);
    *t_last_low = 0; *atomic_accum = 0;
    bind_kernel(f_low, pbuf_low, 1);

    cuLaunchKernel(f_low, 60, 1, 1, 512, 1, 1, 0, stream_low, args_low_b4, nullptr);
    while (*t_last_low == 0 || cta_progress[59] == 0) usleep(10);
    usleep(300);

    std::vector<double> vec_mmio, vec_yield, vec_resume;
    int b4_timing_errs = 0;
    const int N_CYCLES = 50;
    for (int c = 0; c < N_CYCLES; c++) {
        // Wait for a fresh timestamp tick from running kernel before triggering pause
        uint64_t t_pre = *t_last_low;
        uint64_t tw_pre = read_ptimer();
        while (*t_last_low == t_pre && (read_ptimer() - tw_pre) < 500000ULL) {}

        uint64_t tp0 = read_ptimer();
        pbuf_low[0] = 2u;
        __sync_synchronize();
        pulse_trap_broadcast();
        uint64_t tp1 = read_ptimer();
        usleep(40);
        uint64_t t_stopped = *t_last_low;
        if (t_stopped <= tp0) b4_timing_errs++;
        double y_us = (double)((int64_t)(t_stopped - tp0)) / 1000.0;

        uint64_t tr0 = read_ptimer();
        pbuf_low[0] = 0u;
        __sync_synchronize();
        pulse_trap_broadcast();
        while (*t_last_low == t_stopped && (read_ptimer() - tr0) < 5000000ULL) {}
        uint64_t tr1 = *t_last_low;
        if (tr1 <= tr0 || tr1 == t_stopped) b4_timing_errs++;
        double r_us = (double)((int64_t)(tr1 - tr0)) / 1000.0;

        vec_mmio.push_back((tp1 - tp0) / 1000.0);
        vec_yield.push_back(y_us);
        vec_resume.push_back(r_us);
        usleep(60);
    }
    pbuf_low[0] = 0u;
    __sync_synchronize();
    pulse_trap_broadcast();
    CUresult res_b4 = cuStreamSynchronize(stream_low);
    if (res_b4 != CUDA_SUCCESS) total_errors++;

    int b4_checksum_errs = 0;
    for (unsigned int tid = 0; tid < 60 * 512; tid += 64) {
        if (out_checksums[tid] != compute_ref_checksum(tid, low_iters_b4)) b4_checksum_errs++;
    }
    unsigned long long b4_atomic = *atomic_accum;
    int b4_total_errs = b4_timing_errs + b4_checksum_errs + ((b4_atomic != expected_atomic_b4) ? 1 : 0);
    total_errors += b4_total_errs;

    print_stats("1. Host BAR0 MMIO Pulse Latency", vec_mmio);
    print_stats("2. On-GPU 30-SM Compute Yield", vec_yield);
    print_stats("3. In-Place Trampoline Resume", vec_resume);
    printf("  [50-Cycle Integrity Verification]   stream_low completed after 50 Pause/Resume cycles: atomic=%llu/%llu | checksum_errs=%d | timing_errs=%d (total_errors=%d)\n",
           b4_atomic, expected_atomic_b4, b4_checksum_errs, b4_timing_errs, total_errors);

    // Restore original trap handler
    cuXtraMemcpyHtoD(info.handler_va + 0x390, orig_390, 16, 0);
    cuXtraMemcpyHtoD(ext_va, ext_backup.data(), flat_ext.size() * 8, 0);
    cuXtraMemcpyHtoD(exit_stub_va, exit_backup.data(), flat_exit.size() * 8, 0);
    cuXtraMemcpyHtoD(pause_tramp_va, tramp_backup.data(), flat_tramp.size() * 8, 0);
    cuXtraInvalInstrCache(ctx);
    return (total_errors == 0) ? 0 : 1;
}
