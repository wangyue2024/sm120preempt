#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"

#define CHK_RT(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", \
                     (int)_e, cudaGetErrorString(_e), __FILE__, __LINE__); \
        std::exit(2); \
    } \
} while (0)

#define CHK_BLAS(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        std::exit(2); \
    } \
} while (0)

namespace {

// =========================================================================
// Realistic LLM & DNN CUDA Kernels (Non-Idempotent + Tensor/Memory Mix)
// =========================================================================

// Fused Causal Tiled Flash-style Self-Attention (1 head per block-row, runs ~6-8ms on RTX 5060)
__global__ void k_causal_attention(const float *Q, const float *K, const float *V,
                                   float *O, int seq_len, int head_dim, int n_heads) {
    const int head = blockIdx.y;
    const int row  = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= n_heads || row >= seq_len) return;

    const int base = head * seq_len * head_dim;
    const float *q_row = Q + base + row * head_dim;
    float *o_row = O + base + row * head_dim;

    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float acc[8] = {0}; // compute 8 output dims per thread-pass (head_dim <= 64 strided)
    const int d_stride = 8;
    const float inv_sqrt_d = rsqrtf(static_cast<float>(head_dim));

    // Online softmax (FlashAttention-style) over causal keys 0..row
    for (int col = 0; col <= row; ++col) {
        const float *k_col = K + base + col * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q_row[d] * k_col[d];
        float s = dot * inv_sqrt_d;
        float new_max = fmaxf(max_score, s);
        float exp_old = expf(max_score - new_max);
        float exp_new = expf(s - new_max);
        const float *v_col = V + base + col * head_dim;
        for (int d = 0; d < d_stride; ++d) {
            acc[d] = acc[d] * exp_old + exp_new * v_col[d];
        }
        sum_exp = sum_exp * exp_old + exp_new;
        max_score = new_max;
    }
    float inv_sum = 1.0f / (sum_exp + 1e-6f);
    for (int d = 0; d < head_dim; ++d) {
        o_row[d] += acc[d % d_stride] * inv_sum; // NON-IDEMPOTENT residual add!
    }
}

// In-place RMSNorm + RoPE + SwiGLU activation (strictly non-idempotent in-place update)
__global__ void k_rmsnorm_swiglu_inplace(float *x, const float *gate, int rows, int hidden) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float s_sum[128];
    const int tid = threadIdx.x;
    float *xr = x + static_cast<size_t>(r) * hidden;
    const float *gr = gate + static_cast<size_t>(r) * hidden;

    float local = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = xr[i];
        local += v * v;
    }
    s_sum[tid] = local;
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) {
        if (tid < off) s_sum[tid] += s_sum[tid + off];
        __syncthreads();
    }
    float inv_rms = rsqrtf(s_sum[0] / static_cast<float>(hidden) + 1e-5f);
    for (int i = tid; i < hidden; i += blockDim.x) {
        float g = gr[i];
        float silu = g / (1.0f + expf(-g));
        xr[i] = (xr[i] * inv_rms) * silu;
    }
}

// Strictly Non-Idempotent In-Place AdamW Optimizer + Multi-Step Gradient Accumulation
// If re-executed (Mode 1), w, m, v are updated TWICE -> huge numerical corruption!
__global__ void k_adamw_inplace_heavy(float *w, float *m, float *v, const float *g,
                                      int n, int inner_iters, float lr, float wd) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float wi = w[idx];
    float mi = m[idx];
    float vi = v[idx];
    float gi = g[idx];
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-6f;
    for (int it = 0; it < inner_iters; ++it) {
        float g_eff = gi + 1e-4f * sinf(wi + static_cast<float>(it) * 0.01f);
        mi = b1 * mi + (1.0f - b1) * g_eff;
        vi = b2 * vi + (1.0f - b2) * (g_eff * g_eff);
        wi = wi * (1.0f - lr * wd) - lr * mi * rsqrtf(vi + eps);
    }
    w[idx] = wi;
    m[idx] = mi;
    v[idx] = vi;
}

// HP Latency-Critical LLM Decode Step (KV-cache single-token attention + small MLP)
__global__ void k_hp_llm_decode(const float *q, const float *k_cache, const float *v_cache,
                                float *out, int kv_len, int head_dim, int n_heads) {
    const int head = blockIdx.x;
    const int tid  = threadIdx.x;
    if (head >= n_heads) return;
    __shared__ float s_dot[128];
    const float *qh = q + head * head_dim;
    const float *kh = k_cache + head * kv_len * head_dim;
    const float *vh = v_cache + head * kv_len * head_dim;

    float max_s = -1e30f, sum_e = 0.0f, acc = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
        float partial = (tid < head_dim) ? (qh[tid] * kh[t * head_dim + tid]) : 0.0f;
        s_dot[tid] = partial;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) s_dot[tid] += s_dot[tid + off];
            __syncthreads();
        }
        float score = s_dot[0] * 0.125f;
        float nm = fmaxf(max_s, score);
        float eo = expf(max_s - nm);
        float en = expf(score - nm);
        if (tid < head_dim) acc = acc * eo + en * vh[t * head_dim + tid];
        sum_e = sum_e * eo + en;
        max_s = nm;
    }
    if (tid < head_dim) out[head * head_dim + tid] = acc / (sum_e + 1e-6f);
}

struct LatencyStats {
    double p50 = 0, p95 = 0, p99 = 0, max_v = 0, mean = 0;
    double slo_2ms = 0, slo_5ms = 0;
    int count = 0;
};

LatencyStats ComputeStats(std::vector<double> vals) {
    LatencyStats st{};
    if (vals.empty()) return st;
    std::sort(vals.begin(), vals.end());
    st.count = static_cast<int>(vals.size());
    auto pct = [&](double p) -> double {
        size_t idx = static_cast<size_t>(std::ceil(p * vals.size())) - 1;
        if (idx >= vals.size()) idx = vals.size() - 1;
        return vals[idx];
    };
    st.p50   = pct(0.50);
    st.p95   = pct(0.95);
    st.p99   = pct(0.99);
    st.max_v = vals.back();
    st.mean  = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
    int c2 = 0, c5 = 0;
    for (double v : vals) {
        if (v <= 2.0) ++c2;
        if (v <= 5.0) ++c5;
    }
    st.slo_2ms = 100.0 * c2 / vals.size();
    st.slo_5ms = 100.0 * c5 / vals.size();
    return st;
}

// =========================================================================
// Workload 1: LLM Serving — LP Prefill vs HP Decode
// =========================================================================
void RunLlmServing(const std::string &mode, int hp_period_ms, int num_hp_requests) {
    int lp_prio = 0, hp_prio = 0;
    CHK_RT(cudaDeviceGetStreamPriorityRange(&lp_prio, &hp_prio));
    cudaStream_t lp_stream = nullptr, hp_stream = nullptr;
    CHK_RT(cudaStreamCreateWithPriority(&lp_stream, cudaStreamNonBlocking, lp_prio));
    CHK_RT(cudaStreamCreateWithPriority(&hp_stream, cudaStreamNonBlocking, hp_prio));

    HwQueueHandle hwq = 0;
    XQueueHandle xq = 0;
    const bool use_xq = (mode != "native");
    if (use_xq) {
        XPreemptLevel lv = (mode == "lv1") ? kPreemptLevelBlock : kPreemptLevelInterrupt;
        if (CudaQueueCreate(&hwq, lp_stream) != kXSchedSuccess ||
            XQueueCreate(&xq, hwq, lv, kQueueCreateFlagNone) != kXSchedSuccess) {
            std::fprintf(stderr, "XQueueCreate failed for mode=%s\n", mode.c_str());
            std::exit(3);
        }
    }

    cublasHandle_t blas_lp = nullptr, blas_hp = nullptr;
    CHK_BLAS(cublasCreate(&blas_lp));
    CHK_BLAS(cublasCreate(&blas_hp));
    CHK_BLAS(cublasSetStream(blas_lp, lp_stream));
    CHK_BLAS(cublasSetStream(blas_hp, hp_stream));

    // LP Prefill buffers: FP16 TC GEMM (2048x2048x2048) + Causal Attention (seq=512, heads=16, dim=64)
    const int M = 2048, K = 2048, N = 2048;
    const int seq = 512, n_heads = 16, head_dim = 64;
    const size_t gemm_elems = static_cast<size_t>(M) * N;
    const size_t attn_elems = static_cast<size_t>(n_heads) * seq * head_dim;

    __half *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC0 = nullptr;
    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr, *dO0 = nullptr, *dGate = nullptr;
    CHK_RT(cudaMalloc(&dA, gemm_elems * sizeof(__half)));
    CHK_RT(cudaMalloc(&dB, gemm_elems * sizeof(__half)));
    CHK_RT(cudaMalloc(&dC, gemm_elems * sizeof(__half)));
    CHK_RT(cudaMalloc(&dC0, gemm_elems * sizeof(__half)));
    CHK_RT(cudaMalloc(&dQ, attn_elems * sizeof(float)));
    CHK_RT(cudaMalloc(&dK, attn_elems * sizeof(float)));
    CHK_RT(cudaMalloc(&dV, attn_elems * sizeof(float)));
    CHK_RT(cudaMalloc(&dO, attn_elems * sizeof(float)));
    CHK_RT(cudaMalloc(&dO0, attn_elems * sizeof(float)));
    CHK_RT(cudaMalloc(&dGate, attn_elems * sizeof(float)));

    std::vector<__half> hA(gemm_elems), hB(gemm_elems), hC0(gemm_elems);
    for (size_t i = 0; i < gemm_elems; ++i) {
        hA[i]  = __float2half(static_cast<float>((i % 7) - 3) * 0.05f);
        hB[i]  = __float2half(static_cast<float>((i % 5) - 2) * 0.05f);
        hC0[i] = __float2half(static_cast<float>((i % 11) + 1) * 0.25f);
    }
    CHK_RT(cudaMemcpy(dA, hA.data(), gemm_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dB, hB.data(), gemm_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dC0, hC0.data(), gemm_elems * sizeof(__half), cudaMemcpyHostToDevice));

    std::vector<float> hQ(attn_elems), hK(attn_elems), hV(attn_elems), hO0(attn_elems);
    for (size_t i = 0; i < attn_elems; ++i) {
        hQ[i] = std::sin(static_cast<float>(i % 257) * 0.03f) * 0.5f;
        hK[i] = std::cos(static_cast<float>((i * 3) % 257) * 0.03f) * 0.5f;
        hV[i] = static_cast<float>((i % 13) - 6) * 0.1f;
        hO0[i] = static_cast<float>((i % 9) + 1) * 0.2f;
    }
    CHK_RT(cudaMemcpy(dQ, hQ.data(), attn_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dK, hK.data(), attn_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dV, hV.data(), attn_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dO0, hO0.data(), attn_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dGate, hQ.data(), attn_elems * sizeof(float), cudaMemcpyHostToDevice));

    // HP Decode buffers (kv_len=512, 32 heads, dim=64)
    const int hp_heads = 32, hp_kv = 512;
    float *d_hq = nullptr, *d_hk = nullptr, *d_hv = nullptr, *d_ho = nullptr;
    CHK_RT(cudaMalloc(&d_hq, hp_heads * head_dim * sizeof(float)));
    CHK_RT(cudaMalloc(&d_hk, hp_heads * hp_kv * head_dim * sizeof(float)));
    CHK_RT(cudaMalloc(&d_hv, hp_heads * hp_kv * head_dim * sizeof(float)));
    CHK_RT(cudaMalloc(&d_ho, hp_heads * head_dim * sizeof(float)));
    CHK_RT(cudaMemset(d_hq, 0, hp_heads * head_dim * sizeof(float)));
    CHK_RT(cudaMemset(d_hk, 0, hp_heads * hp_kv * head_dim * sizeof(float)));
    CHK_RT(cudaMemset(d_hv, 0, hp_heads * hp_kv * head_dim * sizeof(float)));

    const float alpha = 1.0f, beta = 1.0f; // NON-IDEMPOTENT residual projection!
    auto run_one_prefill_step = [&](cudaStream_t st, cublasHandle_t bh) {
        // 2 FP16 TC GEMM layers (beta=1.0 non-idempotent) + Causal Attention (non-idempotent residual) + RMSNorm/SwiGLU
        CHK_RT(cudaMemcpyAsync(dC, dC0, gemm_elems * sizeof(__half), cudaMemcpyDeviceToDevice, st));
        CHK_RT(cudaMemcpyAsync(dO, dO0, attn_elems * sizeof(float), cudaMemcpyDeviceToDevice, st));
        CHK_BLAS(cublasGemmEx(bh, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                              &alpha, dA, CUDA_R_16F, M, dB, CUDA_R_16F, K,
                              &beta, dC, CUDA_R_16F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        dim3 grid_attn((seq + 63) / 64, n_heads);
        k_causal_attention<<<grid_attn, 64, 0, st>>>(dQ, dK, dV, dO, seq, head_dim, n_heads);
        k_rmsnorm_swiglu_inplace<<<seq, 128, 0, st>>>(dO, dGate, seq, n_heads * head_dim);
    };

    // Standalone reference & baseline timing (on hp_stream, unmanaged)
    run_one_prefill_step(hp_stream, blas_hp);
    CHK_RT(cudaStreamSynchronize(hp_stream));
    std::vector<__half> hC_ref(gemm_elems);
    std::vector<float> hO_ref(attn_elems);
    CHK_RT(cudaMemcpy(hC_ref.data(), dC, gemm_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CHK_RT(cudaMemcpy(hO_ref.data(), dO, attn_elems * sizeof(float), cudaMemcpyDeviceToHost));

    auto t_lp_base0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 5; ++i) run_one_prefill_step(hp_stream, blas_hp);
    CHK_RT(cudaStreamSynchronize(hp_stream));
    double lp_base_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_lp_base0).count() / 5.0;

    // Warm up HP decode + LP instrumentation on lp_stream
    k_hp_llm_decode<<<hp_heads, 128, 0, hp_stream>>>(d_hq, d_hk, d_hv, d_ho, hp_kv, head_dim, hp_heads);
    CHK_RT(cudaStreamSynchronize(hp_stream));
    auto t_hp_base0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) {
        k_hp_llm_decode<<<hp_heads, 128, 0, hp_stream>>>(d_hq, d_hk, d_hv, d_ho, hp_kv, head_dim, hp_heads);
        CHK_RT(cudaStreamSynchronize(hp_stream));
    }
    double hp_base_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_hp_base0).count() / 10.0;

    // Warm up LP XQueue instrumentation so we measure steady-state serving (not first-launch compile!)
    run_one_prefill_step(lp_stream, blas_lp);
    CHK_RT(cudaStreamSynchronize(lp_stream));

    // Co-located Serving Benchmark!
    std::atomic<bool> stop{false};
    std::atomic<int> lp_steps{0};
    double max_err_gemm = 0.0, max_err_attn = 0.0;
    size_t bad_steps = 0;

    std::thread lp_thr([&]() {
        CHK_RT(cudaSetDevice(0));
        std::vector<__half> hC_cur(gemm_elems);
        std::vector<float> hO_cur(attn_elems);
        while (!stop.load(std::memory_order_acquire)) {
            run_one_prefill_step(lp_stream, blas_lp);
            CHK_RT(cudaStreamSynchronize(lp_stream));
            int step_id = lp_steps.fetch_add(1, std::memory_order_acq_rel) + 1;
            // Verify numerical exactness on every 2nd step (outside critical section)
            if (step_id <= 6 && (step_id & 1)) {
                CHK_RT(cudaMemcpy(hC_cur.data(), dC, gemm_elems * sizeof(__half), cudaMemcpyDeviceToHost));
                CHK_RT(cudaMemcpy(hO_cur.data(), dO, attn_elems * sizeof(float), cudaMemcpyDeviceToHost));
                double eg = 0.0, ea = 0.0;
                for (size_t i = 0; i < gemm_elems; i += 31) {
                    eg = std::max(eg, std::fabs(double(__half2float(hC_cur[i])) - double(__half2float(hC_ref[i]))));
                }
                for (size_t i = 0; i < attn_elems; i += 13) {
                    ea = std::max(ea, std::fabs(double(hO_cur[i]) - double(hO_ref[i])));
                }
                if (eg > max_err_gemm) max_err_gemm = eg;
                if (ea > max_err_attn) max_err_attn = ea;
                if (eg > 1e-2 || ea > 1e-2) ++bad_steps;
            }
        }
    });

    // Wait for LP to be actively running
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    std::vector<double> hp_lat_ms, susp_lat_ms;
    hp_lat_ms.reserve(num_hp_requests);
    susp_lat_ms.reserve(num_hp_requests);

    auto t_bench0 = std::chrono::steady_clock::now();
    for (int r = 0; r < num_hp_requests; ++r) {
        std::this_thread::sleep_for(std::chrono::milliseconds(hp_period_ms));
        auto t0 = std::chrono::steady_clock::now();
        if (use_xq) {
            XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue);
        }
        auto t_susp = std::chrono::steady_clock::now();
        k_hp_llm_decode<<<hp_heads, 128, 0, hp_stream>>>(d_hq, d_hk, d_hv, d_ho, hp_kv, head_dim, hp_heads);
        CHK_RT(cudaStreamSynchronize(hp_stream));
        auto t_hp_done = std::chrono::steady_clock::now();
        if (use_xq) {
            XQueueResume(xq, kQueueResumeFlagNone);
        }
        hp_lat_ms.push_back(std::chrono::duration<double, std::milli>(t_hp_done - t0).count());
        susp_lat_ms.push_back(std::chrono::duration<double, std::milli>(t_susp - t0).count());
    }
    auto t_bench1 = std::chrono::steady_clock::now();
    stop.store(true, std::memory_order_release);
    lp_thr.join();

    double total_s = std::chrono::duration<double>(t_bench1 - t_bench0).count();
    double lp_tput = lp_steps.load() / total_s;
    double lp_standalone_tput = 1000.0 / lp_base_ms;
    double lp_ret_pct = 100.0 * lp_tput / lp_standalone_tput;

    LatencyStats hs = ComputeStats(hp_lat_ms);
    LatencyStats ss = ComputeStats(susp_lat_ms);
    std::printf("[LLM_SERVING] mode=%-6s period=%2dms | base_lp=%.2fms base_hp=%.3fms | "
                "HP_ms(P50=%.3f P95=%.3f P99=%.3f Max=%.3f) SLO2ms=%.1f%% SLO5ms=%.1f%% | "
                "SUSP_ms(P50=%.3f P99=%.3f) | LP_steps=%d LP_ret=%.1f%% | "
                "err_gemm=%.2e err_attn=%.2e bad=%zu\n",
                mode.c_str(), hp_period_ms, lp_base_ms, hp_base_ms,
                hs.p50, hs.p95, hs.p99, hs.max_v, hs.slo_2ms, hs.slo_5ms,
                ss.p50, ss.p99, lp_steps.load(), lp_ret_pct,
                max_err_gemm, max_err_attn, bad_steps);

    cublasDestroy(blas_lp); cublasDestroy(blas_hp);
    if (use_xq) { XQueueDestroy(xq); HwQueueDestroy(hwq); }
    cudaStreamDestroy(lp_stream); cudaStreamDestroy(hp_stream);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC0);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dO0); cudaFree(dGate);
    cudaFree(d_hq); cudaFree(d_hk); cudaFree(d_hv); cudaFree(d_ho);
}

// =========================================================================
// Workload 2: Non-Idempotent DNN Training (In-Place AdamW) vs HP Inference
// =========================================================================
void RunDnnTraining(const std::string &mode, int hp_period_ms, int num_hp_requests) {
    int lp_prio = 0, hp_prio = 0;
    CHK_RT(cudaDeviceGetStreamPriorityRange(&lp_prio, &hp_prio));
    cudaStream_t lp_stream = nullptr, hp_stream = nullptr;
    CHK_RT(cudaStreamCreateWithPriority(&lp_stream, cudaStreamNonBlocking, lp_prio));
    CHK_RT(cudaStreamCreateWithPriority(&hp_stream, cudaStreamNonBlocking, hp_prio));

    HwQueueHandle hwq = 0;
    XQueueHandle xq = 0;
    const bool use_xq = (mode != "native");
    if (use_xq) {
        XPreemptLevel lv = (mode == "lv1") ? kPreemptLevelBlock : kPreemptLevelInterrupt;
        if (CudaQueueCreate(&hwq, lp_stream) != kXSchedSuccess ||
            XQueueCreate(&xq, hwq, lv, kQueueCreateFlagNone) != kXSchedSuccess) {
            std::fprintf(stderr, "XQueueCreate failed for mode=%s\n", mode.c_str());
            std::exit(3);
        }
    }

    // 256 blocks x 256 threads = 65536 parameters, 1600 inner AdamW steps (~12ms pure GPU compute, 100% non-idempotent!)
    const int n_params = 256 * 256;
    const int inner_iters = 1600;
    float *dW = nullptr, *dM = nullptr, *dV = nullptr, *dG = nullptr;
    float *dW0 = nullptr, *dM0 = nullptr, *dV0 = nullptr;
    CHK_RT(cudaMalloc(&dW, n_params * sizeof(float)));
    CHK_RT(cudaMalloc(&dM, n_params * sizeof(float)));
    CHK_RT(cudaMalloc(&dV, n_params * sizeof(float)));
    CHK_RT(cudaMalloc(&dG, n_params * sizeof(float)));
    CHK_RT(cudaMalloc(&dW0, n_params * sizeof(float)));
    CHK_RT(cudaMalloc(&dM0, n_params * sizeof(float)));
    CHK_RT(cudaMalloc(&dV0, n_params * sizeof(float)));

    std::vector<float> hW0(n_params), hM0(n_params), hV0(n_params), hG(n_params);
    for (int i = 0; i < n_params; ++i) {
        hW0[i] = static_cast<float>((i % 31) - 15) * 0.05f;
        hM0[i] = 0.001f * static_cast<float>(i % 7);
        hV0[i] = 0.01f + 0.001f * static_cast<float>(i % 5);
        hG[i]  = static_cast<float>((i % 19) - 9) * 0.02f;
    }
    CHK_RT(cudaMemcpy(dW0, hW0.data(), n_params * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dM0, hM0.data(), n_params * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dV0, hV0.data(), n_params * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dG,  hG.data(),  n_params * sizeof(float), cudaMemcpyHostToDevice));

    // HP Inference: cuBLAS FP16 TC batch=16 MLP (512x1024x1024)
    cublasHandle_t blas_hp = nullptr;
    CHK_BLAS(cublasCreate(&blas_hp));
    CHK_BLAS(cublasSetStream(blas_hp, hp_stream));
    __half *dhA = nullptr, *dhB = nullptr, *dhC = nullptr;
    CHK_RT(cudaMalloc(&dhA, 512 * 1024 * sizeof(__half)));
    CHK_RT(cudaMalloc(&dhB, 1024 * 1024 * sizeof(__half)));
    CHK_RT(cudaMalloc(&dhC, 512 * 1024 * sizeof(__half)));
    CHK_RT(cudaMemset(dhA, 0, 512 * 1024 * sizeof(__half)));
    CHK_RT(cudaMemset(dhB, 0, 1024 * 1024 * sizeof(__half)));
    CHK_RT(cudaMemset(dhC, 0, 512 * 1024 * sizeof(__half)));
    const float alpha = 1.0f, beta0 = 0.0f;

    auto run_adamw_step = [&](cudaStream_t st) {
        CHK_RT(cudaMemcpyAsync(dW, dW0, n_params * sizeof(float), cudaMemcpyDeviceToDevice, st));
        CHK_RT(cudaMemcpyAsync(dM, dM0, n_params * sizeof(float), cudaMemcpyDeviceToDevice, st));
        CHK_RT(cudaMemcpyAsync(dV, dV0, n_params * sizeof(float), cudaMemcpyDeviceToDevice, st));
        k_adamw_inplace_heavy<<<256, 256, 0, st>>>(dW, dM, dV, dG, n_params, inner_iters, 1e-3f, 1e-2f);
    };

    // Standalone reference
    run_adamw_step(hp_stream);
    CHK_RT(cudaStreamSynchronize(hp_stream));
    std::vector<float> hW_ref(n_params);
    CHK_RT(cudaMemcpy(hW_ref.data(), dW, n_params * sizeof(float), cudaMemcpyDeviceToHost));

    auto t_lp0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 5; ++i) run_adamw_step(hp_stream);
    CHK_RT(cudaStreamSynchronize(hp_stream));
    double lp_base_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_lp0).count() / 5.0;

    // Warm up HP + LP
    CHK_BLAS(cublasGemmEx(blas_hp, CUBLAS_OP_N, CUBLAS_OP_N, 512, 1024, 1024,
                          &alpha, dhA, CUDA_R_16F, 512, dhB, CUDA_R_16F, 1024,
                          &beta0, dhC, CUDA_R_16F, 512, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHK_RT(cudaStreamSynchronize(hp_stream));
    auto t_hp0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) {
        CHK_BLAS(cublasGemmEx(blas_hp, CUBLAS_OP_N, CUBLAS_OP_N, 512, 1024, 1024,
                              &alpha, dhA, CUDA_R_16F, 512, dhB, CUDA_R_16F, 1024,
                              &beta0, dhC, CUDA_R_16F, 512, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHK_RT(cudaStreamSynchronize(hp_stream));
    }
    double hp_base_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_hp0).count() / 10.0;

    run_adamw_step(lp_stream);
    CHK_RT(cudaStreamSynchronize(lp_stream));

    std::atomic<bool> stop{false};
    std::atomic<int> lp_steps{0};
    double max_err_w = 0.0;
    size_t mismatch_total = 0;
    int checked_steps = 0;

    std::thread lp_thr([&]() {
        CHK_RT(cudaSetDevice(0));
        std::vector<float> hW_cur(n_params);
        while (!stop.load(std::memory_order_acquire)) {
            run_adamw_step(lp_stream);
            CHK_RT(cudaStreamSynchronize(lp_stream));
            lp_steps.fetch_add(1, std::memory_order_acq_rel);
            CHK_RT(cudaMemcpy(hW_cur.data(), dW, n_params * sizeof(float), cudaMemcpyDeviceToHost));
            ++checked_steps;
            for (int i = 0; i < n_params; ++i) {
                double e = std::fabs(double(hW_cur[i]) - double(hW_ref[i]));
                if (e > max_err_w) max_err_w = e;
                if (e > 1e-6) ++mismatch_total;
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    std::vector<double> hp_lat_ms, susp_lat_ms;
    hp_lat_ms.reserve(num_hp_requests);
    susp_lat_ms.reserve(num_hp_requests);

    auto t_b0 = std::chrono::steady_clock::now();
    for (int r = 0; r < num_hp_requests; ++r) {
        std::this_thread::sleep_for(std::chrono::milliseconds(hp_period_ms));
        auto t0 = std::chrono::steady_clock::now();
        if (use_xq) XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue);
        auto t_susp = std::chrono::steady_clock::now();
        CHK_BLAS(cublasGemmEx(blas_hp, CUBLAS_OP_N, CUBLAS_OP_N, 512, 1024, 1024,
                              &alpha, dhA, CUDA_R_16F, 512, dhB, CUDA_R_16F, 1024,
                              &beta0, dhC, CUDA_R_16F, 512, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHK_RT(cudaStreamSynchronize(hp_stream));
        auto t_done = std::chrono::steady_clock::now();
        if (use_xq) XQueueResume(xq, kQueueResumeFlagNone);
        hp_lat_ms.push_back(std::chrono::duration<double, std::milli>(t_done - t0).count());
        susp_lat_ms.push_back(std::chrono::duration<double, std::milli>(t_susp - t0).count());
    }
    auto t_b1 = std::chrono::steady_clock::now();
    stop.store(true, std::memory_order_release);
    lp_thr.join();

    double total_s = std::chrono::duration<double>(t_b1 - t_b0).count();
    double lp_ret_pct = 100.0 * (lp_steps.load() / total_s) / (1000.0 / lp_base_ms);
    LatencyStats hs = ComputeStats(hp_lat_ms);
    LatencyStats ss = ComputeStats(susp_lat_ms);

    std::printf("[DNN_TRAIN  ] mode=%-6s period=%2dms | base_lp=%.2fms base_hp=%.3fms | "
                "HP_ms(P50=%.3f P95=%.3f P99=%.3f Max=%.3f) SLO2ms=%.1f%% SLO5ms=%.1f%% | "
                "SUSP_ms(P50=%.3f P99=%.3f) | LP_steps=%d LP_ret=%.1f%% | "
                "max_err_w=%.2e mismatch=%zu (checked=%d)\n",
                mode.c_str(), hp_period_ms, lp_base_ms, hp_base_ms,
                hs.p50, hs.p95, hs.p99, hs.max_v, hs.slo_2ms, hs.slo_5ms,
                ss.p50, ss.p99, lp_steps.load(), lp_ret_pct,
                max_err_w, mismatch_total, checked_steps);

    cublasDestroy(blas_hp);
    if (use_xq) { XQueueDestroy(xq); HwQueueDestroy(hwq); }
    cudaStreamDestroy(lp_stream); cudaStreamDestroy(hp_stream);
    cudaFree(dW); cudaFree(dM); cudaFree(dV); cudaFree(dG);
    cudaFree(dW0); cudaFree(dM0); cudaFree(dV0);
    cudaFree(dhA); cudaFree(dhB); cudaFree(dhC);
}

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string workload = (argc > 1) ? argv[1] : "llm";
    std::string mode     = (argc > 2) ? argv[2] : "m5";
    int period_ms        = (argc > 3) ? std::atoi(argv[3]) : 15;
    int num_req          = (argc > 4) ? std::atoi(argv[4]) : 25;

    if (workload == "llm") {
        RunLlmServing(mode, period_ms, num_req);
    } else if (workload == "dnn") {
        RunDnnTraining(mode, period_ms, num_req);
    } else {
        std::fprintf(stderr, "Unknown workload: %s (use llm|dnn)\n", workload.c_str());
        return 1;
    }
    return 0;
}
