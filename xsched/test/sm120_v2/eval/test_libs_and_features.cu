#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cufft.h>
#include <cusparse.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"

#define CHK_RT(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA RT error %d (%s) at %s:%d\n", \
                     (int)_e, cudaGetErrorString(_e), __FILE__, __LINE__); \
        return false; \
    } \
} while (0)

#define CHK_BL(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        return false; \
    } \
} while (0)

#define CHK_FFT(call) do { \
    cufftResult _r = (call); \
    if (_r != CUFFT_SUCCESS) { \
        std::fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)_r, __FILE__, __LINE__); \
        return false; \
    } \
} while (0)

#define CHK_SP(call) do { \
    cusparseStatus_t _s = (call); \
    if (_s != CUSPARSE_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuSPARSE error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        return false; \
    } \
} while (0)

#define CHK_XS(call) do { \
    XResult _r = (call); \
    if (_r != kXSchedSuccess) { \
        std::fprintf(stderr, "XSched error %d at %s:%d\n", (int)_r, __FILE__, __LINE__); \
        return false; \
    } \
} while (0)

namespace {

// High-priority canary kernel to prove SM availability while LP is suspended.
__global__ void k_hp_canary(uint32_t *d_out, uint32_t seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_out[i] = seed + static_cast<uint32_t>(i) * 7u;
}

static uint32_t *g_hp_out = nullptr;

bool VerifyHpRunsCleanly(cudaStream_t hp_stream, double *hp_ms_out = nullptr) {
    const int n = 64 * 128;
    if (g_hp_out == nullptr) {
        CHK_RT(cudaMalloc(&g_hp_out, n * sizeof(uint32_t)));
    }
    auto t0 = std::chrono::steady_clock::now();
    k_hp_canary<<<64, 128, 0, hp_stream>>>(g_hp_out, 12345u, n);
    CHK_RT(cudaStreamSynchronize(hp_stream));
    auto t1 = std::chrono::steady_clock::now();
    if (hp_ms_out) *hp_ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::vector<uint32_t> h(n);
    CHK_RT(cudaMemcpyAsync(h.data(), g_hp_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost, hp_stream));
    CHK_RT(cudaStreamSynchronize(hp_stream));
    for (int i = 0; i < n; ++i) {
        if (h[i] != 12345u + static_cast<uint32_t>(i) * 7u) return false;
    }
    return true;
}

// Helper: suspend xq mid-execution after delay_us, run HP canary, then resume xq.
struct PreemptTiming {
    double suspend_ms = 0;
    double hp_ms = 0;
    double resume_total_ms = 0;
    bool hp_ok = false;
};

template <typename LaunchFn>
bool RunWithMidPreempt(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream,
                       int delay_us, int num_preempts, LaunchFn &&launch_fn,
                       PreemptTiming *out = nullptr) {
    PreemptTiming pt{};
    std::atomic<bool> lp_submitted{false};
    std::atomic<bool> lp_done{false};
    std::atomic<bool> lp_ok{true};

    std::thread lp_thr([&]() {
        if (cudaSetDevice(0) != cudaSuccess) {
            lp_ok.store(false);
        } else if (!launch_fn()) {
            lp_ok.store(false);
        }
        lp_submitted.store(true, std::memory_order_release);
        if (cudaStreamSynchronize(lp_stream) != cudaSuccess) {
            lp_ok.store(false);
        }
        lp_done.store(true, std::memory_order_release);
    });

    while (!lp_submitted.load(std::memory_order_acquire) &&
           !lp_done.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    double sum_susp = 0, sum_hp = 0;
    if (const char *np_env = getenv("TEST_NP")) num_preempts = atoi(np_env);
    static int cached_sms = []() {
        int s = 30;
        cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, 0);
        return s > 0 ? s : 30;
    }();
    const int eff_delay_us = std::max(100, delay_us * 30 / std::max(30, cached_sms));
    int actual_preempts = 0;
    auto t_all0 = std::chrono::steady_clock::now();
    for (int p = 0; p < num_preempts; ++p) {
        std::this_thread::sleep_for(std::chrono::microseconds(eff_delay_us));
        if (lp_done.load(std::memory_order_acquire)) break;
        auto ts0 = std::chrono::steady_clock::now();
        XResult sr = XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue);
        auto ts1 = std::chrono::steady_clock::now();
        if (sr != kXSchedSuccess) {
            std::fprintf(stderr, "XQueueSuspend failed: %d\n", (int)sr);
            lp_ok.store(false);
            break;
        }
        sum_susp += std::chrono::duration<double, std::milli>(ts1 - ts0).count();
        double hp_ms = 0;
        bool hok = VerifyHpRunsCleanly(hp_stream, &hp_ms);
        sum_hp += hp_ms;
        pt.hp_ok = (p == 0) ? hok : (pt.hp_ok && hok);
        XResult rr = XQueueResume(xq, kQueueResumeFlagNone);
        if (rr != kXSchedSuccess) {
            std::fprintf(stderr, "XQueueResume failed: %d\n", (int)rr);
            lp_ok.store(false);
            break;
        }
        ++actual_preempts;
    }

    lp_thr.join();
    auto t_all1 = std::chrono::steady_clock::now();
    if (actual_preempts > 0) {
        pt.suspend_ms = sum_susp / actual_preempts;
        pt.hp_ms = sum_hp / actual_preempts;
    } else {
        pt.hp_ok = true; // finished before delay_us
    }
    pt.resume_total_ms = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();
    if (out) *out = pt;
    std::printf("    [preempts=%d/%d avg_susp=%.3fms avg_hp=%.3fms total=%.2fms hp_ok=%d]\n",
                actual_preempts, num_preempts, pt.suspend_ms, pt.hp_ms, pt.resume_total_ms, (int)pt.hp_ok);
    return lp_ok.load() && pt.hp_ok && actual_preempts > 0;
}

// =========================================================================
// LIB-01 & LIB-02: cuBLAS SGEMM (beta=0 idempotent & beta=1 non-idempotent)
// =========================================================================
bool TestCublasSgemm(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream,
                     float beta, const char *tag) {
    const int M = 3072, N = 3072, K = 3072;
    const size_t sz = static_cast<size_t>(M) * N;
    std::vector<float> hA(sz), hB(sz), hC0(sz), hC_ref(sz), hC_pre(sz);
    for (size_t i = 0; i < sz; ++i) {
        hA[i] = static_cast<float>((i * 17u + 3u) % 19u) * 0.01f - 0.09f;
        hB[i] = static_cast<float>((i * 31u + 7u) % 23u) * 0.01f - 0.11f;
        hC0[i] = static_cast<float>((i * 13u + 5u) % 17u) * 0.1f + 1.0f;
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHK_RT(cudaMalloc(&dA, sz * sizeof(float)));
    CHK_RT(cudaMalloc(&dB, sz * sizeof(float)));
    CHK_RT(cudaMalloc(&dC, sz * sizeof(float)));
    CHK_RT(cudaMemcpy(dA, hA.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dB, hB.data(), sz * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    CHK_BL(cublasCreate(&handle));
    const float alpha = 1.0f;

    // Reference on unmanaged hp_stream
    CHK_BL(cublasSetStream(handle, hp_stream));
    CHK_RT(cudaMemcpy(dC, hC0.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    CHK_BL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M, dB, K, &beta, dC, M));
    CHK_RT(cudaStreamSynchronize(hp_stream));
    CHK_RT(cudaMemcpy(hC_ref.data(), dC, sz * sizeof(float), cudaMemcpyDeviceToHost));

    // Preempted on managed lp_stream (run 4 back-to-back or single large? Single beta=1 is non-idempotent!)
    CHK_BL(cublasSetStream(handle, lp_stream));
    CHK_RT(cudaMemcpy(dC, hC0.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 2000, 2, [&]() -> bool {
        return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                           &alpha, dA, M, dB, K, &beta, dC, M) == CUBLAS_STATUS_SUCCESS;
    }, &pt);

    CHK_RT(cudaMemcpy(hC_pre.data(), dC, sz * sizeof(float), cudaMemcpyDeviceToHost));
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    double max_err = 0.0;
    size_t mismatch = 0;
    for (size_t i = 0; i < sz; ++i) {
        double e = std::fabs(static_cast<double>(hC_pre[i]) - static_cast<double>(hC_ref[i]));
        if (e > max_err) max_err = e;
        if (e > 1e-3) ++mismatch;
    }
    bool pass = ok && (mismatch == 0);
    std::printf("[%s] cuBLAS SGEMM %dx%d beta=%.1f: max_err=%.6e mismatch=%zu/%zu -> %s\n",
                tag, M, N, beta, max_err, mismatch, sz, pass ? "PASS" : "FAIL");
    return pass;
}

// =========================================================================
// LIB-03 & LIB-04: cuBLAS GemmEx FP16 / BF16 Tensor Core (beta=1.0 NON-IDEMPOTENT)
// =========================================================================
template <typename ElemT>
bool TestCublasTensorCore(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream,
                          cudaDataType_t dtype, const char *tag, const char *dtype_name) {
    const int M = 4096, N = 4096, K = 4096;
    const size_t sz = static_cast<size_t>(M) * N;
    std::vector<ElemT> hA(sz), hB(sz);
    std::vector<float> hC0(sz), hC_ref(sz), hC_pre(sz);
    for (size_t i = 0; i < sz; ++i) {
        float a = static_cast<float>((i * 11u + 3u) % 13u) * 0.02f - 0.12f;
        float b = static_cast<float>((i * 29u + 5u) % 17u) * 0.02f - 0.16f;
        hA[i] = static_cast<ElemT>(a);
        hB[i] = static_cast<ElemT>(b);
        hC0[i] = static_cast<float>((i * 7u + 1u) % 11u) * 0.25f + 2.0f;
    }

    ElemT *dA = nullptr, *dB = nullptr;
    float *dC = nullptr;
    CHK_RT(cudaMalloc(&dA, sz * sizeof(ElemT)));
    CHK_RT(cudaMalloc(&dB, sz * sizeof(ElemT)));
    CHK_RT(cudaMalloc(&dC, sz * sizeof(float)));
    CHK_RT(cudaMemcpy(dA, hA.data(), sz * sizeof(ElemT), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dB, hB.data(), sz * sizeof(ElemT), cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    CHK_BL(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 1.0f;

    // Reference
    CHK_BL(cublasSetStream(handle, hp_stream));
    CHK_RT(cudaMemcpy(dC, hC0.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    CHK_BL(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                        &alpha, dA, dtype, M, dB, dtype, K,
                        &beta, dC, CUDA_R_32F, M,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHK_RT(cudaStreamSynchronize(hp_stream));
    CHK_RT(cudaMemcpy(hC_ref.data(), dC, sz * sizeof(float), cudaMemcpyDeviceToHost));

    // Preempted on lp_stream (2 rounds of non-idempotent beta=1.0 Tensor Core GEMM)
    CHK_BL(cublasSetStream(handle, lp_stream));
    CHK_RT(cudaMemcpy(dC, hC0.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 1000, 2, [&]() -> bool {
        return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                            &alpha, dA, dtype, M, dB, dtype, K,
                            &beta, dC, CUDA_R_32F, M,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
    }, &pt);

    CHK_RT(cudaMemcpy(hC_pre.data(), dC, sz * sizeof(float), cudaMemcpyDeviceToHost));
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    double max_err = 0.0;
    size_t mismatch = 0;
    for (size_t i = 0; i < sz; ++i) {
        double e = std::fabs(static_cast<double>(hC_pre[i]) - static_cast<double>(hC_ref[i]));
        if (e > max_err) max_err = e;
        if (e > 1e-2) ++mismatch;
    }
    bool pass = ok && (mismatch == 0);
    std::printf("[%s] cuBLAS GemmEx %s TC %dx%d beta=1.0: max_err=%.6e mismatch=%zu/%zu -> %s\n",
                tag, dtype_name, M, N, max_err, mismatch, sz, pass ? "PASS" : "FAIL");
    return pass;
}

// =========================================================================
// LIB-05: cuBLASLt Matmul FP16 Tensor Core (beta=1.0 NON-IDEMPOTENT)
// =========================================================================
bool TestCublasLt(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream) {
    const int M = 4096, N = 4096, K = 4096;
    const size_t sz = static_cast<size_t>(M) * N;
    std::vector<__half> hA(sz), hB(sz);
    std::vector<float> hC0(sz), hC_ref(sz), hC_pre(sz);
    for (size_t i = 0; i < sz; ++i) {
        hA[i] = __float2half(static_cast<float>((i * 5u + 1u) % 9u) * 0.02f - 0.08f);
        hB[i] = __float2half(static_cast<float>((i * 13u + 2u) % 11u) * 0.02f - 0.10f);
        hC0[i] = static_cast<float>((i * 3u + 1u) % 7u) * 0.5f + 1.0f;
    }

    __half *dA = nullptr, *dB = nullptr;
    float *dC = nullptr;
    void *dWs = nullptr;
    const size_t ws_sz = 32 * 1024 * 1024;
    CHK_RT(cudaMalloc(&dA, sz * sizeof(__half)));
    CHK_RT(cudaMalloc(&dB, sz * sizeof(__half)));
    CHK_RT(cudaMalloc(&dC, sz * sizeof(float)));
    CHK_RT(cudaMalloc(&dWs, ws_sz));
    CHK_RT(cudaMemcpy(dA, hA.data(), sz * sizeof(__half), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dB, hB.data(), sz * sizeof(__half), cudaMemcpyHostToDevice));

    cublasLtHandle_t lt = nullptr;
    CHK_BL(cublasLtCreate(&lt));
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t aDesc = nullptr, bDesc = nullptr, cDesc = nullptr;
    CHK_BL(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHK_BL(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16F, M, K, M));
    CHK_BL(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16F, K, N, K));
    CHK_BL(cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_32F, M, N, M));

    cublasLtMatmulPreference_t pref = nullptr;
    CHK_BL(cublasLtMatmulPreferenceCreate(&pref));
    CHK_BL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_sz, sizeof(ws_sz)));
    cublasLtMatmulHeuristicResult_t heur{};
    int returned = 0;
    CHK_BL(cublasLtMatmulAlgoGetHeuristic(lt, opDesc, aDesc, bDesc, cDesc, cDesc, pref, 1, &heur, &returned));
    if (returned == 0) {
        std::fprintf(stderr, "[LIB-05] no cuBLASLt heuristic\n");
        return false;
    }

    const float alpha = 1.0f, beta = 1.0f;
    CHK_RT(cudaMemcpy(dC, hC0.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    CHK_BL(cublasLtMatmul(lt, opDesc, &alpha, dA, aDesc, dB, bDesc, &beta, dC, cDesc, dC, cDesc,
                          &heur.algo, dWs, ws_sz, hp_stream));
    CHK_RT(cudaStreamSynchronize(hp_stream));
    CHK_RT(cudaMemcpy(hC_ref.data(), dC, sz * sizeof(float), cudaMemcpyDeviceToHost));

    CHK_RT(cudaMemcpy(dC, hC0.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 1000, 2, [&]() -> bool {
        return cublasLtMatmul(lt, opDesc, &alpha, dA, aDesc, dB, bDesc, &beta, dC, cDesc, dC, cDesc,
                              &heur.algo, dWs, ws_sz, lp_stream) == CUBLAS_STATUS_SUCCESS;
    }, &pt);

    CHK_RT(cudaMemcpy(hC_pre.data(), dC, sz * sizeof(float), cudaMemcpyDeviceToHost));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(aDesc);
    cublasLtMatrixLayoutDestroy(bDesc);
    cublasLtMatrixLayoutDestroy(cDesc);
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtDestroy(lt);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dWs);

    double max_err = 0.0;
    size_t mismatch = 0;
    for (size_t i = 0; i < sz; ++i) {
        double e = std::fabs(static_cast<double>(hC_pre[i]) - static_cast<double>(hC_ref[i]));
        if (e > max_err) max_err = e;
        if (e > 1e-2) ++mismatch;
    }
    bool pass = ok && (mismatch == 0);
    std::printf("[LIB-05] cuBLASLt FP16 TC %dx%d beta=1.0: max_err=%.6e mismatch=%zu/%zu -> %s\n",
                M, N, max_err, mismatch, sz, pass ? "PASS" : "FAIL");
    return pass;
}

// =========================================================================
// LIB-06: cuFFT In-Place Batched 1D Complex FFT (Strictly NON-IDEMPOTENT!)
// =========================================================================
bool TestCufftInPlace(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream) {
    const int n_fft = 65536;
    const int batch = 256;
    const size_t total = static_cast<size_t>(n_fft) * batch;
    std::vector<cufftComplex> h_in(total), h_ref(total), h_pre(total);
    for (size_t i = 0; i < total; ++i) {
        h_in[i].x = std::sin(static_cast<float>(i % 1024) * 0.01f);
        h_in[i].y = std::cos(static_cast<float>((i * 3) % 1024) * 0.01f);
    }

    cufftComplex *d_buf = nullptr;
    CHK_RT(cudaMalloc(&d_buf, total * sizeof(cufftComplex)));

    cufftHandle plan = 0;
    CHK_FFT(cufftPlan1d(&plan, n_fft, CUFFT_C2C, batch));

    // Reference (in-place forward FFT)
    CHK_FFT(cufftSetStream(plan, hp_stream));
    CHK_RT(cudaMemcpy(d_buf, h_in.data(), total * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    CHK_FFT(cufftExecC2C(plan, d_buf, d_buf, CUFFT_FORWARD));
    CHK_RT(cudaStreamSynchronize(hp_stream));
    CHK_RT(cudaMemcpy(h_ref.data(), d_buf, total * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    // Preempted in-place forward FFT
    CHK_FFT(cufftSetStream(plan, lp_stream));
    CHK_RT(cudaMemcpy(d_buf, h_in.data(), total * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 800, 2, [&]() -> bool {
        return cufftExecC2C(plan, d_buf, d_buf, CUFFT_FORWARD) == CUFFT_SUCCESS;
    }, &pt);

    CHK_RT(cudaMemcpy(h_pre.data(), d_buf, total * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    cufftDestroy(plan);
    cudaFree(d_buf);

    double max_err = 0.0;
    size_t mismatch = 0;
    for (size_t i = 0; i < total; ++i) {
        double ex = std::fabs(h_pre[i].x - h_ref[i].x);
        double ey = std::fabs(h_pre[i].y - h_ref[i].y);
        double e = std::max(ex, ey);
        if (e > max_err) max_err = e;
        if (e > 1e-3) ++mismatch;
    }
    bool pass = ok && (mismatch == 0);
    std::printf("[LIB-06] cuFFT In-Place C2C (%d x %d): max_err=%.6e mismatch=%zu/%zu -> %s\n",
                n_fft, batch, max_err, mismatch, total, pass ? "PASS" : "FAIL");
    return pass;
}

// =========================================================================
// LIB-07: cuSPARSE SpMM (CSR x Dense, beta=1.0 NON-IDEMPOTENT)
// =========================================================================
bool TestCusparseSpMM(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream) {
    const int M = 4096, K = 4096, N = 512;
    const int nnz_per_row = 128;
    const int nnz = M * nnz_per_row;
    std::vector<int> h_row(M + 1), h_col(nnz);
    std::vector<float> h_val(nnz), hB(static_cast<size_t>(K) * N), hC0(static_cast<size_t>(M) * N);
    for (int r = 0; r < M; ++r) {
        h_row[r] = r * nnz_per_row;
        for (int j = 0; j < nnz_per_row; ++j) {
            int idx = r * nnz_per_row + j;
            h_col[idx] = (r * 31 + j * 29) % K;
            h_val[idx] = static_cast<float>((idx % 13) + 1) * 0.01f;
        }
    }
    h_row[M] = nnz;
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = static_cast<float>((i % 19) + 1) * 0.02f;
    for (size_t i = 0; i < hC0.size(); ++i) hC0[i] = static_cast<float>((i % 11) + 1) * 0.5f;

    int *d_row = nullptr, *d_col = nullptr;
    float *d_val = nullptr, *dB = nullptr, *dC = nullptr;
    CHK_RT(cudaMalloc(&d_row, (M + 1) * sizeof(int)));
    CHK_RT(cudaMalloc(&d_col, nnz * sizeof(int)));
    CHK_RT(cudaMalloc(&d_val, nnz * sizeof(float)));
    CHK_RT(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CHK_RT(cudaMalloc(&dC, hC0.size() * sizeof(float)));
    CHK_RT(cudaMemcpy(d_row, h_row.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(d_col, h_col.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(d_val, h_val.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    cusparseHandle_t sp = nullptr;
    CHK_SP(cusparseCreate(&sp));
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr, matC = nullptr;
    CHK_SP(cusparseCreateCsr(&matA, M, K, nnz, d_row, d_col, d_val,
                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHK_SP(cusparseCreateDnMat(&matB, K, N, N, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHK_SP(cusparseCreateDnMat(&matC, M, N, N, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    const float alpha = 1.0f, beta = 1.0f;
    size_t buf_sz = 0;
    CHK_SP(cusparseSpMM_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                   CUSPARSE_SPMM_ALG_DEFAULT, &buf_sz));
    void *d_buf = nullptr;
    if (buf_sz > 0) CHK_RT(cudaMalloc(&d_buf, buf_sz));

    std::vector<float> h_ref(hC0.size()), h_pre(hC0.size());
    CHK_SP(cusparseSetStream(sp, hp_stream));
    CHK_RT(cudaMemcpy(dC, hC0.data(), hC0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHK_SP(cusparseSpMM(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                        CUSPARSE_SPMM_ALG_DEFAULT, d_buf));
    CHK_RT(cudaStreamSynchronize(hp_stream));
    CHK_RT(cudaMemcpy(h_ref.data(), dC, hC0.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CHK_SP(cusparseSetStream(sp, lp_stream));
    CHK_RT(cudaMemcpy(dC, hC0.data(), hC0.size() * sizeof(float), cudaMemcpyHostToDevice));
    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 500, 2, [&]() -> bool {
        return cusparseSpMM(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                            CUSPARSE_SPMM_ALG_DEFAULT, d_buf) == CUSPARSE_STATUS_SUCCESS;
    }, &pt);

    CHK_RT(cudaMemcpy(h_pre.data(), dC, hC0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(sp);
    if (d_buf) cudaFree(d_buf);
    cudaFree(d_row); cudaFree(d_col); cudaFree(d_val); cudaFree(dB); cudaFree(dC);

    double max_err = 0.0;
    size_t mismatch = 0;
    for (size_t i = 0; i < h_ref.size(); ++i) {
        double e = std::fabs(h_pre[i] - h_ref[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-3) ++mismatch;
    }
    bool pass = ok && (mismatch == 0);
    std::printf("[LIB-07] cuSPARSE SpMM (%dx%d nnz=%d beta=1.0): max_err=%.6e mismatch=%zu/%zu -> %s\n",
                M, N, nnz, max_err, mismatch, h_ref.size(), pass ? "PASS" : "FAIL");
    return pass;
}

// =========================================================================
// LIB-08: CUB DeviceRadixSort::SortPairs (8M elements)
// =========================================================================
bool TestCubRadixSort(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream) {
    const int N = 8 * 1024 * 1024;
    std::vector<uint32_t> h_k(N), h_v(N), h_k_ref(N), h_v_ref(N), h_k_pre(N), h_v_pre(N);
    uint32_t rng = 0x12345678u;
    for (int i = 0; i < N; ++i) {
        rng = rng * 1664525u + 1013904223u;
        h_k[i] = rng;
        h_v[i] = static_cast<uint32_t>(i);
    }

    uint32_t *d_kin = nullptr, *d_kout = nullptr, *d_vin = nullptr, *d_vout = nullptr;
    CHK_RT(cudaMalloc(&d_kin, N * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&d_kout, N * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&d_vin, N * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&d_vout, N * sizeof(uint32_t)));
    CHK_RT(cudaMemcpy(d_kin, h_k.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(d_vin, h_v.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    size_t temp_bytes = 0;
    CHK_RT(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, d_kin, d_kout, d_vin, d_vout, N, 0, 32, hp_stream));
    void *d_temp = nullptr;
    CHK_RT(cudaMalloc(&d_temp, temp_bytes));

    // Ref
    CHK_RT(cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_kin, d_kout, d_vin, d_vout, N, 0, 32, hp_stream));
    CHK_RT(cudaStreamSynchronize(hp_stream));
    CHK_RT(cudaMemcpy(h_k_ref.data(), d_kout, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHK_RT(cudaMemcpy(h_v_ref.data(), d_vout, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Preempted
    CHK_RT(cudaMemset(d_kout, 0, N * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_vout, 0, N * sizeof(uint32_t)));
    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 500, 2, [&]() -> bool {
        return cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_kin, d_kout, d_vin, d_vout,
                                               N, 0, 32, lp_stream) == cudaSuccess;
    }, &pt);

    CHK_RT(cudaMemcpy(h_k_pre.data(), d_kout, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHK_RT(cudaMemcpy(h_v_pre.data(), d_vout, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_temp); cudaFree(d_kin); cudaFree(d_kout); cudaFree(d_vin); cudaFree(d_vout);

    size_t mismatch = 0;
    for (int i = 0; i < N; ++i) {
        if (h_k_pre[i] != h_k_ref[i] || h_v_pre[i] != h_v_ref[i]) ++mismatch;
    }
    bool pass = ok && (mismatch == 0);
    std::printf("[LIB-08] CUB DeviceRadixSort (8M pairs): mismatch=%zu/%d -> %s\n",
                mismatch, N, pass ? "PASS" : "FAIL");
    return pass;
}

// =========================================================================
// HW-01: Explicit PTX Tensor Core MMA (mma.sync.aligned.m16n8k16)
// =========================================================================
__global__ void k_ptx_mma_nonidemp(float *d_out, unsigned long long *d_counter, int iters) {
    const int lane = threadIdx.x & 31;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    // Pack two fp16 values per uint32_t
    __half h1 = __float2half(0.25f + (lane & 3) * 0.05f);
    __half h2 = __float2half(0.50f - (lane & 3) * 0.03f);
    uint32_t a0 = *reinterpret_cast<uint32_t *>(&h1);
    __half2 hh = __halves2half2(h1, h2);
    uint32_t pack = *reinterpret_cast<uint32_t *>(&hh);
    a0 = pack;
    uint32_t a1 = pack, a2 = pack, a3 = pack;
    uint32_t b0 = pack, b1 = pack;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    for (int it = 0; it < iters; ++it) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        if ((it & 63) == 0 && lane == 0) {
            atomicAdd(d_counter, 1ULL); // non-idempotent side-effect!
        }
    }
    int base = (warp_global * 32 + lane) * 4;
    d_out[base + 0] = d0;
    d_out[base + 1] = d1;
    d_out[base + 2] = d2;
    d_out[base + 3] = d3;
}

bool TestHwPtxMma(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream) {
    const int blocks = 120, threads = 128, iters = 60000;
    const size_t n_floats = static_cast<size_t>(blocks) * threads * 4;
    float *d_out = nullptr;
    unsigned long long *d_cnt = nullptr;
    CHK_RT(cudaMalloc(&d_out, n_floats * sizeof(float)));
    CHK_RT(cudaMalloc(&d_cnt, sizeof(unsigned long long)));

    std::vector<float> h_ref(n_floats), h_pre(n_floats);
    unsigned long long cnt_ref = 0, cnt_pre = 0;

    CHK_RT(cudaMemset(d_cnt, 0, sizeof(unsigned long long)));
    k_ptx_mma_nonidemp<<<blocks, threads, 0, hp_stream>>>(d_out, d_cnt, iters);
    CHK_RT(cudaStreamSynchronize(hp_stream));
    CHK_RT(cudaMemcpy(h_ref.data(), d_out, n_floats * sizeof(float), cudaMemcpyDeviceToHost));
    CHK_RT(cudaMemcpy(&cnt_ref, d_cnt, sizeof(cnt_ref), cudaMemcpyDeviceToHost));

    CHK_RT(cudaMemset(d_cnt, 0, sizeof(unsigned long long)));
    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 2000, 3, [&]() -> bool {
        k_ptx_mma_nonidemp<<<blocks, threads, 0, lp_stream>>>(d_out, d_cnt, iters);
        return cudaGetLastError() == cudaSuccess;
    }, &pt);
    CHK_RT(cudaMemcpy(h_pre.data(), d_out, n_floats * sizeof(float), cudaMemcpyDeviceToHost));
    CHK_RT(cudaMemcpy(&cnt_pre, d_cnt, sizeof(cnt_pre), cudaMemcpyDeviceToHost));
    cudaFree(d_out); cudaFree(d_cnt);

    size_t mismatch = 0;
    for (size_t i = 0; i < n_floats; ++i) {
        if (h_pre[i] != h_ref[i]) ++mismatch; // exact bitwise float match!
    }
    bool pass = ok && (mismatch == 0) && (cnt_pre == cnt_ref);
    std::printf("[HW-01] PTX TensorCore mma.sync (non-idemp): mismatch=%zu/%zu cnt=%llu/%llu -> %s\n",
                mismatch, n_floats, cnt_pre, cnt_ref, pass ? "PASS" : "FAIL");
    return pass;
}

// =========================================================================
// HW-02: Explicit Async Copy cp.async (LDGSTS) + wait_group
// =========================================================================
__global__ void k_ptx_cp_async(const uint32_t *d_in, uint32_t *d_out, int iters) {
    __shared__ uint32_t smem[2][128];
    const int tid = threadIdx.x;
    const int gbase = blockIdx.x * 128;
    uint32_t acc = 0;
    for (int it = 0; it < iters; ++it) {
        int buf = it & 1;
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[buf][tid]));
        const void *gptr = &d_in[gbase + ((tid + it) & 127)];
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(smem_addr), "l"(gptr));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        acc = (acc * 33u) ^ smem[buf][(tid + 17) & 127];
        __syncthreads();
    }
    d_out[gbase + tid] = acc;
}

bool TestHwCpAsync(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream) {
    const int blocks = 120, threads = 128, iters = 120000;
    const size_t n = static_cast<size_t>(blocks) * threads;
    std::vector<uint32_t> h_in(n), h_ref(n), h_pre(n);
    for (size_t i = 0; i < n; ++i) h_in[i] = static_cast<uint32_t>(i * 2654435761u + 19u);

    uint32_t *d_in = nullptr, *d_out = nullptr;
    CHK_RT(cudaMalloc(&d_in, n * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&d_out, n * sizeof(uint32_t)));
    CHK_RT(cudaMemcpy(d_in, h_in.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    k_ptx_cp_async<<<blocks, threads, 0, hp_stream>>>(d_in, d_out, iters);
    CHK_RT(cudaStreamSynchronize(hp_stream));
    CHK_RT(cudaMemcpy(h_ref.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 2000, 3, [&]() -> bool {
        k_ptx_cp_async<<<blocks, threads, 0, lp_stream>>>(d_in, d_out, iters);
        return cudaGetLastError() == cudaSuccess;
    }, &pt);
    CHK_RT(cudaMemcpy(h_pre.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out);

    size_t mismatch = 0;
    for (size_t i = 0; i < n; ++i) if (h_pre[i] != h_ref[i]) ++mismatch;
    bool pass = ok && (mismatch == 0);
    std::printf("[HW-02] PTX cp.async (LDGSTS) pipeline: mismatch=%zu/%zu -> %s\n",
                mismatch, n, pass ? "PASS" : "FAIL");
    return pass;
}

// =========================================================================
// HW-03: Explicit SMEM mbarrier (mbarrier.init / arrive / try_wait.parity)
// Double-buffered SMEM to ensure zero intra-iteration data race.
// =========================================================================
__global__ void k_ptx_mbarrier(uint32_t *d_out, int iters) {
    __shared__ __align__(8) uint64_t mbar[2];
    __shared__ uint32_t s_data[2][128];
    const int tid = threadIdx.x;
    uint32_t mbar0 = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[0]));
    uint32_t mbar1 = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[1]));
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(mbar0), "r"(128));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(mbar1), "r"(128));
    }
    __syncthreads();

    uint32_t acc = static_cast<uint32_t>(blockIdx.x * 131 + tid * 7);
    uint32_t phase[2] = {0, 0};
    for (int it = 0; it < iters; ++it) {
        int b = it & 1;
        uint32_t ma = (b == 0) ? mbar0 : mbar1;
        s_data[b][tid] = acc ^ static_cast<uint32_t>(it);
        uint64_t tok = 0;
        asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(tok) : "r"(ma));
        uint32_t ready = 0;
        while (!ready) {
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n\t"
                "selp.u32 %0, 1, 0, p;\n\t"
                "}"
                : "=r"(ready) : "r"(ma), "r"(phase[b]));
        }
        phase[b] ^= 1u;
        acc = (acc * 31u) + s_data[b][(tid + 33) & 127];
    }
    d_out[blockIdx.x * 128 + tid] = acc;
}

bool TestHwMbarrier(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream) {
    const int blocks = 120, threads = 128, iters = 150000;
    const size_t n = static_cast<size_t>(blocks) * threads;
    uint32_t *d_out = nullptr;
    CHK_RT(cudaMalloc(&d_out, n * sizeof(uint32_t)));
    std::vector<uint32_t> h_ref(n), h_pre(n);

    k_ptx_mbarrier<<<blocks, threads, 0, hp_stream>>>(d_out, iters);
    CHK_RT(cudaStreamSynchronize(hp_stream));
    CHK_RT(cudaMemcpy(h_ref.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 2000, 3, [&]() -> bool {
        k_ptx_mbarrier<<<blocks, threads, 0, lp_stream>>>(d_out, iters);
        return cudaGetLastError() == cudaSuccess;
    }, &pt);
    CHK_RT(cudaMemcpy(h_pre.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    size_t mismatch = 0;
    for (size_t i = 0; i < n; ++i) if (h_pre[i] != h_ref[i]) ++mismatch;
    bool pass = ok && (mismatch == 0);
    std::printf("[HW-03] PTX SMEM mbarrier (arrive/try_wait): mismatch=%zu/%zu -> %s\n",
                mismatch, n, pass ? "PASS" : "FAIL");
    return pass;
}

// =========================================================================
// HW-04: CUDA Graph Capture + Launch under XSched Level-3 (Fallback Check)
// =========================================================================
__global__ void k_graph_step(float *d_x, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = d_x[i];
    for (int k = 0; k < iters; ++k) v = fmaf(v, 1.000001f, 0.0001f);
    d_x[i] = v;
}

bool TestHwCudaGraph(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream) {
    const int n = 256 * 128;
    float *d_x = nullptr;
    CHK_RT(cudaMalloc(&d_x, n * sizeof(float)));
    std::vector<float> h0(n, 1.0f), h_ref(n), h_pre(n);

    // Capture a 4-node graph (~25ms total) on hp_stream
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    CHK_RT(cudaStreamBeginCapture(hp_stream, cudaStreamCaptureModeGlobal));
    for (int s = 0; s < 4; ++s) {
        k_graph_step<<<256, 128, 0, hp_stream>>>(d_x, n, 1200000);
    }
    CHK_RT(cudaStreamEndCapture(hp_stream, &graph));
    CHK_RT(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    CHK_RT(cudaMemcpy(d_x, h0.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    auto tg0 = std::chrono::steady_clock::now();
    CHK_RT(cudaGraphLaunch(exec, hp_stream));
    CHK_RT(cudaStreamSynchronize(hp_stream));
    auto tg1 = std::chrono::steady_clock::now();
    double graph_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
    CHK_RT(cudaMemcpy(h_ref.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Launch graph on managed lp_stream and try mid-suspend (expected Level-1 fallback!)
    CHK_RT(cudaMemcpy(d_x, h0.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 2000, 1, [&]() -> bool {
        return cudaGraphLaunch(exec, lp_stream) == cudaSuccess;
    }, &pt);
    CHK_RT(cudaMemcpy(h_pre.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaFree(d_x);

    double max_err = 0.0;
    for (int i = 0; i < n; ++i) max_err = std::max(max_err, std::fabs(double(h_pre[i]) - double(h_ref[i])));
    std::printf("[HW-04] CUDA Graph (4-node, %.2fms): max_err=%.6e susp=%.3fms (Lv1 boundary fallback!) -> %s\n",
                graph_ms, max_err, pt.suspend_ms, (ok && max_err == 0.0) ? "PASS (Lv1-fallback)" : "FAIL");
    return ok && (max_err == 0.0);
}

// =========================================================================
// HW-05: Idle Queue Suspend/Resume Safety (Regression for Defect #1)
// =========================================================================
bool TestHwIdleSuspendSafety(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream) {
    (void)hp_stream;
    // Suspend & resume when stream is completely idle, then run a real kernel!
    CHK_XS(XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue));
    CHK_XS(XQueueResume(xq, kQueueResumeFlagNone));

    const int n = 4096;
    uint32_t *d_out = nullptr;
    CHK_RT(cudaMalloc(&d_out, n * sizeof(uint32_t)));
    CHK_RT(cudaMemset(d_out, 0, n * sizeof(uint32_t)));
    k_hp_canary<<<32, 128, 0, lp_stream>>>(d_out, 777u, n);
    CHK_RT(cudaStreamSynchronize(lp_stream));
    std::vector<uint32_t> h(n);
    CHK_RT(cudaMemcpy(h.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    size_t mismatch = 0;
    for (int i = 0; i < n; ++i) if (h[i] != 777u + static_cast<uint32_t>(i) * 7u) ++mismatch;
    bool pass = (mismatch == 0);
    std::printf("[HW-05] Idle Queue Suspend/Resume Safety: mismatch=%zu/%d -> %s\n",
                mismatch, n, pass ? "PASS" : "FAIL");
    return pass;
}

// =========================================================================
// HW-06: Blackwell CTA Cluster (__cluster_dims__(2,1,1) + DSMEM + cluster.sync)
// =========================================================================
__global__ void __cluster_dims__(2, 1, 1) k_cluster_dsmem(uint32_t *d_out, int iters) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    __shared__ uint32_t smem[128];
    const int tid = threadIdx.x;
    const int rank = cluster.block_rank();
    const int peer = rank ^ 1;
    uint32_t acc = static_cast<uint32_t>(blockIdx.x * 17 + tid * 3 + 1);

    for (int it = 0; it < iters; ++it) {
        smem[tid] = acc ^ static_cast<uint32_t>(it);
        cluster.sync();
        uint32_t *peer_smem = cluster.map_shared_rank(smem, peer);
        uint32_t v = peer_smem[(tid + 11) & 127];
        cluster.sync();
        acc = (acc * 33u) + v;
    }
    d_out[blockIdx.x * 128 + tid] = acc;
}

bool TestHwClusterDsmem(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream) {
    const int blocks = 60, threads = 128, iters = 30000;
    const size_t n = static_cast<size_t>(blocks) * threads;
    uint32_t *d_out = nullptr;
    CHK_RT(cudaMalloc(&d_out, n * sizeof(uint32_t)));
    std::vector<uint32_t> h_ref(n), h_pre(n);

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(blocks, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 2; attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;

    cfg.stream = hp_stream;
    CHK_RT(cudaLaunchKernelEx(&cfg, k_cluster_dsmem, d_out, iters));
    CHK_RT(cudaStreamSynchronize(hp_stream));
    CHK_RT(cudaMemcpy(h_ref.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cfg.stream = lp_stream;
    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 2000, 1, [&]() -> bool {
        return cudaLaunchKernelEx(&cfg, k_cluster_dsmem, d_out, iters) == cudaSuccess;
    }, &pt);
    cudaError_t ce = cudaMemcpy(h_pre.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    size_t mismatch = 0;
    if (ce == cudaSuccess) {
        for (size_t i = 0; i < n; ++i) if (h_pre[i] != h_ref[i]) ++mismatch;
    } else {
        mismatch = n;
    }
    bool pass = ok && (ce == cudaSuccess) && (mismatch == 0);
    std::printf("[HW-06] Blackwell CTA Cluster (2x1x1 + DSMEM): ce=%d mismatch=%zu/%zu -> %s\n",
                (int)ce, mismatch, n, pass ? "PASS" : "FAIL");
    return pass;
}

// =========================================================================
// HW-07: Cooperative Groups (cudaLaunchCooperativeKernel + grid.sync())
// Tests Shim coverage boundary (cuLaunchCooperativeKernel currently bypasses XSched).
// =========================================================================
__global__ void k_coop_grid_sync(uint32_t *d_buf, int iters) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    uint32_t acc = static_cast<uint32_t>(idx + 1);
    for (int it = 0; it < iters; ++it) {
        d_buf[idx] = acc;
        grid.sync();
        acc = (acc * 31u) ^ d_buf[(idx + 137) % total];
        grid.sync();
    }
    d_buf[idx] = acc;
}

bool TestHwCoopGroups(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream) {
    const int blocks = 30, threads = 128, iters = 15000;
    const size_t n = static_cast<size_t>(blocks) * threads;
    uint32_t *d_buf = nullptr;
    CHK_RT(cudaMalloc(&d_buf, n * sizeof(uint32_t)));
    std::vector<uint32_t> h_ref(n), h_pre(n);

    int it_arg = iters;
    void *args[] = {&d_buf, &it_arg};
    CHK_RT(cudaLaunchCooperativeKernel((void *)k_coop_grid_sync, dim3(blocks), dim3(threads), args, 0, hp_stream));
    CHK_RT(cudaStreamSynchronize(hp_stream));
    CHK_RT(cudaMemcpy(h_ref.data(), d_buf, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 2000, 1, [&]() -> bool {
        return cudaLaunchCooperativeKernel((void *)k_coop_grid_sync, dim3(blocks), dim3(threads), args, 0, lp_stream) == cudaSuccess;
    }, &pt);
    CHK_RT(cudaMemcpy(h_pre.data(), d_buf, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_buf);

    size_t mismatch = 0;
    for (size_t i = 0; i < n; ++i) if (h_pre[i] != h_ref[i]) ++mismatch;
    std::printf("[HW-07] Cooperative Groups (grid.sync): mismatch=%zu/%zu susp=%.3fms (Shim bypass check) -> %s\n",
                mismatch, n, pt.suspend_ms, (ok && mismatch == 0) ? "PASS (Shim-bypass)" : "FAIL");
    return ok && (mismatch == 0);
}

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string filter = (argc > 1) ? argv[1] : "all";

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceProperties failed\n");
        return 1;
    }
    std::printf("[SM120_V2_LIB_ARCH_SUITE] GPU=%s SM=%d.%d SMs=%d filter=%s\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount, filter.c_str());

    cudaStream_t lp_stream = nullptr, hp_stream = nullptr;
    if (cudaStreamCreateWithFlags(&lp_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&hp_stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamCreate failed\n");
        return 1;
    }

    HwQueueHandle hwq = 0;
    XQueueHandle xq = 0;
    if (CudaQueueCreate(&hwq, lp_stream) != kXSchedSuccess ||
        XQueueCreate(&xq, hwq, kPreemptLevelInterrupt, kQueueCreateFlagNone) != kXSchedSuccess) {
        std::fprintf(stderr, "XQueueCreate(Level 3 Interrupt) failed\n");
        return 1;
    }

    int passed = 0, total = 0;
    auto run_case = [&](const char *id, auto fn) {
        if (filter != "all" && filter != id) return;
        ++total;
        if (fn()) ++passed;
    };

    run_case("LIB-01", [&]() { return TestCublasSgemm(xq, lp_stream, hp_stream, 0.0f, "LIB-01"); });
    run_case("LIB-02", [&]() { return TestCublasSgemm(xq, lp_stream, hp_stream, 1.0f, "LIB-02"); });
    run_case("LIB-03", [&]() { return TestCublasTensorCore<__half>(xq, lp_stream, hp_stream, CUDA_R_16F, "LIB-03", "FP16"); });
    run_case("LIB-04", [&]() { return TestCublasTensorCore<__nv_bfloat16>(xq, lp_stream, hp_stream, CUDA_R_16BF, "LIB-04", "BF16"); });
    run_case("LIB-05", [&]() { return TestCublasLt(xq, lp_stream, hp_stream); });
    run_case("LIB-06", [&]() { return TestCufftInPlace(xq, lp_stream, hp_stream); });
    run_case("LIB-07", [&]() { return TestCusparseSpMM(xq, lp_stream, hp_stream); });
    run_case("LIB-08", [&]() { return TestCubRadixSort(xq, lp_stream, hp_stream); });
    run_case("HW-01",  [&]() { return TestHwPtxMma(xq, lp_stream, hp_stream); });
    run_case("HW-02",  [&]() { return TestHwCpAsync(xq, lp_stream, hp_stream); });
    run_case("HW-03",  [&]() { return TestHwMbarrier(xq, lp_stream, hp_stream); });
    run_case("HW-04",  [&]() { return TestHwCudaGraph(xq, lp_stream, hp_stream); });
    run_case("HW-05",  [&]() { return TestHwIdleSuspendSafety(xq, lp_stream, hp_stream); });
    run_case("HW-06",  [&]() { return TestHwClusterDsmem(xq, lp_stream, hp_stream); });
    run_case("HW-07",  [&]() { return TestHwCoopGroups(xq, lp_stream, hp_stream); });

    std::printf("\n=== LIB & HW FEATURE SUMMARY: %d / %d PASSED ===\n", passed, total);
    XQueueDestroy(xq);
    HwQueueDestroy(hwq);
    cudaStreamDestroy(hp_stream);
    cudaStreamDestroy(lp_stream);
    return (passed == total) ? 0 : 1;
}
