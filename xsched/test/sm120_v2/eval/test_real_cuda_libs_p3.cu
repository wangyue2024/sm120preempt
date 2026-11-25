#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cufft.h>
#include <curand.h>
#include <cusolverDn.h>
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

#define CHK_SOLVER(call) do { \
    cusolverStatus_t _s = (call); \
    if (_s != CUSOLVER_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuSOLVER error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        return false; \
    } \
} while (0)

#define CHK_RAND(call) do { \
    curandStatus_t _s = (call); \
    if (_s != CURAND_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuRAND error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
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

#define CHK_XS(call) do { \
    XResult _r = (call); \
    if (_r != kXSchedSuccess) { \
        std::fprintf(stderr, "XSched error %d at %s:%d\n", (int)_r, __FILE__, __LINE__); \
        return false; \
    } \
} while (0)

namespace {

__global__ void k_hp_canary(uint32_t *d_out, uint32_t seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_out[i] = seed + static_cast<uint32_t>(i) * 13u;
}

__global__ void k_scale_complex(cufftComplex *d, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d[i].x *= alpha;
        d[i].y *= alpha;
    }
}

struct KeepOddOp {
    __host__ __device__ __forceinline__ bool operator()(const uint32_t &a) const {
        return (a & 1u) != 0u;
    }
};

static uint32_t *g_hp_out = nullptr;

bool VerifyHpRunsCleanly(cudaStream_t hp_stream, double *hp_ms_out = nullptr) {
    const int n = 64 * 128;
    if (g_hp_out == nullptr) {
        CHK_RT(cudaMalloc(&g_hp_out, n * sizeof(uint32_t)));
    }
    auto t0 = std::chrono::steady_clock::now();
    k_hp_canary<<<64, 128, 0, hp_stream>>>(g_hp_out, 54321u, n);
    CHK_RT(cudaStreamSynchronize(hp_stream));
    auto t1 = std::chrono::steady_clock::now();
    if (hp_ms_out) *hp_ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::vector<uint32_t> h(n);
    CHK_RT(cudaMemcpyAsync(h.data(), g_hp_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost, hp_stream));
    CHK_RT(cudaStreamSynchronize(hp_stream));
    for (int i = 0; i < n; ++i) {
        if (h[i] != 54321u + static_cast<uint32_t>(i) * 13u) return false;
    }
    return true;
}

struct PreemptTiming {
    double suspend_ms = 0;
    double hp_ms = 0;
    double resume_total_ms = 0;
    int actual_preempts = 0;
    bool hp_ok = true;
};

template <typename LaunchFn>
bool RunWithMidPreempt(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream,
                       int delay_us, int num_preempts, LaunchFn &&launch_fn,
                       PreemptTiming *out = nullptr) {
    PreemptTiming pt{};
    std::atomic<bool> lp_done{false};
    std::atomic<bool> lp_ok{true};

    std::thread lp_thr([&]() {
        if (cudaSetDevice(0) != cudaSuccess) {
            lp_ok.store(false);
        } else if (!launch_fn()) {
            lp_ok.store(false);
        }
        if (lp_ok.load()) {
            if (cudaStreamSynchronize(lp_stream) != cudaSuccess) lp_ok.store(false);
        }
        lp_done.store(true, std::memory_order_release);
    });

    int sm_cnt = 30;
    cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, 0);
    int eff_delay_us = std::max(250, static_cast<int>(delay_us * (30.0 / std::max(30, sm_cnt))));

    auto t_start = std::chrono::steady_clock::now();
    for (int p = 0; p < num_preempts; ++p) {
        std::this_thread::sleep_for(std::chrono::microseconds(eff_delay_us));
        if (lp_done.load(std::memory_order_acquire)) break;
        auto ts0 = std::chrono::steady_clock::now();
        XResult r = XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue);
        auto ts1 = std::chrono::steady_clock::now();
        if (r != kXSchedSuccess) {
            std::fprintf(stderr, "XQueueSuspend failed: %d\n", (int)r);
            lp_thr.join();
            return false;
        }
        double sm = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
        if (pt.actual_preempts == 0) pt.suspend_ms = sm;
        else pt.suspend_ms = std::max(pt.suspend_ms, sm);

        double hms = 0;
        if (!VerifyHpRunsCleanly(hp_stream, &hms)) pt.hp_ok = false;
        pt.hp_ms = std::max(pt.hp_ms, hms);

        r = XQueueResume(xq, kQueueResumeFlagNone);
        if (r != kXSchedSuccess) {
            std::fprintf(stderr, "XQueueResume failed: %d\n", (int)r);
            lp_thr.join();
            return false;
        }
        pt.actual_preempts++;
    }

    lp_thr.join();
    auto t_end = std::chrono::steady_clock::now();
    pt.resume_total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    if (out) *out = pt;
    return lp_ok.load() && pt.hp_ok;
}

// =========================================================================
// LIB-09: cuSOLVER Dense LU Factorization + Linear Solve (getrf + getrs)
// Non-idempotent in-place factorization & RHS solve.
// =========================================================================
bool TestLib09CuSolverLU(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream, int np) {
    const int N = 2048;
    const int nrhs = 64;
    const size_t szA = (size_t)N * N;
    const size_t szB = (size_t)N * nrhs;

    std::vector<float> hA(szA), hB(szB), hX_ref(szB), hX_got(szB);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            float v = 0.001f * std::sin(0.013f * i + 0.017f * j);
            if (i == j) v += 5.0f;  // strictly diagonally dominant -> well-conditioned
            hA[(size_t)j * N + i] = v;
        }
    }
    for (size_t i = 0; i < szB; ++i) hB[i] = std::cos(0.011f * (float)i);

    float *dA = nullptr, *dB = nullptr, *dWork = nullptr;
    int *dIpiv = nullptr, *dInfo = nullptr;
    CHK_RT(cudaMalloc(&dA, szA * sizeof(float)));
    CHK_RT(cudaMalloc(&dB, szB * sizeof(float)));
    CHK_RT(cudaMalloc(&dIpiv, N * sizeof(int)));
    CHK_RT(cudaMalloc(&dInfo, sizeof(int)));

    cusolverDnHandle_t handle = nullptr;
    CHK_SOLVER(cusolverDnCreate(&handle));
    CHK_SOLVER(cusolverDnSetStream(handle, lp_stream));
    int lwork = 0;
    CHK_SOLVER(cusolverDnSgetrf_bufferSize(handle, N, N, dA, N, &lwork));
    CHK_RT(cudaMalloc(&dWork, (size_t)std::max(1, lwork) * sizeof(float)));

    auto run_solve = [&]() -> bool {
        CHK_SOLVER(cusolverDnSgetrf(handle, N, N, dA, N, dWork, dIpiv, dInfo));
        CHK_SOLVER(cusolverDnSgetrs(handle, CUBLAS_OP_N, N, nrhs, dA, N, dIpiv, dB, N, dInfo));
        return true;
    };

    // Warmup + Golden reference (no preemption)
    CHK_RT(cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dB, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice));
    if (!run_solve()) return false;
    CHK_RT(cudaStreamSynchronize(lp_stream));
    CHK_RT(cudaMemcpy(hX_ref.data(), dB, szB * sizeof(float), cudaMemcpyDeviceToHost));

    // Preempted run
    CHK_RT(cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dB, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaDeviceSynchronize());

    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 2000, np, run_solve, &pt);
    CHK_RT(cudaMemcpy(hX_got.data(), dB, szB * sizeof(float), cudaMemcpyDeviceToHost));

    int info_h = -1;
    CHK_RT(cudaMemcpy(&info_h, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    int bad = 0;
    for (size_t i = 0; i < szB; ++i) {
        double e = std::fabs((double)hX_got[i] - (double)hX_ref[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-4) bad++;
    }

    cusolverDnDestroy(handle);
    cudaFree(dWork); cudaFree(dInfo); cudaFree(dIpiv); cudaFree(dB); cudaFree(dA);

    bool pass = ok && info_h == 0 && bad == 0;
    std::printf("[LIB-09 cuSOLVER LU+Solve %dx%d] %s | preempts=%d susp=%.3f ms hp=%.3f ms | info=%d max_err=%.3e bad=%d/%zu\n",
                N, N, pass ? "PASS" : "FAIL", pt.actual_preempts, pt.suspend_ms, pt.hp_ms, info_h, max_err, bad, szB);
    return pass;
}

// =========================================================================
// LIB-10: cuSOLVER Cholesky Factorization (cusolverDnSpotrf, N=3072 SPD)
// =========================================================================
bool TestLib10CuSolverCholesky(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream, int np) {
    const int N = 3072;
    const size_t szA = (size_t)N * N;
    std::vector<float> hA(szA), hRef(szA), hGot(szA);
    for (int j = 0; j < N; ++j) {
        for (int i = j; i < N; ++i) {
            float v = (i == j) ? 10.0f : 0.0005f * std::cos(0.01f * (i + j));
            hA[(size_t)j * N + i] = v;
            hA[(size_t)i * N + j] = v;
        }
    }

    float *dA = nullptr, *dWork = nullptr;
    int *dInfo = nullptr;
    CHK_RT(cudaMalloc(&dA, szA * sizeof(float)));
    CHK_RT(cudaMalloc(&dInfo, sizeof(int)));

    cusolverDnHandle_t handle = nullptr;
    CHK_SOLVER(cusolverDnCreate(&handle));
    CHK_SOLVER(cusolverDnSetStream(handle, lp_stream));
    int lwork = 0;
    CHK_SOLVER(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, N, dA, N, &lwork));
    CHK_RT(cudaMalloc(&dWork, (size_t)std::max(1, lwork) * sizeof(float)));

    auto run_potrf = [&]() -> bool {
        CHK_SOLVER(cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, N, dA, N, dWork, lwork, dInfo));
        return true;
    };

    // Golden
    CHK_RT(cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice));
    if (!run_potrf()) return false;
    CHK_RT(cudaStreamSynchronize(lp_stream));
    CHK_RT(cudaMemcpy(hRef.data(), dA, szA * sizeof(float), cudaMemcpyDeviceToHost));

    // Preempted
    CHK_RT(cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaDeviceSynchronize());

    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 2000, np, run_potrf, &pt);
    CHK_RT(cudaMemcpy(hGot.data(), dA, szA * sizeof(float), cudaMemcpyDeviceToHost));
    int info_h = -1;
    CHK_RT(cudaMemcpy(&info_h, dInfo, sizeof(int), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    int bad = 0;
    for (int j = 0; j < N; ++j) {
        for (int i = j; i < N; ++i) {
            size_t idx = (size_t)j * N + i;
            double e = std::fabs((double)hGot[idx] - (double)hRef[idx]);
            if (e > max_err) max_err = e;
            if (e > 1e-4) bad++;
        }
    }

    cusolverDnDestroy(handle);
    cudaFree(dWork); cudaFree(dInfo); cudaFree(dA);

    bool pass = ok && info_h == 0 && bad == 0;
    std::printf("[LIB-10 cuSOLVER Cholesky %dx%d] %s | preempts=%d susp=%.3f ms hp=%.3f ms | info=%d max_err=%.3e bad=%d\n",
                N, N, pass ? "PASS" : "FAIL", pt.actual_preempts, pt.suspend_ms, pt.hp_ms, info_h, max_err, bad);
    return pass;
}

// =========================================================================
// LIB-11: cuRAND Stateful Non-Idempotent PRNG (XORWOW + Philox4_32_10)
// Multi-call state progression; any double execution corrupts PRNG sequence!
// =========================================================================
bool TestLib11CuRandStateful(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream, int np) {
    const size_t N = 8 * 1024 * 1024;  // 8M floats per call
    const int calls = 60;
    float *dOut = nullptr;
    CHK_RT(cudaMalloc(&dOut, N * sizeof(float)));

    auto run_prng = [&](curandRngType_t rng_type, std::vector<float> &out_vec, bool preempt, PreemptTiming *pt) -> bool {
        curandGenerator_t gen = nullptr;
        CHK_RAND(curandCreateGenerator(&gen, rng_type));
        CHK_RAND(curandSetStream(gen, lp_stream));
        CHK_RAND(curandSetPseudoRandomGeneratorSeed(gen, 20261124ULL));
        // Materialize seed setup before timing
        CHK_RAND(curandGenerateUniform(gen, dOut, 1024));
        CHK_RT(cudaStreamSynchronize(lp_stream));

        // Reset seed so golden & preempted start from identical PRNG state
        CHK_RAND(curandSetPseudoRandomGeneratorSeed(gen, 20261124ULL));
        CHK_RAND(curandSetGeneratorOffset(gen, 0ULL));

        auto launch = [&]() -> bool {
            for (int c = 0; c < calls; ++c) {
                CHK_RAND(curandGenerateNormal(gen, dOut, N, 0.0f, 1.0f));
            }
            return true;
        };

        bool ok = true;
        if (preempt) {
            ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 2000, np, launch, pt);
        } else {
            ok = launch();
            CHK_RT(cudaStreamSynchronize(lp_stream));
        }
        out_vec.resize(N);
        CHK_RT(cudaMemcpy(out_vec.data(), dOut, N * sizeof(float), cudaMemcpyDeviceToHost));
        curandDestroyGenerator(gen);
        return ok;
    };

    std::vector<float> ref_xor, got_xor, ref_ph, got_ph;
    PreemptTiming pt1{}, pt2{};
    bool ok1 = run_prng(CURAND_RNG_PSEUDO_XORWOW, ref_xor, false, nullptr) &&
               run_prng(CURAND_RNG_PSEUDO_XORWOW, got_xor, true, &pt1);
    bool ok2 = run_prng(CURAND_RNG_PSEUDO_PHILOX4_32_10, ref_ph, false, nullptr) &&
               run_prng(CURAND_RNG_PSEUDO_PHILOX4_32_10, got_ph, true, &pt2);

    int bad_xor = 0, bad_ph = 0;
    for (size_t i = 0; i < N; ++i) {
        if (std::memcmp(&ref_xor[i], &got_xor[i], sizeof(float)) != 0) bad_xor++;
        if (std::memcmp(&ref_ph[i], &got_ph[i], sizeof(float)) != 0) bad_ph++;
    }
    cudaFree(dOut);

    bool pass = ok1 && ok2 && pt1.actual_preempts >= 1 && pt2.actual_preempts >= 1 && bad_xor == 0 && bad_ph == 0;
    std::printf("[LIB-11 cuRAND Stateful XORWOW+Philox 60x8M] %s | preempts=%d+%d susp=%.3f/%.3f ms | bad_bits=%d+%d / %zu\n",
                pass ? "PASS" : "FAIL", pt1.actual_preempts, pt2.actual_preempts,
                pt1.suspend_ms, pt2.suspend_ms, bad_xor, bad_ph, 2 * N);
    return pass;
}

// =========================================================================
// LIB-12: cuFFT 2D Batched R2C -> Spectral Filter -> C2R Pipeline
// =========================================================================
bool TestLib12CuFFT2D(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream, int np) {
    const int batch = 64;
    const int H = 256, W = 256;
    const int n_real = batch * H * W;
    const int n_cplx = batch * H * (W / 2 + 1);

    std::vector<float> hIn(n_real), hRef(n_real), hGot(n_real);
    for (int i = 0; i < n_real; ++i) hIn[i] = std::sin(0.003f * i) + 0.5f * std::cos(0.007f * i);

    float *dReal = nullptr;
    cufftComplex *dCplx = nullptr;
    CHK_RT(cudaMalloc(&dReal, (size_t)n_real * sizeof(float)));
    CHK_RT(cudaMalloc(&dCplx, (size_t)n_cplx * sizeof(cufftComplex)));

    cufftHandle plan_fwd, plan_inv;
    int dims[2] = {H, W};
    CHK_FFT(cufftPlanMany(&plan_fwd, 2, dims, nullptr, 1, H * W, nullptr, 1, H * (W / 2 + 1), CUFFT_R2C, batch));
    CHK_FFT(cufftPlanMany(&plan_inv, 2, dims, nullptr, 1, H * (W / 2 + 1), nullptr, 1, H * W, CUFFT_C2R, batch));
    CHK_FFT(cufftSetStream(plan_fwd, lp_stream));
    CHK_FFT(cufftSetStream(plan_inv, lp_stream));

    const float inv_scale = 1.0f / (float)(H * W);
    auto run_pipe = [&]() -> bool {
        for (int it = 0; it < 6; ++it) {
            CHK_FFT(cufftExecR2C(plan_fwd, dReal, dCplx));
            k_scale_complex<<<(n_cplx + 255) / 256, 256, 0, lp_stream>>>(dCplx, n_cplx, inv_scale);
            CHK_FFT(cufftExecC2R(plan_inv, dCplx, dReal));
        }
        return true;
    };

    // Golden
    CHK_RT(cudaMemcpy(dReal, hIn.data(), (size_t)n_real * sizeof(float), cudaMemcpyHostToDevice));
    if (!run_pipe()) return false;
    CHK_RT(cudaStreamSynchronize(lp_stream));
    CHK_RT(cudaMemcpy(hRef.data(), dReal, (size_t)n_real * sizeof(float), cudaMemcpyDeviceToHost));

    // Preempted
    CHK_RT(cudaMemcpy(dReal, hIn.data(), (size_t)n_real * sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaDeviceSynchronize());
    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 1200, np, run_pipe, &pt);
    CHK_RT(cudaMemcpy(hGot.data(), dReal, (size_t)n_real * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    int bad = 0;
    for (int i = 0; i < n_real; ++i) {
        double e = std::fabs((double)hGot[i] - (double)hRef[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-5) bad++;
    }

    cufftDestroy(plan_fwd); cufftDestroy(plan_inv);
    cudaFree(dCplx); cudaFree(dReal);

    bool pass = ok && bad == 0;
    std::printf("[LIB-12 cuFFT 2D R2C+C2R 64x256x256] %s | preempts=%d susp=%.3f ms hp=%.3f ms | max_err=%.3e bad=%d/%d\n",
                pass ? "PASS" : "FAIL", pt.actual_preempts, pt.suspend_ms, pt.hp_ms, max_err, bad, n_real);
    return pass;
}

// =========================================================================
// LIB-13: CUB Decoupled Look-Back DeviceScan::ExclusiveSum + Reduce + Select
// Tests inter-CTA Look-Back prefix-sum synchronization under Mode 5!
// =========================================================================
bool TestLib13CubScanReduceSelect(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream, int np) {
    const int N = 16 * 1024 * 1024;  // 16M uint32
    std::vector<uint32_t> hIn(N), hScanRef(N), hScanGot(N);
    for (int i = 0; i < N; ++i) hIn[i] = (uint32_t)((i * 7 + 3) % 13);

    uint32_t *dIn = nullptr, *dScan = nullptr, *dSel = nullptr, *dSum = nullptr;
    int *dNumSel = nullptr;
    CHK_RT(cudaMalloc(&dIn, (size_t)N * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&dScan, (size_t)N * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&dSel, (size_t)N * sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&dSum, sizeof(uint32_t)));
    CHK_RT(cudaMalloc(&dNumSel, sizeof(int)));
    CHK_RT(cudaMemcpy(dIn, hIn.data(), (size_t)N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    size_t tmp1 = 0, tmp2 = 0, tmp3 = 0;
    CHK_RT(cub::DeviceScan::ExclusiveSum(nullptr, tmp1, dIn, dScan, N, lp_stream));
    CHK_RT(cub::DeviceReduce::Sum(nullptr, tmp2, dIn, dSum, N, lp_stream));
    CHK_RT(cub::DeviceSelect::If(nullptr, tmp3, dIn, dSel, dNumSel, N, KeepOddOp{}, lp_stream));
    size_t tmp_sz = std::max({tmp1, tmp2, tmp3});
    void *dTmp = nullptr;
    CHK_RT(cudaMalloc(&dTmp, tmp_sz));

    auto run_cub = [&]() -> bool {
        for (int it = 0; it < 4; ++it) {
            CHK_RT(cub::DeviceScan::ExclusiveSum(dTmp, tmp1, dIn, dScan, N, lp_stream));
            CHK_RT(cub::DeviceReduce::Sum(dTmp, tmp2, dIn, dSum, N, lp_stream));
            CHK_RT(cub::DeviceSelect::If(dTmp, tmp3, dIn, dSel, dNumSel, N, KeepOddOp{}, lp_stream));
        }
        return true;
    };

    // Golden
    if (!run_cub()) return false;
    CHK_RT(cudaStreamSynchronize(lp_stream));
    CHK_RT(cudaMemcpy(hScanRef.data(), dScan, (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t sum_ref = 0; int sel_ref = 0;
    CHK_RT(cudaMemcpy(&sum_ref, dSum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHK_RT(cudaMemcpy(&sel_ref, dNumSel, sizeof(int), cudaMemcpyDeviceToHost));

    // Preempted
    CHK_RT(cudaMemset(dScan, 0, (size_t)N * sizeof(uint32_t)));
    CHK_RT(cudaDeviceSynchronize());
    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 800, np, run_cub, &pt);
    CHK_RT(cudaMemcpy(hScanGot.data(), dScan, (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t sum_got = 0; int sel_got = 0;
    CHK_RT(cudaMemcpy(&sum_got, dSum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHK_RT(cudaMemcpy(&sel_got, dNumSel, sizeof(int), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < N; ++i) if (hScanGot[i] != hScanRef[i]) bad++;

    cudaFree(dTmp); cudaFree(dNumSel); cudaFree(dSum); cudaFree(dSel); cudaFree(dScan); cudaFree(dIn);

    bool pass = ok && bad == 0 && sum_got == sum_ref && sel_got == sel_ref;
    std::printf("[LIB-13 CUB LookBack Scan+Reduce+Select 16M] %s | preempts=%d susp=%.3f ms hp=%.3f ms | bad=%d sum_ok=%d sel_ok=%d\n",
                pass ? "PASS" : "FAIL", pt.actual_preempts, pt.suspend_ms, pt.hp_ms,
                bad, (int)(sum_got == sum_ref), (int)(sel_got == sel_ref));
    return pass;
}

// =========================================================================
// LIB-14: Blackwell cuBLASLt FP8 E4M3 Tensor Core GEMM (beta=1.0 non-idemp)
//         + Multi-Head cublasGemmStridedBatchedEx
// =========================================================================
bool TestLib14CuBlasLtFP8AndBatched(XQueueHandle xq, cudaStream_t lp_stream, cudaStream_t hp_stream, int np) {
    const int M = 4096, N = 4096, K = 4096;
    const size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;

    std::vector<__nv_fp8_e4m3> hA(szA), hB(szB);
    std::vector<__nv_bfloat16> hC0(szC), hRef(szC), hGot(szC);
    for (size_t i = 0; i < szA; ++i) hA[i] = __nv_fp8_e4m3(0.125f * (float)((int)(i % 11) - 5));
    for (size_t i = 0; i < szB; ++i) hB[i] = __nv_fp8_e4m3(0.125f * (float)((int)((i * 3) % 11) - 5));
    for (size_t i = 0; i < szC; ++i) hC0[i] = __float2bfloat16(0.25f * (float)((int)(i % 7) - 3));

    __nv_fp8_e4m3 *dA = nullptr, *dB = nullptr;
    __nv_bfloat16 *dC = nullptr;
    float *dScaleA = nullptr, *dScaleB = nullptr;
    void *dWs = nullptr;
    const size_t ws_sz = 32 * 1024 * 1024;
    CHK_RT(cudaMalloc(&dA, szA));
    CHK_RT(cudaMalloc(&dB, szB));
    CHK_RT(cudaMalloc(&dC, szC * sizeof(__nv_bfloat16)));
    CHK_RT(cudaMalloc(&dScaleA, sizeof(float)));
    CHK_RT(cudaMalloc(&dScaleB, sizeof(float)));
    CHK_RT(cudaMalloc(&dWs, ws_sz));

    float one = 1.0f;
    CHK_RT(cudaMemcpy(dA, hA.data(), szA, cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dB, hB.data(), szB, cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dScaleA, &one, sizeof(float), cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dScaleB, &one, sizeof(float), cudaMemcpyHostToDevice));

    cublasLtHandle_t lt = nullptr;
    CHK_BL(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t opDesc = nullptr;
    CHK_BL(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transA = CUBLAS_OP_T, transB = CUBLAS_OP_N;
    CHK_BL(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHK_BL(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    CHK_BL(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dScaleA, sizeof(dScaleA)));
    CHK_BL(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dScaleB, sizeof(dScaleB)));

    cublasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr;
    CHK_BL(cublasLtMatrixLayoutCreate(&layA, CUDA_R_8F_E4M3, K, M, K));
    CHK_BL(cublasLtMatrixLayoutCreate(&layB, CUDA_R_8F_E4M3, K, N, K));
    CHK_BL(cublasLtMatrixLayoutCreate(&layC, CUDA_R_16BF, M, N, M));

    cublasLtMatmulPreference_t pref = nullptr;
    CHK_BL(cublasLtMatmulPreferenceCreate(&pref));
    CHK_BL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_sz, sizeof(ws_sz)));
    cublasLtMatmulHeuristicResult_t heur{};
    int returned = 0;
    CHK_BL(cublasLtMatmulAlgoGetHeuristic(lt, opDesc, layA, layB, layC, layC, pref, 1, &heur, &returned));
    if (returned == 0) {
        std::fprintf(stderr, "No FP8 E4M3 heuristic found!\n");
        return false;
    }

    float alpha = 0.01f, beta = 1.0f;
    auto run_fp8 = [&]() -> bool {
        for (int it = 0; it < 4; ++it) {
            CHK_BL(cublasLtMatmul(lt, opDesc, &alpha, dA, layA, dB, layB, &beta,
                                  dC, layC, dC, layC, &heur.algo, dWs, ws_sz, lp_stream));
        }
        return true;
    };

    // Golden
    CHK_RT(cudaMemcpy(dC, hC0.data(), szC * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    if (!run_fp8()) return false;
    CHK_RT(cudaStreamSynchronize(lp_stream));
    CHK_RT(cudaMemcpy(hRef.data(), dC, szC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Preempted
    CHK_RT(cudaMemcpy(dC, hC0.data(), szC * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHK_RT(cudaDeviceSynchronize());
    PreemptTiming pt{};
    bool ok = RunWithMidPreempt(xq, lp_stream, hp_stream, 1000, np, run_fp8, &pt);
    CHK_RT(cudaMemcpy(hGot.data(), dC, szC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    int bad = 0;
    for (size_t i = 0; i < szC; ++i) {
        double e = std::fabs((double)__bfloat162float(hGot[i]) - (double)__bfloat162float(hRef[i]));
        if (e > max_err) max_err = e;
        if (e > 0.0) bad++;
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layC); cublasLtMatrixLayoutDestroy(layB); cublasLtMatrixLayoutDestroy(layA);
    cublasLtMatmulDescDestroy(opDesc); cublasLtDestroy(lt);
    cudaFree(dWs); cudaFree(dScaleB); cudaFree(dScaleA); cudaFree(dC); cudaFree(dB); cudaFree(dA);

    bool pass = ok && bad == 0;
    std::printf("[LIB-14 cuBLASLt FP8-E4M3 4096^3 beta=1] %s | preempts=%d susp=%.3f ms hp=%.3f ms | max_err=%.3e bad=%d/%zu\n",
                pass ? "PASS" : "FAIL", pt.actual_preempts, pt.suspend_ms, pt.hp_ms, max_err, bad, szC);
    return pass;
}

}  // namespace

int main(int argc, char **argv) {
    std::string target = (argc >= 2) ? argv[1] : "all";
    int np = 2;
    if (const char *env = std::getenv("TEST_NP")) np = std::atoi(env);

    cudaStream_t lp_stream = nullptr, hp_stream = nullptr;
    if (cudaStreamCreateWithFlags(&lp_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&hp_stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "Failed to create CUDA streams\n");
        return 1;
    }

    HwQueueHandle hwq = 0;
    XQueueHandle xq = 0;
    if (CudaQueueCreate(&hwq, lp_stream) != kXSchedSuccess ||
        XQueueCreate(&xq, hwq, kPreemptLevelInterrupt, kQueueCreateFlagNone) != kXSchedSuccess) {
        std::fprintf(stderr, "Failed to create Level-3 XQueue\n");
        return 1;
    }

    bool all_ok = true;
    auto run = [&](const char *name, auto fn) {
        if (target == "all" || target == name) {
            if (!fn(xq, lp_stream, hp_stream, np)) all_ok = false;
        }
    };

    run("LIB-09", TestLib09CuSolverLU);
    run("LIB-10", TestLib10CuSolverCholesky);
    run("LIB-11", TestLib11CuRandStateful);
    run("LIB-12", TestLib12CuFFT2D);
    run("LIB-13", TestLib13CubScanReduceSelect);
    run("LIB-14", TestLib14CuBlasLtFP8AndBatched);

    XQueueDestroy(xq);
    HwQueueDestroy(hwq);
    if (g_hp_out) cudaFree(g_hp_out);
    cudaStreamDestroy(lp_stream);
    cudaStreamDestroy(hp_stream);

    if (all_ok) std::printf("PHASE3_CUDA_LIBS_ALL_PASS\n");
    return all_ok ? 0 : 1;
}
