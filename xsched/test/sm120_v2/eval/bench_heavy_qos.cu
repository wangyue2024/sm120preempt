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
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"

#define CHK_RT(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA err %d (%s) at %s:%d\n", (int)_e, cudaGetErrorString(_e), __FILE__, __LINE__); \
        std::exit(2); \
    } \
} while (0)

// Heavy SM-saturating non-idempotent DNN training / LLM attention kernel (~25ms per call on RTX 5060)
// Deliberately uses modest grid (120 blocks x 256 threads = 4 blocks/SM on 30-SM RTX 5060)
// to saturate all 30 SMs completely so Native CUDA Streams suffer true ~15-25ms Head-of-Line blocking!
__global__ void k_heavy_nonidemp_train(float *w, float *m, float *v, const float *g,
                                       int n, int inner_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float wi = w[idx], mi = m[idx], vi = v[idx], gi = g[idx];
    for (int it = 0; it < inner_iters; ++it) {
        float ge = gi + 1e-4f * sinf(wi + static_cast<float>(it) * 0.001f);
        mi = 0.9f * mi + 0.1f * ge;
        vi = 0.999f * vi + 0.001f * ge * ge;
        wi = wi * 0.99999f - 1e-3f * mi * rsqrtf(vi + 1e-6f);
    }
    w[idx] = wi; m[idx] = mi; v[idx] = vi;
}

// High-priority real-time inference / LLM decode kernel (~0.20ms native, needs SMs immediately!)
__global__ void k_hp_realtime_infer(const float *x, float *y, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    for (int it = 0; it < iters; ++it) v = fmaf(v, 0.9995f, 0.0005f);
    y[idx] = v;
}

static double Pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = std::min(v.size() - 1, static_cast<size_t>(std::ceil(p * v.size())) - 1);
    return v[idx];
}

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string mode = (argc > 1) ? argv[1] : "m5";
    int period_ms    = (argc > 2) ? std::atoi(argv[2]) : 15;
    int num_hp       = (argc > 3) ? std::atoi(argv[3]) : 25;
    int lp_iters     = (argc > 4) ? std::atoi(argv[4]) : 45000;

    int lp_prio = 0, hp_prio = 0;
    CHK_RT(cudaDeviceGetStreamPriorityRange(&lp_prio, &hp_prio));
    cudaStream_t lp_s = nullptr, hp_s = nullptr;
    CHK_RT(cudaStreamCreateWithPriority(&lp_s, cudaStreamNonBlocking, lp_prio));
    CHK_RT(cudaStreamCreateWithPriority(&hp_s, cudaStreamNonBlocking, hp_prio));

    HwQueueHandle hwq = 0;
    XQueueHandle xq = 0;
    bool use_xq = (mode != "native");
    if (use_xq) {
        XPreemptLevel lv = (mode == "lv1") ? kPreemptLevelBlock : kPreemptLevelInterrupt;
        if (CudaQueueCreate(&hwq, lp_s) != kXSchedSuccess ||
            XQueueCreate(&xq, hwq, lv, kQueueCreateFlagNone) != kXSchedSuccess) {
            std::fprintf(stderr, "XQueueCreate failed\n");
            return 2;
        }
    }

    const int blocks = 180, threads = 256, n = blocks * threads;
    float *dW, *dM, *dV, *dG, *dW0, *dM0, *dV0, *dhX, *dhY;
    CHK_RT(cudaMalloc(&dW, n*4)); CHK_RT(cudaMalloc(&dM, n*4)); CHK_RT(cudaMalloc(&dV, n*4));
    CHK_RT(cudaMalloc(&dG, n*4)); CHK_RT(cudaMalloc(&dW0, n*4)); CHK_RT(cudaMalloc(&dM0, n*4));
    CHK_RT(cudaMalloc(&dV0, n*4)); CHK_RT(cudaMalloc(&dhX, 60*256*4)); CHK_RT(cudaMalloc(&dhY, 60*256*4));

    std::vector<float> hW0(n), hM0(n), hV0(n), hG(n);
    for (int i = 0; i < n; ++i) {
        hW0[i] = float((i % 29) - 14) * 0.05f;
        hM0[i] = 0.001f * float(i % 7);
        hV0[i] = 0.01f + 0.001f * float(i % 5);
        hG[i]  = float((i % 17) - 8) * 0.02f;
    }
    CHK_RT(cudaMemcpy(dW0, hW0.data(), n*4, cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dM0, hM0.data(), n*4, cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dV0, hV0.data(), n*4, cudaMemcpyHostToDevice));
    CHK_RT(cudaMemcpy(dG,  hG.data(),  n*4, cudaMemcpyHostToDevice));
    CHK_RT(cudaMemset(dhX, 0, 60*256*4));

    auto run_lp = [&](cudaStream_t st) {
        CHK_RT(cudaMemcpyAsync(dW, dW0, n*4, cudaMemcpyDeviceToDevice, st));
        CHK_RT(cudaMemcpyAsync(dM, dM0, n*4, cudaMemcpyDeviceToDevice, st));
        CHK_RT(cudaMemcpyAsync(dV, dV0, n*4, cudaMemcpyDeviceToDevice, st));
        k_heavy_nonidemp_train<<<blocks, threads, 16384, st>>>(dW, dM, dV, dG, n, lp_iters);
    };

    // Standalone reference & baseline
    run_lp(hp_s); CHK_RT(cudaStreamSynchronize(hp_s));
    std::vector<float> hW_ref(n);
    CHK_RT(cudaMemcpy(hW_ref.data(), dW, n*4, cudaMemcpyDeviceToHost));
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; ++i) run_lp(hp_s);
    CHK_RT(cudaStreamSynchronize(hp_s));
    double base_lp = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t0).count() / 3.0;

    k_hp_realtime_infer<<<60, 256, 0, hp_s>>>(dhX, dhY, 60*256, 500);
    CHK_RT(cudaStreamSynchronize(hp_s));
    auto th0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) {
        k_hp_realtime_infer<<<60, 256, 0, hp_s>>>(dhX, dhY, 60*256, 500);
        CHK_RT(cudaStreamSynchronize(hp_s));
    }
    double base_hp = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - th0).count() / 10.0;

    // Warmup LP instrumentation
    run_lp(lp_s); CHK_RT(cudaStreamSynchronize(lp_s));

    std::atomic<bool> stop{false};
    std::atomic<int> lp_steps{0};
    double max_err = 0.0;
    size_t mismatch = 0;
    int checked = 0;

    std::thread lp_thr([&]() {
        CHK_RT(cudaSetDevice(0));
        std::vector<float> hW_cur(n);
        while (!stop.load(std::memory_order_acquire)) {
            run_lp(lp_s);
            CHK_RT(cudaStreamSynchronize(lp_s));
            lp_steps.fetch_add(1, std::memory_order_acq_rel);
            CHK_RT(cudaMemcpy(hW_cur.data(), dW, n*4, cudaMemcpyDeviceToHost));
            ++checked;
            for (int i = 0; i < n; ++i) {
                double e = std::fabs(double(hW_cur[i]) - double(hW_ref[i]));
                if (e > max_err) max_err = e;
                if (e > 1e-6) ++mismatch;
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::vector<double> hp_ms, susp_ms;
    auto tb0 = std::chrono::steady_clock::now();
    for (int r = 0; r < num_hp; ++r) {
        std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
        auto s0 = std::chrono::steady_clock::now();
        if (use_xq) XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue);
        auto s1 = std::chrono::steady_clock::now();
        k_hp_realtime_infer<<<60, 256, 0, hp_s>>>(dhX, dhY, 60*256, 500);
        CHK_RT(cudaStreamSynchronize(hp_s));
        auto s2 = std::chrono::steady_clock::now();
        if (use_xq) XQueueResume(xq, kQueueResumeFlagNone);
        susp_ms.push_back(std::chrono::duration<double,std::milli>(s1 - s0).count());
        hp_ms.push_back(std::chrono::duration<double,std::milli>(s2 - s0).count());
    }
    auto tb1 = std::chrono::steady_clock::now();
    stop.store(true, std::memory_order_release);
    lp_thr.join();

    double total_s = std::chrono::duration<double>(tb1 - tb0).count();
    double lp_ret = 100.0 * (lp_steps.load() / total_s) / (1000.0 / base_lp);
    int slo2 = 0, slo5 = 0;
    for (double x : hp_ms) { if (x <= 2.0) ++slo2; if (x <= 5.0) ++slo5; }
    std::printf("[HEAVY_QOS] mode=%-6s period=%2dms | base_lp=%.2fms base_hp=%.3fms | "
                "HP_ms(P50=%.3f P95=%.3f P99=%.3f Max=%.3f) SLO2ms=%.1f%% SLO5ms=%.1f%% | "
                "SUSP_ms(P50=%.3f P99=%.3f) | LP_steps=%d LP_ret=%.1f%% | "
                "max_err=%.2e mismatch=%zu/%d\n",
                mode.c_str(), period_ms, base_lp, base_hp,
                Pct(hp_ms,0.50), Pct(hp_ms,0.95), Pct(hp_ms,0.99), Pct(hp_ms,1.0),
                100.0*slo2/hp_ms.size(), 100.0*slo5/hp_ms.size(),
                Pct(susp_ms,0.50), Pct(susp_ms,0.99),
                lp_steps.load(), lp_ret, max_err, mismatch, checked);

    if (use_xq) { XQueueDestroy(xq); HwQueueDestroy(hwq); }
    cudaStreamDestroy(lp_s); cudaStreamDestroy(hp_s);
    return 0;
}
