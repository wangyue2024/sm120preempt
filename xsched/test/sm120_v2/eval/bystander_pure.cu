#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <unistd.h>

__global__ void k_bystander(float *out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 1.0f + float(i & 31) * 0.01f;
    for (int k = 0; k < iters; ++k) v = fmaf(v, 0.9999f, 0.0001f);
    out[i] = v;
}
static double Pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = std::min(v.size() - 1, (size_t)std::ceil(p * v.size()) - 1);
    return v[idx];
}
int main() {
    const int n = 60 * 256;
    float *d_out = nullptr;
    cudaMalloc(&d_out, n * 4);
    k_bystander<<<60, 256>>>(d_out, n, 4000);
    cudaDeviceSynchronize();
    std::vector<float> ref(n), cur(n);
    cudaMemcpy(ref.data(), d_out, n * 4, cudaMemcpyDeviceToHost);

    std::vector<double> lats;
    size_t err_cnt = 0;
    for (int i = 0; i < 80; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        k_bystander<<<60, 256>>>(d_out, n, 4000);
        cudaError_t e = cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        if (e != cudaSuccess) { ++err_cnt; break; }
        lats.push_back(std::chrono::duration<double,std::milli>(t1 - t0).count());
        usleep(2000);
    }
    cudaMemcpy(cur.data(), d_out, n * 4, cudaMemcpyDeviceToHost);
    size_t mis = 0;
    for (int i = 0; i < n; ++i) if (cur[i] != ref[i]) ++mis;
    printf("[BYSTANDER] n=%zu P50=%.3fms P95=%.3fms P99=%.3fms Max=%.3fms err=%zu mis=%zu\n",
           lats.size(), Pct(lats,0.5), Pct(lats,0.95), Pct(lats,0.99), Pct(lats,1.0), err_cnt, mis);
    return 0;
}
