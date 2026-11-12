#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void accum_kernel(unsigned long long *counter, int iters) {
    for (int i = 0; i < iters; ++i) {
        atomicAdd(counter, 1ULL);
    }
}

int main(int argc, char **argv) {
    int blocks = (argc > 1) ? atoi(argv[1]) : 64;
    int threads = (argc > 2) ? atoi(argv[2]) : 256;
    int iters = (argc > 3) ? atoi(argv[3]) : 1000;

    printf("[ACCUM] Launching non-idempotent atomic accumulator (blocks=%d, threads=%d, iters=%d)...\n",
           blocks, threads, iters);

    unsigned long long *d_counter;
    cudaMalloc(&d_counter, sizeof(unsigned long long));
    cudaMemset(d_counter, 0, sizeof(unsigned long long));

    accum_kernel<<<blocks, threads>>>(d_counter, iters);
    cudaDeviceSynchronize();

    unsigned long long h_counter = 0;
    cudaMemcpy(&h_counter, d_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long expected = (unsigned long long)blocks * threads * iters;
    int mismatch = (h_counter == expected) ? 0 : 1;

    printf("[ACCUM] Status: %s, counter=%llu, expected=%llu, mismatch == %d\n",
           mismatch == 0 ? "PASS" : "FAIL", h_counter, expected, mismatch);

    cudaFree(d_counter);
    return mismatch;
}
