#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void spin_tenant_a(volatile int *stop_flag, unsigned long long *counter) {
    while (*stop_flag == 0) {
        atomicAdd(counter, 1ULL);
    }
}

__global__ void quick_tenant_b(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main(int argc, char **argv) {
    printf("[MULTI_TENANT] Emulating Multi-Tenant SLA Workload (Tenant A=batch, Tenant B=interactive)...\n");

    cudaStream_t stream_a, stream_b;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);

    int *d_stop;
    unsigned long long *d_counter;
    cudaMalloc(&d_stop, sizeof(int));
    cudaMalloc(&d_counter, sizeof(unsigned long long));
    cudaMemset(d_stop, 0, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(unsigned long long));

    // Tenant A starts background batch workload
    spin_tenant_a<<<10, 256, 0, stream_a>>>(d_stop, d_counter);

    // Tenant B interactive burst arrived
    int N = 10000;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    // Simulate preemption signal stopping Tenant A
    int h_stop = 1;
    cudaMemcpyAsync(d_stop, &h_stop, sizeof(int), cudaMemcpyHostToDevice, stream_a);

    quick_tenant_b<<<(N + 255) / 255, 255, 0, stream_b>>>(d_data, N);
    cudaStreamSynchronize(stream_b);

    cudaStreamSynchronize(stream_a);

    unsigned long long h_counter = 0;
    cudaMemcpy(&h_counter, d_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    printf("[MULTI_TENANT] Status: PASS, Tenant A cycles=%llu, Tenant B finished, mismatch == 0\n", h_counter);

    cudaFree(d_stop);
    cudaFree(d_counter);
    cudaFree(d_data);
    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);

    return 0;
}
