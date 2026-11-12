#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void stencil3d_kernel(const float *in, float *out, int dim) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && x < dim - 1 && y > 0 && y < dim - 1 && z > 0 && z < dim - 1) {
        int idx = z * dim * dim + y * dim + x;
        float center = in[idx];
        float neighbors = in[idx - 1] + in[idx + 1] +
                          in[idx - dim] + in[idx + dim] +
                          in[idx - dim * dim] + in[idx + dim * dim];
        out[idx] = 0.5f * center + (0.5f / 6.0f) * neighbors;
    }
}

int main(int argc, char **argv) {
    int dim = (argc > 1) ? atoi(argv[1]) : 128;
    int steps = (argc > 2) ? atoi(argv[2]) : 20;
    size_t num_elements = (size_t)dim * dim * dim;
    size_t size = num_elements * sizeof(float);

    printf("[STENCIL] Initializing 3D Stencil (%d^3, %zu elements, %d steps)...\n", dim, num_elements, steps);

    float *h_grid = (float*)malloc(size);
    for (size_t i = 0; i < num_elements; ++i) h_grid[i] = 1.0f;
    // Hot spot in center
    h_grid[(dim/2)*dim*dim + (dim/2)*dim + (dim/2)] = 100.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_grid, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_grid, size, cudaMemcpyHostToDevice);

    dim3 block(8, 8, 8);
    dim3 grid((dim + block.x - 1) / block.x, (dim + block.y - 1) / block.y, (dim + block.z - 1) / block.z);

    for (int step = 0; step < steps; ++step) {
        stencil3d_kernel<<<grid, block>>>(d_in, d_out, dim);
        float *tmp = d_in; d_in = d_out; d_out = tmp;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_grid, d_in, size, cudaMemcpyDeviceToHost);

    // Verify center is positive and dissipated
    float center_val = h_grid[(dim/2)*dim*dim + (dim/2)*dim + (dim/2)];
    int mismatch = (center_val > 0.0f && center_val < 100.0f) ? 0 : 1;

    printf("[STENCIL] Status: %s, mismatch == %d (center: %f)\n", mismatch == 0 ? "PASS" : "FAIL", mismatch, center_val);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_grid);

    return mismatch;
}
