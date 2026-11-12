#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void attention_score_kernel(const float *Q, const float *K, float *scores, int seq_len, int head_dim) {
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (q_idx < seq_len && k_idx < seq_len) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += Q[q_idx * head_dim + d] * K[k_idx * head_dim + d];
        }
        scores[q_idx * seq_len + k_idx] = dot / sqrtf((float)head_dim);
    }
}

int main(int argc, char **argv) {
    int seq_len = (argc > 1) ? atoi(argv[1]) : 512;
    int head_dim = 64;
    size_t qk_size = seq_len * head_dim * sizeof(float);
    size_t score_size = seq_len * seq_len * sizeof(float);

    printf("[ATTENTION] Emulating Transformer Attention (seq_len=%d, head_dim=%d)...\n", seq_len, head_dim);

    float *h_Q = (float*)malloc(qk_size);
    float *h_K = (float*)malloc(qk_size);
    float *h_scores = (float*)malloc(score_size);

    for (int i = 0; i < seq_len * head_dim; ++i) {
        h_Q[i] = 0.5f;
        h_K[i] = 0.5f;
    }

    float *d_Q, *d_K, *d_scores;
    cudaMalloc(&d_Q, qk_size);
    cudaMalloc(&d_K, qk_size);
    cudaMalloc(&d_scores, score_size);

    cudaMemcpy(d_Q, h_Q, qk_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qk_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((seq_len + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);

    attention_score_kernel<<<grid, block>>>(d_Q, d_K, d_scores, seq_len, head_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_scores, d_scores, score_size, cudaMemcpyDeviceToHost);

    // Verify first score: head_dim * (0.5 * 0.5) / sqrt(64) = 64 * 0.25 / 8 = 2.0
    float expected = (head_dim * 0.25f) / sqrtf((float)head_dim);
    int mismatch = 0;
    if (fabs(h_scores[0] - expected) > 1e-3) {
        mismatch++;
        printf("[ATTENTION] Mismatch at score[0]: got %f, expected %f\n", h_scores[0], expected);
    }

    printf("[ATTENTION] Status: %s, mismatch == %d\n", mismatch == 0 ? "PASS" : "FAIL", mismatch);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_scores);
    free(h_Q);
    free(h_K);
    free(h_scores);

    return mismatch;
}
