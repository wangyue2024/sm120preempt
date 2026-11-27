// Maximum-register test kernel: keeps 250 floats live in registers through a long loop. Compiled
// with -maxrregcount=255 ptxas reports 255 registers and no spills; the SASS uses R0..R252 (ptxas
// never allocates R253 / R254 on sm_120), so every allocated register is live when a trap arrives.
#include <cuda_runtime.h>
#ifndef NACC
#define NACC 250
#endif
#ifndef LB
#define LB 1
#endif
#if LB
#define REGPRESS_BOUNDS __launch_bounds__(128)
#else
#define REGPRESS_BOUNDS            // -maxrregcount is ignored for kernels with launch bounds
#endif
extern "C" __global__ void REGPRESS_BOUNDS regpress255(const float *in, float *out, int iters)
{
    float acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = in[(threadIdx.x * 7 + i) & 4095];
    for (int t = 0; t < iters; ++t) {
        const float k = 1.0f + 1e-6f * (float)(t & 7);
#pragma unroll
        for (int i = 0; i < NACC; ++i) acc[i] = fmaf(acc[i], k, 1e-7f * (float)i);
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < NACC; ++i) s = fmaf(acc[i], (float)(i + 1), s);
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

extern "C" int launch_regpress255(const float *in, float *out, int iters, int blocks, void *stream)
{
    regpress255<<<blocks, 128, 0, (cudaStream_t)stream>>>(in, out, iters);
    return (int)cudaGetLastError();
}
