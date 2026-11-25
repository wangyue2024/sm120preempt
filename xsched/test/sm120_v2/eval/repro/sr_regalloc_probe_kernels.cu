// Each kernel keeps N values live; the S2R SR_TID.X feeding `t` is binary-patched to SR_REGALLOC.
template <int N>
__global__ void probe(unsigned *out, const unsigned *src)
{
    unsigned t = threadIdx.x;
    unsigned v[N];
#pragma unroll
    for (int k = 0; k < N; ++k) v[k] = __ldg(src + k) * (k + 3);
    unsigned acc = 0;
#pragma unroll 1
    for (int it = 0; it < 4; ++it) {
#pragma unroll
        for (int k = 0; k < N; ++k) { v[k] = v[k] * 2654435761u + v[(k + 1) % N]; }
    }
#pragma unroll
    for (int k = 0; k < N; ++k) acc ^= v[k];
    if (blockIdx.x == 0) { out[0] = t; out[1] = acc; }
}
template __global__ void probe<8>(unsigned *, const unsigned *);
template __global__ void probe<12>(unsigned *, const unsigned *);
template __global__ void probe<20>(unsigned *, const unsigned *);
template __global__ void probe<40>(unsigned *, const unsigned *);
template __global__ void probe<64>(unsigned *, const unsigned *);
template __global__ void probe<120>(unsigned *, const unsigned *);
template __global__ void probe<200>(unsigned *, const unsigned *);
