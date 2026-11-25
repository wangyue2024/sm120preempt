#include <cstdio>
__global__ void k(unsigned *out) {
    extern __shared__ unsigned char dyn[];
    unsigned tot, dsz;
    asm volatile("mov.u32 %0, %%total_smem_size;" : "=r"(tot));
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
    if (threadIdx.x == 0 && blockIdx.x == 0) { if (dsz) dyn[0] = 1; out[0] = tot; out[1] = dsz; }
}
__global__ void k_static(unsigned *out) {
    __shared__ unsigned char st[36352];
    unsigned tot; asm volatile("mov.u32 %0, %%total_smem_size;" : "=r"(tot));
    st[threadIdx.x] = 1; if (threadIdx.x == 0 && blockIdx.x == 0) { out[0] = tot + st[5]; }
}
int main() {
    unsigned *d, h[2];
    cudaMalloc(&d, 8);
    for (int dyn : {0, 32, 1000, 3000, 4096, 8000, 20000, 40000}) {
        k<<<1, 256, dyn>>>(d); { cudaError_t e = cudaDeviceSynchronize(); if (e) { printf("err %s\n", cudaGetErrorString(e)); return 1; } } cudaMemcpy(h, d, 8, cudaMemcpyDeviceToHost);
        unsigned formula = dyn > 0 ? ((dyn + 0x400 + 4095) / 4096) * 4096 : 0;
        printf("dyn=%6d  total_smem_size=%6u (0x%05x)  dynamic=%6u  | engine smem_area=%6u  %s\n", dyn, h[0], h[0], h[1], formula,
               (h[0] > 0x400 && h[0] - 0x400 > formula - (formula ? 0x400 : 0)) ? "<-- SAVE (SR_SMEMSZ-0x400) EXCEEDS AREA" : "");
    }
    k_static<<<1, 256>>>(d); cudaMemcpy(h, d, 4, cudaMemcpyDeviceToHost);
    printf("static=36352 total_smem_size=%u (0x%x) engine smem_area=%u\n", h[0] - 0, h[0], ((36352 + 0x400 + 4095) / 4096) * 4096);
}
