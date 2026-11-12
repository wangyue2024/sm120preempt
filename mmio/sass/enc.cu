__global__ void k(unsigned long long *p, unsigned long long base) {
    unsigned sm, wid; unsigned long long t;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(wid));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    unsigned long long *q = (unsigned long long *)(base + (unsigned long long)sm * 64 + wid * 8);
    *q = t;
    atomicAdd((unsigned long long *)base, 1ull);
}
