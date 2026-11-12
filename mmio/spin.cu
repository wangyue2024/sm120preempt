// Long-running spin kernel; prints progress counter from mapped host memory.
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
__global__ void spin(volatile unsigned long long *ctr, long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {
        if (threadIdx.x == 0 && blockIdx.x == 0) *ctr += 1;
    }
}
int main(int argc, char **argv) {
    double secs = argc > 1 ? atof(argv[1]) : 5;
    int blocks = argc > 2 ? atoi(argv[2]) : 30;
    unsigned long long *h; cudaHostAlloc(&h, 8, cudaHostAllocMapped); *h = 0;
    unsigned long long *d; cudaHostGetDevicePointer(&d, h, 0);
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    spin<<<blocks, 128>>>(d, (long long)(secs * clk * 1000.0));
    printf("launched pid=%d\n", getpid()); fflush(stdout);
    cudaError_t e;
    while ((e = cudaStreamQuery(0)) == cudaErrorNotReady) { usleep(200000); printf("ctr=%llu\n", *h); fflush(stdout);}
    printf("done: %s ctr=%llu\n", cudaGetErrorString(e), *h);
    return 0;
}
