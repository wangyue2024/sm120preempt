// cave_probe.cu - User-Space Hardware Architecture & 4-Stage Code Cave Verifier
// Runs WITHOUT root privileges! Uses CUDA Driver Internal Export Tables (libcuxtra)
// to inspect the active GPU's .text.traphandler arena and verify all Code Caves
// on RTX 5060, RTX 5090, or other NVIDIA GPUs/drivers.

#include "arena.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

__global__ void dummy_init_kernel() {}

struct CaveRegion {
    const char *name;
    uint64_t rel_from_handler;
    size_t expected_start_off;
    size_t required_instrs;
    const char *role;
};

int main(int argc, char **argv) {
    bool dump_files = (argc > 1 && strcmp(argv[1], "--dump") == 0);

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        fprintf(stderr, "[-] Failed to query CUDA device 0\n");
        return 1;
    }
    char bus_id[64] = {0};
    cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), 0);

    printf("====================================================================================\n");
    printf(" [1] GPU Hardware & Architecture Summary\n");
    printf("====================================================================================\n");
    printf("  GPU Model                 : %s\n", prop.name);
    printf("  Compute Capability        : sm_%d%d (Major=%d, Minor=%d)\n",
           prop.major, prop.minor, prop.major, prop.minor);
    printf("  PCI Bus ID                : %s\n", bus_id);
    printf("  SM Count                  : %d SMs\n", prop.multiProcessorCount);
    printf("  Max Threads / SM          : %d threads (%d warps/SM, %d threads total across GPU)\n",
           prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerMultiProcessor / 32,
           prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
    printf("  Max Threads / CTA         : %d threads\n", prop.maxThreadsPerBlock);
    printf("  Max 32-bit Regs / SM      : %d regs (%d KiB/SM)\n",
           prop.regsPerMultiprocessor, prop.regsPerMultiprocessor * 4 / 1024);
    printf("  Max 32-bit Regs / CTA     : %d regs\n", prop.regsPerBlock);
    printf("  Shared Memory / SM        : %zu bytes (%zu KiB)\n",
           prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor / 1024);
    printf("  Max Opt-in Shared / CTA   : %zu bytes (%zu KiB)\n",
           prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin / 1024);

    cudaFree(nullptr);
    dummy_init_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    auto info = sm120arena::probe(ctx, 0);

    printf("\n====================================================================================\n");
    printf(" [2] CUDA Context Internal Arena & .text.traphandler Probe (via libcuxtra)\n");
    printf("====================================================================================\n");
    sm120arena::report(info);
    if (!info.ok || !info.handler_is_memobj) {
        fprintf(stderr, "[-] ERROR: Could not locate live .text.traphandler memobj in CUDA context arena!\n");
        return 1;
    }

    // Read standard SM120 NOP encoding from handler+0x900 (known trailing NOP padding)
    uint64_t ref_nop_w0 = 0x0000000000007918ULL;
    if (info.handler.size() >= 0x8f0) {
        memcpy(&ref_nop_w0, &info.handler[0x8e0], 8);
    }

    printf("\n====================================================================================\n");
    printf(" [3] Multi-Stage Hardware Trap Code Cave Verification\n");
    printf("====================================================================================\n");
    const CaveRegion caves[] = {
        {"Cave 0 (tools+0xA00)",  0x4400, 0x0a00,  62, "Dispatcher + Modes 1/2/3/4 + Mode 5 Entry Jump"},
        {"Cave 1 (ext_code)",     0x5700, 0x0110, 152, "Mode 5 Evict/Restore Main + SMEM + URs + R16..R95"},
        {"Cave 2 (ext2_code)",    0x6200, 0x0110, 100, "Mode 5 Stage-2 Chained Extension: R96..R191"},
        {"Cave 3 (ext3_code)",    0x6c00, 0x0110,  64, "Mode 5 Stage-3 Chained Extension: R192..R252"},
        {"Reserve (r_07700)",     0x7700, 0x0110,   0, "Unused Reserve Sub-Page Cave (143 instrs free)"},
    };

    bool all_ok = true;
    for (const auto &c : caves) {
        uint64_t base_va = info.handler_va + c.rel_from_handler;
        cuxtra::etbl::CUtoolsMemObjHandle_st *obj = nullptr;
        unsigned long long off = 0;
        unsigned long sz = 0;
        bool found = cuxtra::cuda::FindMemObjHandle(ctx, base_va, &obj, &off, &sz);
        if (!found || sz <= c.expected_start_off) {
            printf("  [-] %-22s @ VA 0x%llx : NOT FOUND or too small (found=%d, size=0x%lx)\n",
                   c.name, (unsigned long long)(base_va + c.expected_start_off), (int)found, sz);
            all_ok = false;
            continue;
        }
        size_t cave_bytes = sz - c.expected_start_off;
        size_t max_instrs = cave_bytes / 16;
        bool ok = (max_instrs >= c.required_instrs);
        if (!ok) all_ok = false;
        printf("  [%s] %-22s @ VA 0x%llx : obj_sz=0x%04lx, cave=%4zu B (%3zu instrs capacity, need %3zu) | %s\n",
               ok ? "+" : "-", c.name,
               (unsigned long long)(base_va + c.expected_start_off),
               sz, cave_bytes, max_instrs, c.required_instrs, c.role);
    }

    if (dump_files) {
        FILE *f1 = fopen("live_handler.bin", "wb");
        if (f1) { fwrite(info.handler.data(), 1, info.handler.size(), f1); fclose(f1); }
        FILE *f2 = fopen("tools_image.bin", "wb");
        if (f2) { fwrite(info.tools.data(), 1, info.tools.size(), f2); fclose(f2); }
        printf("\n  [*] Dumped live_handler.bin (%zu B) and tools_image.bin (%zu B)\n",
               info.handler.size(), info.tools.size());
    }

    printf("\n====================================================================================\n");
    if (all_ok) {
        printf(" >>> [PASS] All 4 Hardware Trap Code Caves Verified Compatible on %s! <<<\n", prop.name);
    } else {
        printf(" >>> [WARN] Some Code Cave offsets differ on this GPU/driver — run with --dump to inspect! <<<\n");
    }
    printf("====================================================================================\n");
    return all_ok ? 0 : 1;
}
