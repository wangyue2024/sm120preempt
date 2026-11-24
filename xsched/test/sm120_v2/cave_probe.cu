#include <cuda.h>
#include <cstdio>

#include "xsched/cuda/hal/level3/sm120_trap_engine.h"

int main()
{
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuInit failed\n");
        return 1;
    }

    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuDeviceGet failed\n");
        return 1;
    }

    char name[128] = {};
    cuDeviceGetName(name, sizeof(name), dev);
    int major = 0, minor = 0, sm_cnt = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&sm_cnt, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);

    CUcontext ctx = nullptr;
    if (cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuCtxCreate failed\n");
        return 1;
    }

    xsched::cuda::sm120::Sm120Arena arena(ctx);
    const bool ok_disc = arena.Discover(true);
    const bool ok_bar0 = arena.EnsureBar0Mapped();
    std::printf("[CAVE_PROBE] GPU=%s SM=%d.%d SMs=%d discover=%s bar0=%s\n",
                name, major, minor, sm_cnt,
                ok_disc ? "OK" : "FAIL",
                ok_bar0 ? "MAPPED" : "NO_PERM");
    if (!ok_disc) {
        cuCtxDestroy(ctx);
        return 1;
    }

    std::printf("  arena_base=%#llx handler_va=%#llx tools_pc=%#llx tools_sz=%#llx\n",
                static_cast<unsigned long long>(arena.ArenaBaseVA()),
                static_cast<unsigned long long>(arena.HandlerVA()),
                static_cast<unsigned long long>(arena.ToolsPcVA()),
                static_cast<unsigned long long>(arena.ToolsSize()));

    uint32_t total_slots = 0;
    const auto &caves = arena.Caves();
    for (size_t i = 0; i < caves.size(); ++i) {
        total_slots += caves[i].capacity_instrs;
        std::printf("  cave[%2zu]: va=%#llx obj_base=%#llx obj_sz=%#x slots=%3u (%5u B)\n",
                    i,
                    static_cast<unsigned long long>(caves[i].va),
                    static_cast<unsigned long long>(caves[i].obj_base),
                    caves[i].obj_size,
                    caves[i].capacity_instrs,
                    caves[i].capacity_instrs * 16u);
    }

    std::vector<size_t> selected;
    const bool ok_sel = arena.SelectWorkingCaves(9, 144, selected);
    std::printf("  total_caves=%zu total_slots=%u select_9_ge_144=%s\n",
                caves.size(), total_slots, ok_sel ? "PASS" : "FAIL");

    auto eng = xsched::cuda::sm120::Sm120TrapEngine::Instance(ctx);
    const bool ok_eng = eng->Init();
    std::printf("  engine_init=%s has_bar0=%s\n",
                ok_eng ? "PASS" : "FAIL",
                eng->HasBar0() ? "true" : "false");
    eng->RestoreOriginal();

    cuCtxDestroy(ctx);
    return (ok_disc && ok_sel && ok_eng) ? 0 : 1;
}
