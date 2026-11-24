#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/level3/sm120_sass.h"

namespace xsched::cuda::sm120
{

/// @brief Executable code cave located after an official sub-object's `BRA self`.
struct ArenaCave
{
    uint64_t va = 0;
    uint64_t obj_base = 0;
    uint32_t obj_size = 0;
    uint32_t capacity_instrs = 0;
};

/// @brief Discovers, validates, patches, and restores Blackwell SM120 .text.traphandler
/// arena code caves via libcuxtra, and exposes direct PCIe BAR0 trap pulse triggering.
class Sm120Arena
{
public:
    Sm120Arena(CUcontext ctx);
    ~Sm120Arena();

    bool Discover(bool verify_patch_site = true);
    bool WriteCave(size_t cave_idx, const std::vector<sass::Ins> &code);
    bool PatchHandlerHook(uint64_t target_va);
    void RestoreAll();

    bool EnsureBar0Mapped();
    bool HasBar0() const { return bar0_ != nullptr; }
    bool FireEvictPulse();

    bool     IsDiscovered()    const { return discovered_; }
    bool     IsHookPatched()   const { return hook_patched_; }
    CUcontext Context()        const { return ctx_; }
    uint64_t ArenaBaseVA()     const { return arena_base_va_; }
    uint64_t HandlerVA()       const { return handler_va_; }
    uint64_t ToolsPcVA()       const { return tools_pc_; }
    uint64_t ToolsSize()       const { return tools_sz_; }
    const std::vector<ArenaCave> &Caves() const { return caves_; }

    /// @brief Select the canonical 7-cave working set (tools cave first, then live_handler,
    /// then 5 additional sub-object caves with >= min_slots instructions).
    bool SelectWorkingCaves(size_t needed, uint32_t min_slots, std::vector<size_t> &out_indices) const;

private:
    struct PatchedRegion
    {
        uint64_t va = 0;
        std::vector<uint64_t> backup;
    };

    CUcontext ctx_ = nullptr;
    CUstream op_stream_ = nullptr;
    bool discovered_ = false;
    bool hook_patched_ = false;
    uint64_t arena_base_va_ = 0;
    uint64_t handler_va_ = 0;
    uint64_t tools_pc_ = 0;
    uint64_t tools_sz_ = 0;
    uint64_t orig_390_[2] = {};
    uint64_t call_380_[2] = {};

    std::vector<ArenaCave> caves_;
    std::vector<PatchedRegion> patched_caves_;

    volatile uint32_t *bar0_ = nullptr;
    int bar0_fd_ = -1;
    std::mutex mu_;
};

} // namespace xsched::cuda::sm120
