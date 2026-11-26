#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda { class InstrMemAllocator; }

namespace xsched::cuda::sm120
{

inline constexpr size_t   kGuardianBytes          = 0x500; // 80 SASS slots (128B aligned)
inline constexpr uint32_t kGuardianBarOff         = 39u * 16u; // 0x270: BAR.SYNC 0 gate offset
inline constexpr uint32_t kGuardianRestorePathOff = 44u * 16u; // 0x2c0: Restore/Replay path offset
inline constexpr uint32_t kGuardianReplayCIdxOff  = 59u * 16u; // 0x3b0: replay_state_c entry offset
inline constexpr uint32_t kGuardianReplayBarOff   = 76u * 16u; // 0x4c0: Final BAR.SYNC 0 before exit/fallthrough
inline constexpr uint32_t kKernelDescMagic        = 0x534D3132u; // "SM12"

/// @brief Immutable device-resident metadata bound to c[0][0x178..0x17f] per CUfunction.
struct alignas(16) Sm120KernelDesc
{
    uint64_t ep_inst_va;       // +0x00: Instrumented entry point VA
    uint32_t guardian_bar_off; // +0x08: Byte offset of BAR.SYNC 0 inside Guardian
    uint32_t lmem_static_bytes;// +0x0c: Align16(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
    uint32_t lmem_cap_bytes;   // +0x10: Max per-thread LMEM snapshot capacity (0 = disabled)
    uint32_t bar_cnt;          // +0x14: Clamped >= 1
    uint32_t flags;            // +0x18: bit0=has_lmem, bit1=has_device_calls, bit2=has_hw_mbarrier
    uint32_t magic;            // +0x1c: kKernelDescMagic
};
static_assert(sizeof(Sm120KernelDesc) == 32, "Sm120KernelDesc ABI mismatch");

struct Sm120KernelRecord
{
    CUfunction    func = nullptr;
    CUdeviceptr   ep_orig = 0;
    CUdeviceptr   ep_inst = 0;
    CUdeviceptr   kdesc_dev = 0;
    Sm120KernelDesc desc{};
    int           raw_num_regs = 0;
    int           clamped_regs = 0;
    int           static_smem_bytes = 0;
    bool          has_hw_mbarrier = false;
    bool          has_calls = false;
    bool          loop_free = false;  // no backward / indirect branches (short, bounded CTAs)
};

/// @brief Permanently instruments each managed CUfunction once on first launch with
/// the v2 branchless 3D Entry Guardian (0 state-C leaks, BPT.TRAP 0x1 self-restore).
class Sm120GuardianRegistry
{
public:
    static std::shared_ptr<Sm120GuardianRegistry> Instance(CUcontext ctx);

    Sm120GuardianRegistry(CUcontext ctx, CUdevice dev);
    ~Sm120GuardianRegistry();

    const Sm120KernelRecord *EnsureInstrumented(CUfunction func);
    const Sm120KernelRecord *Find(CUfunction func);
    size_t InstrumentedCount();

private:
    CUcontext ctx_ = nullptr;
    CUdevice  dev_ = 0;
    CUstream  op_stream_ = nullptr;
    std::unique_ptr<InstrMemAllocator> instr_mem_;
    std::mutex mu_;
    std::unordered_map<CUfunction, Sm120KernelRecord> records_;
};

} // namespace xsched::cuda::sm120
