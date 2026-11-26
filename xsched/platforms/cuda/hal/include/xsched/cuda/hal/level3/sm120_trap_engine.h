#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/level3/sm120_arena.h"
#include "xsched/cuda/hal/level3/sm120_guardian.h"

namespace xsched::cuda::sm120
{

/// @brief Simplified Blackwell SM120 preemption modes (v2).
enum class Sm120PreemptMode : uint32_t
{
    kEarlyExit = 1, // Mode 1: Idempotent fast early-exit + CTA-granular replay
    kTrapFrame = 5, // Mode 5: Non-idempotent universal trap-frame evict + BPT.TRAP restore
};

enum Sm120CommandWord : uint32_t
{
    kCmdNormal       = 0,
    kCmdMode1Evict   = 1,
    kCmdMode1Replay  = 2,
    kCmdMode5Evict   = 5,
    kCmdMode5Restore = 6,
};

enum Sm120CtaState : uint32_t
{
    kCtaStateDone   = 0, // State A (natural completion / cleared)
    kCtaStateReplay = 1, // Mode 1 replay required
    kCtaStateQueued = 2, // State C (drained before entering user code)
    kCtaStateSaved  = 5, // State B (trap frame + SMEM/LMEM/bars/regs saved in slot_id)
};

struct alignas(16) Sm120CtaEntry
{
    uint32_t state;
    uint32_t entry_dec;
    uint32_t slot_id;
    uint32_t warp_mask; // bit w = 1 iff warp w was alive at Mode 5 Evict
};
static_assert(sizeof(Sm120CtaEntry) == 16, "Sm120CtaEntry ABI mismatch");

/// @brief Zero-copy mapped host-device control block per XQueue/CUDA stream.
/// First 32 bytes [0x00..0x1f] hold per-round mutable run state and can be cleared
/// with a single 32-byte DMA transfer/memset after a restored kernel finishes.
struct alignas(16) Sm120ControlHeader
{
    uint32_t cmd;                 // +0x00: Sm120CommandWord
    uint32_t evicted_slots;       // +0x04: ATOMG.E.ADD counter for State B compact slot_id
    uint64_t active_kernel_idx;   // +0x08: Claimed 64-bit kernel_idx
    uint32_t restored_slots;      // +0x10: REDG.E.ADD counter on Cave 1 restore finish
    uint32_t drained_cnt;         // +0x14: Mode 5 State C exits
    uint32_t mode1_exit_cnt;      // +0x18: Mode 1 exits
    uint32_t is_restore_wave;     // +0x1c: 1 while Mode 1/5 restore wave active (cleared by 32B DMA)

    uint64_t arena_va;            // +0x20: Pooled device u_arena VA
    uint32_t slot_stride_bytes;   // +0x28: Bytes per evicted CTA slot in u_arena
    uint32_t max_slots;           // +0x2c: Capacity of u_arena in compact slots
    uint32_t thread_area_off;     // +0x30: Offset within slot to per-thread frame area
    uint32_t thread_frame_stride; // +0x34: Bytes per thread frame
    uint32_t smem_area_off;       // +0x38: Offset within slot to SMEM dump
    uint32_t lmem_area_off;       // +0x3c: Offset within slot to LMEM dump

    uint32_t lmem_cap_bytes;      // +0x40: Effective per-thread LMEM capacity
    uint32_t max_ctas;            // +0x44: Allocated CTA entry capacity in u_control
    uint32_t overflow_err;        // +0x48: Set if slot_id >= max_slots
    uint32_t trap_inflight;       // +0x4c: Active warps currently inside Cave 1 (cmd == 6)
};
static_assert(sizeof(Sm120ControlHeader) == 0x50, "Sm120ControlHeader ABI mismatch");

struct Sm120StreamTelemetry
{
    uint32_t cmd = 0;
    uint32_t evicted_slots = 0;
    uint32_t restored_slots = 0;
    uint32_t drained_cnt = 0;
    uint32_t mode1_exit_cnt = 0;
    uint32_t overflow_err = 0;
    uint32_t is_restore_wave = 0;
    uint64_t active_kernel_idx = 0;
    size_t   arena_bytes = 0;
};

/// @brief Context-wide static 9-cave SM120 trap handler engine.
/// Installed ONCE per CUcontext; zero runtime cave re-patching across heterogeneous streams.
class Sm120TrapEngine
{
public:
    static std::shared_ptr<Sm120TrapEngine> Instance(CUcontext ctx);
    static std::mutex &LaunchMutex(CUcontext ctx);

    Sm120TrapEngine(CUcontext ctx, CUdevice dev);
    ~Sm120TrapEngine();

    bool Init();
    bool IsReady() const { return ready_; }
    bool HasBar0() const { return arena_.HasBar0(); }
    bool FireEvictPulse();
    void RestoreOriginal();

    int SmCount() const { return sm_count_; }
    int MaxThreadsPerSm() const { return max_threads_per_sm_; }
    CUcontext Context() const { return ctx_; }
    const Sm120Arena &Arena() const { return arena_; }
    CUfunction CondClearFunc() const { return cond_clear_fn_; }

private:
    bool EmitStaticHandler();
    bool LoadHelperModule();

    CUcontext ctx_ = nullptr;
    CUdevice  dev_ = 0;
    int sm_count_ = 84;
    int max_threads_per_sm_ = 1536;
    bool ready_ = false;
    Sm120Arena arena_;
    CUmodule helper_mod_ = nullptr;
    CUfunction cond_clear_fn_ = nullptr;
    std::mutex mu_;
};

/// @brief Per-stream state owning zero-copy u_control and dynamic compact-slot pooled u_arena.
class Sm120StreamContext
{
public:
    Sm120StreamContext(CUcontext ctx, CUdevice dev, std::shared_ptr<Sm120TrapEngine> engine);
    ~Sm120StreamContext();

    /// @brief Ensure guardian instrumentation, capacity of u_control and pooled u_arena,
    /// and populate c[0][0x170..0x187] DebuggerParams before kernel launch.
    /// @note Callers launching the same CUfunction concurrently across threads must serialize
    /// PrepareLaunch + LaunchWrapper around the function's DebuggerParams.
    bool PrepareLaunch(CUfunction func,
                       uint32_t num_ctas,
                       uint32_t threads_per_cta,
                       uint32_t dyn_smem_bytes,
                       uint64_t kernel_idx,
                       Sm120PreemptMode mode,
                       CUstream stream = nullptr,
                       const void *launch_handle = nullptr);

    /// @brief Monotonic count of PrepareLaunch() calls (snapshot before a stream synchronization).
    uint64_t LaunchSeq() const;

    /// @brief Write {cmd, is_restore_wave} into every gate mirror line (stream-ordered; caller holds mu_
    /// and synchronizes the stream before firing a pulse / relaunching).
    void WriteGateMirrorAsync(uint32_t cmd, uint32_t is_restore_wave, CUstream stream);
    /// @brief Write only the cmd word of every gate mirror line (PreemptStream).
    void WriteGateMirrorCmdAsync(uint32_t cmd, CUstream stream);

    /// @brief Phase 3.3 arena trimming, called after the stream was synchronized. If no launch was
    /// prepared since `seq_before_sync` and no eviction/restore state is pending, frees arenas
    /// retired by growth and, if the current arena stayed > 2x (and > 64 MiB) above the largest
    /// requirement for a window of synchronizations, frees it too (re-allocated on demand).
    /// @return bytes released.
    size_t TrimArenas(uint64_t seq_before_sync);

    /// @brief Zero c[0][0x170..0x187] so an instrumented function bypasses the Guardian cleanly.
    /// No-op (no driver call) if the window is already zero.
    static void ClearDebuggerParams(CUfunction func);

    /// @brief Whether a function launched through `launch_handle` (the CUkernel / CUfunction passed
    /// to cuLaunchKernel) may still hold non-zero Mode 5 debugger params. Lets pass-through
    /// launches skip resolving the CUfunction. Call with the context's LaunchMutex held.
    static bool MaybeParamsSet(const void *launch_handle);

    /// @brief Evict running/queued waves on this stream with 1 single BAR0 pulse.
    bool PreemptStream(Sm120PreemptMode mode);

    /// @brief Configure control header for 0-pulse BPT.TRAP restore / Mode 1 replay.
    /// Returns active_kernel_idx (0 if none was recorded).
    uint64_t PrepareRestore(Sm120PreemptMode mode);

    /// @brief Asynchronously clear the 32-byte mutable run-state header after restored kernel completes.
    bool EnqueuePostRestoreClear(CUstream stream);

    /// @brief Explicitly reset mutable header & CtaEntries (useful in standalone tests).
    void ResetControlSync();

    Sm120StreamTelemetry Telemetry() const;
    bool ReadCtaEntries(std::vector<Sm120CtaEntry> &out, uint32_t count) const;

    /// @brief Diagnostics (XSCHED_SM120_TRACE_LAUNCH): after a Mode 5 evict, log up to
    /// `max_ctas` State-B CTAs of the active kernel with slot id, warp mask and the saved
    /// per-lane TRAP_RETURN_PC expressed as an offset into the ORIGINAL kernel SASS
    /// (pc - ep_inst - kGuardianBytes), for correlation with `nvdisasm -b SM120`.
    /// The trap-frame layout is recomputed from `rec` (the header layout slot may already have
    /// been overwritten by queued kernels' stream-ordered layout DMAs during the drain).
    void DebugDumpEvicted(const Sm120KernelRecord *rec, uint32_t num_ctas, uint32_t threads_per_cta,
                          uint32_t dyn_smem_bytes, uint32_t max_ctas) const;
    CUdeviceptr ControlDevVA() const { return ctrl_dev_; }
    volatile uint32_t *ControlHostPtr() const { return reinterpret_cast<volatile uint32_t *>(ctrl_host_); }

private:
    bool EnsureControlCapacity(uint32_t num_ctas);
    bool EnsureArenaPool(size_t required_bytes);
    bool ReadHeaderLocked(Sm120ControlHeader *out) const;
    bool WriteHeaderSliceLocked(size_t offset, const void *src, size_t bytes);

    CUcontext ctx_ = nullptr;
    CUdevice  dev_ = 0;
    std::shared_ptr<Sm120TrapEngine> engine_;
    std::shared_ptr<Sm120GuardianRegistry> guardian_reg_;

    // VRAM-resident control header + CtaEntry[] (GPU L2-cached, zero PCIe traffic in Guardian/Trap!)
    CUdeviceptr ctrl_dev_ = 0;
    size_t ctrl_bytes_ = 0;
    uint32_t ctrl_cta_cap_ = 0;
    // Pinned 80B host staging buffer + dedicated high-priority Copy Engine stream for DMA header sync
    Sm120ControlHeader *ctrl_host_ = nullptr;
    CUstream ctrl_op_stream_ = nullptr;

    CUdeviceptr arena_dev_ = 0;
    size_t arena_bytes_ = 0;
    uint64_t launch_seq_ = 0;                 // PrepareLaunch() calls
    size_t window_max_required_ = 0;          // largest arena requirement in the trim window
    uint32_t window_syncs_ = 0;               // synchronizations in the trim window
    std::vector<CUdeviceptr> retired_arenas_;
    std::vector<CUdeviceptr> retired_ctrls_;
    uint8_t *layout_stage_ = nullptr;
    size_t layout_stage_cursor_ = 0;
    // Last arena layout enqueued to the control header (Phase 3.3): the 36 B HtoD copy is skipped
    // when the next launch on the same stream would write identical bytes to the same header.
    // Device code never writes these header fields; header reallocation changes ctrl_dev_.
    uint8_t last_layout_[36] = {};
    CUdeviceptr last_layout_ctrl_ = 0;
    CUstream last_layout_stream_ = nullptr;
    CUdeviceptr saved_arena_dev_ = 0;
    bool is_restore_cmd_active_ = false;
    CUstream last_stream_ = nullptr;
    std::map<uint64_t, CUdeviceptr> kernel_arena_map_;

    uint64_t last_kernel_idx_ = 0;
    mutable std::mutex mu_;
};

} // namespace xsched::cuda::sm120
