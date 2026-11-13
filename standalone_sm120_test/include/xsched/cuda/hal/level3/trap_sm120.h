#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>

#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda
{

struct Sm120RmCtxHandles {
    int      ctl_fd;
    uint32_t hClient;
    uint32_t hDevice;
    uint32_t hSubDevice;
    uint32_t hTsg;
    uint32_t hChannel;
    uint32_t hCompute;
    uint32_t hDebugger;
    uint32_t hDebuggerClient;
    uint32_t hShareClient;
};

class TrapSm120Context
{
public:
    TrapSm120Context() = default;
    ~TrapSm120Context();

    static TrapSm120Context &Instance();

    bool Init(CUcontext ctx, CUstream op_stream = nullptr);
    void BindContextSlot(int slot_index) { target_slot_ = slot_index; }
    void Arm(bool armed);
    bool Trigger();
    bool Resume();
    bool PostRecover();

    // Direct in-process RM 0x83de + 0xa06c0105 trap/preempt pulse
    bool RmFastTrapPulse83de();

    // Fine-grained 3-granularity hardware trap preemption (.text.traphandler + 0x0390 + BAR0 0x424684):
    //   Mode 1: Per-Stream Early-Exit + XSched Per-CTA Restore Flags
    //   Mode 2: Non-Abort Resumable Per-Stream Pause & In-Place Resume (Non-Idempotent Safe)
    //   Mode 3: Sub-Stream Per-CTA Selective Pause & In-Place Resume Inside a Single Kernel
    //   Mode 4: Signal an opt-in kernel to checkpoint and exit at a CTA-safe boundary.
    bool InstallFineGrainedTrapHandler(CUcontext ctx);
    bool BindKernelPreemptBuffer(CUfunction func, uint64_t preempt_buf_va, int64_t kernel_idx, uint64_t guardian_va = 0);
    // Bind while no launch of func is in flight. control_va is the existing mapped
    // command buffer, request_va is a separate, zero-initialized device uint32_t.
    // After requesting a checkpoint, retire the stream before clearing the two
    // words and relaunching. The caller owns the checkpoint and stream ordering.
    bool BindKernelCheckpointRequest(CUfunction func, uint64_t control_va, uint64_t request_va);
    bool RequestStreamCheckpoint(volatile uint32_t *control_host);
    // Mode 5 (Path A): Zero-source-modification arbitrary-PC hardware trap-frame + SMEM evict
    // and atomic RTT restore via Global Arena c[0][0x178].
    bool BindKernelTrapFrameEvict(CUfunction func, uint64_t control_va, uint64_t dump_stub_va);
    bool RequestStreamTrapFrameEvict(volatile uint32_t *control_host, int num_ctas = 0);
    bool PrepareKernelTrapFrameRestore(CUfunction func);
    bool RequestStreamTrapFrameRestore(volatile uint32_t *control_host, int num_ctas = 0);
    bool FinishKernelTrapFrameRestore();
    bool TriggerFineGrainedTrapPulse();
    bool EarlyExitStream(volatile uint32_t *preempt_buf_host);
    bool PauseStreamInPlace(volatile uint32_t *preempt_buf_host);
    bool ResumeStreamInPlace(volatile uint32_t *preempt_buf_host);
    bool ConfigureAndPauseCtasInPlace(volatile uint32_t *preempt_buf_host, const uint32_t *cta_modes, int num_ctas);
    bool RestoreOriginalTrapHandler();

    uint64_t TrapCounterValue() const;
    uint64_t ArmedFlagDevVA() const { return flag_dev_va_; }
    uint64_t CounterDevVA()   const { return counter_dev_va_; }
    bool     IsReady()        const { return ready_; }
    uint64_t LiveTrapVA()     const { return live_va_; }
    uint64_t CaveVA()         const { return cave_va_; }
    uint64_t TrampolineVA()   const { return tramp_va_; }

    static bool GetLatestRmHandles(Sm120RmCtxHandles *out);
    static int  GetCapturedContextCount();
    static bool GetRmHandlesByIndex(int idx, Sm120RmCtxHandles *out);

private:
    bool LocateTrapBody(uint64_t *out_base, uint64_t *out_live);
    bool PatchTrapBody();
    bool ConnectAgent();
    bool EnsureDebuggerHandle();

    CUcontext ctx_            = nullptr;
    int       target_slot_    = 0;
    uint64_t  base_va_        = 0;
    uint64_t  live_va_        = 0;
    uint64_t  slot20_va_      = 0;
    uint64_t  cave_va_        = 0;
    uint64_t  tramp_va_       = 0;

    volatile uint64_t *flag_host_    = nullptr;
    uint64_t           flag_dev_va_  = 0;
    volatile uint64_t *counter_host_ = nullptr;
    uint64_t           counter_dev_va_ = 0;

    uint64_t  orig_slot20_[2] = {0, 0};
    bool      patched_        = false;
    bool      ready_          = false;

    int       agent_sock_     = -1;
    std::mutex agent_mu_;
};

} // namespace xsched::cuda
