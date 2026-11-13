#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/level3/trap_sm120.h"

namespace xsched::cuda
{

enum class Sm120ResumableBackend : int {
    kAuto            = 0,
    kCilpTsgPreempt  = 1, // Path A: 0x20801210 CILP/CTA + 0xa06c0103 + 0xa06c0105 + 0xa06c0101
    kDebuggerSuspend = 2, // Path B: 0x83de0317 {1, 0} + 0x83de0318
};

class ResumableSm120Context
{
public:
    ResumableSm120Context(CUcontext ctx = nullptr);
    ~ResumableSm120Context() = default;

    static std::shared_ptr<ResumableSm120Context> Instance(CUcontext ctx);

    void BindContextSlot(int slot_index);
    bool EnsureHandles(int ctx_index = -1);

    bool Init(CUcontext ctx = nullptr,
              Sm120ResumableBackend backend = Sm120ResumableBackend::kAuto);

    bool Pause(uint64_t *latency_us_out = nullptr);
    bool Resume(uint64_t *latency_us_out = nullptr);

    bool SuspendOnly();
    bool ResumeOnly();
    bool PreemptPulse();

    void Interrupt() { Pause(nullptr); }
    void Restore()   { Resume(nullptr); }

    bool IsPaused() const { return paused_.load(std::memory_order_acquire); }
    Sm120ResumableBackend ActiveBackend() const { return active_backend_; }
    const char *ModeName() const;
    const Sm120RmCtxHandles &Handles() const { return handles_; }

private:
    bool InitCilpMode();
    bool PauseViaCilpTsg();
    bool ResumeViaCilpTsg();

    bool PauseViaDebugger();
    bool ResumeViaDebugger();

    CUcontext             ctx_            = nullptr;
    int                   ctx_slot_       = 0;
    Sm120RmCtxHandles     handles_        = {};
    Sm120ResumableBackend active_backend_ = Sm120ResumableBackend::kCilpTsgPreempt;
    std::atomic<bool>     paused_{false};
    bool                  initialized_    = false;
    bool                  cilp_enabled_   = false;
    bool                  dbg_suspended_  = false;
    std::mutex            mtx_;
};

} // namespace xsched::cuda
