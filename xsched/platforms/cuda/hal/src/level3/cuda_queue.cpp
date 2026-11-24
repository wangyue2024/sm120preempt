#include <cstdlib>
#include <mutex>

#include "xsched/utils/env.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/level3/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

bool CudaQueueLv3Trap::IsSm120Stream(CUstream stream)
{
    (void)stream;
    CUdevice dev = 0;
    if (Driver::CtxGetDevice(&dev) != CUDA_SUCCESS) return false;
    int major = 0, minor = 0;
    if (Driver::DeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS ||
        Driver::DeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS) {
        return false;
    }
    return (major * 10 + minor) == 120;
}

sm120::Sm120PreemptMode CudaQueueLv3Trap::SelectSm120Mode(const std::shared_ptr<HwCommand> &hw_cmd) const
{
    if (const char *env = getenv("XSCHED_SM120_FORCE_MODE1")) {
        if (env[0] == '1') return sm120::Sm120PreemptMode::kEarlyExit;
    }
    if (hw_cmd != nullptr && hw_cmd->GetProps(kCommandPropertyIdempotent) != 0) {
        return sm120::Sm120PreemptMode::kEarlyExit;
    }
    return sm120::Sm120PreemptMode::kTrapFrame;
}

CudaQueueLv3Trap::CudaQueueLv3Trap(CUstream stream)
    : CudaQueueLv2(stream, !IsSm120Stream(stream))
    , is_sm120_(instrument_manager_ == nullptr)
{
    if (is_sm120_) {
        sm120_engine_ = sm120::Sm120TrapEngine::Instance(context_);
        sm120_stream_ = std::make_unique<sm120::Sm120StreamContext>(
            context_, cudevice_, sm120_engine_);
    } else {
        interrupt_context_ = InterruptContext::Instance(context_);
    }
}

void CudaQueueLv3Trap::Launch(std::shared_ptr<HwCommand> hw_cmd)
{
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
    if (is_sm120_) {
        if (kernel != nullptr) {
            last_kernel_ = hw_cmd;
            relaunched_ = true;
            const auto mode = SelectSm120Mode(hw_cmd);
            const uint64_t kidx = static_cast<uint64_t>(hw_cmd->GetIdx());
            std::lock_guard<std::mutex> lock(sm120::Sm120TrapEngine::LaunchMutex(context_));
            if (level_ >= kPreemptLevelDeactivate) {
                XASSERT(sm120_stream_->PrepareLaunch(
                            kernel->kFuncHandle, kernel->BlockCnt(),
                            kernel->ThreadCnt(), kernel->DynSmemBytes(),
                            kidx, mode, kStream),
                        "sm120 PrepareLaunch failed");
            } else {
                sm120::Sm120StreamContext::ClearDebuggerParams(kernel->kFuncHandle);
            }
            CUDA_ASSERT(kernel->LaunchWrapper(kStream));
            return;
        }
        CudaQueueLv1::Launch(hw_cmd);
        return;
    }

    if (kernel != nullptr) {
        last_kernel_ = hw_cmd;
        relaunched_ = true;
        XINFO("trap queue: remembered kernel cmd idx=%ld", (long)hw_cmd->GetIdx());
    }
    CudaQueueLv2::Launch(hw_cmd);
}

void CudaQueueLv3Trap::Deactivate()
{
    XASSERT(level_ >= kPreemptLevelDeactivate, "Deactivate() not supported on level-%d", level_);
    if (is_sm120_) {
        // Suspend() invokes Deactivate() and then Interrupt(). Only trigger Mode 1 early-exit
        // directly when operating strictly at level-2; level-3 triggers once inside Interrupt().
        if (level_ == kPreemptLevelDeactivate) {
            last_preempt_mode_ = sm120::Sm120PreemptMode::kEarlyExit;
            sm120_stream_->PreemptStream(last_preempt_mode_);
        }
        return;
    }
    CudaQueueLv2::Deactivate();
}

void CudaQueueLv3Trap::Interrupt()
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Interrupt() not supported on level-%d", level_);
    if (is_sm120_) {
        last_preempt_mode_ = SelectSm120Mode(last_kernel_);
        sm120_stream_->PreemptStream(last_preempt_mode_);
        return;
    }
    // wait until the preempt flag is set
    XINFO("Interrupt: syncing op stream");
    CUDA_ASSERT(Driver::StreamSynchronize(instrument_manager_->OpStream()));
    XINFO("Interrupt: op stream idle, triggering");
    // FIXME: what if multiple threads call Interrupt()?
    interrupt_context_->Interrupt();
}

void CudaQueueLv3Trap::Restore(const CommandLog &)
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Restore() not supported on level-%d", level_);
    if (is_sm120_) {
        // SM120 v2 restores cleanly in Reactivate(log) via BPT.TRAP 0x1 with zero BAR0 pulses.
        return;
    }
}

void CudaQueueLv3Trap::Reactivate(const CommandLog &log)
{
    if (is_sm120_) {
        this->Synchronize();
        const auto t = sm120_stream_->Telemetry();
        const bool has_preempted =
            (t.evicted_slots > 0 || t.drained_cnt > 0 || t.mode1_exit_cnt > 0);
        if (!has_preempted) {
            // Stream had no evicted or drained CTAs (idle or completed before the pulse).
            // Reset control state back to cmd=0 so future launches are never poisoned
            // with cmd=6 (kCmdMode5Restore), which would cause state==0 CTAs to exit early.
            sm120_stream_->ResetControlSync();
            return;
        }
        const uint64_t active_idx = sm120_stream_->PrepareRestore(last_preempt_mode_);
        bool cleared_post_active = false;
        for (const auto &cmd : log) {
            if (active_idx != 0 && static_cast<uint64_t>(cmd->GetIdx()) < active_idx) continue;
            this->Launch(cmd);
            if (!cleared_post_active &&
                (active_idx == 0 || static_cast<uint64_t>(cmd->GetIdx()) == active_idx)) {
                sm120_stream_->EnqueuePostRestoreClear(kStream);
                cleared_post_active = true;
            }
        }
        if (!cleared_post_active && log.empty() && last_kernel_ != nullptr) {
            this->Launch(last_kernel_);
            sm120_stream_->EnqueuePostRestoreClear(kStream);
            cleared_post_active = true;
        }
        if (!cleared_post_active) {
            sm120_stream_->ResetControlSync();
        }
        return;
    }

    relaunched_ = false;
    CudaQueueLv2::Reactivate(log);
    XINFO("trap reactivate: log=%zu last_kernel=%p relaunched=%d", log.size(),
          (void *)last_kernel_.get(), (int)relaunched_);
    if (relaunched_ || last_kernel_ == nullptr) return;
    this->Launch(last_kernel_);
}

void CudaQueueLv3Trap::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    if (is_sm120_) {
        if (level >= kPreemptLevelDeactivate) {
            sm120_engine_->Init();
        }
        level_ = level;
        return;
    }
    if (level == kPreemptLevelInterrupt) {
        instrument_manager_->NotifyTrapInstrumented();
        interrupt_context_->InstrumentTrapHandler();
    }
    level_ = level;
}

void CudaQueueLv3Trap::OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    if (is_sm120_) {
        this->CudaQueueLv1::OnHwCommandSubmit(hw_cmd);
        if (level_ < kPreemptLevelDeactivate) return;

        if (std::dynamic_pointer_cast<CudaGraphCommand>(hw_cmd) != nullptr) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                XWARN("CUDA graph cannot support trap-based level-3 preemption, "
                      "falling back to level-1");
            }
            return;
        }
        auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
        if (kernel != nullptr) {
            sm120::Sm120GuardianRegistry::Instance(context_)->EnsureInstrumented(kernel->kFuncHandle);
            kernel->killable = true;
            last_kernel_ = hw_cmd;
        }
        return;
    }

    this->CudaQueueLv2::OnHwCommandSubmit(hw_cmd);
    if (level_ < kPreemptLevelInterrupt) return;

    if (std::dynamic_pointer_cast<CudaGraphCommand>(hw_cmd) != nullptr) {
        // do nothing here, will automatically fallback to wait-based preemption
        static bool warned = false;
        if (!warned) {
            warned = true;
            XWARN("CUDA graph cannot support trap-based level-3 preemption, "
                  "falling back to level-1");
        }
        return;
    }
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
    // TODO: assign kernel_command->killable
    if (kernel != nullptr) {
        kernel->killable = true;
        last_kernel_ = hw_cmd;
        XINFO("trap queue: remembered kernel cmd idx=%ld", (long)hw_cmd->GetIdx());
    }
}

CudaQueueLv3Tsg::CudaQueueLv3Tsg(CUstream stream): CudaQueueLv1(stream)
{
    tsg_context_ = TsgContext::Instance(context_);
}

void CudaQueueLv3Tsg::Interrupt()
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Interrupt() not supported on level-%d", level_);
    tsg_context_->Interrupt();
}

void CudaQueueLv3Tsg::Restore(const CommandLog &)
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Restore() not supported on level-%d", level_);
    tsg_context_->Restore();
}

void CudaQueueLv3Tsg::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    level_ = level;
}

CudaQueueLv3Resumable::CudaQueueLv3Resumable(CUstream stream): CudaQueueLv1(stream)
{
    resumable_context_ = ResumableSm120Context::Instance(context_);
}

void CudaQueueLv3Resumable::Interrupt()
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Interrupt() not supported on level-%d", level_);
    resumable_context_->Interrupt();
}

void CudaQueueLv3Resumable::Restore(const CommandLog &)
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Restore() not supported on level-%d", level_);
    resumable_context_->Restore();
}

void CudaQueueLv3Resumable::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    level_ = level;
}
