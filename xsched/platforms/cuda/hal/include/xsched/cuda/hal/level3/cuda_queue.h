#pragma once

#include <mutex>

#include "xsched/cuda/hal/level2/cuda_queue.h"
#include "xsched/cuda/hal/level3/interrupt.h"
#include "xsched/cuda/hal/level3/tsg.h"
#include "xsched/cuda/hal/level3/resumable_sm120.h"
#include "xsched/cuda/hal/level3/sm120_trap_engine.h"

namespace xsched::cuda
{

class CudaQueueLv3Trap : public CudaQueueLv2
{
public:
    CudaQueueLv3Trap(CUstream stream);
    virtual ~CudaQueueLv3Trap() = default;

    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Synchronize() override;
    virtual void Deactivate() override;
    virtual void Interrupt() override;
    virtual void Restore(const preempt::CommandLog &log) override;
    virtual void Reactivate(const preempt::CommandLog &log) override;
    virtual void OnPreemptLevelChange(XPreemptLevel level) override;
    virtual void OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> hw_cmd) override;

    virtual bool          SupportDynamicLevel()  override { return true; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelInterrupt; }

    sm120::Sm120StreamContext *Sm120Stream() const { return sm120_stream_.get(); }

protected:
    static bool IsSm120Stream(CUstream stream);
    sm120::Sm120PreemptMode SelectSm120Mode(const std::shared_ptr<preempt::HwCommand> &hw_cmd) const;

    std::shared_ptr<InterruptContext> interrupt_context_ = nullptr;
    const bool is_sm120_ = false;
    std::shared_ptr<sm120::Sm120TrapEngine> sm120_engine_ = nullptr;
    std::unique_ptr<sm120::Sm120StreamContext> sm120_stream_ = nullptr;
    sm120::Sm120PreemptMode last_preempt_mode_ = sm120::Sm120PreemptMode::kTrapFrame;
    int sm_count_ = 0;
    mutable std::mutex last_kernel_mtx_;
    std::shared_ptr<preempt::HwCommand> last_kernel_ = nullptr;
    bool relaunched_ = false;   // Reactivate relaunched a kernel this round
};

class CudaQueueLv3Tsg : public CudaQueueLv1
{
public:
    CudaQueueLv3Tsg(CUstream stream);
    virtual ~CudaQueueLv3Tsg() = default;

    virtual void Interrupt() override;
    virtual void Restore(const preempt::CommandLog &log) override;
    virtual void OnPreemptLevelChange(XPreemptLevel level) override;

    virtual bool          SupportDynamicLevel()  override { return true; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelInterrupt; }

private:
    std::shared_ptr<TsgContext> tsg_context_ = nullptr;
};

class CudaQueueLv3Resumable : public CudaQueueLv1
{
public:
    CudaQueueLv3Resumable(CUstream stream);
    virtual ~CudaQueueLv3Resumable() = default;

    virtual void Interrupt() override;
    virtual void Restore(const preempt::CommandLog &log) override;
    virtual void OnPreemptLevelChange(XPreemptLevel level) override;

    virtual bool          SupportDynamicLevel()  override { return true; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelInterrupt; }

private:
    std::shared_ptr<ResumableSm120Context> resumable_context_ = nullptr;
};

} // namespace xsched::cuda
