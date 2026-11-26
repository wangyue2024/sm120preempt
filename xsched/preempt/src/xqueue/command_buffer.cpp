#include "xsched/utils/xassert.h"
#include "xsched/preempt/sched/agent.h"
#include "xsched/preempt/xqueue/command_buffer.h"
#include "xsched/preempt/xqueue/spin.h"

using namespace xsched::sched;
using namespace xsched::preempt;

CommandBuffer::CommandBuffer(XQueueHandle xq_h): kXQueueHandle(xq_h)
{
    Enqueue(std::make_shared<XQueueWaitAllCommand>());
    XASSERT(last_enqueued_cmd_ != nullptr, "last_enqueued_cmd_ should not be nullptr");
}

void CommandBuffer::WaitForNextIdle()
{
    std::unique_lock<std::mutex> lock(mtx_);
    if (xq_state_ == kQueueStateIdle) return;
    uint64_t current_idle_cnt = idle_cnt_;
    while (current_idle_cnt == idle_cnt_) idle_cv_.wait(lock);
}

XQueueState CommandBuffer::GetXQueueState()
{
    std::lock_guard<std::mutex> lock(mtx_);
    return xq_state_;
}

void CommandBuffer::ForEachCommand(std::function<bool (std::shared_ptr<XCommand>)> func)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (last_dequeued_cmd_ != nullptr && last_dequeued_cmd_->GetState() < kCommandStateCompleted) {
        func(last_dequeued_cmd_);
    }
    for (auto xcmd : cmds_) {
        if (!func(xcmd)) break;
    }
}

std::shared_ptr<XCommand> CommandBuffer::Dequeue()
{
    std::unique_lock<std::mutex> lock(mtx_);

    auto pop_front = [this]() {
        auto xcmd = cmds_.front();
        cmds_.pop_front();
        pending_.fetch_sub(1, std::memory_order_relaxed);
        last_dequeued_cmd_ = xcmd;
        const auto type = xcmd->GetType();
        worker_holds_ordered_ = (type == kCommandTypeHardware || type == kCommandTypeHostFunction);
        XDEBG("xcmd (%p) dequeued from cmd buf (%p)", xcmd.get(), this);
        return xcmd;
    };

    while (true) {
        // Never dequeue while a direct launch is in progress: it is ordered before anything that
        // gets enqueued meanwhile.
        while (direct_gate_) cv_.wait(lock);

        if (!cmds_.empty()) return pop_front();

        XDEBG("cmd buf (%p) emptied, last_enqueued_cmd_ (%p), type (%d)",
              this, last_enqueued_cmd_.get(), last_enqueued_cmd_->GetType());
        switch (last_enqueued_cmd_->GetType())
        {
        case kCommandTypeXQueueWaitAll:
        {
            if (xq_state_ == kQueueStateReady) {
                xq_state_ = kQueueStateIdle;
                idle_cnt_++;
                idle_cv_.notify_all();
                SchedAgent::SendEvent(std::make_shared<XQueueIdleEvent>(kXQueueHandle));
            }

            // Sleep until a command is enqueued or a direct launch happened (after which the
            // last enqueued command is a hardware command that needs tracking).
            const uint64_t seq = direct_seq_.load(std::memory_order_relaxed);
            if (SpinBudgetUs() > 0) {
                // Bounded spin before sleeping: back-to-back submissions avoid a futex wake-up.
                lock.unlock();
                SpinUntil([this, seq]() {
                    return pending_.load(std::memory_order_acquire) > 0 ||
                           direct_seq_.load(std::memory_order_acquire) != seq;
                });
                lock.lock();
            }
            while (cmds_.empty() && direct_seq_.load(std::memory_order_relaxed) == seq) {
                worker_asleep_ = true;
                cv_.wait(lock);
                worker_asleep_ = false;
            }
            continue;
        }

        case kCommandTypeBatchSynchronize:
        {
            auto xcmd = std::make_shared<XQueueWaitAllCommand>();
            xcmd->SetState(kCommandStatePending);
            last_enqueued_cmd_ = xcmd;
            last_dequeued_cmd_ = xcmd;
            worker_holds_ordered_ = false;
            XDEBG("generated XQueueWaitAllCommand (%p) dequeued from cmd buf (%p)", xcmd.get(), this);
            return xcmd;
        }

        default:
        {
            // About to generate a background batch synchronization, which blocks the worker until
            // the oldest in-flight command completes. Commands that arrive meanwhile could only be
            // launched afterwards, so first give back-to-back submissions a short window.
            if (SpinBudgetUs() > 0 && pending_.load(std::memory_order_acquire) == 0) {
                lock.unlock();
                SpinUntil([this]() { return pending_.load(std::memory_order_acquire) > 0; });
                lock.lock();
                if (!cmds_.empty() || direct_gate_) continue;
            }
            auto xcmd = std::make_shared<BatchSynchronizeCommand>();
            xcmd->SetState(kCommandStatePending);
            last_enqueued_cmd_ = xcmd;
            last_dequeued_cmd_ = xcmd;
            worker_holds_ordered_ = false;
            XDEBG("generated BatchSynchronizeCommand (%p) dequeued from cmd buf (%p)",
                  xcmd.get(), this);
            return xcmd;
        }
        }
    }

    XASSERT(false, "should not reach here");
    return nullptr;
}

void CommandBuffer::WorkerDone()
{
    std::lock_guard<std::mutex> lock(mtx_);
    worker_holds_ordered_ = false;
}

bool CommandBuffer::AllOrderedLaunched()
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (worker_holds_ordered_ || direct_gate_) return false;
    for (const auto &xcmd : cmds_) {
        const auto type = xcmd->GetType();
        if (type == kCommandTypeHardware || type == kCommandTypeHostFunction) return false;
    }
    return true;
}

bool CommandBuffer::TryBeginDirect(std::shared_ptr<XCommand> xcmd, bool allow_awake)
{
    auto ready_time = std::chrono::system_clock::now();
    std::lock_guard<std::mutex> lock(mtx_);
    if (!cmds_.empty() || worker_holds_ordered_ || direct_gate_) return false;
    // Direct submit only pays off when handing over would need a futex wake-up of a sleeping
    // worker; an awake (spinning) worker takes the command almost for free and pipelines the
    // driver launch cost with this thread.
    // An awake worker is still taken over when the caller predicts a single-command call (the
    // previous call between two WaitAll()s had one command): no burst to pipeline with.
    if (!worker_asleep_ && !allow_awake) return false;
    direct_gate_ = true;
    xcmd->SetState(kCommandStatePending);
    if (xcmd->GetType() == kCommandTypeHardware && xq_state_ == kQueueStateIdle) {
        xq_state_ = kQueueStateReady;
        SchedAgent::SendEvent(std::make_shared<XQueueReadyEvent>(kXQueueHandle, ready_time));
    }
    return true;
}

void CommandBuffer::EndDirect(std::shared_ptr<XCommand> xcmd, bool launched)
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        direct_gate_ = false;
        last_enqueued_cmd_ = xcmd;
        if (launched) {
            last_dequeued_cmd_ = xcmd;
            direct_seq_.fetch_add(1, std::memory_order_release);
        } else {
            cmds_.emplace_back(xcmd);
            pending_.fetch_add(1, std::memory_order_release);
        }
    }
    cv_.notify_all();
}

void CommandBuffer::Enqueue(std::shared_ptr<XCommand> xcmd)
{
    auto ready_time = std::chrono::system_clock::now();
    XASSERT(xcmd != nullptr, "xcmd should not be nullptr");
    xcmd->SetState(kCommandStatePending);

    mtx_.lock();

    if (xcmd->GetType() == kCommandTypeHardware && xq_state_ == kQueueStateIdle) {
        xq_state_ = kQueueStateReady;
        SchedAgent::SendEvent(std::make_shared<XQueueReadyEvent>(kXQueueHandle, ready_time));
    }

    last_enqueued_cmd_ = xcmd;
    cmds_.emplace_back(xcmd);
    pending_.fetch_add(1, std::memory_order_release);
    XDEBG("xcmd (%p) type (%d) enqueued to cmd buf (%p)", xcmd.get(), xcmd->GetType(), this);

    mtx_.unlock();
    cv_.notify_all();
}

void CommandBuffer::DropAll()
{
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto &xcmd : cmds_) xcmd->SetState(kCommandStateCompleted);
    cmds_.clear();
    pending_.store(0, std::memory_order_release);
}

std::shared_ptr<XQueueWaitAllCommand> CommandBuffer::EnqueueXQueueWaitAllCommand()
{
    std::unique_lock<std::mutex> lock(mtx_);

    if (xq_state_ == kQueueStateIdle) {
        // If the XQueue is idle, syncing on this XQueue should return. Here, we should return
        // a completed XCommand so that syncing on this XQueue will return immediately.
        auto wait_all_cmd = std::dynamic_pointer_cast<XQueueWaitAllCommand>(last_enqueued_cmd_);
        XASSERT(last_enqueued_cmd_->GetType() == kCommandTypeXQueueWaitAll &&
                wait_all_cmd != nullptr,
                "last_enqueued_cmd_ must be an XQueueWaitAllCommand if the XQueue is idle");
        return wait_all_cmd;
    }

    if (last_enqueued_cmd_->GetType() == kCommandTypeXQueueWaitAll) {
        auto wait_all_cmd = std::dynamic_pointer_cast<XQueueWaitAllCommand>(last_enqueued_cmd_);
        XASSERT(wait_all_cmd != nullptr, "last_enqueued_cmd_ must be an XQueueWaitAllCommand");
        return wait_all_cmd;
    }

    auto xcmd = std::make_shared<XQueueWaitAllCommand>();
    xcmd->SetState(kCommandStatePending);
    last_enqueued_cmd_ = xcmd;
    cmds_.emplace_back(xcmd);
    pending_.fetch_add(1, std::memory_order_release);

    lock.unlock();
    cv_.notify_all();
    return xcmd;
}
