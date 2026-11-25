#include "xsched/utils/xassert.h"
#include "xsched/preempt/xqueue/xcommand.h"
#include "xsched/preempt/xqueue/spin.h"

using namespace xsched::preempt;

XCommandState XCommand::GetState()
{
    std::lock_guard<std::mutex> lock(mtx_);
    return state_;
}

void XCommand::SetState(XCommandState new_state)
{
    XASSERT(new_state >= kCommandStateCreated && new_state <= kCommandStateCompleted,
            "invalid state: %d", new_state);
    mtx_.lock();
    XCommandState old_state = state_;
    XASSERT(new_state >= old_state, "state should not go back");
    state_ = new_state;
    state_atomic_.store(static_cast<int>(new_state), std::memory_order_release);
    mtx_.unlock();

    if (new_state == old_state) return; // only notify when state changes
    state_cv_.notify_all();
    for (int i = old_state + 1; i <= new_state; i++) {
        for (auto listener : state_listeners_) listener(XCommandState(i));
    }
}

void XCommand::WaitUntil(XCommandState state)
{
    // Bounded spin first (fast completions / short GPU work), then block on the condition variable.
    if (SpinUntil([&]() { return state_atomic_.load(std::memory_order_acquire) >= static_cast<int>(state); })) {
        return;
    }
    std::unique_lock<std::mutex> lock(mtx_);
    while ((int)state_ < (int)state) state_cv_.wait(lock);
}
