#pragma once

#include <map>
#include <unordered_map>

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched
{

class HighestPriorityFirstPolicy : public Policy
{
public:
    HighestPriorityFirstPolicy(): Policy(kPolicyHighestPriorityFirst) {}
    virtual ~HighestPriorityFirstPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    Priority GetPriority(XQueueHandle handle);
    std::unordered_map<XQueueHandle, Priority> priorities_;
    // Phase 3.5 resume delay (XSCHED_HPF_RESUME_DELAY_US): last time each priority level had a ready
    // XQueue, per device.
    std::map<XDevice, std::map<Priority, TimePoint>> last_ready_;
    // DEF-41 (Phase 3.5): Level-3 peer rotation. Two processes running Level-3 queues at the same time
    // are two time-sliced contexts; a context switch while one has warps in the trap handler times
    // out (Xid 109). At the top priority of a device only one process's Level-3 queues are resumed at
    // a time; ownership rotates every XSCHED_HPF_LV3_QUANTUM_US (default 20000) among processes with
    // ready Level-3 queues. XSCHED_HPF_LV3_EXCLUSIVE=0 disables it.
    std::map<XDevice, PID> lv3_owner_;
    std::map<XDevice, TimePoint> lv3_owner_since_;
};

} // namespace xsched::sched
