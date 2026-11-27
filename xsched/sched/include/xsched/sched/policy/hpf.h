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
};

} // namespace xsched::sched
