#include <cstdlib>
#include <map>
#include <set>

#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/hpf.h"

using namespace xsched::sched;

// Resume delay (Phase 3.5): a lower-priority XQueue stays suspended until no higher-priority XQueue
// on its device has been ready for XSCHED_HPF_RESUME_DELAY_US (default 0 = resume at once). A
// decode-style high-priority client goes idle between steps for a few hundred microseconds; resuming
// the low-priority queue in every gap and suspending it again at the next step costs an eviction and
// a restore per step at Level-3 (and a ready-gate wait for the high-priority launch).
static std::chrono::microseconds ResumeDelay()
{
    static const std::chrono::microseconds d = [] {
        const char *v = std::getenv("XSCHED_HPF_RESUME_DELAY_US");
        return std::chrono::microseconds(v != nullptr ? std::atoll(v) : 0);
    }();
    return d;
}

static bool Lv3Exclusive()
{
    static const bool on = [] {
        const char *v = std::getenv("XSCHED_HPF_LV3_EXCLUSIVE");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}

static std::chrono::microseconds Lv3Quantum()
{
    static const std::chrono::microseconds q = [] {
        const char *v = std::getenv("XSCHED_HPF_LV3_QUANTUM_US");
        return std::chrono::microseconds(v != nullptr ? std::atoll(v) : 20000);
    }();
    return q;
}

void HighestPriorityFirstPolicy::Sched(const Status &status)
{
    const auto now = std::chrono::system_clock::now();
    const auto delay = ResumeDelay();
    // find the running highest priority task of each device
    std::map<XDevice, Priority> running_prio_max;
    for (auto &status : status.xqueue_status) {
        if (!status.second->ready) continue;
        XQueueHandle handle = status.second->handle;
        Priority priority = GetPriority(handle);
        if (delay.count() > 0) last_ready_[status.second->device][priority] = now;

        auto prio_it = running_prio_max.find(status.second->device);
        if (prio_it == running_prio_max.end()) {
            running_prio_max[status.second->device] = priority;
        } else if (priority > prio_it->second) {
            prio_it->second = priority;
        }
    }

    // DEF-41: choose the one process per device whose Level-3 queues at the top priority may run.
    std::map<XDevice, PID> lv3_run;
    if (Lv3Exclusive()) {
        std::map<XDevice, std::set<PID>> ready_pids, any_pids;
        for (auto &st : status.xqueue_status) {
            const XQueueStatus &q = *st.second;
            if (q.level < kPreemptLevelInterrupt) continue;
            auto pm = running_prio_max.find(q.device);
            const Priority top = pm == running_prio_max.end() ? PRIORITY_MIN : pm->second;
            if (GetPriority(q.handle) < top) continue;
            any_pids[q.device].insert(q.pid);
            if (q.ready) ready_pids[q.device].insert(q.pid);
        }
        for (auto &[dev, pids] : any_pids) {
            if (pids.size() < 2) continue;              // a single Level-3 process: nothing to arbitrate
            const auto &ready = ready_pids[dev];
            auto own = lv3_owner_.find(dev);
            const bool have_owner = own != lv3_owner_.end() && pids.count(own->second);
            const bool owner_ready = have_owner && ready.count(own->second);
            const bool expired = have_owner && now >= lv3_owner_since_[dev] + Lv3Quantum();
            PID next = have_owner ? own->second : *pids.begin();
            if (!ready.empty() && (!owner_ready || expired)) {
                // round robin: the first ready process after the current owner
                auto it = ready.upper_bound(have_owner ? own->second : 0);
                next = it != ready.end() ? *it : *ready.begin();
            }
            if (!have_owner || next != own->second) {
                lv3_owner_[dev] = next;
                lv3_owner_since_[dev] = now;
            }
            lv3_run[dev] = lv3_owner_[dev];
            if (ready.size() > 1 || (ready.size() == 1 && !ready.count(lv3_owner_[dev]))) {
                this->AddTimer(lv3_owner_since_[dev] + Lv3Quantum());
            }
        }
    }

    // suspend all xqueues with lower priority
    // and resume all xqueues with higher priority
    for (auto &status : status.xqueue_status) {
        XQueueHandle handle = status.second->handle;
        Priority priority = GetPriority(handle);

        // get the running highest priority task of the device
        Priority prio_max = PRIORITY_MIN;
        auto prio_it = running_prio_max.find(status.second->device);
        if (prio_it != running_prio_max.end()) prio_max = prio_it->second;

        if (priority < prio_max) {
            this->Suspend(handle);
            continue;
        }
        if (status.second->level >= kPreemptLevelInterrupt) {
            auto run = lv3_run.find(status.second->device);
            if (run != lv3_run.end() && run->second != status.second->pid) {
                this->Suspend(handle);              // DEF-41: another process's turn
                continue;
            }
        }
        if (delay.count() > 0) {
            // A higher priority level that was ready recently keeps this queue suspended for now.
            TimePoint latest{};
            auto dev_it = last_ready_.find(status.second->device);
            if (dev_it != last_ready_.end()) {
                for (auto it = dev_it->second.upper_bound(priority); it != dev_it->second.end(); ++it) {
                    if (it->second > latest) latest = it->second;
                }
            }
            if (latest.time_since_epoch().count() != 0 && now < latest + delay) {
                this->Suspend(handle);
                this->AddTimer(latest + delay);
                continue;
            }
        }
        this->Resume(handle);
    }
}

void HighestPriorityFirstPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    if (hint->Type() != kHintTypePriority) return;
    auto h = std::dynamic_pointer_cast<const PriorityHint>(hint);
    XASSERT(h != nullptr, "hint type not match");

    Priority priority = h->Prio();
    if (priority < PRIORITY_MIN) priority = PRIORITY_MIN;
    if (priority > PRIORITY_MAX) priority = PRIORITY_MAX;
    if (priority != h->Prio()) {
        XWARN("priority %d not in range [%d, %d], overide priority for XQueue 0x" FMT_64X " to %d",
              h->Prio(), PRIORITY_MIN, PRIORITY_MAX, h->Handle(), priority);
    }

    XINFO("set priority %d for XQueue 0x" FMT_64X, priority, h->Handle());
    priorities_[h->Handle()] = priority;
}

Priority HighestPriorityFirstPolicy::GetPriority(XQueueHandle handle)
{
    auto it = priorities_.find(handle);
    if (it != priorities_.end()) return it->second;
    // if priority not found, use default priority
    return PRIORITY_DEFAULT;
}
