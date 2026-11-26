#pragma once

#include <map>
#include <list>
#include <deque>
#include <mutex>
#include <thread>
#include <chrono>
#include <memory>
#include <condition_variable>

#include "xsched/utils/common.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/status.h"
#include "xsched/sched/scheduler/scheduler.h"

namespace xsched::sched
{

class LocalScheduler : public Scheduler
{
public:
    LocalScheduler(XPolicyType type);
    virtual ~LocalScheduler();

    virtual void Run() override;
    virtual void Stop() override;
    virtual void RecvEvent(std::shared_ptr<const Event> event) override;

    void SetPolicy(XPolicyType type);
    XPolicyType GetPolicy() const { return policy_type_; }

    void WaitAllOperations();
    void WaitOperations(PID pid);
    /// @brief Wait until every operation issued to pid has completed, at most timeout.
    bool WaitOperationsFor(PID pid, std::chrono::milliseconds timeout);
    /// @brief DEF-27 (cross-process server only): each round first sends the operations that
    /// suspend queues, waits (bounded) until those processes report them complete, and only then
    /// sends the other processes their operations. Together with the client ready gate, a
    /// higher-priority queue does not launch while a lower-priority one is still being preempted.
    void SetSuspendFirst(bool on, std::chrono::milliseconds timeout)
    { suspend_first_ = on; suspend_wait_ = timeout; }
    OperationId NextOperationId(PID pid);

private:
    void Worker();
    void ExecuteOperations();
    void CreateXQueueStatus(PID pid, const std::string &cmdline, XQueueHandle handle, XDevice dev,
                            XPreemptLevel level, int64_t threshold, int64_t batch_size,
                            bool ready, std::chrono::system_clock::time_point ready_time);
    void UpdateStatus(std::shared_ptr<const Event> event);
    void Suspend(XQueueHandle handle);
    void Resume(XQueueHandle handle);
    void AddTimer(const std::chrono::system_clock::time_point time_point);

    Status status_;
    std::deque<std::chrono::system_clock::time_point> timers_;

    XPolicyType policy_type_;
    std::unique_ptr<Policy> policy_ = nullptr;
    std::unique_ptr<std::thread> thread_ = nullptr;

    std::mutex event_mtx_;
    std::condition_variable event_cv_;
    std::unique_ptr<std::list<std::shared_ptr<const Event>>> event_queue_ = nullptr;

    struct OperationInfo
    {
        OperationId issued_id = 0;
        OperationId completed_id = 0;
    };

    std::mutex op_mtx_;
    std::condition_variable op_cv_;
    std::map<PID, OperationInfo> ops_;

    bool suspend_first_ = false;
    std::chrono::milliseconds suspend_wait_{100};
};

} // namespace xsched::sched
