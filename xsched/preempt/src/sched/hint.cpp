#include <memory>
#include <mutex>
#include <unordered_map>

#include "xsched/hint.h"
#include "xsched/utils/common.h"
#include "xsched/sched/protocol/hint.h"
#include "xsched/preempt/sched/agent.h"
#include "xsched/preempt/xqueue/xqueue.h"

using namespace xsched::sched;
using namespace xsched::preempt;

EXPORT_C_FUNC XResult XHintSetScheduler(XSchedulerType scheduler, XPolicyType policy)
{
    return SchedAgent::SetScheduler(scheduler, policy);
}

// Last priority hint per XQueue, kept client-side so that internal helper queues (e.g. a stream
// promoted to an XQueue by a cross-stream event wait) can inherit the priority of the queue they
// depend on (see platforms/cuda/shim/src/event_bridge.cpp).
static std::mutex g_prio_mtx;
static std::unordered_map<XQueueHandle, Priority> g_prio;

EXPORT_C_FUNC XResult XHintPriority(XQueueHandle xq, Priority prio)
{
    XResult res = XQueueManager::Exists(xq);
    if (res != kXSchedSuccess) {
        XWARN("XQueue with handle 0x" FMT_64X " does not exist", xq);
        return res;
    }
    {
        std::lock_guard<std::mutex> lk(g_prio_mtx);
        g_prio[xq] = prio;
    }
    SchedAgent::SendHint(std::make_shared<PriorityHint>(xq, prio));
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XHintGetPriority(XQueueHandle xq, Priority *prio)
{
    if (prio == nullptr) return kXSchedErrorInvalidValue;
    std::lock_guard<std::mutex> lk(g_prio_mtx);
    auto it = g_prio.find(xq);
    if (it == g_prio.end()) return kXSchedErrorNotFound;
    *prio = it->second;
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XHintUtilization(XQueueHandle xq, Utilization util)
{
    XResult res = XQueueManager::Exists(xq);
    if (res != kXSchedSuccess) {
        XWARN("XQueue with handle 0x" FMT_64X " does not exist", xq);
        return res;
    }
    SchedAgent::SendHint(std::make_shared<UtilizationHint>(GetProcessId(), xq, util));
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XHintTimeslice(Timeslice ts_us)
{
    SchedAgent::SendHint(std::make_shared<TimesliceHint>(ts_us));
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XHintLaxity(XQueueHandle xq, Laxity lax_us, Priority lax_prio, Priority crit_prio)
{
    XResult res = XQueueManager::Exists(xq);
    if (res != kXSchedSuccess) {
        XWARN("XQueue with handle 0x" FMT_64X " does not exist", xq);
        return res;
    }
    SchedAgent::SendHint(std::make_shared<LaxityHint>(xq, lax_us, lax_prio, crit_prio));
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XHintDeadline(XQueueHandle xq, Deadline ddl_us)
{
    XResult res = XQueueManager::Exists(xq);
    if (res != kXSchedSuccess) {
        XWARN("XQueue with handle 0x" FMT_64X " does not exist", xq);
        return res;
    }
    SchedAgent::SendHint(std::make_shared<DeadlineHint>(xq, ddl_us));
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XHintKDeadline(size_t k)
{
    SchedAgent::SendHint(std::make_shared<KDeadlineHint>(k));
    return kXSchedSuccess;
}
