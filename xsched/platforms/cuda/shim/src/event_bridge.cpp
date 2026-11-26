// Non-blocking cross-stream event waits (Phase 3.5, DEF-29, found with vLLM).
//
// cuStreamWaitEvent(S, E) where E was recorded on an XQueue and S is an ordinary (non-XQueue)
// stream used to block the calling thread until E had completed (xevent->Wait()), because the GPU
// cannot be trusted with the dependency: at Level-2/3 a preempted kernel ends early (trapped CTAs
// exit, later CTAs are killed at entry by the Entry Guardian) and an event recorded after it would
// signal before the work is really done; the XQueue relaunches the remaining work on resume. The
// host-side block serialized the application's CPU work with the GPU: vLLM's per-step
// `copy_stream.wait_stream(main_stream)` cost 1.8 ms per decode step (-22% tokens/s at Level-1).
//
// - Level-1 recording queue: launched work is never discarded, so once the event-record command is
//   in flight (launched on the hardware stream) the native GPU-side wait is exact: wait until
//   launched, then cuStreamWaitEvent.
// - Level-2/3 recording queue: the waiting stream S is promoted to an internal Level-1 XQueue and the
//   wait is submitted to it as a CudaEventWaitCommand. Its launch worker (not the caller) waits until
//   XSched reports the record command complete (HwCommand::Wait accounts for preemption and
//   re-execution); later work on S queues behind it in the new XQueue. The GPU never waits on host
//   progress: a first design that made S wait on the GPU for a host-written flag (cuStreamWaitValue32)
//   deadlocked under vLLM, because cuModuleLoadData holds a driver lock while it waits for the GPU to
//   go idle, the GPU waited for the flag, and the flag needed the XQueue worker's next cuLaunchKernel,
//   which waited for that lock.
//   The promoted queue inherits the recording queue's priority hint (a global HPF scheduler must not
//   suspend a high-priority process's helper queue as "low priority"), and is drained and destroyed
//   with its stream (StreamPromotionOnDestroy).
//
// XSCHED_CUDA_ASYNC_EVENT_WAIT = 0 restores the blocking behaviour; 1 (default) enables both paths;
// 2 is a test-only UNSAFE mode (plain GPU-side wait at every level) that demonstrates the hazard.

#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "xsched/hint.h"
#include "xsched/xqueue.h"
#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/cuda/hal.h"
#include "xsched/cuda/shim/shim.h"
#include "xsched/cuda/hal/common/cuda_command.h"

namespace xsched::cuda
{

namespace
{

int Mode()
{
    static const int m = [] {
        const char *v = std::getenv("XSCHED_CUDA_ASYNC_EVENT_WAIT");
        return (v == nullptr || v[0] == '\0') ? 1 : std::atoi(v);
    }();
    return m;
}

struct Promoted
{
    HwQueueHandle hwq = 0;
    XQueueHandle xq = 0;
};

std::mutex g_mtx;
std::unordered_map<CUstream, Promoted> g_promoted;
std::atomic<uint64_t> g_lv1_waits{0}, g_promoted_waits{0}, g_blocking_waits{0}, g_promotions{0};

std::shared_ptr<preempt::XQueue> Promote(CUstream stream, XQueueHandle recording)
{
    std::lock_guard<std::mutex> lk(g_mtx);
    // Another thread may have promoted the stream meanwhile.
    auto existing = preempt::HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (existing != nullptr) return existing;
    HwQueueHandle hwq = 0;
    XQueueHandle xq = 0;
    if (CudaQueueCreate(&hwq, stream) != kXSchedSuccess) return nullptr;
    if (XQueueCreate(&xq, hwq, kPreemptLevelBlock, kQueueCreateFlagNone) != kXSchedSuccess) {
        HwQueueDestroy(hwq);
        return nullptr;
    }
    Priority prio = 0;
    if (XHintGetPriority(recording, &prio) == kXSchedSuccess) XHintPriority(xq, prio);
    g_promoted[stream] = Promoted{hwq, xq};
    g_promotions.fetch_add(1, std::memory_order_relaxed);
    static bool logged = false;
    if (!logged) {
        logged = true;
        XINFO("stream %p waits on an event of a level-2/3 XQueue: promoted to a level-1 XQueue so the "
              "wait does not block the caller (XSCHED_CUDA_ASYNC_EVENT_WAIT=0 disables)", (void *)stream);
    }
    return preempt::XQueueManager::Get(xq);
}

} // namespace

bool AsyncCrossStreamWait(CUstream stream, CUevent event, unsigned int flags,
                          const std::shared_ptr<CudaEventRecordCommand> &xevent)
{
    const int mode = Mode();
    if (mode <= 0) {
        g_blocking_waits.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const XQueueHandle xqh = xevent->GetXQueueHandle();
    auto xq = preempt::XQueueManager::Get(xqh);
    if (xq == nullptr) {
        // The recording XQueue is gone: its commands were completed or dropped.
        xevent->Wait();
        return Driver::StreamWaitEvent(stream, event, flags) == CUDA_SUCCESS;
    }
    if (xq->GetPreemptLevel() <= kPreemptLevelBlock || mode == 2) {
        xevent->WaitUntil(preempt::kCommandStateInFlight);
        g_lv1_waits.fetch_add(1, std::memory_order_relaxed);
        return Driver::StreamWaitEvent(stream, event, flags) == CUDA_SUCCESS;
    }
    auto wq = Promote(stream, xqh);
    if (wq == nullptr) {
        g_blocking_waits.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    wq->Submit(std::make_shared<CudaEventWaitCommand>(xevent, flags));
    g_promoted_waits.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void StreamPromotionOnDestroy(CUstream stream)
{
    Promoted p;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_promoted.find(stream);
        if (it == g_promoted.end()) return;
        p = it->second;
        g_promoted.erase(it);
    }
    // cuStreamDestroy lets queued work finish; destroying an XQueue drops its pending commands, so
    // drain first.
    if (auto xq = preempt::XQueueManager::Get(p.xq)) xq->SubmitWaitAll()->Wait();
    XQueueDestroy(p.xq);
    HwQueueDestroy(p.hwq);
}

} // namespace xsched::cuda

/// Counters for tests: [Level-1 launched-then-GPU waits, waits submitted to promoted streams,
/// blocking waits, streams promoted].
EXPORT_C_FUNC int XCudaEventWaitStats(uint64_t *out, int n, int reset)
{
    using namespace xsched::cuda;
    std::atomic<uint64_t> *c[4] = {&g_lv1_waits, &g_promoted_waits, &g_blocking_waits, &g_promotions};
    int k = n < 4 ? n : 4;
    for (int i = 0; i < k; ++i) out[i] = reset ? c[i]->exchange(0) : c[i]->load();
    return k;
}
