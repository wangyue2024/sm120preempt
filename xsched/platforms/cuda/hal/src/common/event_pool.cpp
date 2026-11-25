#include "xsched/utils/log.h"
#include "xsched/cuda/hal/common/event_pool.h"

using namespace xsched::cuda;

void *ContextEventPool::Create()
{
    CUcontext cur_ctx = nullptr;
    CUDA_ASSERT(Driver::CtxGetCurrent(&cur_ctx));
    if (cur_ctx != ctx_) CUDA_ASSERT(Driver::CtxSetCurrent(ctx_));

    CUevent event = nullptr;
    CUDA_ASSERT(Driver::EventCreate(&event, CU_EVENT_DISABLE_TIMING));
    if (cur_ctx != ctx_) CUDA_ASSERT(Driver::CtxSetCurrent(cur_ctx));
    return event;
}

void ContextEventPool::DrainAll()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (void *obj : pool_) {
        if (obj == nullptr) continue;
        /// Best-effort: the event may not belong to the thread's current
        /// context. Any event that fails to be destroyed here will be
        /// reclaimed by the driver when the context is destroyed.
        Driver::EventDestroy((CUevent)obj);
    }
    pool_.clear();
}

std::mutex CudaEventPool::mutex_;
std::map<CUcontext, std::shared_ptr<ContextEventPool>> CudaEventPool::pools_;

CUevent CudaEventPool::Pop(CUcontext ctx)
{
    mutex_.lock();
    auto it = pools_.find(ctx);
    if (it == pools_.end()) {
        auto pool = std::make_shared<ContextEventPool>(ctx);
        pools_[ctx] = pool;
        it = pools_.find(ctx);
    }
    CUevent event = (CUevent)it->second->Pop();
    mutex_.unlock();
    return event;
}

void CudaEventPool::Push(CUcontext ctx, CUevent event)
{
    mutex_.lock();
    auto it = pools_.find(ctx);
    if (it == pools_.end()) {
        /// The pool has been cleared (the context is being destroyed or
        /// already destroyed). Destroy the event to avoid leaking it; this
        /// is a no-op if the context is already gone.
        mutex_.unlock();
        XWARN("cuda event pool not found for context (%p), destroying event %p", ctx, event);
        Driver::EventDestroy(event);
        return;
    }
    it->second->Push(event);
    mutex_.unlock();
}

void CudaEventPool::Clear(CUcontext ctx)
{
    std::shared_ptr<ContextEventPool> pool = nullptr;
    mutex_.lock();
    auto it = pools_.find(ctx);
    if (it != pools_.end()) {
        pool = it->second;
        pools_.erase(it);
    }
    mutex_.unlock();

    if (pool == nullptr) return;
    // Destroy the cached events outside the lock; the context is still
    // alive at this point (Clear is called before the physical destroy).
    pool->DrainAll();
}
