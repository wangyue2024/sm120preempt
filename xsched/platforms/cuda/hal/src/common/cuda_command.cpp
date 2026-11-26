#include <cstring>
#include <cuxtra/cuxtra.h>

#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/options.h"
#include "xsched/cuda/hal/common/event_pool.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/common/cuda_command.h"

using namespace xsched::cuda;

CudaCommand::CudaCommand(preempt::XCommandProperties props)
    : HwCommand(GetCudaLv3Implementation() == kCudaLv3ImplementationTsg
                ? props | preempt::kCommandPropertyDeactivatable : props)
{
    CUDA_ASSERT(Driver::CtxGetCurrent(&ctx_));
    XASSERT(ctx_ != nullptr, "current context of the calling thread is nullptr");
}

CudaCommand::~CudaCommand()
{
    if (following_event_ == nullptr) return;
    CudaEventPool::Push(ctx_, following_event_);
}

void CudaCommand::Synchronize()
{
    XASSERT(following_event_ != nullptr,
            "following_event_ is nullptr, EnableSynchronization() should be called first");
    CUDA_ASSERT(Driver::EventSynchronize(following_event_));
}

bool CudaCommand::Synchronizable()
{
    return following_event_ != nullptr;
}

bool CudaCommand::EnableSynchronization()
{
    if (following_event_ != nullptr) return true;
    following_event_ = CudaEventPool::Pop(ctx_);
    return following_event_ != nullptr;
}

CUresult CudaCommand::LaunchWrapper(CUstream stream)
{
    CUresult ret = Launch(stream);
    if (UNLIKELY(ret != CUDA_SUCCESS)) return ret;
    if (following_event_ != nullptr) ret = Driver::EventRecord(following_event_, stream);
    return ret;
}

CudaKernelCommand::CudaKernelCommand(CUfunction func, void **params, void **extra, bool deep_copy)
    : CudaCommand(preempt::kCommandPropertyDeactivatable)
    // cuXtraKernelGetFunction() will convert func to CUfunction if func is a CUkernel.
    , kFunc(func), kFuncHandle(deep_copy ? cuXtraKernelGetFunction((CUkernel)func) : nullptr)
    , params_(params), extra_(extra), param_cnt_(deep_copy ? cuXtraGetParamCount(kFuncHandle) : 0)
{
    if (!deep_copy) return;

    /// @FIXME: even if param_cnt_ == 0, params or extra can be non-nullptr.
    if (param_cnt_ == 0) return;
    if (params == nullptr && extra == nullptr) {
        XWARN("params and extra of kernel (%p) are both nullptr", func);
        return;
    }
    if (params != nullptr && extra != nullptr) {
        XWARN("illegally using both params (%p) and extra (%p) of kernel %p", params, extra, func);
        return;
    }

    deep_copy_ = true;
    params_ = (void **)malloc(param_cnt_ * sizeof(void *));

    if (params != nullptr) {
        // Allocate a continuous buffer for all of the params
        // buffer size = last param offset + last param size
        size_t buffer_size = 0;
        for (size_t i = 0; i < param_cnt_; ++i) {
            size_t offset = 0, size = 0;
            cuXtraGetParamInfo(kFuncHandle, i, &offset, &size, nullptr);
            if (offset + size > buffer_size) buffer_size = offset + size;
        }
        if (buffer_size == 0) buffer_size = 1;
        param_data_ = (char *)malloc(buffer_size);

        for (size_t i = 0; i < param_cnt_; ++i) {
            size_t offset = 0, size = 0;
            cuXtraGetParamInfo(kFuncHandle, i, &offset, &size, nullptr);
            params_[i] = (void*)&param_data_[offset];
            memcpy(params_[i], params[i], size);
        }
    } else if (extra != nullptr) {
        // Get extra buffer and its size from extra map.
        void *extra_buffer = nullptr;
        cuXtraGetExtraBuffer(extra, &extra_buffer, &extra_buffer_size_);
        // We have checked param_cnt_ > 0.
        XASSERT(extra_buffer != nullptr && extra_buffer_size_ > 0,
                "invalid extra buffer (%p) and size (%zu)", extra_buffer, extra_buffer_size_);

        // Deep-copy extra buffer to extra_data_.
        extra_data_ = (char *)malloc(extra_buffer_size_);
        memcpy(extra_data_, extra_buffer, extra_buffer_size_);

        // Set extra map.
        extra_ = (void **)malloc(5 * sizeof(void *));
        extra_[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
        extra_[1] = extra_data_;
        extra_[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
        extra_[3] = (void *)&extra_buffer_size_;
        extra_[4] = CU_LAUNCH_PARAM_END;

        // Set params_ to point to extra_data_.
        for (size_t i = 0; i < param_cnt_; ++i) {
            size_t offset, size;
            cuXtraGetParamInfo(kFuncHandle, i, &offset, &size, nullptr);
            XASSERT(offset + size <= extra_buffer_size_,
                    "extra[%zu] out of range: offset (%zu) + size (%zu) > buffer size (%zu)",
                    i, offset, size, extra_buffer_size_);
            params_[i] = (void*)&extra_data_[offset];
        }
    }
}

CudaKernelCommand::~CudaKernelCommand()
{
    if (!deep_copy_) return;
    if (param_data_) free(param_data_);
    if (extra_data_) free(extra_data_);
    if (params_) free(params_);
    if (extra_) free(extra_);
}

CUfunction CudaKernelCommand::FuncHandle() const
{
    if (kFuncHandle != nullptr) return kFuncHandle;
    if (lazy_func_handle_ == nullptr) lazy_func_handle_ = cuXtraKernelGetFunction((CUkernel)kFunc);
    return lazy_func_handle_;
}

bool CudaKernelCommand::EnsureDynamicSmem(unsigned int shm) const
{
    int max_dyn = 0;
    if (Driver::FuncGetAttribute(&max_dyn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, FuncHandle()) != CUDA_SUCCESS) {
        return false;
    }
    if (static_cast<int>(shm) <= max_dyn) return false;
    return Driver::FuncSetAttribute(
               FuncHandle(), CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
               static_cast<int>(shm)) == CUDA_SUCCESS;
}

CudaKernelLaunchExCommand::CudaKernelLaunchExCommand(const CUlaunchConfig *cfg, CUfunction func,
                                                     void **params, void **extra, bool deep_copy)
    : CudaKernelCommand(func, params, extra, deep_copy)
{
    if (cfg == nullptr) {
        XWARN("CUlaunchConfig of %p is nullptr", func);
        return;
    }

    memcpy(&cfg_, cfg, sizeof(CUlaunchConfig));
    if (cfg->attrs != nullptr && cfg->numAttrs > 0 && deep_copy) {
        cfg_.attrs = (CUlaunchAttribute *)malloc(cfg->numAttrs * sizeof(CUlaunchAttribute));
        memcpy(cfg_.attrs, cfg->attrs, cfg->numAttrs * sizeof(CUlaunchAttribute));
        attrs_deep_copied_ = true;
    }
    if (IsCooperative()) {
        ClearProps(preempt::kCommandPropertyDeactivatable);
    }
}

CudaKernelLaunchExCommand::~CudaKernelLaunchExCommand()
{
    if (cfg_.attrs == nullptr || !attrs_deep_copied_) return;
    free(cfg_.attrs);
}

bool CudaKernelLaunchExCommand::IsCooperative() const
{
    if (cfg_.attrs == nullptr) return false;
    for (unsigned int i = 0; i < cfg_.numAttrs; ++i) {
        if (cfg_.attrs[i].id == CU_LAUNCH_ATTRIBUTE_COOPERATIVE && cfg_.attrs[i].value.cooperative != 0) {
            return true;
        }
    }
    return false;
}

unsigned int CudaKernelLaunchExCommand::ClusterSize() const
{
    if (cfg_.attrs == nullptr) return 1;
    for (unsigned int i = 0; i < cfg_.numAttrs; ++i) {
        if (cfg_.attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
            const auto &cd = cfg_.attrs[i].value.clusterDim;
            const unsigned int sz = cd.x * cd.y * cd.z;
            return sz > 0 ? sz : 1;
        }
    }
    return 1;
}

CUresult CudaKernelLaunchExCommand::Launch(CUstream stream)
{
    cfg_.hStream = stream;
    void **params = params_;
    if (deep_copy_ && extra_ != nullptr) params = nullptr;
    CUresult r = Driver::LaunchKernelEx(&cfg_, kFunc, params, extra_);
    if (UNLIKELY(r == CUDA_ERROR_INVALID_VALUE && cfg_.sharedMemBytes > 0 &&
                 EnsureDynamicSmem(cfg_.sharedMemBytes))) {
        r = Driver::LaunchKernelEx(&cfg_, kFunc, params, extra_);
    }
    return r;
}

CudaEventRecordCommand::CudaEventRecordCommand(CUevent event)
    : CudaCommand(preempt::kCommandPropertyIdempotent), event_(event)
{
    XASSERT(event_ != nullptr, "cuda event should not be nullptr");
}

CudaEventRecordCommand::~CudaEventRecordCommand()
{
    if (event_ == nullptr || (!destroy_event_)) return;
    CUDA_ASSERT(Driver::EventDestroy_v2(event_));
}

void CudaEventWaitCommand::BeforeLaunch()
{
    if (event_cmd_) event_cmd_->Wait(); // recorded on an XQueue, wait it on the XQueue
}

CUresult CudaEventWaitCommand::Launch(CUstream stream)
{
    if (!event_) return CUDA_SUCCESS; // already waited in BeforeLaunch()
    return Driver::StreamWaitEvent(stream, event_, flags_);
}

#if defined(_WIN32)
extern "C" {
CUfunction cuXtraKernelGetFunction(CUkernel kernel)
{
    return reinterpret_cast<CUfunction>(kernel);
}

void cuXtraGetExtraBuffer(void **extra, void **buffer, size_t *size)
{
    if (buffer) *buffer = nullptr;
    if (size) *size = 0;
    if (!extra) return;
    for (void **p = extra; *p != nullptr; ) {
        void *tag = *p++;
        if (tag == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            if (buffer) *buffer = *p;
            p++;
        } else if (tag == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            if (size) *size = *reinterpret_cast<size_t *>(*p);
            p++;
        } else {
            p++;
        }
    }
}
}
#endif
