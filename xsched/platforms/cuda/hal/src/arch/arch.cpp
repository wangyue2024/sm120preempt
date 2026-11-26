#include <strings.h>
#include "xsched/cuda/hal/level2/guardian.h"
#include "xsched/cuda/hal/level3/trap.h"
#include "xsched/cuda/hal/level3/cuda_queue.h"
#include "xsched/cuda/hal/common/levels.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/options.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/arch/sm35.h"
#include "xsched/cuda/hal/arch/sm70.h"
#include "xsched/cuda/hal/arch/sm86.h"
#include "xsched/cuda/hal/arch/sm120.h"
#include "xsched/cuda/hal/level3/gdb_interrupt.h"
#include "xsched/cuda/hal/level3/sm120_trap_engine.h"
// NEW_CUDA_ARCH: New CUDA architecture support goes here

using namespace xsched::cuda;
using namespace xsched::preempt;

// XSCHED_SM120_L3_BACKEND=GDB: allow a scheduler-side cuda-gdb agent to attach
// to this process, which is what the level-3 interrupt backend drives.  Inert
// unless the env var is set.
static void MaybeAllowGdbBackend()
{
    static bool done = false;
    if (done) return;
    done = true;
    const char *backend = getenv("XSCHED_SM120_L3_BACKEND");
    if (backend == nullptr) backend = getenv("XSCHED_SM120_L3_MODE");
    if (backend == nullptr || strcasecmp(backend, "GDB") != 0) return;
    if (!GdbInterruptContext::IsAvailable()) {
        XWARN("XSCHED_SM120_L3_BACKEND=GDB but cuda-gdb was not found in PATH");
        return;
    }
    if (GdbInterruptContext::AllowDebuggerAttach()) {
        XINFO("level-3 GDB backend enabled: this process accepts a debugger attach");
    } else {
        XWARN("level-3 GDB backend: prctl(PR_SET_PTRACER) failed");
    }
}

static int32_t GetArch(CUdevice dev)
{
    // see https://developer.nvidia.com/cuda-legacy-gpus
    // see https://developer.nvidia.com/cuda-gpus
    int major, minor;
    CUDA_ASSERT(Driver::DeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CUDA_ASSERT(Driver::DeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    return major * 10 + minor;
}

std::shared_ptr<Guardian> Guardian::Instance(CUdevice dev)
{
    switch (GetArch(dev)) {
    case 35:
        return std::make_shared<GuardianSM35>();
    case 70:
        return std::make_shared<GuardianSM70>();
    case 86:
        return std::make_shared<GuardianSM86>();
    case 120: // Blackwell: RTX 50 series
        return std::make_shared<GuardianSM120>();
    // NEW_CUDA_ARCH: New CUDA architecture support goes here
    default:
        return nullptr;
    }
}

std::shared_ptr<TarpHandler> TarpHandler::Instance(CUdevice dev)
{
    switch (GetArch(dev)) {
    case 70:
        return std::make_shared<TarpHandlerSM70>();
    case 86:
        return std::make_shared<TarpHandlerSM86>();
    case 120: // Blackwell: RTX 50 series (Linux; Windows falls back to Lv2)
        return std::make_shared<TarpHandlerSM120>();
    // NEW_CUDA_ARCH: New CUDA architecture support goes here
    default:
        return nullptr;
    }
}

std::shared_ptr<HwQueue> xsched::cuda::CudaQueueCreate(CUstream stream)
{
    CUdevice dev;
    CUcontext stream_ctx;
    CUcontext current_ctx;
    CUDA_ASSERT(Driver::StreamGetCtx(stream, &stream_ctx));
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_ctx));
    XASSERT(current_ctx == stream_ctx,
            "create CudaQueue failed: current context (%p) does not match stream context (%p)",
            current_ctx, stream_ctx);
    CUDA_ASSERT(Driver::CtxGetDevice(&dev));

    const int32_t arch = GetArch(dev);

#if defined(_WIN32)
    // Windows: only the Blackwell (sm120) level-2 path is hardware-verified
    // on the standard driver (test/sm120_mve, T6).  The trap (L3) path is
    // unavailable there (cuXtraGetTrapHandlerInfo aborts, T1-P6) and sm120
    // has no TarpHandlerSM120, so sm120 must NOT fall into CudaQueueLv3Trap.
    // Other architectures keep the legacy level-1 fallback.
    if (arch == 120) return std::make_shared<CudaQueueLv2>(stream);
    return std::make_shared<CudaQueueLv1>(stream);
#endif

    if (GetCudaLv3Implementation() == kCudaLv3ImplementationTsg) {
        return std::make_shared<CudaQueueLv3Tsg>(stream);
    }

    switch (arch) {
    case 35: // Kepler: K20, K40, GTX TITAN
        return std::make_shared<CudaQueueLv2>(stream);
    case 70: // Volta: V100, GV100
    case 86: // Ampere: RTX3050 - RTX 3090 Ti
        return std::make_shared<CudaQueueLv3Trap>(stream);
    case 120: { // Blackwell: RTX 50 series.
        const char *mode = getenv("XSCHED_SM120_L3_MODE");
        if (mode == nullptr || mode[0] == '\0') {
            mode = getenv("XSCHED_SM120_L3_BACKEND");
        }
        if (mode != nullptr) {
            if (strcasecmp(mode, "resumable") == 0) {
                XINFO("level-3 sm120 resumable backend enabled");
                return std::make_shared<CudaQueueLv3Resumable>(stream);
            }
            if (strcasecmp(mode, "trap") == 0) {
                XINFO("level-3 sm120 trap backend enabled");
                return std::make_shared<CudaQueueLv3Trap>(stream);
            }
            if (strcasecmp(mode, "tsg") == 0) {
                XINFO("level-3 sm120 tsg backend enabled");
                return std::make_shared<CudaQueueLv3Tsg>(stream);
            }
            if (strcasecmp(mode, "gdb") == 0) {
                MaybeAllowGdbBackend();
                return std::make_shared<CudaQueueLv3Tsg>(stream);
            }
        }
        XINFO("level-3 sm120 trap backend enabled (default)");
        return std::make_shared<CudaQueueLv3Trap>(stream);
    }
    // NEW_CUDA_ARCH: New CUDA architecture support goes here
    default:
        return std::make_shared<CudaQueueLv1>(stream);
    }
}

CUresult xsched::cuda::DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream)
{
    CUcontext current_ctx;
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_ctx));
    // Per-thread cache of the (stream, context) check and the device architecture: pass-through
    // launches of unmanaged (e.g. high-priority) streams are on the latency-critical path, and
    // these lookups cost several driver calls per launch (Phase 3.3).
    thread_local CUstream cached_stream = nullptr;
    thread_local CUcontext cached_ctx = nullptr;
    thread_local int32_t cached_arch = -1;
    int32_t arch;
    if (cached_arch >= 0 && stream == cached_stream && current_ctx == cached_ctx) {
        arch = cached_arch;
    } else {
        CUdevice dev;
        CUcontext stream_ctx;
        CUDA_ASSERT(Driver::StreamGetCtx(stream, &stream_ctx));
        XASSERT(current_ctx == stream_ctx,
                "direct launch kernel failed: current context (%p) does not match stream context (%p)",
                current_ctx, stream_ctx);
        CUDA_ASSERT(Driver::CtxGetDevice(&dev));
        arch = GetArch(dev);
        cached_stream = stream;
        cached_ctx = current_ctx;
        cached_arch = arch;
    }

#if defined(_WIN32)
    // Windows: sm120 uses the level-2 direct path so that an already
    // instrumented kernel stays safe when launched on an unmanaged stream
    // (the preempt-buffer field of the debugger window is populated before
    // the passthrough launch).  Other architectures keep the plain
    // level-1 launch.
    if (arch == 120) return CudaQueueLv2::DirectLaunch(kernel, current_ctx, stream);
    return CudaQueueLv1::DirectLaunch(kernel, stream);
#endif

    if (GetCudaLv3Implementation() == kCudaLv3ImplementationTsg) {
        return CudaQueueLv3Tsg::DirectLaunch(kernel, stream);
    }

    switch (arch) {
    case 35:
        return CudaQueueLv2::DirectLaunch(kernel, current_ctx, stream);
    case 70:
    case 86:
        return CudaQueueLv3Trap::DirectLaunch(kernel, current_ctx, stream);
    case 120: {
        // Under the LaunchMutex an LP Mode 5 launch cannot set the params between this check and
        // the launch. Resolve the CUfunction only if this launch handle may have params set.
        std::lock_guard<std::mutex> lock(sm120::Sm120TrapEngine::LaunchMutex(current_ctx));
        if (sm120::Sm120StreamContext::MaybeParamsSet(kernel->kFunc)) {
            sm120::Sm120StreamContext::ClearDebuggerParams(kernel->FuncHandle());
        }
        return kernel->LaunchWrapper(stream);
    }
    // NEW_CUDA_ARCH: New CUDA architecture support goes here
    default:
        return CudaQueueLv1::DirectLaunch(kernel, stream);
    }
}
