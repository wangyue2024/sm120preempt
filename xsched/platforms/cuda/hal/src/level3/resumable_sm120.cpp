#include "xsched/cuda/hal/level3/resumable_sm120.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <sys/ioctl.h>

#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"

namespace xsched::cuda
{

namespace
{

using IoctlFn = int (*)(int, unsigned long, ...);

IoctlFn GetRealIoctl()
{
    static IoctlFn fn = nullptr;
    if (!fn) {
        fn = reinterpret_cast<IoctlFn>(dlsym(RTLD_NEXT, "ioctl"));
    }
    return fn;
}

struct NvOs54Parameters {
    uint32_t hClient;
    uint32_t hObject;
    uint32_t cmd;
    uint32_t flags;
    uint64_t params;
    uint32_t paramsSize;
    uint32_t status;
};

constexpr unsigned long kNvEscRmControl = 0xc020462aUL;

// NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE (0x20801210)
// Authoritative 32-byte layout from open-gpu-kernel-modules ctrl2080gr.h
struct Nv2080CtrlGrSetCtxswPreemptionModeParams {
    uint32_t flags;             // 0x00: bit 0 = set gfxp, bit 1 = set cilp
    uint32_t hChannel;          // 0x04: TSG or channel handle
    uint32_t gfxpPreemptMode;   // 0x08: 0=WFI, 1=CTA, 2=GFXP
    uint32_t cilpPreemptMode;   // 0x0c: 0=WFI, 1=CTA, 2=CILP
    uint8_t  grRouteInfo[16];   // 0x10..0x1f: NV2080_CTRL_GR_ROUTE_info (zero)
};
static_assert(sizeof(Nv2080CtrlGrSetCtxswPreemptionModeParams) == 0x20,
              "NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE params must be 32 bytes");

// NVA06C_CTRL_CMD_GPFIFO_SCHEDULE (0xa06c0101)
struct NvA06cCtrlGpfifoScheduleParams {
    uint8_t bEnable;
    uint8_t bSkipSubmit;
    uint8_t pad[2];
};

// NVA06C_CTRL_CMD_SET_TIMESLICE (0xa06c0103)
struct NvA06cCtrlTimesliceParams {
    uint64_t timesliceUs;
};

// NVA06C_CTRL_CMD_PREEMPT (0xa06c0105)
struct NvA06cCtrlPreemptParams {
    uint8_t  bWait;
    uint8_t  bManualTimeout;
    uint8_t  pad[2];
    uint32_t timeoutUs;
};

// NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT (0x83de0317)
struct Nv83deSuspendParams {
    uint32_t waitForEvent;
    uint32_t status;
};

uint32_t RmControlCall(int ctl_fd, uint32_t hClient, uint32_t hObject,
                       uint32_t cmd, void *params, uint32_t paramsSize)
{
    if (ctl_fd < 0 || hClient == 0 || hObject == 0) {
        return 0xffffffffu;
    }
    NvOs54Parameters p{};
    p.hClient    = hClient;
    p.hObject    = hObject;
    p.cmd        = cmd;
    p.flags      = 0;
    p.params     = reinterpret_cast<uint64_t>(params);
    p.paramsSize = paramsSize;
    p.status     = 0xffffffffu;

    IoctlFn real_ioctl = GetRealIoctl();
    int ret = real_ioctl ? real_ioctl(ctl_fd, kNvEscRmControl, &p)
                         : ::ioctl(ctl_fd, kNvEscRmControl, &p);
    if (ret < 0) return 0xffffffffu;
    return p.status;
}

uint64_t NowMicros()
{
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

} // namespace

ResumableSm120Context::ResumableSm120Context(CUcontext ctx) : ctx_(ctx)
{
    Init(ctx, Sm120ResumableBackend::kAuto);
}

std::shared_ptr<ResumableSm120Context> ResumableSm120Context::Instance(CUcontext ctx)
{
    static std::mutex s_mtx;
    static std::map<CUcontext, std::shared_ptr<ResumableSm120Context>> s_map;
    std::lock_guard<std::mutex> lock(s_mtx);
    auto it = s_map.find(ctx);
    if (it != s_map.end()) return it->second;
    auto inst = std::make_shared<ResumableSm120Context>(ctx);
    s_map[ctx] = inst;
    return inst;
}

void ResumableSm120Context::BindContextSlot(int slot_index)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (ctx_slot_ != slot_index) {
        ctx_slot_ = slot_index;
        initialized_ = false;
        cilp_enabled_ = false;
    }
    EnsureHandles(slot_index);
}

bool ResumableSm120Context::EnsureHandles(int ctx_index)
{
    if (ctx_index >= 0) {
        ctx_slot_ = ctx_index;
    }
    if (!TrapSm120Context::GetRmHandlesByIndex(ctx_slot_, &handles_)) {
        if (!TrapSm120Context::GetLatestRmHandles(&handles_)) {
            return false;
        }
    }
    return handles_.ctl_fd >= 0 && handles_.hClient != 0 && handles_.hTsg != 0;
}

bool ResumableSm120Context::Init(CUcontext ctx, Sm120ResumableBackend backend)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (ctx != nullptr) ctx_ = ctx;
    EnsureHandles(-1);

    if (backend == Sm120ResumableBackend::kAuto) {
        const char *env_b = std::getenv("XSCHED_SM120_RESUMABLE_BACKEND");
        if (env_b && (std::strcmp(env_b, "DEBUGGER") == 0 ||
                      std::strcmp(env_b, "SUSPEND") == 0)) {
            active_backend_ = Sm120ResumableBackend::kDebuggerSuspend;
        } else {
            active_backend_ = Sm120ResumableBackend::kCilpTsgPreempt;
        }
    } else {
        active_backend_ = backend;
    }

    InitCilpMode();
    initialized_ = true;
    return true;
}

bool ResumableSm120Context::InitCilpMode()
{
    if (!EnsureHandles(-1) || handles_.hSubDevice == 0) {
        return false;
    }

    uint32_t target_h = handles_.hTsg ? handles_.hTsg : handles_.hChannel;
    if (target_h == 0) return false;

    // Step 1: Try COMPUTE_CILP (2), fallback to COMPUTE_CTA (1) via 0x20801210 (32-byte params)
    // Per ctrl2080gr.h: bit 0 (0x1u) is FLAGS_CILP_SET; bit 1 (0x2u) is FLAGS_GFXP_SET.
    Nv2080CtrlGrSetCtxswPreemptionModeParams mode{};
    mode.flags           = 0x1u; // bit 0: NV2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_FLAGS_CILP_SET
    mode.hChannel        = target_h;
    mode.gfxpPreemptMode = 0;
    mode.cilpPreemptMode = 2u;   // NV2080_CTRL_SET_CTXSW_PREEMPTION_MODE_COMPUTE_CILP

    uint32_t st = RmControlCall(handles_.ctl_fd, handles_.hClient,
                                handles_.hSubDevice, 0x20801210u,
                                &mode, sizeof(mode));
    if (st != 0) {
        mode.flags           = 0x1u;
        mode.cilpPreemptMode = 1u; // NV2080_CTRL_SET_CTXSW_PREEMPTION_MODE_COMPUTE_CTA
        st = RmControlCall(handles_.ctl_fd, handles_.hClient,
                           handles_.hSubDevice, 0x20801210u,
                           &mode, sizeof(mode));
    }
    cilp_enabled_ = (st == 0);
    return cilp_enabled_;
}

bool ResumableSm120Context::PreemptPulse()
{
    if (!EnsureHandles(-1)) {
        return false;
    }
    if (!cilp_enabled_) {
        InitCilpMode();
    }

    // Reduce TSG timeslice to minimum before preemption so runlist yields immediately
    NvA06cCtrlTimesliceParams ts{};
    ts.timesliceUs = 1;
    RmControlCall(handles_.ctl_fd, handles_.hClient,
                  handles_.hTsg, 0xa06c0103u, &ts, sizeof(ts));

    // Issue synchronous hardware TSG context-switch preemption (0xa06c0105)
    NvA06cCtrlPreemptParams pr{};
    pr.bWait          = 1;
    pr.bManualTimeout = 1;
    pr.timeoutUs      = 100000;
    uint32_t st = RmControlCall(handles_.ctl_fd, handles_.hClient,
                                handles_.hTsg, 0xa06c0105u, &pr, sizeof(pr));
    return (st == 0);
}

bool ResumableSm120Context::SuspendOnly()
{
    if (!EnsureHandles(-1)) {
        return false;
    }
    if (handles_.hDebugger != 0) {
        uint32_t hc = handles_.hDebuggerClient ? handles_.hDebuggerClient : handles_.hClient;
        Nv83deSuspendParams sp{1u, 0u};
        uint32_t st = RmControlCall(handles_.ctl_fd, hc,
                                    handles_.hDebugger, 0x83de0317u, &sp, sizeof(sp));
        if (st == 0) {
            dbg_suspended_ = true;
            return true;
        }
    }
    return PreemptPulse();
}

bool ResumableSm120Context::ResumeOnly()
{
    if (!EnsureHandles(-1)) {
        return false;
    }
    bool ok = true;
    if (dbg_suspended_ && handles_.hDebugger != 0) {
        uint32_t hc = handles_.hDebuggerClient ? handles_.hDebuggerClient : handles_.hClient;
        uint32_t dummy = 0;
        uint32_t st = RmControlCall(handles_.ctl_fd, hc,
                                    handles_.hDebugger, 0x83de0318u, &dummy, sizeof(dummy));
        dbg_suspended_ = false;
        ok = (st == 0);
    }

    // Restore normal timeslice (2000 us)
    NvA06cCtrlTimesliceParams ts{};
    ts.timesliceUs = 2000;
    uint32_t ts_st = RmControlCall(handles_.ctl_fd, handles_.hClient,
                                   handles_.hTsg, 0xa06c0103u, &ts, sizeof(ts));
    return ok && (ts_st == 0);
}

bool ResumableSm120Context::PauseViaCilpTsg()
{
    return PreemptPulse();
}

bool ResumableSm120Context::ResumeViaCilpTsg()
{
    return ResumeOnly();
}

bool ResumableSm120Context::PauseViaDebugger()
{
    return SuspendOnly();
}

bool ResumableSm120Context::ResumeViaDebugger()
{
    return ResumeOnly();
}

bool ResumableSm120Context::Pause(uint64_t *latency_us_out)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!initialized_) {
        EnsureHandles(-1);
        InitCilpMode();
        initialized_ = true;
    }
    uint64_t t0 = NowMicros();
    bool ok = false;
    if (active_backend_ == Sm120ResumableBackend::kDebuggerSuspend) {
        ok = PauseViaDebugger();
    } else {
        ok = PauseViaCilpTsg();
    }
    uint64_t dt = NowMicros() - t0;
    if (latency_us_out) *latency_us_out = dt;
    if (ok) {
        paused_.store(true, std::memory_order_release);
    }
    return ok;
}

bool ResumableSm120Context::Resume(uint64_t *latency_us_out)
{
    std::lock_guard<std::mutex> lock(mtx_);
    uint64_t t0 = NowMicros();
    bool ok = false;
    if (active_backend_ == Sm120ResumableBackend::kDebuggerSuspend) {
        ok = ResumeViaDebugger();
    } else {
        ok = ResumeViaCilpTsg();
    }
    uint64_t dt = NowMicros() - t0;
    if (latency_us_out) *latency_us_out = dt;
    if (ok) {
        paused_.store(false, std::memory_order_release);
    }
    return ok;
}

const char *ResumableSm120Context::ModeName() const
{
    if (active_backend_ == Sm120ResumableBackend::kDebuggerSuspend || dbg_suspended_) {
        return "suspend_resume_0x83de0317_0x83de0318";
    }
    if (cilp_enabled_) {
        return "cilp_cta_preempt_0x20801210_0xa06c0105";
    }
    return "tsg_preempt_0xa06c0105";
}

} // namespace xsched::cuda
