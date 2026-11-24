#include "xsched/cuda/hal/level3/sm120_rm_handles.h"

#include <cstdarg>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <sys/ioctl.h>

namespace xsched::cuda
{

namespace
{

struct NVOS00
{
    uint32_t hRoot;
    uint32_t hObjectParent;
    uint32_t hObjectOld;
    int32_t  status;
};

struct NVOS21
{
    uint32_t hRoot;
    uint32_t hParent;
    uint32_t hObject;
    uint32_t hClass;
    uint64_t pAllocParms;
    uint64_t pRightsRequested;
    uint32_t paramsSize;
    uint32_t flags;
    int32_t  status;
};
static_assert(sizeof(NVOS21) == 48, "RM NVOS21 ABI mismatch");

constexpr unsigned long kNvEscRmFree  = 0xc0104629UL;
constexpr unsigned long kNvEscRmAlloc = 0xc030462bUL;

constexpr uint32_t kClassRoot     = 0x0041;
constexpr uint32_t kClassDevice   = 0x0080;
constexpr uint32_t kClassSubdev   = 0x2080;
constexpr uint32_t kClassTsg      = 0xa06c;
constexpr uint32_t kClassChannel  = 0xc96f;
constexpr uint32_t kClassCompute  = 0xcec0;
constexpr uint32_t kClassSmdbgssn = 0x83de;

constexpr int kMaxCapturedCtx = 16;

struct CapturedState
{
    int               fd = -1;
    uint32_t          client = 0;
    uint32_t          device = 0;
    uint32_t          subdev = 0;
    uint32_t          tsg = 0;
    uint32_t          channel = 0;
    uint32_t          session = 0;
    uint32_t          dbg_client = 0;
    uint32_t          pending_tsg = 0;
    uint32_t          pending_channel = 0;
    Sm120RmCtxHandles ctx_table[kMaxCapturedCtx] = {};
    int               ctx_count = 0;
    bool              pin_debugger_handle = true;
    std::mutex        mu;
};

CapturedState g_rm;
using IoctlFn = int (*)(int, unsigned long, ...);
IoctlFn g_real_ioctl = nullptr;
thread_local int t_in_hook = 0;

void RecordToCtxTable(uint32_t hClass, uint32_t hObj, uint32_t hRoot, int fd)
{
    if (hClass == kClassTsg) {
        g_rm.pending_tsg = hObj;
        g_rm.pending_channel = 0;
    } else if (hClass == kClassChannel) {
        if (g_rm.pending_channel == 0) g_rm.pending_channel = hObj;
    } else if (hClass == kClassCompute) {
        if (g_rm.pending_tsg != 0 &&
            (g_rm.ctx_count == 0 || g_rm.ctx_table[g_rm.ctx_count - 1].hTsg != g_rm.pending_tsg)) {
            int idx = (g_rm.ctx_count < kMaxCapturedCtx) ? g_rm.ctx_count++ : (kMaxCapturedCtx - 1);
            Sm120RmCtxHandles &slot = g_rm.ctx_table[idx];
            memset(&slot, 0, sizeof(slot));
            slot.ctl_fd     = (g_rm.fd >= 0) ? g_rm.fd : fd;
            slot.hClient    = g_rm.client ? g_rm.client : hRoot;
            slot.hDevice    = g_rm.device;
            slot.hSubDevice = g_rm.subdev;
            slot.hTsg       = g_rm.pending_tsg;
            slot.hChannel   = g_rm.pending_channel;
            slot.hCompute   = hObj;
        } else if (g_rm.ctx_count > 0 && g_rm.ctx_table[g_rm.ctx_count - 1].hCompute == 0) {
            g_rm.ctx_table[g_rm.ctx_count - 1].hCompute = hObj;
        }
    } else if (hClass == kClassSmdbgssn && g_rm.ctx_count > 0) {
        Sm120RmCtxHandles &cur = g_rm.ctx_table[g_rm.ctx_count - 1];
        cur.hDebugger       = hObj;
        cur.hDebuggerClient = hRoot;
    }
}

void LearnRmAlloc(int fd, void *arg)
{
    NVOS21 a{};
    memcpy(&a, arg, sizeof(a));
    if (a.status != 0 || a.hObject == 0) return;

    std::lock_guard<std::mutex> lock(g_rm.mu);
    switch (a.hClass) {
    case kClassRoot:
        if (g_rm.fd < 0) g_rm.fd = fd;
        if (!g_rm.client) g_rm.client = a.hObject;
        break;
    case kClassDevice:
        if (!g_rm.device) {
            g_rm.device = a.hObject;
            g_rm.client = a.hRoot;
            g_rm.fd     = fd;
        }
        break;
    case kClassSubdev:
        if (!g_rm.subdev) g_rm.subdev = a.hObject;
        break;
    case kClassTsg:
        if (!g_rm.tsg) g_rm.tsg = a.hObject;
        if (g_rm.fd < 0) g_rm.fd = fd;
        if (!g_rm.client) g_rm.client = a.hRoot;
        RecordToCtxTable(a.hClass, a.hObject, a.hRoot, fd);
        break;
    case kClassChannel:
        if (!g_rm.channel) g_rm.channel = a.hObject;
        RecordToCtxTable(a.hClass, a.hObject, a.hRoot, fd);
        break;
    case kClassCompute:
        if (g_rm.pending_tsg != 0) g_rm.tsg = g_rm.pending_tsg;
        if (g_rm.pending_channel != 0) g_rm.channel = g_rm.pending_channel;
        RecordToCtxTable(a.hClass, a.hObject, a.hRoot, fd);
        break;
    case kClassSmdbgssn:
        g_rm.session    = a.hObject;
        g_rm.dbg_client = a.hRoot;
        RecordToCtxTable(a.hClass, a.hObject, a.hRoot, fd);
        break;
    default:
        break;
    }
}

bool ShouldBlockFree(void *arg)
{
    std::lock_guard<std::mutex> lock(g_rm.mu);
    if (!g_rm.pin_debugger_handle || g_rm.session == 0) return false;
    auto *fr = reinterpret_cast<NVOS00 *>(arg);
    if (fr->hObjectOld == g_rm.session || fr->hObjectOld == g_rm.dbg_client) {
        fr->status = 0;
        return true;
    }
    return false;
}

} // namespace

bool Sm120RmHandleStore::GetLatest(Sm120RmCtxHandles *out)
{
    if (!out) return false;
    std::lock_guard<std::mutex> lock(g_rm.mu);
    if (g_rm.ctx_count > 0) {
        *out = g_rm.ctx_table[g_rm.ctx_count - 1];
        return out->ctl_fd >= 0 && out->hClient != 0 && out->hTsg != 0;
    }
    if (g_rm.fd < 0 || g_rm.client == 0 || g_rm.tsg == 0) return false;
    memset(out, 0, sizeof(*out));
    out->ctl_fd          = g_rm.fd;
    out->hClient         = g_rm.client;
    out->hDevice         = g_rm.device;
    out->hSubDevice      = g_rm.subdev;
    out->hTsg            = g_rm.tsg;
    out->hChannel        = g_rm.channel;
    out->hDebugger       = g_rm.session;
    out->hDebuggerClient = g_rm.dbg_client;
    return true;
}

int Sm120RmHandleStore::GetCount()
{
    std::lock_guard<std::mutex> lock(g_rm.mu);
    return g_rm.ctx_count;
}

bool Sm120RmHandleStore::GetByIndex(int idx, Sm120RmCtxHandles *out)
{
    if (!out) return false;
    std::lock_guard<std::mutex> lock(g_rm.mu);
    if (idx < 0 || idx >= g_rm.ctx_count) return false;
    *out = g_rm.ctx_table[idx];
    return out->ctl_fd >= 0 && out->hClient != 0 && out->hTsg != 0;
}

void Sm120RmHandleStore::SetPinDebuggerHandle(bool pin)
{
    std::lock_guard<std::mutex> lock(g_rm.mu);
    g_rm.pin_debugger_handle = pin;
}

} // namespace xsched::cuda

extern "C" int ioctl(int fd, unsigned long req, ...)
{
    va_list ap;
    va_start(ap, req);
    void *arg = va_arg(ap, void *);
    va_end(ap);

    if (!xsched::cuda::g_real_ioctl) {
        xsched::cuda::g_real_ioctl =
            reinterpret_cast<xsched::cuda::IoctlFn>(dlsym(RTLD_NEXT, "ioctl"));
    }
    if (!xsched::cuda::t_in_hook && arg &&
        (_IOC_NR(req) == 0x29 || req == xsched::cuda::kNvEscRmFree)) {
        if (xsched::cuda::ShouldBlockFree(arg)) return 0;
    }

    int r = xsched::cuda::g_real_ioctl(fd, req, arg);
    if (xsched::cuda::t_in_hook || r != 0 || !arg ||
        (_IOC_NR(req) != 0x2b && req != xsched::cuda::kNvEscRmAlloc)) {
        return r;
    }

    xsched::cuda::t_in_hook = 1;
    xsched::cuda::LearnRmAlloc(fd, arg);
    xsched::cuda::t_in_hook = 0;
    return r;
}
