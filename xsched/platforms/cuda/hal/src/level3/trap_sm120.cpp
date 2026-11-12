// trap_sm120.cpp - XSched level-3 trap backend for Blackwell sm120 (Linux).
//
// Mechanism (validated in test/sm120_l3_trap_v2, FINDINGS_ZH.md v2.0):
//   * the live driver .text.traphandler (arena base + 0x15f0f00) is patched at
//     +0x390 with CALL.ABS to a cave in the slot-20 object (+0xA00);
//   * the cave is a harmless return stub by default.  To preempt, it is
//     rewritten to "redirect the saved TRAP_RETURN_PC to an EXIT trampoline"
//     (cave+0x70).  The handler finishes restoring the trap state first, so the
//     warp exits cleanly and the context survives;
//   * an external agent holds a 0x83de session bound to {app client, KGO} and
//     sends 0x83de0317{0,0} when asked; the app grants it access with the
//     NV01_ROOT 0xd04 inherited-share policy (late grant works).
//
// The app and agent talk over a POSIX shm control block (trap_sm120_shm.h).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cerrno>
#include <cstdint>
#include <csignal>
#include <csetjmp>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <time.h>
#include <vector>
#include <string>
#include <link.h>
#include <thread>

#include <cuxtra/cuxtra.h>

#include "xsched/utils/log.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/level3/trap_sm120.h"
#include "xsched/cuda/hal/level3/trap_sm120_shm.h"

using namespace xsched::cuda;

// ------------------------------------------------------------------ RM ABI
struct NVOS21 { uint32_t hRoot, hParent, hObject, hClass;
                uint64_t pAllocParms, pRightsRequested;
                uint32_t paramsSize, flags; int32_t status; };
struct NVOS54 { uint32_t hClient, hObject, cmd, flags;
                uint64_t params; uint32_t paramsSize; int32_t status; };
static_assert(sizeof(NVOS21) == 48 && sizeof(NVOS54) == 32, "RM ABI");

struct NVOS00 { uint32_t hRoot, hObjectParent, hObjectOld; int32_t status; };

static const unsigned long NV_ESC_RM_FREE    = 0xc0104629UL;
static const unsigned long NV_ESC_RM_ALLOC   = 0xc030462bUL;
static const unsigned long NV_ESC_RM_CONTROL = 0xc020462aUL;
static const uint32_t CLASS_ROOT = 0x41, CLASS_DEVICE = 0x80, CLASS_SUBDEV = 0x2080,
                      CLASS_TSG = 0xa06c, CLASS_CHANNEL = 0xc96f,
                      CLASS_COMPUTE = 0xcec0, CLASS_SMDBGSSN = 0x83de;

namespace {
struct Handles {
    int fd = -1;
    uint32_t client = 0, device = 0, subdev = 0, tsg = 0, channel = 0, session = 0, dbg_client = 0;
    uint32_t ncompute = 0, computes[L3_TRAP_MAX_KGO] = {};
} H;

constexpr int kMaxCapturedCtx = 16;
Sm120RmCtxHandles g_ctx_table[kMaxCapturedCtx] = {};
int g_ctx_count = 0;
uint32_t g_pending_tsg = 0;
uint32_t g_pending_channel = 0;
bool g_pin_debugger_handle = true;

int (*real_ioctl)(int, unsigned long, ...) = nullptr;
__thread int in_hook = 0;

void RecordToCtxTable(uint32_t hClass, uint32_t hObj, uint32_t hRoot, int fd)
{
    if (hClass == CLASS_TSG) {
        g_pending_tsg = hObj;
        g_pending_channel = 0;
    } else if (hClass == CLASS_CHANNEL) {
        if (g_pending_channel == 0) g_pending_channel = hObj;
    } else if (hClass == CLASS_COMPUTE) {
        if (g_pending_tsg != 0 &&
            (g_ctx_count == 0 || g_ctx_table[g_ctx_count - 1].hTsg != g_pending_tsg)) {
            int idx = (g_ctx_count < kMaxCapturedCtx) ? g_ctx_count++ : (kMaxCapturedCtx - 1);
            g_ctx_table[idx].ctl_fd     = (H.fd >= 0) ? H.fd : fd;
            g_ctx_table[idx].hClient    = H.client ? H.client : hRoot;
            g_ctx_table[idx].hDevice    = H.device;
            g_ctx_table[idx].hSubDevice = H.subdev;
            g_ctx_table[idx].hTsg       = g_pending_tsg;
            g_ctx_table[idx].hChannel   = g_pending_channel;
            g_ctx_table[idx].hCompute   = hObj;
        } else if (g_ctx_count > 0 && g_ctx_table[g_ctx_count - 1].hCompute == 0) {
            g_ctx_table[g_ctx_count - 1].hCompute = hObj;
        }
    } else if (hClass == CLASS_SMDBGSSN && g_ctx_count > 0) {
        Sm120RmCtxHandles &cur = g_ctx_table[g_ctx_count - 1];
        cur.hDebugger       = hObj;
        cur.hDebuggerClient = hRoot;
    }
}

void LearnAlloc(int fd, void *arg)
{
    NVOS21 a; memcpy(&a, arg, sizeof a);
    if (a.status != 0 || a.hObject == 0) return;
    switch (a.hClass) {
    case CLASS_ROOT:
        if (H.fd < 0) H.fd = fd;
        if (!H.client) H.client = a.hObject;
        break;
    case CLASS_DEVICE:
        if (!H.device) { H.device = a.hObject; H.client = a.hRoot; H.fd = fd; }
        break;
    case CLASS_SUBDEV:
        if (!H.subdev) H.subdev = a.hObject;
        break;
    case CLASS_TSG:
        if (!H.tsg) H.tsg = a.hObject;
        if (H.fd < 0) H.fd = fd;
        if (!H.client) H.client = a.hRoot;
        RecordToCtxTable(a.hClass, a.hObject, a.hRoot, fd);
        break;
    case CLASS_CHANNEL:
        if (!H.channel) H.channel = a.hObject;
        RecordToCtxTable(a.hClass, a.hObject, a.hRoot, fd);
        break;
    case CLASS_COMPUTE:
        if (g_pending_tsg != 0) H.tsg = g_pending_tsg;
        if (g_pending_channel != 0) H.channel = g_pending_channel;
        if (H.ncompute < L3_TRAP_MAX_KGO) H.computes[H.ncompute++] = a.hObject;
        RecordToCtxTable(a.hClass, a.hObject, a.hRoot, fd);
        break;
    case CLASS_SMDBGSSN:
        H.session = a.hObject;
        H.dbg_client = a.hRoot;
        RecordToCtxTable(a.hClass, a.hObject, a.hRoot, fd);
        break;
    }
}
} // namespace

extern "C" int ioctl(int fd, unsigned long req, ...)
{
    va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
    if (!real_ioctl) real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
    if (!in_hook && arg && (_IOC_NR(req) == 0x29 || req == NV_ESC_RM_FREE) && g_pin_debugger_handle) {
        NVOS00 *fr = reinterpret_cast<NVOS00 *>(arg);
        if (H.session != 0 && (fr->hObjectOld == H.session || fr->hRoot == H.dbg_client)) {
            fr->status = 0;
            return 0;
        }
    }
    int r = real_ioctl(fd, req, arg);
    if (in_hook || r || !arg || (_IOC_NR(req) != 0x2b && req != NV_ESC_RM_ALLOC)) return r;
    in_hook = 1; LearnAlloc(fd, arg); in_hook = 0;
    return r;
}

// ------------------------------------------------------------ helpers
static int rm_control(int fd, uint32_t client, uint32_t obj, uint32_t cmd, void *p, uint32_t n)
{
    NVOS54 c; memset(&c, 0, sizeof c);
    c.hClient = client; c.hObject = obj; c.cmd = cmd;
    c.params = (uint64_t)(uintptr_t)p; c.paramsSize = n;
    if (!real_ioctl) real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
    int r = real_ioctl(fd, NV_ESC_RM_CONTROL, &c);
    return r ? (0x1000000 | r) : c.status;
}

static CUuuid UuidOf(const char *hex)
{
    CUuuid u;
    for (int i = 0; i < 16; i++) { unsigned v; sscanf(hex + i * 2, "%2x", &v); u.bytes[i] = (char)v; }
    return u;
}
static uint32_t OpcodeAt(const uint8_t *p) { uint64_t w; memcpy(&w, p, 8); return (uint32_t)w; }
static uint32_t Imm32At(const uint8_t *p) { uint64_t w; memcpy(&w, p, 8); return (uint32_t)(w >> 32); }

namespace {
struct TrapState {
    bool inited = false, patched = false;
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    uint64_t handler_va = 0, cave_va = 0, live_base = 0;
    uint64_t ret_stub[16], armed_stub[16];
    uint8_t orig_390[16] = {};
    uint64_t call_words[2] = {};
    int agent_pid = -1;
    L3TrapShm *shm = nullptr;
    char shm_name[64] = {};
    uint64_t drv_preempt_buf = 0;   // driver-patched preemption buffer (handler reads it)
    uint64_t scratch_buf = 0;       // driver-patched scratchpad (host rw-s mapping)
    // device-side cave writer (host instruction writes serialise behind the
    // app's running kernels, a concurrent kernel does not)
    CUmodule wr_mod = nullptr;
    CUfunction wr_fn = nullptr;
    CUstream wr_stream = nullptr;
    int wr_lowpri = 0, wr_highpri = 0;
    CUdeviceptr wr_buf = 0;      // [0,0x80) return stub, [0x80,0x100) armed
    // Host-mapped record page the trap can store into; the CPU reads it
    // directly, so probe results survive a parked/debug-state context.
    volatile uint32_t *rec_host = nullptr;
    CUdeviceptr rec_dev = 0;
};
TrapState T;
} // namespace

// A concurrent kernel that copies one of the two stubs into the cave.  Host
// copies to instruction memory wait for the app's blocking streams; a kernel
// launched on our own non-blocking stream does not.
static const char *kWriterPtx =
    ".version 8.7\n.target sm_120\n.address_size 64\n"
    ".visible .entry l3write(.param .u64 p, .param .u64 src)\n{\n"
    "  .reg .u32 %rtid, %r0;\n"
    "  .reg .u64 %rd1, %rd2, %rd3, %rd4, %rd5;\n"
    "  ld.param.u64 %rd1, [p];\n"
    "  ld.param.u64 %rd2, [src];\n"
    "  mov.u32 %rtid, %tid.x;\n"
    "  mul.wide.u32 %rd3, %rtid, 4;\n"
    "  add.s64 %rd4, %rd1, %rd3;\n"
    "  add.s64 %rd5, %rd2, %rd3;\n"
    "  ld.global.u32 %r0, [%rd5];\n"
    "  st.global.u32 [%rd4], %r0;\n"
    "  membar.sys;\n"
    "  ret;\n}\n";

// cuMemHostAlloc is not part of the generated Driver wrapper table.
static CUresult HostAllocPinned(void **p, size_t n, unsigned flags)
{
    static CUresult (*fn)(void **, size_t, unsigned) = nullptr;
    if (fn == nullptr) {
        void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (h == nullptr) h = dlopen("libcuda.so", RTLD_NOW | RTLD_NOLOAD);
        if (h != nullptr) {
            fn = (CUresult (*)(void **, size_t, unsigned))dlsym(h, "cuMemHostAlloc");
        }
        if (fn == nullptr) return CUDA_ERROR_NOT_FOUND;
    }
    return fn(p, n, flags);
}

static bool InitWriter()
{
    if (T.wr_fn != nullptr) return true;
    CUresult r;
    if ((r = Driver::ModuleLoadData(&T.wr_mod, kWriterPtx)) != CUDA_SUCCESS) {
        XWARN("sm120 trap: cave writer module load -> %d", (int)r);
        return false;
    }
    if ((r = Driver::ModuleGetFunction(&T.wr_fn, T.wr_mod, "l3write")) != CUDA_SUCCESS) {
        XWARN("sm120 trap: cave writer function -> %d", (int)r);
        T.wr_fn = nullptr;
        return false;
    }
    if ((r = Driver::CtxGetStreamPriorityRange(&T.wr_lowpri, &T.wr_highpri)) != CUDA_SUCCESS)
        T.wr_highpri = 0;
    if ((r = Driver::StreamCreateWithPriority(&T.wr_stream, CU_STREAM_NON_BLOCKING,
                                              T.wr_highpri)) != CUDA_SUCCESS) {
        XWARN("sm120 trap: cave writer stream -> %d", (int)r);
        T.wr_fn = nullptr;
        return false;
    }
    if ((r = Driver::MemAlloc_v2(&T.wr_buf, 0x200)) != CUDA_SUCCESS) {
        XWARN("sm120 trap: cave writer buffer -> %d", (int)r);
        T.wr_fn = nullptr;
        return false;
    }
    {
        void *hp = nullptr;
        if (HostAllocPinned(&hp, 0x1000, CU_MEMHOSTALLOC_DEVICEMAP) == CUDA_SUCCESS) {
            memset(hp, 0, 0x1000);
            T.rec_host = (volatile uint32_t *)hp;
            // UVA: the device address of a host allocation is the host VA.
            T.rec_dev = (CUdeviceptr)(uintptr_t)hp;
            XINFO("sm120 trap: record page host=%p dev=%#llx", hp,
                  (unsigned long long)T.rec_dev);
        } else {
            XWARN("sm120 trap: host record page unavailable");
        }
    }
    r = Driver::MemcpyHtoD_v2_ptds(T.wr_buf, T.ret_stub, sizeof T.ret_stub);
    if (r == CUDA_SUCCESS)
        r = Driver::MemcpyHtoD_v2_ptds(T.wr_buf + 0x80, T.armed_stub, sizeof T.armed_stub);
    if (r != CUDA_SUCCESS) {
        XWARN("sm120 trap: cave writer stub copy -> %d", (int)r);
        T.wr_fn = nullptr;
        return false;
    }
    if (getenv("XSCHED_SM120_TRAP_REASON_PROBE") != nullptr) {
        std::thread([]() {
            sleep(4);
            Driver::CtxSetCurrent(T.ctx);
            uint32_t v[4] = {0, 0, 0, 0};
            CUresult rr = Driver::MemcpyDtoHAsync_v2_ptsz(v, T.wr_buf + 0x100,
                                                          sizeof v, T.wr_stream);
            if (rr == CUDA_SUCCESS) {
                for (int i = 0; i < 500; i++) {
                    if (Driver::StreamQuery_ptsz(T.wr_stream) == CUDA_SUCCESS) break;
                    usleep(10000);
                }
            }
            uint64_t ldg[2] = {0, 0};
            Driver::MemcpyDtoHAsync_v2_ptsz(ldg, T.wr_buf + 0x40, sizeof ldg, T.wr_stream);
            for (int i = 0; i < 200; i++) {
                if (Driver::StreamQuery_ptsz(T.wr_stream) == CUDA_SUCCESS) break;
                usleep(10000);
            }
            XINFO("sm120 trap: reason probe buf %#x %#x %#x %#x ldg=%#llx (r=%d)",
                  v[0], v[1], v[2], v[3], (unsigned long long)ldg[0], (int)rr);
        }).detach();
    }
    return true;
}

static bool LaunchWriter(bool armed)
{
    if (!InitWriter()) return false;
    uint64_t dst = T.cave_va, src = T.wr_buf + (armed ? 0x80 : 0);
    void *args[2] = { &dst, &src };
    CUresult r = Driver::LaunchKernel_ptsz(T.wr_fn, 1, 1, 1, 32, 1, 1, 0,
                                           T.wr_stream, args, nullptr);
    if (r != CUDA_SUCCESS) { XWARN("sm120 trap: cave writer launch -> %d", (int)r); return false; }
    // wait for the tiny kernel without synchronising the app's streams
    CUresult last = CUDA_SUCCESS;
    for (int i = 0; i < 2000; i++) {
        last = Driver::StreamQuery_ptsz(T.wr_stream);
        if (last == CUDA_SUCCESS) return true;
        usleep(10);
    }
    XWARN("sm120 trap: cave writer did not finish (query=%d)", (int)last);
    return false;
}

// Copy a 32-bit value to an arbitrary device VA with the same concurrent
// writer kernel used for the cave (src is prepared in w r_buf+0x1c0).
[[maybe_unused]] static bool DeviceWrite32(uint64_t dst_va, uint32_t value)
{
    if (!InitWriter()) return false;
    if (Driver::MemcpyHtoDAsync_v2_ptsz(T.wr_buf + 0x1c0, &value, sizeof value,
                                        T.wr_stream) != CUDA_SUCCESS)
        return false;
    uint64_t dst = dst_va, src = T.wr_buf + 0x1c0;
    void *args[2] = { &dst, &src };
    if (Driver::LaunchKernel_ptsz(T.wr_fn, 1, 1, 1, 1, 1, 1, 0, T.wr_stream,
                                  args, nullptr) != CUDA_SUCCESS)
        return false;
    for (int i = 0; i < 2000; i++) {
        CUresult q = Driver::StreamQuery_ptsz(T.wr_stream);
        if (q == CUDA_SUCCESS) return true;
        if (q != CUDA_ERROR_NOT_READY) { XWARN("sm120 trap: dev write query %d", (int)q); return false; }
        usleep(10);
    }
    XWARN("sm120 trap: dev write timed out");
    return false;
}

// ------------------------------------------------------------ arena + patch
static bool ReadImageSlice(CUdeviceptr pc, size_t size, std::vector<uint8_t> &out)
{
    out.resize(size);
    return cuXtraMemcpyDtoH(out.data(), pc, size, T.stream) == size;
}

// The driver traps kernels at launch once debugger parameters are set.  The
// sm120 handler has no benign path for such a trap, so try to turn the launch
// blocking off and keep only the explicit 0x83de0317 trap.
static void ClearDebuggerLaunchBlocking()
{
    if (getenv("XSCHED_SM120_TRAP_DBGFLAGS") == nullptr) return;
    void *h = nullptr;
    const char *paths[] = {"libcuda.so.1", "libcuda.so"};
    for (const char *p : paths) {
        h = dlopen(p, RTLD_NOW | RTLD_NOLOAD);
        if (h) break;
    }
    if (h == nullptr) { XWARN("sm120 trap: libcuda not found for debug flags"); return; }
    // Launch blocking makes every kernel launch wait for the debugger (the
    // replay saw cuLaunchKernel stall ~4.6 s), so switch it OFF; keep the
    // other two flags on.  Never touch cudbgAttachHandlerAvailable /
    // cudbgDebuggerCapabilities.
    int *lb = (int *)dlsym(h, "cudbgEnableLaunchBlocking");
    if (lb) { XINFO("sm120 trap: cudbgEnableLaunchBlocking = %d -> 0", *lb); *lb = 0; }
    const char *names[] = {"cudbgEnablePreemptionDebugging", "cudbgDebuggerInitialized"};
    for (const char *n : names) {
        int *v = (int *)dlsym(h, n);
        if (v) { XINFO("sm120 trap: %s = %d -> 1", n, *v); *v = 1; }
    }
}

static void InitCudbg()
{
    if (getenv("XSCHED_SM120_TRAP_CUDBG") == nullptr) return;
    void *lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (lc == nullptr) lc = dlopen("libcuda.so", RTLD_NOW | RTLD_NOLOAD);
    if (lc == nullptr) { XWARN("sm120 trap: libcuda not loaded for cudbg init"); return; }
    int (*init1)(int) = (int (*)(int))dlsym(lc, "cudbgApiInit");
    if (init1 == nullptr) { XWARN("sm120 trap: cudbgApiInit not exported"); return; }
    int mode = getenv("XSCHED_SM120_TRAP_CUDBG_MODE") ? atoi(getenv("XSCHED_SM120_TRAP_CUDBG_MODE")) : 1;
    int r = init1(mode);
    XINFO("sm120 trap: cudbgApiInit(%d) -> %d", mode, r);
    if (getenv("XSCHED_SM120_TRAP_CUDBG_ATTACH") != nullptr) {
        // cudbgApiAttach() drives the debugger-state transition cuda-gdb
        // performs on attach (contexts switched to the debug vtbl).  Without
        // it the driver suspends every new launch on a stop nobody answers.
        int (*attach)() = (int (*)())dlsym(lc, "cudbgApiAttach");
        if (attach == nullptr) { XWARN("sm120 trap: cudbgApiAttach not exported"); return; }
        int a = attach();
        XINFO("sm120 trap: cudbgApiAttach() -> %d", a);
    }
}

static bool ArmAtTrigger()
{
    return getenv("XSCHED_SM120_TRAP_ARM_PBUF_PROBE") != nullptr ||
           getenv("XSCHED_SM120_TRAP_ARM_SENTINEL") != nullptr ||
           getenv("XSCHED_SM120_TRAP_ARM_PBUF_WRITE") != nullptr;
}

static bool ProbeAndPatch()
{
    const void *tt_raw = nullptr, *mt_raw = nullptr;
    CUuuid te = UuidOf("cc529e945c4e9d46837c9625868334e4");
    CUuuid me = UuidOf("bfdb432dbf3c5a4a945eb34029e81e75");
    if (Driver::GetExportTable(&tt_raw, &te) != CUDA_SUCCESS ||
        Driver::GetExportTable(&mt_raw, &me) != CUDA_SUCCESS ||
        tt_raw == nullptr || mt_raw == nullptr) {
        XWARN("sm120 trap: export tables unavailable");
        return false;
    }
    const void **tt = (const void **)tt_raw, **mt = (const void **)mt_raw;
    struct { uint64_t flags, ssize; void *handle; uint64_t stub, hsize, rsv; } ti;
    memset(&ti, 0, sizeof ti);
    if (((CUresult (*)(CUcontext, void *))tt[20])(T.ctx, &ti) != CUDA_SUCCESS || !ti.handle) {
        XWARN("sm120 trap: slot20 getter failed");
        return false;
    }
    CUdeviceptr pc = 0; uint64_t sz = 0;
    if (((CUresult (*)(CUcontext, void *, CUdeviceptr *))mt[24])(T.ctx, ti.handle, &pc) != CUDA_SUCCESS ||
        ((CUresult (*)(void *, uint64_t *))mt[25])(ti.handle, &sz) != CUDA_SUCCESS || sz < 0x1300) {
        XWARN("sm120 trap: slot20 image unavailable (pc=%#llx sz=%#llx)",
              (unsigned long long)pc, (unsigned long long)sz);
        return false;
    }
    std::vector<uint8_t> img;
    if (!ReadImageSlice(pc, sz, img)) { XWARN("sm120 trap: image read failed"); return false; }
    uint64_t trt = (uint64_t)Imm32At(&img[0x130]) | ((uint64_t)Imm32At(&img[0x140]) << 32);
    uint64_t base = trt - 0xc38300ULL;
    uint64_t live = base + 0x15f0f00ULL;

    std::vector<uint8_t> hdr;
    if (!ReadImageSlice(live, 0xa0, hdr)) { XWARN("sm120 trap: live handler read failed"); return false; }
    static const uint32_t fp[10] = {
        0x0000795c, 0x1a007956, 0xff007387, 0xff007387, 0xff007387,
        0x00047202, 0x00057202, 0xff067803, 0x17077355, 0xff007387,
    };
    for (int i = 0; i < 10; i++) {
        if (OpcodeAt(&hdr[i * 16]) != fp[i]) {
            XWARN("sm120 trap: live handler fingerprint mismatch at +%#x", i * 16);
            return false;
        }
    }

    uint64_t cave = pc + 0xa00;
    uint64_t tramp = cave + 0x80;
    const uint64_t exit_tramp[2] = {
        0x794dULL, ((uint64_t(0xfea00u) << 32) | 0x3800000ULL) // EXIT (normal mode)
    };

    // return stub: displaced MOV R3,RZ + return to +0x3a0
    for (size_t off = 0; off < 0x80; off += 16) {
        T.ret_stub[off/8] = 0x7918ULL; T.ret_stub[off/8+1] = ((uint64_t(0xfc000u) << 32) | 0x0ULL);
        T.armed_stub[off/8] = 0x7918ULL; T.armed_stub[off/8+1] = ((uint64_t(0xfc000u) << 32) | 0x0ULL);
    }
    // Keep R2 (the reason mask) intact: zeroing it makes the handler skip
    // the real handling of a trap and re-execute the faulting instruction.
    T.ret_stub[0] = ((uint64_t(0xffu) << 32) | 0x37202ULL); T.ret_stub[1] = ((uint64_t(0x3ff000u) << 32) | 0xf00ULL); // MOV R3,RZ
    T.ret_stub[2] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); T.ret_stub[3] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
    T.ret_stub[4] = 0xc087950ULL; T.ret_stub[5] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET

    // Reason-guarded armed stub (FINDINGS v4.3, unconditional default): only a
    // trap whose reason has bit0 (the explicit 0x83de0317 request; launch-time
    // traps carry 0x8) is redirected to the EXIT trampoline at cave+0x80.
    // Everything else takes the exact return path of the original handler.
    // LOP3 encoding is copied verbatim from the live handler (+0x3b0).
    if (getenv("XSCHED_SM120_TRAP_LEGACY_EXIT") == nullptr) {
        T.armed_stub[0] = ((uint64_t)(uint32_t)tramp << 32) | (20ULL << 16) | 0x00007802ULL;
        T.armed_stub[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                          // MOV R20, lo
        T.armed_stub[2] = ((uint64_t)(uint32_t)(tramp >> 32) << 32) | (21ULL << 16) | 0x00007802ULL;
        T.armed_stub[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                          // MOV R21, hi
        T.armed_stub[4] = ((uint64_t(0x1u) << 32) | 0x2ff7812ULL); T.armed_stub[5] = ((uint64_t(0x3fda00u) << 32) | 0x780c0ffULL); // LOP3 P0,RZ,R2,1
        T.armed_stub[6] = 0x48947ULL; T.armed_stub[7] = ((uint64_t(0x3ff200u) << 32) | 0x3800000ULL); // @!P0 BRA +0x10
        T.armed_stub[8] = ((uint64_t(0xffffa800u) << 32) | 0xff147387ULL); T.armed_stub[9] = ((uint64_t(0x33e400u) << 32) | 0x100a00ULL); // STL.64 [0xffffa8], R20
        T.armed_stub[10] = ((uint64_t(0xffu) << 32) | 0x37202ULL); T.armed_stub[11] = ((uint64_t(0x3ff000u) << 32) | 0xf00ULL); // MOV R3, RZ
        T.armed_stub[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); T.armed_stub[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        T.armed_stub[14] = 0xc087950ULL; T.armed_stub[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET.ABS.NODEC
        goto armed_ready;
    }
    // Legacy stub (only when XSCHED_SM120_TRAP_LEGACY_EXIT is set):
    T.armed_stub[0] = ((uint64_t(0x5c00u) << 32) | 0xff047b82ULL); T.armed_stub[1] = ((uint64_t(0xfc000u) << 32) | 0x800ULL); // LDC R4,c[0x170]
    T.armed_stub[2] = ((uint64_t(0x5d00u) << 32) | 0xff057b82ULL); T.armed_stub[3] = ((uint64_t(0xfc000u) << 32) | 0x800ULL); // LDC R5,c[0x174]
    T.armed_stub[4] = 0x40a7980ULL; T.armed_stub[5] = ((uint64_t(0xea200u) << 32) | 0x100900ULL); // LD.E R10,[R4]
    T.armed_stub[6] = ((uint64_t(0xffu) << 32) | 0xa00720cULL); T.armed_stub[7] = ((uint64_t(0x4fda00u) << 32) | 0x3f05270ULL); // ISETP.NE.AND P0,R10,RZ
    T.armed_stub[8] = 0x94dULL; T.armed_stub[9] = ((uint64_t(0xfea00u) << 32) | 0x3800000ULL); // @P0 EXIT
    T.armed_stub[10] = ((uint64_t(0xffu) << 32) | 0x37202ULL); T.armed_stub[11] = ((uint64_t(0x3ff000u) << 32) | 0xf00ULL); // MOV R3,RZ
    T.armed_stub[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); T.armed_stub[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
    T.armed_stub[14] = 0xc087950ULL; T.armed_stub[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
armed_ready:;

    // CALL.ABS (T-0x100)>>8 at handler+0x390
    std::vector<uint8_t> orig;
    if (!ReadImageSlice(live + 0x380, 16, orig)) return false;
    uint64_t o0, o1; memcpy(&o0, &orig[0], 8); memcpy(&o1, &orig[8], 8);
    uint64_t enc = (cave - 0x100) >> 8;
    uint64_t call[2] = {
        ((enc & 0xffffffffULL) << 32) | (o0 & 0xffffffffULL),
        (o1 & ~0xffULL) | ((enc >> 32) & 0xffULL),
    };

    {
        std::vector<uint8_t> orig390;
        if (!ReadImageSlice(live + 0x390, 16, orig390)) return false;
        memcpy(T.orig_390, orig390.data(), 16);
    }
    T.call_words[0] = call[0]; T.call_words[1] = call[1];
    T.handler_va = live; T.cave_va = cave; T.live_base = base;
    T.drv_preempt_buf = base - 0x2000000ULL;
    T.scratch_buf = base + 0x1a00000ULL;
    XINFO("sm120 trap: live_base=%#llx driver preempt_buf=%#llx scratch=%#llx (handler reads +4/+8)",
          (unsigned long long)base, (unsigned long long)T.drv_preempt_buf,
          (unsigned long long)T.scratch_buf);
    if (getenv("XSCHED_SM120_TRAP_DUMP_READ") != nullptr) {
        // Dump the live do_read_trap_reason and print every MOV-imm32, so we
        // can recover the driver-patched preemption-buffer address
        // (CUDA_PREEMPTION_BUFFER_PATCH_ADDR).
        uint64_t rd_va = base + 0x15f2100ULL;
        std::vector<uint8_t> rd(0x600);
        if (ReadImageSlice(rd_va, rd.size(), rd)) {
            uint32_t imm120 = 0, imm130 = 0;
            bool have120 = false, have130 = false;
            for (size_t o = 0; o + 16 <= rd.size(); o += 16) {
                uint64_t w0, w1; memcpy(&w0, &rd[o], 8); memcpy(&w1, &rd[o + 8], 8);
                if ((w0 & 0xffffULL) == 0x7802ULL) {
                    uint32_t imm = (uint32_t)(w0 >> 32);
                    uint32_t reg = (uint32_t)((w0 >> 16) & 0xff);
                    if (o == 0x120) { imm120 = imm; have120 = true; }
                    if (o == 0x130) { imm130 = imm; have130 = true; }
                    if (imm != 0) {
                        XINFO("sm120 trap: do_read+%#zx MOV R%u, %#x", o, reg, imm);
                    }
                }
                (void)w1;
            }
            // The driver may relocate the preemption buffer when the debugger
            // session is created.  do_read+0x120/0x130 are MOV low/high of the
            // patched pbuf address; use the post-session value if it changed.
            if (have120 && have130) {
                uint64_t np = (uint64_t)imm120 | ((uint64_t)imm130 << 32);
                if (np != 0 && np != T.drv_preempt_buf) {
                    XINFO("sm120 trap: pbuf relocated by session: %#llx -> %#llx",
                          (unsigned long long)T.drv_preempt_buf,
                          (unsigned long long)np);
                    T.drv_preempt_buf = np;
                } else if (np != 0) {
                    XINFO("sm120 trap: pbuf immediate %#llx (same as base-0x2000000)",
                          (unsigned long long)np);
                }
            }
            if (getenv("XSCHED_SM120_TRAP_DUMP_RAW") != nullptr) {
                for (size_t o = 0x100; o < 0x200; o += 16) {
                    uint64_t w0, w1; memcpy(&w0, &rd[o], 8); memcpy(&w1, &rd[o + 8], 8);
                    XINFO("sm120 trap: do_read raw +%#zx: %016llx %016llx", o,
                          (unsigned long long)w0, (unsigned long long)w1);
                }
            }
        } else {
            XWARN("sm120 trap: cannot read do_read_trap_reason at %#llx",
                  (unsigned long long)rd_va);
        }
    }
    if (getenv("XSCHED_SM120_TRAP_DUMP_ACCESS") != nullptr) {
        uint32_t dv = 0;
        CUresult rr = Driver::MemcpyDtoH_v2(&dv, T.drv_preempt_buf + 4, sizeof dv);
        uint32_t one = 2, back = 0;
        size_t hw = cuXtraMemcpyHtoD(T.drv_preempt_buf + 4, &one, sizeof one, T.stream);
        cuXtraMemcpyDtoH(&back, T.drv_preempt_buf + 4, sizeof back, T.stream);
        XINFO("sm120 trap: drv_pbuf access: dtoh r=%d val=%#x; xtra htod w=%zu back=%#x",
              (int)rr, dv, hw, back);
    }
    if (getenv("XSCHED_SM120_TRAP_NO_PATCH") != nullptr) {
        XINFO("sm120 trap: live=%#llx cave=%#llx tramp=%#llx (no patch)",
              (unsigned long long)live, (unsigned long long)cave,
              (unsigned long long)tramp);
        return true;
    }
    bool always = getenv("XSCHED_SM120_TRAP_ALWAYS_ARMED") != nullptr;
    bool ret_only = getenv("XSCHED_SM120_TRAP_RETURN_ONLY") != nullptr;
    if (always || ret_only) {
        const uint64_t *stub = ret_only ? T.ret_stub : T.armed_stub;
        size_t w0 = cuXtraMemcpyHtoD(cave, stub, sizeof T.ret_stub, T.stream);
        size_t wt = cuXtraMemcpyHtoD(tramp, exit_tramp, sizeof exit_tramp, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        uint64_t rb[2] = {0, 0};
        cuXtraMemcpyDtoH(rb, live + 0x390, 16, T.stream);
        bool ok = w0 == sizeof T.ret_stub && wt == sizeof exit_tramp && w1 == 16 && rb[0] == call[0] && rb[1] == call[1];
        XINFO("sm120 trap: live=%#llx cave=%#llx tramp=%#llx always-armed=%s",
              (unsigned long long)live, (unsigned long long)cave,
              (unsigned long long)tramp, ok ? "ok" : "FAILED");
        return ok;
    }
    if (getenv("XSCHED_SM120_TRAP_PROBE_ONLY") != nullptr) {
        XINFO("sm120 trap: live=%#llx cave=%#llx tramp=%#llx (probe only)",
              (unsigned long long)live, (unsigned long long)cave,
              (unsigned long long)tramp);
        return true;
    }
    // Default: install the call site now, before any kernel runs.  Rewriting
    // the call site while kernels run blocks in the driver (it is executing
    // code), while the cave can still be switched at runtime -- this is the
    // same split run_l3kill.sh relies on.
    InitWriter();
    if (ArmAtTrigger()) {
        // Arm() swaps this in *after* the launch (and before the 0x83de
        // session exists), so the only trap it sees is the explicit 0317.
        // The cave reads one 32-bit word of the driver's preemption buffer
        // with the handler's own LD.E.CONSTANT and records it in wr_buf, so
        // the host can read it without touching the driver VA.  The sentinel
        // variant stores R2 (reason mask) instead, proving the cave ran.
        bool sentinel = getenv("XSCHED_SM120_TRAP_ARM_SENTINEL") != nullptr;
        bool wr_probe = getenv("XSCHED_SM120_TRAP_ARM_PBUF_WRITE") != nullptr;
        uint64_t off = getenv("XSCHED_SM120_TRAP_DRVBUF_OFF")
                     ? strtoull(getenv("XSCHED_SM120_TRAP_DRVBUF_OFF"), nullptr, 0) : 4ULL;
        uint64_t tva = T.drv_preempt_buf + off;
        uint64_t buf = T.rec_dev ? T.rec_dev + 0x100 : T.wr_buf + 0x100;
        T.armed_stub[0] = (((uint64_t)(uint32_t)buf) << 32) | (8ULL << 16) | 0x00007802ULL;
        T.armed_stub[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                  // MOV R8, buf lo
        T.armed_stub[2] = (((uint64_t)(uint32_t)(buf >> 32)) << 32) | (9ULL << 16) | 0x00007802ULL;
        T.armed_stub[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                  // MOV R9, buf hi
        if (wr_probe) {
            // Can the trap write the driver's preemption buffer?  With
            // WValue=0 store R2 (the 0317 reason, 0x1); otherwise store the
            // given command word (2 = preempt_save), which the handler's next
            // dispatch-loop iteration would pick up as 0x10|0x20.
            uint32_t wval = getenv("XSCHED_SM120_TRAP_ARM_PBUF_WVAL")
                          ? (uint32_t)strtoul(getenv("XSCHED_SM120_TRAP_ARM_PBUF_WVAL"),
                                              nullptr, 0) : 0;
            T.armed_stub[4] = (((uint64_t)(uint32_t)tva) << 32) | (6ULL << 16) | 0x00007802ULL;
            T.armed_stub[5] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                              // MOV R6, tva lo
            T.armed_stub[6] = (((uint64_t)(uint32_t)(tva >> 32)) << 32) | (7ULL << 16) | 0x00007802ULL;
            T.armed_stub[7] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                              // MOV R7, tva hi
            if (wval == 0) {
                T.armed_stub[8] = 0x6027385ULL; T.armed_stub[9] = ((uint64_t(0xfe200u) << 32) | 0x100902ULL);  // ST.E [R6],R2
                T.armed_stub[10] = 0x8027385ULL; T.armed_stub[11] = ((uint64_t(0xfe200u) << 32) | 0x100902ULL); // ST.E [R8],R2
            } else {
                T.armed_stub[8] = (((uint64_t)wval) << 32) | (10ULL << 16) | 0x00007802ULL;
                T.armed_stub[9] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                          // MOV R10, wval
                T.armed_stub[10] = 0x60a7385ULL; T.armed_stub[11] = ((uint64_t(0xfe200u) << 32) | 0x10090aULL); // ST.E [R6],R10
            }
            T.armed_stub[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); T.armed_stub[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
            T.armed_stub[14] = 0xc087950ULL; T.armed_stub[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        } else if (sentinel) {
            T.armed_stub[4] = 0x8027385ULL; T.armed_stub[5] = ((uint64_t(0xfe200u) << 32) | 0x100902ULL); // ST.E [R8],R2
            T.armed_stub[6] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); T.armed_stub[7] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
            T.armed_stub[8] = 0xc087950ULL; T.armed_stub[9] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        } else {
            T.armed_stub[4] = (((uint64_t)(uint32_t)tva) << 32) | (6ULL << 16) | 0x00007802ULL;
            T.armed_stub[5] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                              // MOV R6, tva lo
            T.armed_stub[6] = (((uint64_t)(uint32_t)(tva >> 32)) << 32) | (7ULL << 16) | 0x00007802ULL;
            T.armed_stub[7] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                              // MOV R7, tva hi
            T.armed_stub[8] = 0x60a7980ULL; T.armed_stub[9] = ((uint64_t(0x322400u) << 32) | 0x108900ULL);  // LD.E.CONSTANT R10,[R6]
            T.armed_stub[10] = 0x8007385ULL; T.armed_stub[11] = ((uint64_t(0xfe200u) << 32) | 0x10090aULL); // ST.E [R8],R10
            T.armed_stub[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); T.armed_stub[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
            T.armed_stub[14] = 0xc087950ULL; T.armed_stub[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        }
        // LaunchWriter(true) copies wr_buf+0x80, so refresh it with the probe.
        CUresult cr = Driver::MemcpyHtoD_v2_ptds(T.wr_buf + 0x80, T.armed_stub,
                                                 sizeof T.armed_stub);
        XINFO("sm120 trap: arm %s off=%llu tva=%#llx buf=%#llx upload=%d",
              wr_probe ? "pbuf write" : (sentinel ? "sentinel" : "pbuf probe"),
              (unsigned long long)off, (unsigned long long)tva,
              (unsigned long long)buf, (int)cr);
        if (getenv("XSCHED_SM120_TRAP_DUMP_READ") == nullptr) {
            XWARN("sm120 trap: arm probe without DUMP_READ: address may be stale (%#llx)",
                  (unsigned long long)T.drv_preempt_buf);
        }
    }
    if (getenv("XSCHED_SM120_TRAP_PBUF_LOAD_PROBE") != nullptr) {
        // Replace the driver's own pbuf load (do_read+0x140) with a jump to a
        // cave that executes the same load, stores R0 (cmd) to wr_buf, and
        // jumps back to +0x150.  Unlike re-reading pbuf at +0x390, this runs
        // where the driver has the pbuf mapped.
        uint64_t rd_va = base + 0x15f2100ULL;
        uint64_t site = rd_va + 0x140ULL;
        std::vector<uint8_t> ld;
        if (!ReadImageSlice(site, 16, ld)) {
            XWARN("sm120 trap: pbuf load probe cannot read site");
            return false;
        }
        uint64_t lw0, lw1; memcpy(&lw0, &ld[0], 8); memcpy(&lw1, &ld[8], 8);
        auto setjmp_instr = [](uint8_t *in, uint64_t addr) {
            uint64_t w0 = ((uint64_t(0x34567890u) << 32) | 0x794aULL), w1 = ((uint64_t(0xfea00u) << 32) | 0x3807f12ULL);
            memcpy(in, &w0, 8); memcpy(in + 8, &w1, 8);
            in[4] = (uint8_t)(addr & 0xff); in[5] = (uint8_t)((addr >> 8) & 0xff);
            in[6] = (uint8_t)((addr >> 16) & 0xff); in[7] = (uint8_t)((addr >> 24) & 0xff);
            in[8] = (uint8_t)((addr >> 32) & 0xff); in[9] = (uint8_t)((addr >> 40) & 0xff);
        };
        uint64_t rec[16] = {};
        uint64_t buf = T.wr_buf + 0x100;
        bool benign = (strcmp(getenv("XSCHED_SM120_TRAP_PBUF_LOAD_PROBE"), "benign") == 0);
        if (benign) {
            // Control: only re-execute the displaced MOV R3 and jump back, to
            // test the JMP encoding/site without the pbuf load/store.
            rec[0] = ((uint64_t(0x630b0u) << 32) | 0x37802ULL); rec[1] = ((uint64_t(0x3fec00u) << 32) | 0xf00ULL); // MOV R3,0x630b0
            setjmp_instr((uint8_t *)&rec[2], site + 16);                 // JMP back +0x150
            for (int i = 4; i < 16; i += 2) {
                rec[i] = 0x7918ULL; rec[i + 1] = ((uint64_t(0xfc000u) << 32) | 0x0ULL);
            }
        } else {
            rec[0] = (((uint64_t)(uint32_t)buf) << 32) | (8ULL << 16) | 0x00007802ULL;
            rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                              // MOV R8, lo
            rec[2] = (((uint64_t)(uint32_t)(buf >> 32)) << 32) | (9ULL << 16) | 0x00007802ULL;
            rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                              // MOV R9, hi
            rec[4] = lw0; rec[5] = lw1;                                  // driver's pbuf load -> R0
            rec[6] = 0x8007385ULL; rec[7] = ((uint64_t(0xfe200u) << 32) | 0x100900ULL); // ST.E [R8],R0
            setjmp_instr((uint8_t *)&rec[8], site + 16);                 // JMP back +0x150
            for (int i = 10; i < 16; i += 2) {
                rec[i] = 0x7918ULL; rec[i + 1] = ((uint64_t(0xfc000u) << 32) | 0x0ULL);
            }
        }
        uint8_t jmp_site[16];
        setjmp_instr(jmp_site, cave);
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(site, jmp_site, sizeof jmp_site, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: pbuf load probe site=%#llx cave=%#llx w=%zu/%zu buf=%#llx ld=%016llx/%016llx",
              (unsigned long long)site, (unsigned long long)cave, w0, w1,
              (unsigned long long)buf, (unsigned long long)lw0, (unsigned long long)lw1);
        return w0 == sizeof rec && w1 == sizeof jmp_site;
    }
    if (getenv("XSCHED_SM120_TRAP_PBUF_READ_PROBE") != nullptr) {
        // Capture the value the DRIVER already loaded from its preemption
        // buffer.  Live do_read:
        //   +0x120 MOV R4, pbuf_lo ; +0x130 MOV R5, pbuf_hi
        //   +0x140 LD.E.CONSTANT R0,[R4+4]   (cmd; 2 on 0317)
        //   +0x150 MOV R3, 0x630b0            (displaced by the call)
        // The pbuf is only mapped on the driver's path; re-reading it from a
        // +0x390 cave faults at launch (700).  So patch +0x150 and store R0.
        uint64_t rd_va = base + 0x15f2100ULL;
        uint64_t site = rd_va + 0x150ULL;
        std::vector<uint8_t> orig;
        if (!ReadImageSlice(site, 16, orig)) {
            XWARN("sm120 trap: pbuf read probe cannot read do_read site");
            return false;
        }
        uint64_t o0, o1; memcpy(&o0, &orig[0], 8); memcpy(&o1, &orig[8], 8);
        uint64_t enc = (cave - 0x100) >> 8;
        uint64_t call2[2] = {
            ((enc & 0xffffffffULL) << 32) | (o0 & 0xffffffffULL),
            (o1 & ~0xffULL) | ((enc >> 32) & 0xffULL),
        };
        uint64_t rec[16] = {};
        uint64_t buf = T.wr_buf + 0x100;
        rec[0] = (((uint64_t)(uint32_t)buf) << 32) | (8ULL << 16) | 0x00007802ULL;
        rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                  // MOV R8, lo
        rec[2] = (((uint64_t)(uint32_t)(buf >> 32)) << 32) | (9ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                  // MOV R9, hi
        rec[4] = 0x8007385ULL; rec[5] = ((uint64_t(0xfe200u) << 32) | 0x100900ULL);  // ST.E [R8],R0
        rec[6] = ((uint64_t(0x630b0u) << 32) | 0x37802ULL); rec[7] = ((uint64_t(0x3fec00u) << 32) | 0xf00ULL);  // MOV R3,0x630b0
        rec[8] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[9] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL);  // IADD R12
        rec[10] = 0xc087950ULL; rec[11] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        rec[12] = 0x7918ULL; rec[13] = ((uint64_t(0xfc000u) << 32) | 0x0ULL); // NOP
        rec[14] = 0x7918ULL; rec[15] = ((uint64_t(0xfc000u) << 32) | 0x0ULL); // NOP
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(site, call2, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: pbuf read probe site=%#llx cave=%#llx w=%zu/%zu buf=%#llx",
              (unsigned long long)site, (unsigned long long)cave, w0, w1,
              (unsigned long long)buf);
        return w0 == sizeof rec && w1 == 16;
    }    if (getenv("XSCHED_SM120_TRAP_DRVBUF_RW") != nullptr) {
        // Write R2 (reason, =1) into the driver's preemption buffer +4 and read
        // it back with the handler's own LD.E.CONSTANT; store the readback into
        // our buffer.  If it reads 1 the trap context can write that area.
        uint64_t off = getenv("XSCHED_SM120_TRAP_DRVBUF_OFF")
                     ? strtoull(getenv("XSCHED_SM120_TRAP_DRVBUF_OFF"), nullptr, 0) : 4ULL;
        uint64_t tva = T.drv_preempt_buf + off;
        uint64_t rec[16] = {};
        uint64_t buf = T.wr_buf + 0x100;
        rec[0] = (((uint64_t)(uint32_t)buf) << 32) | (8ULL << 16) | 0x00007802ULL;
        rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                   // MOV R8, buf lo
        rec[2] = (((uint64_t)(uint32_t)(buf >> 32)) << 32) | (9ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                   // MOV R9, buf hi
        rec[4] = (((uint64_t)(uint32_t)tva) << 32) | (6ULL << 16) | 0x00007802ULL;
        rec[5] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                   // MOV R6, lo
        rec[6] = (((uint64_t)(uint32_t)(tva >> 32)) << 32) | (7ULL << 16) | 0x00007802ULL;
        rec[7] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                   // MOV R7, hi
        rec[8] = 0x6027385ULL; rec[9] = ((uint64_t(0xfe200u) << 32) | 0x100902ULL);   // ST.E [R6],R2  (write +4)
        rec[10] = 0x8027385ULL; rec[11] = ((uint64_t(0xfe200u) << 32) | 0x100902ULL); // ST.E [R8],R2  (record R2)
        rec[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        rec[14] = 0xc087950ULL; rec[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: drvbuf RW probe off=%llu tva=%#llx w=%zu/%zu",
              (unsigned long long)off, (unsigned long long)tva, w0, w1);
        return w0 == sizeof rec && w1 == 16;
    }
    if (getenv("XSCHED_SM120_TRAP_DRVBUF_WRITE") != nullptr) {
        // Can the TRAP context write the driver's preemption buffer?  Write a
        // magic to +4 here; the next experiment reads it back with the
        // read-probe (two-run check, since the host cannot read that VA).
        uint64_t off = getenv("XSCHED_SM120_TRAP_DRVBUF_OFF")
                     ? strtoull(getenv("XSCHED_SM120_TRAP_DRVBUF_OFF"), nullptr, 0) : 4ULL;
        uint64_t tva = T.drv_preempt_buf + off;
        if (getenv("XSCHED_SM120_TRAP_SCRATCH") != nullptr) {
            // trap -> host write test: the cave stores a magic into the
            // scratchpad; the host reads it back through the rw-s mapping.
            tva = T.scratch_buf + off;
        }
        uint32_t magic = 0x5a5a1234;
        uint64_t rec[16] = {};
        rec[0] = (((uint64_t)(uint32_t)tva) << 32) | (6ULL << 16) | 0x00007802ULL;
        rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[2] = (((uint64_t)(uint32_t)(tva >> 32)) << 32) | (7ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[4] = (((uint64_t)magic) << 32) | (10ULL << 16) | 0x00007802ULL;
        rec[5] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[6] = 0xa007385ULL; rec[7] = ((uint64_t(0xfe200u) << 32) | 0x10090aULL);   // ST.E [R6],R10
        rec[8] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[9] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL);   // IADD R12
        rec[10] = 0xc087950ULL; rec[11] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: drvbuf WRITE probe off=%llu tva=%#llx magic=%#x w=%zu/%zu",
              (unsigned long long)off, (unsigned long long)tva, magic, w0, w1);
        return w0 == sizeof rec && w1 == 16;
    }
    if (getenv("XSCHED_SM120_TRAP_DRVBUF_PROBE") != nullptr) {
        // Read the DRIVER's preemption buffer (base-0x2000000) with the exact
        // load flavour the handler itself uses (LD.E.CONSTANT).  This makes the
        // driver's writes (exit flag / command word / preempt_idx) observable.
        uint64_t off = getenv("XSCHED_SM120_TRAP_DRVBUF_OFF")
                     ? strtoull(getenv("XSCHED_SM120_TRAP_DRVBUF_OFF"), nullptr, 0) : 4ULL;
        uint64_t tva = T.drv_preempt_buf + off;
        if (getenv("XSCHED_SM120_TRAP_SCRATCH") != nullptr) {
            // host -> trap read test: CPU writes the sentinel through the
            // /dev/nvidiactl rw-s scratchpad mapping, the cave reads it back
            // with LD.E.CONSTANT (the driver's own load flavour).
            tva = T.scratch_buf + off;
            uint64_t sentinel = ((uint64_t(0x11223344u) << 32) | 0x55667788ULL);
            if (getenv("XSCHED_SM120_TRAP_SCRATCH_NOWRITE") == nullptr) {
                memcpy((void *)tva, &sentinel, sizeof sentinel);
                __sync_synchronize();
                XINFO("sm120 trap: scratch host wrote %#llx to %#llx",
                      (unsigned long long)sentinel, (unsigned long long)tva);
            }
        }
        uint64_t rec[16] = {};
        uint64_t buf = T.wr_buf + 0x100;
        rec[0] = (((uint64_t)(uint32_t)buf) << 32) | (8ULL << 16) | 0x00007802ULL;
        rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[2] = (((uint64_t)(uint32_t)(buf >> 32)) << 32) | (9ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[4] = (((uint64_t)(uint32_t)tva) << 32) | (6ULL << 16) | 0x00007802ULL;
        rec[5] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[6] = (((uint64_t)(uint32_t)(tva >> 32)) << 32) | (7ULL << 16) | 0x00007802ULL;
        rec[7] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[8] = 0x60a7980ULL; rec[9] = ((uint64_t(0x322400u) << 32) | 0x108900ULL);   // LD.E.CONSTANT R10,[R6]
        rec[10] = 0x8007385ULL; rec[11] = ((uint64_t(0xfe200u) << 32) | 0x10090aULL); // ST.E [R8],R10
        rec[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        rec[14] = 0xc087950ULL; rec[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        uint64_t sentinel = ((uint64_t(0xdeadbeefu) << 32) | 0xcafef00dULL);
        cuXtraMemcpyHtoD(buf, &sentinel, sizeof sentinel, T.stream);
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: drvbuf probe off=%llu tva=%#llx w=%zu/%zu", (unsigned long long)off,
              (unsigned long long)tva, w0, w1);
        return w0 == sizeof rec && w1 == 16;
    }
    if (getenv("XSCHED_SM120_TRAP_CONST_PROBE") != nullptr) {
        // Same as MEM_PROBE but using LD.E.CONSTANT -- the exact load flavour
        // the driver's own do_read_trap_reason uses (bytes taken from the
        // plugin disassembly at +0x220).
        uint64_t off = getenv("XSCHED_SM120_TRAP_MEM_OFF")
                     ? strtoull(getenv("XSCHED_SM120_TRAP_MEM_OFF"), nullptr, 0) : 0x40ULL;
        uint64_t tva = T.wr_buf + off;
        uint64_t magic = ((uint64_t(0x11223344u) << 32) | 0x55667788ULL);
        size_t mw = cuXtraMemcpyHtoD(tva, &magic, sizeof magic, T.stream);
        uint64_t rec[16] = {};
        uint64_t buf = T.wr_buf + 0x100;
        rec[0] = (((uint64_t)(uint32_t)buf) << 32) | (8ULL << 16) | 0x00007802ULL;
        rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[2] = (((uint64_t)(uint32_t)(buf >> 32)) << 32) | (9ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[4] = (((uint64_t)(uint32_t)tva) << 32) | (6ULL << 16) | 0x00007802ULL;
        rec[5] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[6] = (((uint64_t)(uint32_t)(tva >> 32)) << 32) | (7ULL << 16) | 0x00007802ULL;
        rec[7] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[8] = 0x60a7980ULL; rec[9] = ((uint64_t(0x322400u) << 32) | 0x108900ULL);   // LD.E.CONSTANT R10,[R6]
        rec[10] = 0x8007385ULL; rec[11] = ((uint64_t(0xfe200u) << 32) | 0x10090aULL); // ST.E [R8],R10
        rec[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        rec[14] = 0xc087950ULL; rec[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        uint64_t sentinel = ((uint64_t(0xdeadbeefu) << 32) | 0xcafef00dULL);
        cuXtraMemcpyHtoD(buf, &sentinel, sizeof sentinel, T.stream);
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: const probe tva=%#llx magic=%#llx w=%zu/%zu (mw=%zu)", (unsigned long long)tva,
              (unsigned long long)magic, w0, w1, mw);
        return w0 == sizeof rec && w1 == 16;
    }
    if (getenv("XSCHED_SM120_TRAP_SRC") != nullptr) {
        // Try every remaining memory type as a host->handler channel:
        //   instr   = cuXtraInstrMemBlockAlloc (instruction memory block)
        //   managed = cuMemAllocManaged
        //   scratch = driver scratchpad (read-only)
        //   pause   = driver pause_reason_table (read-only)
        const char *src = getenv("XSCHED_SM120_TRAP_SRC");
        static CUdeviceptr stva = 0;
        static uint64_t magic = ((uint64_t(0x5a5a5a5au) << 32) | 0x12345678ULL);
        static bool ready = false;
        static int alloc_r = -1;
        if (!ready) {
            if (!strcmp(src, "instr")) {
                stva = cuXtraInstrMemBlockAlloc(T.ctx, 0x1000);
                alloc_r = (int)cuXtraInstrMemcpyHtoD(stva, &magic, 8, T.stream);
            } else if (!strcmp(src, "managed")) {
                CUresult r = Driver::MemAllocManaged(&stva, 4096, 1 /* ATTACH_GLOBAL */);
                alloc_r = (int)r;
                if (r == CUDA_SUCCESS) Driver::MemcpyHtoD_v2_ptds(stva, &magic, 8);
            } else if (!strcmp(src, "scratch")) {
                stva = base + 0x1a00000ULL; alloc_r = 0;   // read-only
            } else if (!strcmp(src, "pause")) {
                stva = base + 0x1000000ULL; alloc_r = 0;   // read-only
            } else {
                XWARN("sm120 trap: unknown SRC %s", src);
                return false;
            }
            ready = true;
            uint64_t hv = 0;
            cuXtraMemcpyDtoH(&hv, stva, 8, T.stream);
            XINFO("sm120 trap: src=%s va=%#llx r=%d magic=%#llx hostread=%#llx", src,
                  (unsigned long long)stva, alloc_r, (unsigned long long)magic,
                  (unsigned long long)hv);
        }
        uint64_t rec[16] = {};
        uint64_t buf = T.wr_buf + 0x100;
        rec[0] = (((uint64_t)(uint32_t)buf) << 32) | (8ULL << 16) | 0x00007802ULL;
        rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[2] = (((uint64_t)(uint32_t)(buf >> 32)) << 32) | (9ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[4] = (((uint64_t)(uint32_t)stva) << 32) | (6ULL << 16) | 0x00007802ULL;
        rec[5] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[6] = (((uint64_t)(uint32_t)(stva >> 32)) << 32) | (7ULL << 16) | 0x00007802ULL;
        rec[7] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[8] = 0x60a7980ULL; rec[9] = ((uint64_t(0xea200u) << 32) | 0x100900ULL);   // LD.E R10,[R6]
        rec[10] = 0x8007385ULL; rec[11] = ((uint64_t(0xfe200u) << 32) | 0x10090aULL); // ST.E [R8],R10
        rec[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        rec[14] = 0xc087950ULL; rec[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        uint64_t sentinel = ((uint64_t(0xdeadbeefu) << 32) | 0xcafef00dULL);
        cuXtraMemcpyHtoD(buf, &sentinel, sizeof sentinel, T.stream);
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: src probe installed src=%s va=%#llx w=%zu/%zu", src,
              (unsigned long long)stva, w0, w1);
        return w0 == sizeof rec && w1 == 16;
    }
    if (getenv("XSCHED_SM120_TRAP_UVA_PROBE") != nullptr) {
        // Read host-mapped (UVA) memory from the trap context: different
        // aperture than device memory, and a candidate host->handler channel.
        static void *hp = nullptr;
        static CUdeviceptr dp = 0;
        if (hp == nullptr) {
            CUresult r1 = Driver::MemHostAlloc(&hp, 4096,
                CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE);
            CUresult r2 = Driver::MemHostGetDevicePointer_v2(&dp, hp, 0);
            XINFO("sm120 trap: uva alloc host=%p dev=%#llx r=%d/%d", hp,
                  (unsigned long long)dp, (int)r1, (int)r2);
            memset(hp, 0, 4096);
        }
        uint64_t tva = (uint64_t)dp + 0x40;
        uint64_t magic = ((uint64_t(0xa5a5c0deu) << 32) | 0x12345678ULL);
        memcpy((char *)hp + 0x40, &magic, sizeof magic);
        uint64_t rec[16] = {};
        uint64_t buf = T.wr_buf + 0x100;
        rec[0] = (((uint64_t)(uint32_t)buf) << 32) | (8ULL << 16) | 0x00007802ULL;
        rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[2] = (((uint64_t)(uint32_t)(buf >> 32)) << 32) | (9ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[4] = (((uint64_t)(uint32_t)tva) << 32) | (6ULL << 16) | 0x00007802ULL;
        rec[5] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[6] = (((uint64_t)(uint32_t)(tva >> 32)) << 32) | (7ULL << 16) | 0x00007802ULL;
        rec[7] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[8] = 0x60a7980ULL; rec[9] = ((uint64_t(0xea200u) << 32) | 0x100900ULL);   // LD.E R10,[R6]
        rec[10] = 0x8007385ULL; rec[11] = ((uint64_t(0xfe200u) << 32) | 0x10090aULL); // ST.E [R8],R10
        rec[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        rec[14] = 0xc087950ULL; rec[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        uint64_t sentinel = ((uint64_t(0xdeadbeefu) << 32) | 0xcafef00dULL);
        cuXtraMemcpyHtoD(buf, &sentinel, sizeof sentinel, T.stream);
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: uva probe tva=%#llx magic=%#llx w=%zu/%zu", (unsigned long long)tva,
              (unsigned long long)magic, w0, w1);
        return w0 == sizeof rec && w1 == 16;
    }
    if (getenv("XSCHED_SM120_TRAP_LDG_PROBE") != nullptr) {
        // Same as MEM_PROBE, but with the descriptor-based global load the
        // compiler emits on sm_120:  LDCU.64 UR4,c[0x0][0x358] + LDG.E.64.
        uint64_t off = getenv("XSCHED_SM120_TRAP_MEM_OFF")
                     ? strtoull(getenv("XSCHED_SM120_TRAP_MEM_OFF"), nullptr, 0) : 0x40ULL;
        uint64_t tva = T.wr_buf + off;
        uint64_t magic = ((uint64_t(0x11223344u) << 32) | 0x55667788ULL);
        size_t mw = cuXtraMemcpyHtoD(tva, &magic, sizeof magic, T.stream);
        uint64_t rec[16] = {};
        rec[0] = ((uint64_t(0x6b00u) << 32) | 0xff0477acULL); rec[1] = ((uint64_t(0xe2e00u) << 32) | 0x8000a00ULL); // LDCU.64 UR4,c[0x358]
        rec[2] = (((uint64_t)(uint32_t)tva) << 32) | (6ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                   // MOV R6, lo
        rec[4] = (((uint64_t)(uint32_t)(tva >> 32)) << 32) | (7ULL << 16) | 0x00007802ULL;
        rec[5] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                   // MOV R7, hi
        rec[6] = ((uint64_t(0x4u) << 32) | 0x60a7981ULL); rec[7] = ((uint64_t(0x1eac00u) << 32) | 0xc1e1b00ULL);   // LDG.E.64 R10,desc[UR4][R6]
        rec[8] = ((uint64_t(0x1u) << 32) | 0xa0a7835ULL); rec[9] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL);   // IADD R10,R10,1
        rec[10] = ((uint64_t(0xau) << 32) | 0x6007986ULL); rec[11] = ((uint64_t(0x4fe200u) << 32) | 0xc101b04ULL); // STG.E.64 desc[UR4][R6],R10
        rec[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        rec[14] = 0xc087950ULL; rec[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: ldg probe tva=%#llx magic=%#llx w=%zu/%zu", (unsigned long long)tva,
              (unsigned long long)magic, mw, w1);
        return w0 == sizeof rec && w1 == 16;
    }
    if (getenv("XSCHED_SM120_TRAP_MEM_PROBE") != nullptr) {
        // Read our OWN device allocation (host-initialized magic) from the trap
        // context: tests whether the anomaly is "global reads are dead" or
        // "only foreign (preempt buffer) allocations are invisible".
        uint64_t off = getenv("XSCHED_SM120_TRAP_MEM_OFF")
                     ? strtoull(getenv("XSCHED_SM120_TRAP_MEM_OFF"), nullptr, 0) : 0x40ULL;
        bool code = getenv("XSCHED_SM120_TRAP_MEM_CODE") != nullptr;
        // code variant: read the handler's own code page (known content).
        uint64_t tva = code ? (live + 0x390) : (T.wr_buf + off);
        uint64_t magic = ((uint64_t(0x11223344u) << 32) | 0x55667788ULL);
        size_t mw = 0;
        if (!code) mw = cuXtraMemcpyHtoD(tva, &magic, sizeof magic, T.stream);
        uint64_t hv = 0;
        cuXtraMemcpyDtoH(&hv, tva, sizeof hv, T.stream);
        uint64_t rec[16] = {};
        uint64_t buf = T.wr_buf + 0x100;
        rec[0] = (((uint64_t)(uint32_t)buf) << 32) | (8ULL << 16) | 0x00007802ULL;
        rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[2] = (((uint64_t)(uint32_t)(buf >> 32)) << 32) | (9ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[4] = (((uint64_t)(uint32_t)tva) << 32) | (6ULL << 16) | 0x00007802ULL;
        rec[5] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[6] = (((uint64_t)(uint32_t)(tva >> 32)) << 32) | (7ULL << 16) | 0x00007802ULL;
        rec[7] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[8] = 0x60a7980ULL; rec[9] = ((uint64_t(0xea200u) << 32) | 0x100900ULL);   // LD.E R10,[R6]
        rec[10] = 0x8007385ULL; rec[11] = ((uint64_t(0xfe200u) << 32) | 0x10090aULL); // ST.E [R8],R10
        rec[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        rec[14] = 0xc087950ULL; rec[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        uint64_t sentinel = ((uint64_t(0xdeadbeefu) << 32) | 0xcafef00dULL);   // proves the cave ran
        cuXtraMemcpyHtoD(buf, &sentinel, sizeof sentinel, T.stream);
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: mem probe off=%#llx tva=%#llx code=%d magic=%#llx w=%zu/%zu hostread=%#llx call=%#llx",
              (unsigned long long)off, (unsigned long long)tva, (int)code,
              (unsigned long long)magic, mw, w1, (unsigned long long)hv,
              (unsigned long long)call[0]);
        return w0 == sizeof rec && w1 == 16;
    }
    if (getenv("XSCHED_SM120_TRAP_TABLE_PROBE") != nullptr) {
        // Read a driver-managed trap area (trap_reason_table / scratchpad /
        // pause_reason_table) from inside the trap context.  Those areas are
        // mapped for the driver's own handler, so they are the best candidate
        // for a host->handler channel.  The magic is written at Init; the
        // cave stores whatever it reads back into wr_buf+0x100.
        uint64_t off = getenv("XSCHED_SM120_TRAP_TABLE_OFF")
                     ? strtoull(getenv("XSCHED_SM120_TRAP_TABLE_OFF"), nullptr, 0)
                     : 0xc38300ULL;
        uint64_t tva = base + off;
        // Read-only: capture what the host sees at that address, without
        // writing the driver's area (writing it faulted the context).
        uint64_t magic = ((uint64_t(0x5aa5c0deu) << 32) | 0x12345678ULL);
        size_t mw = cuXtraMemcpyHtoD(tva, &magic, sizeof magic, T.stream);
        uint64_t hv = 0;
        cuXtraMemcpyDtoH(&hv, tva, sizeof hv, T.stream);
        XINFO("sm120 trap: table magic write w=%zu hostread=%#llx", mw, (unsigned long long)hv);
        uint64_t rec[16] = {};
        uint64_t buf = T.wr_buf + 0x100;
        rec[0] = (((uint64_t)(uint32_t)buf) << 32) | (8ULL << 16) | 0x00007802ULL;
        rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[2] = (((uint64_t)(uint32_t)(buf >> 32)) << 32) | (9ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        // R6/R7 hold the address, R10 the loaded value: R0-R5 and R12 must be
        // preserved -- the handler writes `STL.64 [R1], R2` right after the
        // cave returns.
        rec[4] = (((uint64_t)(uint32_t)tva) << 32) | (6ULL << 16) | 0x00007802ULL;
        rec[5] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[6] = (((uint64_t)(uint32_t)(tva >> 32)) << 32) | (7ULL << 16) | 0x00007802ULL;
        rec[7] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[8] = 0x60a7980ULL; rec[9] = ((uint64_t(0xea200u) << 32) | 0x100900ULL);   // LD.E R10,[R6]
        rec[10] = 0x8007385ULL; rec[11] = ((uint64_t(0xfe200u) << 32) | 0x10090aULL); // ST.E [R8],R10
        rec[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        rec[14] = 0xc087950ULL; rec[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: table probe installed off=%#llx tva=%#llx magic=%#llx w=%zu/%zu",
              (unsigned long long)off, (unsigned long long)tva,
              (unsigned long long)magic, w0, w1);
        return w0 == sizeof rec && w1 == 16;
    }
    if (getenv("XSCHED_SM120_TRAP_FLAG_PROBE") != nullptr) {
        // Same as the param probe, but ALSO performs the real LD.E from the
        // preempt buffer pointer and records the value: proves whether the
        // trap context can read the scheduler's flag (XSched Lv3 depends on it).
        uint64_t rec[16] = {};
        uint64_t buf = T.wr_buf + 0x100;
        rec[0] = (((uint64_t)(uint32_t)buf) << 32) | (8ULL << 16) | 0x00007802ULL;
        rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[2] = (((uint64_t)(uint32_t)(buf >> 32)) << 32) | (9ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        // 64-bit pointer = c[0x170]:c[0x174] -> R4:R5 (the earlier probe loaded
        // only the low half and used R0:R1, so its result was meaningless).
        rec[4] = ((uint64_t(0x5c00u) << 32) | 0xff047b82ULL); rec[5] = ((uint64_t(0xfc000u) << 32) | 0x800ULL);   // LDC R4,c[0x170]
        rec[6] = ((uint64_t(0x5d00u) << 32) | 0xff057b82ULL); rec[7] = ((uint64_t(0xfc000u) << 32) | 0x800ULL);   // LDC R5,c[0x174]
        rec[8] = 0x40a7980ULL; rec[9] = ((uint64_t(0xea200u) << 32) | 0x100900ULL);   // LD.E R10,[R4]
        rec[10] = 0x8007385ULL; rec[11] = ((uint64_t(0xfe200u) << 32) | 0x10090aULL); // ST.E [R8],R10
        rec[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        rec[14] = 0xc087950ULL; rec[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        rec[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        rec[14] = 0xc087950ULL; rec[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: flag probe installed w=%zu/%zu buf=%#llx", w0, w1,
              (unsigned long long)buf);
        return w0 == sizeof rec && w1 == 16;
    }
    if (const char *p = getenv("XSCHED_SM120_TRAP_DBGPARAM_PROBE")) {
        (void)p;
        // Read one 32-bit debugger-param constant (c[0x0][off]) and store it to
        // wr_buf+0x100 for the reason_probe reader.  Use DBGPARAM_OFF to pick
        // 0x170 (preempt buffer ptr), 0x178 (guardian), 0x180 (kernel_idx?),
        // 0x188 (killable?).  The displaced MOV R3,RZ is restored before RET.
        uint64_t off = 0x180;
        if (const char *po = getenv("XSCHED_SM120_TRAP_DBGPARAM_OFF")) {
            off = strtoull(po, nullptr, 0);
        }
        uint64_t imm = (off & 0xffffffffULL) << 6;   // c[off] encoding: off*64
        uint64_t rec[16] = {};
        uint64_t buf = T.wr_buf + 0x100;
        rec[0] = (((uint64_t)(uint32_t)buf) << 32) | (8ULL << 16) | 0x00007802ULL;
        rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                  // MOV R8, lo
        rec[2] = (((uint64_t)(uint32_t)(buf >> 32)) << 32) | (9ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                  // MOV R9, hi
        rec[4] = (imm << 32) | (0xffULL << 24) | (10ULL << 16) | 0x00007b82ULL;
        rec[5] = ((uint64_t(0xfc000u) << 32) | 0x800ULL);                                  // LDC R10,c[off]
        rec[6] = 0x8007385ULL; rec[7] = ((uint64_t(0xfe200u) << 32) | 0x10090aULL);  // ST.E [R8],R10
        rec[8] = ((uint64_t(0xffu) << 32) | 0x37202ULL); rec[9] = ((uint64_t(0x3ff000u) << 32) | 0xf00ULL);  // MOV R3,RZ
        rec[10] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[11] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        rec[12] = 0xc087950ULL; rec[13] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        rec[14] = 0x7918ULL; rec[15] = ((uint64_t(0xfc000u) << 32) | 0x0ULL); // NOP
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: dbgparam probe off=%#llx (c[0x%llx]) installed w=%zu/%zu buf=%#llx",
              (unsigned long long)off, (unsigned long long)off, w0, w1,
              (unsigned long long)buf);
        return w0 == sizeof rec && w1 == 16;
    }
    if (getenv("XSCHED_SM120_TRAP_PARAM_PROBE") != nullptr) {
        uint64_t rec[16] = {};
        uint64_t lo = (uint64_t)(uint32_t)(T.wr_buf + 0x100);
        uint64_t hi = (uint64_t)(uint32_t)((T.wr_buf + 0x100) >> 32);
        rec[0] = (lo << 32) | (8ULL << 16) | 0x00007802ULL; rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[2] = (hi << 32) | (9ULL << 16) | 0x00007802ULL; rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);
        rec[4] = ((uint64_t(0x5c00u) << 32) | 0xff007b82ULL); rec[5] = ((uint64_t(0xfc000u) << 32) | 0x800ULL);   // LDC R0,c[0x170]
        rec[6] = 0x8007385ULL; rec[7] = ((uint64_t(0xfe200u) << 32) | 0x100902ULL);   // ST.E [R8],R2
        rec[8] = ((uint64_t(0x4u) << 32) | 0x8007385ULL); rec[9] = ((uint64_t(0xfe200u) << 32) | 0x100900ULL);   // ST.E [R8+4],R0
        rec[10] = ((uint64_t(0xffu) << 32) | 0x37202ULL); rec[11] = ((uint64_t(0x3ff000u) << 32) | 0xf00ULL); // MOV R3,RZ
        rec[12] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[13] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL); // IADD R12
        rec[14] = 0xc087950ULL; rec[15] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: param probe installed w=%zu/%zu buf=%#llx", w0, w1,
              (unsigned long long)(T.wr_buf + 0x100));
        return w0 == sizeof rec && w1 == 16;
    }
    if (getenv("XSCHED_SM120_TRAP_REASON_PROBE") != nullptr) {
        // Cave that records the reason mask (R2) the handler computed, so we
        // can tell a launch-time trap from the explicit 0317 trap.
        uint64_t rec[16] = {};
        rec[0] = ((uint64_t)(uint32_t)(T.wr_buf + 0x100) << 32) | (8ULL << 16) | 0x00007802ULL;
        rec[1] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                  // MOV R8, lo
        rec[2] = ((uint64_t)(uint32_t)((T.wr_buf + 0x100) >> 32) << 32) | (9ULL << 16) | 0x00007802ULL;
        rec[3] = ((uint64_t(0x3fe200u) << 32) | 0xf00ULL);                                  // MOV R9, hi
        rec[4] = 0x8027385ULL;                                  // ST.E [R8], R2
        rec[5] = ((uint64_t(0xfe200u) << 32) | 0x100900ULL);
        rec[6] = ((uint64_t(0xffu) << 32) | 0x37202ULL); rec[7] = ((uint64_t(0x3ff000u) << 32) | 0xf00ULL);  // MOV R3,RZ
        rec[8] = ((uint64_t(0x10u) << 32) | 0xc0c7835ULL); rec[9] = ((uint64_t(0x3fe600u) << 32) | 0x78e0000ULL);  // IADD R12
        rec[10] = 0xc087950ULL; rec[11] = ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL); // RET
        size_t w0 = cuXtraMemcpyHtoD(cave, rec, sizeof rec, T.stream);
        size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
        cuXtraInvalInstrCache(T.ctx);
        XINFO("sm120 trap: reason probe installed w=%zu/%zu buf=%#llx",
              w0, w1, (unsigned long long)(T.wr_buf + 0x100));
        return w0 == sizeof rec && w1 == 16;
    }
    // The cave is armed *before* the first launch: at level 3 the guardian
    // spins waiting for the trap, so the GPU is fully occupied and neither a
    // host copy nor a device writer can run once the kernels are up.
    bool pre_arm = getenv("XSCHED_SM120_TRAP_PRE_RETURN") == nullptr && !ArmAtTrigger();
    const uint64_t *stub = pre_arm ? T.armed_stub : T.ret_stub;
    size_t w0 = cuXtraMemcpyHtoD(cave, stub, sizeof T.ret_stub, T.stream);
    size_t wt = cuXtraMemcpyHtoD(tramp, exit_tramp, sizeof exit_tramp, T.stream);
    size_t w1 = cuXtraMemcpyHtoD(live + 0x390, call, 16, T.stream);
    cuXtraInvalInstrCache(T.ctx);
    uint64_t rb[2] = {0, 0};
    cuXtraMemcpyDtoH(rb, live + 0x390, 16, T.stream);
    bool ok = w0 == sizeof T.ret_stub && wt == sizeof exit_tramp && w1 == 16 && rb[0] == call[0] && rb[1] == call[1];
    XINFO("sm120 trap: live=%#llx cave=%#llx tramp=%#llx patched=%s (%s)",
          (unsigned long long)live, (unsigned long long)cave,
          (unsigned long long)tramp, ok ? "ok" : "FAILED", pre_arm ? "armed" : "return");
    return ok;
}

// ------------------------------------------------------------ public API
TrapSm120Context::~TrapSm120Context() = default;

TrapSm120Context &TrapSm120Context::Instance()
{
    static TrapSm120Context inst;
    return inst;
}

bool TrapSm120Context::GetLatestRmHandles(Sm120RmCtxHandles *out)
{
    if (!out) return false;
    if (g_ctx_count > 0) {
        *out = g_ctx_table[g_ctx_count - 1];
        if (out->hDebugger == 0 && H.session != 0) {
            out->hDebugger       = H.session;
            out->hDebuggerClient = H.dbg_client ? H.dbg_client : H.client;
        }
        return (out->ctl_fd >= 0 && out->hClient != 0);
    }
    out->ctl_fd          = H.fd;
    out->hClient         = H.client;
    out->hDevice         = H.device;
    out->hSubDevice      = H.subdev;
    out->hTsg            = H.tsg;
    out->hChannel        = H.channel;
    out->hCompute        = H.ncompute > 0 ? H.computes[H.ncompute - 1] : 0;
    out->hDebugger       = H.session;
    out->hDebuggerClient = H.dbg_client ? H.dbg_client : H.client;
    out->hShareClient    = 0;
    return (out->ctl_fd >= 0 && out->hClient != 0);
}

int TrapSm120Context::GetCapturedContextCount()
{
    return g_ctx_count;
}

bool TrapSm120Context::GetRmHandlesByIndex(int idx, Sm120RmCtxHandles *out)
{
    if (!out || idx < 0 || idx >= g_ctx_count) return false;
    *out = g_ctx_table[idx];
    if (out->hDebugger == 0 && H.session != 0) {
        out->hDebugger       = H.session;
        out->hDebuggerClient = H.dbg_client ? H.dbg_client : H.client;
    }
    return (out->ctl_fd >= 0 && out->hClient != 0);
}

uint64_t TrapSm120Context::TrapCounterValue() const
{
    return counter_host_ ? *counter_host_ : (T.rec_host ? T.rec_host[0x40] : 0);
}

bool TrapSm120Context::PostRecover()
{
    return Resume();
}

bool TrapSm120Context::RmFastTrapPulse83de()
{
    Sm120RmCtxHandles h{};
    if (!GetRmHandlesByIndex(target_slot_, &h)) {
        GetLatestRmHandles(&h);
    }
    int fd = (h.ctl_fd >= 0) ? h.ctl_fd : H.fd;
    uint32_t hClient = h.hClient ? h.hClient : H.client;
    uint32_t hTsg = h.hTsg ? h.hTsg : H.tsg;
    uint32_t hSession = h.hDebugger ? h.hDebugger : H.session;
    uint32_t hDbgClient = h.hDebuggerClient ? h.hDebuggerClient : (H.dbg_client ? H.dbg_client : hClient);

    if (fd < 0 || hClient == 0) return false;
    bool ok = false;
    if (hSession != 0) {
        struct { uint32_t action; uint32_t pad; } mmu_dbg{1u, 0u};
        rm_control(fd, hDbgClient, hSession, 0x83de0329u, &mmu_dbg, sizeof mmu_dbg);

        struct { uint32_t waitForEvent; uint32_t status; } susp{0u, 0u};
        int st1 = rm_control(fd, hDbgClient, hSession, 0x83de0317u, &susp, sizeof susp);

        uint32_t dummy = 0;
        int st2 = rm_control(fd, hDbgClient, hSession, 0x83de0310u, &dummy, sizeof dummy);
        int st3 = rm_control(fd, hDbgClient, hSession, 0x83de0318u, &dummy, sizeof dummy);
        (void)st3;
        if (st1 == 0 || st2 == 0) ok = true;
    }
    if (hTsg != 0) {
        struct { uint64_t timesliceUs; } ts{1ULL};
        rm_control(fd, hClient, hTsg, 0xa06c0103u, &ts, sizeof ts);
        struct { uint8_t bWait; uint8_t bManualTimeout; uint8_t pad[2]; uint32_t timeoutUs; } pr{1, 1, {0, 0}, 50000};
        int st = rm_control(fd, hClient, hTsg, 0xa06c0105u, &pr, sizeof pr);
        if (st == 0) ok = true;
    }
    return ok;
}

bool TrapSm120Context::Init(CUcontext ctx, CUstream stream)
{
    ctx_ = ctx;
    if (!T.inited) {
        T.inited = true; T.ctx = ctx; T.stream = stream;
        prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
        ClearDebuggerLaunchBlocking();
        InitCudbg();

        const char *name = getenv("XSCHED_SM120_TRAP_SHM");
        const char *agent_cfg = getenv("XSCHED_SM120_TRAP_AGENT");
        if ((name != nullptr && name[0] != '\0') || (agent_cfg != nullptr && agent_cfg[0] != '\0')) {
            snprintf(T.shm_name, sizeof T.shm_name, "%s", name ? name : L3_TRAP_SHM_NAME);
            int fd = shm_open(T.shm_name, O_CREAT | O_RDWR, 0666);
            if (fd >= 0 && ftruncate(fd, sizeof(L3TrapShm)) == 0) {
                T.shm = (L3TrapShm *)mmap(nullptr, sizeof(L3TrapShm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                close(fd);
                if (T.shm == MAP_FAILED) T.shm = nullptr;
            } else if (fd >= 0) {
                close(fd);
            }
            if (T.shm != nullptr) {
                memset(T.shm, 0, sizeof *T.shm);
                T.shm->pid = (uint32_t)getpid();
                if (agent_cfg && *agent_cfg) {
                    pid_t pid = fork();
                    if (pid == 0) {
                        execl(agent_cfg, agent_cfg, T.shm_name, (char *)nullptr);
                        _exit(127);
                    }
                    T.agent_pid = (int)pid;
                }
                for (int i = 0; i < 100 && T.shm->agent_ready == 0; i++) usleep(5000);
                if (H.client && H.device && T.shm->agent_ready) {
                    struct { uint32_t target, rights, type; } p;
                    p = { H.client, 1, 0x0002 };
                    T.shm->policy_status[0] = rm_control(H.fd, H.client, H.client, 0xd04, &p, sizeof p);
                    p = { H.client, 4, 0x40002 };
                    T.shm->policy_status[1] = rm_control(H.fd, H.client, H.client, 0xd04, &p, sizeof p);
                    p = { T.shm->agent_client, 1, 0x40003 };
                    T.shm->policy_status[2] = rm_control(H.fd, H.client, H.client, 0xd04, &p, sizeof p);
                    T.shm->app_client = H.client;
                    T.shm->device = H.device;
                    T.shm->subdev = H.subdev;
                    T.shm->app_session = H.session;
                    T.shm->ncompute = H.ncompute < L3_TRAP_MAX_KGO ? H.ncompute : L3_TRAP_MAX_KGO;
                    for (uint32_t i = 0; i < T.shm->ncompute; i++) T.shm->computes[i] = H.computes[i];
                    __sync_synchronize();
                    T.shm->magic = L3_TRAP_SHM_MAGIC;
                }
            }
        }
        T.patched = ProbeAndPatch();
        if (T.shm) T.shm->trap_patch_ok = T.patched ? 1 : 0;
    }
    base_va_  = T.live_base;
    live_va_  = T.handler_va;
    cave_va_  = T.cave_va;
    tramp_va_ = T.cave_va ? (T.cave_va + 0x80) : 0;
    patched_  = T.patched;
    ready_    = T.patched;
    return ready_;
}

void TrapSm120Context::Arm(bool armed)
{
    if (flag_host_) *flag_host_ = armed ? 1ULL : 0ULL;
    if (!T.patched) return;
    if (getenv("XSCHED_SM120_TRAP_CAVE") == nullptr) return;
    if (getenv("XSCHED_SM120_TRAP_KEEP_RET") != nullptr) return;
    if (getenv("XSCHED_SM120_TRAP_NO_PATCH") != nullptr) return;
    if (armed) {
        if (getenv("XSCHED_SM120_TRAP_PRE_RETURN") == nullptr && !ArmAtTrigger()) {
            if (T.shm) T.shm->arm_count = (uint32_t)(T.shm->arm_count + 1);
            return;
        }
        if (getenv("XSCHED_SM120_TRAP_INSTR_WRITE") != nullptr) {
            cuXtraInstrMemcpyHtoD(T.cave_va, T.armed_stub, sizeof T.armed_stub, T.stream);
            cuXtraInvalInstrCache(T.ctx);
        } else if (getenv("XSCHED_SM120_TRAP_HOST_ARM") != nullptr) {
            cuXtraMemcpyHtoD(T.cave_va, T.armed_stub, sizeof T.armed_stub, T.stream);
            cuXtraInvalInstrCache(T.ctx);
        } else {
            LaunchWriter(true);
        }
        if (T.shm) T.shm->arm_count = (uint32_t)(T.shm->arm_count + 1);
    } else {
        if (getenv("XSCHED_SM120_TRAP_PRE_RETURN") == nullptr) return;
        if (getenv("XSCHED_SM120_TRAP_HOST_ARM") != nullptr) {
            cuXtraMemcpyHtoD(T.cave_va, T.ret_stub, sizeof T.ret_stub, T.stream);
            cuXtraInvalInstrCache(T.ctx);
        } else {
            LaunchWriter(false);
        }
    }
}

bool TrapSm120Context::Trigger()
{
    if (!T.patched) return false;
    if (getenv("XSCHED_SM120_TRAP_NO_TRIGGER")) return true;
    if (T.shm == nullptr || T.shm->agent_ready == 0) {
        return RmFastTrapPulse83de();
    }
    bool no_patch = getenv("XSCHED_SM120_TRAP_NO_PATCH") != nullptr;
    bool always = no_patch || getenv("XSCHED_SM120_TRAP_ALWAYS_ARMED") != nullptr;
    bool ret_only = getenv("XSCHED_SM120_TRAP_RETURN_ONLY") != nullptr;
    if (ret_only) always = true;
    if (!always) Arm(true);

    bool lazy = getenv("L3AGENT_EAGER") == nullptr;
    if (lazy && T.shm->session != 0) lazy = false;
    if (lazy) {
        uint32_t sg = T.shm->session_req + 1;
        T.shm->session_req = sg;
        for (int i = 0; i < 400 && T.shm->session_ready != sg; i++) usleep(5000);
        if (T.shm->session_ready != sg || T.shm->session_status != 0) {
            if (!always) Arm(false);
            return RmFastTrapPulse83de();
        }
    }

    uint32_t gen = T.shm->trigger_req + 1;
    T.shm->trigger_req = gen;
    for (int i = 0; i < 40 && T.shm->trigger_ack != gen; i++) usleep(1000);
    if (!always) usleep(20000);

    bool hold = getenv("L3AGENT_HOLD") != nullptr ||
                getenv("XSCHED_SM120_TRAP_KEEP_SESSION") != nullptr;
    if (lazy && !hold) {
        uint32_t fg = T.shm->session_free_req + 1;
        T.shm->session_free_req = fg;
        for (int i = 0; i < 20 && T.shm->session_free_ack != fg; i++) usleep(1000);
    }
    if (!always) Arm(false);
    return true;
}

bool TrapSm120Context::Resume()
{
    if (T.shm == nullptr || T.shm->agent_ready == 0) {
        Sm120RmCtxHandles h{};
        if (!GetRmHandlesByIndex(target_slot_, &h)) {
            GetLatestRmHandles(&h);
        }
        int fd = (h.ctl_fd >= 0) ? h.ctl_fd : H.fd;
        uint32_t hClient = h.hClient ? h.hClient : H.client;
        uint32_t hTsg = h.hTsg ? h.hTsg : H.tsg;
        uint32_t hSession = h.hDebugger ? h.hDebugger : H.session;
        uint32_t hDbgClient = h.hDebuggerClient ? h.hDebuggerClient : (H.dbg_client ? H.dbg_client : hClient);
        if (fd >= 0 && hTsg != 0) {
            struct { uint64_t timesliceUs; } ts{2000ULL};
            rm_control(fd, hClient, hTsg, 0xa06c0103u, &ts, sizeof ts);
        }
        if (fd >= 0 && hSession != 0) {
            uint32_t dummy = 0;
            return rm_control(fd, hDbgClient, hSession, 0x83de0318u, &dummy, sizeof dummy) == 0;
        }
        return true;
    }
    uint32_t gen = T.shm->resume_req + 1;
    T.shm->resume_req = gen;
    for (int i = 0; i < 400 && T.shm->resume_ack != gen; i++) usleep(5000);
    bool ok = T.shm->resume_ack == gen;
    if (ok && getenv("XSCHED_SM120_TRAP_FREE_AFTER_RESUME") != nullptr) {
        uint32_t fg = T.shm->session_free_req + 1;
        T.shm->session_free_req = fg;
        for (int i = 0; i < 100 && T.shm->session_free_ack != fg; i++) usleep(1000);
    }
    return ok;
}

// ------------------------------------------------------------ Fine-Grained 3-Granularity Trap Engine
namespace {
struct FineGrainedState {
    bool installed = false;
    bool callable = false;
    CUcontext ctx = nullptr;
    uint64_t handler_va = 0;
    uint64_t ext_va = 0;
    uint64_t ext2_va = 0;
    uint64_t ext3_va = 0;
    uint64_t exit_stub_va = 0;
    uint64_t pause_tramp_va = 0;
    uint64_t orig_390[2] = {};
    uint64_t bound_func_pc = 0;
    uint64_t patched_func_pc = 0;
    uint64_t func_entry_backup[4] = {};
    bool func_entry_patched = false;
    std::vector<uint64_t> ext_backup;
    std::vector<uint64_t> ext2_backup;
    std::vector<uint64_t> ext3_backup;
    std::vector<uint64_t> exit_backup;
    std::vector<uint64_t> tramp_backup;
    volatile uint32_t *bar0 = nullptr;
    int bar0_fd = -1;
    CUresult (*get_func_va)(CUcontext, void *, CUdeviceptr *) = nullptr;
} FG;

constexpr uint32_t kCheckpointCommand = 4;
constexpr int64_t kCheckpointKernelTag = INT64_MIN;
constexpr uint32_t kTrapFrameEvictCommand = 5;
constexpr uint32_t kTrapFrameRestoreCommand = 6;
constexpr uint32_t kTrapFrameDrainCommand = 7;
constexpr int64_t kTrapFrameEvictKernelTag = INT64_MIN + 0x100000000LL;
constexpr uint32_t kTrapFrameRestoreMagic = 0x52455354u; // 'REST'
constexpr uint32_t kTrapFrameEvictGuard   = 0x45564354u; // 'EVCT'
std::mutex fine_grained_mu;

bool FineGrainedContextIsCurrent()
{
    CUcontext current = nullptr;
    return FG.installed && Driver::CtxGetCurrent(&current) == CUDA_SUCCESS && current == FG.ctx;
}

bool WriteFineGrainedTrapCode(CUdeviceptr addr, const void *data, size_t bytes)
{
    return cuXtraMemcpyHtoD(addr, data, bytes, 0) == bytes;
}

bool RestoreFineGrainedTrapCode()
{
    FG.callable = false;
    if (FG.func_entry_patched && FG.patched_func_pc) {
        WriteFineGrainedTrapCode(FG.patched_func_pc, FG.func_entry_backup, sizeof(FG.func_entry_backup));
        FG.func_entry_patched = false;
        FG.patched_func_pc = 0;
    }
    // Remove the entry hook before replacing the code it can reach.
    if (!WriteFineGrainedTrapCode(FG.handler_va + 0x390, FG.orig_390, sizeof(FG.orig_390))) {
        return false;
    }
    cuXtraInvalInstrCache(FG.ctx);
    const bool ok = WriteFineGrainedTrapCode(FG.ext_va, FG.ext_backup.data(), FG.ext_backup.size() * 8) &&
                    (!FG.ext2_va || FG.ext2_backup.empty() ||
                     WriteFineGrainedTrapCode(FG.ext2_va, FG.ext2_backup.data(), FG.ext2_backup.size() * 8)) &&
                    (!FG.ext3_va || FG.ext3_backup.empty() ||
                     WriteFineGrainedTrapCode(FG.ext3_va, FG.ext3_backup.data(), FG.ext3_backup.size() * 8)) &&
                    WriteFineGrainedTrapCode(FG.exit_stub_va, FG.exit_backup.data(), FG.exit_backup.size() * 8) &&
                    WriteFineGrainedTrapCode(FG.pause_tramp_va, FG.tramp_backup.data(), FG.tramp_backup.size() * 8);
    FG.installed = !ok;
    return ok;
}

struct FgInstr { uint64_t w0, w1; };

static inline uint64_t fg_ctl() { return ((uint64_t(0xfda00u) << 32) | 0x0ULL); }
static inline FgInstr fg_nop() { return {0x7918ULL, 0x0ULL | fg_ctl()}; }
static inline FgInstr fg_mov_imm_w01(int rd, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)rd << 16) | 0x7802ULL, ((uint64_t(0x3fda00u) << 32) | 0xf00ULL)};
}
static inline FgInstr fg_mov_reg(int rd, int ra) {
    return {((uint64_t)ra << 24) | ((uint64_t)rd << 16) | 0x7202ULL, ((uint64_t(0x1fda00u) << 32) | 0xf00ULL)};
}
static inline FgInstr fg_iadd_imm(int rd, int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | ((uint64_t)rd << 16) | 0x7835ULL, 0x078e0000ULL | fg_ctl()};
}
static inline FgInstr fg_iadd3_imm(int rd, int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | ((uint64_t)rd << 16) | 0x7810ULL, 0x07ffe0ffULL | fg_ctl()};
}
static inline FgInstr fg_iadd3_imm_req10(int rd, int ra, uint32_t imm) {
    FgInstr i = fg_iadd3_imm(rd, ra, imm);
    i.w1 |= ((uint64_t(0x300000u) << 32) | 0x0ULL);
    return i;
}
static inline FgInstr fg_ret_r12() { return {0xc087950ULL, ((uint64_t(0x3fc000u) << 32) | 0x3a00000ULL)}; }
static inline FgInstr fg_exit() { return {0x794dULL, ((uint64_t(0xfea00u) << 32) | 0x3800000ULL)}; }
static inline FgInstr fg_ldc_u32(int rd, uint32_t off) {
    uint64_t encoded_off = (uint64_t)(off >> 2) << 8;
    return {(encoded_off << 32) | 0xff000000ULL | ((uint64_t)rd << 16) | 0x7b82ULL, (0x000e34ULL << 40) | 0x800ULL};
}
static inline FgInstr fg_ld_e_strong_gpu_r0_r4() {
    return {(4ULL << 24) | 0x7980ULL, (0x001e34ULL << 40) | 0x10e900ULL};
}
static inline FgInstr fg_ld_e_strong_gpu_r0_r6() {
    return {(6ULL << 24) | 0x7980ULL, (0x001e34ULL << 40) | 0x10e900ULL};
}
static inline FgInstr fg_ldl_32_r0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0xff007983ULL, ((uint64_t(0x322400u) << 32) | 0x100800ULL)};
}
static inline FgInstr fg_stl_32_r0_p0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0xff000387ULL, ((uint64_t(0x33e400u) << 32) | 0x100800ULL)};
}
static inline FgInstr fg_nanosleep_imm(uint32_t imm_ns) {
    return {((uint64_t)imm_ns << 32) | 0x795dULL, ((uint64_t(0xfea00u) << 32) | 0x3800000ULL)};
}
static inline FgInstr fg_ldl_64_r0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0xff007983ULL, ((uint64_t(0x322400u) << 32) | 0x100a00ULL)};
}
static inline FgInstr fg_stl_64_r0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0xff007387ULL, ((uint64_t(0x33e400u) << 32) | 0x100a00ULL)};
}
static inline FgInstr fg_stl_32_r0(uint32_t lmem_off) {
    FgInstr i = fg_stl_64_r0(lmem_off);
    i.w1 ^= 0x200ULL;
    return i;
}
static inline FgInstr fg_ldl_128_r8(uint32_t lmem_off) {
    FgInstr i = fg_ldl_64_r0(lmem_off);
    i.w0 |= (8ULL << 16);
    i.w1 = (i.w1 | (0x10ULL << 40)) ^ 0x600ULL;
    return i;
}
static inline FgInstr fg_stl_128_r8(uint32_t lmem_off) {
    FgInstr i = fg_stl_64_r0(lmem_off);
    i.w0 |= (8ULL << 32);
    i.w1 = (i.w1 | (0x10ULL << 40)) ^ 0x600ULL;
    return i;
}
static inline FgInstr fg_iadd_imm_req10(int rd, int ra, uint32_t imm) {
    FgInstr i = fg_iadd_imm(rd, ra, imm);
    i.w1 |= ((uint64_t(0x300000u) << 32) | 0x0ULL);
    return i;
}
static inline FgInstr fg_imad_r0_r3_imm_r0(uint32_t imm) {
    return {((uint64_t)imm << 32) | (3ULL << 24) | 0x7824ULL, ((uint64_t(0xfda00u) << 32) | 0x78e0200ULL)};
}
static inline FgInstr fg_imad_shl_u32_rz(int rd, int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | ((uint64_t)rd << 16) | 0x7824ULL, ((uint64_t(0x3fda00u) << 32) | 0x78e02ffULL)};
}
static inline FgInstr fg_imad_wide_u32(int rd, int ra, uint32_t imm, int rc) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | ((uint64_t)rd << 16) | 0x7825ULL, ((uint64_t(0x3fda00u) << 32) | 0x78e0000ULL) | (uint64_t)rc};
}
static inline FgInstr fg_ld_e_64_r8_r4(uint32_t off) {
    FgInstr i = fg_ld_e_strong_gpu_r0_r4();
    i.w0 |= ((uint64_t)off << 32) | (8ULL << 16);
    i.w1 ^= 0xe200ULL;
    return i;
}
static inline FgInstr fg_ld_e_strong_gpu_r0_r6_off(uint32_t off) {
    FgInstr i = fg_ld_e_strong_gpu_r0_r6();
    i.w0 |= ((uint64_t)off << 32);
    return i;
}
static inline FgInstr fg_ld_e_strong_gpu_r0_r8_off(uint32_t off) {
    return {((uint64_t)off << 32) | (8ULL << 24) | 0x7980ULL, (0x003e34ULL << 40) | 0x10e900ULL};
}
static inline FgInstr fg_st_e_r6_off_r0(uint32_t off) {
    return {((uint64_t)off << 32) | (6ULL << 24) | 0x7385ULL, (0x0033f4ULL << 40) | 0x10e900ULL};
}
static inline FgInstr fg_st_e_r6_off_r1(uint32_t off) {
    FgInstr i = fg_st_e_r6_off_r0(off);
    i.w1 |= 1ULL;
    return i;
}
static inline FgInstr fg_st_e_128_r6_r8(uint32_t off) {
    FgInstr i = fg_st_e_r6_off_r0(off);
    i.w1 ^= 0x408ULL;
    return i;
}
static inline FgInstr fg_ld_e_128_r8_r6(uint32_t off) {
    FgInstr i = fg_ld_e_strong_gpu_r0_r6_off(off);
    i.w0 |= (8ULL << 16);
    i.w1 = (i.w1 | (0x002000ULL << 40)) ^ 0x400ULL;
    return i;
}
static inline FgInstr fg_st_e_r8_off_r0(uint32_t off) {
    return {((uint64_t)off << 32) | (8ULL << 24) | 0x7385ULL, (0x0033f4ULL << 40) | 0x10e900ULL};
}
static inline FgInstr fg_st_e_r8_off_r1(uint32_t off) {
    return {((uint64_t)off << 32) | (8ULL << 24) | 0x7385ULL, (0x0033f4ULL << 40) | 0x100901ULL};
}
static inline FgInstr fg_st_e_64_r6_r14(uint32_t off) {
    return {((uint64_t)off << 32) | (6ULL << 24) | 0x7385ULL, (0x0033f4ULL << 40) | 0x10eb0eULL};
}
static inline FgInstr fg_ld_e_64_r14_r6(uint32_t off) {
    return {((uint64_t)off << 32) | (6ULL << 24) | (14ULL << 16) | 0x7980ULL, (0x003e34ULL << 40) | 0x10eb00ULL};
}
static inline FgInstr fg_ldcu_64_ur_358(int urd) {
    return {(0x6b00ULL << 32) | 0xff0077acULL | ((uint64_t)urd << 16), (0x003e34ULL << 40) | 0x08000a00ULL};
}
static inline FgInstr fg_mov_r_ur(int rd, int ura) {
    return {((uint64_t)ura << 32) | ((uint64_t)rd << 16) | 0x7c02ULL, (0x003fdaULL << 40) | 0x08000f00ULL};
}
static inline FgInstr fg_r2ur(int urd, int ra) {
    return {((uint64_t)ra << 24) | ((uint64_t)urd << 16) | 0x72caULL, (0x003fdaULL << 40) | 0x000e0000ULL};
}
static inline FgInstr fg_uisetp_up0_true() {
    return {((uint64_t(0xffu) << 32) | 0x3f00728cULL), (0x000fdaULL << 40) | 0x0bf02270ULL};
}
static inline FgInstr fg_lds_r0_r4() {
    return {(0x400ULL << 40) | (4ULL << 24) | 0x7984ULL, (0x003e34ULL << 40) | 0x800ULL};
}
static inline FgInstr fg_sts_r4_r0() {
    return {(0x400ULL << 40) | (4ULL << 24) | 0x7388ULL, (0x0033f4ULL << 40) | 0x800ULL};
}
static inline FgInstr fg_stl_64_r0_p0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0xff000387ULL, ((uint64_t(0x33e400u) << 32) | 0x100a00ULL)};
}
static inline FgInstr fg_isetp_ne_rz(int ra) {
    return {((uint64_t(0xffu) << 32) | 0x720cULL) | ((uint64_t)ra << 24), ((uint64_t(0x1fda00u) << 32) | 0x3f05270ULL)};
}
static inline FgInstr fg_isetp_ne_imm(int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL, ((uint64_t(0x1fda00u) << 32) | 0x3f05270ULL)};
}
static inline FgInstr fg_isetp_ne_imm_and_p0(int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL, ((uint64_t(0x1fda00u) << 32) | 0x705270ULL)};
}
static inline FgInstr fg_isetp_eq_imm(int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL, ((uint64_t(0x1fda00u) << 32) | 0x3f02270ULL)};
}
static inline FgInstr fg_isetp_ge_u32_imm(int ra, uint32_t imm) {
    FgInstr i = fg_isetp_eq_imm(ra, imm);
    i.w1 ^= 0x4200ULL;
    return i;
}
static inline FgInstr fg_isetp_eq_imm_or_p0(int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL, ((uint64_t(0x1fda00u) << 32) | 0x702470ULL)};
}
static inline uint64_t fg_encode_bra_w0(int64_t n_instrs, uint32_t low16) {
    int64_t off = n_instrs * 4;
    uint64_t lo8 = (uint64_t)(off & 0xff);
    uint64_t hi = (uint64_t)((off >> 8) & 0x3fffffffLL);
    return (hi << 34) | (lo8 << 16) | low16;
}
static inline uint64_t fg_encode_bra_w1(int64_t n_instrs) {
    int64_t off = n_instrs * 4;
    uint64_t hi18 = (uint64_t)((off >> 38) & 0x3ffffLL);
    return ((uint64_t(0xfea00u) << 32) | 0x3800000ULL | hi18);
}
static inline FgInstr fg_bra_p0(int64_t n_instrs) {
    return {fg_encode_bra_w0(n_instrs, 0x0947), fg_encode_bra_w1(n_instrs)};
}
static inline FgInstr fg_bra_not_p0(int64_t n_instrs) {
    return {fg_encode_bra_w0(n_instrs, 0x8947), fg_encode_bra_w1(n_instrs)};
}
static inline FgInstr fg_bra_rel(int64_t n_instrs) {
    return {fg_encode_bra_w0(n_instrs, 0x7947), fg_encode_bra_w1(n_instrs)};
}
static inline FgInstr fg_bra_to(uint64_t from_va, uint64_t to_va) {
    int64_t n_instrs = (int64_t(to_va) - int64_t(from_va + 16)) >> 4;
    return fg_bra_rel(n_instrs);
}
static inline FgInstr fg_bra_p0_to(uint64_t from_va, uint64_t to_va) {
    int64_t n_instrs = (int64_t(to_va) - int64_t(from_va + 16)) >> 4;
    return fg_bra_p0(n_instrs);
}
static inline FgInstr fg_bra_not_p0_to(uint64_t from_va, uint64_t to_va) {
    int64_t n_instrs = (int64_t(to_va) - int64_t(from_va + 16)) >> 4;
    return fg_bra_not_p0(n_instrs);
}
static inline FgInstr fg_s2r_tid_x(int rd) {
    return {((uint64_t)rd << 16) | 0x7919ULL, (0x000e34ULL << 40) | 0x2100ULL};
}
static inline FgInstr fg_s2r_tid_y(int rd) {
    return {((uint64_t)rd << 16) | 0x7919ULL, (0x000e74ULL << 40) | 0x2200ULL};
}
static inline FgInstr fg_s2r_ctaid_x_w1(int rd) {
    return {((uint64_t)rd << 16) | 0x7919ULL, (0x000e74ULL << 40) | 0x2500ULL};
}
static inline FgInstr fg_s2r_laneid(int rd) {
    return {((uint64_t)rd << 16) | 0x7919ULL, (0x000e74ULL << 40)};
}
static inline FgInstr fg_ldc_u32_w2(int rd, uint32_t off) {
    FgInstr i = fg_ldc_u32(rd, off);
    i.w1 = (0x000eb4ULL << 40) | 0x800ULL;
    return i;
}
static inline FgInstr fg_ldc_u32_w3(int rd, uint32_t off) {
    FgInstr i = fg_ldc_u32(rd, off);
    i.w1 = (0x000ef4ULL << 40) | 0x800ULL;
    return i;
}
static inline FgInstr fg_ldc_u32_req04(int rd, uint32_t off) {
    FgInstr i = fg_ldc_u32(rd, off);
    i.w1 |= (0x004000ULL << 40);
    return i;
}
static inline FgInstr fg_ldc_64_r8_178() {
    return {(0x5e00ff08ULL << 16) | 0x7b82ULL, (0x000e34ULL << 40) | 0xa00ULL};
}
static inline FgInstr fg_imad_regs(int rd, int ra, int rb, int rc, uint64_t req_mask) {
    return {((uint64_t)rb << 32) | ((uint64_t)ra << 24) | ((uint64_t)rd << 16) | 0x7224ULL,
            ((req_mask << 12 | 0xfdaULL) << 40) | 0x078e0200ULL | (uint64_t)rc};
}
static inline FgInstr fg_imad_rz(int rd, int ra, int rb) {
    return fg_imad_regs(rd, ra, rb, 0xff, 0x0c);
}
static inline FgInstr fg_imad_iadd(int rd, int ra, int rc) {
    return {(1ULL << 32) | ((uint64_t)ra << 24) | ((uint64_t)rd << 16) | 0x7824ULL,
            (0x000fdaULL << 40) | 0x078e0200ULL | (uint64_t)rc};
}
static inline FgInstr fg_iadd3_neg4(int rd, int ra) {
    return {(0xfffffffcULL << 32) | ((uint64_t)ra << 24) | ((uint64_t)rd << 16) | 0x7810ULL,
            (0x000fdaULL << 40) | 0x07ffe0ffULL};
}
static inline FgInstr fg_iadd3_imm_req01(int rd, int ra, uint32_t imm) {
    FgInstr i = fg_iadd3_imm(rd, ra, imm);
    i.w1 |= (0x001000ULL << 40);
    return i;
}
static inline FgInstr fg_shf_r_u32(int rd, int rc, uint32_t shift) {
    return {((uint64_t)shift << 32) | (0xffULL << 24) | ((uint64_t)rd << 16) | 0x7819ULL,
            (0x000fdaULL << 40) | 0x00011600ULL | (uint64_t)rc};
}
static inline FgInstr fg_st_e_64_r6_rs(uint32_t off, int rs) {
    FgInstr i = fg_st_e_64_r6_r14(off);
    i.w1 = (i.w1 & ~0xffULL) | (uint64_t)(rs & 0xff);
    return i;
}
static inline FgInstr fg_ld_e_64_rd_r6(int rd, uint32_t off) {
    FgInstr i = fg_ld_e_64_r14_r6(off);
    i.w0 = (i.w0 & ~(0xffULL << 16)) | ((uint64_t)(rd & 0xff) << 16);
    return i;
}
static inline FgInstr fg_st_e_32_r6_rs(uint32_t off, int rs) {
    FgInstr i = fg_st_e_r6_off_r0(off);
    i.w1 |= (uint64_t)(rs & 0xff);
    return i;
}
static inline FgInstr fg_ld_e_32_rd_r6(int rd, uint32_t off) {
    FgInstr i = fg_ld_e_strong_gpu_r0_r6_off(off);
    i.w0 |= ((uint64_t)(rd & 0xff) << 16);
    i.w1 |= (0x002000ULL << 40);
    return i;
}
static inline FgInstr fg_st_e_128_r6_rs(uint32_t off, int rs) {
    FgInstr i = fg_st_e_r6_off_r0(off);
    i.w1 = (i.w1 ^ 0x400ULL) | (uint64_t)(rs & 0xff);
    return i;
}
static inline FgInstr fg_ld_e_128_rd_r6(int rd, uint32_t off) {
    FgInstr i = fg_ld_e_strong_gpu_r0_r6_off(off);
    i.w0 |= ((uint64_t)(rd & 0xff) << 16);
    i.w1 = (i.w1 | (0x002000ULL << 40)) ^ 0x400ULL;
    return i;
}
static inline FgInstr fg_r2ur_req04(int urd, int ra) {
    FgInstr i = fg_r2ur(urd, ra);
    i.w1 |= (0x004000ULL << 40);
    return i;
}
static inline FgInstr fg_st_e_r8_off_r5(uint32_t off) {
    FgInstr i = fg_st_e_r8_off_r1(off);
    i.w1 = (i.w1 & ~0xffULL) | 5ULL;
    return i;
}
static inline FgInstr fg_isetp_eq_imm_req02(int ra, uint32_t imm) {
    FgInstr i = fg_isetp_eq_imm(ra, imm);
    i.w1 |= (0x002000ULL << 40);
    return i;
}
static inline FgInstr fg_redg_add_p0_r8_4_r0() {
    return {(4ULL << 40) | (8ULL << 24) | 0x098eULL, (0x0033daULL << 40) | 0x0010e1ffULL};
}
static inline FgInstr fg_isetp_ge_u32_r0_r5() {
    return {(5ULL << 32) | 0x720cULL, (0x001fdaULL << 40) | 0x03f06070ULL};
}
static inline FgInstr fg_bar_sync_defer_p0() {
    return {0x0b1dULL, (0x000fe2ULL << 40) | 0x00010000ULL};
}
static inline FgInstr fg_b2r_warp_r8() {
    return {0x8731cULL, (0x003224ULL << 40) | 0x8000ULL};
}

static bool EnsureBar0Mapped() {
    if (FG.bar0 != nullptr) return true;
    const char *res_path = getenv("XSCHED_SM120_BAR0_RESOURCE");
    if (!res_path) res_path = "/sys/bus/pci/devices/0000:01:00.0/resource0";
    int fd = open(res_path, O_RDWR | O_SYNC);
    if (fd >= 0) {
        void *p = mmap(nullptr, 0x1000000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p != MAP_FAILED) {
            FG.bar0 = (volatile uint32_t *)p;
            FG.bar0_fd = fd;
            return true;
        }
        close(fd);
    }
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd >= 0) {
        void *p = mmap(nullptr, 0x1000000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0xb2000000ULL);
        if (p != MAP_FAILED) {
            FG.bar0 = (volatile uint32_t *)p;
            FG.bar0_fd = fd;
            return true;
        }
        close(fd);
    }
    return false;
}

void PulseFineGrainedTrap()
{
    FG.bar0[0x424684 >> 2] = 0x80001c01u;
    (void)FG.bar0[0x424684 >> 2];
    FG.bar0[0x424684 >> 2] = 0x00001c01u;
}

static void BuildFineGrainedExt3Code(int num_regs, uint64_t ext_va, uint64_t ext2_va,
                                     uint64_t ext3_va, int notify_idx,
                                     std::vector<FgInstr> &ext3_code)
{
    constexpr int kExt3RestoreIdx = 76;
    ext3_code.clear();
    ext3_code.reserve(140);
    // Section 0 (idx 0 .. 11): clean_exit_redirect (neutralize SIMT divergence before exit_stub)
    ext3_code.push_back(fg_mov_imm_w01(8, 0xffffffffu));                       // idx 0: R8 = 0xffffffff
    ext3_code.push_back(fg_mov_imm_w01(9, 0));                                 // idx 1: R9 = 0
    ext3_code.push_back(fg_mov_imm_w01(10, 0));                                // idx 2: R10 = 0
    ext3_code.push_back(fg_mov_imm_w01(11, 0));                                // idx 3: R11 = 0
    ext3_code.push_back(fg_stl_128_r8(0xffff90));                              // idx 4: THREAD_STATE_ENUM.0..3 = {0xffffffff,0,0,0}
    ext3_code.push_back(fg_mov_imm_w01(8, 0));                                 // idx 5: R8 = 0
    ext3_code.push_back(fg_stl_128_r8(0xffffb0));                              // idx 6: Rpc.LO/HI, OPT_STACK, B0 = {0,0,0,0}
    ext3_code.push_back(fg_mov_imm_w01(0, 0xffffffffu));                       // idx 7: R0 = 0xffffffff
    ext3_code.push_back(fg_stl_32_r0(0xfffffc));                               // idx 8: TRAP_RETURN_MASK = 0xffffffff
    ext3_code.push_back(fg_mov_imm_w01(0, 0));                                 // idx 9: R0 = 0
    ext3_code.push_back(fg_stl_32_r0(0xffffa0));                               // idx 10: THREAD_STATE_ENUM.4 = 0
    ext3_code.push_back(fg_bra_to(ext3_va + 11 * 16, ext_va + 47 * 16));       // idx 11: BRA exit_redirect (ext idx 47)

    // Support all architectural registers up to num_regs = 255 (user R0..R252, trap-reserved R253..R254).
    // Exact branchless generation for R14 .. R[max_reg] with zero gaps for any register count 1..255.
    const int max_reg = std::min(num_regs - 3, 252);
    // Section 1 (idx 12 .. 75): Save R14 .. R[max_reg] using 16B-aligned ST.E.128 + exact tail
    int r = 14;
    if (r <= max_reg) {
        if (r + 1 <= max_reg) {
            ext3_code.push_back(fg_st_e_64_r6_rs(0x40 + 4 * r, r));
            r += 2;
        } else {
            ext3_code.push_back(fg_st_e_32_r6_rs(0x40 + 4 * r, r));
            r += 1;
        }
    }
    while (r + 3 <= max_reg) {
        ext3_code.push_back(fg_st_e_128_r6_rs(0x40 + 4 * r, r));
        r += 4;
    }
    if (r + 1 <= max_reg) {
        ext3_code.push_back(fg_st_e_64_r6_rs(0x40 + 4 * r, r));
        r += 2;
    }
    if (r <= max_reg) {
        ext3_code.push_back(fg_st_e_32_r6_rs(0x40 + 4 * r, r));
        r += 1;
    }
    ext3_code.push_back(fg_bra_to(ext3_va + ext3_code.size() * 16, ext2_va + notify_idx * 16));
    while (int(ext3_code.size()) < kExt3RestoreIdx) {
        ext3_code.push_back(fg_nop());
    }

    // Section 2 (idx 76 .. 139): Restore R14 .. R[max_reg] using 16B-aligned LD.E.128 + exact tail
    r = 14;
    if (r <= max_reg) {
        if (r + 1 <= max_reg) {
            ext3_code.push_back(fg_ld_e_64_rd_r6(r, 0x40 + 4 * r));
            r += 2;
        } else {
            ext3_code.push_back(fg_ld_e_32_rd_r6(r, 0x40 + 4 * r));
            r += 1;
        }
    }
    while (r + 3 <= max_reg) {
        ext3_code.push_back(fg_ld_e_128_rd_r6(r, 0x40 + 4 * r));
        r += 4;
    }
    if (r + 1 <= max_reg) {
        ext3_code.push_back(fg_ld_e_64_rd_r6(r, 0x40 + 4 * r));
        r += 2;
    }
    if (r <= max_reg) {
        ext3_code.push_back(fg_ld_e_32_rd_r6(r, 0x40 + 4 * r));
        r += 1;
    }
    ext3_code.push_back(fg_bra_to(ext3_va + ext3_code.size() * 16, ext2_va + notify_idx * 16));
    while (ext3_code.size() < 140) {
        ext3_code.push_back(fg_nop());
    }
}

void BindFineGrainedKernel(CUfunction func, uint64_t preempt_buf_va,
                           int64_t kernel_idx, uint64_t guardian_va)
{
    struct {
        uint64_t preempt_buf;
        uint64_t guardian;
        int64_t kernel_idx;
        uint32_t sched_enabled;
    } __attribute__((packed)) params{preempt_buf_va, guardian_va, kernel_idx, 1u};
    cuXtraSetDebuggerParams(func, &params, sizeof(params));
}
} // namespace

bool TrapSm120Context::InstallFineGrainedTrapHandler(CUcontext ctx)
{
    std::lock_guard<std::mutex> lock(fine_grained_mu);
    CUcontext current = nullptr;
    if (!ctx || Driver::CtxGetCurrent(&current) != CUDA_SUCCESS || current != ctx) return false;
    if (FG.installed) return FG.callable && FG.ctx == ctx;
    EnsureBar0Mapped();

    const void *tt_raw = nullptr, *mt_raw = nullptr;
    CUuuid te = UuidOf("cc529e945c4e9d46837c9625868334e4");
    CUuuid me = UuidOf("bfdb432dbf3c5a4a945eb34029e81e75");
    if (Driver::GetExportTable(&tt_raw, &te) != CUDA_SUCCESS ||
        Driver::GetExportTable(&mt_raw, &me) != CUDA_SUCCESS ||
        tt_raw == nullptr || mt_raw == nullptr) {
        return false;
    }
    const void **tt = (const void **)tt_raw, **mt = (const void **)mt_raw;
    struct { uint64_t flags, ssize; void *handle; uint64_t stub, hsize, rsv; } ti;
    memset(&ti, 0, sizeof ti);
    if (((CUresult (*)(CUcontext, void *))tt[20])(ctx, &ti) != CUDA_SUCCESS || !ti.handle) {
        return false;
    }
    CUdeviceptr tools_pc = 0; uint64_t tools_sz = 0;
    if (((CUresult (*)(CUcontext, void *, CUdeviceptr *))mt[24])(ctx, ti.handle, &tools_pc) != CUDA_SUCCESS ||
        ((CUresult (*)(void *, uint64_t *))mt[25])(ti.handle, &tools_sz) != CUDA_SUCCESS || tools_sz < 0x1300) {
        return false;
    }
    std::vector<uint8_t> tools_img(tools_sz);
    if (cuXtraMemcpyDtoH(tools_img.data(), tools_pc, tools_sz, 0) != tools_sz) return false;

    uint64_t trt = (uint64_t)Imm32At(&tools_img[0x130]) | ((uint64_t)Imm32At(&tools_img[0x140]) << 32);
    uint64_t base = trt - 0xc38300ULL;
    uint64_t handler_va = base + 0x15f0f00ULL;

    std::vector<uint8_t> handler_img(0x1200);
    if (cuXtraMemcpyDtoH(handler_img.data(), handler_va, handler_img.size(), 0) != handler_img.size()) return false;

    // The instruction encodings and hook layout below target the 595.99.02
    // GB206 handler. Do not chain over an existing hook or another layout.
    FgInstr hook_site, previous_call;
    memcpy(&hook_site, &handler_img[0x390], sizeof(hook_site));
    memcpy(&previous_call, &handler_img[0x380], sizeof(previous_call));
    if (hook_site.w0 != ((uint64_t(0xffu) << 32) | 0x37202ULL) || hook_site.w1 != ((uint64_t(0x3ff000u) << 32) | 0xf00ULL) ||
        uint32_t(previous_call.w0) != 0x00407943u) {
        return false;
    }

    FgInstr test_r2_ctxsw;
    memcpy(&test_r2_ctxsw.w0, &handler_img[0x3b0], 8);
    memcpy(&test_r2_ctxsw.w1, &handler_img[0x3b8], 8);
    test_r2_ctxsw.w0 = (test_r2_ctxsw.w0 & 0xffffffffULL) | (0x11ULL << 32);
    test_r2_ctxsw.w1 = (test_r2_ctxsw.w1 & ~((uint64_t(0xffe00u) << 32) | 0x0ULL)) | ((uint64_t(0xfda00u) << 32) | 0x0ULL);

    uint64_t lop_w0, lop_w1;
    memcpy(&lop_w0, &handler_img[0x220], 8);
    memcpy(&lop_w1, &handler_img[0x228], 8);
    if (lop_w0 != ((uint64_t(0x4u) << 32) | 0x2027812ULL) || lop_w1 != ((uint64_t(0x3fec00u) << 32) | 0x78ec0ffULL)) {
        return false;
    }
    FgInstr clear_r2_bit8 = {(lop_w0 & 0xffffffffULL) | (0xfffffff7ULL << 32), lop_w1};

    FgInstr stl_64_rpc;
    memcpy(&stl_64_rpc.w0, &handler_img[0x110], 8);
    memcpy(&stl_64_rpc.w1, &handler_img[0x118], 8);

    FgInstr b2r_warp_r8 = fg_b2r_warp_r8();

    uint64_t ext_va = tools_pc + 0x980;
    uint64_t ext2_va = handler_va + 0x860;
    uint64_t ext3_va = handler_va + 0x2520;
    uint64_t exit_stub_va = tools_pc + 0x12b0;
    uint64_t pause_tramp_va = tools_pc + 0x12d0;
    uint32_t tramp0_lo = (uint32_t)pause_tramp_va;
    uint32_t tramp1_lo = (uint32_t)(pause_tramp_va + 0x10);
    uint32_t tramp2_lo = (uint32_t)(pause_tramp_va + 0x20);
    uint32_t tramp_hi  = (uint32_t)(pause_tramp_va >> 32);
    constexpr int kRestoreIdx = 0;
    constexpr int kNotifyIdx = 41;
    constexpr int kNotifyBase = 41;
    constexpr int kExt3CleanExitIdx = 0;
    constexpr int kExt3SaveIdx = 12;
    constexpr int kExt3RestoreIdx = 76;

    std::vector<FgInstr> ext_code = {
        test_r2_ctxsw,                                                     // (R2 & 0x11) != 0?
        fg_bra_p0(56),                                                     // @P0 BRA done
        {((uint64_t(0x2u) << 32) | 0xa7202ULL), ((uint64_t(0xfda00u) << 32) | 0xf00ULL)},                    // MOV R10, R2 (unmodified reason)
        clear_r2_bit8,                                                     // R2 &= ~0x8
        // A broadcast hits unrelated kernels in this context as well. They
        // must return without entering the driver's debugger pause routine.
        fg_ldc_u32(4, 0x170),                                              // R4 = preempt_buf lo
        fg_ldc_u32(5, 0x174),                                              // R5 = preempt_buf hi
        fg_isetp_ne_rz(4),                                                 // P0 = (R4 != 0)
        fg_bra_not_p0(50),                                                 // @!P0 BRA done
        {0x7919ULL, ((uint64_t(0xe6200u) << 32) | 0x2600ULL)},                    // S2R R0, SR_CTAID.Y
        {((uint64_t(0xdd00u) << 32) | 0xff037b82ULL), ((uint64_t(0xe6200u) << 32) | 0x800ULL)},                    // LDC R3, c[0x0][0x374]
        {0x67919ULL, ((uint64_t(0x1e6200u) << 32) | 0x2500ULL)},                    // S2R R6, SR_CTAID.X
        {0x87919ULL, ((uint64_t(0xe2c00u) << 32) | 0x2700ULL)},                    // S2R R8, SR_CTAID.Z
        {((uint64_t(0xdc00u) << 32) | 0xff077b82ULL), ((uint64_t(0xe2200u) << 32) | 0x800ULL)},                    // LDC R7, c[0x0][0x370]
        {((uint64_t(0x8u) << 32) | 0x3007224ULL), ((uint64_t(0x3fda00u) << 32) | 0x78e0200ULL)},                    // IMAD R0, R3, R8, R0
        {((uint64_t(0x7u) << 32) | 0x7224ULL), ((uint64_t(0x1fda00u) << 32) | 0x78e0206ULL)},                    // IMAD R0, R0, R7, R6
        {((uint64_t(0x4u) << 32) | 0x37810ULL), ((uint64_t(0xfda00u) << 32) | 0x7ffe000ULL)},                    // IADD3 R3, PT, PT, R0, 0x4, R0
        {((uint64_t(0x4u) << 32) | 0x3067825ULL), ((uint64_t(0xfda00u) << 32) | 0x78e0004ULL)},                    // IMAD.WIDE.U32 R6, R3, 0x4, R4
        fg_ld_e_strong_gpu_r0_r4(),                                        // LD.E.STRONG.GPU R0, [R4]
        fg_isetp_ne_imm(0, 3),                                             // P0 = (R0 != 3)
        fg_bra_p0(1),                                                      // @P0 BRA check_mode
        fg_ld_e_strong_gpu_r0_r6(),                                        // LD.E.STRONG.GPU R0, [R6]
        // check_mode (also the target of the mode-3 fast path).
        fg_isetp_ge_u32_imm(0, kCheckpointCommand),                        // P0 = (R0 >= 4) (Modes 4 & 5)
        fg_bra_p0(38),                                                     // @P0 BRA checkpoint_signal (idx 62)
        fg_isetp_ne_imm(0, 2),                                             // P0 = (R0 != 2)
        fg_bra_p0(15),                                                     // @P0 BRA check_exit
        // do_pause (100% divergence-free predicated redirect to pause_tramp_va!)
        fg_mov_imm_w01(0, 1),                                              // MOV R0, 1
        fg_isetp_ne_rz(0),                                                 // P0 = true across all 32 lanes
        fg_ldl_64_r0(0xffffa8),                                            // LDL.64 R0, [0xffffa8]
        fg_isetp_ne_imm_and_p0(0, tramp0_lo),                              // P0 = P0 && (R0 != tramp0_lo)
        fg_isetp_ne_imm_and_p0(0, tramp1_lo),                              // P0 = P0 && (R0 != tramp1_lo)
        fg_isetp_ne_imm_and_p0(0, tramp2_lo),                              // P0 = P0 && (R0 != tramp2_lo)
        fg_stl_64_r0_p0(0xffff88),                                         // @P0 STL.64 [0xffff88], R0
        fg_mov_imm_w01(0, tramp0_lo),                                      // MOV R0, tramp0_lo (NANOSLEEP 0x100; BRA)
        fg_mov_imm_w01(1, tramp_hi),                                       // MOV R1, tramp_hi
        fg_stl_64_r0_p0(0xffffa8),                                         // @P0 STL.64 [0xffffa8], R0
        fg_ldl_32_r0(0xfffffc),                                            // LDL.32 R0, [0xfffffc] (TRAP_RETURN_MASK)
        fg_isetp_ne_imm_and_p0(0, 0xffffffffu),                            // P0 = P0 && (TRAP_RETURN_MASK != 0xffffffff)
        fg_mov_imm_w01(0, tramp2_lo),                                      // MOV R0, tramp2_lo (BRA self for diverged sub-mask)
        fg_stl_32_r0_p0(0xffffa8),                                         // @P0 STL.32 [0xffffa8], R0
        fg_bra_rel(17),                                                    // BRA done_r1
        // check_exit
        fg_isetp_ne_imm(0, 1),                                             // P0 = (R0 != 1)
        fg_bra_p0(9),                                                      // @P0 BRA check_resume
        fg_mov_imm_w01(0, 1),                                              // MOV R0, 1
        {((uint64_t(0x4u) << 32) | 0x6007385ULL), ((uint64_t(0xfda00u) << 32) | 0x100900ULL)},                    // ST.E [R6+0x4], R0
        fg_ldc_u32(8, 0x180),                                              // LDC R8, c[0x0][0x180]
        fg_ldc_u32(9, 0x184),                                              // LDC R9, c[0x0][0x184]
        {((uint64_t(0x8u) << 32) | 0x4007385ULL), ((uint64_t(0x1fda00u) << 32) | 0x100b08ULL)},                    // ST.E.64 [R4+0x8], R8
        fg_mov_imm_w01(0, (uint32_t)exit_stub_va),                         // MOV R0, lo32(exit_stub_va)
        fg_mov_imm_w01(1, (uint32_t)(exit_stub_va >> 32)),                 // MOV R1, hi32(exit_stub_va)
        fg_stl_64_r0(0xffffa8),                                            // STL.64 [0xffffa8], R0
        fg_bra_rel(6),                                                     // BRA done_r1
        // check_resume
        fg_ldl_64_r0(0xffffa8),                                            // LDL.64 R0, [0xffffa8]
        fg_isetp_eq_imm(0, tramp0_lo),                                     // P0 = (R0 == tramp0_lo)
        fg_isetp_eq_imm_or_p0(0, tramp1_lo),                               // P0 = P0 || (R0 == tramp1_lo)
        fg_isetp_eq_imm_or_p0(0, tramp2_lo),                               // P0 = P0 || (R0 == tramp2_lo)
        fg_ldl_64_r0(0xffff88),                                            // LDL.64 R0, [0xffff88]
        fg_stl_64_r0_p0(0xffffa8),                                         // @P0 STL.64 [0xffffa8], R0
        // done_r1 / done (idx 58)
        fg_mov_imm_w01(1, 0xffff80),                                       // MOV R1, 0xffff80
        fg_mov_imm_w01(3, 0),                                              // MOV R3, 0
        fg_iadd_imm(12, 12, 0x10),                                         // IADD R12, R12, 0x10
        fg_ret_r12(),                                                      // RET.ABS R12 0x20
        // checkpoint_signal (idx 61). Dispatch Mode 4 vs Mode 5 (Path A).
        fg_isetp_ge_u32_imm(10, 0),                                         // idx 61: P0 = true (bound Mode 4/5 command active)
        fg_bra_not_p0(-6),                                                  // idx 62: @!P0 BRA done (idx 57)
        fg_mov_reg(1, 0),                                                  // idx 63: R1 = command (4, 5, or 6)
        fg_ldc_u32(0, 0x184),                                              // idx 64: Require explicit Mode 4/5 binding
        fg_isetp_eq_imm(0, uint64_t(kCheckpointKernelTag) >> 32),          // idx 65
        fg_bra_not_p0(-10),                                                // idx 66: @!P0 BRA done (idx 57)
        fg_isetp_ne_imm(1, kCheckpointCommand),                            // idx 67: P0 = (R1 != 4)
        fg_bra_p0(7),                                                      // idx 68: @P0 BRA path_a_entry (idx 76)
        {((uint64_t(0x5e00u) << 32) | 0xff047b82ULL), ((uint64_t(0xe2200u) << 32) | 0xa00ULL)},                    // idx 69: LDC.64 R4, c[0][0x178]
        {0x7919ULL, ((uint64_t(0xe6400u) << 32) | 0x0ULL)},                    // idx 70: S2R R0, SR_LANEID
        fg_isetp_ne_rz(5),                                                 // idx 71
        {((uint64_t(0xffu) << 32) | 0x720cULL), ((uint64_t(0x2fda00u) << 32) | 0x702270ULL)},                    // idx 72: P0 &= (lane == 0)
        fg_mov_imm_w01(0, 1),                                              // idx 73
        {0x400098eULL, ((uint64_t(0xfda00u) << 32) | 0xf10e1ffULL)},                    // idx 74: @P0 REDG.E.OR.STRONG.GPU [R4.64+URZ], R0
        fg_bra_rel(-19),                                                   // idx 75: BRA done (idx 57)
        // path_a_entry (idx 76): Universal Zero-Source-Modification Mode 5 Evict & Restore
        fg_s2r_tid_x(4),                                                   // idx 76: S2R R4, SR_TID.X (&wr=0x0)
        fg_s2r_tid_y(5),                                                   // idx 77: S2R R5, SR_TID.Y (&wr=0x1)
        fg_ldc_u32_w2(0, 0x360),                                           // idx 78: LDC R0, c[0][0x360] (blockDim.x, &wr=0x2)
        fg_ldc_u32_w3(10, 0x364),                                          // idx 79: LDC R10, c[0][0x364] (blockDim.y, &wr=0x3)
        fg_imad_regs(4, 5, 0, 4, 0x07),                                    // idx 80: tid = tid.y * blockDim.x + tid.x (&req=0x07)
        fg_ldc_64_r8_178(),                                                // idx 81: LDC.64 R8, c[0][0x178] (arena_va, &wr=0x0)
        fg_imad_rz(5, 0, 10),                                              // idx 82: threads_per_cta = blockDim.x * blockDim.y (&req=0x0c)
        fg_iadd3_neg4(0, 3),                                               // idx 83: R0 = 2 * cta_id
        fg_shf_r_u32(0, 0, 1),                                             // idx 84: R0 = cta_id
        fg_imad_regs(10, 0, 5, 4, 0x00),                                   // idx 85: global_tid = cta_id * threads_per_cta + tid
        fg_imad_wide_u32(6, 10, 0x480, 8),                                 // idx 86: my_frame_va = arena_va + global_tid * 1152 (&req=0x10)
        fg_ld_e_strong_gpu_r0_r6_off(0x70),                                // idx 87: R0 = [R6+0x70] (status, &wr=0x0)
        fg_imad_shl_u32_rz(4, 4, 4),                                       // idx 88: R4 = tid * 4 (SMEM offset)
        fg_imad_shl_u32_rz(5, 5, 4),                                       // idx 89: R5 = threads_per_cta * 4 (SMEM stride)
        fg_iadd3_imm(10, 3, 444),                                          // idx 90: R10 = 2 * (cta_id + 224)
        fg_imad_wide_u32(8, 10, 0x8000, 8),                                // idx 91: R8:R9 = arena_va + (cta_id + 224) * 65536
        fg_imad_wide_u32(8, 4, 1, 8),                                      // idx 92: R8:R9 += tid * 4
        fg_isetp_eq_imm(1, kTrapFrameRestoreCommand),                      // idx 93: P0 = (R1 == 6) (&req=0x01)
        fg_bra_p0_to(ext_va + 94 * 16, ext2_va + kRestoreIdx * 16),        // idx 94: @P0 BRA path_a_restore (ext2 idx kRestoreIdx)
        fg_isetp_ne_rz(0),                                                 // idx 95: P0 = (R0 != 0)
        fg_bra_p0(-40),                                                    // idx 96: @P0 BRA done_r1 (idx 57)
        // path_a_evict (idx 97): Cooperative SMEM save + 112B hardware frame + jump to ext2
        fg_st_e_r6_off_r1(0x70),                                           // idx 97: [R6+0x70] = R1 (5=evicted, 7=drained)
        fg_isetp_eq_imm(1, kTrapFrameDrainCommand),                        // idx 98: P0 = (R1 == 7)
        fg_bra_p0_to(ext_va + 99 * 16, ext2_va + kNotifyIdx * 16),         // idx 99: @P0 BRA path_a_notify (fast drain unstarted wave)
        fg_lds_r0_r4(),                                                    // idx 100: LDS R0, [R4] (&req=0x03 &wr=0x0)
        fg_st_e_r8_off_r0(0x0),                                            // idx 101: ST.E.STRONG.GPU [R8], R0 (&req=0x03 &rd=0x1)
        fg_imad_iadd(4, 4, 5),                                             // idx 102: R4 += R5
        fg_imad_wide_u32(8, 5, 1, 8),                                      // idx 103: R8:R9 += R5
        fg_isetp_ge_u32_imm(4, 0x10000),                                   // idx 104: P0 = (R4 >= 65536)
        fg_bra_not_p0(-6),                                                 // idx 105: @!P0 BRA idx 100
        fg_ldl_128_r8(0xffff90),                                           // idx 106: THREAD_STATE_ENUM.0..3
        fg_st_e_128_r6_r8(0x00),                                           // idx 107
        fg_ldl_128_r8(0xffffa0),                                           // idx 108: THREAD_STATE_ENUM.4 + TRAP_RETURN_PC
        fg_st_e_128_r6_r8(0x10),                                           // idx 109
        fg_ldl_128_r8(0xffffb0),                                           // idx 110: Rpc.LO, Rpc.HI, OPT_STACK, B0
        fg_st_e_128_r6_r8(0x20),                                           // idx 111
        fg_ldl_128_r8(0xffffc0),                                           // idx 112: R0..R3
        fg_st_e_128_r6_r8(0x30),                                           // idx 113
        fg_ldl_128_r8(0xffffd0),                                           // idx 114: R4..R7
        fg_st_e_128_r6_r8(0x40),                                           // idx 115
        fg_ldl_128_r8(0xffffe0),                                           // idx 116: R8..R11
        fg_st_e_128_r6_r8(0x50),                                           // idx 117
        fg_ldl_128_r8(0xfffff0),                                           // idx 118: R12, R13, PR, TRAP_RETURN_MASK
        fg_st_e_128_r6_r8(0x60),                                           // idx 119
        fg_mov_r_ur(8, 4),                                                 // idx 120
        fg_mov_r_ur(9, 5),                                                 // idx 121
        fg_mov_r_ur(10, 6),                                                // idx 122
        fg_mov_r_ur(11, 7),                                                // idx 123
        fg_st_e_128_r6_rs(0x440, 8),                                       // idx 124: save UR4..UR7 at +0x440
        fg_mov_r_ur(8, 8),                                                 // idx 125
        fg_mov_r_ur(9, 9),                                                 // idx 126
        fg_mov_r_ur(10, 10),                                               // idx 127
        fg_mov_r_ur(11, 11),                                               // idx 128
        fg_st_e_128_r6_rs(0x450, 8),                                       // idx 129: save UR8..UR11 at +0x450
        fg_mov_r_ur(8, 12),                                                // idx 130
        fg_mov_r_ur(9, 13),                                                // idx 131
        fg_mov_r_ur(10, 14),                                               // idx 132
        fg_mov_r_ur(11, 15),                                               // idx 133
        fg_st_e_128_r6_rs(0x460, 8),                                       // idx 134: save UR12..UR15 at +0x460
        b2r_warp_r8,                                                       // idx 135: B2R.WARP R8 (&wr=0x0)
        fg_st_e_32_r6_rs(0x74, 8),                                         // idx 136: save B2R.WARP to [R6+0x74]
        fg_bra_to(ext_va + 137 * 16, ext3_va + kExt3SaveIdx * 16),         // idx 137: BRA ext3_save (save R14..R[N-3])
    };
    while (ext_code.size() < 147) {
        ext_code.push_back(fg_nop());
    }
    std::vector<FgInstr> ext2_code = {
        // Section 1: path_a_restore (idx 0 .. idx 40): Cooperative SMEM + 112B frame + UR4..UR15 + jump to ext3_restore
        fg_isetp_eq_imm(0, 0),                                             // idx 0: P0 = (status == 0, pre-completed CTA)
        fg_bra_p0_to(ext2_va + (kRestoreIdx + 1) * 16, ext3_va + kExt3CleanExitIdx * 16), // idx 1: @P0 BRA clean_exit_redirect
        fg_isetp_ne_imm(0, kTrapFrameEvictCommand),                        // idx 2: P0 = (status != 5, already restored)
        fg_bra_p0_to(ext2_va + (kRestoreIdx + 3) * 16, ext_va + 57 * 16),  // idx 3: @P0 BRA done_r1 (ext idx 57)
        fg_st_e_r6_off_r1(0x70),                                           // idx 4: [R6+0x70] = 6 (restored)
        fg_ld_e_strong_gpu_r0_r8_off(0x0),                                 // idx 5: LD.E.STRONG.GPU R0, [R8] (&req=0x03 &wr=0x0)
        fg_sts_r4_r0(),                                                    // idx 6: STS [R4], R0 (&req=0x03 &rd=0x1)
        fg_imad_iadd(4, 4, 5),                                             // idx 7: R4 += R5
        fg_imad_wide_u32(8, 5, 1, 8),                                      // idx 8: R8:R9 += R5
        fg_isetp_ge_u32_imm(4, 0x10000),                                   // idx 9: P0 = (R4 >= 65536)
        fg_bra_not_p0(-6),                                                 // idx 10: @!P0 BRA idx 5
        fg_ld_e_128_r8_r6(0x00),                                           // idx 11: THREAD_STATE_ENUM.0..3
        fg_stl_128_r8(0xffff90),                                           // idx 12
        fg_ld_e_128_r8_r6(0x10),                                           // idx 13: THREAD_STATE_ENUM.4 + TRAP_RETURN_PC
        fg_stl_128_r8(0xffffa0),                                           // idx 14
        fg_ld_e_128_r8_r6(0x20),                                           // idx 15: Rpc.LO, Rpc.HI, OPT_STACK, B0
        fg_stl_128_r8(0xffffb0),                                           // idx 16
        fg_ld_e_128_r8_r6(0x30),                                           // idx 17: R0..R3
        fg_stl_128_r8(0xffffc0),                                           // idx 18
        fg_ld_e_128_r8_r6(0x40),                                           // idx 19: R4..R7
        fg_stl_128_r8(0xffffd0),                                           // idx 20
        fg_ld_e_128_r8_r6(0x50),                                           // idx 21: R8..R11
        fg_stl_128_r8(0xffffe0),                                           // idx 22
        fg_ld_e_128_r8_r6(0x60),                                           // idx 23: R12, R13, PR, TRAP_RETURN_MASK
        fg_stl_128_r8(0xfffff0),                                           // idx 24
        fg_ld_e_128_r8_r6(0x440),                                          // idx 25: UR4..UR7 -> R8..R11 (&wr=0x0)
        fg_r2ur_req04(4, 8),                                               // idx 26
        fg_r2ur(5, 9),                                                     // idx 27
        fg_r2ur(6, 10),                                                    // idx 28
        fg_r2ur(7, 11),                                                    // idx 29
        fg_ld_e_128_r8_r6(0x450),                                          // idx 30: UR8..UR11 -> R8..R11 (&wr=0x0)
        fg_r2ur_req04(8, 8),                                               // idx 31
        fg_r2ur(9, 9),                                                     // idx 32
        fg_r2ur(10, 10),                                                   // idx 33
        fg_r2ur(11, 11),                                                   // idx 34
        fg_ld_e_128_r8_r6(0x460),                                          // idx 35: UR12..UR15 -> R8..R11 (&wr=0x0)
        fg_r2ur_req04(12, 8),                                              // idx 36
        fg_r2ur(13, 9),                                                    // idx 37
        fg_r2ur(14, 10),                                                   // idx 38
        fg_r2ur(15, 11),                                                   // idx 39
        fg_bra_to(ext2_va + 40 * 16, ext3_va + kExt3RestoreIdx * 16),      // idx 40: BRA ext3_restore (restore R14..R[N-3])

        // Section 2: path_a_notify (idx 41 .. idx 62): Multi-warp atomic notify + Cooperative SMEM Restore Barrier
        fg_ldc_u32(8, 0x170),                                              // idx 41: lo32(control_va) (&wr=0x0)
        fg_ldc_u32(9, 0x174),                                              // idx 42: hi32(control_va) (&wr=0x0)
        fg_s2r_laneid(0),                                                  // idx 43: S2R R0, SR_LANEID (&wr=0x1)
        fg_imad_wide_u32(8, 3, 4, 8),                                      // idx 44: R8:R9 = &control[4 + 2*cta_id] (&req=0x01)
        fg_iadd3_imm_req01(5, 5, 124),                                     // idx 45: R5 = threads_per_cta * 4 + 124 (&req=0x01)
        fg_shf_r_u32(5, 5, 7),                                             // idx 46: R5 = warps_per_cta = W
        fg_isetp_eq_imm(1, kTrapFrameDrainCommand),                        // idx 47: P0 = (R1 == 7, drain wave without setting w)
        fg_bra_p0(1),                                                      // idx 48: @P0 BRA idx 50
        fg_st_e_r8_off_r5(0x0),                                            // idx 49: control[4 + 2*cta_id + 0] = W
        fg_isetp_eq_imm_req02(0, 0),                                       // idx 50: P0 = (lane_id == 0) (&req=0x02)
        fg_mov_imm_w01(0, 1),                                              // idx 51: R0 = 1
        fg_redg_add_p0_r8_4_r0(),                                          // idx 52: @P0 REDG.E.ADD.STRONG.GPU [R8.64+0x4], R0
        fg_isetp_ne_imm(1, kTrapFrameRestoreCommand),                      // idx 53: P0 = (R1 != 6, i.e. Evict 5 or Drain 7)
        fg_bra_p0_to(ext2_va + (kNotifyBase + 13) * 16, ext3_va + kExt3CleanExitIdx * 16), // idx 54: @P0 BRA clean_exit_redirect
        fg_imad_shl_u32_rz(5, 5, 2),                                       // idx 55: R5 = 2 * W
        fg_ld_e_strong_gpu_r0_r8_off(0x4),                                 // idx 56: R0 = control[4 + 2*cta_id + 1] (&wr=0x0)
        fg_isetp_ge_u32_r0_r5(),                                           // idx 57: P0 = (R0 >= 2 * W) (&req=0x01)
        fg_bra_not_p0(-3),                                                 // idx 58: @!P0 BRA idx 56
        fg_ld_e_strong_gpu_r0_r6_off(0x74),                                // idx 59: R0 = saved B2R.WARP (&wr=0x0)
        fg_isetp_ge_u32_imm(0, 0x1000),                                    // idx 60: P0 = (R0 >= 0x1000) (&req=0x01)
        fg_bar_sync_defer_p0(),                                            // idx 61: @P0 BAR.SYNC.DEFER_BLOCKING 0x0
        fg_bra_to(ext2_va + (kNotifyBase + 21) * 16, ext_va + 57 * 16),    // idx 62: BRA done_r1 (ext idx 57 -> RTT.FALLTHROUGH)
    };
    while (ext2_code.size() < 80) {
        ext2_code.push_back(fg_nop());
    }
    std::vector<FgInstr> ext3_code;
    BuildFineGrainedExt3Code(255, ext_va, ext2_va, ext3_va, kNotifyIdx, ext3_code);

    std::vector<FgInstr> exit_stub = { fg_exit(), fg_exit() };
    std::vector<FgInstr> pause_tramp = { fg_nanosleep_imm(0x100), fg_bra_rel(-2), fg_bra_rel(-1) };

    std::vector<uint64_t> flat_ext, flat_ext2, flat_ext3, flat_exit, flat_tramp;
    for (auto &ins : ext_code) { flat_ext.push_back(ins.w0); flat_ext.push_back(ins.w1); }
    for (auto &ins : ext2_code) { flat_ext2.push_back(ins.w0); flat_ext2.push_back(ins.w1); }
    for (auto &ins : ext3_code) { flat_ext3.push_back(ins.w0); flat_ext3.push_back(ins.w1); }
    for (auto &ins : exit_stub) { flat_exit.push_back(ins.w0); flat_exit.push_back(ins.w1); }
    for (auto &ins : pause_tramp) { flat_tramp.push_back(ins.w0); flat_tramp.push_back(ins.w1); }

    memcpy(FG.orig_390, &handler_img[0x390], 16);
    FG.ext_backup.resize(flat_ext.size());
    FG.ext2_backup.resize(flat_ext2.size());
    FG.ext3_backup.resize(flat_ext3.size());
    FG.exit_backup.resize(flat_exit.size());
    FG.tramp_backup.resize(flat_tramp.size());
    if (cuXtraMemcpyDtoH(FG.ext_backup.data(), ext_va, flat_ext.size() * 8, 0) != flat_ext.size() * 8 ||
        cuXtraMemcpyDtoH(FG.ext2_backup.data(), ext2_va, flat_ext2.size() * 8, 0) != flat_ext2.size() * 8 ||
        cuXtraMemcpyDtoH(FG.ext3_backup.data(), ext3_va, flat_ext3.size() * 8, 0) != flat_ext3.size() * 8 ||
        cuXtraMemcpyDtoH(FG.exit_backup.data(), exit_stub_va, flat_exit.size() * 8, 0) != flat_exit.size() * 8 ||
        cuXtraMemcpyDtoH(FG.tramp_backup.data(), pause_tramp_va, flat_tramp.size() * 8, 0) != flat_tramp.size() * 8) {
        return false;
    }

    uint64_t o0, o1;
    memcpy(&o0, &handler_img[0x380], 8);
    memcpy(&o1, &handler_img[0x388], 8);
    uint64_t call_off = ext_va >> 2, branch_inst[2];
    branch_inst[0] = (((call_off >> 8) & 0x3fffffffULL) << 34) |
                     ((call_off & 0xffULL) << 16) | (o0 & 0xffffULL);
    branch_inst[1] = (o1 & ~0xffULL) | ((call_off >> 38) & 0xffULL);

    FG.ctx = ctx;
    FG.get_func_va = (CUresult (*)(CUcontext, void *, CUdeviceptr *))mt[24];
    FG.handler_va = handler_va;
    FG.ext_va = ext_va;
    FG.ext2_va = ext2_va;
    FG.ext3_va = ext3_va;
    FG.exit_stub_va = exit_stub_va;
    FG.pause_tramp_va = pause_tramp_va;
    FG.installed = true;
    if (!WriteFineGrainedTrapCode(exit_stub_va, flat_exit.data(), flat_exit.size() * 8) ||
        !WriteFineGrainedTrapCode(pause_tramp_va, flat_tramp.data(), flat_tramp.size() * 8) ||
        !WriteFineGrainedTrapCode(ext3_va, flat_ext3.data(), flat_ext3.size() * 8) ||
        !WriteFineGrainedTrapCode(ext2_va, flat_ext2.data(), flat_ext2.size() * 8) ||
        !WriteFineGrainedTrapCode(ext_va, flat_ext.data(), flat_ext.size() * 8) ||
        !WriteFineGrainedTrapCode(handler_va + 0x390, branch_inst, sizeof(branch_inst))) {
        RestoreFineGrainedTrapCode();
        return false;
    }
    cuXtraInvalInstrCache(ctx);
    FG.callable = true;
    return true;
}

bool TrapSm120Context::BindKernelPreemptBuffer(CUfunction func, uint64_t preempt_buf_va,
                                               int64_t kernel_idx, uint64_t guardian_va)
{
    if (preempt_buf_va == 0) {
        FG.bound_func_pc = 0;
    }
    BindFineGrainedKernel(func, preempt_buf_va, kernel_idx, guardian_va);
    return true;
}

bool TrapSm120Context::TriggerFineGrainedTrapPulse()
{
    std::lock_guard<std::mutex> lock(fine_grained_mu);
    if (!EnsureBar0Mapped()) return false;
    PulseFineGrainedTrap();
    return true;
}

bool TrapSm120Context::BindKernelCheckpointRequest(CUfunction func, uint64_t control_va,
                                                  uint64_t request_va)
{
    std::lock_guard<std::mutex> lock(fine_grained_mu);
    if (!FG.callable || !FineGrainedContextIsCurrent() || !func ||
        !uint32_t(control_va) || !(request_va >> 32) ||
        ((control_va | request_va) & 3u)) {
        return false;
    }
    BindFineGrainedKernel(func, control_va, kCheckpointKernelTag, request_va);
    return true;
}

bool TrapSm120Context::RequestStreamCheckpoint(volatile uint32_t *control_host)
{
    std::lock_guard<std::mutex> lock(fine_grained_mu);
    if (!FG.callable || !FineGrainedContextIsCurrent() || !control_host || !FG.bar0) return false;
    control_host[0] = kCheckpointCommand;
    __sync_synchronize();
    PulseFineGrainedTrap();
    return true;
}

bool TrapSm120Context::BindKernelTrapFrameEvict(CUfunction func, uint64_t control_va,
                                                uint64_t trap_frames_va)
{
    std::lock_guard<std::mutex> lock(fine_grained_mu);
    if (!FG.callable || !FineGrainedContextIsCurrent() || !func ||
        !uint32_t(control_va) || !uint32_t(trap_frames_va)) {
        return false;
    }
    int num_regs = 32;
    CUresult r_attr = Driver::FuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
    if (r_attr != CUDA_SUCCESS || num_regs < 2) {
        num_regs = 32;
    }
    std::vector<FgInstr> ext3_code;
    BuildFineGrainedExt3Code(num_regs, FG.ext_va, FG.ext2_va, FG.ext3_va, 41, ext3_code);
    std::vector<uint64_t> flat_ext3;
    flat_ext3.reserve(ext3_code.size() * 2);
    for (auto &ins : ext3_code) {
        flat_ext3.push_back(ins.w0);
        flat_ext3.push_back(ins.w1);
    }
    if (!WriteFineGrainedTrapCode(FG.ext3_va, flat_ext3.data(), flat_ext3.size() * 8)) {
        return false;
    }
    cuXtraInvalInstrCache(FG.ctx);
    FG.bound_func_pc = cuXtraGetEntryPoint(func);
    const int64_t tagged_idx = int64_t(uint64_t(kCheckpointKernelTag) | uint32_t(num_regs));
    BindFineGrainedKernel(func, control_va, tagged_idx, trap_frames_va);
    return true;
}

bool TrapSm120Context::RequestStreamTrapFrameEvict(volatile uint32_t *control_host, int num_ctas)
{
    std::lock_guard<std::mutex> lock(fine_grained_mu);
    if (!FG.callable || !FineGrainedContextIsCurrent() || !control_host || !FG.bar0) return false;
    control_host[0] = kTrapFrameEvictCommand;
    __sync_synchronize();
    if (num_ctas <= 0) {
        PulseFineGrainedTrap();
        return true;
    }
    int last_done = 0;
    uint32_t warps_per_cta = 0;
    bool drain_mode = false;
    auto last_progress = std::chrono::steady_clock::now();
    for (int pulse = 0; pulse < 2000; ++pulse) {
        PulseFineGrainedTrap();
        const int poll_window_us = drain_mode ? 6 : ((pulse == 0) ? 110 : 20);
        const auto wait_end = std::chrono::steady_clock::now() + std::chrono::microseconds(poll_window_us);
        while (std::chrono::steady_clock::now() < wait_end) {
            int evicted_done = 0;
            int drained_done = 0;
            int active_partial = 0;
            for (int b = 0; b < num_ctas; ++b) {
                const uint32_t w = __atomic_load_n(&control_host[4 + 2 * b + 0], __ATOMIC_ACQUIRE);
                const uint32_t ack = __atomic_load_n(&control_host[4 + 2 * b + 1], __ATOMIC_ACQUIRE);
                if (w > 0) {
                    if (warps_per_cta == 0) warps_per_cta = w;
                    if (ack >= w) {
                        ++evicted_done;
                    } else if (ack > 0) {
                        ++active_partial;
                    }
                } else if (ack > 0) {
                    if (warps_per_cta > 0 && ack >= warps_per_cta) {
                        ++drained_done;
                    } else {
                        ++active_partial;
                    }
                }
            }
            const int total_done = evicted_done + drained_done;
            const auto now = std::chrono::steady_clock::now();
            if (total_done != last_done || active_partial > 0) {
                last_done = total_done;
                last_progress = now;
            }
            if (num_ctas > 30 && !drain_mode && active_partial == 0 &&
                (evicted_done >= 30 ||
                 (evicted_done > 0 &&
                  std::chrono::duration_cast<std::chrono::microseconds>(now - last_progress).count() >= 25))) {
                control_host[0] = kTrapFrameDrainCommand;
                __sync_synchronize();
                drain_mode = true;
                last_progress = now;
                const auto spawn_wait = std::chrono::steady_clock::now() + std::chrono::microseconds(3);
                while (std::chrono::steady_clock::now() < spawn_wait) {}
                break;
            }
            if ((total_done == num_ctas && active_partial == 0) ||
                (drain_mode && total_done > 0 && active_partial == 0 &&
                 std::chrono::duration_cast<std::chrono::microseconds>(now - last_progress).count() >= 24) ||
                (pulse >= 10 && total_done > 0 && active_partial == 0 &&
                 std::chrono::duration_cast<std::chrono::microseconds>(now - last_progress).count() >= 200)) {
                control_host[0] = 0;
                __sync_synchronize();
                return true;
            }
        }
    }
    control_host[0] = 0;
    __sync_synchronize();
    return false;
}

bool TrapSm120Context::PrepareKernelTrapFrameRestore(CUfunction func)
{
    std::lock_guard<std::mutex> lock(fine_grained_mu);
    if (!FG.callable || !FineGrainedContextIsCurrent() || !func) return false;
    CUdeviceptr func_pc = cuXtraGetEntryPoint(func);
    if (!func_pc) return false;
    if (FG.func_entry_patched && FG.patched_func_pc) {
        WriteFineGrainedTrapCode(FG.patched_func_pc, FG.func_entry_backup, sizeof(FG.func_entry_backup));
        FG.func_entry_patched = false;
        FG.patched_func_pc = 0;
    }
    if (cuXtraMemcpyDtoH(FG.func_entry_backup, func_pc, sizeof(FG.func_entry_backup), 0) != sizeof(FG.func_entry_backup)) {
        return false;
    }
    // 15-instruction 3-state conditional entry trampoline at handler_va + 0x0c60 (ext2_va + 64 * 16):
    // Distinguishes:
    //   State A (Pre-completed before preemption): ack == 0 && w == 0 -> immediate EXIT
    //   State B (Evicted mid-execution by cmd=5):  w > 0              -> spin in NANOSLEEP for RTT trap
    //   State C (Drained unstarted by cmd=7):      w == 0 && ack > 0  -> execute orig_entry_inst + BRA (func_pc + 0x10)
    const uint64_t entry_tramp_va = FG.ext2_va + 64 * 16;
    FgInstr orig_entry_inst = {FG.func_entry_backup[0], FG.func_entry_backup[1]};
    std::vector<FgInstr> entry_tramp = {
        fg_ldc_u32(4, 0x170),
        fg_ldc_u32(5, 0x174),
        fg_s2r_ctaid_x_w1(0),
        fg_imad_wide_u32(6, 0, 8, 4),
        fg_ld_e_strong_gpu_r0_r6_off(0x14),
        fg_isetp_ne_rz(0),
        fg_bra_p0(1),
        fg_exit(),
        fg_ld_e_strong_gpu_r0_r6_off(0x10),
        fg_isetp_ne_rz(0),
        fg_bra_not_p0(2),
        fg_nanosleep_imm(0x40),
        fg_bra_rel(-2),
        orig_entry_inst,
        fg_bra_to(entry_tramp_va + 14 * 16, func_pc + 16),
    };
    std::vector<uint64_t> flat_entry_tramp;
    for (auto &ins : entry_tramp) {
        flat_entry_tramp.push_back(ins.w0);
        flat_entry_tramp.push_back(ins.w1);
    }
    if (!WriteFineGrainedTrapCode(entry_tramp_va, flat_entry_tramp.data(), flat_entry_tramp.size() * 8)) {
        return false;
    }
    FgInstr jmp_tramp = fg_bra_to(func_pc, entry_tramp_va);
    uint64_t stub[4] = {jmp_tramp.w0, jmp_tramp.w1, FG.func_entry_backup[2], FG.func_entry_backup[3]};
    if (!WriteFineGrainedTrapCode(func_pc, stub, sizeof(stub))) return false;
    cuXtraInvalInstrCache(FG.ctx);
    FG.patched_func_pc = func_pc;
    FG.func_entry_patched = true;
    return true;
}

bool TrapSm120Context::RequestStreamTrapFrameRestore(volatile uint32_t *control_host, int num_ctas)
{
    std::lock_guard<std::mutex> lock(fine_grained_mu);
    if (!FG.callable || !FineGrainedContextIsCurrent() || !control_host || !FG.bar0) return false;
    control_host[0] = kTrapFrameRestoreCommand;
    __sync_synchronize();
    if (num_ctas <= 0) {
        PulseFineGrainedTrap();
        return true;
    }
    int target_evicted = 0;
    for (int b = 0; b < num_ctas; ++b) {
        if (__atomic_load_n(&control_host[4 + 2 * b + 0], __ATOMIC_ACQUIRE) > 0) {
            ++target_evicted;
        }
    }
    if (target_evicted == 0) target_evicted = num_ctas;
    int last_done = 0;
    int done = 0;
    int active_partial = 0;
    for (int pulse = 0; pulse < 4000; ++pulse) {
        const int wait_us = (done > 0 && done == last_done && active_partial == 0) ? 250 : 25;
        last_done = done;
        PulseFineGrainedTrap();
        const auto wait_end = std::chrono::steady_clock::now() + std::chrono::microseconds(wait_us);
        while (std::chrono::steady_clock::now() < wait_end) {
            done = 0;
            active_partial = 0;
            for (int b = 0; b < num_ctas; ++b) {
                const uint32_t w = __atomic_load_n(&control_host[4 + 2 * b + 0], __ATOMIC_ACQUIRE);
                const uint32_t ack = __atomic_load_n(&control_host[4 + 2 * b + 1], __ATOMIC_ACQUIRE);
                if (w > 0 && ack >= 2u * w) {
                    ++done;
                } else if (w > 0 && ack > w) {
                    ++active_partial;
                }
            }
            if (done >= target_evicted) {
                control_host[0] = 0;
                __sync_synchronize();
                return true;
            }
        }
    }
    control_host[0] = 0;
    __sync_synchronize();
    return false;
}

bool TrapSm120Context::FinishKernelTrapFrameRestore()
{
    std::lock_guard<std::mutex> lock(fine_grained_mu);
    if (!FineGrainedContextIsCurrent()) return false;
    if (FG.func_entry_patched && FG.patched_func_pc) {
        if (!WriteFineGrainedTrapCode(FG.patched_func_pc, FG.func_entry_backup, sizeof(FG.func_entry_backup))) {
            return false;
        }
        cuXtraInvalInstrCache(FG.ctx);
        FG.func_entry_patched = false;
        FG.patched_func_pc = 0;
    }
    return true;
}

bool TrapSm120Context::EarlyExitStream(volatile uint32_t *preempt_buf_host)
{
    if (!preempt_buf_host) return false;
    preempt_buf_host[0] = 1u;
    __sync_synchronize();
    return TriggerFineGrainedTrapPulse();
}

bool TrapSm120Context::PauseStreamInPlace(volatile uint32_t *preempt_buf_host)
{
    if (!preempt_buf_host) return false;
    preempt_buf_host[0] = 2u;
    __sync_synchronize();
    return TriggerFineGrainedTrapPulse();
}

bool TrapSm120Context::ResumeStreamInPlace(volatile uint32_t *preempt_buf_host)
{
    if (!preempt_buf_host) return false;
    preempt_buf_host[0] = 0u;
    __sync_synchronize();
    return TriggerFineGrainedTrapPulse();
}

bool TrapSm120Context::ConfigureAndPauseCtasInPlace(volatile uint32_t *preempt_buf_host,
                                                    const uint32_t *cta_modes, int num_ctas)
{
    if (!preempt_buf_host || !cta_modes || num_ctas <= 0) return false;
    for (int cta = 0; cta < num_ctas; ++cta) {
        preempt_buf_host[4 + 2 * cta] = cta_modes[cta];
    }
    preempt_buf_host[0] = 3u;
    __sync_synchronize();
    return TriggerFineGrainedTrapPulse();
}

bool TrapSm120Context::RestoreOriginalTrapHandler()
{
    std::lock_guard<std::mutex> lock(fine_grained_mu);
    if (!FG.installed) return true;
    return FineGrainedContextIsCurrent() && RestoreFineGrainedTrapCode();
}
