#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdarg>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace {

constexpr unsigned long kNvEscRmAlloc   = 0xc030462bUL;
constexpr unsigned long kNvEscRmControl = 0xc020462aUL;

struct NVOS21 {
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

struct NVOS54 {
    uint32_t hClient;
    uint32_t hObject;
    uint32_t cmd;
    uint32_t flags;
    uint64_t params;
    uint32_t paramsSize;
    int32_t  status;
};
static_assert(sizeof(NVOS21) == 48 && sizeof(NVOS54) == 32, "RM ABI size check");

// Corrected 56-byte NV2080_CTRL_GPU_REG_OP layout (32-byte core + 24-byte tail)
// where regOffset is at +12 (+0x0c), vs the buggy 28/32-byte layout in regops.cpp
// where regAndNackIndex was placed at +4 and +12 was left 0.
struct Nv2080CtrlGpuRegOp56 {
    uint8_t  regOp;           // +0: 0=READ_32, 1=WRITE_32
    uint8_t  regType;         // +1: 0=GLOBAL, 1=GR_CTX
    uint8_t  regStatus;       // +2: out status (0=SUCCESS)
    uint8_t  regQuad;         // +3
    uint32_t regGroupMask;    // +4
    uint32_t regSubGroupMask; // +8
    uint32_t regOffset;       // +12: BAR0 PRI offset
    uint32_t regValueHi;      // +16
    uint32_t regValueLo;      // +20
    uint32_t regAndNMaskHi;   // +24
    uint32_t regAndNMaskLo;   // +28
    uint64_t rsvd[3];         // +32..+55
};
static_assert(sizeof(Nv2080CtrlGpuRegOp56) == 56, "NV2080_CTRL_GPU_REG_OP 56-byte wrapper");

// Exact 48-byte (0x30) NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS required by RM control table
struct Nv2080CtrlGpuExecRegOpsParams {
    uint32_t hClientTarget;     // +0x00
    uint32_t hChannelTarget;    // +0x04
    uint32_t bNonTransactional; // +0x08
    uint32_t reserved00[2];     // +0x0c
    uint32_t regOpCount;        // +0x14
    uint64_t pRegOps;           // +0x18
    uint64_t grRouteInfo[2];    // +0x20..+0x2f
};
static_assert(sizeof(Nv2080CtrlGpuExecRegOpsParams) == 48, "NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS must be 48 bytes (0x30)");

struct RmHandles {
    int      fd         = -1;
    uint32_t hClient    = 0;
    uint32_t hDevice    = 0;
    uint32_t hSubdevice = 0;
    uint32_t hTsg       = 0;
    uint32_t hChannel   = 0;
    uint32_t hCompute   = 0;
} g_rm;

int (*g_real_ioctl)(int, unsigned long, ...) = nullptr;
__thread int g_in_hook = 0;

void RecordRmAlloc(int fd, void *arg)
{
    NVOS21 a{};
    std::memcpy(&a, arg, sizeof(a));
    if (a.status != 0 || a.hObject == 0) return;
    switch (a.hClass) {
    case 0x0041U: // NV01_ROOT
        if (g_rm.fd < 0) g_rm.fd = fd;
        if (!g_rm.hClient) g_rm.hClient = a.hObject;
        break;
    case 0x0080U: // NV01_DEVICE_0
        if (!g_rm.hDevice) {
            g_rm.hDevice = a.hObject;
            g_rm.hClient = a.hRoot;
            g_rm.fd      = fd;
        }
        break;
    case 0x2080U: // NV20_SUBDEVICE_0
        if (!g_rm.hSubdevice) g_rm.hSubdevice = a.hObject;
        break;
    case 0xa06cU: // KEPLER_CHANNEL_GROUP_A
        if (!g_rm.hTsg) g_rm.hTsg = a.hObject;
        break;
    case 0xc96fU: // BLACKWELL_CHANNEL_GPFIFO_A
        if (!g_rm.hChannel) g_rm.hChannel = a.hObject;
        break;
    case 0xcec0U: // BLACKWELL_COMPUTE_A
        if (!g_rm.hCompute) g_rm.hCompute = a.hObject;
        break;
    default:
        break;
    }
}

} // namespace

extern "C" int ioctl(int fd, unsigned long req, ...) noexcept
{
    va_list ap;
    va_start(ap, req);
    void *arg = va_arg(ap, void *);
    va_end(ap);
    if (!g_real_ioctl) {
        g_real_ioctl = reinterpret_cast<int (*)(int, unsigned long, ...)>(dlsym(RTLD_NEXT, "ioctl"));
    }
    int r = g_real_ioctl(fd, req, arg);
    if (!g_in_hook && r == 0 && arg != nullptr && (_IOC_NR(req) == 0x2b || req == kNvEscRmAlloc)) {
        g_in_hook = 1;
        RecordRmAlloc(fd, arg);
        g_in_hook = 0;
    }
    return r;
}

namespace {

static int RmControl(int fd, uint32_t hClient, uint32_t hObject, uint32_t cmd,
                     void *params, uint32_t paramsSize, uint32_t *statusOut)
{
    NVOS54 p{};
    p.hClient    = hClient;
    p.hObject    = hObject;
    p.cmd        = cmd;
    p.params     = reinterpret_cast<uint64_t>(params);
    p.paramsSize = paramsSize;
    if (!g_real_ioctl) {
        g_real_ioctl = reinterpret_cast<int (*)(int, unsigned long, ...)>(dlsym(RTLD_NEXT, "ioctl"));
    }
    int r = g_real_ioctl(fd, kNvEscRmControl, &p);
    if (statusOut) *statusOut = static_cast<uint32_t>(p.status);
    return r;
}

static bool ReadBar0Reg32(const RmHandles &rm, uint32_t regOffset, uint32_t &valOut, uint8_t &regStatusOut, uint32_t &rmStatusOut)
{
    Nv2080CtrlGpuRegOp56 op{};
    op.regOp         = 0; // READ_32
    op.regType       = 0; // GLOBAL
    op.regOffset     = regOffset;
    op.regAndNMaskLo = 0xffffffffU;
    op.regAndNMaskHi = 0xffffffffU;

    Nv2080CtrlGpuExecRegOpsParams params{};
    params.bNonTransactional = 1;
    params.regOpCount        = 1;
    params.pRegOps           = reinterpret_cast<uint64_t>(&op);

    int rc = RmControl(rm.fd, rm.hClient, rm.hSubdevice, 0x20800122U, &params, sizeof(params), &rmStatusOut);
    if (rc != 0) return false;
    regStatusOut = op.regStatus;
    valOut       = op.regValueLo;
    return true;
}

__global__ void ReadDeviceVaKernel(uint8_t *dst, uint64_t src_va, size_t nbytes)
{
    const uint8_t *src = reinterpret_cast<const uint8_t *>(src_va);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nbytes; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

static bool ReadGpuVaSlice(uint64_t src_va, size_t nbytes, std::vector<uint8_t> &out)
{
    out.assign(nbytes, 0);
    uint8_t *d_buf = nullptr;
    if (cudaMalloc(&d_buf, nbytes) != cudaSuccess) return false;
    ReadDeviceVaKernel<<<1, 256>>>(d_buf, src_va, nbytes);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_buf);
        return false;
    }
    err = cudaMemcpy(out.data(), d_buf, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_buf);
    return err == cudaSuccess;
}

static CUuuid UuidOf(const char *hex)
{
    CUuuid u{};
    for (int i = 0; i < 16; ++i) {
        unsigned int v = 0;
        std::sscanf(hex + 2 * i, "%02x", &v);
        u.bytes[i] = static_cast<char>(v);
    }
    return u;
}

static uint32_t Imm32At(const uint8_t *insn)
{
    uint32_t v = 0;
    std::memcpy(&v, insn + 4, 4);
    return v;
}

static uint32_t OpcodeAt(const uint8_t *insn)
{
    uint32_t v = 0;
    std::memcpy(&v, insn, 4);
    return v;
}

struct UtilityArenaInfo {
    uint64_t slot20_pc = 0;
    uint64_t slot20_sz = 0;
    uint64_t util_base = 0;
};

static bool DiscoverUtilityArena(UtilityArenaInfo &info)
{
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx) return false;
    const void *tt_raw = nullptr, *mt_raw = nullptr;
    CUuuid te = UuidOf("cc529e945c4e9d46837c9625868334e4");
    CUuuid me = UuidOf("bfdb432dbf3c5a4a945eb34029e81e75");
    if (cuGetExportTable(&tt_raw, &te) != CUDA_SUCCESS ||
        cuGetExportTable(&mt_raw, &me) != CUDA_SUCCESS ||
        !tt_raw || !mt_raw) {
        return false;
    }
    const void **tt = reinterpret_cast<const void **>(const_cast<void *>(tt_raw));
    const void **mt = reinterpret_cast<const void **>(const_cast<void *>(mt_raw));
    struct { uint64_t flags, ssize; void *handle; uint64_t stub, hsize, rsv; } ti{};
    auto fn_slot20 = reinterpret_cast<CUresult (*)(CUcontext, void *)>(const_cast<void *>(tt[20]));
    if (fn_slot20(ctx, &ti) != CUDA_SUCCESS || !ti.handle) return false;

    CUdeviceptr pc = 0;
    uint64_t sz = 0;
    auto fn_pc = reinterpret_cast<CUresult (*)(CUcontext, void *, CUdeviceptr *)>(const_cast<void *>(mt[24]));
    auto fn_sz = reinterpret_cast<CUresult (*)(void *, uint64_t *)>(const_cast<void *>(mt[25]));
    if (fn_pc(ctx, ti.handle, &pc) != CUDA_SUCCESS ||
        fn_sz(ti.handle, &sz) != CUDA_SUCCESS || sz < 0x1300) {
        return false;
    }
    std::vector<uint8_t> img;
    if (!ReadGpuVaSlice(static_cast<uint64_t>(pc), static_cast<size_t>(sz), img)) return false;
    uint64_t trt = static_cast<uint64_t>(Imm32At(&img[0x130])) |
                   (static_cast<uint64_t>(Imm32At(&img[0x140])) << 32);
    info.slot20_pc = static_cast<uint64_t>(pc);
    info.slot20_sz = sz;
    info.util_base = trt - 0xc38300ULL;
    return true;
}

static bool ProbeGspClient(const RmHandles &rm)
{
    // On open-gpu-kernel-modules 595.58.03 (Blackwell SM120), pGpu+0x911 is
    // IS_GSP_CLIENT(pGpu) == 1 (NOT a regops disable gate).  We verify on live GPU:
    //  1) Subdevice handle rm.hSubdevice != 0 is active on GSP-RM;
    //  2) 0x20800122 (NV2080_CTRL_CMD_GPU_EXEC_REG_OPS) is NOT gated off by pGpu+0x911:
    //     reading NV_PTIMER_TIME_0 (0x9400) succeeds with rmStatus=0x0, regStatus=0x0,
    //     returning a live non-zero timer value.
    cudaDeviceProp prop{};
    bool dev_ok = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) && (prop.major == 12);

    uint32_t ptimer = 0;
    uint8_t regSt = 0xff;
    uint32_t rmSt = 0xff;
    ReadBar0Reg32(rm, 0x00009400U, ptimer, regSt, rmSt);

    bool is_gsp = dev_ok && (rm.hSubdevice != 0) && (rmSt == 0) && (regSt == 0) && (ptimer != 0);
    std::printf("[PROBE gsp_client] GPU=%s (SM %d.%d) is GSP client (IS_GSP_CLIENT / pGpu+0x911==1): 0x20800122 PTIMER=0x%08x rmSt=0x%x regSt=0x%x -> %s\n",
                prop.name, prop.major, prop.minor, ptimer, rmSt, regSt, is_gsp ? "[PASS]" : "[FAIL]");
    return is_gsp;
}

static bool ProbeRegOpsPtimer(const RmHandles &rm)
{
    uint32_t ptimer0 = 0, ptimer1 = 0;
    uint8_t regSt0 = 0xff, regSt1 = 0xff;
    uint32_t rmSt0 = 0xff, rmSt1 = 0xff;

    if (!ReadBar0Reg32(rm, 0x00009400U, ptimer0, regSt0, rmSt0)) return false;
    ::usleep(1000);
    if (!ReadBar0Reg32(rm, 0x00009400U, ptimer1, regSt1, rmSt1)) return false;

    bool ok = (rmSt0 == 0 && regSt0 == 0 && rmSt1 == 0 && regSt1 == 0 && ptimer1 != ptimer0);
    std::printf("[PROBE regops_ptimer] 0x20800122 56-byte RegOp (+12 regOffset=0x9400): PTIMER_0=0x%08x PTIMER_1=0x%08x delta=%u ns -> %s\n",
                ptimer0, ptimer1, (ptimer1 - ptimer0), ok ? "[PASS]" : "[FAIL]");
    return ok;
}

static bool ProbeReg419e84AndTraphandler(const RmHandles &rm, const UtilityArenaInfo &arena)
{
    uint32_t val419e84 = 0;
    uint8_t regSt = 0xff;
    uint32_t rmSt = 0xff;
    if (!ReadBar0Reg32(rm, 0x00419e84U, val419e84, regSt, rmSt)) return false;

    // On Blackwell SM120, 0x419e84 is unmapped/relocated: raw BAR0 MMIO returns PRI poison
    // 0xbadf1100 (NV_PPRIV_SYS_PRI_ERROR_CODE) and user-mode 0x20800122 rejects it in
    // gpuValidateRegOffset_IMPL (rmSt=0x3a, regSt=0x20, val=0x00000000).
    bool reg_blocked = (rmSt != 0) || (regSt != 0) || (val419e84 == 0xbadf1100U);

    uint64_t traphandler_va = arena.util_base + 0x15f0f00ULL;
    uint64_t trap_table_va  = arena.util_base + 0x15f5300ULL;
    int64_t  rel_offset     = static_cast<int64_t>(traphandler_va) - static_cast<int64_t>(trap_table_va);
    std::vector<uint8_t> hdr;
    if (!ReadGpuVaSlice(traphandler_va, 0xa0, hdr)) return false;

    static const uint32_t fp[10] = {
        0x0000795c, 0x1a007956, 0xff007387, 0xff007387, 0xff007387,
        0x00047202, 0x00057202, 0xff067803, 0x17077355, 0xff007387,
    };
    bool fp_ok = true;
    for (int i = 0; i < 10; ++i) {
        if (OpcodeAt(&hdr[i * 16]) != fp[i]) {
            fp_ok = false;
            break;
        }
    }

    bool ok = (reg_blocked && rel_offset == -0x4400 && fp_ok);
    std::printf("[PROBE reg_419e84_and_traphandler] 0x419e84 raw_val=0x%08x (rmSt=0x%x regSt=0x%x; BAR0 PRI poison=0xbadf1100) .text.traphandler=0x%llx (offset -0x4400 / %lld from .text.trap_entry_table) op0=0x%08x -> %s\n",
                val419e84, rmSt, regSt,
                static_cast<unsigned long long>(traphandler_va),
                static_cast<long long>(rel_offset),
                OpcodeAt(&hdr[0]),
                ok ? "[PASS]" : "[FAIL]");
    return ok;
}

static bool ProbePriRelocation(const RmHandles &rm)
{
    uint32_t v0 = 0, v1 = 0, v2 = 0, vPtimer = 0;
    uint8_t r0 = 0, r1 = 0, r2 = 0, rPt = 0;
    uint32_t s0 = 0, s1 = 0, s2 = 0, sPt = 0;
    ReadBar0Reg32(rm, 0x00419e84U, v0, r0, s0);
    ReadBar0Reg32(rm, 0x00501e84U, v1, r1, s1);
    ReadBar0Reg32(rm, 0x00509e84U, v2, r2, s2);
    ReadBar0Reg32(rm, 0x00009400U, vPtimer, rPt, sPt);

    bool b0 = (s0 != 0) || (r0 != 0) || (v0 == 0xbadf1100U);
    bool b1 = (s1 != 0) || (r1 != 0) || (v1 == 0xbadf1100U);
    bool b2 = (s2 != 0) || (r2 != 0) || (v2 == 0xbadf1100U);

    bool ok = (b0 && b1 && b2 && sPt == 0 && rPt == 0 && vPtimer != 0);
    std::printf("[PROBE pri_relocation] legacy PRI offsets relocated/rejected (BAR0 poison=0xbadf1100): 0x419e84=(val=0x%08x rmSt=0x%x regSt=0x%x) 0x501e84=(val=0x%08x rmSt=0x%x regSt=0x%x) 0x509e84=(val=0x%08x rmSt=0x%x regSt=0x%x) (PTIMER 0x9400=0x%08x rmSt=0x%x regSt=0x%x) -> %s\n",
                v0, s0, r0, v1, s1, r1, v2, s2, r2, vPtimer, sPt, rPt, ok ? "[PASS]" : "[FAIL]");
    return ok;
}

static bool ProbeCaveBoundary(const UtilityArenaInfo &arena)
{
    uint64_t cave_va  = arena.slot20_pc + 0xa00ULL;
    uint64_t cave_end = cave_va + 0x100ULL;
    uint64_t img_end  = arena.slot20_pc + arena.slot20_sz;
    std::vector<uint8_t> slice;
    bool read_ok = ReadGpuVaSlice(cave_va, 0x100, slice);
    bool in_bounds = (arena.slot20_sz >= 0x1300ULL) && (cave_end <= img_end);
    bool ok = read_ok && in_bounds;
    std::printf("[PROBE cave_boundary] slot20=[0x%llx, 0x%llx) sz=0x%llx cave=[0x%llx, 0x%llx) -> %s\n",
                static_cast<unsigned long long>(arena.slot20_pc),
                static_cast<unsigned long long>(img_end),
                static_cast<unsigned long long>(arena.slot20_sz),
                static_cast<unsigned long long>(cave_va),
                static_cast<unsigned long long>(cave_end),
                ok ? "[PASS]" : "[FAIL]");
    return ok;
}

} // namespace

int main(int argc, char **argv)
{
    const char *probe = "all";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--all") == 0) {
            probe = "all";
        } else if (std::strcmp(argv[i], "--probe") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing argument for --probe\n");
                return 1;
            }
            probe = argv[++i];
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--all] [--probe <name>]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown or invalid argument: %s\n", argv[i]);
            return 1;
        }
    }

    if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
        std::fprintf(stderr, "CUDA initialization failed\n");
        return 1;
    }

    UtilityArenaInfo arena;
    if (g_rm.hSubdevice == 0 || !DiscoverUtilityArena(arena)) {
        std::fprintf(stderr, "Failed to discover RM handles (hSubdevice=0x%x) or utility arena (base=0x%llx)\n",
                     g_rm.hSubdevice, static_cast<unsigned long long>(arena.util_base));
        return 1;
    }

    bool ok = true;
    if (std::strcmp(probe, "all") == 0) {
        ok = ProbeGspClient(g_rm) && ok;
        ok = ProbeRegOpsPtimer(g_rm) && ok;
        ok = ProbeReg419e84AndTraphandler(g_rm, arena) && ok;
        ok = ProbePriRelocation(g_rm) && ok;
        ok = ProbeCaveBoundary(arena) && ok;
        if (ok) {
            std::printf("[VERIFY_PASS] VERIFY_PASS: all 5 SM120 hardware compendium claims verified on live GPU [PASS]\n");
        }
    } else if (std::strcmp(probe, "gsp_client") == 0) {
        ok = ProbeGspClient(g_rm);
    } else if (std::strcmp(probe, "regops_ptimer") == 0) {
        ok = ProbeRegOpsPtimer(g_rm);
    } else if (std::strcmp(probe, "reg_419e84_and_traphandler") == 0) {
        ok = ProbeReg419e84AndTraphandler(g_rm, arena);
    } else if (std::strcmp(probe, "pri_relocation") == 0) {
        ok = ProbePriRelocation(g_rm);
    } else if (std::strcmp(probe, "cave_boundary") == 0) {
        ok = ProbeCaveBoundary(arena);
    } else {
        std::fprintf(stderr, "Unknown probe: %s\n", probe);
        ok = false;
    }

    return ok ? 0 : 1;
}
