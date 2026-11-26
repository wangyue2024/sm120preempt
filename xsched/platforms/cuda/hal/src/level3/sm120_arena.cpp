#include "xsched/cuda/hal/level3/sm120_arena.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cuxtra/cuxtra.h>

#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/level3/sm120_bar0_proxy.h"
#include "xsched/utils/log.h"

namespace cuxtra::etbl
{
struct CUtoolsMemObjHandle_st;
}

namespace cuxtra::cuda
{
bool FindMemObjHandle(CUctx_st *ctx, unsigned long long va,
                      cuxtra::etbl::CUtoolsMemObjHandle_st **obj,
                      unsigned long long *offset, unsigned long *size);
}

namespace xsched::cuda::sm120
{

namespace
{

constexpr uint64_t kBraSelfW0            = 0xfffffffc00fc7947ULL;
constexpr uint64_t kTrapReasonTableOff   = 0xc38300ULL;
constexpr uint64_t kLiveHandlerOff       = 0x15f0f00ULL;
constexpr uint64_t kToolsOffsetFromLive  = 0x4400ULL;
constexpr uint64_t kScanWindowBytes      = 0x8200ULL;
constexpr uint64_t kHookSiteW0           = (0xffULL << 32) | 0x37202ULL;
constexpr uint64_t kHookSiteW1           = (0x3ff000ULL << 32) | 0xf00ULL;
constexpr size_t   kBar0MapBytes         = 0x01000000u; // 16 MiB covers 0x424684

// Relocation-free leading opcodes of .text.traphandler on sm_120.
constexpr uint32_t kTrapHandlerFp[10] = {
    0x0000795cU, 0x1a007956U, 0xff007387U, 0xff007387U, 0xff007387U,
    0x00047202U, 0x00057202U, 0xff067803U, 0x17077355U, 0xff007387U,
};

CUuuid ParseUuid(const char *hex)
{
    CUuuid u{};
    for (int i = 0; i < 16; ++i) {
        unsigned v = 0;
        if (sscanf(hex + i * 2, "%2x", &v) == 1) {
            u.bytes[i] = static_cast<char>(v);
        }
    }
    return u;
}

uint32_t ReadU32LE(const uint8_t *p)
{
    uint32_t v = 0;
    memcpy(&v, p, sizeof(v));
    return v;
}

uint32_t Imm32At(const uint8_t *p)
{
    uint64_t w = 0;
    memcpy(&w, p, sizeof(w));
    return static_cast<uint32_t>(w >> 32);
}

bool HasHandlerFingerprint(const uint8_t *buf, size_t len)
{
    if (len < 160) return false;
    for (size_t i = 0; i < 10; ++i) {
        if (ReadU32LE(buf + i * 16) != kTrapHandlerFp[i]) return false;
    }
    return true;
}

void EncodeCallAbs(uint64_t target_va, const uint64_t ref[2], uint64_t out[2])
{
    const uint64_t call_off = target_va >> 2;
    out[0] = (((call_off >> 8) & 0x3fffffffULL) << 34) |
             ((call_off & 0xffULL) << 16) |
             (ref[0] & 0xffffULL);
    out[1] = (ref[1] & ~0xffULL) | ((call_off >> 38) & 0xffULL);
}

bool TryMmapBar0(const char *path, volatile uint32_t **out_ptr, int *out_fd)
{
    if (!path || !path[0]) return false;
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) return false;
    void *p = mmap(nullptr, kBar0MapBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p != MAP_FAILED) {
        *out_ptr = reinterpret_cast<volatile uint32_t *>(p);
        *out_fd = fd;
        return true;
    }
    close(fd);
    return false;
}

} // namespace

Sm120Arena::Sm120Arena(CUcontext ctx) : ctx_(ctx) {}

Sm120Arena::~Sm120Arena()
{
    RestoreAll();
    CUcontext cur = nullptr;
    if (op_stream_ != nullptr &&
        Driver::CtxGetCurrent(&cur) == CUDA_SUCCESS && cur == ctx_) {
        Driver::StreamDestroy_v2(op_stream_);
        op_stream_ = nullptr;
    }
    if (bar0_ != nullptr) {
        munmap(const_cast<uint32_t *>(bar0_), kBar0MapBytes);
        bar0_ = nullptr;
    }
    if (bar0_fd_ >= 0) {
        close(bar0_fd_);
        bar0_fd_ = -1;
    }
    if (proxy_fd_ >= 0) {
        close(proxy_fd_);
        proxy_fd_ = -1;
    }
}

bool Sm120Arena::Discover(bool verify_patch_site)
{
    std::lock_guard<std::mutex> lock(mu_);
    if (discovered_) return true;

    CUcontext current = nullptr;
    if (!ctx_ || Driver::CtxGetCurrent(&current) != CUDA_SUCCESS || current != ctx_) {
        return false;
    }
    if (op_stream_ == nullptr &&
        Driver::StreamCreate(&op_stream_, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
        return false;
    }

    const void *tt_raw = nullptr;
    const void *mt_raw = nullptr;
    CUuuid te = ParseUuid("cc529e945c4e9d46837c9625868334e4");
    CUuuid me = ParseUuid("bfdb432dbf3c5a4a945eb34029e81e75");
    if (Driver::GetExportTable(&tt_raw, &te) != CUDA_SUCCESS ||
        Driver::GetExportTable(&mt_raw, &me) != CUDA_SUCCESS ||
        !tt_raw || !mt_raw) {
        XWARN("sm120_arena: internal CUDA export tables unavailable");
        return false;
    }

    const void **tt = reinterpret_cast<const void **>(const_cast<void *>(tt_raw));
    const void **mt = reinterpret_cast<const void **>(const_cast<void *>(mt_raw));

    struct
    {
        uint64_t flags;
        uint64_t ssize;
        void    *handle;
        uint64_t stub;
        uint64_t hsize;
        uint64_t rsv;
    } ti{};
    if (reinterpret_cast<CUresult (*)(CUcontext, void *)>(const_cast<void *>(tt[20]))(ctx_, &ti) != CUDA_SUCCESS ||
        !ti.handle) {
        return false;
    }

    CUdeviceptr tools_pc_dev = 0;
    uint64_t tools_sz = 0;
    if (reinterpret_cast<CUresult (*)(CUcontext, void *, CUdeviceptr *)>(
            const_cast<void *>(mt[24]))(ctx_, ti.handle, &tools_pc_dev) != CUDA_SUCCESS ||
        reinterpret_cast<CUresult (*)(void *, uint64_t *)>(
            const_cast<void *>(mt[25]))(ti.handle, &tools_sz) != CUDA_SUCCESS ||
        tools_sz < 0x1300) {
        return false;
    }

    std::vector<uint8_t> tools_img(tools_sz);
    if (cuXtraMemcpyDtoH(tools_img.data(), tools_pc_dev, tools_sz, op_stream_) != tools_sz) {
        return false;
    }

    const uint64_t trap_reason_tbl =
        static_cast<uint64_t>(Imm32At(&tools_img[0x130])) |
        (static_cast<uint64_t>(Imm32At(&tools_img[0x140])) << 32);
    arena_base_va_ = trap_reason_tbl - kTrapReasonTableOff;
    tools_pc_      = static_cast<uint64_t>(tools_pc_dev);
    tools_sz_      = tools_sz;

    handler_va_ = (tools_pc_ > kToolsOffsetFromLive)
                ? (tools_pc_ - kToolsOffsetFromLive)
                : (arena_base_va_ + kLiveHandlerOff);

    std::vector<uint8_t> handler_img(0x1200);
    if (cuXtraMemcpyDtoH(handler_img.data(), handler_va_, handler_img.size(), op_stream_) != handler_img.size() ||
        !HasHandlerFingerprint(handler_img.data(), handler_img.size())) {
        handler_va_ = arena_base_va_ + kLiveHandlerOff;
        if (cuXtraMemcpyDtoH(handler_img.data(), handler_va_, handler_img.size(), op_stream_) != handler_img.size() ||
            !HasHandlerFingerprint(handler_img.data(), handler_img.size())) {
            XWARN("sm120_arena: failed .text.traphandler fingerprint check at %#llx",
                  static_cast<unsigned long long>(handler_va_));
            return false;
        }
    }

    memcpy(orig_390_, &handler_img[0x390], sizeof(orig_390_));
    memcpy(call_380_, &handler_img[0x380], sizeof(call_380_));
    if (verify_patch_site) {
        if (orig_390_[0] != kHookSiteW0 || orig_390_[1] != kHookSiteW1 ||
            (static_cast<uint32_t>(call_380_[0]) & 0xffffU) != 0x7943U) {
            XWARN("sm120_arena: unexpected instruction at +0x380/+0x390");
            return false;
        }
    }

    // Dynamically walk every sub-object in [handler_va_, handler_va_ + kScanWindowBytes)
    // and anchor each cave immediately after its `BRA self` (`0xfffffffc00fc7947`).
    caves_.clear();
    uint64_t cursor = handler_va_;
    const uint64_t scan_end = handler_va_ + kScanWindowBytes;
    while (cursor < scan_end) {
        cuxtra::etbl::CUtoolsMemObjHandle_st *obj = nullptr;
        unsigned long long obj_base_raw = 0;
        unsigned long obj_sz = 0;
        if (!cuxtra::cuda::FindMemObjHandle(reinterpret_cast<CUctx_st *>(ctx_),
                                            cursor, &obj, &obj_base_raw, &obj_sz) ||
            obj_base_raw == 0 || obj_sz < 0x100) {
            cursor += 0x100;
            continue;
        }
        const uint64_t obj_base = static_cast<uint64_t>(obj_base_raw);
        std::vector<uint64_t> words(obj_sz / 8);
        if (cuXtraMemcpyDtoH(words.data(), obj_base, words.size() * 8, op_stream_) == words.size() * 8) {
            uint64_t bra_off = 0;
            bool found_bra = false;
            for (size_t i = 0; i + 1 < words.size(); i += 2) {
                if (words[i] == kBraSelfW0) {
                    bra_off = static_cast<uint64_t>(i * 8);
                    found_bra = true;
                    break;
                }
            }
            if (found_bra) {
                uint64_t start_off = bra_off + 0x10;
                if (obj_base == handler_va_ && start_off < 0x860) {
                    start_off = 0x860; // Keep live_handler +0x830..+0x850 margin untouched
                }
                if (start_off < obj_sz) {
                    ArenaCave cave;
                    cave.va              = obj_base + start_off;
                    cave.obj_base        = obj_base;
                    cave.obj_size        = static_cast<uint32_t>(obj_sz);
                    cave.capacity_instrs = static_cast<uint32_t>((obj_sz - start_off) / 16);
                    caves_.push_back(cave);
                }
            }
        }
        const uint64_t next_cursor = obj_base + obj_sz;
        cursor = next_cursor > (cursor + 0x100ULL) ? next_cursor : (cursor + 0x100ULL);
    }

    discovered_ = !caves_.empty();
    return discovered_;
}

bool Sm120Arena::SelectWorkingCaves(size_t needed, uint32_t min_slots,
                                    std::vector<size_t> &out_indices) const
{
    out_indices.clear();
    if (!discovered_ || caves_.empty()) return false;

    int tools_idx = -1;
    int live_idx  = -1;
    for (size_t i = 0; i < caves_.size(); ++i) {
        if (caves_[i].capacity_instrs < min_slots) continue;
        if (caves_[i].obj_base == tools_pc_) tools_idx = static_cast<int>(i);
        else if (caves_[i].obj_base == handler_va_) live_idx = static_cast<int>(i);
    }
    if (tools_idx < 0 || live_idx < 0) return false;

    out_indices.push_back(static_cast<size_t>(tools_idx));
    if (needed > 1) out_indices.push_back(static_cast<size_t>(live_idx));

    for (size_t i = 0; i < caves_.size() && out_indices.size() < needed; ++i) {
        if (static_cast<int>(i) == tools_idx || static_cast<int>(i) == live_idx) continue;
        if (caves_[i].capacity_instrs >= min_slots) {
            out_indices.push_back(i);
        }
    }
    return out_indices.size() == needed;
}

bool Sm120Arena::WriteCave(size_t cave_idx, const std::vector<sass::Ins> &code)
{
    std::lock_guard<std::mutex> lock(mu_);
    if (!discovered_ || cave_idx >= caves_.size()) return false;
    const ArenaCave &cave = caves_[cave_idx];
    if (code.empty() || code.size() > cave.capacity_instrs) return false;

    const size_t bytes = code.size() * sizeof(sass::Ins);
    PatchedRegion reg;
    reg.va = cave.va;
    reg.backup.resize(bytes / 8);
    if (cuXtraMemcpyDtoH(reg.backup.data(), cave.va, bytes, op_stream_) != bytes) {
        return false;
    }
    if (cuXtraMemcpyHtoD(cave.va, code.data(), bytes, op_stream_) != bytes) {
        return false;
    }
    patched_caves_.push_back(std::move(reg));
    return true;
}

bool Sm120Arena::PatchHandlerHook(uint64_t target_va)
{
    std::lock_guard<std::mutex> lock(mu_);
    if (!discovered_) return false;
    uint64_t patch_390[2] = {};
    EncodeCallAbs(target_va, call_380_, patch_390);
    if (cuXtraMemcpyHtoD(handler_va_ + 0x390, patch_390, sizeof(patch_390), op_stream_) != sizeof(patch_390)) {
        return false;
    }
    Driver::StreamSynchronize(op_stream_);
    cuXtraInvalInstrCache(ctx_);
    hook_patched_ = true;
    return true;
}

void Sm120Arena::RestoreAll()
{
    std::lock_guard<std::mutex> lock(mu_);
    if (!discovered_) return;
    CUcontext current = nullptr;
    if (Driver::CtxGetCurrent(&current) != CUDA_SUCCESS || current != ctx_) {
        hook_patched_ = false;
        patched_caves_.clear();
        return;
    }

    if (hook_patched_) {
        cuXtraMemcpyHtoD(handler_va_ + 0x390, orig_390_, sizeof(orig_390_), op_stream_);
        Driver::StreamSynchronize(op_stream_);
        cuXtraInvalInstrCache(ctx_);
        hook_patched_ = false;
    }
    for (auto it = patched_caves_.rbegin(); it != patched_caves_.rend(); ++it) {
        const size_t bytes = it->backup.size() * 8;
        cuXtraMemcpyHtoD(it->va, it->backup.data(), bytes, op_stream_);
    }
    patched_caves_.clear();
    Driver::StreamSynchronize(op_stream_);
    cuXtraInvalInstrCache(ctx_);
}

bool Sm120Arena::EnsureBar0Mapped()
{
    std::lock_guard<std::mutex> lock(mu_);
    if (bar0_ != nullptr) return true;
    {
        std::lock_guard<std::mutex> plock(proxy_mu_);
        if (proxy_fd_ >= 0) return true;
    }
    const char *force = getenv("XSCHED_SM120_BAR0_FORCE_PROXY");
    if (!(force && force[0] == '1') && MapBar0DirectLocked()) return true;
    return ConnectProxyLocked();
}

std::string Sm120Arena::ExpectedBdf()
{
    CUdevice dev = 0;
    int dom = 0, bus = 0, slot = 0;
    if (Driver::CtxGetDevice(&dev) != CUDA_SUCCESS ||
        Driver::DeviceGetAttribute(&dom, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev) != CUDA_SUCCESS ||
        Driver::DeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev) != CUDA_SUCCESS ||
        Driver::DeviceGetAttribute(&slot, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev) != CUDA_SUCCESS) {
        return std::string();
    }
    char bdf[32];
    snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.0", dom, bus, slot);
    return bdf;
}

bool Sm120Arena::ProxyRoundTrip(uint16_t op, int32_t *status, char *bdf_out)
{
    // Caller holds proxy_mu_.
    namespace bp = bar0proxy;
    if (proxy_fd_ < 0) return false;
    bp::Request rq = {bp::kMagic, bp::kVersion, op, ++proxy_seq_};
    bp::Reply rp = {};
    static const int timeout_ms = [] {
        const char *v = getenv("XSCHED_SM120_BAR0_PROXY_TIMEOUT_MS");
        return v ? std::max(1, atoi(v)) : 200;
    }();
    struct pollfd pfd = {proxy_fd_, POLLIN, 0};
    bool ok = send(proxy_fd_, &rq, sizeof(rq), MSG_NOSIGNAL) == static_cast<ssize_t>(sizeof(rq)) &&
              poll(&pfd, 1, timeout_ms) == 1 && (pfd.revents & POLLIN) &&
              recv(proxy_fd_, &rp, sizeof(rp), 0) == static_cast<ssize_t>(sizeof(rp)) &&
              rp.magic == bp::kMagic && rp.seq == rq.seq;
    if (!ok) {
        // Broken or desynchronized connection: drop it; EnsureBar0Mapped reconnects later.
        XWARN("sm120_arena: BAR0 proxy request failed; reconnecting on next pulse");
        close(proxy_fd_);
        proxy_fd_ = -1;
        return false;
    }
    *status = rp.status;
    if (bdf_out) {
        memcpy(bdf_out, rp.bdf, sizeof(rp.bdf));
        bdf_out[sizeof(rp.bdf) - 1] = '\0';
    }
    return true;
}

bool Sm120Arena::ConnectProxyLocked()
{
    namespace bp = bar0proxy;
    std::lock_guard<std::mutex> plock(proxy_mu_);
    if (proxy_fd_ >= 0) return true;
    const char *env = getenv("XSCHED_SM120_BAR0_PROXY");
    if (env && strcmp(env, "0") == 0) return false;
    const auto now = std::chrono::steady_clock::now();
    if (now < proxy_retry_after_) return false;
    proxy_retry_after_ = now + std::chrono::milliseconds(200);

    const std::string want = ExpectedBdf();
    std::string path = (env && env[0]) ? std::string(env)
                                        : std::string(bp::kDefaultDir) + "/bar0-" + want + ".sock";
    if (path.empty() || (!(env && env[0]) && want.empty())) return false;

    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) return false;
    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        if (!proxy_logged_) XWARN("sm120_arena: BAR0 proxy path too long (max %zu bytes): %s",
                                  sizeof(addr.sun_path) - 1, path.c_str());
        proxy_logged_ = true;
        close(fd);
        return false;
    }
    memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    if (connect(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) != 0) {
        if (!proxy_logged_ && errno != ENOENT) {
            XWARN("sm120_arena: cannot connect to BAR0 proxy %s: %s", path.c_str(), strerror(errno));
            proxy_logged_ = true;
        }
        close(fd);
        return false;
    }
    proxy_fd_ = fd;
    int32_t st = -1;
    char served[sizeof(bp::Reply::bdf)] = {};
    if (!ProxyRoundTrip(bp::kOpHello, &st, served) || st != bp::kOk) {
        if (proxy_fd_ >= 0) { close(proxy_fd_); proxy_fd_ = -1; }
        return false;
    }
    if (!want.empty() && want != served) {
        if (!proxy_logged_) XWARN("sm120_arena: BAR0 proxy %s serves %s but this context runs on %s; not using it",
                                  path.c_str(), served, want.c_str());
        proxy_logged_ = true;
        close(proxy_fd_);
        proxy_fd_ = -1;
        return false;
    }
    XINFO("sm120_arena: trap pulses via BAR0 proxy %s (%s, non-root, DEF-10)", path.c_str(), served);
    return true;
}

bool Sm120Arena::MapBar0DirectLocked()
{
    if (TryMmapBar0(getenv("XSCHED_SM120_BAR0_RESOURCE"), &bar0_, &bar0_fd_)) return true;
    if (TryMmapBar0(getenv("BAR0_PATH"), &bar0_, &bar0_fd_)) return true;

    CUdevice dev = 0;
    uint64_t phys_bar0 = 0xb2000000ULL;
    if (Driver::CtxGetDevice(&dev) == CUDA_SUCCESS) {
        int dom = 0, bus = 0, slot = 0;
        if (Driver::DeviceGetAttribute(&dom, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev) == CUDA_SUCCESS &&
            Driver::DeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev) == CUDA_SUCCESS &&
            Driver::DeviceGetAttribute(&slot, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev) == CUDA_SUCCESS) {
            char bdf_path[128];
            snprintf(bdf_path, sizeof(bdf_path),
                     "/sys/bus/pci/devices/%04x:%02x:%02x.0/resource0", dom, bus, slot);
            if (TryMmapBar0(bdf_path, &bar0_, &bar0_fd_)) return true;

            char rpath[128];
            snprintf(rpath, sizeof(rpath),
                     "/sys/bus/pci/devices/%04x:%02x:%02x.0/resource", dom, bus, slot);
            if (FILE *fr = fopen(rpath, "r")) {
                unsigned long long start = 0;
                if (fscanf(fr, "0x%llx", &start) == 1 && start != 0) {
                    phys_bar0 = static_cast<uint64_t>(start);
                }
                fclose(fr);
            }
        }
    }

    DIR *dir = opendir("/sys/bus/pci/devices");
    if (dir) {
        struct dirent *ent = nullptr;
        while ((ent = readdir(dir)) != nullptr) {
            if (ent->d_name[0] == '.') continue;
            char vpath[512], cpath[512], rpath[512];
            snprintf(vpath, sizeof(vpath), "/sys/bus/pci/devices/%s/vendor", ent->d_name);
            FILE *fv = fopen(vpath, "r");
            if (!fv) continue;
            char vbuf[32] = {};
            const bool is_nv = (fgets(vbuf, sizeof(vbuf), fv) != nullptr) && (strstr(vbuf, "10de") != nullptr);
            fclose(fv);
            if (!is_nv) continue;

            snprintf(cpath, sizeof(cpath), "/sys/bus/pci/devices/%s/class", ent->d_name);
            FILE *fc = fopen(cpath, "r");
            if (!fc) continue;
            char cbuf[32] = {};
            const bool is_gpu = (fgets(cbuf, sizeof(cbuf), fc) != nullptr) && (strstr(cbuf, "0x03") != nullptr);
            fclose(fc);
            if (!is_gpu) continue;

            snprintf(rpath, sizeof(rpath), "/sys/bus/pci/devices/%s/resource0", ent->d_name);
            if (TryMmapBar0(rpath, &bar0_, &bar0_fd_)) {
                closedir(dir);
                return true;
            }
        }
        closedir(dir);
    }

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd >= 0) {
        void *p = mmap(nullptr, kBar0MapBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       static_cast<off_t>(phys_bar0));
        if (p != MAP_FAILED) {
            bar0_ = reinterpret_cast<volatile uint32_t *>(p);
            bar0_fd_ = fd;
            return true;
        }
        close(fd);
    }
    return false;
}

bool Sm120Arena::FireEvictPulse()
{
    if (!EnsureBar0Mapped()) return false;
    if (bar0_ != nullptr) {
        FireBar0Pulse(&bar0_[kBar0TrapRegOff >> 2], &bar0_[kBar0PtimerLoOff >> 2]);
        return true;
    }
    std::lock_guard<std::mutex> plock(proxy_mu_);
    int32_t st = -1;
    if (!ProxyRoundTrip(bar0proxy::kOpPulse, &st, nullptr)) return false;
    if (st != bar0proxy::kOk) {
        XWARN("sm120_arena: BAR0 proxy refused pulse (status %d%s)", st,
              st == bar0proxy::kRateLimited ? ", rate limited" : "");
        return false;
    }
    return true;
}

} // namespace xsched::cuda::sm120
