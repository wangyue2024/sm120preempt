#include "xsched/cuda/hal/level3/sm120_guardian.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#include <cuxtra/cuxtra.h>

#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/level2/mm.h"
#include "xsched/cuda/hal/level3/sm120_sass.h"
#include "xsched/utils/log.h"

namespace xsched::cuda::sm120
{

namespace
{

uint32_t AlignUp16(uint32_t v) { return (v + 15u) & ~15u; }

uint32_t ConfiguredLmemCapBytes(uint32_t static_aligned, bool has_calls)
{
    if (static_aligned == 0 && !has_calls) return 0;
    uint32_t def_cap = std::max(static_aligned, 4096u);
    if (const char *env = getenv("XSCHED_SM120_LMEM_CAP_BYTES")) {
        unsigned long long parsed = strtoull(env, nullptr, 0);
        if (parsed >= static_aligned && parsed <= (256ULL << 10)) {
            def_cap = AlignUp16(static_cast<uint32_t>(parsed));
        }
    }
    return def_cap;
}

bool BinaryHasDeviceCalls(const uint8_t *bin, size_t bytes)
{
    if (!bin || bytes < 16) return false;
    const size_t n_ins = bytes / 16;
    for (size_t i = 0; i < n_ins; ++i) {
        uint64_t w0 = 0;
        memcpy(&w0, bin + i * 16, sizeof(w0));
        const uint32_t op12 = static_cast<uint32_t>(w0 & 0xfffU);
        if (op12 == 0x943U || op12 == 0x944U) {
            return true;
        }
    }
    return false;
}

bool BinaryHasHwMbarrier(const uint8_t *bin, size_t bytes)
{
    if (!bin || bytes < 16) return false;
    const size_t n_ins = bytes / 16;
    for (size_t i = 0; i < n_ins; ++i) {
        uint64_t w0 = 0;
        memcpy(&w0, bin + i * 16, sizeof(w0));
        const uint32_t op9 = static_cast<uint32_t>(w0 & 0x1ffU);
        // SM120 hardware-managed SMEM mbarrier instructions (SYNCS.ARRIVE/PHASECHK/EXCH.64).
        if (op9 == 0x1a7U || op9 == 0x1b2U) {
            return true;
        }
    }
    return false;
}

std::vector<sass::Ins> BuildEntryGuardian()
{
    using namespace sass;
    constexpr size_t kSlots = kGuardianBytes / 16; // 80 slots
    std::vector<Ins> g;
    g.reserve(kSlots);

    // [0..3]: Fast bypass if c[0][0x170..0x177] == 0 (unmanaged / high-priority direct launches).
    g.push_back(ldc_u32(4, 0x170));                            // 0
    g.push_back(ldc_u32(5, 0x174));                            // 1
    g.push_back(lop3_or3_p0(4, 5, kRegZero));                  // 2: P0 = ((R4 | R5) != 0)
    const int br_unmanaged = static_cast<int>(g.size());
    g.push_back(nop());                                        // 3 -> orig_kernel (+0x500)

    // [4..11]: Full 3D linear cta_id -> R3; CtaEntry ptr -> R6:R7 = u_ctrl + 0x50 + 16*cta_id.
    g.push_back(s2r(0, kSrCtaIdZ));                            // 4
    g.push_back(ldc_u32(6, 0x374));                            // 5: gridDim.y
    g.push_back(s2r(7, kSrCtaIdY));                            // 6
    g.push_back(ldc_u32(8, 0x370));                            // 7: gridDim.x
    g.push_back(s2r(9, kSrCtaIdX));                            // 8
    g.push_back(imad_regs(3, 0, 6, 7));                        // 9
    g.push_back(imad_regs(3, 3, 8, 9));                        // 10: R3 = cta_id
    g.push_back(imad_wide_u32(6, 3, 16, 4));                   // 11: R6:R7 = u_ctrl + 16*cta_id

    // [12..14]: Route by is_restore_wave (u_ctrl + 0x1c) so ALL warps of a CTA take
    // restore_path unanimously even if Preempt #2 flips cmd 6->5 or 2->1 mid-wave.
    g.push_back(ld_e_32(1, 4, 0x1c));                          // 12: is_restore_wave
    g.push_back(isetp_ne_imm(1, 0));                           // 13
    const int br_restore_wave = static_cast<int>(g.size());    // 14
    g.push_back(nop());                                        // 14 -> @P0 restore_path

    // [15..23]: Normal/Evict multi-kernel FIFO guard (is_restore_wave == 0): if cmd != 0 &&
    //           active_idx != 0 && my_idx != active_idx, queued K2 exits without touching K1.
    g.push_back(ld_e_32(0, 4, 0x0));                           // 15: cmd
    g.push_back(isetp_ne_imm(0, 0));                           // 16: P0 = (cmd != 0)
    const int br_cmd0_skip = static_cast<int>(g.size());       // 17
    g.push_back(nop());                                        // 17 -> norm_gate
    g.push_back(ld_e_64(10, 4, 0x08));                         // 18: active_kernel_idx
    g.push_back(lop3_or3_p0(10, 11, kRegZero));                // 19: P0 = (active != 0)
    const int br_no_active = static_cast<int>(g.size());       // 20
    g.push_back(nop());                                        // 20 -> norm_gate
    g.push_back(ldc_64(8, 0x180));                             // 21: my_idx
    g.push_back(isetp_ne_u64_regs(8, 10));                     // 22: P0 = (my_idx != active)
    g.push_back(exit_inst(kP0));                               // 23: queued K2 exits safely!

    // [24..43]: norm_gate — 100% branchless intra-warp consensus before BAR.SYNC 0.
    // is_restore_wave is guaranteed 0 here (restore waves route exclusively to restore_path).
    const int norm_gate_idx = static_cast<int>(g.size());      // 24
    g.push_back(s2r(8, kSrTidX));                              // 24
    g.push_back(s2r(9, kSrTidY));                              // 25
    g.push_back(s2r(10, kSrTidZ));                             // 26
    g.push_back(lop3_or3_p0(8, 9, 10));                        // 27: P0 = (tid != 0)
    g.push_back(ld_e_32(0, 4, 0x0));                           // 28: R0 = cmd (coherent L2 read)
    g.push_back(isetp_ne_and_notp0(0, 0));                     // 29: P0 = (tid==0 && cmd!=0)
    g.push_back(mov_imm(2, 2));                                // 30: R2 = 2 (State C entry_dec)
    g.push_back(isetp_eq_imm(0, 1, kP1));                      // 31: P1 = (cmd == 1)
    g.push_back(mov_imm(2, 1, kP1));                           // 32: @P1 R2 = 1 (Mode 1 replay)
    g.push_back(st_e_32(6, 0x50, 2, kP0));                     // 33: @P0 cta[b].state = R2
    g.push_back(st_e_32(6, 0x54, 2, kP0));                     // 34: @P0 cta[b].entry_dec = R2
    g.push_back(ldc_64(8, 0x180));                             // 35: R8:R9 = my_idx
    g.push_back(st_e_64(4, 0x08, 8, kP0));                     // 36: @P0 active_kernel_idx = my_idx
    g.push_back(mov_imm(0, 1));                                // 37
    g.push_back(redg_add_u32(4, 0x14, 0, kP0));                // 38: @P0 drained_cnt++
    // Index 39 MUST match kGuardianBarOff (0x270).
    const int bar_idx = static_cast<int>(g.size());            // 39
    g.push_back(bar_sync0());                                  // 39: BAR.SYNC.DEFER_BLOCKING 0x0
    g.push_back(ld_e_32(2, 6, 0x54));                          // 40: cta[b].entry_dec
    g.push_back(isetp_ne_imm(2, 0));                           // 41
    g.push_back(exit_inst(kP0));                               // 42: early-exit if entry_dec!=0
    const int br_run_orig = static_cast<int>(g.size());        // 43
    g.push_back(nop());                                        // 43 -> orig_kernel (+0x500)

    // [44..58]: restore_path — multi-kernel isolation & unanimous state dispatch (is_restore_wave != 0).
    const int restore_path_idx = static_cast<int>(g.size());   // 44
    g.push_back(ld_e_64(10, 4, 0x08));                         // 44: active_kernel_idx
    g.push_back(mov_imm(2, 0));                                // 45
    g.push_back(mov_imm(3, 0));                                // 46
    g.push_back(isetp_ne_u64_regs(10, 2));                     // 47: P0 = (active != 0)
    const int br_rest_no_active = static_cast<int>(g.size());  // 48
    g.push_back(nop());                                        // 48 -> do_restore_dispatch
    g.push_back(ldc_64(8, 0x180));                             // 49: my_idx
    g.push_back(isetp_ne_u64_regs(8, 10));                     // 50: P0 = (my_idx != active)
    g.push_back(exit_inst(kP0));                               // 51: queued K2 exits while restore/evict active

    // [52..58]: do_restore_dispatch (my_idx == active_idx).
    const int rest_disp_idx = static_cast<int>(g.size());      // 52
    g.push_back(ld_e_32(3, 6, 0x50));                          // 52: R3 = cta[b].state (kept for replay_state_c)
    g.push_back(isetp_eq_imm(3, 0));                           // 53: state == 0 -> State A
    g.push_back(exit_inst(kP0));                               // 54: completed CTA exits!
    g.push_back(isetp_eq_imm(3, 5));                           // 55: state == 5 -> State B
    const int br_state_c = static_cast<int>(g.size());         // 56
    g.push_back(nop());                                        // 56 -> @!P0 replay_state_c
    g.push_back(bpt_trap1());                                  // 57: State B self-trap restore!
    g.push_back(exit_inst());                                  // 58: safety guard

    // [59..79]: replay_state_c — unanimous consensus for State C / Mode 1 CTAs.
    // Handles both normal Restore (cmd==2/6 -> (cmd&1)==0: clear state=0 & run orig)
    // and mid-restore Preempt #2 (cmd==1/5 -> (cmd&1)!=0: keep state, set entry_dec=2 & exit!).
    const int replay_c_idx = static_cast<int>(g.size());       // 59
    g.push_back(bar_sync0());                                  // 59: wait for all warps to finish reading state
    g.push_back(s2r(8, kSrTidX));                              // 60
    g.push_back(s2r(9, kSrTidY));                              // 61
    g.push_back(s2r(10, kSrTidZ));                             // 62
    g.push_back(lop3_or3_p0(8, 9, 10));                        // 63: P0 = (tid != 0)
    g.push_back(ld_e_32(0, 4, 0x0));                           // 64: R0 = cmd
    g.push_back(lop3_and_imm(0, 0, 1u));                       // 65: R0 = cmd & 1 (0=restore, 1=evict)
    g.push_back(isetp_eq_imm(0, 0, kP1));                      // 66: P1 = (restore)
    g.push_back(mov_imm(2, 2));                                // 67: R2 = 2 (evict entry_dec)
    g.push_back(mov_imm(2, 0, kP1));                           // 68: @P1 R2 = 0 (restore entry_dec)
    g.push_back(mov_imm(3, 0, kP1));                           // 69: @P1 R3 = 0 (restore clears state)
    g.push_back(st_e_32(6, 0x50, 3, kNotP0));                  // 70: tid==0 -> cta[b].state = R3
    g.push_back(st_e_32(6, 0x54, 2, kNotP0));                  // 71: tid==0 -> cta[b].entry_dec = R2
    g.push_back(isetp_ne_and_notp0(0, 0));                     // 72: P0 = (tid==0 && (cmd&1)!=0)
    g.push_back(mov_imm(1, 1));                                // 73
    g.push_back(redg_add_u32(4, 0x14, 1, kP0));                // 74: @P0 drained_cnt++
    g.push_back(nop());                                        // 75: pad
    const int replay_bar_idx = static_cast<int>(g.size());     // 76 (0x4c0)
    g.push_back(bar_sync0());                                  // 76: consensus barrier
    g.push_back(ld_e_32(2, 6, 0x54));                          // 77: cta[b].entry_dec
    g.push_back(isetp_ne_imm(2, 0));                           // 78
    g.push_back(exit_inst(kP0));                               // 79: unanimous exit on Preempt #2; else +0x500

    const int end_idx = static_cast<int>(g.size()); // 80 -> +0x500 (orig_kernel)
    if (bar_idx * 16 != static_cast<int>(kGuardianBarOff) ||
        restore_path_idx * 16 != static_cast<int>(kGuardianRestorePathOff) ||
        replay_c_idx * 16 != static_cast<int>(kGuardianReplayCIdxOff) ||
        replay_bar_idx * 16 != static_cast<int>(kGuardianReplayBarOff)) {
        XERRO("sm120_guardian: offset mismatch bar=%#x(exp %#x) rest=%#x(exp %#x) "
              "rep_c=%#x(exp %#x) rep_bar=%#x(exp %#x)",
              bar_idx * 16, kGuardianBarOff,
              restore_path_idx * 16, kGuardianRestorePathOff,
              replay_c_idx * 16, kGuardianReplayCIdxOff,
              replay_bar_idx * 16, kGuardianReplayBarOff);
    }

    g[br_unmanaged]      = bra_rel(end_idx - br_unmanaged - 1, kNotP0);
    g[br_restore_wave]   = bra_rel(restore_path_idx - br_restore_wave - 1, kP0);
    g[br_cmd0_skip]      = bra_rel(norm_gate_idx - br_cmd0_skip - 1, kNotP0);
    g[br_no_active]      = bra_rel(norm_gate_idx - br_no_active - 1, kNotP0);
    g[br_run_orig]       = bra_rel(end_idx - br_run_orig - 1, kPT);
    g[br_rest_no_active] = bra_rel(rest_disp_idx - br_rest_no_active - 1, kNotP0);
    g[br_state_c]        = bra_rel(replay_c_idx - br_state_c - 1, kNotP0);

    return g;
}

} // namespace

std::shared_ptr<Sm120GuardianRegistry> Sm120GuardianRegistry::Instance(CUcontext ctx)
{
    static std::mutex s_mtx;
    static std::map<CUcontext, std::shared_ptr<Sm120GuardianRegistry>> s_map;
    std::lock_guard<std::mutex> lock(s_mtx);
    auto it = s_map.find(ctx);
    if (it != s_map.end()) return it->second;

    CUdevice dev = 0;
    CUDA_ASSERT(Driver::CtxGetDevice(&dev));
    auto reg = std::make_shared<Sm120GuardianRegistry>(ctx, dev);
    s_map[ctx] = reg;
    return reg;
}

Sm120GuardianRegistry::Sm120GuardianRegistry(CUcontext ctx, CUdevice dev)
    : ctx_(ctx), dev_(dev)
{
    int lp = 0, hp = 0;
    if (Driver::CtxGetStreamPriorityRange(&lp, &hp) == CUDA_SUCCESS) {
        CUDA_ASSERT(Driver::StreamCreateWithPriority(&op_stream_, CU_STREAM_NON_BLOCKING, hp));
    } else {
        CUDA_ASSERT(Driver::StreamCreate(&op_stream_, CU_STREAM_NON_BLOCKING));
    }
    instr_mem_ = std::make_unique<InstrMemAllocator>(ctx_, dev_);
}

Sm120GuardianRegistry::~Sm120GuardianRegistry()
{
    CUcontext cur = nullptr;
    if (Driver::CtxGetCurrent(&cur) == CUDA_ERROR_DEINITIALIZED || cur != ctx_) {
        return;
    }
    for (const auto &kv : records_) {
        if (kv.second.kdesc_dev != 0) {
            Driver::MemFree_v2(kv.second.kdesc_dev);
        }
        if (kv.second.ep_orig != 0) {
            cuXtraSetEntryPoint(kv.first, kv.second.ep_orig);
        }
    }
    records_.clear();
    if (op_stream_ != nullptr) {
        Driver::StreamDestroy_v2(op_stream_);
        op_stream_ = nullptr;
    }
}

const Sm120KernelRecord *Sm120GuardianRegistry::Find(CUfunction func)
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = records_.find(func);
    return (it != records_.end()) ? &it->second : nullptr;
}

size_t Sm120GuardianRegistry::InstrumentedCount()
{
    std::lock_guard<std::mutex> lock(mu_);
    return records_.size();
}

const Sm120KernelRecord *Sm120GuardianRegistry::EnsureInstrumented(CUfunction func)
{
    if (!func) return nullptr;
    std::lock_guard<std::mutex> lock(mu_);
    auto it = records_.find(func);
    if (it != records_.end()) return &it->second;

    const void *orig_bin = nullptr;
    size_t orig_bin_sz = 0;
    cuXtraGetBinary(ctx_, func, &orig_bin, &orig_bin_sz, false);
    if (!orig_bin || orig_bin_sz == 0) {
        XWARN("sm120_guardian: cuXtraGetBinary returned empty image for func=%p",
              static_cast<void *>(func));
        return nullptr;
    }

    const CUdeviceptr ep_orig = cuXtraGetEntryPoint(func);

    int raw_regs = 16;
    if (Driver::FuncGetAttribute(&raw_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, func) != CUDA_SUCCESS) {
        raw_regs = static_cast<int>(cuXtraGetLocalRegsPerThread(func));
    }
    int local_bytes = 0;
    Driver::FuncGetAttribute(&local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func);
    int static_smem = 0;
    Driver::FuncGetAttribute(&static_smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func);

    const uint32_t raw_bars = static_cast<uint32_t>(cuXtraGetBarrierCnt(func));
    const uint32_t clamped_bars = std::max(raw_bars, 1u);
    const int clamped_regs = std::max(raw_regs, 16);

    const bool has_mbar = BinaryHasHwMbarrier(static_cast<const uint8_t *>(orig_bin), orig_bin_sz);
    if (has_mbar) {
        Sm120KernelRecord rec;
        rec.func              = func;
        rec.ep_orig           = ep_orig;
        rec.ep_inst           = ep_orig;
        rec.kdesc_dev         = 0;
        rec.desc.flags        = 4u;
        rec.raw_num_regs      = raw_regs;
        rec.clamped_regs      = raw_regs;
        rec.static_smem_bytes = static_smem;
        rec.has_hw_mbarrier   = true;
        auto [ins_it, inserted] = records_.emplace(func, rec);
        (void)inserted;
        XINFO("sm120_guardian: detected SM120 SYNCS/mbarrier in func=%p (regs=%d static_smem=%d) -> safe level-1 boundary",
              static_cast<void *>(func), raw_regs, static_smem);
        return &ins_it->second;
    }

    const bool has_calls = BinaryHasDeviceCalls(static_cast<const uint8_t *>(orig_bin), orig_bin_sz);
    const uint32_t lmem_static = AlignUp16(static_cast<uint32_t>(std::max(local_bytes, 0)));
    const uint32_t lmem_cap = ConfiguredLmemCapBytes(lmem_static, has_calls);

    const size_t total_alloc = (kGuardianBytes + orig_bin_sz + 127ULL) & ~127ULL;
    const CUdeviceptr ep_inst = instr_mem_->Alloc(total_alloc);
    if ((ep_inst & 0x7fULL) != 0) {
        XERRO("sm120_guardian: unaligned ep_inst %#llx",
              static_cast<unsigned long long>(ep_inst));
        return nullptr;
    }

    static const std::vector<sass::Ins> guardian_code = BuildEntryGuardian();
    cuXtraInstrMemcpyHtoD(ep_inst, guardian_code.data(), kGuardianBytes, op_stream_);
    cuXtraInstrMemcpyHtoD(ep_inst + kGuardianBytes, orig_bin, orig_bin_sz, op_stream_);
    Driver::StreamSynchronize(op_stream_);

    cuXtraSetEntryPoint(func, ep_inst);
    if (static_cast<size_t>(clamped_regs) > cuXtraGetLocalRegsPerThread(func)) {
        cuXtraSetLocalRegsPerThread(func, static_cast<size_t>(clamped_regs));
    }
    if (static_cast<size_t>(clamped_bars) > cuXtraGetBarrierCnt(func)) {
        cuXtraSetBarrierCnt(func, static_cast<size_t>(clamped_bars));
    }
    cuXtraInvalInstrCache(ctx_);

    Sm120KernelDesc desc{};
    desc.ep_inst_va       = static_cast<uint64_t>(ep_inst);
    desc.guardian_bar_off = kGuardianBarOff;
    desc.lmem_static_bytes = lmem_static;
    desc.lmem_cap_bytes   = lmem_cap;
    desc.bar_cnt          = clamped_bars;
    desc.flags            = ((lmem_cap > 0) ? 1u : 0u) | (has_calls ? 2u : 0u);
    desc.magic            = kKernelDescMagic;

    CUdeviceptr kdesc_dev = 0;
    CUDA_ASSERT(Driver::MemAlloc_v2(&kdesc_dev, sizeof(desc)));
    CUDA_ASSERT(Driver::MemcpyHtoDAsync_v2(kdesc_dev, &desc, sizeof(desc), op_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));

    Sm120KernelRecord rec;
    rec.func              = func;
    rec.ep_orig           = ep_orig;
    rec.ep_inst           = ep_inst;
    rec.kdesc_dev         = kdesc_dev;
    rec.desc              = desc;
    rec.raw_num_regs      = raw_regs;
    rec.clamped_regs      = clamped_regs;
    rec.static_smem_bytes = static_smem;

    auto [ins_it, inserted] = records_.emplace(func, rec);
    (void)inserted;
    XINFO("sm120_guardian: instrumented func=%p orig_ep=%#llx -> ep_inst=%#llx "
          "regs=%d->%d bars=%u lmem=%u(cap=%u,calls=%d) static_smem=%d",
          static_cast<void *>(func),
          static_cast<unsigned long long>(ep_orig),
          static_cast<unsigned long long>(ep_inst),
          raw_regs, clamped_regs, clamped_bars, lmem_static, lmem_cap,
          has_calls ? 1 : 0, static_smem);
    return &ins_it->second;
}

} // namespace xsched::cuda::sm120
