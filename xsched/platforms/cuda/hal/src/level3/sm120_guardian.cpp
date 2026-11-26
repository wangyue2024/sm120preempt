#include "xsched/cuda/hal/level3/sm120_guardian.h"

#include <algorithm>
#include <cstdio>
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

// Per-thread LMEM bytes the Mode 5 trap frame must be able to save (DEF-23, 2026-11-25):
//  - no device calls: the local frame is fixed -> exactly the static local size;
//  - device calls: callee frames live in the per-thread stack, which the hardware bounds by the
//    context's CU_LIMIT_STACK_SIZE -> static local + stack limit (a deeper stack faults by itself).
// The previous fixed max(static, 4096 B) reserved 4 KiB per thread for every such kernel (e.g.
// 231 MiB arena for LayerNorm on RTX 5060). XSCHED_SM120_LMEM_CAP_BYTES still overrides.
uint32_t ConfiguredLmemCapBytes(uint32_t static_aligned, bool has_calls, size_t stack_limit)
{
    if (static_aligned == 0 && !has_calls) return 0;
    const size_t stack = has_calls ? std::min<size_t>(stack_limit, 256u << 10) : 0;
    uint32_t def_cap = AlignUp16(static_cast<uint32_t>(std::min<size_t>(static_aligned + stack, 256u << 10)));
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
        // 0x943/0x944: CALL.REL(.NOINC); 0x343: CALL.ABS.NOINC Rx (DEF-22: previously missed, e.g.
        // PyTorch reduce_kernel / direct_copy / gather kernels -> calls=0, no stack save area).
        static const bool detect_abs = []() {
            const char *e = getenv("XSCHED_SM120_DETECT_CALL_ABS");
            return !(e != nullptr && e[0] == '0');
        }();
        // DEF-24: 0x344 = CALL.REL.NOINC Rx (indirect call through a function pointer); 0x950 =
        // RET (callee code is part of this image, so something calls it).
        if (op12 == 0x943U || op12 == 0x944U || op12 == 0x344U || op12 == 0x950U ||
            (detect_abs && op12 == 0x343U)) {
            return true;
        }
    }
    return false;
}

bool BinaryIsLoopFree(const uint8_t *bin, size_t bytes)
{
    // DEF-17 router: a kernel image without backward branches (and without indirect branches,
    // whose targets are unknown) executes each instruction at most once per thread, so its CTAs
    // are short and bounded. The trailing "BRA self" padding after EXIT (offset -16) is ignored.
    if (!bin || bytes < 16) return false;
    const size_t n_ins = bytes / 16;
    for (size_t i = 0; i < n_ins; ++i) {
        uint64_t w0 = 0, w1 = 0;
        memcpy(&w0, bin + i * 16, sizeof(w0));
        memcpy(&w1, bin + i * 16 + 8, sizeof(w1));
        const uint32_t op12 = static_cast<uint32_t>(w0 & 0xfffU);
        if (op12 == 0x949U) return false;           // BRX (indirect branch)
        if (op12 != 0x947U) continue;               // BRA / BRA.U
        // Relative offset in 4-byte units, relative to the next instruction:
        // bits [16,24) of w0 = low 8 bits, bits [34,74) of the 128-bit word = high 40 bits.
        const uint64_t lo = (w0 >> 16) & 0xffULL;
        const uint64_t hi = ((w0 >> 34) | (w1 << 30)) & ((1ULL << 40) - 1);
        int64_t units = static_cast<int64_t>(lo | (hi << 8));
        if (units & (1LL << 47)) units -= (1LL << 48);
        if (units * 4 < -16) return false;          // backward branch -> loop
    }
    return true;
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
    // Entry Guardian variants (DEF-17, 2026-11-25):
    //  - default : original fully serialized issue (every LDC/S2R/LD waits before the next one), but
    //              without the redundant slot-28 cmd re-read (pure ordering argument, no async window).
    //  - XSCHED_SM120_GUARDIAN_V1=1 : exact original guardian (with the slot-28 re-read).
    //  - XSCHED_SM120_GUARDIAN_V2=1 : EXPERIMENTAL, additionally issues the control-header loads and the
    //    S2R/LDC chains asynchronously (scoreboards SB2..SB5). 20-35% lower Lv3 overhead on short-CTA
    //    kernels, but LLM-08 (two Lv3 queues + HP stream) failed 11/61 vs 0/78 with serialized issue on
    //    RTX 5080; suspected: traps delivered while guardian-issued ops are outstanding. Not default.
    //    Sub-switches for bisecting: XSCHED_SM120_GUARDIAN_V2_SERIAL_HDR=1 / _SERIAL_SREG=1.
    auto env_on = [](const char *k) { const char *e = getenv(k); return e != nullptr && e[0] == '1'; };
    const bool exact_v1 = env_on("XSCHED_SM120_GUARDIAN_V1");
    const bool exp_v2 = !exact_v1 && env_on("XSCHED_SM120_GUARDIAN_V2");
    const bool v2_reload = exact_v1;
    const bool v2_par_hdr = exp_v2 && !env_on("XSCHED_SM120_GUARDIAN_V2_SERIAL_HDR");
    const bool v2_par_sreg = exp_v2 && !env_on("XSCHED_SM120_GUARDIAN_V2_SERIAL_SREG");
    // v2 helpers (DEF-17): issue a variable-latency producer (LDC/S2R/LD) on scoreboard `sb` without
    // waiting (original stall/yield kept), and add scoreboard waits to its first consumer.
    auto issue_on = [v2_par_sreg](Ins ins, uint32_t sb) -> Ins {
        if (!v2_par_sreg) return ins;
        const uint64_t ctl = (ins.w1 >> 41) & 0x1fffffULL;
        return with_ctl(ins, static_cast<uint32_t>(ctl & 0xf), static_cast<uint32_t>((ctl >> 4) & 1), sb, 7, 0);
    };
    auto wait_on = [v2_par_sreg](Ins ins, uint32_t mask) -> Ins { return v2_par_sreg ? with_extra_wait(ins, mask) : ins; };
    constexpr uint32_t kSb2 = 1u << 2, kSb3 = 1u << 3, kSb4 = 1u << 4, kSb5 = 1u << 5;

    // [0..3]: Fast bypass if c[0][0x170..0x177] == 0 (unmanaged / high-priority direct launches).
    g.push_back(issue_on(ldc_u32(4, 0x170), 2));               // 0
    g.push_back(issue_on(ldc_u32(5, 0x174), 3));               // 1
    g.push_back(wait_on(lop3_or3_p0(4, 5, kRegZero), kSb2 | kSb3)); // 2: P0 = ((R4 | R5) != 0)
    const int br_unmanaged = static_cast<int>(g.size());
    g.push_back(nop());                                        // 3 -> orig_kernel (+0x500)

    // [4..11]: Full 3D linear cta_id -> R3; CtaEntry ptr -> R6:R7 = u_ctrl + 0x50 + 16*cta_id.
    g.push_back(issue_on(s2r(0, kSrCtaIdZ), 2));               // 4
    g.push_back(issue_on(ldc_u32(6, 0x374), 3));               // 5: gridDim.y
    g.push_back(issue_on(s2r(7, kSrCtaIdY), 4));               // 6
    g.push_back(issue_on(ldc_u32(8, 0x370), 5));               // 7: gridDim.x
    g.push_back(s2r(9, kSrCtaIdX));                            // 8 (SB0, as v1)
    g.push_back(wait_on(imad_regs(3, 0, 6, 7), kSb2 | kSb3 | kSb4)); // 9
    g.push_back(wait_on(imad_regs(3, 3, 8, 9), kSb5));         // 10: R3 = cta_id
    g.push_back(imad_wide_u32(6, 3, 16, 4));                   // 11: R6:R7 = u_ctrl + 16*cta_id

    // [12..17] experimental v2: issue the two control-header loads back-to-back on independent
    // scoreboards (SB2 / SB3) and wait once; default / v1 keep the original serialized sequence.
    // Route by is_restore_wave (u_ctrl + 0x1c) so ALL warps of a CTA take restore_path
    // unanimously even if Preempt #2 flips cmd 6->5 or 2->1 mid-wave.
    int br_restore_wave = -1;
    int br_cmd0_skip = -1;
    if (v2_par_hdr) {
        g.push_back(with_ctl(ld_e_32(1, 4, 0x1c), 1, 1, 2, 7, 0)); // 12: is_restore_wave -> SB2
        g.push_back(with_ctl(ld_e_32(0, 4, 0x0), 1, 1, 3, 7, 0));  // 13: cmd -> SB3
        g.push_back(with_extra_wait(isetp_ne_imm(1, 0), (1u << 2) | (1u << 3))); // 14: + wait SB2|SB3
        br_restore_wave = static_cast<int>(g.size());           // 15
        g.push_back(nop());                                     // 15 -> @P0 restore_path
        g.push_back(isetp_ne_imm(0, 0));                        // 16: P0 = (cmd != 0)
        br_cmd0_skip = static_cast<int>(g.size());              // 17
        g.push_back(nop());                                     // 17 -> norm_gate
    } else {
        g.push_back(ld_e_32(1, 4, 0x1c));                       // 12: is_restore_wave
        g.push_back(isetp_ne_imm(1, 0));                        // 13
        br_restore_wave = static_cast<int>(g.size());           // 14
        g.push_back(nop());                                     // 14 -> @P0 restore_path
        // [15..23]: Normal/Evict multi-kernel FIFO guard (is_restore_wave == 0): if cmd != 0 &&
        //           active_idx != 0 && my_idx != active_idx, queued K2 exits without touching K1.
        g.push_back(ld_e_32(0, 4, 0x0));                        // 15: cmd
        g.push_back(isetp_ne_imm(0, 0));                        // 16: P0 = (cmd != 0)
        br_cmd0_skip = static_cast<int>(g.size());              // 17
        g.push_back(nop());                                     // 17 -> norm_gate
    }
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
    g.push_back(issue_on(s2r(8, kSrTidX), 2));                 // 24
    g.push_back(issue_on(s2r(9, kSrTidY), 3));                 // 25
    g.push_back(issue_on(s2r(10, kSrTidZ), 4));                // 26
    g.push_back(wait_on(lop3_or3_p0(8, 9, 10), kSb2 | kSb3 | kSb4)); // 27: P0 = (tid != 0)
    // 28: exact v1 re-reads cmd; default/v2 reuse R0 = cmd from the earlier read (R0 is untouched on
    // every path to here).
    // Safe: a CTA that observed cmd == 0 was resident before the host published cmd, and the BAR0
    // pulse is fired only after that write completes, so such a CTA is either finished or trapped.
    g.push_back(v2_reload ? ld_e_32(0, 4, 0x0) : nop());       // 28
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
    const bool barred = GuardianBarRedEnabled();
    const int bar_idx = static_cast<int>(g.size());            // 39
    int br_run_orig = -1;
    if (barred) {
        // OR over P0 = (tid==0 && cmd!=0): entry_dec can only have been written by this Guardian
        // if the result is true (the trap handler never completes this barrier: trapped warps
        // decide from entry_dec in Cave 0). If B2R runs after a State B restore its value is
        // undefined, but both outcomes are correct there (entry_dec == 0 => run).
        g.push_back(bar_red_or0_p0());                         // 39: BAR.RED.OR 0x0, P0
        g.push_back(b2r_result_p(kP1));                        // 40: P1 = OR result (SB1)
        g.push_back(with_extra_wait(ld_e_32(2, 6, 0x54, kP1), 1u << 1)); // 41: @P1 entry_dec
        g.push_back(isetp_ne_and_pred(2, 0, kP0, kP1));       // 42: P0 = entry_dec!=0 && P1
        g.push_back(exit_inst(kP0));                           // 43: early-exit if drained
        br_run_orig = static_cast<int>(g.size());              // 44
        g.push_back(nop());                                    // 44 -> orig_kernel (+0x500)
    } else {
        g.push_back(bar_sync0());                              // 39: BAR.SYNC.DEFER_BLOCKING 0x0
        g.push_back(ld_e_32(2, 6, 0x54));                      // 40: cta[b].entry_dec
        g.push_back(isetp_ne_imm(2, 0));                       // 41
        g.push_back(exit_inst(kP0));                           // 42: early-exit if entry_dec!=0
        br_run_orig = static_cast<int>(g.size());              // 43
        g.push_back(nop());                                    // 43 -> orig_kernel (+0x500)
    }

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
    if (!barred) g.push_back(nop());                           // 75: pad (dropped in the BAR.RED layout)
    const int replay_bar_idx = static_cast<int>(g.size());     // 76 (0x4c0)
    g.push_back(bar_sync0());                                  // 76: consensus barrier
    g.push_back(ld_e_32(2, 6, 0x54));                          // 77: cta[b].entry_dec
    g.push_back(isetp_ne_imm(2, 0));                           // 78
    g.push_back(exit_inst(kP0));                               // 79: unanimous exit on Preempt #2; else +0x500

    const int end_idx = static_cast<int>(g.size()); // 80 -> +0x500 (orig_kernel)
    if (bar_idx * 16 != static_cast<int>(kGuardianBarOff) ||
        restore_path_idx * 16 != static_cast<int>(GuardianRestorePathOff()) ||
        replay_c_idx * 16 != static_cast<int>(GuardianReplayCIdxOff()) ||
        replay_bar_idx * 16 != static_cast<int>(kGuardianReplayBarOff)) {
        XERRO("sm120_guardian: offset mismatch bar=%#x(exp %#x) rest=%#x(exp %#x) "
              "rep_c=%#x(exp %#x) rep_bar=%#x(exp %#x)",
              bar_idx * 16, kGuardianBarOff,
              restore_path_idx * 16, GuardianRestorePathOff(),
              replay_c_idx * 16, GuardianReplayCIdxOff(),
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

bool GuardianBarRedEnabled()
{
    // Default on (Phase 3.3): XSCHED_SM120_GUARDIAN_BARRED=0 restores the BAR.SYNC norm_gate;
    // XSCHED_SM120_GUARDIAN_V1=1 (exact original Guardian) also disables it.
    static const bool on = []() {
        const char *v1 = getenv("XSCHED_SM120_GUARDIAN_V1");
        if (v1 != nullptr && v1[0] == '1') return false;
        const char *e = getenv("XSCHED_SM120_GUARDIAN_BARRED");
        return !(e != nullptr && e[0] == '0');
    }();
    return on;
}

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
    // DEF-20 fix (2026-11-25): round the register allocation up to an EVEN count so the trap engine's
    // 64-bit register-pair cascade (Cave 2/5) never straddles the allocation boundary. 255 is the
    // per-thread maximum and is left as is (known residual: R254 of a 255-register kernel is not saved).
    int clamped_regs = std::max(raw_regs, 16);
    if (clamped_regs < 255) clamped_regs = (clamped_regs + 1) & ~1;

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
    const bool loop_free = !has_calls &&
        BinaryIsLoopFree(static_cast<const uint8_t *>(orig_bin), orig_bin_sz);
    const uint32_t lmem_static = AlignUp16(static_cast<uint32_t>(std::max(local_bytes, 0)));
    size_t stack_limit = 1024;
    if (Driver::CtxGetLimit(&stack_limit, CU_LIMIT_STACK_SIZE) != CUDA_SUCCESS) stack_limit = 4096;
    const uint32_t lmem_cap = ConfiguredLmemCapBytes(lmem_static, has_calls, stack_limit);

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
    rec.has_calls         = has_calls;
    rec.loop_free         = loop_free;

    auto [ins_it, inserted] = records_.emplace(func, rec);
    (void)inserted;
    const char *fname = nullptr;
    if (Driver::FuncGetName(&fname, func) != CUDA_SUCCESS) fname = nullptr;
    if (const char *dump_dir = getenv("XSCHED_SM120_DUMP_DIR")) {
        // Debug aid: dump the original SASS image (disassemble with `nvdisasm -b SM120 <file>`).
        char path[4096];
        snprintf(path, sizeof(path), "%s/k_%p.bin", dump_dir, static_cast<void *>(func));
        if (FILE *fp = fopen(path, "wb")) {
            fwrite(orig_bin, 1, orig_bin_sz, fp);
            fclose(fp);
        }
        snprintf(path, sizeof(path), "%s/k_%p.name", dump_dir, static_cast<void *>(func));
        if (FILE *fp = fopen(path, "w")) {
            fprintf(fp, "%s\n", fname ? fname : "?");
            fclose(fp);
        }
        snprintf(path, sizeof(path), "%s/entry_guardian.bin", dump_dir);
        if (FILE *fp = fopen(path, "wb")) {
            fwrite(guardian_code.data(), 1, kGuardianBytes, fp);
            fclose(fp);
        }
    }
    XINFO("sm120_guardian: instrumented func=%p orig_ep=%#llx -> ep_inst=%#llx "
          "regs=%d->%d bars=%u lmem=%u(cap=%u,calls=%d) loop_free=%d static_smem=%d bin=%zuB name=%.120s",
          static_cast<void *>(func),
          static_cast<unsigned long long>(ep_orig),
          static_cast<unsigned long long>(ep_inst),
          raw_regs, clamped_regs, clamped_bars, lmem_static, lmem_cap,
          has_calls ? 1 : 0, loop_free ? 1 : 0, static_smem, orig_bin_sz, fname ? fname : "?");
    return &ins_it->second;
}

} // namespace xsched::cuda::sm120
