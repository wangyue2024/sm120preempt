#include "xsched/cuda/hal/level3/sm120_trap_engine.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <vector>

#include <cuxtra/cuxtra.h>

#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/level3/sm120_sass.h"
#include "xsched/utils/env.h"
#include "xsched/utils/log.h"

namespace xsched::cuda::sm120
{

namespace
{

using namespace sass;

uint32_t AlignUp(uint32_t v, uint32_t align)
{
    return (v + align - 1u) & ~(align - 1u);
}

size_t AlignUpSize(size_t v, size_t align)
{
    return (v + align - 1ULL) & ~(align - 1ULL);
}

// Compute 3D tid -> tid_reg, 3D threads_per_cta -> tpc_reg, 3D cta_id -> cta_reg.
// Clobbers: R0, R1, R2, R3, R6, R7 (preserves R4:R5 and R12).
void EmitCompute3DCoords(std::vector<Ins> &code, int cta_reg, int tid_reg, int tpc_reg)
{
    code.push_back(s2r(0, kSrTidX));
    code.push_back(s2r(1, kSrTidY));
    code.push_back(s2r(2, kSrTidZ));
    code.push_back(ldc_u32(6, 0x360)); // blockDim.x
    code.push_back(ldc_u32(7, 0x364)); // blockDim.y
    code.push_back(ldc_u32(3, 0x368)); // blockDim.z
    code.push_back(imad_regs(tid_reg, 2, 7, 1));
    code.push_back(imad_regs(tid_reg, tid_reg, 6, 0));
    code.push_back(imad_regs(tpc_reg, 6, 7, kRegZero));
    code.push_back(imad_regs(tpc_reg, tpc_reg, 3, kRegZero));

    code.push_back(s2r(0, kSrCtaIdZ));
    code.push_back(ldc_u32(6, 0x374)); // gridDim.y
    code.push_back(s2r(7, kSrCtaIdY));
    code.push_back(ldc_u32(2, 0x370)); // gridDim.x
    code.push_back(s2r(3, kSrCtaIdX));
    code.push_back(imad_regs(cta_reg, 0, 6, 7));
    code.push_back(imad_regs(cta_reg, cta_reg, 2, 3));
}

// Compute slot_base = arena_va + slot_id * slot_stride_bytes into dst_even:dst_even+1.
// Reads R4:R5 = u_control_va and [0xffff7c] = slot_id. Clobbers tmp0, tmp1.
void EmitLoadSlotBase(std::vector<Ins> &code, int dst_even, int tmp_slot, int tmp_stride)
{
    code.push_back(ldl_32(tmp_slot, kRegZero, 0xffff7c));
    code.push_back(ld_e_64(dst_even, 4, offsetof(Sm120ControlHeader, arena_va)));
    code.push_back(ld_e_32(tmp_stride, 4, offsetof(Sm120ControlHeader, slot_stride_bytes)));
    code.push_back(imad_wide_u32_regs(dst_even, tmp_slot, tmp_stride, dst_even));
}

// Compute thread_base (or lmem_base) into dst_even:dst_even+1.
void EmitComputePerThreadPtr(std::vector<Ins> &code,
                             int dst_even,
                             uint32_t area_off_field,
                             uint32_t stride_field,
                             int tmp0,
                             int tmp1)
{
    EmitLoadSlotBase(code, dst_even, tmp0, tmp1);
    code.push_back(ld_e_32(tmp0, 4, area_off_field));
    code.push_back(imad_wide_u32(dst_even, tmp0, 1, dst_even));
    code.push_back(ldl_32(tmp0, kRegZero, 0xffff74)); // tid
    code.push_back(ld_e_32(tmp1, 4, stride_field));
    code.push_back(imad_wide_u32_regs(dst_even, tmp0, tmp1, dst_even));
}

// Shared SMEM save/restore loop (`is_save = true` copies SMEM->u_arena, `false` u_arena->SMEM).
void EmitDynamicSmemLoop(std::vector<Ins> &code, bool is_save)
{
    code.push_back(s2r(8, kSrSmemSz));
    // Guard against SR_SMEMSZ <= 0x400 (e.g. 0-SMEM kernels where SR_SMEMSZ == 0 or 0x400):
    // subtracting 0x400 unconditionally would wrap unsigned to 0xFFFFFC00 -> illegal SMEM fault.
    code.push_back(isetp_ge_u32_imm(8, 0x401u));
    const int br_skip = static_cast<int>(code.size());
    code.push_back(nop());
    code.push_back(iadd_imm(8, 8, static_cast<uint32_t>(-0x400)));

    code.push_back(ldl_32(9, kRegZero, 0xffff74));  // tid
    code.push_back(ldl_32(10, kRegZero, 0xffff78)); // threads_per_cta

    int br_partial_skip = -1;
    if (is_save) {
        // full_mask = ((1u << (((tpc + 31) >> 5) - 1)) << 1) - 1
        code.push_back(iadd_imm(0, 10, 31u));
        code.push_back(shf_r_u32(0, 0, 5));
        code.push_back(iadd_imm(0, 0, static_cast<uint32_t>(-1)));
        code.push_back(shf_l_one(0, 0));
        code.push_back(lea_shl(0, 0, 1));
        code.push_back(iadd_imm(0, 0, static_cast<uint32_t>(-1)));

        code.push_back(ldl_32(1, kRegZero, 0xffff70)); // cta_id
        code.push_back(imad_wide_u32(6, 1, 16, 4));
        code.push_back(ld_e_32(1, 6, 0x5c));            // cta[b].warp_mask
        code.push_back(isetp_eq_u32_regs(1, 0));
        const int br_full_cta = static_cast<int>(code.size());
        code.push_back(nop());

        // Partial CTA fallback (some warps exited): lowest surviving warp copies full SMEM coalesced.
        code.push_back(iadd3_sub_regs(2, kRegZero, 1));
        code.push_back(lop3_and_regs(2, 1, 2));         // R2 = lowest live warp bit
        code.push_back(shf_r_u32(3, 9, 5));             // R3 = warp_id
        code.push_back(shf_l_one(3, 3));                // R3 = 1u << warp_id
        code.push_back(isetp_eq_u32_regs(3, 2));
        br_partial_skip = static_cast<int>(code.size());
        code.push_back(nop());
        code.push_back(lop3_and_imm(9, 9, 31u));        // lane_id
        code.push_back(mov_imm(10, 32u));               // 32-lane stride

        const int full_cta_idx = static_cast<int>(code.size());
        code[br_full_cta] = bra_rel(full_cta_idx - br_full_cta - 1, kP0);
    }

    code.push_back(lea_shl(9, 9, 2));               // R9 = init_tid * 4
    code.push_back(lea_shl(10, 10, 2));             // R10 = stride * 4
    EmitLoadSlotBase(code, 6, 0, 1);                // preserve R8, R9, R10
    code.push_back(ld_e_32(1, 4, offsetof(Sm120ControlHeader, smem_area_off)));
    code.push_back(imad_wide_u32(6, 1, 1, 6));
    code.push_back(imad_wide_u32(6, 9, 1, 6));

    const int loop_head = static_cast<int>(code.size());
    code.push_back(isetp_ge_u32_regs(9, 8));
    const int br_done = static_cast<int>(code.size());
    code.push_back(nop());
    if (is_save) {
        code.push_back(lds_u32(0, 9, 0x400));
        code.push_back(st_e_32(6, 0x0, 0));
    } else {
        code.push_back(ld_e_32(0, 6, 0x0));
        code.push_back(sts_u32(9, 0x400, 0));
    }
    code.push_back(imad_add(9, 9, 10));
    code.push_back(imad_wide_u32(6, 10, 1, 6));
    const int back_idx = static_cast<int>(code.size());
    code.push_back(bra_rel(loop_head - back_idx - 1));

    const int done_idx = static_cast<int>(code.size());
    code[br_skip] = bra_rel(done_idx - br_skip - 1, kNotP0);
    code[br_done] = bra_rel(done_idx - br_done - 1, kP0);
    if (br_partial_skip >= 0) {
        code[br_partial_skip] = bra_rel(done_idx - br_partial_skip - 1, kNotP0);
    }
}

// Cave 2 (is_save=true) and Cave 5 (is_save=false): Static SR_REGALLOC + BRX R14..R252 cascade.
std::vector<Ins> BuildRegCascadeCave(bool is_save, uint64_t cave_va, uint64_t return_va)
{
    std::vector<Ins> c;
    c.reserve(136);

    // Header: compute r_top = min(SR_REGALLOC - 1, 252). Skip if < 14.
    // DEF-20 (2026-11-25): SR_REGALLOC reports EXACTLY the kernel's register count
    // (== CU_FUNC_ATTRIBUTE_NUM_REGS, measured on GB206/GB203 with a patched S2R probe); it does
    // NOT include two reserved registers. The previous "SR_REGALLOC - 3" silently skipped the top
    // 2-3 live registers of every kernel (R14/R15 entirely for 14..16-register kernels), causing
    // Mode 5 restore crashes (700/717) and silent corruption (DEF-14/15/16).
    c.push_back(s2r(4, kSrRegAlloc));
    c.push_back(iadd_imm(4, 4, static_cast<uint32_t>(-1)));
    c.push_back(isetp_ge_u32_imm(4, 14));
    const int br_too_low = static_cast<int>(c.size());
    c.push_back(nop());

    // Handle R253 explicitly only when SR_REGALLOC >= 256 (r_top >= 253),
    // avoiding trap-reserved R253 if hardware ever reports SR_REGALLOC == 255.
    c.push_back(isetp_ge_u32_imm(4, 253));
    if (is_save) {
        c.push_back(st_e_32(6, 0x434, 253, kP0));
    } else {
        c.push_back(ld_e_32(253, 6, 0x434, kP0));
    }

    c.push_back(isetp_ge_u32_imm(4, 252));
    c.push_back(mov_imm(4, 252, kP0));
    c.push_back(with_pred(lop3_and_imm(4, 4, 0xfffffffeU), kNotP0));
    c.push_back(iadd3_sub_from_imm(10, 4, 252u));
    c.push_back(lea_shl(10, 10, 3)); // ((252 - r_start) / 2) * 16 bytes
    c.push_back(mov_imm(11, 0));     // SM120 BRX R10 consumes 64-bit offset pair R10:R11
    c.push_back(brx(10));

    // Index 0: R252 at +0x430 (32-bit); Index 1..119: R250..R14 paired 64-bit.
    if (is_save) {
        c.push_back(st_e_32(6, 0x430, 252));
        for (int r = 250; r >= 14; r -= 2) {
            c.push_back(st_e_64(6, static_cast<uint32_t>(0x78 + (r - 14) * 4), r));
        }
    } else {
        c.push_back(ld_e_32(252, 6, 0x430));
        for (int r = 250; r >= 14; r -= 2) {
            c.push_back(ld_e_64(r, 6, static_cast<uint32_t>(0x78 + (r - 14) * 4)));
        }
    }

    const int done_idx = static_cast<int>(c.size());
    c[br_too_low] = bra_rel(done_idx - br_too_low - 1, kNotP0);

    // Reload R4:R5 = u_control_va so R4:R5 ABI stays unbroken for Cave 0/1!
    c.push_back(ldc_u32(4, 0x170));
    c.push_back(ldc_u32(5, 0x174));
    c.push_back(bra_to(cave_va + c.size() * 16, return_va));
    return c;
}

// Cave 3: Named Barriers 0..15 + B2R.WARP Save+Clear, Mode 1/State C Clear+Exit, and Restore.
// Runs on Thread 0 / Elected Lane 0 so 1-thread blocks through 1024-thread blocks all work uniformly.
struct Cave3Layout
{
    std::vector<Ins> code;
    size_t save_idx = 0;
    size_t clear_exit_idx = 0;
    size_t rest_idx = 0;
};

Cave3Layout BuildNamedBarriersCave(uint64_t cave_va,
                                   uint64_t ev_ret_va,
                                   uint64_t rest_ret_va,
                                   uint64_t clean_exit_va)
{
    Cave3Layout L;
    auto &c = L.code;
    c.reserve(112);

    // --- Section 0: Save B2R.WARP + B2R[0..15] and Clear Barriers + R2B.WARP ---
    L.save_idx = 0;
    EmitLoadSlotBase(c, 6, 9, 10);
    c.push_back(ldl_32(8, kRegZero, 0xffff74)); // tid
    c.push_back(shf_r_u32(10, 8, 5));          // warp_id
    c.push_back(imad_wide_u32(2, 10, 0x160, 6));
    c.push_back(b2r_warp(11));
    c.push_back(elect_p0());
    c.push_back(st_e_32(2, 0x144, 11, kP0));   // warp_base + 0x104

    // Wait for all warps in CTA before dynamic CTA leader captures and clears CTA barriers 0..15.
    c.push_back(bar_sync_trap());
    c.push_back(ldl_32(8, kRegZero, 0xffff6c)); // is_leader (dynamic CTA leader)
    c.push_back(isetp_ne_imm(8, 0, kP0));
    for (int grp = 0; grp < 4; ++grp) {
        for (int off = 0; off < 4; ++off) {
            c.push_back(b2r_bar(kP0, 8 + off, grp * 4 + off));
        }
        c.push_back(st_e_128(6, static_cast<uint32_t>(grp * 16), 8, kP0));
    }
    for (int b = 0; b < 16; ++b) {
        c.push_back(r2b_bar(kP0, b, kRegZero));
    }
    c.push_back(r2b_warp(kPT, kRegZero));
    c.push_back(bar_sync_trap());
    c.push_back(bra_to(cave_va + c.size() * 16, ev_ret_va));

    // --- Section 1: Clear Barriers 0..15 + R2B.WARP and jump to clean_exit (Mode 1 / State C / A) ---
    // Never execute bar_sync_trap() here: sibling warps may have already exited at Guardian Slot 49 or 61.
    L.clear_exit_idx = c.size();
    c.push_back(elect_p0());
    for (int b = 0; b < 16; ++b) {
        c.push_back(r2b_bar(kP0, b, kRegZero));
    }
    c.push_back(r2b_warp(kPT, kRegZero));
    c.push_back(bra_to(cave_va + c.size() * 16, clean_exit_va));

    // --- Section 2: Restore Named Barriers 0..15 (post-sync R2B.WARP runs in Cave 1!) ---
    L.rest_idx = c.size();
    EmitLoadSlotBase(c, 6, 9, 10);
    c.push_back(ldl_32(8, kRegZero, 0xffff74)); // tid
    c.push_back(isetp_eq_imm(8, 0, kP0));
    for (int grp = 0; grp < 4; ++grp) {
        c.push_back(ld_e_128(8, 6, static_cast<uint32_t>(grp * 16), kP0));
        for (int off = 0; off < 4; ++off) {
            c.push_back(r2b_bar(kP0, grp * 4 + off, 8 + off));
        }
    }
    c.push_back(bar_sync_trap());
    c.push_back(bra_to(cave_va + c.size() * 16, rest_ret_va));
    return L;
}

// Cave 4 (is_save=true) and Cave 6 (is_save=false): Per-Warp UR0..UR63 + UPR + B0..B15 on Elected Lane.
std::vector<Ins> BuildUniformCave(bool is_save, uint64_t cave_va, uint64_t return_va)
{
    std::vector<Ins> c;
    c.reserve(120);

    EmitLoadSlotBase(c, 6, 9, 10);
    c.push_back(ldl_32(8, kRegZero, 0xffff74)); // tid
    c.push_back(shf_r_u32(10, 8, 5));          // warp_id
    c.push_back(imad_wide_u32(6, 10, 0x160, 6));
    c.push_back(iadd64_imm(6, 6, 0x40u));      // R6:R7 = warp_base
    c.push_back(elect_p0());                   // P0 = single elected active lane in warp

    if (!is_save) {
        // Restore UPR FIRST via temporary UR7 before overwriting UR0..UR63 with original UR7.
        c.push_back(ld_e_32(8, 6, 0x100, kP0));
        c.push_back(r2ur(7, 8, kP0));
        c.push_back(ur2up_ur7());
    }

    // 16 chunks x 4 URs (UR0..UR63) at warp_base + [0x000..0x0ff].
    for (int chunk = 0; chunk < 16; ++chunk) {
        if (is_save) {
            for (int i = 0; i < 4; ++i) {
                c.push_back(mov_r_ur(8 + i, chunk * 4 + i, kP0));
            }
            c.push_back(st_e_128(6, static_cast<uint32_t>(chunk * 16), 8, kP0));
        } else {
            c.push_back(ld_e_128(8, 6, static_cast<uint32_t>(chunk * 16), kP0));
            for (int i = 0; i < 4; ++i) {
                c.push_back(r2ur(chunk * 4 + i, 8 + i, kP0));
            }
        }
    }

    if (is_save) {
        // Safe to clobber UR7 with UPR now that original UR7 is stored in chunk 1.
        c.push_back(up2ur_ur7());
        c.push_back(mov_r_ur(8, 7, kP0));
        c.push_back(st_e_32(6, 0x100, 8, kP0));

        // Save convergence barriers B0..B15 to [warp_base + 0x110..0x14f].
        c.push_back(ldl_32(8, kRegZero, 0xffffbc, kP0)); // B0
        for (int b = 1; b <= 3; ++b) c.push_back(bmov_clear_r_b(kP0, 8 + b, b));
        c.push_back(st_e_128(6, 0x110, 8, kP0));
        for (int grp = 0; grp < 3; ++grp) {
            for (int i = 0; i < 4; ++i) {
                c.push_back(bmov_clear_r_b(kP0, 8 + i, 4 + grp * 4 + i));
            }
            c.push_back(st_e_128(6, static_cast<uint32_t>(0x120 + grp * 16), 8, kP0));
        }
    } else {
        // Restore B1..B15 (B0 lives in [0xffffbc] restored in official .text.traphandler epilogue).
        c.push_back(ld_e_128(8, 6, 0x110, kP0));
        for (int b = 1; b <= 3; ++b) c.push_back(bmov_b_r(kP0, b, 8 + b));
        for (int grp = 0; grp < 3; ++grp) {
            c.push_back(ld_e_128(8, 6, static_cast<uint32_t>(0x120 + grp * 16), kP0));
            for (int i = 0; i < 4; ++i) {
                c.push_back(bmov_b_r(kP0, 4 + grp * 4 + i, 8 + i));
            }
        }
    }

    c.push_back(bra_to(cave_va + c.size() * 16, return_va));
    return c;
}

// Cave 7: Dynamic Stack-Spill & __noinline__ LMEM Save + Restore.
struct Cave7Layout
{
    std::vector<Ins> code;
    size_t save_idx = 0;
    size_t rest_idx = 0;
};

Cave7Layout BuildLmemCave(uint64_t cave_va, uint64_t ev_ret_va, uint64_t rest_ret_va)
{
    Cave7Layout L;
    auto &c = L.code;
    c.reserve(80);

    // --- Section 0: LMEM Save ---
    L.save_idx = 0;
    c.push_back(ldc_64(8, 0x178)); // kdesc_va
    c.push_back(ld_e_32(0, 8, offsetof(Sm120KernelDesc, lmem_cap_bytes)));
    c.push_back(isetp_eq_imm(0, 0));
    const int br_save_skip = static_cast<int>(c.size());
    c.push_back(nop());

    // Compute static_lo = stack_top - lmem_static; extend down to user_R1 if has_calls.
    // Never clobber R12:R13 (64-bit link register for RET.ABS back to .text.traphandler!).
    c.push_back(ldc_u32(10, 0x37c)); // R10 = stack_top (e.g. 0xfffdc0)
    c.push_back(ld_e_32(11, 8, offsetof(Sm120KernelDesc, lmem_static_bytes)));
    c.push_back(iadd3_sub_regs(11, 10, 11)); // R11 = static_lo
    c.push_back(ld_e_32(1, 8, offsetof(Sm120KernelDesc, flags)));
    c.push_back(lop3_tst_imm_p0(1, 2u)); // bit 1 = has_device_calls
    const int br_no_calls = static_cast<int>(c.size());
    c.push_back(nop());

    c.push_back(ldl_32(2, kRegZero, 0xffffc4)); // user R1
    c.push_back(lop3_and_imm(2, 2, 0xfffffff0U));
    c.push_back(iadd3_sub_regs(3, 10, 0));      // min_lo = stack_top - lmem_cap
    c.push_back(isetp_lt_u32_regs(2, 11, kP0)); // P0 = (user_R1 < static_lo)
    c.push_back(isetp_lt_u32_regs(2, 3, kP1));  // P1 = (user_R1 < min_lo)
    c.push_back(mov_reg(2, 3, kP1));            // R2 = max(user_R1, min_lo)
    c.push_back(mov_reg(11, 2, kP0));           // if user_R1 < static_lo: R11 = max(user_R1, min_lo)

    const int save_ptr_idx = static_cast<int>(c.size());
    c[br_no_calls] = bra_rel(save_ptr_idx - br_no_calls - 1, kNotP0);
    c.push_back(mov_reg(0, 10)); // R0 = stack_top (preserved across loop; R10 reused below)

    // Store save_lo (R11) to [thread_base + 0x70] using R6:R7 temporary, then R2 = save_lo.
    c.push_back(mov_reg(2, 11)); // preserve save_lo in R2 across EmitComputePerThreadPtr (clobbers R10,R11)
    EmitComputePerThreadPtr(c, 6,
                            offsetof(Sm120ControlHeader, thread_area_off),
                            offsetof(Sm120ControlHeader, thread_frame_stride),
                            10, 11);
    c.push_back(st_e_32(6, 0x70, 2)); // [thread_base + 0x70] = save_lo

    EmitComputePerThreadPtr(c, 6,
                            offsetof(Sm120ControlHeader, lmem_area_off),
                            offsetof(Sm120ControlHeader, lmem_cap_bytes),
                            10, 11);

    const int sloop = static_cast<int>(c.size());
    c.push_back(isetp_ge_u32_regs(2, 0));
    const int br_save_done = static_cast<int>(c.size());
    c.push_back(nop());
    c.push_back(ldl_128(8, 2, 0));
    c.push_back(st_e_128(6, 0, 8));
    c.push_back(iadd_imm(2, 2, 16));
    c.push_back(iadd64_imm(6, 6, 16));
    c.push_back(bra_rel(sloop - static_cast<int>(c.size()) - 1));

    const int save_done = static_cast<int>(c.size());
    c[br_save_skip] = bra_rel(save_done - br_save_skip - 1, kP0);
    c[br_save_done] = bra_rel(save_done - br_save_done - 1, kP0);
    c.push_back(bra_to(cave_va + c.size() * 16, ev_ret_va));

    // --- Section 1: LMEM Restore ---
    L.rest_idx = c.size();
    c.push_back(ldc_64(8, 0x178));
    c.push_back(ld_e_32(0, 8, offsetof(Sm120KernelDesc, lmem_cap_bytes)));
    c.push_back(isetp_eq_imm(0, 0));
    const int br_rest_skip = static_cast<int>(c.size());
    c.push_back(nop());

    c.push_back(ldc_u32(0, 0x37c)); // R0 = stack_top
    EmitComputePerThreadPtr(c, 6,
                            offsetof(Sm120ControlHeader, thread_area_off),
                            offsetof(Sm120ControlHeader, thread_frame_stride),
                            10, 11);
    c.push_back(ld_e_32(2, 6, 0x70)); // R2 = saved_lmem_lo

    EmitComputePerThreadPtr(c, 6,
                            offsetof(Sm120ControlHeader, lmem_area_off),
                            offsetof(Sm120ControlHeader, lmem_cap_bytes),
                            10, 11);

    const int rloop = static_cast<int>(c.size());
    c.push_back(isetp_ge_u32_regs(2, 0));
    const int br_rest_done = static_cast<int>(c.size());
    c.push_back(nop());
    c.push_back(ld_e_128(8, 6, 0));
    c.push_back(stl_128(2, 0, 8));
    c.push_back(iadd_imm(2, 2, 16));
    c.push_back(iadd64_imm(6, 6, 16));
    c.push_back(bra_rel(rloop - static_cast<int>(c.size()) - 1));

    const int rest_done = static_cast<int>(c.size());
    c[br_rest_skip] = bra_rel(rest_done - br_rest_skip - 1, kP0);
    c[br_rest_done] = bra_rel(rest_done - br_rest_done - 1, kP0);
    c.push_back(bra_to(cave_va + c.size() * 16, rest_ret_va));
    return L;
}

// Cave 8: Dynamic SR_SMEMSZ Save/Restore and 112B HW Trap Frame Save/Restore
// (tail-chaining directly into Cave 2 Save GPRs and Cave 5 Restore GPRs).
struct Cave8Layout
{
    std::vector<Ins> code;
    size_t smem_save_idx = 0;
    size_t smem_rest_idx = 0;
    size_t hwframe_save_idx = 0;
    size_t hwframe_rest_idx = 0;
};

Cave8Layout BuildSmemHwFrameCave(uint64_t cave_va,
                                  uint64_t ev_smem_ret_va,
                                  uint64_t rest_smem_ret_va,
                                  uint64_t cave2_save_regs_va,
                                  uint64_t cave5_rest_regs_va)
{
    Cave8Layout L;
    auto &c = L.code;
    c.reserve(116);

    L.smem_save_idx = c.size();
    EmitDynamicSmemLoop(c, /*is_save=*/true);
    c.push_back(bra_to(cave_va + c.size() * 16, ev_smem_ret_va));

    L.smem_rest_idx = c.size();
    EmitDynamicSmemLoop(c, /*is_save=*/false);
    c.push_back(bra_to(cave_va + c.size() * 16, rest_smem_ret_va));

    L.hwframe_save_idx = c.size();
    EmitComputePerThreadPtr(c, 6,
                            offsetof(Sm120ControlHeader, thread_area_off),
                            offsetof(Sm120ControlHeader, thread_frame_stride),
                            10, 11);
    for (int k = 0; k < 7; ++k) {
        c.push_back(ldl_128(8, kRegZero, static_cast<uint32_t>(0xffff90 + 16 * k)));
        c.push_back(st_e_128(6, static_cast<uint32_t>(16 * k), 8));
    }
    c.push_back(bra_to(cave_va + c.size() * 16, cave2_save_regs_va));

    L.hwframe_rest_idx = c.size();
    EmitComputePerThreadPtr(c, 6,
                            offsetof(Sm120ControlHeader, thread_area_off),
                            offsetof(Sm120ControlHeader, thread_frame_stride),
                            10, 11);
    for (int k = 0; k < 7; ++k) {
        c.push_back(ld_e_128(8, 6, static_cast<uint32_t>(16 * k)));
        c.push_back(stl_128(kRegZero, static_cast<uint32_t>(0xffff90 + 16 * k), 8));
    }
    c.push_back(bra_to(cave_va + c.size() * 16, cave5_rest_regs_va));
    return L;
}

} // namespace

std::shared_ptr<Sm120TrapEngine> Sm120TrapEngine::Instance(CUcontext ctx)
{
    static std::mutex s_mtx;
    static std::map<CUcontext, std::shared_ptr<Sm120TrapEngine>> s_map;
    std::lock_guard<std::mutex> lock(s_mtx);
    auto it = s_map.find(ctx);
    if (it != s_map.end()) return it->second;

    CUdevice dev = 0;
    CUDA_ASSERT(Driver::CtxGetDevice(&dev));
    auto eng = std::make_shared<Sm120TrapEngine>(ctx, dev);
    s_map[ctx] = eng;
    return eng;
}

std::mutex &Sm120TrapEngine::LaunchMutex(CUcontext ctx)
{
    static std::mutex s_mtx;
    static std::map<CUcontext, std::unique_ptr<std::mutex>> s_map;
    std::lock_guard<std::mutex> lock(s_mtx);
    auto &ptr = s_map[ctx];
    if (!ptr) ptr = std::make_unique<std::mutex>();
    return *ptr;
}

Sm120TrapEngine::Sm120TrapEngine(CUcontext ctx, CUdevice dev)
    : ctx_(ctx), dev_(dev), arena_(ctx)
{
    int sm = 0, mt = 0;
    if (Driver::DeviceGetAttribute(&sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_) == CUDA_SUCCESS && sm > 0) {
        sm_count_ = sm;
    }
    if (Driver::DeviceGetAttribute(&mt, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev_) == CUDA_SUCCESS && mt > 0) {
        max_threads_per_sm_ = mt;
    }
}

Sm120TrapEngine::~Sm120TrapEngine()
{
    RestoreOriginal();
    if (helper_mod_ != nullptr) {
        CUcontext cur = nullptr;
        if (Driver::CtxGetCurrent(&cur) == CUDA_SUCCESS && cur == ctx_) {
            Driver::ModuleUnload(helper_mod_);
        }
        helper_mod_ = nullptr;
        cond_clear_fn_ = nullptr;
    }
}

bool Sm120TrapEngine::LoadHelperModule()
{
    if (cond_clear_fn_ != nullptr) return true;
    // Conditional post-restore clear (DEF-12): clears header[0..31] and every gate mirror line
    // (Phase 3.3) ONLY if cmd is still 6 or 2 (no-op if Preempt #2 flipped cmd to 5/1). Launched with
    // kGateMirrorEntries threads: all threads read cmd before the barrier, then thread t clears mirror t
    // and thread 0 clears the header.
    static constexpr const char *kCondClearPtx =
        ".version 8.0\n"
        ".target sm_80\n"
        ".address_size 64\n"
        ".visible .entry sm120_cond_clear_hdr(.param .u64 p_ctrl) {\n"
        "    .reg .u64 %rd<4>;\n"
        "    .reg .u32 %r<3>;\n"
        "    .reg .pred %p<3>;\n"
        "    ld.param.u64 %rd1, [p_ctrl];\n"
        "    ld.global.cg.u32 %r1, [%rd1];\n"
        "    setp.eq.u32 %p0, %r1, 6;\n"
        "    setp.eq.u32 %p1, %r1, 2;\n"
        "    or.pred %p0, %p0, %p1;\n"
        "    bar.sync 0;\n"
        "    @!%p0 bra DONE;\n"
        "    mov.u64 %rd0, 0;\n"
        "    mov.u32 %r2, %tid.x;\n"
        "    mul.wide.u32 %rd2, %r2, 128;\n"
        "    sub.u64 %rd3, %rd1, 32768;\n"
        "    add.u64 %rd3, %rd3, %rd2;\n"
        "    st.global.cg.u64 [%rd3], %rd0;\n"
        "    setp.ne.u32 %p2, %r2, 0;\n"
        "    @%p2 bra DONE;\n"
        "    st.global.cg.u64 [%rd1 + 0], %rd0;\n"
        "    st.global.cg.u64 [%rd1 + 8], %rd0;\n"
        "    st.global.cg.u64 [%rd1 + 16], %rd0;\n"
        "    st.global.cg.u64 [%rd1 + 24], %rd0;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    if (Driver::ModuleLoadData(&helper_mod_, kCondClearPtx) != CUDA_SUCCESS || !helper_mod_) {
        XWARN("sm120_trap_engine: failed to load cond_clear_hdr PTX helper");
        return false;
    }
    if (Driver::ModuleGetFunction(&cond_clear_fn_, helper_mod_, "sm120_cond_clear_hdr") != CUDA_SUCCESS ||
        !cond_clear_fn_) {
        XWARN("sm120_trap_engine: failed to resolve sm120_cond_clear_hdr");
        return false;
    }
    return true;
}

bool Sm120TrapEngine::Init()
{
    std::lock_guard<std::mutex> lock(mu_);
    if (ready_) return true;
    if (!arena_.Discover(true)) return false;
    LoadHelperModule();
    if (!EmitStaticHandler()) {
        arena_.RestoreAll();
        return false;
    }
    arena_.EnsureBar0Mapped();
    // OBS-8 (Phase 3.4-2): the evict pulse is a GPU-wide broadcast. On a GPU that also drives a
    // display (kernel execution timeout active), Level-3 stress runs coincided with occasional
    // MMU faults (Xid 31) in desktop graphics processes, which recovered. Warn once.
    CUdevice dev = 0;
    int exec_timeout = 0;
    if (Driver::CtxGetDevice(&dev) == CUDA_SUCCESS &&
        Driver::DeviceGetAttribute(&exec_timeout, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, dev) ==
            CUDA_SUCCESS && exec_timeout != 0) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            XWARN("sm120_trap_engine: this GPU drives a display; Level-3 trap pulses can "
                  "occasionally fault desktop graphics clients (OBS-8), prefer a GPU without display");
        }
    }
    ready_ = true;
    return true;
}

void Sm120TrapEngine::RestoreOriginal()
{
    std::lock_guard<std::mutex> lock(mu_);
    arena_.RestoreAll();
    ready_ = false;
}

bool Sm120TrapEngine::FireEvictPulse()
{
    return arena_.FireEvictPulse();
}

bool Sm120TrapEngine::EmitStaticHandler()
{
    std::vector<size_t> cidx;
    if (!arena_.SelectWorkingCaves(9, 144, cidx)) {
        XWARN("sm120_trap_engine: insufficient trap arena caves (found %zu)", arena_.Caves().size());
        return false;
    }
    const auto &all_caves = arena_.Caves();
    const uint64_t cave_va[9] = {
        all_caves[cidx[0]].va, all_caves[cidx[1]].va, all_caves[cidx[2]].va,
        all_caves[cidx[3]].va, all_caves[cidx[4]].va, all_caves[cidx[5]].va,
        all_caves[cidx[6]].va, all_caves[cidx[7]].va, all_caves[cidx[8]].va,
    };

    // Pass 1: build Cave 0 & Cave 1 skeletons to obtain exact cross-cave return VAs,
    // then generate Caves 2..8 with exact target VAs and finalize branches in Pass 2.
    std::vector<Ins> c0, c1;

    // ----------------------------------------------------------------------
    // Build Cave 1 (Mode 5 Restore Main + clean_exit_redirect + exit_stub)
    // ----------------------------------------------------------------------
    constexpr size_t kC1RestoreEntryIdx = 0;
    // Track warps currently inside Cave 1 synchronously so PreemptStream (cmd 6 -> 5/1)
    // never fires BAR0 while a collaterally-trapped warp is still waiting on PCIe reads.
    c1.push_back(elect_p0());
    c1.push_back(mov_imm(0, 1));
    c1.push_back(atomg_add_u32(1, 4, offsetof(Sm120ControlHeader, trap_inflight), 0, kP0));
    EmitCompute3DCoords(c1, 8, 9, 10);

    // Multi-kernel filter: if active_kernel_idx != 0 && my_idx != active_kernel_idx -> NO-OP RTT.
    c1.push_back(ld_e_64(6, 4, offsetof(Sm120ControlHeader, active_kernel_idx)));
    c1.push_back(mov_imm(2, 0));
    c1.push_back(mov_imm(3, 0));
    c1.push_back(isetp_ne_u64_regs(6, 2));        // P0 = (active != 0)
    const int br_c1_no_active = static_cast<int>(c1.size());
    c1.push_back(nop());                          // @!P0 -> chk_state5
    c1.push_back(ldc_64(0, 0x180));               // R0:R1 = my_idx
    c1.push_back(isetp_ne_u64_regs(0, 6));        // P0 = (my_idx != active)
    const size_t c1_br_wrong_kernel = c1.size();
    c1.push_back(nop());                          // @P0 -> cave0_done_clear_r2 (NO-OP RTT)

    // SM-wide BPT.TRAP 0x1 filter: only State B (cta[b].state == 5) restores.
    // Co-resident State A (0), State C (2), and already-restored State B (0) CTAs on the same SM
    // get collaterally trapped and must return cleanly via NO-OP RTT.
    const int c1_chk_state5_idx = static_cast<int>(c1.size());
    c1[br_c1_no_active] = bra_rel(c1_chk_state5_idx - br_c1_no_active - 1, kNotP0);
    c1.push_back(imad_wide_u32(6, 8, 16, 4));     // R6:R7 = u_ctrl + 16*cta_id
    c1.push_back(ld_e_32(2, 6, 0x50));            // R2 = cta[b].state
    c1.push_back(isetp_ne_imm(2, kCtaStateSaved));// P0 = (state != 5)
    const size_t c1_br_not_state_b = c1.size();
    c1.push_back(nop());                          // @P0 -> cave0_done_clear_r2 (NO-OP RTT)

    c1.push_back(ld_e_32(11, 6, 0x58)); // slot_id
    c1.push_back(stl_128(kRegZero, 0xffff70, 8));

    // Check whether this warp was alive at Mode 5 Evict (cta[b].warp_mask bit w).
    c1.push_back(ld_e_32(1, 6, 0x5c));            // R1 = cta[b].warp_mask
    c1.push_back(shf_r_u32(2, 9, 5));             // R2 = warp_id
    c1.push_back(shf_l_one(3, 2));                // R3 = 1u << warp_id
    c1.push_back(lop3_tst_regs_p0(1, 3));         // P0 = ((warp_mask & my_bit) != 0)
    c1.push_back(mov_imm(0, 0));
    c1.push_back(mov_imm(0, 1, kP0));
    c1.push_back(stl_32(kRegZero, 0xffff6c, 0));  // [0xffff6c] = is_live_warp

    const size_t c1_call_smem_idx = c1.size();
    c1.push_back(nop()); // -> cave8_smem_restore
    const size_t c1_after_smem_idx = c1.size();

    c1.push_back(ldl_32(0, kRegZero, 0xffff6c));
    c1.push_back(isetp_eq_imm(0, 0));
    const int br_c1_skip_lmem = static_cast<int>(c1.size());
    c1.push_back(nop()); // @P0 -> c1_after_lmem_idx
    const size_t c1_call_lmem_idx = c1.size();
    c1.push_back(nop()); // -> cave7_lmem_restore
    const size_t c1_after_lmem_idx = c1.size();
    c1[br_c1_skip_lmem] = bra_rel(static_cast<int>(c1_after_lmem_idx) - br_c1_skip_lmem - 1, kP0);

    const size_t c1_call_bars_idx = c1.size();
    c1.push_back(nop()); // -> cave3_rest_bars
    const size_t c1_after_bars_idx = c1.size();

    c1.push_back(ldl_32(0, kRegZero, 0xffff6c));
    c1.push_back(isetp_eq_imm(0, 0));
    const int br_c1_skip_regs = static_cast<int>(c1.size());
    c1.push_back(nop()); // @P0 -> c1_finalize_idx

    const size_t c1_call_unif_idx = c1.size();
    c1.push_back(nop()); // -> cave6_rest_uniforms
    const size_t c1_after_unif_idx = c1.size();

    const size_t c1_call_hwregs_idx = c1.size();
    c1.push_back(nop()); // -> cave8_hwframe_restore -> cave5_rest_regs
    const size_t c1_after_hwregs_idx = c1.size();

    // Pre-load per-warp B2R.WARP state into R8 BEFORE restored_slots++ (preserve R12:R13!).
    EmitLoadSlotBase(c1, 6, 9, 10);
    c1.push_back(ldl_32(1, kRegZero, 0xffff74)); // tid
    c1.push_back(shf_r_u32(10, 1, 5));          // warp_id
    c1.push_back(imad_wide_u32(6, 10, 0x160, 6));
    c1.push_back(ld_e_32(8, 6, 0x144));         // R8 = warp_base + 0x104 (B2R.WARP)

    const int c1_finalize_idx = static_cast<int>(c1.size());
    c1[br_c1_skip_regs] = bra_rel(c1_finalize_idx - br_c1_skip_regs - 1, kP0);

    // Finalize restore: clear cta[b].state/entry_dec/warp_mask, BAR.SYNC.TRAP, restored_slots++,
    // exit dead warps cleanly, then R2B.WARP on live warps!
    c1.push_back(ldl_128(0, kRegZero, 0xffff70)); // R0=cta_id, R1=tid, R3=slot_id
    c1.push_back(imad_wide_u32(6, 0, 16, 4));
    c1.push_back(isetp_eq_imm(1, 0));
    c1.push_back(mov_imm(2, 0));
    c1.push_back(st_e_32(6, 0x50, 2, kP0));
    c1.push_back(st_e_32(6, 0x54, 2, kP0));
    c1.push_back(st_e_32(6, 0x5c, 2, kP0));
    c1.push_back(bar_sync_trap());
    c1.push_back(isetp_eq_imm(1, 0));
    c1.push_back(mov_imm(2, 1));
    c1.push_back(redg_add_u32(4, offsetof(Sm120ControlHeader, restored_slots), 2, kP0));

    c1.push_back(ldl_32(0, kRegZero, 0xffff6c)); // preserves R8 (B2R.WARP)
    c1.push_back(isetp_eq_imm(0, 0));
    const int br_c1_dead_exit = static_cast<int>(c1.size());
    c1.push_back(nop()); // @P0 -> c1_dead_dec_exit_idx

    c1.push_back(elect_p0());
    c1.push_back(r2b_warp(kP0, 8));
    c1.push_back(mov_imm(0, 0xffffffffu));
    c1.push_back(redg_add_u32(4, offsetof(Sm120ControlHeader, trap_inflight), 0, kP0));
    const size_t c1_ret_rtt_idx = c1.size();
    c1.push_back(nop()); // -> cave0_done_clear_r2

    const size_t c1_exit_stub_idx = c1.size();
    c1.push_back(exit_inst());
    const uint64_t exit_stub_va = cave_va[1] + c1_exit_stub_idx * 16;

    const size_t c1_dead_dec_exit_idx = c1.size();
    c1[br_c1_dead_exit] = bra_rel(static_cast<int>(c1_dead_dec_exit_idx) - br_c1_dead_exit - 1, kP0);
    c1.push_back(elect_p0());
    c1.push_back(mov_imm(0, 0xffffffffu));
    c1.push_back(redg_add_u32(4, offsetof(Sm120ControlHeader, trap_inflight), 0, kP0));

    const size_t c1_clean_exit_idx = c1.size();
    c1.push_back(mov_imm(8, 0xffffffffu));
    c1.push_back(mov_imm(9, 0));
    c1.push_back(mov_imm(10, 0));
    c1.push_back(mov_imm(11, 0));
    c1.push_back(stl_128(kRegZero, 0xffff90, 8));
    c1.push_back(mov_imm(8, 0));
    c1.push_back(stl_128(kRegZero, 0xffffb0, 8));
    c1.push_back(mov_imm(0, 0xffffffffu));
    c1.push_back(stl_32(kRegZero, 0xfffffc, 0));
    c1.push_back(mov_imm(0, 0));
    c1.push_back(stl_32(kRegZero, 0xffffa0, 0));
    c1.push_back(mov_imm(0, static_cast<uint32_t>(exit_stub_va)));
    c1.push_back(mov_imm(1, static_cast<uint32_t>(exit_stub_va >> 32)));
    c1.push_back(stl_64(kRegZero, 0xffffa8, 0));
    const size_t c1_clean_exit_ret_idx = c1.size();
    c1.push_back(nop()); // -> cave0_done_clear_r2

    // Collateral-trap / wrong-kernel exit from Cave 1:
    // Re-read cmd while trap_inflight > 0; if PreemptStream already switched cmd to 5 or 1,
    // dispatch straight into Cave 0 Evict so this warp can never escape to user mode and
    // miss the 1-shot BAR0 pulse.
    const size_t c1_noop_recheck_idx = c1.size();
    c1[c1_br_wrong_kernel] = bra_rel(static_cast<int>(c1_noop_recheck_idx) -
                                     static_cast<int>(c1_br_wrong_kernel) - 1, kP0);
    c1[c1_br_not_state_b]  = bra_rel(static_cast<int>(c1_noop_recheck_idx) -
                                     static_cast<int>(c1_br_not_state_b) - 1, kP0);
    c1.push_back(ld_e_32(1, 4, 0x0));
    c1.push_back(elect_p0());
    c1.push_back(mov_imm(0, 0xffffffffu));
    c1.push_back(redg_add_u32(4, offsetof(Sm120ControlHeader, trap_inflight), 0, kP0));
    c1.push_back(isetp_eq_imm(1, kCmdMode5Evict));
    const size_t c1_recheck_m5_idx = c1.size();
    c1.push_back(nop()); // @P0 -> cave0_coords_idx
    c1.push_back(isetp_eq_imm(1, kCmdMode1Evict));
    const size_t c1_recheck_m1_idx = c1.size();
    c1.push_back(nop()); // @P0 -> cave0_coords_idx
    const size_t c1_recheck_rtt_idx = c1.size();
    c1.push_back(nop()); // -> cave0_done_clear_r2

    // Re-entrancy-safe command dispatcher (entered from Cave 0 whenever cmd != 0):
    // Any warp already redirected to exit_stub_va (user-mode EXIT) must always take NO-OP RTT,
    // preventing re-entry if a recovery BAR0 pulse or collateral BPT.TRAP hits it before EXIT.
    const size_t c1_dispatch_guard_idx = c1.size();
    c1.push_back(ldl_64(6, kRegZero, 0xffffa8));
    c1.push_back(mov_imm(8, static_cast<uint32_t>(exit_stub_va)));
    c1.push_back(mov_imm(9, static_cast<uint32_t>(exit_stub_va >> 32)));
    c1.push_back(isetp_ne_u64_regs(6, 8));
    const size_t c1_br_exit_stub_noop = c1.size();
    c1.push_back(nop()); // @!P0 -> cave0_done_clear_r2

    c1.push_back(isetp_eq_imm(1, kCmdMode5Restore));
    c1.push_back(bra_rel(static_cast<int>(kC1RestoreEntryIdx) -
                         static_cast<int>(c1.size()) - 1, kP0));
    c1.push_back(isetp_eq_imm(1, kCmdMode1Evict));
    const size_t c1_br_disp_m1 = c1.size();
    c1.push_back(nop()); // @P0 -> cave0_coords_idx
    c1.push_back(isetp_eq_imm(1, kCmdMode5Evict));
    const size_t c1_br_disp_m5 = c1.size();
    c1.push_back(nop()); // @P0 -> cave0_coords_idx
    const size_t c1_br_disp_noop = c1.size();
    c1.push_back(nop()); // -> cave0_done_clear_r2

    // ----------------------------------------------------------------------
    // Build Cave 0 (Trap Dispatcher + 3-Region Zero-Race Boundary + Mode 1/5 Evict)
    // ----------------------------------------------------------------------
    c0.push_back(lop3_tst_imm_p0(2, 0x11));
    const int br_ctxsw = static_cast<int>(c0.size());
    c0.push_back(nop());
    c0.push_back(lop3_and_imm(2, 2, 0xfffffff7U));

    c0.push_back(ldc_u32(4, 0x170));
    c0.push_back(ldc_u32(5, 0x174));
    c0.push_back(lop3_or3_p0(4, 5, kRegZero));
    const int br_unmanaged = static_cast<int>(c0.size());
    c0.push_back(nop());

    c0.push_back(ld_e_32(1, 4, 0x0));
    c0.push_back(isetp_eq_imm(1, 0));
    const int br_cmd0 = static_cast<int>(c0.size());
    c0.push_back(nop());
    c0.push_back(bra_to(cave_va[0] + c0.size() * 16,
                        cave_va[1] + c1_dispatch_guard_idx * 16));

    const int coords_idx = static_cast<int>(c0.size());
    EmitCompute3DCoords(c0, 8, 9, 10);
    c0.push_back(mov_imm(11, 0));
    c0.push_back(stl_128(kRegZero, 0xffff70, 8)); // {cta_id, tid, threads_per_cta, 0}

    // Multi-kernel FIFO stream claim/filter.
    c0.push_back(ldc_64(6, 0x180));
    c0.push_back(mov_imm(2, 0));
    c0.push_back(mov_imm(3, 0));
    c0.push_back(isetp_ne_u64_regs(6, 2));
    const int br_untagged = static_cast<int>(c0.size());
    c0.push_back(nop());
    c0.push_back(ld_e_64(10, 4, offsetof(Sm120ControlHeader, active_kernel_idx)));
    c0.push_back(isetp_ne_u64_regs(10, 2));
    const int br_has_active = static_cast<int>(c0.size());
    c0.push_back(nop());
    c0.push_back(elect_p0()); // elected active lane claims active_kernel_idx
    c0.push_back(st_e_64(4, offsetof(Sm120ControlHeader, active_kernel_idx), 6, kP0));
    const int br_claimed = static_cast<int>(c0.size());
    c0.push_back(nop());
    const int chk_match_idx = static_cast<int>(c0.size());
    c0[br_has_active] = bra_rel(chk_match_idx - br_has_active - 1, kP0);
    c0.push_back(isetp_ne_u64_regs(6, 10));
    c0.push_back(bra_to(cave_va[0] + c0.size() * 16,
                        cave_va[1] + c1_clean_exit_idx * 16, kP0));

    const int boundary_idx = static_cast<int>(c0.size());
    c0[br_untagged] = bra_rel(boundary_idx - br_untagged - 1, kNotP0);
    c0[br_claimed]  = bra_rel(boundary_idx - br_claimed - 1, kPT);

    // Handle cmd == 1 (Mode 1 idempotent early exit) directly with ZERO bar_sync_trap():
    // sibling warps may have already exited at Guardian Slot 49 (BAR.SYNC 0 + EXIT).
    c0.push_back(ld_e_32(1, 4, 0x0));
    c0.push_back(isetp_ne_imm(1, 1));
    const int br_m5_boundary = static_cast<int>(c0.size());
    c0.push_back(nop());

    c0.push_back(ldl_32(8, kRegZero, 0xffff70)); // cta_id
    c0.push_back(ldl_32(9, kRegZero, 0xffff74)); // tid
    c0.push_back(imad_wide_u32(6, 8, 16, 4));    // R6:R7 = u_ctrl + 16*cta_id
    c0.push_back(ld_e_32(2, 6, 0x54));           // entry_dec
    c0.push_back(isetp_eq_imm(9, 0, kP0));       // P0 = (tid == 0)
    c0.push_back(isetp_eq_and_pred(2, 0, kP0, kP0)); // P0 = (tid == 0 && entry_dec == 0)
    c0.push_back(mov_imm(0, kCtaStateReplay));   // 1 (reused for state and redg_add)
    c0.push_back(st_e_32(6, 0x50, 0, kP0));
    c0.push_back(redg_add_u32(4, offsetof(Sm120ControlHeader, mode1_exit_cnt), 0, kP0));
    const int br_clear_bars_m1 = static_cast<int>(c0.size());
    c0.push_back(nop());                         // -> clear_bars_exit_va (no bar_sync_trap!)

    // Direct 4-interval unsigned 64-bit Guardian boundary check for cmd == 5:
    //   Interval 1 [0x000..0x2e0]: pre/at BAR.SYNC 0 -> mark State A (3) or C (2), sync
    //   Interval 2 [0x2f0..0x320]: post-BAR.SYNC 0 -> if entry_dec!=0 escape w/o bar_sync_trap; else sync State B
    //   Interval 3 [0x330..0x410]: restore_path -> if State A escape w/o bar_sync_trap; else mark State C
    //   Interval 4 [0x420..0x4f0]: replay_state_c -> mark State C, sync
    const int m5_boundary_idx = static_cast<int>(c0.size());
    c0[br_m5_boundary] = bra_rel(m5_boundary_idx - br_m5_boundary - 1, kP0);
    c0.push_back(ldl_32(8, kRegZero, 0xffff70)); // cta_id
    c0.push_back(imad_wide_u32(2, 8, 16, 4));    // R2:R3 = u_ctrl + 16*cta_id (keep intact!)
    c0.push_back(ldl_64(6, kRegZero, 0xffffa8)); // TRAP_RETURN_PC
    c0.push_back(ldc_64(8, 0x178));              // kdesc_va
    c0.push_back(ld_e_64(10, 8, offsetof(Sm120KernelDesc, ep_inst_va)));
    c0.push_back(iadd64_sub_regs(6, 6, 10));     // R6:R7 = pc_off

    // Interval 1: pc_off < kGuardianBarOff (strictly before BAR.SYNC 0: no sibling warp can have exited!)
    // Note: if pc_off == kGuardianBarOff, this warp may have just released BAR.SYNC 0 while
    // sibling warps already exited at slot 42 (when entry_dec != 0), so it MUST take Interval 2!
    c0.push_back(isetp_lt_u64_imm(6, kGuardianBarOff));
    const int br_r1 = static_cast<int>(c0.size());
    c0.push_back(nop());

    // Interval 2: kGuardianBarOff <= pc_off < kGuardianRestorePathOff
    c0.push_back(isetp_lt_u64_imm(6, GuardianRestorePathOff()));
    const int br_not_r2 = static_cast<int>(c0.size());
    c0.push_back(nop());
    const int r2_postbar_chk_idx = static_cast<int>(c0.size());
    c0.push_back(ld_e_32(0, 2, 0x54));           // entry_dec
    // Only escape without bar_sync_trap if Guardian decided drain (entry_dec == 2);
    // entry_dec == 4 means a sibling warp inside Cave 0 marked drain and is waiting at bar_sync_trap!
    c0.push_back(isetp_eq_imm(0, 2));
    const int br_escape_postbar = static_cast<int>(c0.size());
    c0.push_back(nop());                         // @P0 -> clear_bars_exit_va
    const int br_postbar_body = static_cast<int>(c0.size());
    c0.push_back(nop());                         // -> post_gate_sync_idx (State B or entry_dec==4)

    // Interval 3: pc_off < kGuardianReplayCIdxOff
    const int chk_r3_idx = static_cast<int>(c0.size());
    c0[br_not_r2] = bra_rel(chk_r3_idx - br_not_r2 - 1, kNotP0);
    c0.push_back(isetp_lt_u64_imm(6, GuardianReplayCIdxOff()));
    const int br_not_r3 = static_cast<int>(c0.size());
    c0.push_back(nop());
    const int chk_state_a_idx = static_cast<int>(c0.size());
    c0.push_back(ld_e_32(0, 4, offsetof(Sm120ControlHeader, is_restore_wave)));
    c0.push_back(ld_e_32(10, 2, 0x50));          // cta[b].state (do NOT clobber R3!)
    c0.push_back(isetp_ne_imm(0, 0, kP0));
    c0.push_back(isetp_eq_and_pred(10, 0, kP0, kP0));
    const int br_escape_state_a = static_cast<int>(c0.size());
    c0.push_back(nop());                         // @P0 -> clear_bars_exit_va
    const int br_r3_mark_c = static_cast<int>(c0.size());
    c0.push_back(nop());                         // -> mark_c_idx

    // Interval 4: pc_off < kGuardianReplayBarOff -> mark_c_idx (escape if entry_dec==2; else set 4 & sync).
    // Interval 5: kGuardianReplayBarOff <= pc_off < kGuardianBytes -> at/post-BAR.SYNC 0 exit check in replay_state_c
    //             (reuses r2_postbar_chk_idx: if entry_dec == 2 escape w/o bar_sync_trap; else sync).
    const int chk_r4_idx = static_cast<int>(c0.size());
    c0[br_not_r3] = bra_rel(chk_r4_idx - br_not_r3 - 1, kNotP0);
    c0.push_back(isetp_lt_u64_imm(6, kGuardianReplayBarOff));
    const int br_r4_mark_c = static_cast<int>(c0.size());
    c0.push_back(nop());                         // @P0 -> mark_c_idx
    c0.push_back(isetp_lt_u64_imm(6, kGuardianBytes));
    c0.push_back(bra_rel(r2_postbar_chk_idx - static_cast<int>(c0.size()) - 1, kP0));
    const int br_not_in_guardian = static_cast<int>(c0.size());
    c0.push_back(nop());                         // -> post_gate_sync_idx (user body)

    // Interval 1 jumps directly to chk_state_a_idx: if (is_restore_wave && state==0) escape w/o bar_sync_trap
    // (sibling warps may have exited at slot 54), else fall into mark_c_idx (if entry_dec==2 escape w/o
    // bar_sync_trap; else write entry_dec=4 and sync at bar_sync_trap).
    c0[br_r1] = bra_rel(chk_state_a_idx - br_r1 - 1, kP0);

    const int mark_c_idx = static_cast<int>(c0.size());
    c0[br_r3_mark_c] = bra_rel(mark_c_idx - br_r3_mark_c - 1, kPT);
    c0[br_r4_mark_c] = bra_rel(mark_c_idx - br_r4_mark_c - 1, kP0);
    c0.push_back(ld_e_32(0, 2, 0x54));           // entry_dec
    c0.push_back(isetp_eq_imm(0, 2));            // Guardian already decided drain -> sibling may have exited!
    c0.push_back(bra_rel(br_escape_postbar - static_cast<int>(c0.size()) - 1, kP0));
    c0.push_back(mov_imm(0, 4));                 // 4 = trap-handler decided State C (all warps alive)
    c0.push_back(st_e_32(2, 0x54, 0));           // cta[b].entry_dec = 4

    const int post_gate_sync_idx = static_cast<int>(c0.size());
    c0[br_postbar_body]    = bra_rel(post_gate_sync_idx - br_postbar_body - 1, kPT);
    c0[br_not_in_guardian] = bra_rel(post_gate_sync_idx - br_not_in_guardian - 1, kPT);
    c0.push_back(bar_sync_trap());

    // Consensus drain check for State C (entry_dec != 0, i.e., 4) vs State B (0).
    c0.push_back(ldl_32(8, kRegZero, 0xffff70)); // cta_id
    c0.push_back(ldl_32(9, kRegZero, 0xffff74)); // tid
    c0.push_back(imad_wide_u32(6, 8, 16, 4));
    c0.push_back(ld_e_32(2, 6, 0x54));           // entry_dec

    c0.push_back(isetp_ne_imm(2, 0));            // State C (entry_dec != 0)?
    const int br_state_b_evict = static_cast<int>(c0.size());
    c0.push_back(nop());

    // State C drained path (cmd == 5 && entry_dec != 0).
    c0.push_back(isetp_eq_imm(9, 0));
    c0.push_back(mov_imm(2, kCtaStateQueued));
    c0.push_back(st_e_32(6, 0x50, 2, kP0));
    c0.push_back(mov_imm(0, 1));
    c0.push_back(redg_add_u32(4, offsetof(Sm120ControlHeader, drained_cnt), 0, kP0));
    const int br_clear_bars_c = static_cast<int>(c0.size());
    c0.push_back(nop());

    // Mode 5 State B Evict: record surviving warp_mask and elect dynamic CTA leader for slot_id allocation.
    const int state_b_idx = static_cast<int>(c0.size());
    c0[br_state_b_evict] = bra_rel(state_b_idx - br_state_b_evict - 1, kNotP0);
    c0.push_back(shf_r_u32(10, 9, 5));                                                      // warp_id
    c0.push_back(shf_l_one(3, 10));                                                         // my_bit = 1u << warp_id
    c0.push_back(elect_p0());                                                               // P0 = elected active lane
    c0.push_back(atomg_or_u32(0, 6, 0x5c, 3, kP0));                                         // @P0 R0 = old warp_mask; mask |= my_bit
    c0.push_back(isetp_eq_and_pred(0, 0, kP0, kP0));                                        // P0 = (elected && old == 0)
    c0.push_back(mov_imm(0, 0));
    c0.push_back(mov_imm(0, 1, kP0));
    c0.push_back(stl_32(kRegZero, 0xffff6c, 0));                                            // [0xffff6c] = is_leader
    c0.push_back(mov_imm(0, 1));
    c0.push_back(atomg_add_u32(11, 4, offsetof(Sm120ControlHeader, evicted_slots), 0, kP0));
    c0.push_back(st_e_32(6, 0x58, 11, kP0));
    c0.push_back(bar_sync_trap());
    c0.push_back(ld_e_32(11, 6, 0x58));
    c0.push_back(stl_32(kRegZero, 0xffff7c, 11));
    c0.push_back(ld_e_32(0, 4, offsetof(Sm120ControlHeader, max_slots)));
    c0.push_back(isetp_lt_u32_regs(11, 0));
    const int br_slot_ok = static_cast<int>(c0.size());
    c0.push_back(nop());
    c0.push_back(mov_imm(0, 1));
    c0.push_back(st_e_32(4, offsetof(Sm120ControlHeader, overflow_err), 0));
    c0.push_back(bra_to(cave_va[0] + c0.size() * 16,
                        cave_va[1] + c1_clean_exit_idx * 16));

    const int slot_ok_idx = static_cast<int>(c0.size());
    c0[br_slot_ok] = bra_rel(slot_ok_idx - br_slot_ok - 1, kP0);

    // Jump Chain: Cave 3 (Bars) -> Cave 4 (Uniforms) -> Cave 8 (SMEM) -> Cave 7 (LMEM) -> Cave 8 (HW Frame) -> Cave 2 (GPRs).
    const size_t c0_call_bars_idx = c0.size();
    c0.push_back(nop());
    const size_t c0_after_bars_idx = c0.size();

    const size_t c0_call_unif_idx = c0.size();
    c0.push_back(nop());
    const size_t c0_after_unif_idx = c0.size();

    const size_t c0_call_smem_idx = c0.size();
    c0.push_back(nop());
    const size_t c0_after_smem_idx = c0.size();

    const size_t c0_call_lmem_idx = c0.size();
    c0.push_back(nop());
    const size_t c0_after_lmem_idx = c0.size();

    const size_t c0_call_hwregs_idx = c0.size();
    c0.push_back(nop());
    const size_t c0_after_hwregs_idx = c0.size();

    // Finalize Evict: BAR.SYNC.TRAP -> cta[b].state = 5 -> clean exit!
    c0.push_back(bar_sync_trap());
    c0.push_back(ldl_32(8, kRegZero, 0xffff70));
    c0.push_back(ldl_32(9, kRegZero, 0xffff6c));
    c0.push_back(imad_wide_u32(6, 8, 16, 4));
    c0.push_back(isetp_ne_imm(9, 0));
    c0.push_back(mov_imm(2, kCtaStateSaved));
    c0.push_back(st_e_32(6, 0x50, 2, kP0));
    c0.push_back(bra_to(cave_va[0] + c0.size() * 16,
                        cave_va[1] + c1_clean_exit_idx * 16));

    const size_t c0_done_clear_r2_idx = c0.size();
    c0[br_cmd0] = bra_rel(static_cast<int>(c0_done_clear_r2_idx) - br_cmd0 - 1, kP0);
    c0.push_back(mov_imm(2, 0));

    const size_t c0_done_ctxsw_idx = c0.size();
    c0[br_ctxsw]     = bra_rel(static_cast<int>(c0_done_ctxsw_idx) - br_ctxsw - 1, kP0);
    c0[br_unmanaged] = bra_rel(static_cast<int>(c0_done_ctxsw_idx) - br_unmanaged - 1, kNotP0);
    c0.push_back(mov_imm(1, 0xffff80u));
    c0.push_back(mov_imm(3, 0));
    c0.push_back(iadd_imm(12, 12, 0x10u));
    c0.push_back(ret_r12());

    // ----------------------------------------------------------------------
    // Build Caves 2..8 and wire inter-cave branches
    // ----------------------------------------------------------------------
    Cave3Layout c3 = BuildNamedBarriersCave(cave_va[3],
                                            cave_va[0] + c0_after_bars_idx * 16,
                                            cave_va[1] + c1_after_bars_idx * 16,
                                            cave_va[1] + c1_clean_exit_idx * 16);
    std::vector<Ins> c4 = BuildUniformCave(/*is_save=*/true, cave_va[4],
                                           cave_va[0] + c0_after_unif_idx * 16);
    std::vector<Ins> c2 = BuildRegCascadeCave(/*is_save=*/true, cave_va[2],
                                              cave_va[0] + c0_after_hwregs_idx * 16);
    std::vector<Ins> c5 = BuildRegCascadeCave(/*is_save=*/false, cave_va[5],
                                              cave_va[1] + c1_after_hwregs_idx * 16);
    std::vector<Ins> c6 = BuildUniformCave(/*is_save=*/false, cave_va[6],
                                           cave_va[1] + c1_after_unif_idx * 16);
    Cave7Layout c7 = BuildLmemCave(cave_va[7],
                                   cave_va[0] + c0_after_lmem_idx * 16,
                                   cave_va[1] + c1_after_lmem_idx * 16);
    Cave8Layout c8 = BuildSmemHwFrameCave(cave_va[8],
                                          cave_va[0] + c0_after_smem_idx * 16,
                                          cave_va[1] + c1_after_smem_idx * 16,
                                          cave_va[2],
                                          cave_va[5]);

    const uint64_t clear_bars_exit_va = cave_va[3] + c3.clear_exit_idx * 16;
    c0[br_escape_postbar] = bra_to(cave_va[0] + br_escape_postbar * 16, clear_bars_exit_va, kP0);
    c0[br_escape_state_a] = bra_to(cave_va[0] + br_escape_state_a * 16, clear_bars_exit_va, kP0);
    c0[br_clear_bars_c]   = bra_to(cave_va[0] + br_clear_bars_c * 16,   clear_bars_exit_va);
    c0[br_clear_bars_m1]  = bra_to(cave_va[0] + br_clear_bars_m1 * 16,  clear_bars_exit_va);

    c0[c0_call_bars_idx]   = bra_to(cave_va[0] + c0_call_bars_idx * 16,   cave_va[3] + c3.save_idx * 16);
    c0[c0_call_unif_idx]   = bra_to(cave_va[0] + c0_call_unif_idx * 16,   cave_va[4]);
    c0[c0_call_smem_idx]   = bra_to(cave_va[0] + c0_call_smem_idx * 16,   cave_va[8] + c8.smem_save_idx * 16);
    c0[c0_call_lmem_idx]   = bra_to(cave_va[0] + c0_call_lmem_idx * 16,   cave_va[7] + c7.save_idx * 16);
    c0[c0_call_hwregs_idx] = bra_to(cave_va[0] + c0_call_hwregs_idx * 16, cave_va[8] + c8.hwframe_save_idx * 16);

    c1[c1_recheck_m5_idx]     = bra_to(cave_va[1] + c1_recheck_m5_idx * 16,
                                       cave_va[0] + static_cast<size_t>(coords_idx) * 16, kP0);
    c1[c1_recheck_m1_idx]     = bra_to(cave_va[1] + c1_recheck_m1_idx * 16,
                                       cave_va[0] + static_cast<size_t>(coords_idx) * 16, kP0);
    c1[c1_recheck_rtt_idx]    = bra_to(cave_va[1] + c1_recheck_rtt_idx * 16,
                                       cave_va[0] + c0_done_clear_r2_idx * 16);
    c1[c1_br_exit_stub_noop]  = bra_to(cave_va[1] + c1_br_exit_stub_noop * 16,
                                       cave_va[0] + c0_done_clear_r2_idx * 16, kNotP0);
    c1[c1_br_disp_m1]         = bra_to(cave_va[1] + c1_br_disp_m1 * 16,
                                       cave_va[0] + static_cast<size_t>(coords_idx) * 16, kP0);
    c1[c1_br_disp_m5]         = bra_to(cave_va[1] + c1_br_disp_m5 * 16,
                                       cave_va[0] + static_cast<size_t>(coords_idx) * 16, kP0);
    c1[c1_br_disp_noop]       = bra_to(cave_va[1] + c1_br_disp_noop * 16,
                                       cave_va[0] + c0_done_clear_r2_idx * 16);
    c1[c1_call_smem_idx]   = bra_to(cave_va[1] + c1_call_smem_idx * 16,   cave_va[8] + c8.smem_rest_idx * 16);
    c1[c1_call_lmem_idx]   = bra_to(cave_va[1] + c1_call_lmem_idx * 16,   cave_va[7] + c7.rest_idx * 16);
    c1[c1_call_bars_idx]   = bra_to(cave_va[1] + c1_call_bars_idx * 16,   cave_va[3] + c3.rest_idx * 16);
    c1[c1_call_unif_idx]   = bra_to(cave_va[1] + c1_call_unif_idx * 16,   cave_va[6]);
    c1[c1_call_hwregs_idx] = bra_to(cave_va[1] + c1_call_hwregs_idx * 16, cave_va[8] + c8.hwframe_rest_idx * 16);
    c1[c1_ret_rtt_idx]     = bra_to(cave_va[1] + c1_ret_rtt_idx * 16,     cave_va[0] + c0_done_clear_r2_idx * 16);
    c1[c1_clean_exit_ret_idx] = bra_to(cave_va[1] + c1_clean_exit_ret_idx * 16,
                                       cave_va[0] + c0_done_clear_r2_idx * 16);

    const std::vector<Ins> *bins[9] = {&c0, &c1, &c2, &c3.code, &c4, &c5, &c6, &c7.code, &c8.code};
    for (int i = 0; i < 9; ++i) {
        if (bins[i]->size() > all_caves[cidx[i]].capacity_instrs) {
            XERRO("sm120_trap_engine: cave %d overflow (%zu > %u slots)",
                  i, bins[i]->size(), all_caves[cidx[i]].capacity_instrs);
            return false;
        }
        if (!arena_.WriteCave(cidx[i], *bins[i])) return false;
    }

    if (!arena_.PatchHandlerHook(cave_va[0])) return false;

    XINFO("sm120_trap_engine: installed static 9-cave engine "
          "(c0=%zu/%u c1=%zu/%u c2=%zu/%u c3=%zu/%u c4=%zu/%u c5=%zu/%u c6=%zu/%u c7=%zu/%u c8=%zu/%u)",
          c0.size(), all_caves[cidx[0]].capacity_instrs,
          c1.size(), all_caves[cidx[1]].capacity_instrs,
          c2.size(), all_caves[cidx[2]].capacity_instrs,
          c3.code.size(), all_caves[cidx[3]].capacity_instrs,
          c4.size(), all_caves[cidx[4]].capacity_instrs,
          c5.size(), all_caves[cidx[5]].capacity_instrs,
          c6.size(), all_caves[cidx[6]].capacity_instrs,
          c7.code.size(), all_caves[cidx[7]].capacity_instrs,
          c8.code.size(), all_caves[cidx[8]].capacity_instrs);
    return true;
}

// ==========================================================================
// Sm120StreamContext (VRAM-Resident Control Plane + High-Priority CE DMA)
// ==========================================================================

namespace
{

constexpr size_t kLayoutStageSlots = 256;
constexpr size_t kLayoutStageBytes = kLayoutStageSlots * 64;

struct ScopedCudaCtx
{
    bool pushed = false;
    explicit ScopedCudaCtx(CUcontext target)
    {
        if (target == nullptr) return;
        CUcontext cur = nullptr;
        if (Driver::CtxGetCurrent(&cur) != CUDA_SUCCESS || cur != target) {
            if (Driver::CtxPushCurrent_v2(target) == CUDA_SUCCESS) {
                pushed = true;
            }
        }
    }
    ~ScopedCudaCtx()
    {
        if (pushed) {
            CUcontext popped = nullptr;
            (void)Driver::CtxPopCurrent_v2(&popped);
        }
    }
};

} // namespace

Sm120StreamContext::Sm120StreamContext(CUcontext ctx, CUdevice dev,
                                       std::shared_ptr<Sm120TrapEngine> engine)
    : ctx_(ctx), dev_(dev), engine_(std::move(engine))
{
    ScopedCudaCtx guard(ctx_);
    guardian_reg_ = Sm120GuardianRegistry::Instance(ctx_);

    int lp = 0, hp = 0;
    if (Driver::CtxGetStreamPriorityRange(&lp, &hp) == CUDA_SUCCESS) {
        if (Driver::StreamCreateWithPriority(&ctrl_op_stream_, CU_STREAM_NON_BLOCKING, hp) != CUDA_SUCCESS) {
            Driver::StreamCreate(&ctrl_op_stream_, CU_STREAM_NON_BLOCKING);
        }
    } else {
        Driver::StreamCreate(&ctrl_op_stream_, CU_STREAM_NON_BLOCKING);
    }

    void *hdr_pin = nullptr;
    if (Driver::MemHostAlloc(&hdr_pin, 128, CU_MEMHOSTALLOC_PORTABLE) == CUDA_SUCCESS && hdr_pin) {
        memset(hdr_pin, 0, 128);
        ctrl_host_ = reinterpret_cast<Sm120ControlHeader *>(hdr_pin);
    }

    void *stage = nullptr;
    if (Driver::MemHostAlloc(&stage, kLayoutStageBytes, CU_MEMHOSTALLOC_PORTABLE) == CUDA_SUCCESS && stage) {
        memset(stage, 0, kLayoutStageBytes);
        layout_stage_ = reinterpret_cast<uint8_t *>(stage);
    }

    EnsureControlCapacity(65536);
}

Sm120StreamContext::~Sm120StreamContext()
{
    CUcontext cur = nullptr;
    if (Driver::CtxGetCurrent(&cur) == CUDA_ERROR_DEINITIALIZED) {
        return;
    }
    ScopedCudaCtx guard(ctx_);
    if (ctrl_op_stream_ != nullptr) {
        (void)Driver::StreamSynchronize(ctrl_op_stream_);
        Driver::StreamDestroy_v2(ctrl_op_stream_);
        ctrl_op_stream_ = nullptr;
    }
    for (CUdeviceptr old_arena : retired_arenas_) {
        if (old_arena != 0) Driver::MemFree_v2(old_arena);
    }
    retired_arenas_.clear();
    if (arena_dev_ != 0) {
        Driver::MemFree_v2(arena_dev_);
        arena_dev_ = 0;
        arena_bytes_ = 0;
    }
    // retired_ctrls_ hold allocation bases; the live allocation starts kGateMirrorBytes before ctrl_dev_.
    for (CUdeviceptr old_ctrl : retired_ctrls_) {
        if (old_ctrl != 0) Driver::MemFree_v2(old_ctrl);
    }
    retired_ctrls_.clear();
    if (ctrl_dev_ != 0) {
        Driver::MemFree_v2(ctrl_dev_ - kGateMirrorBytes);
        ctrl_dev_ = 0;
    }
    if (ctrl_host_ != nullptr) {
        Driver::MemFreeHost(ctrl_host_);
        ctrl_host_ = nullptr;
    }
    if (layout_stage_ != nullptr) {
        Driver::MemFreeHost(layout_stage_);
        layout_stage_ = nullptr;
    }
}

bool Sm120StreamContext::ReadHeaderLocked(Sm120ControlHeader *out) const
{
    if (!out || ctrl_dev_ == 0 || ctrl_host_ == nullptr || ctrl_op_stream_ == nullptr) return false;
    ScopedCudaCtx guard(ctx_);
    if (Driver::MemcpyDtoHAsync_v2(ctrl_host_, ctrl_dev_, sizeof(Sm120ControlHeader), ctrl_op_stream_) != CUDA_SUCCESS) {
        return false;
    }
    if (Driver::StreamSynchronize(ctrl_op_stream_) != CUDA_SUCCESS) return false;
    memcpy(out, ctrl_host_, sizeof(Sm120ControlHeader));
    static const bool check_mirror = GetEnvOption("XSCHED_SM120_GATE_MIRROR_CHECK", false);
    if (check_mirror) {
        // DEBUG: every gate mirror line must equal {cmd, is_restore_wave} of the primary header.
        static thread_local std::vector<uint32_t> buf(kGateMirrorBytes / 4);
        if (Driver::MemcpyDtoH_v2(buf.data(), ctrl_dev_ - kGateMirrorBytes, kGateMirrorBytes) == CUDA_SUCCESS) {
            int bad = 0, first = -1;
            for (uint32_t i = 0; i < kGateMirrorEntries; ++i) {
                const uint32_t c = buf[i * (kGateMirrorStride / 4)], r = buf[i * (kGateMirrorStride / 4) + 1];
                if (c != out->cmd || r != out->is_restore_wave) { if (first < 0) first = static_cast<int>(i); ++bad; }
            }
            if (bad) {
                const uint32_t i = static_cast<uint32_t>(first);
                XWARN("sm120_gate_mirror MISMATCH: %d lines, first line %u = {cmd %u, rw %u} vs header {cmd %u, rw %u}",
                      bad, i, buf[i * 32], buf[i * 32 + 1], out->cmd, out->is_restore_wave);
            }
        }
    }
    return true;
}

bool Sm120StreamContext::WriteHeaderSliceLocked(size_t offset, const void *src, size_t bytes)
{
    if (!src || ctrl_dev_ == 0 || ctrl_host_ == nullptr || ctrl_op_stream_ == nullptr) return false;
    if (offset + bytes > sizeof(Sm120ControlHeader)) return false;
    ScopedCudaCtx guard(ctx_);
    uint8_t *dst_pin = reinterpret_cast<uint8_t *>(ctrl_host_) + offset;
    memcpy(dst_pin, src, bytes);
    if (Driver::MemcpyHtoDAsync_v2(ctrl_dev_ + offset, dst_pin, bytes, ctrl_op_stream_) != CUDA_SUCCESS) {
        return false;
    }
    return Driver::StreamSynchronize(ctrl_op_stream_) == CUDA_SUCCESS;
}

bool Sm120StreamContext::EnsureControlCapacity(uint32_t num_ctas)
{
    const uint32_t needed_ctas = std::max(num_ctas, 65536u);
    if (ctrl_dev_ != 0 && needed_ctas <= ctrl_cta_cap_) return true;

    ScopedCudaCtx guard(ctx_);
    // If growing u_control after kernels have already been queued on last_stream_,
    // wait for the stream so no in-flight kernel references the retired ctrl_dev_.
    if (ctrl_dev_ != 0 && last_stream_ != nullptr) {
        (void)Driver::StreamSynchronize(last_stream_);
    }

    // Layout: [gate mirror: kGateMirrorEntries x 128 B][Sm120ControlHeader][CtaEntry x needed_ctas];
    // ctrl_dev_ (u_control) points at the header, the mirror sits at negative offsets from it.
    const size_t new_bytes = AlignUpSize(kGateMirrorBytes + sizeof(Sm120ControlHeader) +
                                         static_cast<size_t>(needed_ctas) * sizeof(Sm120CtaEntry),
                                         4096ULL);
    CUdeviceptr new_base = 0;
    if (Driver::MemAlloc_v2(&new_base, new_bytes) != CUDA_SUCCESS || new_base == 0) {
        return false;
    }
    const CUdeviceptr new_dev = new_base + kGateMirrorBytes;
    if (ctrl_op_stream_ != nullptr) {
        Driver::MemsetD8Async(new_base, 0, new_bytes, ctrl_op_stream_);
        if (ctrl_dev_ != 0) {
            const size_t copy_bytes = kGateMirrorBytes + sizeof(Sm120ControlHeader) +
                                      static_cast<size_t>(ctrl_cta_cap_) * sizeof(Sm120CtaEntry);
            Driver::MemcpyDtoDAsync_v2(new_base, ctrl_dev_ - kGateMirrorBytes, copy_bytes, ctrl_op_stream_);
            retired_ctrls_.push_back(ctrl_dev_ - kGateMirrorBytes);
        }
        Driver::MemsetD32Async(new_dev + offsetof(Sm120ControlHeader, max_ctas),
                               needed_ctas, 1, ctrl_op_stream_);
        Driver::StreamSynchronize(ctrl_op_stream_);
    } else {
        Driver::MemsetD8_v2(new_base, 0, new_bytes);
        if (ctrl_dev_ != 0) {
            const size_t copy_bytes = kGateMirrorBytes + sizeof(Sm120ControlHeader) +
                                      static_cast<size_t>(ctrl_cta_cap_) * sizeof(Sm120CtaEntry);
            Driver::MemcpyDtoD_v2(new_base, ctrl_dev_ - kGateMirrorBytes, copy_bytes);
            retired_ctrls_.push_back(ctrl_dev_ - kGateMirrorBytes);
        }
        Driver::MemsetD32_v2(new_dev + offsetof(Sm120ControlHeader, max_ctas), needed_ctas, 1);
    }

    ctrl_dev_ = new_dev;
    ctrl_bytes_ = new_bytes;
    ctrl_cta_cap_ = needed_ctas;
    if (ctrl_host_ != nullptr) ctrl_host_->max_ctas = needed_ctas;
    return true;
}

bool Sm120StreamContext::EnsureArenaPool(size_t required_bytes)
{
    if (required_bytes > window_max_required_) window_max_required_ = required_bytes;
    if (required_bytes == 0) return true;
    if (arena_dev_ != 0 && required_bytes <= arena_bytes_) return true;

    const size_t alloc_bytes = AlignUpSize(required_bytes, 65536ULL);
    CUdeviceptr new_arena = 0;
    if (Driver::MemAlloc_v2(&new_arena, alloc_bytes) != CUDA_SUCCESS || !new_arena) {
        return false;
    }
    if (arena_dev_ != 0) {
        retired_arenas_.push_back(arena_dev_);
    }
    arena_dev_ = new_arena;
    arena_bytes_ = alloc_bytes;
    return true;
}

uint64_t Sm120StreamContext::LaunchSeq() const
{
    std::lock_guard<std::mutex> lock(mu_);
    return launch_seq_;
}

size_t Sm120StreamContext::TrimArenas(uint64_t seq_before_sync)
{
    static const bool enabled = GetEnvOption("XSCHED_SM120_ARENA_TRIM", true);
    if (!enabled) return 0;
    std::lock_guard<std::mutex> lock(mu_);
    // Something was launched after the synchronization started (it may use the current arena), or
    // an evicted kernel's state still lives in an arena: nothing is safe to free.
    if (launch_seq_ != seq_before_sync || saved_arena_dev_ != 0 || is_restore_cmd_active_) return 0;
    size_t freed = 0;
    if (!retired_arenas_.empty()) {
        ScopedCudaCtx guard(ctx_);
        for (CUdeviceptr old_arena : retired_arenas_) {
            if (old_arena != 0 && Driver::MemFree_v2(old_arena) == CUDA_SUCCESS) freed += 1;
        }
        retired_arenas_.clear();
    }
    constexpr uint32_t kWindowSyncs = 64;
    static const size_t kMinTrimBytes = []() {
        int64_t mb = 64;
        GetEnvInt64("XSCHED_SM120_ARENA_TRIM_MIN_MB", mb);
        return static_cast<size_t>(mb < 0 ? 0 : mb) << 20;
    }();
    if (++window_syncs_ >= kWindowSyncs) {
        if (arena_dev_ != 0 && arena_bytes_ > kMinTrimBytes && arena_bytes_ > 2 * window_max_required_) {
            ScopedCudaCtx guard(ctx_);
            if (Driver::MemFree_v2(arena_dev_) == CUDA_SUCCESS) {
                XINFO("sm120_arena: trimmed idle arena %zu B (window max requirement %zu B)",
                      arena_bytes_, window_max_required_);
                freed += arena_bytes_;
                arena_dev_ = 0;
                arena_bytes_ = 0;
            }
        }
        window_syncs_ = 0;
        window_max_required_ = 0;
    }
    if (freed) kernel_arena_map_.clear();   // no launched kernel can still reference a freed arena
    return freed;
}

namespace
{
// Functions whose debugger-params window currently holds a non-zero (Mode 5) value. Written only
// by PrepareLaunch(); ClearDebuggerParams() skips the driver call for every other function
// (never instrumented, or already cleared). Both callers hold the context's LaunchMutex, so an
// LP Mode 5 launch cannot interleave between a clear and the following pass-through launch.
// g_params_set maps each such CUfunction to the launch handle it was launched with (CUkernel or
// CUfunction as passed to cuLaunchKernel); g_orig_cnt counts entries per launch handle so that
// pass-through launches can decide without resolving the CUfunction (MaybeParamsSet()).
std::mutex g_params_mtx;
std::unordered_map<CUfunction, const void *> g_params_set;
std::unordered_map<const void *, int> g_orig_cnt;
std::atomic<size_t> g_params_cnt{0};
} // namespace

void Sm120StreamContext::ClearDebuggerParams(CUfunction func)
{
    static const bool always = GetEnvOption("XSCHED_SM120_ALWAYS_CLEAR_PARAMS", false);
    if (!always) {
        std::lock_guard<std::mutex> lk(g_params_mtx);
        auto it = g_params_set.find(func);
        if (it == g_params_set.end()) return;   // already zero
        auto oc = g_orig_cnt.find(it->second);
        if (oc != g_orig_cnt.end() && --oc->second <= 0) g_orig_cnt.erase(oc);
        g_params_set.erase(it);
        g_params_cnt.store(g_params_set.size(), std::memory_order_release);
    }
    uint8_t zeros[24] = {};
    cuXtraSetDebuggerParams(func, zeros, sizeof(zeros));
}

bool Sm120StreamContext::MaybeParamsSet(const void *launch_handle)
{
    static const bool always = GetEnvOption("XSCHED_SM120_ALWAYS_CLEAR_PARAMS", false);
    if (always) return true;
    if (g_params_cnt.load(std::memory_order_acquire) == 0) return false;
    std::lock_guard<std::mutex> lk(g_params_mtx);
    return g_orig_cnt.count(launch_handle) != 0;
}

bool Sm120StreamContext::PrepareLaunch(CUfunction func,
                                       uint32_t num_ctas,
                                       uint32_t threads_per_cta,
                                       uint32_t dyn_smem_bytes,
                                       uint64_t kernel_idx,
                                       Sm120PreemptMode mode,
                                       CUstream stream,
                                       const void *launch_handle)
{
    if (launch_handle == nullptr) launch_handle = func;
    std::lock_guard<std::mutex> lock(mu_);
    if (!engine_->IsReady() && !engine_->Init()) return false;

    const Sm120KernelRecord *rec = guardian_reg_->EnsureInstrumented(func);
    if (!rec) return false;
    if (!EnsureControlCapacity(num_ctas)) return false;

    last_kernel_idx_ = kernel_idx;
    ++launch_seq_;
    if (stream != nullptr) last_stream_ = stream;

    if (mode == Sm120PreemptMode::kTrapFrame) {
        const uint32_t warps = std::max((threads_per_cta + 31u) / 32u, 1u);
        const uint32_t warp_area_off = 0x40u;
        const uint32_t thread_area_off = AlignUp(warp_area_off + warps * 0x160u, 128u);

        // Conservative SM120 register-file ceiling per thread (+16 slack for HW +2 trap regs & quantum).
        const uint32_t reg_ceil = std::min(AlignUp(static_cast<uint32_t>(rec->clamped_regs) + 16u, 16u), 256u);
        const uint32_t gpr_tail_regs = (reg_ceil > 14u) ? (reg_ceil - 14u) : 0u;
        const uint32_t thread_frame_stride = AlignUp(0x78u + gpr_tail_regs * 4u, 16u);

        const uint32_t smem_area_off =
            AlignUp(thread_area_off + threads_per_cta * thread_frame_stride, 128u);
        const uint32_t total_user_smem =
            static_cast<uint32_t>(rec->static_smem_bytes) + dyn_smem_bytes;
        // Hardware allocates SMEM in 4096B quanta AFTER adding the 0x400B driver-reserved region,
        // so SR_SMEMSZ can reach AlignUp(total_user_smem + 0x400, 4096) (e.g., 36352B -> 40960B).
        const uint32_t smem_area =
            (total_user_smem > 0u) ? AlignUp(total_user_smem + 0x400u, 4096u) : 0u;

        const uint32_t lmem_area_off = smem_area_off + smem_area;
        const uint32_t lmem_cap = rec->desc.lmem_cap_bytes;
        const uint32_t lmem_area = AlignUp(threads_per_cta * lmem_cap, 128u);
        const uint32_t slot_stride = AlignUp(lmem_area_off + lmem_area, 128u);

        // Compute resident CTA upper bound on GPU using occupancy when available.
        int max_blocks_sm = 0;
        if (Driver::OccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_sm, func, static_cast<int>(threads_per_cta), dyn_smem_bytes) != CUDA_SUCCESS ||
            max_blocks_sm <= 0) {
            max_blocks_sm = std::max(engine_->MaxThreadsPerSm() / static_cast<int>(threads_per_cta), 1);
        }
        const uint32_t slack = std::max(64u, static_cast<uint32_t>(engine_->SmCount() * 2));
        const uint32_t max_resident =
            static_cast<uint32_t>(engine_->SmCount() * max_blocks_sm) + slack;
        const uint32_t max_slots = std::min(num_ctas, max_resident);

        if (!EnsureArenaPool(static_cast<size_t>(slot_stride) * max_slots)) return false;

        const bool is_restore = is_restore_cmd_active_ && (saved_arena_dev_ != 0);
        const uint64_t effective_arena = static_cast<uint64_t>(is_restore ? saved_arena_dev_ : arena_dev_);
        if (kernel_idx != 0) {
            kernel_arena_map_[kernel_idx] = static_cast<CUdeviceptr>(effective_arena);
            while (kernel_arena_map_.size() > 256) {
                kernel_arena_map_.erase(kernel_arena_map_.begin());
            }
        }

        struct __attribute__((packed)) Layout36
        {
            uint64_t arena_va;
            uint32_t slot_stride_bytes;
            uint32_t max_slots;
            uint32_t thread_area_off;
            uint32_t thread_frame_stride;
            uint32_t smem_area_off;
            uint32_t lmem_area_off;
            uint32_t lmem_cap_bytes;
        } layout{
            effective_arena,
            slot_stride,
            max_slots,
            thread_area_off,
            thread_frame_stride,
            smem_area_off,
            lmem_area_off,
            lmem_cap
        };
        static_assert(sizeof(Layout36) == 36, "Layout36 size mismatch");

        static const bool layout_cache = GetEnvOption("XSCHED_SM120_LAYOUT_CACHE", true);
        if (layout_cache && stream != nullptr && stream == last_layout_stream_ &&
            ctrl_dev_ == last_layout_ctrl_ && memcmp(last_layout_, &layout, sizeof(layout)) == 0) {
            // Identical layout already enqueued on this stream (stream-ordered before this kernel).
        } else if (stream != nullptr && layout_stage_ != nullptr) {
            memcpy(last_layout_, &layout, sizeof(layout));
            last_layout_ctrl_ = ctrl_dev_;
            last_layout_stream_ = stream;
            // Destination ctrl_dev_ is in GPU GDDR7 VRAM: single strictly-async HtoD DMA on stream!
            const size_t slot_off = (layout_stage_cursor_ % kLayoutStageSlots) * 64u;
            ++layout_stage_cursor_;
            void *host_slot = layout_stage_ + slot_off;
            memcpy(host_slot, &layout, sizeof(layout));
            Driver::MemcpyHtoDAsync_v2(
                ctrl_dev_ + offsetof(Sm120ControlHeader, arena_va),
                host_slot, sizeof(layout), stream);
        } else {
            WriteHeaderSliceLocked(offsetof(Sm120ControlHeader, arena_va), &layout, sizeof(layout));
            last_layout_ctrl_ = 0;  // synchronous write: no stream-ordered cache entry
            last_layout_stream_ = nullptr;
        }
    }

    struct
    {
        uint64_t u_control_va;
        uint64_t kdesc_va;
        uint64_t kernel_idx;
    } __attribute__((packed)) dbg{
        static_cast<uint64_t>(ctrl_dev_),
        static_cast<uint64_t>(rec->kdesc_dev),
        kernel_idx
    };
    static_assert(sizeof(dbg) == 24, "DebuggerParams window size must be 24B");
    {
        std::lock_guard<std::mutex> lk(g_params_mtx);
        auto [it, inserted] = g_params_set.emplace(func, launch_handle);
        if (inserted) {
            g_orig_cnt[launch_handle]++;
        } else if (it->second != launch_handle) {
            auto oc = g_orig_cnt.find(it->second);
            if (oc != g_orig_cnt.end() && --oc->second <= 0) g_orig_cnt.erase(oc);
            it->second = launch_handle;
            g_orig_cnt[launch_handle]++;
        }
        g_params_cnt.store(g_params_set.size(), std::memory_order_release);
    }
    cuXtraSetDebuggerParams(func, &dbg, sizeof(dbg));
    return true;
}

bool Sm120StreamContext::PreemptStream(Sm120PreemptMode mode)
{
    std::lock_guard<std::mutex> lock(mu_);
    if (!engine_->IsReady() || ctrl_dev_ == 0) return false;
    ScopedCudaCtx guard(ctx_);

    Sm120ControlHeader hdr{};
    if (!ReadHeaderLocked(&hdr)) return false;

    // If a prior Mode 5 restore is in flight, wait until all State B CTAs self-trapped & restored,
    // and all warps (including collaterally-trapped warps on the same SM) have drained Cave 1.
    const bool was_mode5_restore = (hdr.cmd == kCmdMode5Restore);
    if (was_mode5_restore) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (true) {
            const uint32_t target_restored =
                (hdr.max_slots > 0) ? std::min(hdr.evicted_slots, hdr.max_slots) : hdr.evicted_slots;
            if (hdr.restored_slots >= target_restored && hdr.trap_inflight == 0u) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                XWARN("sm120_trap_engine: timed out waiting for prior restore (%u/%u, max=%u, inflight=%u)",
                      hdr.restored_slots, hdr.evicted_slots, hdr.max_slots, hdr.trap_inflight);
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(25));
            if (!ReadHeaderLocked(&hdr)) break;
        }
    }

    is_restore_cmd_active_ = false;

    // Capture currently active arena VA as fallback; PrepareRestore will resolve the exact
    // arena recorded for active_kernel_idx in kernel_arena_map_.
    saved_arena_dev_ = static_cast<CUdeviceptr>(hdr.arena_va);

    // Reset per-round mutable run state and overflow_err, preserving is_restore_wave (+0x1c).
    // If preempting mid-restore (cmd==6 or 2), ALSO preserve active_kernel_idx (+0x08..+0x0f) so
    // queued K2 exits cleanly at Entry Guardian Slot 51 instead of stealing active_kernel_idx!
    const bool was_restore_wave = (hdr.cmd == kCmdMode5Restore || hdr.cmd == kCmdMode1Replay);
    const uint32_t new_cmd = (mode == Sm120PreemptMode::kEarlyExit)
                           ? kCmdMode1Evict : kCmdMode5Evict;
    ctrl_host_->cmd = new_cmd;
    ctrl_host_->evicted_slots = 0u;
    ctrl_host_->restored_slots = 0u;
    ctrl_host_->drained_cnt = 0u;
    ctrl_host_->mode1_exit_cnt = 0u;
    ctrl_host_->overflow_err = 0u;
    Driver::MemcpyHtoDAsync_v2(ctrl_dev_ + offsetof(Sm120ControlHeader, overflow_err),
                               &ctrl_host_->overflow_err, sizeof(uint32_t), ctrl_op_stream_);
    if (was_restore_wave) {
        Driver::MemcpyHtoDAsync_v2(ctrl_dev_ + offsetof(Sm120ControlHeader, evicted_slots),
                                   &ctrl_host_->evicted_slots, sizeof(uint32_t), ctrl_op_stream_);
        Driver::MemcpyHtoDAsync_v2(ctrl_dev_ + offsetof(Sm120ControlHeader, restored_slots),
                                   &ctrl_host_->restored_slots, 3 * sizeof(uint32_t), ctrl_op_stream_);
    } else {
        ctrl_host_->active_kernel_idx = 0ULL;
        Driver::MemcpyHtoDAsync_v2(ctrl_dev_ + 4, reinterpret_cast<uint8_t *>(ctrl_host_) + 4,
                                   6 * sizeof(uint32_t), ctrl_op_stream_);
    }
    Driver::MemcpyHtoDAsync_v2(ctrl_dev_, ctrl_host_, sizeof(uint32_t), ctrl_op_stream_);
    // Gate mirror: cmd word only. is_restore_wave must NOT be written here: `hdr` may be stale (the
    // post-restore clear kernel can reset the header's is_restore_wave after it was read), and the
    // header itself only gets cmd from this function. Mirrors therefore follow exactly the header's
    // writers: cmd here, {cmd, is_restore_wave} in PrepareRestore / clear kernel / reset.
    WriteGateMirrorCmdAsync(new_cmd, ctrl_op_stream_);
    Driver::StreamSynchronize(ctrl_op_stream_);

    if (was_mode5_restore) {
        std::this_thread::sleep_for(std::chrono::microseconds(30));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (ReadHeaderLocked(&hdr) && hdr.trap_inflight != 0u) {
            if (std::chrono::steady_clock::now() >= deadline) {
                XWARN("sm120_trap_engine: timed out waiting for trap_inflight drain (%u)",
                      hdr.trap_inflight);
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(25));
        }
        std::this_thread::sleep_for(std::chrono::microseconds(30));
    }

    if (!engine_->FireEvictPulse()) {
        // At most one warning per second (a stopped proxy would otherwise log every pulse).
        static std::atomic<int64_t> last_warn_ms{-1000000};
        static std::atomic<uint64_t> failed{0};
        const uint64_t n = failed.fetch_add(1) + 1;
        const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        int64_t prev = last_warn_ms.load();
        if (now_ms - prev >= 1000 && last_warn_ms.compare_exchange_strong(prev, now_ms)) {
            XWARN("sm120_trap_engine: BAR0 pulse failed (%llu so far; run as root or start xsched-bar0d; "
                  "waiting for the kernel instead)", static_cast<unsigned long long>(n));
        }
        return false;
    }

    // Progress-aware bounded confirmation: on the normal fast path StreamQuery returns
    // CUDA_SUCCESS in < 1 ms with ZERO extra DMA overhead; only sample counters via CE DMA
    // every 2 ms if the stream stays active (display/shared GPU recovery protection).
    if (last_stream_ != nullptr) {
        uint32_t last_prog = 0;
        auto last_prog_tp = std::chrono::steady_clock::now();
        auto last_sample_tp = last_prog_tp;
        constexpr int kMaxRecoveryPulses = 8;
        int retries = 0;
        while (Driver::StreamQuery(last_stream_) == CUDA_ERROR_NOT_READY) {
            const auto now = std::chrono::steady_clock::now();
            const int64_t since_sample_us =
                std::chrono::duration_cast<std::chrono::microseconds>(now - last_sample_tp).count();
            if (since_sample_us >= 2000LL) {
                last_sample_tp = now;
                if (ReadHeaderLocked(&hdr)) {
                    const uint32_t prog = hdr.evicted_slots + hdr.drained_cnt + hdr.mode1_exit_cnt;
                    if (prog != last_prog) {
                        last_prog = prog;
                        last_prog_tp = now;
                    } else {
                        const int64_t stall_us =
                            std::chrono::duration_cast<std::chrono::microseconds>(now - last_prog_tp).count();
                        const int64_t threshold_us = std::min<int64_t>(15000LL << std::min(retries, 2), 60000LL);
                        if (stall_us >= threshold_us) {
                            if (retries >= kMaxRecoveryPulses) {
                                XWARN("sm120_trap_engine: stream still busy after %d recovery pulses "
                                      "(ev=%u dr=%u m1=%u cmd=%u active=%llu max_ctas=%u inflight=%u ctrl=%#llx)",
                                      retries, hdr.evicted_slots, hdr.drained_cnt, hdr.mode1_exit_cnt,
                                      hdr.cmd, (unsigned long long)hdr.active_kernel_idx, hdr.max_ctas,
                                      hdr.trap_inflight, (unsigned long long)ctrl_dev_);
                                break;
                            }
                            ++retries;
                            XDEBG("sm120_trap_engine: recovery pulse #%d after %lld us stall "
                                  "(ev=%u dr=%u m1=%u)",
                                  retries, static_cast<long long>(stall_us),
                                  hdr.evicted_slots, hdr.drained_cnt, hdr.mode1_exit_cnt);
                            engine_->FireEvictPulse();
                            last_prog_tp = std::chrono::steady_clock::now();
                        }
                    }
                }
            }
            std::this_thread::yield();
        }
    }
    return true;
}

uint64_t Sm120StreamContext::PrepareRestore(Sm120PreemptMode mode)
{
    std::lock_guard<std::mutex> lock(mu_);
    if (ctrl_dev_ == 0) return 0;
    ScopedCudaCtx guard(ctx_);

    Sm120ControlHeader hdr{};
    (void)ReadHeaderLocked(&hdr);

    uint64_t idx = hdr.active_kernel_idx;
    if (idx == 0) idx = last_kernel_idx_;

    // Resolve exact arena VA used when the evicted kernel was originally launched so
    // back-to-back multi-kernel arena growth can never mismatch Evict vs Restore.
    if (idx != 0) {
        auto it = kernel_arena_map_.find(idx);
        if (it != kernel_arena_map_.end() && it->second != 0) {
            saved_arena_dev_ = it->second;
        }
    }

    const uint32_t restore_cmd = (mode == Sm120PreemptMode::kEarlyExit)
                               ? kCmdMode1Replay : kCmdMode5Restore;
    is_restore_cmd_active_ = (mode == Sm120PreemptMode::kTrapFrame);

    // Update restored_slots=0 (+0x10), is_restore_wave=1 (+0x1c), then cmd=restore_cmd (+0x00)
    // atomically in GPU VRAM via high-priority Copy Engine stream.
    ctrl_host_->restored_slots = 0u;
    ctrl_host_->is_restore_wave = 1u;
    ctrl_host_->cmd = restore_cmd;
    Driver::MemcpyHtoDAsync_v2(ctrl_dev_ + offsetof(Sm120ControlHeader, restored_slots),
                               &ctrl_host_->restored_slots, sizeof(uint32_t), ctrl_op_stream_);
    Driver::MemcpyHtoDAsync_v2(ctrl_dev_ + offsetof(Sm120ControlHeader, is_restore_wave),
                               &ctrl_host_->is_restore_wave, sizeof(uint32_t), ctrl_op_stream_);
    Driver::MemcpyHtoDAsync_v2(ctrl_dev_ + offsetof(Sm120ControlHeader, cmd),
                               &ctrl_host_->cmd, sizeof(uint32_t), ctrl_op_stream_);
    WriteGateMirrorAsync(restore_cmd, 1u, ctrl_op_stream_);   // before the restore-wave relaunch
    Driver::StreamSynchronize(ctrl_op_stream_);
    return idx;
}

void Sm120StreamContext::WriteGateMirrorCmdAsync(uint32_t cmd, CUstream stream)
{
    if (ctrl_dev_ == 0) return;
    Driver::MemsetD2D32Async(ctrl_dev_ - kGateMirrorBytes, kGateMirrorStride, cmd, 1, kGateMirrorEntries, stream);
}

void Sm120StreamContext::WriteGateMirrorAsync(uint32_t cmd, uint32_t is_restore_wave, CUstream stream)
{
    if (ctrl_dev_ == 0) return;
    const CUdeviceptr base = ctrl_dev_ - kGateMirrorBytes;
    // Strided 32-bit memsets: word 0 (cmd) and word 1 (is_restore_wave) of every 128 B mirror line.
    Driver::MemsetD2D32Async(base, kGateMirrorStride, cmd, 1, kGateMirrorEntries, stream);
    Driver::MemsetD2D32Async(base + 4, kGateMirrorStride, is_restore_wave, 1, kGateMirrorEntries, stream);
}

bool Sm120StreamContext::EnqueuePostRestoreClear(CUstream stream)
{
    std::lock_guard<std::mutex> lock(mu_);
    saved_arena_dev_ = 0;
    is_restore_cmd_active_ = false;
    if (ctrl_dev_ == 0) return false;
    ScopedCudaCtx guard(ctx_);

    CUfunction fn = engine_->CondClearFunc();
    if (fn != nullptr) {
        // Launch 1-thread GPU-native conditional clear (~2 us, strictly ordered on stream):
        // clears ctrl_dev_[0..31] ONLY if cmd == 6 or 2 (no-op if Preempt #2 flipped cmd to 5/1!).
        CUdeviceptr dev_arg = ctrl_dev_;
        void *params[] = { &dev_arg };
        return Driver::LaunchKernel(fn, 1, 1, 1, kGateMirrorEntries, 1, 1, 0, stream, params, nullptr) ==
               CUDA_SUCCESS;
    }
    return Driver::MemsetD8Async(ctrl_dev_ - kGateMirrorBytes, 0, kGateMirrorBytes, stream) == CUDA_SUCCESS &&
           Driver::MemsetD8Async(ctrl_dev_, 0, 32, stream) == CUDA_SUCCESS;
}

void Sm120StreamContext::ResetControlSync()
{
    std::lock_guard<std::mutex> lock(mu_);
    saved_arena_dev_ = 0;
    is_restore_cmd_active_ = false;
    if (ctrl_dev_ == 0 || ctrl_op_stream_ == nullptr) return;
    ScopedCudaCtx guard(ctx_);

    Driver::MemsetD8Async(ctrl_dev_ - kGateMirrorBytes, 0, kGateMirrorBytes, ctrl_op_stream_);
    Driver::MemsetD8Async(ctrl_dev_, 0, 32, ctrl_op_stream_);
    Driver::MemsetD8Async(ctrl_dev_ + offsetof(Sm120ControlHeader, overflow_err), 0, 8, ctrl_op_stream_);
    const size_t entries_bytes = static_cast<size_t>(ctrl_cta_cap_) * sizeof(Sm120CtaEntry);
    if (entries_bytes > 0) {
        Driver::MemsetD8Async(ctrl_dev_ + sizeof(Sm120ControlHeader), 0, entries_bytes, ctrl_op_stream_);
    }
    Driver::MemsetD32Async(ctrl_dev_ + offsetof(Sm120ControlHeader, max_ctas),
                           ctrl_cta_cap_, 1, ctrl_op_stream_);
    Driver::StreamSynchronize(ctrl_op_stream_);
}

Sm120StreamTelemetry Sm120StreamContext::Telemetry() const
{
    std::lock_guard<std::mutex> lock(mu_);
    Sm120StreamTelemetry t;
    Sm120ControlHeader hdr{};
    if (!ReadHeaderLocked(&hdr)) return t;
    t.cmd               = hdr.cmd;
    t.evicted_slots     = hdr.evicted_slots;
    t.restored_slots    = hdr.restored_slots;
    t.drained_cnt       = hdr.drained_cnt;
    t.mode1_exit_cnt    = hdr.mode1_exit_cnt;
    t.overflow_err      = hdr.overflow_err;
    t.is_restore_wave   = hdr.is_restore_wave;
    t.active_kernel_idx = hdr.active_kernel_idx;
    t.arena_bytes       = arena_bytes_;
    return t;
}

void Sm120StreamContext::DebugDumpEvicted(const Sm120KernelRecord *rec, uint32_t num_ctas,
                                          uint32_t threads_per_cta, uint32_t dyn_smem_bytes,
                                          uint32_t max_ctas) const
{
    std::lock_guard<std::mutex> lock(mu_);
    if (ctrl_dev_ == 0 || ctrl_op_stream_ == nullptr || rec == nullptr) return;
    const CUdeviceptr ep_inst = rec->ep_inst;
    const CUdeviceptr ep_orig = rec->ep_orig;
    // Recompute the Mode 5 slot layout exactly as PrepareLaunch() does (diagnostic copy).
    const uint32_t lw = std::max((threads_per_cta + 31u) / 32u, 1u);
    const uint32_t l_thread_area_off = AlignUp(0x40u + lw * 0x160u, 128u);
    const uint32_t l_reg_ceil = std::min(AlignUp(static_cast<uint32_t>(rec->clamped_regs) + 16u, 16u), 256u);
    const uint32_t l_gpr_tail = (l_reg_ceil > 14u) ? (l_reg_ceil - 14u) : 0u;
    const uint32_t l_thr_stride = AlignUp(0x78u + l_gpr_tail * 4u, 16u);
    const uint32_t l_smem_off = AlignUp(l_thread_area_off + threads_per_cta * l_thr_stride, 128u);
    const uint32_t l_user_smem = static_cast<uint32_t>(rec->static_smem_bytes) + dyn_smem_bytes;
    const uint32_t l_smem_area = (l_user_smem > 0u) ? AlignUp(l_user_smem + 0x400u, 4096u) : 0u;
    const uint32_t l_lmem_off = l_smem_off + l_smem_area;
    const uint32_t l_slot_stride =
        AlignUp(l_lmem_off + AlignUp(threads_per_cta * rec->desc.lmem_cap_bytes, 128u), 128u);
    ScopedCudaCtx guard(ctx_);
    Sm120ControlHeader hdr{};
    if (!ReadHeaderLocked(&hdr)) return;
    const uint32_t n = std::min(num_ctas, ctrl_cta_cap_);
    std::vector<Sm120CtaEntry> ents(n);
    if (n == 0 ||
        Driver::MemcpyDtoHAsync_v2(ents.data(), ctrl_dev_ + sizeof(Sm120ControlHeader),
                                   static_cast<size_t>(n) * sizeof(Sm120CtaEntry),
                                   ctrl_op_stream_) != CUDA_SUCCESS ||
        Driver::StreamSynchronize(ctrl_op_stream_) != CUDA_SUCCESS) {
        return;
    }
    uint32_t cnt[8] = {};
    for (const auto &e : ents) cnt[std::min(e.state, 7u)]++;
    XINFO("sm120_dump: active-kernel layout stride=%u thr_off=%#x thr_stride=%u smem_off=%#x smem_area=%u "
          "(regs=%d static_smem=%d dyn_smem=%u thr=%u)", l_slot_stride, l_thread_area_off, l_thr_stride,
          l_smem_off, l_smem_area, rec->clamped_regs, rec->static_smem_bytes, dyn_smem_bytes,
          threads_per_cta);
    XINFO("sm120_dump: hdr(last DMA) cmd=%u evicted=%u drained=%u active=%llu arena=%#llx stride=%u "
          "max_slots=%u thr_off=%#x thr_stride=%u smem_off=%#x | states: 0=%u 1=%u 2=%u 5=%u other=%u",
          hdr.cmd, hdr.evicted_slots, hdr.drained_cnt,
          static_cast<unsigned long long>(hdr.active_kernel_idx),
          static_cast<unsigned long long>(hdr.arena_va), hdr.slot_stride_bytes, hdr.max_slots,
          hdr.thread_area_off, hdr.thread_frame_stride, hdr.smem_area_off,
          cnt[0], cnt[1], cnt[2], cnt[5], n - cnt[0] - cnt[1] - cnt[2] - cnt[5]);
    uint32_t shown = 0;
    const uint32_t warps = std::max((threads_per_cta + 31u) / 32u, 1u);
    for (uint32_t b = 0; b < n && shown < max_ctas; ++b) {
        if (ents[b].state != 5u) continue;
        ++shown;
        const CUdeviceptr arena = saved_arena_dev_ ? saved_arena_dev_ : static_cast<CUdeviceptr>(hdr.arena_va);
        const CUdeviceptr slot = arena + static_cast<CUdeviceptr>(ents[b].slot_id) * l_slot_stride;
        std::string pcs;
        for (uint32_t w = 0; w < warps && w < 32; ++w) {
            if (((ents[b].warp_mask >> w) & 1u) == 0) { pcs += "w" + std::to_string(w) + ":dead "; continue; }
            // Lanes 0 and 31 of each live warp: 112B HW frame, TRAP_RETURN_PC at +0x18.
            for (uint32_t lane : {0u, 31u}) {
                const uint32_t tid = w * 32u + lane;
                if (tid >= threads_per_cta) break;
                uint64_t pc = 0;
                const CUdeviceptr fr = slot + l_thread_area_off +
                                       static_cast<CUdeviceptr>(tid) * l_thr_stride;
                Driver::MemcpyDtoHAsync_v2(&pc, fr + 0x18, sizeof(pc), ctrl_op_stream_);
                Driver::StreamSynchronize(ctrl_op_stream_);
                char buf[64];
                if (pc >= ep_inst + 0x500 && pc < ep_inst + 0x500 + (1u << 24)) {
                    snprintf(buf, sizeof(buf), "w%u.%u:+%#llx ", w, lane,
                             static_cast<unsigned long long>(pc - ep_inst - 0x500));
                } else if (pc >= ep_inst && pc < ep_inst + 0x500) {
                    snprintf(buf, sizeof(buf), "w%u.%u:G%#llx ", w, lane,
                             static_cast<unsigned long long>(pc - ep_inst));
                } else if (ep_orig != 0 && pc >= ep_orig && pc < ep_orig + (1u << 24)) {
                    snprintf(buf, sizeof(buf), "w%u.%u:ORIG+%#llx ", w, lane,
                             static_cast<unsigned long long>(pc - ep_orig));
                } else {
                    snprintf(buf, sizeof(buf), "w%u.%u:abs%#llx ", w, lane,
                             static_cast<unsigned long long>(pc));
                }
                pcs += buf;
            }
            if (pcs.size() > 600) { pcs += "..."; break; }
        }
        XINFO("sm120_dump: cta=%u slot=%u warp_mask=%#x entry_dec=%u pcs=[%s]", b, ents[b].slot_id,
              ents[b].warp_mask, ents[b].entry_dec, pcs.c_str());
    }
}

bool Sm120StreamContext::ReadCtaEntries(std::vector<Sm120CtaEntry> &out, uint32_t count) const
{
    std::lock_guard<std::mutex> lock(mu_);
    if (ctrl_dev_ == 0 || ctrl_op_stream_ == nullptr || count > ctrl_cta_cap_) return false;
    out.resize(count);
    if (count == 0) return true;
    ScopedCudaCtx guard(ctx_);
    if (Driver::MemcpyDtoHAsync_v2(out.data(), ctrl_dev_ + sizeof(Sm120ControlHeader),
                                   static_cast<size_t>(count) * sizeof(Sm120CtaEntry),
                                   ctrl_op_stream_) != CUDA_SUCCESS) {
        return false;
    }
    return Driver::StreamSynchronize(ctrl_op_stream_) == CUDA_SUCCESS;
}

} // namespace xsched::cuda::sm120
