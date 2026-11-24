#include "xsched/cuda/hal/level3/sm120_trap_engine.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

#include <cuxtra/cuxtra.h>

#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/level3/sm120_sass.h"
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

    // Header: compute r_start = min(SR_REGALLOC - 3, 252). Skip if < 14.
    c.push_back(s2r(4, kSrRegAlloc));
    c.push_back(iadd_imm(4, 4, static_cast<uint32_t>(-3)));
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
}

bool Sm120TrapEngine::Init()
{
    std::lock_guard<std::mutex> lock(mu_);
    if (ready_) return true;
    if (!arena_.Discover(true)) return false;
    if (!EmitStaticHandler()) {
        arena_.RestoreAll();
        return false;
    }
    arena_.EnsureBar0Mapped();
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
    c0.push_back(isetp_lt_u64_imm(6, kGuardianRestorePathOff));
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
    c0.push_back(isetp_lt_u64_imm(6, kGuardianReplayCIdxOff));
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
// Sm120StreamContext
// ==========================================================================

namespace
{

constexpr size_t kLayoutStageSlots = 256;
constexpr size_t kLayoutStageBytes = kLayoutStageSlots * 64;

struct PostRestoreClearArgs
{
    std::weak_ptr<void> state;
    uint64_t expected_epoch = 0;
    void (*clear_fn)(const std::shared_ptr<void> &, uint64_t) = nullptr;
};

void CUDA_CB PostRestoreClearHostCb(void *ud)
{
    auto *args = static_cast<PostRestoreClearArgs *>(ud);
    if (!args) return;
    auto st = args->state.lock();
    const uint64_t expected = args->expected_epoch;
    auto fn = args->clear_fn;
    delete args;
    if (st && fn) fn(st, expected);
}

} // namespace

Sm120StreamContext::Sm120StreamContext(CUcontext ctx, CUdevice dev,
                                       std::shared_ptr<Sm120TrapEngine> engine)
    : ctx_(ctx), dev_(dev), engine_(std::move(engine)),
      clear_state_(std::make_shared<PostRestoreClearState>())
{
    guardian_reg_ = Sm120GuardianRegistry::Instance(ctx_);
    void *stage = nullptr;
    if (Driver::MemHostAlloc(&stage, kLayoutStageBytes, CU_MEMHOSTALLOC_DEVICEMAP) == CUDA_SUCCESS && stage) {
        memset(stage, 0, kLayoutStageBytes);
        layout_stage_ = reinterpret_cast<uint8_t *>(stage);
    }
    if (Driver::MemAlloc_v2(&layout_stage_dev_, kLayoutStageBytes + 64u) == CUDA_SUCCESS &&
        layout_stage_dev_ != 0) {
        Driver::MemsetD8_v2(layout_stage_dev_, 0, kLayoutStageBytes + 64u);
    }
    EnsureControlCapacity(65536);
}

Sm120StreamContext::~Sm120StreamContext()
{
    if (clear_state_) {
        std::lock_guard<std::mutex> lock(clear_state_->mu);
        ++clear_state_->epoch;
        clear_state_->ctrl_host = nullptr;
    }
    CUcontext cur = nullptr;
    if (Driver::CtxGetCurrent(&cur) == CUDA_ERROR_DEINITIALIZED || cur != ctx_) {
        return;
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
    for (void *old_ctrl : retired_ctrls_) {
        if (old_ctrl != nullptr) Driver::MemFreeHost(old_ctrl);
    }
    retired_ctrls_.clear();
    if (ctrl_host_ != nullptr) {
        Driver::MemFreeHost(ctrl_host_);
        ctrl_host_ = nullptr;
        ctrl_dev_ = 0;
    }
    if (layout_stage_ != nullptr) {
        Driver::MemFreeHost(layout_stage_);
        layout_stage_ = nullptr;
    }
    if (layout_stage_dev_ != 0) {
        Driver::MemFree_v2(layout_stage_dev_);
        layout_stage_dev_ = 0;
    }
}

bool Sm120StreamContext::EnsureControlCapacity(uint32_t num_ctas)
{
    const uint32_t needed_ctas = std::max(num_ctas, 65536u);
    if (ctrl_host_ != nullptr && needed_ctas <= ctrl_cta_cap_) return true;

    // If growing u_control after kernels have already been queued on last_stream_,
    // wait for the stream so no in-flight kernel references the retired ctrl_dev_.
    if (ctrl_host_ != nullptr && last_stream_ != nullptr) {
        CUcontext cur = nullptr;
        if (Driver::CtxGetCurrent(&cur) == CUDA_SUCCESS && cur == ctx_) {
            (void)Driver::StreamSynchronize(last_stream_);
        }
    }

    const size_t new_bytes = AlignUpSize(sizeof(Sm120ControlHeader) +
                                         static_cast<size_t>(needed_ctas) * sizeof(Sm120CtaEntry),
                                         4096ULL);
    void *new_host = nullptr;
    CUresult r = Driver::MemHostAlloc(&new_host, new_bytes, CU_MEMHOSTALLOC_DEVICEMAP);
    if (r != CUDA_SUCCESS || !new_host) return false;
    memset(new_host, 0, new_bytes);

    CUdeviceptr new_dev = reinterpret_cast<CUdeviceptr>(new_host);
    (void)Driver::MemHostGetDevicePointer_v2(&new_dev, new_host, 0);

    auto *new_hdr = reinterpret_cast<Sm120ControlHeader *>(new_host);
    if (ctrl_host_ != nullptr) {
        memcpy(new_hdr, ctrl_host_, sizeof(Sm120ControlHeader));
        const size_t old_entries = static_cast<size_t>(ctrl_cta_cap_) * sizeof(Sm120CtaEntry);
        if (old_entries > 0) {
            memcpy(reinterpret_cast<uint8_t *>(new_hdr) + sizeof(Sm120ControlHeader),
                   reinterpret_cast<uint8_t *>(ctrl_host_) + sizeof(Sm120ControlHeader),
                   old_entries);
        }
        retired_ctrls_.push_back(ctrl_host_);
    }

    new_hdr->max_ctas = needed_ctas;
    ctrl_host_ = new_hdr;
    ctrl_dev_  = new_dev;
    ctrl_bytes_ = new_bytes;
    ctrl_cta_cap_ = needed_ctas;
    if (clear_state_) {
        std::lock_guard<std::mutex> lock(clear_state_->mu);
        clear_state_->ctrl_host = ctrl_host_;
    }
    return true;
}

bool Sm120StreamContext::EnsureArenaPool(size_t required_bytes)
{
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

void Sm120StreamContext::ClearDebuggerParams(CUfunction func)
{
    uint8_t zeros[24] = {};
    cuXtraSetDebuggerParams(func, zeros, sizeof(zeros));
}

bool Sm120StreamContext::PrepareLaunch(CUfunction func,
                                       uint32_t num_ctas,
                                       uint32_t threads_per_cta,
                                       uint32_t dyn_smem_bytes,
                                       uint64_t kernel_idx,
                                       Sm120PreemptMode mode,
                                       CUstream stream)
{
    std::lock_guard<std::mutex> lock(mu_);
    if (!engine_->IsReady() && !engine_->Init()) return false;

    const Sm120KernelRecord *rec = guardian_reg_->EnsureInstrumented(func);
    if (!rec) return false;
    if (!EnsureControlCapacity(num_ctas)) return false;

    last_kernel_idx_ = kernel_idx;
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
        const uint32_t smem_area =
            AlignUp(static_cast<uint32_t>(rec->static_smem_bytes) + dyn_smem_bytes, 4096u);

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
        const uint32_t max_resident =
            static_cast<uint32_t>(engine_->SmCount() * max_blocks_sm) + 16u;
        const uint32_t max_slots = std::min(num_ctas, max_resident);

        if (!EnsureArenaPool(static_cast<size_t>(slot_stride) * max_slots)) return false;

        const bool is_restore =
            (ctrl_host_ != nullptr) &&
            (__atomic_load_n(&ctrl_host_->cmd, __ATOMIC_ACQUIRE) == kCmdMode5Restore) &&
            (saved_arena_dev_ != 0);
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

        if (stream != nullptr && layout_stage_ != nullptr && layout_stage_dev_ != 0) {
            // Queue layout update FIFO on stream via Device VRAM staging:
            // Pinned-Host -> Device (HtoDAsync) -> Pinned-Host ctrl_host_ (DtoHAsync).
            // Direct HtoDAsync to host-mapped ctrl_dev_ is Host->Host and synchronously blocks CPU!
            const size_t slot_off = (layout_stage_cursor_ % kLayoutStageSlots) * 64u;
            ++layout_stage_cursor_;
            void *host_slot = layout_stage_ + slot_off;
            const CUdeviceptr dev_slot = layout_stage_dev_ + slot_off;
            memcpy(host_slot, &layout, sizeof(layout));
            Driver::MemcpyHtoDAsync_v2(dev_slot, host_slot, sizeof(layout), stream);
            Driver::MemcpyDtoHAsync_v2(
                reinterpret_cast<uint8_t *>(ctrl_host_) + offsetof(Sm120ControlHeader, arena_va),
                dev_slot, sizeof(layout), stream);
        } else {
            memcpy(&ctrl_host_->arena_va, &layout, sizeof(layout));
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
    cuXtraSetDebuggerParams(func, &dbg, sizeof(dbg));
    return true;
}

bool Sm120StreamContext::PreemptStream(Sm120PreemptMode mode)
{
    std::lock_guard<std::mutex> lock(mu_);
    if (!engine_->IsReady() || !ctrl_host_) return false;

    // If a prior Mode 5 restore is in flight, wait until all State B CTAs self-trapped & restored,
    // and all warps (including collaterally-trapped warps on the same SM) have drained Cave 1.
    const bool was_mode5_restore = (__atomic_load_n(&ctrl_host_->cmd, __ATOMIC_ACQUIRE) == kCmdMode5Restore);
    if (was_mode5_restore) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (__atomic_load_n(&ctrl_host_->restored_slots, __ATOMIC_ACQUIRE) <
                   __atomic_load_n(&ctrl_host_->evicted_slots, __ATOMIC_ACQUIRE) ||
               __atomic_load_n(&ctrl_host_->trap_inflight, __ATOMIC_ACQUIRE) != 0u) {
            if (std::chrono::steady_clock::now() >= deadline) {
                XWARN("sm120_trap_engine: timed out waiting for prior restore (%u/%u, inflight=%u)",
                      ctrl_host_->restored_slots, ctrl_host_->evicted_slots, ctrl_host_->trap_inflight);
                break;
            }
            std::this_thread::yield();
        }
    }

    // Invalidate any pending PostRestoreClearHostCb from an earlier restore wave so that
    // when this preemption drains the stream early, the stale callback does not wipe
    // our newly recorded eviction counters [0x00..0x1f].
    if (clear_state_) {
        std::lock_guard<std::mutex> clk(clear_state_->mu);
        ++clear_state_->epoch;
    }

    // Capture currently active arena VA as fallback; PrepareRestore will resolve the exact
    // arena recorded for active_kernel_idx in kernel_arena_map_.
    saved_arena_dev_ = static_cast<CUdeviceptr>(
        __atomic_load_n(&ctrl_host_->arena_va, __ATOMIC_ACQUIRE));

    // Reset per-round mutable run state [0x00..0x18] before firing the 1-shot pulse.
    // Note: is_restore_wave (+0x1c) is preserved across multi-round preempts until post-restore clear.
    __atomic_store_n(&ctrl_host_->evicted_slots, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&ctrl_host_->active_kernel_idx, 0ULL, __ATOMIC_RELEASE);
    __atomic_store_n(&ctrl_host_->restored_slots, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&ctrl_host_->drained_cnt, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&ctrl_host_->mode1_exit_cnt, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&ctrl_host_->overflow_err, 0u, __ATOMIC_RELEASE);

    const uint32_t new_cmd = (mode == Sm120PreemptMode::kEarlyExit)
                           ? kCmdMode1Evict : kCmdMode5Evict;
    __atomic_store_n(&ctrl_host_->cmd, new_cmd, __ATOMIC_SEQ_CST);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    if (was_mode5_restore) {
        // Any warp that entered Cave 1 right around the cmd store will re-read cmd == new_cmd
        // and self-dispatch into Cave 0 Evict; wait for trap_inflight to hit 0 and give
        // pure-L1 .text.traphandler epilogues (~50 instrs, zero PCIe) margin to finish RTT.
        std::this_thread::sleep_for(std::chrono::microseconds(30));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (__atomic_load_n(&ctrl_host_->trap_inflight, __ATOMIC_ACQUIRE) != 0u) {
            if (std::chrono::steady_clock::now() >= deadline) {
                XWARN("sm120_trap_engine: timed out waiting for trap_inflight drain (%u)",
                      ctrl_host_->trap_inflight);
                break;
            }
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(30));
    }

    if (!engine_->FireEvictPulse()) {
        XWARN("sm120_trap_engine: BAR0 pulse failed (check resource0 permissions)");
        return false;
    }

    // Progress-aware bounded confirmation: on an idle GPU the single pulse above drains the
    // stream immediately (0 retries, 0 extra latency). On a display/shared GPU (where another
    // context briefly owns GR during the 1 us pulse) or during back-to-back Restore->Evict
    // under heavy PCIe atomic contention (where 1 SM may still be finishing RTT), issue a
    // strictly bounded recovery pulse only when eviction counters stall while the stream is
    // still active.
    CUcontext cur = nullptr;
    if (last_stream_ != nullptr &&
        Driver::CtxGetCurrent(&cur) == CUDA_SUCCESS && cur == ctx_) {
        auto read_progress = [this]() -> uint32_t {
            return __atomic_load_n(&ctrl_host_->evicted_slots, __ATOMIC_ACQUIRE) +
                   __atomic_load_n(&ctrl_host_->drained_cnt, __ATOMIC_ACQUIRE) +
                   __atomic_load_n(&ctrl_host_->mode1_exit_cnt, __ATOMIC_ACQUIRE);
        };
        uint32_t last_prog = read_progress();
        auto last_prog_tp = std::chrono::steady_clock::now();
        constexpr int kMaxRecoveryPulses = 8;
        int retries = 0;
        while (Driver::StreamQuery(last_stream_) == CUDA_ERROR_NOT_READY) {
            const uint32_t prog = read_progress();
            const auto now = std::chrono::steady_clock::now();
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
                              "(ev=%u dr=%u m1=%u)",
                              retries, ctrl_host_->evicted_slots,
                              ctrl_host_->drained_cnt, ctrl_host_->mode1_exit_cnt);
                        break;
                    }
                    ++retries;
                    XDEBG("sm120_trap_engine: recovery pulse #%d after %lld us stall "
                          "(ev=%u dr=%u m1=%u)",
                          retries, static_cast<long long>(stall_us),
                          ctrl_host_->evicted_slots, ctrl_host_->drained_cnt,
                          ctrl_host_->mode1_exit_cnt);
                    engine_->FireEvictPulse();
                    last_prog_tp = std::chrono::steady_clock::now();
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
    if (!ctrl_host_) return 0;

    uint64_t idx = __atomic_load_n(&ctrl_host_->active_kernel_idx, __ATOMIC_ACQUIRE);
    if (idx == 0) idx = last_kernel_idx_;

    // Resolve exact arena VA used when the evicted kernel was originally launched so
    // back-to-back multi-kernel arena growth can never mismatch Evict vs Restore.
    if (idx != 0) {
        auto it = kernel_arena_map_.find(idx);
        if (it != kernel_arena_map_.end() && it->second != 0) {
            saved_arena_dev_ = it->second;
        }
    }

    if (clear_state_) {
        std::lock_guard<std::mutex> clk(clear_state_->mu);
        last_restore_epoch_ = ++clear_state_->epoch;
    }

    const uint32_t restore_cmd = (mode == Sm120PreemptMode::kEarlyExit)
                               ? kCmdMode1Replay : kCmdMode5Restore;
    __atomic_store_n(&ctrl_host_->restored_slots, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&ctrl_host_->is_restore_wave, 1u, __ATOMIC_RELEASE);
    __atomic_store_n(&ctrl_host_->cmd, restore_cmd, __ATOMIC_SEQ_CST);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    return idx;
}

bool Sm120StreamContext::EnqueuePostRestoreClear(CUstream stream)
{
    std::lock_guard<std::mutex> lock(mu_);
    saved_arena_dev_ = 0;
    if (ctrl_host_ == nullptr || !clear_state_) return false;
    auto *args = new PostRestoreClearArgs{
        clear_state_,
        last_restore_epoch_,
        [](const std::shared_ptr<void> &raw, uint64_t expected) {
            auto st = std::static_pointer_cast<PostRestoreClearState>(raw);
            std::lock_guard<std::mutex> clk(st->mu);
            if (st->epoch != expected || st->ctrl_host == nullptr) return;
            memset(st->ctrl_host, 0, 32);
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
        }
    };
    if (Driver::LaunchHostFunc(stream, PostRestoreClearHostCb, args) != CUDA_SUCCESS) {
        delete args;
        return false;
    }
    return true;
}

void Sm120StreamContext::ResetControlSync()
{
    std::lock_guard<std::mutex> lock(mu_);
    saved_arena_dev_ = 0;
    if (clear_state_) {
        std::lock_guard<std::mutex> clk(clear_state_->mu);
        ++clear_state_->epoch;
    }
    if (!ctrl_host_) return;
    memset(ctrl_host_, 0, 32);
    ctrl_host_->overflow_err = 0;
    ctrl_host_->trap_inflight = 0;
    const size_t entries_bytes = static_cast<size_t>(ctrl_cta_cap_) * sizeof(Sm120CtaEntry);
    if (entries_bytes > 0) {
        memset(reinterpret_cast<uint8_t *>(ctrl_host_) + sizeof(Sm120ControlHeader), 0, entries_bytes);
    }
    ctrl_host_->max_ctas = ctrl_cta_cap_;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

Sm120StreamTelemetry Sm120StreamContext::Telemetry() const
{
    std::lock_guard<std::mutex> lock(mu_);
    Sm120StreamTelemetry t;
    if (!ctrl_host_) return t;
    t.cmd               = __atomic_load_n(&ctrl_host_->cmd, __ATOMIC_ACQUIRE);
    t.evicted_slots     = __atomic_load_n(&ctrl_host_->evicted_slots, __ATOMIC_ACQUIRE);
    t.restored_slots    = __atomic_load_n(&ctrl_host_->restored_slots, __ATOMIC_ACQUIRE);
    t.drained_cnt       = __atomic_load_n(&ctrl_host_->drained_cnt, __ATOMIC_ACQUIRE);
    t.mode1_exit_cnt    = __atomic_load_n(&ctrl_host_->mode1_exit_cnt, __ATOMIC_ACQUIRE);
    t.overflow_err      = __atomic_load_n(&ctrl_host_->overflow_err, __ATOMIC_ACQUIRE);
    t.is_restore_wave   = __atomic_load_n(&ctrl_host_->is_restore_wave, __ATOMIC_ACQUIRE);
    t.active_kernel_idx = __atomic_load_n(&ctrl_host_->active_kernel_idx, __ATOMIC_ACQUIRE);
    t.arena_bytes       = arena_bytes_;
    return t;
}

} // namespace xsched::cuda::sm120
