#pragma once

#include <cstdint>

namespace xsched::cuda::sm120::sass
{

/// @brief 128-bit Blackwell (sm_120) SASS instruction word pair.
struct Ins
{
    uint64_t w0;
    uint64_t w1;
};

// Predicate encodings (low 4 bits placed at w0[15:12]).
inline constexpr int kP0    = 0;
inline constexpr int kP1    = 1;
inline constexpr int kP2    = 2;
inline constexpr int kP3    = 3;
inline constexpr int kP4    = 4;
inline constexpr int kP5    = 5;
inline constexpr int kP6    = 6;
inline constexpr int kPT    = 7;
inline constexpr int kNotP0 = 8;

inline constexpr int kRegZero = 0xff; // RZ / URZ

// SM120 Special Register indices (verified on RTX 5080 + nvdisasm 12.9).
inline constexpr uint32_t kSrLaneId            = 0x00;
inline constexpr uint32_t kSrTidX              = 0x21;
inline constexpr uint32_t kSrTidY              = 0x22;
inline constexpr uint32_t kSrTidZ              = 0x23;
inline constexpr uint32_t kSrCtaIdX            = 0x25;
inline constexpr uint32_t kSrCtaIdY            = 0x26;
inline constexpr uint32_t kSrCtaIdZ            = 0x27;
inline constexpr uint32_t kSrSmemSz            = 0x32;
inline constexpr uint32_t kSrEqMask            = 0x38;
inline constexpr uint32_t kSrRegAlloc          = 0x3d;
inline constexpr uint32_t kSrGlobalErrorStatus = 0x40;
inline constexpr uint32_t kSrWarpErrorStatus   = 0x42;

inline constexpr Ins with_pred(Ins ins, int pred)
{
    ins.w0 = (ins.w0 & ~(0xfULL << 12)) | ((static_cast<uint64_t>(pred) & 0xfULL) << 12);
    return ins;
}

inline constexpr Ins with_req(Ins ins, uint64_t req_mask)
{
    ins.w1 = (ins.w1 & ~(0x3fULL << 52)) | ((req_mask & 0x3fULL) << 52);
    return ins;
}

// --------------------------------------------------------------------------
// Control flow, barriers & trap lifecycle
// --------------------------------------------------------------------------

inline constexpr Ins nop() { return {0x0000000000007918ULL, 0x000fda0000000000ULL}; }

inline constexpr Ins exit_inst(int pred = kPT)
{
    return with_pred({0x794dULL, (0xfea00ULL << 32) | 0x3800000ULL}, pred);
}

inline constexpr Ins bpt_trap1()
{
    return {0x000000040000795cULL, 0x002fea0000300000ULL};
}

inline constexpr Ins ret_r12()
{
    return {0x0c087950ULL, (0x3fc000ULL << 32) | 0x3a00000ULL};
}

inline constexpr uint64_t bra_w0(int64_t n_instrs, int pred)
{
    const int64_t off = n_instrs * 4;
    const uint64_t lo8 = static_cast<uint64_t>(off & 0xffLL);
    const uint64_t hi  = static_cast<uint64_t>((off >> 8) & 0x3fffffffLL);
    return (hi << 34) | (lo8 << 16) | ((static_cast<uint64_t>(pred) & 0xfULL) << 12) | 0x947ULL;
}

inline constexpr uint64_t bra_w1(int64_t n_instrs)
{
    const int64_t off = n_instrs * 4;
    const uint64_t hi18 = static_cast<uint64_t>((off >> 38) & 0x3ffffLL);
    return (0xfea00ULL << 32) | 0x3800000ULL | hi18;
}

inline constexpr Ins bra_rel(int64_t n_instrs, int pred = kPT)
{
    return {bra_w0(n_instrs, pred), bra_w1(n_instrs)};
}

inline constexpr Ins bra_to(uint64_t from_va, uint64_t to_va, int pred = kPT)
{
    const int64_t n_instrs = (static_cast<int64_t>(to_va) - static_cast<int64_t>(from_va + 16)) >> 4;
    return bra_rel(n_instrs, pred);
}

inline constexpr Ins brx(int rn)
{
    return {(static_cast<uint64_t>(rn & 0xff) << 24) | 0x7949ULL, 0x003ff20003800000ULL};
}

inline constexpr Ins bar_sync0(int pred = kPT)
{
    return with_pred({0x7b1dULL, 0x000fec0000010000ULL}, pred);
}

inline constexpr Ins bar_sync_trap()
{
    return {0x0000000000007b1dULL, 0x000fc00000008000ULL};
}

// --------------------------------------------------------------------------
// Data movement & 32/64-bit integer ALU
// --------------------------------------------------------------------------

inline constexpr Ins mov_imm(int rd, uint32_t imm, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(imm) << 32) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7802ULL,
                      (0x3fda00ULL << 32) | 0x0f00ULL},
                     pred);
}

inline constexpr Ins mov_reg(int rd, int rs, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(rs & 0xff) << 32) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7202ULL,
                      (0x3fda00ULL << 32) | 0x0f00ULL},
                     pred);
}

inline constexpr Ins iadd_imm(int rd, int ra, uint32_t imm)
{
    return {(static_cast<uint64_t>(imm) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7835ULL,
            (0x3fda00ULL << 32) | 0x078e0000ULL};
}

inline constexpr Ins iadd3_sub_regs(int rd, int ra, int rb)
{
    // IADD3 Rd, PT, PT, Ra, -Rb, RZ
    return {(1ULL << 63) |
            (static_cast<uint64_t>(rb & 0xff) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7210ULL,
            0x003fde0007ffe0ffULL};
}

inline constexpr Ins iadd3_sub_from_imm(int rd, int ra, uint32_t imm)
{
    // IADD3 Rd, PT, PT, -Ra, imm, RZ (rd = imm - ra)
    return {(static_cast<uint64_t>(imm) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7810ULL,
            0x003fde0007ffe1ffULL};
}

inline constexpr Ins iadd64_imm(int rd, int ra, uint32_t imm)
{
    // IADD.64 Rd:Rd+1, Ra:Ra+1, sign_extend(imm)
    return {(static_cast<uint64_t>(imm) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7835ULL,
            (0x3fde00ULL << 32) | 0x078e0200ULL};
}

inline constexpr Ins iadd64_sub_regs(int rd, int ra, int rb)
{
    // IADD.64 Rd:Rd+1, Ra:Ra+1, -Rb:Rb+1 (even registers)
    return {(1ULL << 63) |
            (static_cast<uint64_t>(rb & 0xff) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7235ULL,
            0x003fde00078e0200ULL};
}

inline constexpr Ins lea_shl(int rd, int ra, uint32_t shift)
{
    // LEA Rd, Ra, RZ, shift (shift in 0..31)
    return {(0xffULL << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7211ULL,
            0x003fde00078e00ffULL | (static_cast<uint64_t>(shift & 0x1fU) << 11)};
}

inline constexpr Ins imad_regs(int rd, int ra, int rb, int rc)
{
    return {(static_cast<uint64_t>(rb & 0xff) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7224ULL,
            (0x3fda00ULL << 32) | 0x078e0200ULL | static_cast<uint64_t>(rc & 0xff)};
}

inline constexpr Ins imad_add(int rd, int ra, int rc)
{
    return {(1ULL << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7824ULL,
            (0x3fda00ULL << 32) | 0x078e0200ULL | static_cast<uint64_t>(rc & 0xff)};
}

inline constexpr Ins imad_shl_u32(int rd, int ra, uint32_t imm)
{
    return {(static_cast<uint64_t>(imm) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7824ULL,
            (0x3fda00ULL << 32) | 0x078e02ffULL};
}

inline constexpr Ins imad_wide_u32(int rd, int ra, uint32_t imm, int rc)
{
    return {(static_cast<uint64_t>(imm) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7825ULL,
            (0x3fda00ULL << 32) | 0x078e0000ULL | static_cast<uint64_t>(rc & 0xff)};
}

inline constexpr Ins imad_wide_u32_regs(int rd, int ra, int rb, int rc)
{
    return {(static_cast<uint64_t>(rb & 0xff) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7225ULL,
            (0x3fda00ULL << 32) | 0x078e0000ULL | static_cast<uint64_t>(rc & 0xff)};
}

inline constexpr Ins lop3_and_imm(int rd, int ra, uint32_t imm)
{
    return {(static_cast<uint64_t>(imm) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7812ULL,
            (0x3fde00ULL << 32) | 0x078ec0ffULL};
}

inline constexpr Ins lop3_and_regs(int rd, int ra, int rb)
{
    // LOP3.LUT Rd, Ra, Rb, RZ, 0xc0, !PT -> Rd = Ra & Rb
    return {(static_cast<uint64_t>(rb & 0xff) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7212ULL,
            (0x3fde00ULL << 32) | 0x078ec0ffULL};
}

inline constexpr Ins lop3_tst_imm_p0(int ra, uint32_t imm)
{
    // LOP3.LUT P0, RZ, Ra, imm, RZ, 0xc0, !PT -> P0 = ((Ra & imm) != 0)
    return {(static_cast<uint64_t>(imm) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) | (0xffULL << 16) | 0x7812ULL,
            (0x3fde00ULL << 32) | 0x0780c0ffULL};
}

inline constexpr Ins lop3_tst_regs_p0(int ra, int rb)
{
    // LOP3.LUT P0, RZ, Ra, Rb, RZ, 0xc0, !PT -> P0 = ((Ra & Rb) != 0)
    return {(static_cast<uint64_t>(rb & 0xff) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) | (0xffULL << 16) | 0x7212ULL,
            (0x3fde00ULL << 32) | 0x0780c0ffULL};
}

inline constexpr Ins lop3_or3_p0(int ra, int rb, int rc)
{
    // LOP3.LUT P0, RZ, Ra, Rb, Rc, 0xfe, !PT -> P0 = ((Ra | Rb | Rc) != 0)
    return {(static_cast<uint64_t>(rb & 0xff) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) | (0xffULL << 16) | 0x7212ULL,
            (0x3fde00ULL << 32) | 0x0780fe00ULL | static_cast<uint64_t>(rc & 0xff)};
}

inline constexpr Ins shf_r_u32(int rd, int rc, uint32_t shift)
{
    // SHF.R.U32.HI Rd, RZ, shift, Rc (Rd = Rc >> shift)
    return {(static_cast<uint64_t>(shift) << 32) |
            (0xffULL << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7819ULL,
            (0x3fda00ULL << 32) | 0x00011600ULL | static_cast<uint64_t>(rc & 0xff)};
}

inline constexpr Ins shf_l_one(int rd, int rshift)
{
    // SHF.L.U32.HI Rd, RZ, Rshift, 0x1 -> Rd = 1u << (Rshift & 31)
    return {(1ULL << 32) |
            (0xffULL << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7419ULL,
            (0x3fde00ULL << 32) | 0x00010600ULL | static_cast<uint64_t>(rshift & 0xff)};
}

// --------------------------------------------------------------------------
// Predicates & Special Registers
// --------------------------------------------------------------------------

inline constexpr Ins s2r(int rd, uint32_t sr)
{
    return {(static_cast<uint64_t>(rd & 0xff) << 16) | 0x7919ULL,
            (0x003224ULL << 40) | (static_cast<uint64_t>(sr) << 8)};
}

inline constexpr Ins r2p_pr_0xf(int ra)
{
    return {(0x0fULL << 32) | (static_cast<uint64_t>(ra & 0xff) << 24) | 0x7804ULL,
            0x003ff40000000000ULL};
}

inline constexpr Ins isetp_imm_raw(int pred_dst, int ra, uint32_t imm, uint32_t flags_lo)
{
    return {(static_cast<uint64_t>(imm) << 32) | (static_cast<uint64_t>(ra & 0xff) << 24) | 0x780cULL,
            (0x3fde00ULL << 32) | (static_cast<uint64_t>(pred_dst & 0x7) << 17) | flags_lo};
}

inline constexpr Ins isetp_eq_imm(int ra, uint32_t imm, int pred_dst = kP0)
{
    return isetp_imm_raw(pred_dst, ra, imm, 0x03f02270ULL);
}

inline constexpr Ins isetp_eq_and_pred(int ra, uint32_t imm, int pred_dst, int pred_src)
{
    const uint32_t ps = (static_cast<uint32_t>(pred_src & 0xf) << 23) | 0x00702270U;
    return isetp_imm_raw(pred_dst, ra, imm, ps);
}

inline constexpr Ins elect_p0()
{
    // ELECT P0, URZ, PT — elects lowest currently active lane in warp.
    return {0x0000000000ff782fULL, 0x003fde0003800000ULL};
}

inline constexpr Ins isetp_ne_imm(int ra, uint32_t imm, int pred_dst = kP0)
{
    return isetp_imm_raw(pred_dst, ra, imm, 0x03f05270ULL);
}

inline constexpr Ins isetp_le_u32_imm(int ra, uint32_t imm, int pred_dst = kP0)
{
    return isetp_imm_raw(pred_dst, ra, imm, 0x03f03070ULL);
}

inline constexpr Ins isetp_ge_u32_imm(int ra, uint32_t imm, int pred_dst = kP0)
{
    return isetp_imm_raw(pred_dst, ra, imm, 0x03f06070ULL);
}

inline constexpr Ins isetp_lt_u64_imm(int ra, uint32_t imm, int pred_dst = kP0)
{
    return isetp_imm_raw(pred_dst, ra, imm, 0x03f11070ULL);
}

inline constexpr Ins isetp_le_u64_imm(int ra, uint32_t imm, int pred_dst = kP0)
{
    return isetp_imm_raw(pred_dst, ra, imm, 0x03f13070ULL);
}

inline constexpr Ins isetp_ge_u64_imm(int ra, uint32_t imm, int pred_dst = kP0)
{
    return isetp_imm_raw(pred_dst, ra, imm, 0x03f16070ULL);
}

inline constexpr Ins isetp_ne_and_notp0(int ra, uint32_t imm)
{
    // ISETP.NE.AND P0, PT, Ra, imm, !P0 (branchless consensus!)
    return isetp_imm_raw(kP0, ra, imm, 0x04705270ULL);
}

inline constexpr Ins isetp_lt_u32_regs(int ra, int rb, int pred_dst = kP0)
{
    return {(static_cast<uint64_t>(rb & 0xff) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) | 0x720cULL,
            (0x3fde00ULL << 32) | (static_cast<uint64_t>(pred_dst & 0x7) << 17) | 0x03f01070ULL};
}

inline constexpr Ins isetp_eq_u32_regs(int ra, int rb, int pred_dst = kP0)
{
    return {(static_cast<uint64_t>(rb & 0xff) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) | 0x720cULL,
            (0x3fde00ULL << 32) | (static_cast<uint64_t>(pred_dst & 0x7) << 17) | 0x03f02070ULL};
}

inline constexpr Ins isetp_ge_u32_regs(int ra, int rb, int pred_dst = kP0)
{
    return {(static_cast<uint64_t>(rb & 0xff) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) | 0x720cULL,
            (0x3fde00ULL << 32) | (static_cast<uint64_t>(pred_dst & 0x7) << 17) | 0x03f06070ULL};
}

inline constexpr Ins isetp_ne_u64_regs(int ra, int rb, int pred_dst = kP0)
{
    return {(static_cast<uint64_t>(rb & 0xff) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) | 0x720cULL,
            (0x3fde00ULL << 32) | (static_cast<uint64_t>(pred_dst & 0x7) << 17) | 0x03f15270ULL};
}

// --------------------------------------------------------------------------
// Constant, Shared, Local & Global Memory
// --------------------------------------------------------------------------

inline constexpr Ins ldc_u32(int rd, uint32_t off)
{
    const uint64_t enc = static_cast<uint64_t>(off >> 2) << 8;
    return {(enc << 32) | 0xff000000ULL | (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7b82ULL,
            (0x003e34ULL << 40) | 0x0800ULL};
}

inline constexpr Ins ldc_64(int rd, uint32_t off)
{
    const uint64_t enc = static_cast<uint64_t>(off >> 2) << 8;
    return {(enc << 32) | 0xff000000ULL | (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7b82ULL,
            (0x003e34ULL << 40) | 0x0a00ULL};
}

inline constexpr Ins lds_u32(int rd, int ra, uint32_t off)
{
    return {(static_cast<uint64_t>(off & 0xffffffU) << 40) |
            (static_cast<uint64_t>(ra & 0xff) << 24) |
            (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7984ULL,
            (0x003e34ULL << 40) | 0x0800ULL};
}

inline constexpr Ins sts_u32(int ra, uint32_t off, int rs)
{
    return {(static_cast<uint64_t>(off & 0xffffffU) << 40) |
            (static_cast<uint64_t>(rs & 0xff) << 32) |
            (static_cast<uint64_t>(ra & 0xff) << 24) | 0x7388ULL,
            (0x0033f4ULL << 40) | 0x0800ULL};
}

inline constexpr Ins ldl_32(int rd, int ra, uint32_t off, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(off & 0xffffffU) << 40) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7983ULL,
                      0x0032240000100800ULL},
                     pred);
}

inline constexpr Ins ldl_64(int rd, int ra, uint32_t off, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(off & 0xffffffU) << 40) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7983ULL,
                      0x0032240000100a00ULL},
                     pred);
}

inline constexpr Ins ldl_128(int rd, int ra, uint32_t off, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(off & 0xffffffU) << 40) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7983ULL,
                      0x0032240000100c00ULL},
                     pred);
}

inline constexpr Ins stl_32(int ra, uint32_t off, int rs, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(off & 0xffffffU) << 40) |
                      (static_cast<uint64_t>(rs & 0xff) << 32) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) | 0x7387ULL,
                      0x0033e40000100800ULL},
                     pred);
}

inline constexpr Ins stl_64(int ra, uint32_t off, int rs, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(off & 0xffffffU) << 40) |
                      (static_cast<uint64_t>(rs & 0xff) << 32) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) | 0x7387ULL,
                      0x0033e40000100a00ULL},
                     pred);
}

inline constexpr Ins stl_128(int ra, uint32_t off, int rs, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(off & 0xffffffU) << 40) |
                      (static_cast<uint64_t>(rs & 0xff) << 32) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) | 0x7387ULL,
                      0x0033e40000100c00ULL},
                     pred);
}

inline constexpr Ins ld_e_32(int rd, int ra, uint32_t off, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(off) << 32) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7980ULL,
                      (0x003e34ULL << 40) | 0x10e900ULL},
                     pred);
}

inline constexpr Ins ld_e_64(int rd, int ra, uint32_t off, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(off) << 32) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7980ULL,
                      (0x003e34ULL << 40) | 0x10eb00ULL},
                     pred);
}

inline constexpr Ins ld_e_128(int rd, int ra, uint32_t off, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(off) << 32) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7980ULL,
                      (0x003e34ULL << 40) | 0x10ed00ULL},
                     pred);
}

inline constexpr Ins st_e_32(int ra, uint32_t off, int rs, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(off) << 32) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) | 0x7385ULL,
                      (0x0033f4ULL << 40) | 0x10e900ULL | static_cast<uint64_t>(rs & 0xff)},
                     pred);
}

inline constexpr Ins st_e_64(int ra, uint32_t off, int rs, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(off) << 32) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) | 0x7385ULL,
                      (0x0033f4ULL << 40) | 0x10eb00ULL | static_cast<uint64_t>(rs & 0xff)},
                     pred);
}

inline constexpr Ins st_e_128(int ra, uint32_t off, int rs, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(off) << 32) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) | 0x7385ULL,
                      (0x0033f4ULL << 40) | 0x10ed00ULL | static_cast<uint64_t>(rs & 0xff)},
                     pred);
}

inline constexpr Ins redg_add_u32(int ra, uint32_t byte_off, int rs, int pred = kP0)
{
    return with_pred({(static_cast<uint64_t>(byte_off) << 40) |
                      (static_cast<uint64_t>(rs & 0xff) << 32) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) | 0x798eULL,
                      (0x0033daULL << 40) | 0x0010e1ffULL},
                     pred);
}

inline constexpr Ins atomg_add_u32(int rd, int ra, uint32_t byte_off, int rs, int pred = kP0)
{
    return with_pred({(static_cast<uint64_t>(byte_off) << 40) |
                      (static_cast<uint64_t>(rs & 0xff) << 32) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x73a8ULL,
                      (0x003224ULL << 40) | 0x001ee1ffULL},
                     pred);
}

inline constexpr Ins atomg_or_u32(int rd, int ra, uint32_t byte_off, int rs, int pred = kP0)
{
    return with_pred({(static_cast<uint64_t>(byte_off) << 40) |
                      (static_cast<uint64_t>(rs & 0xff) << 32) |
                      (static_cast<uint64_t>(ra & 0xff) << 24) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x73a8ULL,
                      (0x003224ULL << 40) | 0x031ee1ffULL},
                     pred);
}

// --------------------------------------------------------------------------
// Named barriers, uniform state, uniform predicates & convergence barriers
// --------------------------------------------------------------------------

inline constexpr Ins b2r_bar(int pred, int rd, int bar_id)
{
    return with_pred({(static_cast<uint64_t>(bar_id & 0xf) << 54) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x731cULL,
                      0x0032240000000000ULL},
                     pred);
}

inline constexpr Ins b2r_warp(int rd)
{
    return {(static_cast<uint64_t>(rd & 0xff) << 16) | 0x731cULL, 0x0032240000008000ULL};
}

inline constexpr Ins r2b_bar(int pred, int bar_id, int rs)
{
    return with_pred({(static_cast<uint64_t>(bar_id & 0xf) << 54) |
                      (static_cast<uint64_t>(rs & 0xff) << 32) | 0x731eULL,
                      0x0033e40000000000ULL},
                     pred);
}

inline constexpr Ins r2b_warp(int pred, int rs)
{
    return with_pred({(static_cast<uint64_t>(rs & 0xff) << 32) | 0x731eULL,
                      0x0033e40000008000ULL},
                     pred);
}

inline constexpr Ins mov_r_ur(int rd, int ur, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(ur & 0xff) << 32) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7c02ULL,
                      (0x003fdaULL << 40) | 0x08000f00ULL},
                     pred);
}

inline constexpr Ins r2ur(int urd, int ra, int pred = kPT)
{
    return with_pred({(static_cast<uint64_t>(ra & 0xff) << 24) |
                      (static_cast<uint64_t>(urd & 0xff) << 16) | 0x72caULL,
                      (0x003fdaULL << 40) | 0x000e0000ULL},
                     pred);
}

inline constexpr Ins up2ur_ur7() { return {0x000000ffff077883ULL, 0x003fde0008000000ULL}; }
inline constexpr Ins ur2up_ur7() { return {0x000000ff07007884ULL, 0x003fde0008000000ULL}; }

inline constexpr Ins bmov_clear_r_b(int pred, int rd, int b_idx)
{
    return with_pred({(static_cast<uint64_t>(b_idx & 0x3f) << 24) |
                      (static_cast<uint64_t>(rd & 0xff) << 16) | 0x7355ULL,
                      0x003e2a0000100000ULL},
                     pred);
}

inline constexpr Ins bmov_b_r(int pred, int b_idx, int rs)
{
    return with_pred({(static_cast<uint64_t>(rs & 0xff) << 32) |
                      (static_cast<uint64_t>(b_idx & 0x3f) << 24) | 0x7356ULL,
                      0x0033ea0000000000ULL},
                     pred);
}

} // namespace xsched::cuda::sm120::sass
