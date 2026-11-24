#include "xsched/cuda/hal/level3/sm120_sass.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace xsched::cuda::sm120::sass;

int main()
{
    std::vector<Ins> v = {
        nop(),
        exit_inst(),
        exit_inst(kP0),
        bpt_trap1(),
        ret_r12(),
        bra_rel(3),
        bra_rel(-2, kNotP0),
        brx(10),
        bar_sync0(),
        bar_sync_trap(),
        mov_imm(1, 0xffff80u),
        mov_imm(2, 1, kP1),
        mov_reg(9, 8, kP0),
        iadd_imm(12, 12, 0x10u),
        iadd3_sub_regs(7, 10, 11),
        iadd3_sub_from_imm(10, 4, 252u),
        iadd64_imm(6, 6, 0x40u),
        iadd64_sub_regs(6, 6, 8),
        lea_shl(10, 10, 3),
        imad_regs(3, 0, 6, 7),
        imad_add(4, 4, 5),
        imad_shl_u32(4, 4, 4),
        imad_wide_u32(6, 3, 16, 4),
        imad_wide_u32_regs(6, 10, 11, 6),
        lop3_and_imm(2, 2, 0xfffffff7u),
        lop3_tst_imm_p0(2, 0x11),
        lop3_or3_p0(8, 9, 10),
        shf_r_u32(10, 8, 5),
        s2r(4, kSrRegAlloc),
        s2r(5, kSrSmemSz),
        r2p_pr_0xf(5),
        elect_p0(),
        isetp_eq_imm(0, 6),
        isetp_eq_and_pred(3, 0, kP0, kP0),
        isetp_ne_imm(1, 0),
        isetp_le_u32_imm(0, 31),
        isetp_ge_u32_imm(4, 0x10000u),
        isetp_lt_u64_imm(6, 0x230u),
        isetp_le_u64_imm(6, 0x280u),
        isetp_ge_u64_imm(6, 0x500u),
        isetp_ne_and_notp0(0, 0),
        isetp_lt_u32_regs(8, 9),
        isetp_ge_u32_regs(9, 8),
        isetp_ne_u64_regs(8, 10),
        ldc_u32(4, 0x170),
        ldc_64(8, 0x180),
        lds_u32(0, 4, 0x400),
        sts_u32(4, 0x400, 5),
        ldl_32(1, kRegZero, 0xffffc4u),
        ldl_64(6, kRegZero, 0xffffa8u),
        ldl_128(8, 2, 0),
        stl_32(kRegZero, 0xffffa0u, 0),
        stl_64(kRegZero, 0xffffa8u, 0),
        stl_128(2, 0, 8),
        ld_e_32(0, 4, 0x0),
        ld_e_64(10, 4, 0x08),
        ld_e_128(8, 6, 0x10),
        st_e_32(6, 0x50, 2, kNotP0),
        st_e_64(6, 0x78, 14),
        st_e_128(6, 0x10, 8, kP4),
        redg_add_u32(4, 0x14, 0, kP0),
        atomg_add_u32(11, 4, 0x04, 0, kP0),
        b2r_bar(kP1, 9, 5),
        b2r_warp(8),
        r2b_bar(kP2, 9, 10),
        r2b_warp(kPT, kRegZero),
        mov_r_ur(8, 24, kP0),
        r2ur(24, 8, kP1),
        up2ur_ur7(),
        ur2up_ur7(),
        bmov_clear_r_b(kP0, 5, 1),
        bmov_b_r(kP0, 1, 5),
        lop3_and_regs(1, 2, 3),
        lop3_tst_regs_p0(1, 3),
        shf_l_one(3, 10),
        isetp_eq_u32_regs(3, 1),
        atomg_or_u32(0, 6, 0x5c, 3, kP0),
        st_e_64(6, 0x430, 252),
        ld_e_64(252, 6, 0x430),
    };

    const char *out_path = "/tmp/sm120_sass_check.bin";
    FILE *f = fopen(out_path, "wb");
    if (!f) {
        perror("fopen");
        return 1;
    }
    fwrite(v.data(), sizeof(Ins), v.size(), f);
    fclose(f);
    printf("Wrote %zu SASS instructions to %s\n", v.size(), out_path);
    return 0;
}
