// Tiny hand-assembler for the handful of SM120 SASS forms we inject into the
// live trap handler.  Every cave is checked with `nvdisasm -b SM120` before use.
#pragma once
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace s120 {
struct I { uint64_t w0, w1; };
// control: stall[41:44] yield[45] wbar[46:48] rbar[49:51] wait[52:57]
inline uint64_t ctl(int stall = 15, int wait = 0x00, int wb = 7, int rb = 7, int yld = 1) {
    (void)stall; (void)wait; (void)wb; (void)rb; (void)yld;
    return 0x000fda0000000000ULL;
}
const int RZ = 255;
inline I nop() { return {0x0000000000007918ULL, 0x000fc00000000000ULL}; }
inline I mov_imm(int rd, uint32_t imm, uint64_t c = ctl()) {
    return {((uint64_t)imm << 32) | ((uint64_t)rd << 16) | 0x7802ULL, 0x0000000000000f00ULL | c};
}
// S2R rd, SR (SR_VIRTUALSMID=0x43, SR_VIRTID=0x03, SR_LANEID=0x00); result via write barrier wb
inline I s2r(int rd, int sr, int wb = 0, uint64_t extra = 0) {
    return {((uint64_t)rd << 16) | 0x7919ULL, ((uint64_t)sr << 8) | ctl(15, 0x00, wb) | extra};
}
inline I cs2r_globaltimer(int rd) { return {((uint64_t)rd << 16) | 0x7805ULL, 0x15200ULL | ctl()}; }
// IMAD.WIDE.U32 rd(64) = ra * imm + rc(64)
inline I imad_wide_u32_imm(int rd, int ra, uint32_t imm, int rc) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | ((uint64_t)rd << 16) | 0x7825ULL,
            0x00078e0000ULL | (uint64_t)rc | ctl()};
}
// generic (no descriptor) store: ST.E[.64] [ra+off], rs
inline I st_e(int ra, int32_t off, int rs, bool b64 = false) {
    return {((uint64_t)(uint32_t)off << 32) | ((uint64_t)ra << 24) | 0x7385ULL,
            (b64 ? 0x100b00ULL : 0x100900ULL) | (uint64_t)rs | ctl()};
}
inline I iadd_imm(int rd, int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | ((uint64_t)rd << 16) | 0x7835ULL, 0x078e0000ULL | ctl()};
}
inline I ret_r12() { return {0x000000000c087950ULL, 0x003fc00003a00000ULL}; }   // RET.ABS R12 0x20
inline I exit_() { return {0x000000000000794dULL, 0x000fea0003800000ULL}; }

inline std::vector<uint64_t> flat(const std::vector<I> &v) {
    std::vector<uint64_t> o; for (auto &i : v) { o.push_back(i.w0); o.push_back(i.w1); } return o;
}
inline void disasm(const std::vector<I> &v, const char *path) {
    FILE *f = fopen(path, "wb"); for (auto &i : v) { fwrite(&i.w0, 8, 1, f); fwrite(&i.w1, 8, 1, f); } fclose(f);
    std::string cmd = std::string("/home/wy/cuda-12.9/bin/nvdisasm -b SM120 ") + path + " | grep -v headerflags";
    if (system(cmd.c_str())) {}
}
}  // namespace s120
