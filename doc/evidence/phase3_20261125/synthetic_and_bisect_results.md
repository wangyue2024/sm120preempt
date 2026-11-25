# Synthetic Mode 5 tests, SR_REGALLOC probe and bisect results (2026-11-25)

All runs: RTX 5060 (GB206, 30 SMs) unless noted. Synthetic kernels: 60 CTAs x 256 threads (single wave, so every
preemption catches CTAs mid-body = State B), ~40 ms, in-place non-idempotent LCG accumulation, bit-exact compare
against a native golden run. `np` = number of suspend/resume pairs per run (at ~6 ms, +9 ms, +12 ms).
Sources: `xsched/test/sm120_v2/eval/test_partial_exit_p3.cu`, `test_convbar_p3.cu`.

## 1. Before any fix (engine as of v2.2.1 + working-tree DEF-11/12/13)

| Case | What it exercises | regs | np=1 | np=3 |
|---|---|---:|---|---|
| K0 uniform | control | <14 | - | PASS 5/5 |
| K1 half_exit | odd lanes EXIT at entry | <14 | - | PASS 5/5 |
| K2 staggered_exit | lanes EXIT one by one (partially exited warps) | <14 | - | PASS 5/5 |
| K3 staggered_reconv | intra-warp divergence + `__syncwarp` | <14 | - | PASS 5/5 |
| C0 shallow | 1 divergent loop + break (B0) | 11 | PASS | PASS 5/5 |
| C1 deep | 6 nested divergent loops + break (B0..B5) | 15 | PASS 3/3 | **FAIL 5/5** (11008/15360 threads wrong, no error) |
| D1..D6 | nest depth 1..6 (B0..B5) | 13 | - | PASS 5/5 each |
| R0 k_regs<1> | 1 divergent loop + extra live regs | 12 | PASS | PASS |
| R1 k_regs<4> | idem | 16 | **FAIL** | PASS |
| R2 k_regs<16> | idem | 26 | PASS | **FAIL** |
| R3 k_regs<48> | idem | 55 | **FAIL** (990 / 1000 / 974 / 11838 wrong; uniform over lanes & warps; 364 whole warps wrong) | **FAIL** |
| C1 with np=0 | no preemption | 15 | PASS 3/3 (np=0) | |

Conclusion at this stage: every kernel with <= 13 registers passes; every kernel using R14+ fails at least sometimes;
convergence-barrier index (B0..B5) alone is NOT the trigger (D6 passes).

## 2. SR_REGALLOC probe (patched `S2R SR_TID.X` -> `S2R SR_REGALLOC`)

| live values N | CU_FUNC_ATTRIBUTE_NUM_REGS | SR_REGALLOC (plain) | SR_REGALLOC (Lv3, before fix) | SR_REGALLOC (Lv3, even-rounding fix) |
|---:|---:|---:|---:|---:|
| 8 | 16 | 16 | 16 | 16 |
| 12 | 20 | 20 | 20 | 20 |
| 20 | 25 | 25 | 25 | **26** |
| 40 | 45 | 45 | 45 | **46** |
| 64 | 69 | 69 | 69 | **70** |
| 120 | 125 | 125 | 125 | **126** |
| 200 | 206 | 206 | 206 | 206 |

`SR_REGALLOC` == the kernel's exact register count (no extra reserved registers). The engine's Cave 2/5 cascade used
`r_top = SR_REGALLOC - 3` -> the top 1..3 live registers were never saved/restored (R14/R15 entirely for 14..16-register
kernels). The instrumentation's `cuXtraSetLocalRegsPerThread` DOES change `SR_REGALLOC` (25 -> 26 etc.).

## 3. Fix steps (both in the working tree, uncommitted)

Step A: `sm120_trap_engine.cpp` Cave 2/5 header `SR_REGALLOC - 3` -> `SR_REGALLOC - 1`.

| Case | regs | np=1 | np=3 |
|---|---:|---|---|
| R1 | 16 | PASS 5/5 | PASS 5/5 |
| R2 | 26 | PASS 5/5 | PASS 5/5 |
| R3 | 55 (odd) | **FAIL 5/5** | **FAIL 5/5** |
| C1 | 15 (odd, clamped 16) | **FAIL 5/5** | PASS 5/5 |
| k_regs_dump<48> | 64 | PASS 3/3 | |

-> odd register counts still fail: with odd N the 64-bit pair access `R(N-1):R(N)` straddles the allocation end.

Step B: `sm120_guardian.cpp` round the instrumented register allocation up to an even count
(`clamped_regs = (max(raw,16)+1) & ~1`).

| Case | np=1 | np=3 |
|---|---|---|
| C1 | PASS 5/5, PASS 10/10 | PASS 5/5 |
| R1 / R2 / R3 | PASS 5/5 each | PASS 5/5 each |
| K0..K3 partial-exit suite | | PASS 5/5 each |

## 4. Remaining failures after A+B: restore-wave re-preemption (DEF-21) and the working-tree DEF-12 change (DEF-18)

WT = working tree (A+B + uncommitted DEF-12 restore-path change); VAR = A+B with the three DEF-12 hunks reverted to HEAD
(`sm120_cond_clear_hdr` unconditional `is_restore_wave` clear, `PreemptStream` zeroing `active_kernel_idx`, no
post-active relaunch in `Reactivate`).

| Test | WT | VAR |
|---|---|---|
| C1 alone, np=1 x10 | PASS 10/10 | PASS 10/10 |
| C1 alone, np=2 x10 | **FAIL 0/10** | PASS 10/10 |
| C1 alone, np=3 x10 | FAIL 0/10 (one run) / PASS 10/10 (another run) | PASS 10/10 |
| all 12 kernels sequentially, np=2 | all PASS | all PASS |
| all 12 kernels sequentially, np=3, rep 1 | C0 FAIL, rest PASS | C0 FAIL, rest PASS |
| all 12 kernels sequentially, np=3, rep 2 | C1 FAIL, rest PASS | C1 FAIL, rest PASS |
| partial-exit suite np=3 | PASS | PASS |

## 5. Real PyTorch / Triton ops at Level-3 after A+B (RTX 5060, 15 mid-job preemptions each, per-call bit-exact)

| op | before fix | WT after A+B | VAR after A+B |
|---|---|---|---|
| cast_bool_to_i64 | crash 717 (100%) | PASS 5/5 | PASS 5/5 |
| triton_rmsnorm | silent corruption, 1-2/5 OK | PASS 5/5 | PASS 5/5 |
| sort_16m | crash 700 | PASS 5/5 | PASS 5/5 |
| topk_sampling | crash 717/700 | PASS 5/5 | PASS 5/5 |
| cumsum_64m (int) | crash | PASS 5/5 | PASS 5/5 |
| index_add_int32 | crash (5060) | PASS 5/5 | PASS 5/5 |
| transpose_copy / gemm_bf16_4k | PASS | PASS 5/5 | PASS 5/5 |

## 6. LIB-08 (CUB DeviceRadixSort 8M, NP=2) bisect on RTX 5080, before the register fix

| Build | runs | pass | fail rate |
|---|---:|---:|---:|
| HEAD (committed v2.2.1) | 40 (10 + 30 interleaved) | 34 | 15% |
| Working tree (DEF-12 change) | 50 (10 + 10 + 30 interleaved) | 31 | 38% |

Fisher exact two-sided p = 0.019. HW-06: HEAD 2/3, WT 2/4 pass. LLM-07 m5: HEAD 2/3, WT 2/4 pass.
After-fix 5080 numbers: see `fix_5080_*` files / the defect ledger.
