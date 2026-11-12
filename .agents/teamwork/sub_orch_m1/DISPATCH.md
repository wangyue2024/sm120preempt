## 2026-10-31T18:40:00Z

### Task: Milestone M1 Sub-Orchestrator (`M1_critical_verify_and_l2_base`)

- **Your Identity**: `sub_orch_m1` (Sub-Orchestrator for Milestone M1, `teamwork_orchestrator`)
- **Your Working Directory**: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/sub_orch_m1`
- **Parent Conversation ID**: `257da9c1-15a6-4890-8b45-11ac94dbad91`
- **Original Request Record**: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/ORIGINAL_REQUEST.md` (MUST read and include in every subagent dispatch)
- **Project Scope Document**: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/PROJECT.md`
- **Phase 0 Survey Reports (MUST READ)**:
  1. `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_1/handoff.md`
  2. `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_2/handoff.md`
  3. `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_3/handoff.md`

#### Scope & Deliverables for Milestone M1
This milestone fits a single Explorer → Worker → Reviewer/Challenger/Auditor iteration loop (Step 2B of Project Pattern).

1. **SM120 Level 2 & Level 3 Scaffolding Integration in `xsched/`**:
   - Integrate the hardware-verified `origin/release/sm120-level2` (`eebe930`) implementation (`GuardianSM120`, `c[0x0][0x170..0x188]`, `c[0x0][0x374..0x378]`, `tools/instrument/extract_sass.py`) and the SM120 Level 3 scaffolding from `origin/dev/Linuxlevel3` (`aa69ca7`) into `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched`.
   - Fix `TarpHandlerSM120::SetJumpInstruction` in `platforms/cuda/hal/src/arch/sm120.cpp` so it no longer uses the illegal SM86 48-bit `JMP` (`0x345678900000794a`), and ensure `xsched` builds cleanly into `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/build_sm120/` and passes SM120 Level 2 preemption tests on the live RTX 5060.

2. **Empirical Root-Cause Verification Suite (`xsched/test/sm120_l3_verify/` -> `sm120_verify_compendium`)**:
   - Build and run a standalone, reproducible verification binary `sm120_verify_compendium` (built into `xsched/build_sm120/bin/sm120_verify_compendium` and `xsched/test/sm120_l3_verify/sm120_verify_compendium`) on the live RTX 5060 (`595.99.02`, root password `123` if needed) that empirically tests and prints objective verification output (`VERIFY_PASS`) for:
     a) **Debunking `pGpu+0x911 == 0`**: Prove via `NV2080_CTRL_CMD_GPU_EXEC_REG_OPS` (`0x20800122`) with the properly aligned 56-byte `NV2080_CTRL_GPU_REG_OP` struct (`regOffset` at byte offset `+12`, `hClientTarget=0, hChannelTarget=0`) that `0x20800122` is NOT skipped by a `pGpu+0x911 == 0` gate, and reads live advancing Blackwell `PTIMER` (`0x9400`) values in user space.
     b) **Why `0x419e84` (`cuXtraTriggerTrap`) Failed**: Prove that for non-root `0x419e84` returns `0x1f` (`gpuValidateRegOffset_IMPL` user register access map), and for root (`CAP_SYS_ADMIN`) `0x20800122` and BAR0 read **`0xbadf1100`** (PRI address decode/routing error on Blackwell `GB206` because `SM_DBGR_CONTROL0` moved from its Volta–Ada offset `0x419e84`).
     c) **SM120 Trap Handler Layout & Why `trap_sm120.cpp` Failed**: Verify `CU_ETID_ToolsTrapHandler` slot 20 (`slot20_pc`, `.text.do_pause`) vs live `.text.traphandler` (`slot20_pc - 0x4400`), verify the `slot20_pc + 0xA00` code cave, and verify that `0x83de0317` / `0x83de031d` / `0x20801210` (`GR_SET_CTXSW_PREEMPTION_MODE`) work when invoked with correct structs/parameters and without `cuXtraInstrMemBlockAlloc`.

3. **Comprehensive Paper & Compendium Verification Report**:
   - Write `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/docs/sm120_level3_analysis_and_verification.md` containing:
     - Complete analysis of the XSched OSDI'25 paper's Level 1 / Level 2 / Level 3 preemption model and original implementation (`cuXtraGetTrapHandlerInfo`, `check_preempt_trap`, `cuXtraTriggerTrap`, `CudaQueueLv2::Reactivate`).
     - Point-by-point critical evaluation of `Level3forLinux/xsched/test/sm120_l3_trap_v2/compendium` backed by kernel source (`open-gpu-kernel-modules-595.99.02`) and live RTX 5060 empirical output from `sm120_verify_compendium`.

#### Exclusive Write Ownership
- `xsched/docs/sm120_level3_analysis_and_verification.md`
- `xsched/test/sm120_l3_verify/`
- `xsched/platforms/cuda/hal/` (setting up L2/L3 base for M2 and M3)
- `xsched/tools/instrument/`
- Do NOT write to `xsched/test/sm120_l3_e2e/` (owned by `sub_orch_e2e_test`).

#### Completion Criteria
- Run the full iteration loop (3 Explorers → 1 Worker → 2 Reviewers + 2 Challengers + 1 Forensic Auditor → Gate `GATE_STATUS.md`).
- When Gate Result is **PASS**, write `handoff.md` in your working directory and notify `257da9c1-15a6-4890-8b45-11ac94dbad91` via `send_message`.
