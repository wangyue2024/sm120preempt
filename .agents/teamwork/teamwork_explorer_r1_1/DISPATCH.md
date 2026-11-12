## 2026-10-31T18:46:30Z

### Task: Iteration 1 Explorer 1 — SM120 Level 2 Merge + `.text.traphandler` Cave & Clean Trap-Exit SASS Blueprint

- **Your Identity**: `teamwork_explorer_r1_1` (`teamwork_explorer`)
- **Your Working Directory**: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_1`
- **Parent Conversation ID**: `257da9c1-15a6-4890-8b45-11ac94dbad91`
- **Original Request Record**: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/ORIGINAL_REQUEST.md` (MUST read first)
- **Project Scope Document**: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/PROJECT.md`
- **Phase 0 Survey Reports**:
  - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_1/handoff.md`
  - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_2/handoff.md`

#### Objectives
Produce an exact, byte-level implementation blueprint for the Worker covering:
1. **Merging `origin/release/sm120-level2` (`eebe930`) and SM120 Level 3 files into `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched`**:
   - Inspect exact diffs and CMakeLists.txt in `xsched` so both Level 2 (`GuardianSM120`) and Level 3 (`CudaQueueLv3Trap` / `CudaQueueLv3Resumable`) compile cleanly on this machine (note: `/usr/bin/nvcc` is 12.4, or check if a newer CUDA toolkit e.g. `/usr/local/cuda*` exists on disk!).
2. **Exact SM120 `.text.traphandler` Patch & Clean Trap-Exit SASS Encoding**:
   - Inspect `Level3forLinux/xsched/platforms/cuda/hal/src/level3/trap_sm120.cpp`, `arch/sm120.cpp`, and `Level3forLinux/xsched/test/sm120_l3_trap_v2/FINDINGS_ZH.md` (§11, §22–§28).
   - Document the exact 16-byte SM120 SASS encodings for:
     a) `CALL.ABS.NOINC` patched at `live_traphandler + 0x390` (`slot20_pc - 0x4400 + 0x390`) targeting `slot20_pc + 0xA00`.
     b) The cave instructions at `slot20_pc + 0xA00` that:
        - Read `preempt_buffer` (`c[0x0][0x170..0x174]`), `entry_point_guardian` (`c[0x0][0x178..0x17c]`), `kernel_idx` (`c[0x0][0x180..0x184]`), `killable` (`c[0x0][0x188]`).
        - Write `global_exit_flag = 1` (`ST.E [R4], 1`) and/or `preempt_idx = kernel_idx` and `block_restore_flag = 1` (`ST.E` to `preempt_buffer + 16 + 8 * cta_id + 4`), or redirect the trap return PC `(R12, R13)` to a normal-mode `EXIT` instruction before executing `IADD R12, R12, 0x10; RET.ABS.NODEC R12 0x20` so the SM exits hardware trap mode cleanly without `CUDA_ERROR_ILLEGAL_INSTRUCTION (715)` or `CUDA_ERROR_LAUNCH_FAILED (719)`.
        - Ensure `cuXtraInstrMemBlockAlloc` is NOT called when `NV83DE` is used on SM120.
3. Write your complete blueprint to `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_1/handoff.md` and notify `257da9c1-15a6-4890-8b45-11ac94dbad91`.
