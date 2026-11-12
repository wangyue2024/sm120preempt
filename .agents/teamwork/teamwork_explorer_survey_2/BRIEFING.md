# BRIEFING — 2026-10-31T18:03:00Z

## Mission
Thoroughly inspect and critically evaluate prior SM120 (RTX 5060 Blackwell) Level 3 trap preemption research in `Level3forLinux` and `sm120_l3_trap_v2/compendium`.

## 🔒 My Identity
- Archetype: teamwork_explorer
- Roles: Investigation, Synthesis
- Working directory: /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_2
- Original parent: 257da9c1-15a6-4890-8b45-11ac94dbad91
- Milestone: Phase 0 Survey

## 🔒 Key Constraints
- Read-only investigation — do NOT implement or modify source code outside working directory
- Critically evaluate all claims, methods, and conclusions in `compendium` rather than accepting them at face value

## Current Parent
- Conversation ID: 257da9c1-15a6-4890-8b45-11ac94dbad91
- Updated: 2026-10-31T18:03:00Z

## Investigation State
- **Explored paths**:
  - `Level3forLinux/xsched/test/sm120_l3_trap_v2/compendium/` (`README.md`, `01`–`07`)
  - `Level3forLinux/xsched/test/sm120_l3_trap_v2/report/` (`00`, `02`–`07`, `10`, `kb/ABI.md`, `kb/Methods.md`)
  - `Level3forLinux/xsched/test/sm120_l3_trap_v2/` (`FINDINGS_ZH.md`, `RM_REVERSE_ZH.md`, `L3_TRAP_SM120_REPORT_ZH.md`, `probe/regops.cpp`, `probe/victim_spin.cu`, `tools/probe_regops_root.sh`, `tools/trigger_root_test.sh`, `tools/local_root_battery.sh`, `tools/nvidia_dbg/nvidia_dbg.c`)
  - `Level3forLinux/xsched/test/sm120_l3_probe/L3_FINAL_SUMMARY_ZH.md`
  - `Level3forLinux/xsched/platforms/cuda/hal/src/level3/trap_sm120.cpp`, `src/arch/sm120.cpp`, `src/arch/arch.cpp`
  - `Level3forLinux/refs/open-gpu-kernel-modules/` (`subdevice_ctrl_gpu_regops.c`, `gpu_access.c`, `kernel_sm_debugger_session_ctrl.c`)
- **Key findings**:
  1. `compendium/` (frozen 2026-10-26) builds its central thesis around a false reverse-engineering claim that `pGpu+0x911 == 0` is a "boot-time debug gate" blocking `0x20800122` and `0x83de031d`. Their own `report/07` (2026-10-27) and `refs/open-gpu-kernel-modules` prove `pGpu+0x911` is `IS_GSP_CLIENT(pGpu)` (= 1 on RTX 5060), and their earlier `regops.cpp` had two bugs (`hClientTarget != 0, hChannelTarget == 0` -> `0x3a`; `regOffset` at `+10` instead of `+12` -> reading offset `0`).
  2. With `regops.cpp` fixed, `0x20800122` works on SM120. Non-root access to `0x419e84` fails in `gpuValidateRegOffset_IMPL` (`0x1f`, user register access map). Root bypasses the map (`osIsAdministrator()`), forwards to GSP-RM (`status=0`), and reads `0xbadf1100` (hardware PRI error poison — `SM_DBGR_CONTROL0` moved from `0x419e84` on Blackwell GB20x or requires `grRouteInfo`/context routing).
  3. In `FINDINGS_ZH.md` v1.8–v2.0, `0x83de0317{0,0}` on a plain context (without `cuXtraInstrMemBlockAlloc`) triggered a benign trap (`reason=1`), executed the cave at `slot20 + 0xA00`, wrote `1` to XSched's `global_exit_flag` via `LDC R4, c[0x0][0x170]; ST.E [R4], R10`, returned cleanly via `RET.ABS`, and allowed the L2 guardian loop to exit in `6–89 ms` with `cudaSuccess` and zero context corruption. This hybrid trap-assisted L2 flag exit was abandoned in `compendium/` because `CudaQueueLv3Trap` called `cuXtraInstrMemBlockAlloc` (which causes driver context cancel under `0317{1,0}`) and used raw `EXIT` or `LD.E` inside the cave.
  4. `platforms/cuda/hal/src/arch/sm120.cpp` still contains the broken SM86 `JMP` encoding (`0x345678900000794a`) and patches `slot20 + 0x880` (`do_pause`) when `XSCHED_SM120_TRAP_CAVE` is unset.
- **Unexplored areas**: None within scope.

## Key Decisions Made
- Synthesize all findings into the 5-component `handoff.md` report structured around the 4 core evaluation questions.

## Artifact Index
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_2/BRIEFING.md` — Persistent working memory
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_2/progress.md` — Liveness heartbeat and progress tracking
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_2/handoff.md` — Final 5-component critical evaluation report
