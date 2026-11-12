# BRIEFING — 2026-10-31T17:55:00Z

## Mission
Survey XSched paper(s) (focusing deeply on Level 1/2/3 preemption models and the original Level 3 implementation method) and inspect git repositories, branches, commits, and code across `level3opencode` and related `xsched` directories.

## 🔒 My Identity
- Archetype: teamwork_explorer
- Roles: explorer, synthesizer
- Working directory: /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_1
- Original parent: 257da9c1-15a6-4890-8b45-11ac94dbad91
- Milestone: Phase 0 Survey

## 🔒 Key Constraints
- Read-only investigation — do NOT implement or modify source code
- Write only to `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_1/`
- Provide exact file paths, line numbers, commit hashes/branches, and paper quotes/sections

## Current Parent
- Conversation ID: 257da9c1-15a6-4890-8b45-11ac94dbad91
- Updated: 2026-10-31T18:18:00Z

## Investigation State
- **Explored paths**:
  - Papers/Docs: `xsched/docs/xsched-osdi25.pdf` (all 13 pages), `xsched/docs/xsched-osdi25-talk.pdf` (36 slides), `xsched/docs/xsched-intro-2025-en/zh.md`.
  - Core XQueue/Preempt: `xsched/preempt/src/xqueue/{async_xqueue.cpp,launch_worker.cpp}`.
  - CUDA HAL (`main` @ `f49289f`): `platforms/cuda/hal/include/xsched/cuda/hal/level{1,2,3}/*`, `platforms/cuda/hal/src/level{1,2,3}/*`, `platforms/cuda/hal/src/arch/{arch.cpp,sm70.cpp,sm86.cpp}`, `platforms/cuda/hal/inject/{Makefile,inject.cu,inject_sm86.asm,trap_handler_sm86.asm}`.
  - `cuxtra` interface & reverse-engineering: `3rdparty/cuxtra/include/cuxtra/cuxtra.h`, `test/sm120_l3_probe/evidence/cuxtra_analysis.txt`.
  - Git branches in `level3opencode/xsched`: `main` (`f49289f`), `origin/release/sm120-level2` (`eebe930`), `origin/sm120-level2` (`6c07b2e`), `origin/evidence/sm120-level2` (`bbf1c13`), `origin/docs/sm120-release-audit` (`e83a89d`), `origin/sm120-level3-probe` (`2eb2e63`), `origin/dev/Linuxlevel3` (`aa69ca7`).
- **Key findings**:
  - Paper defines 3 hierarchical preemption levels on `hwQueue`: Lv1 (`launch`/`sync`, blocks pending commands, latency $\approx \text{THRESHOLD} \times T$), Lv2 (`deactivate`/`reactivate`, DBI guardian `check_preempt` + `restore_exec` via `cuxtra` export tables, latency $\approx 1T$), Lv3 (`interrupt`/`restore`, preempts running command, 32 $\mu$s P99 on GV100).
  - Paper's NVIDIA Lv3 has two variants: TSG-based (`cuXtraSetTimeslice(ctx, 0)`, RM class `0xa06c` cmd `0xa06c0104`, context-wide) and Queue/Trap-based (`cuXtraTriggerTrap(ctx)` via RM subdevice `0x20800122` `NV2080_CTRL_CMD_GPU_EXEC_REG_OPS` writing `0x419e84` bit 31 + DBI-instrumented trap handler `check_preempt_trap` aborting idempotent warps of deactivated `hwQueue` and re-executing unfinished CTAs via `restore_exec`).
  - SM120 (Blackwell) Lv2 is completed and verified in `origin/release/sm120-level2` (`GuardianSM120` in `sm120.h`/`sm120.cpp`, debugger params window mapped at `c[0x0][0x170..0x188]`, grid dims at `c[0x0][0x374..0x378]`, offline SASS generator `tools/instrument/extract_sass.py`).
  - SM120 Lv3 investigation in `origin/dev/Linuxlevel3` proved that Driver 595 disables user-reachable reg-ops (`0x20800122` / `0x83de031d`) at GPU init via `pGpu+0x911 == 0`, blocking `cuXtraTriggerTrap` (`0x419e84`). `origin/dev/Linuxlevel3` implements two alternative backends (`GdbInterruptContext` and `TrapSm120Context` via RM `0x83de0317`) and documents all ABI offsets (`slot 20`, live trap handler at `base+0x15f0f00`, patch site `+0x390`).
- **Unexplored areas**: None within survey scope.

## Key Decisions Made
- Synthesize paper analysis, `main` branch Lv1/Lv2/Lv3 architecture, SM120 Lv2 branch implementation, and SM120 Lv3 branch findings into `handoff.md`.

## Artifact Index
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_1/DISPATCH.md` — Dispatch log
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_1/BRIEFING.md` — Persistent working memory
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_1/progress.md` — Liveness heartbeat and progress log
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_1/handoff.md` — Final 5-component survey handoff report
