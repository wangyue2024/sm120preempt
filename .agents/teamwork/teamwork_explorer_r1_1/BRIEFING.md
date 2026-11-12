# BRIEFING — 2026-10-31T19:12:00Z

## Mission
Produce an exact, byte-level implementation blueprint for merging SM120 Level 2/3 into `level3opencode/xsched`, building cleanly on this host, and encoding the SM120 `.text.traphandler` patch (`CALL.ABS.NOINC`) and `slot20_pc + 0xA00` cave with clean trap-mode exit (`RET.ABS.NODEC`).

## 🔒 My Identity
- Archetype: teamwork_explorer
- Roles: investigation, synthesis
- Working directory: /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_1
- Original parent: 257da9c1-15a6-4890-8b45-11ac94dbad91
- Milestone: M1_critical_verify_and_l2_base / M2_sm120_trap_preempt

## 🔒 Key Constraints
- Read-only investigation — do NOT implement or modify project source files
- Write only to `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_1/`

## Current Parent
- Conversation ID: 257da9c1-15a6-4890-8b45-11ac94dbad91
- Updated: 2026-10-31T19:12:00Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md`, `PROJECT.md`, `DISPATCH.md`
  - `teamwork_explorer_survey_1/handoff.md`, `teamwork_explorer_survey_2/handoff.md`
  - `Level3forLinux/xsched/build/CMakeCache.txt` & `/home/wy/cuda-12.9/bin/nvcc`
  - `git diff main..origin/release/sm120-level2` & `origin/release/sm120-level2..origin/dev/Linuxlevel3`
  - `Level3forLinux/xsched/platforms/cuda/hal/include/xsched/cuda/hal/arch/sm120.h`
  - `Level3forLinux/xsched/platforms/cuda/hal/src/arch/{sm120.cpp,arch.cpp}`
  - `Level3forLinux/xsched/platforms/cuda/hal/src/level3/{interrupt.cpp,cuda_queue.cpp,trap_sm120.cpp}`
  - `Level3forLinux/xsched/test/sm120_l3_trap_v2/{FINDINGS_ZH.md,run_l3kill.sh,run_l3qspin.sh,probe/arena.h,probe/l3x.cu,probe/l3qspin.cu}`
- **Key findings**:
  - CUDA 12.9 is installed at `/home/wy/cuda-12.9/bin/nvcc` with `/usr/bin/g++-13`; configuring CMake with `-DSHIM_SOFTLINK=ON` and running `--target install` creates `output/lib/libcuda.so.1 -> libshimcuda.so`, which is required for `libcudart` `<<<>>>` interception.
  - `InterruptContext::InterruptContext` in `interrupt.cpp` allocates `InstrMemAllocator` (`cuXtraInstrMemBlockAlloc`) before checking `sm120_trap_`, which turns `0x83de0317` into context cancellation and causes `InstrumentTrapHandler()` to patch `slot20_pc + 0x880` with the broken SM86 `0x345678900000794a` jump.
  - Verified exact 16-byte SM120 SASS encodings for `CALL.ABS.NOINC`, `LDC` from `c[0x0][0x170..0x188]`, `ST.E` to `preempt_buffer`, `STL.64 [0xffffa8], R0` (`TRAP_RETURN_PC` redirection to normal-mode `EXIT` trampoline at `cave + 0x70`), and `IADD R12, R12, 0x10; RET.ABS.NODEC R12 0x20`.
- **Unexplored areas**: None — investigation complete.

## Key Decisions Made
- Provide complete byte-level C++ / SASS implementation blueprint in `handoff.md` covering both M1 (Level 2 merge + Level 3 integration + build) and M2 (`.text.traphandler` patch + cave SASS encodings + `InterruptContext` fix).

## Artifact Index
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_1/BRIEFING.md` — Working memory
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_1/progress.md` — Liveness heartbeat
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_1/handoff.md` — 5-Component Handoff Report & Implementation Blueprint
