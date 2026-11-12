## 2026-10-31T19:14:24Z
You are `teamwork_explorer_r1_3_gen2` (Type: `teamwork_explorer`).
Your working directory is `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_3_gen2`.
Your Parent Conversation ID is `257da9c1-15a6-4890-8b45-11ac94dbad91`.

### Mandatory First Steps
1. Read `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/ORIGINAL_REQUEST.md` first.
2. Read `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_3_gen2/DISPATCH.md`.
3. Read `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/PROJECT.md` and `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/TEST_INFRA.md`.
4. Read the Phase 0 survey reports:
   - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_2/handoff.md`
   - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_3/handoff.md`
5. Create `BRIEFING.md` and `progress.md` in your working directory `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_3_gen2`.

### Objectives
Produce an exact implementation blueprint for the Worker covering:
1. **Blackwell (`GB202`/`GB206`) PRI Offsets & `libcuda.so.595.99.02` Trap Trigger Inspection**:
   - Inspect `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/open-gpu-kernel-modules-595.99.02/src/common/inc/swref/published/blackwell/` and `src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gpu.h` to list published Blackwell PRI offsets (`PTIMER` `0x9400`, `PMC_BOOT_0`, `NV_RUNLIST_PREEMPT`, `NV_CHRAM_CHANNEL`, `PGRAPH`/`GR` offsets) and `NV2080_CTRL_GPU_REG_OP` 56-byte struct layout.
   - Check `/usr/lib/x86_64-linux-gnu/libcuda.so.595.99.02` (read-only disassembly/search around `0x419e84` / `ToolsTrapHandler`) to verify whether `cuXtraTriggerTrap` hardcodes `0x419e84`.
2. **Design of `sm120_verify_compendium` (`xsched/test/sm120_l3_verify/`) and `xsched/docs/sm120_level3_analysis_and_verification.md`**:
   - Read `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/test/sm120_l3_e2e/tier1_feature_coverage.sh`, `tier2_boundary_corner.sh`, `tier3_cross_feature.sh`, and `tier4_real_world.sh` to see the exact CLI flags and output strings expected from `sm120_verify_compendium`, `sm120_l3_trap_bench`, `sm120_l3_resumable_bench`, and `xsched/docs/sm120_level3_analysis_and_verification.md`!
   - Specify the exact C++/CUDA implementation and CLI interface for `sm120_verify_compendium` (and document requirements for `sm120_l3_trap_bench` & `sm120_l3_resumable_bench`) so they match all E2E test assertions.
3. Write your complete blueprint to `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_3_gen2/handoff.md` and notify `257da9c1-15a6-4890-8b45-11ac94dbad91` via `send_message`.
