## 2026-10-31T18:46:30Z

### Task: Iteration 1 Explorer 3 — Blackwell PRI Register Map, `0x83de031d` RegOps, & Compendium Verification Suite Blueprint

- **Your Identity**: `teamwork_explorer_r1_3` (`teamwork_explorer`)
- **Your Working Directory**: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_3`
- **Parent Conversation ID**: `257da9c1-15a6-4890-8b45-11ac94dbad91`
- **Original Request Record**: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/ORIGINAL_REQUEST.md` (MUST read first)
- **Project Scope Document**: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/PROJECT.md`
- **Phase 0 Survey Reports**:
  - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_2/handoff.md`
  - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_3/handoff.md`

#### Objectives
Produce an exact implementation blueprint for the Worker covering:
1. **Blackwell (`GB202`/`GB206`) PRI Offsets & `libcuda.so.595.99.02` Trap Trigger Inspection**:
   - Inspect `open-gpu-kernel-modules-595.99.02/src/common/inc/swref/published/blackwell/` and `src/nvidia/src/kernel/gpu/` to list all published Blackwell PRI offsets (`PTIMER`, `PMC_BOOT_0`, `NV_RUNLIST_PREEMPT`, `NV_CHRAM_CHANNEL`, `PGRAPH` / `GR` offsets) and `NV2080_CTRL_GPU_REG_OP` struct layout (`src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gpu.h`).
   - Also inspect `/usr/lib/x86_64-linux-gnu/libcuda.so.595.99.02` (read-only `objdump` / symbol / constant check around `ToolsTrapHandler` and `cuXtraTriggerTrap`) to see whether `libcuda.so.595.99.02` hardcodes `0x419e84` or has a Blackwell-specific register offset / `0x83de` path.
2. **Design of `sm120_verify_compendium` (`xsched/test/sm120_l3_verify/`) and `xsched/docs/sm120_level3_analysis_and_verification.md`**:
   - Specify the exact C++/CUDA implementation for `sm120_verify_compendium` that empirically tests on the live RTX 5060:
     - Test 1: Buggy `RegOp32` (`regOffset` at `+10`) vs fixed 56-byte `NV2080_CTRL_GPU_REG_OP` (`regOffset` at `+12`) calling `0x20800122` on Blackwell `PTIMER` (`0x9400`), proving `pGpu+0x911` (`IS_GSP_CLIENT == 1`) does NOT block `0x20800122`.
     - Test 2: Calling `0x20800122` on `0x419e84` (non-root -> `0x1f` `gpuValidateRegOffset_IMPL`; root -> `status=0`, read value `0xbadf1100` PRI error poison), proving why `cuXtraTriggerTrap` failed on SM120.
     - Test 3: Calling `CU_ETID_ToolsTrapHandler` slot 20 (`GetInfoBlackwell`) vs slot 19 (`CUDA_ERROR_INVALID_DEVICE = 101`), dumping `slot20_pc` (`.text.do_pause`) and `slot20_pc - 0x4400` (live `.text.traphandler`), and verifying the `slot20_pc + 0xA00` code cave.
     - Test 4: Testing `NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE` with the true command ID `0x20801210` vs `compendium`'s wrong command ID `0x20801201`.
3. Write your complete blueprint to `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_3/handoff.md` and notify `257da9c1-15a6-4890-8b45-11ac94dbad91`.
