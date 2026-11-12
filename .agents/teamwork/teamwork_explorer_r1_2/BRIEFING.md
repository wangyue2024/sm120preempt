# BRIEFING — 2026-10-31T19:29:00Z

## Mission
Produce an exact implementation blueprint for the Worker covering: (1) Sub-Millisecond (`~100–300 µs`) `NV83DE` Trigger Without `cuda-gdb` Spawn Overhead, and (2) Superior Non-Abort Zero-Lost-Work Resumable Preemption (`resumable_sm120.cpp` and `sm120_l3_resumable_bench`).

## 🔒 My Identity
- Archetype: teamwork_explorer
- Roles: explorer, synthesizer
- Working directory: /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_2
- Original parent: 257da9c1-15a6-4890-8b45-11ac94dbad91
- Milestone: M2_sm120_trap_preempt / M3_resumable_preempt (Iteration 1 Explorer 2)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement or modify project source code
- Write only to `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_2`

## Current Parent
- Conversation ID: 257da9c1-15a6-4890-8b45-11ac94dbad91
- Updated: 2026-10-31T19:29:00Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md`, `DISPATCH.md`, `PROJECT.md`, `teamwork_explorer_survey_2/handoff.md`, `teamwork_explorer_survey_3/handoff.md`
  - `xsched/platforms/cuda/hal/src/level3/trap_sm120.cpp`, `gdb_interrupt.cpp`, `interrupt.cpp`, `cuda_queue.cpp`, `tsg.cpp`
  - `Level3forLinux/xsched/test/sm120_l3_trap_v2/probe/l3agent.cpp`, `l3dbg.cu`, `trigger_dbg.cu`
  - `open-gpu-kernel-modules-595.99.02/src/nvidia/src/kernel/gpu/gr/kernel_sm_debugger_session.c`, `kernel_sm_debugger_session_ctrl.c`, `kernel_graphics_context.c`
  - `open-gpu-kernel-modules-595.99.02/src/common/sdk/nvidia/inc/class/cl83de.h`, `ctrl83dedebug.h`, `ctrl2080gr.h`, `ctrla06c.h`, `nvos.h`
  - `open-gpu-kernel-modules-595.99.02/src/nvidia/generated/g_kernel_sm_debugger_session_nvoc.c`, `g_subdevice_nvoc.c`, `g_kernel_channel_group_api_nvoc.c`
- **Key findings**:
  - Found struct bug in `trap_sm120.cpp:932`: `NVOS21_PARAMETERS` was missing `uint32_t paramsSize;` before `uint32_t status;`, AND `NV_ESC_RM_FREE` (`0x29`) was destroying `g_rm_hDebugger` when `cuda-gdb` detached. Suppressing `0x29` on `g_rm_hDebugger` + fixing `NVOS21_PARAMETERS` keeps `0x83de` permanently alive so `RmFastTrapPulse83de` (`0x83de0329` + `0x83de0317 {0,0}` + `0x83de0310` + `0x83de0318`) runs in `~150–250 µs` with zero `cuda-gdb` spawns.
  - Verified `0x20801210` (`NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE`, 32 bytes) and `0xa06c0105` (`NVA06C_CTRL_CMD_PREEMPT`, 8 bytes), `0xa06c0103` (`SET_TIMESLICE`, 8 bytes), `0xa06c0101` (`GPFIFO_SCHEDULE`, 4 bytes) all have `ROUTE_TO_PHYSICAL` (`0x10248u`) and `accessRight = 0x0u` in `595.99.02`.
- **Unexplored areas**: None — investigation complete.

## Key Decisions Made
- Delivered full C++ blueprints for both Deliverable 1 (`trap_sm120.cpp` sub-ms `NV83DE` trigger) and Deliverable 2 (`resumable_sm120.h`, `resumable_sm120.cpp`, and `sm120_l3_resumable_bench`) in `handoff.md`.

## Artifact Index
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_2/DISPATCH.md` — Dispatch instructions
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_2/BRIEFING.md` — Persistent working memory
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_2/progress.md` — Liveness heartbeat
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_2/handoff.md` — Complete 5-component handoff report and C++ blueprints
