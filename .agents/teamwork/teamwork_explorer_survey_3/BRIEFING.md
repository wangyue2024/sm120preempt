# BRIEFING — 2026-10-31T18:31:00Z

## Mission
Inspect live Blackwell (RTX 5060) hardware/driver environment and deeply analyze NVIDIA open-gpu-kernel-modules (595.99.02 & 550.163.01), RM control calls, and CUDA/debugger/driver mechanisms for SM120/GB20x Level 3 trap preemption and sub-millisecond resumable preemption.

## 🔒 My Identity
- Archetype: teamwork_explorer
- Roles: Investigation, Synthesis
- Working directory: /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_3
- Original parent: 257da9c1-15a6-4890-8b45-11ac94dbad91
- Milestone: Phase 0 Survey — Live Blackwell Hardware/Driver Environment & Preemption/Trap Mechanisms

## 🔒 Key Constraints
- Read-only investigation — do NOT implement or modify source code or kernel state
- Write only to `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_3`

## Current Parent
- Conversation ID: 257da9c1-15a6-4890-8b45-11ac94dbad91
- Updated: 2026-10-31T18:31:00Z

## Investigation State
- **Explored paths**:
  - Live system: `uname -a`, `nvidia-smi -q`, `lspci -vvnn -s 01:00.0`, `/proc/driver/nvidia/params`, `libcuda.so.595.99.02`, `libcudadebugger.so.595.99.02`, `/usr/include/cupti_checkpoint.h`
  - `open-gpu-kernel-modules-595.99.02/src/common/sdk/nvidia/inc/ctrl/ctrl83de/ctrl83dedebug.h`
  - `open-gpu-kernel-modules-595.99.02/src/nvidia/src/kernel/gpu/gr/kernel_sm_debugger_session.c` & `kernel_sm_debugger_session_ctrl.c`
  - `open-gpu-kernel-modules-595.99.02/src/nvidia/generated/g_kernel_sm_debugger_session_nvoc.c`
  - `open-gpu-kernel-modules-595.99.02/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gr.h` & `src/nvidia/src/kernel/gpu/gr/kernel_graphics_context.c`
  - `open-gpu-kernel-modules-595.99.02/src/common/sdk/nvidia/inc/ctrl/ctrla06c.h` & `src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c`
  - `open-gpu-kernel-modules-595.99.02/src/common/inc/swref/published/blackwell/gb202/dev_runlist.h` & `src/nvidia/src/kernel/gpu/fifo/arch/blackwell/kernel_fifo_gb202.c`
  - `open-gpu-kernel-modules-595.99.02/src/nvidia/src/kernel/gpu/subdevice/subdevice_ctrl_gpu_regops.c` & `gpu_register_access_map.c`
- **Key findings**:
  - Host runs Linux `7.0.0-38-generic` (SecureBoot disabled) with RTX 5060 (`GB206`, `sm_120`, 30 SMs, 64MB BAR0) and driver/GSP firmware `595.99.02` (`EnableGpuFirmware=18`).
  - `open-gpu-kernel-modules-595.99.02` matches the running driver; `550.163.01` has zero Blackwell files.
  - Hardware CILP (`COMPUTE_CILP = 2`) and CTA (`COMPUTE_CTA = 1`) non-abort preemption are supported via `NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE` (`0x20801210`) + `NVA06C_CTRL_CMD_PREEMPT` (`0xa06c0105`) / `NV_RUNLIST_PREEMPT` (`0x014`).
  - `NV83DE` (`GT200_DEBUGGER`) commands (`SUSPEND_CONTEXT`, `RESUME_CONTEXT`, `SET_SINGLE_SM_STOP_TRIGGER`, `SET_SINGLE_SM_RUN_TRIGGER`, `SET_DROP_DEFERRED_RC`, `CLEAR_ALL_SM_ERROR_STATES`) are routed by CPU-RM to GSP-RM and support both resumable pause/resume and paper-style trap-abort preemption.
- **Unexplored areas**:
  - None (Phase 0 survey scope 100% complete).

## Key Decisions Made
- Completed full verification of live environment and `595.99.02` source tree; documented both non-abort resumable preemption (`CILP`/`CTA` and `NV83DE` suspend/resume) and paper-style trap-abort preemption in `handoff.md`.

## Artifact Index
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_3/DISPATCH.md` — Dispatch log
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_3/BRIEFING.md` — Persistent working memory
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_3/progress.md` — Heartbeat and progress log
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_survey_3/handoff.md` — 5-component Handoff Report
