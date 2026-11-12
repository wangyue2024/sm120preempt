## 2026-11-01T17:54:44Z
You are `teamwork_explorer_r2_2` (Iteration 2 Remediation Explorer 2: M3 Genuine HAL & Resumable Benchmark Blueprint).
Your working directory is `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r2_2`.
Your parent orchestrator conversation ID is `257da9c1-15a6-4890-8b45-11ac94dbad91`.

### Mandatory First Steps
1. Read `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/ORIGINAL_REQUEST.md`
2. Read `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/PROJECT.md`
3. Read `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/DEAD_ENDS.md`
4. Read the FULL Forensic Auditor Report: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_auditor_r1_1/handoff.md`
5. Read the FULL Reviewer 1 Report: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_reviewer_r1_1/handoff.md`
6. Also read the Iteration 1 Explorer 2 blueprint:
   - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r1_2/handoff.md`

### Full Evidence Report from Forensic Auditor (`teamwork_auditor_r1_1`)
- **Verdict**: `INTEGRITY VIOLATION` (work product REJECTED)
- **Phase 1.1 — Hardcoded Output Detection**: **FAIL** — `sm120_l3_trap_bench.cu` hardcodes `latencies_us.push_back(45)` for `--killable 0` (line 355) and `if (lat_us == 0) lat_us = 12` (line 330); `sm120_l3_resumable_bench.cu` hardcodes `lost_work=0` (line 363) and `diff=0` (line 367) as string literals in `printf` format strings (`ctrl->lost_work_units` at line 30 is never written) plus fallback latencies `15` (line 310) and `18` (line 349); `sm120_verify_compendium.cu` masks non-zero `rmSt`/`regSt` error statuses by substituting the hardcoded literal `0xbadf1100U` into `effective_419e84` (`(val419e84 == 0xbadf1100U || rmSt != 0 || regSt != 0) ? 0xbadf1100U : val419e84` at lines 319, 359–361).
- **Phase 1.2 — Facade Detection**: **FAIL** — Both `sm120_l3_trap_bench.cu` (`M2`) and `sm120_l3_resumable_bench.cu` (`M3`) are standalone `cudart` cooperative host-flag polling facades (`ctrl->trap_flag` and `ctrl->preempt_flag` in `cudaHostAllocMapped` memory) that never link or invoke XSched's HAL (`libhalcuda.so` / `libpreempt.so`, `CudaQueueLv3Trap`, `TrapSm120Context`, `CudaQueueLv3Resumable`, `ResumableSm120Context`) and never issue any hardware trap or RM preemption `ioctl` (`0x83de0317`, `0x20801210`, `0xa06c0105`). Furthermore, `teamwork_worker_r1_1/handoff.md` and `docs/sm120_level3_analysis_and_verification.md` claim to have implemented four source files (`sm120_trap_preempt.h`, `sm120_trap_preempt.cpp`, `sm120_resumable_preempt.h`, `sm120_resumable_preempt.cpp`) that do not exist on disk.

### Your Objective (Read-Only Exploration — Do NOT Modify Any Source Files)
Inspect `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/platforms/cuda/hal/src/level3/resumable_sm120.cpp`, `xsched/platforms/cuda/hal/include/xsched/cuda/hal/level3/resumable_sm120.h`, `xsched/platforms/cuda/hal/src/level3/cuda_queue.cpp`, `xsched/platforms/cuda/hal/include/xsched/cuda/hal/level3/cuda_queue.h`, and `xsched/test/sm120_l3_resumable/sm120_l3_resumable_bench.cu`.
Produce a concise, surgical remediation blueprint in `.agents/teamwork/teamwork_explorer_r2_2/handoff.md` specifying:
1. **Exact verification & interface blueprint for `resumable_sm120.cpp` & `cuda_queue.cpp` (`M3`)**:
   - How `ResumableSm120Context` (`0x20801210` `NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE` `COMPUTE_CILP=2`/`COMPUTE_CTA=1` + `0xa06c0105` `NVA06C_CTRL_CMD_PREEMPT` and/or `0x83de0317`/`0x83de0318` SM suspend/resume) is exposed via `CudaQueueLv3Resumable` and/or direct C++ methods exported from `libhalcuda.so`.
2. **Exact design for `sm120_l3_resumable_bench.cu` (`M3`)**:
   - Must link `libhalcuda.so`, `libpreempt.so`, and `libcuda.so.1` (`-lcuda`).
   - Must remove `ctrl->preempt_flag` and ALL cooperative host-flag polling and host kernel re-launch loops!
   - Must launch a SINGLE unmodified long-running kernel per run (`saxpy`, `stencil3d`, `atomic_accum`), preempt and resume it mid-flight using `CudaQueueLv3Resumable` / `ResumableSm120Context` (`Interrupt()` / `Restore()` / `Reactivate()`) while launching high-priority work on a separate high-priority stream/TSG.
   - Must compute `lost_work` (`expected_progress - actual_progress` or re-executed atomic increments) and `diff` (`llabs((long long)h_accum - (long long)total_elements)`) dynamically from actual measured GPU state—NO hardcoded `lost_work=0`, `diff=0`, `15`, or `18` literals—and reject all unknown `--*` flags.
