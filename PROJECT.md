# Project: XSched Blackwell (RTX 5060 / SM120) Level 3 Trap & Resumable Preemption

## Architecture
- **Target Hardware & Driver**: NVIDIA GeForce RTX 5060 (`GB206`, Compute Capability `12.0` / `sm_120`, 30 SMs, 64 MB BAR0) on Ubuntu 26.04.1 LTS (`Linux 7.0.0-38-generic`, SecureBoot disabled) running NVIDIA Open GPU Kernel Modules & GSP Firmware `595.99.02`.
- **XSched Preemption Hierarchy (`xsched/preempt` & `xsched/platforms/cuda/hal`)**:
  - **Level 1 (`kPreemptLevelBlock`)**: User-space `LaunchWorker` pause/resume with progressive command launching (`threshold_`, `batch_size_`).
  - **Level 2 (`kPreemptLevelDeactivate`)**: Dynamic SASS binary instrumentation (`GuardianSM120`: `check_preempt` at kernel entry + `restore_exec` for selective CTA replay) using `cuxtra` export tables and SM120 constant memory Bank 0 (`c[0x0][0x170..0x188]` for `DebuggerParams`, `c[0x0][0x374..0x378]` for `gridDim`).
  - **Level 3 (`kPreemptLevelInterrupt`) on Blackwell (`SM120`)**:
    1. **SM120 Trap-Based Preemption (`CudaQueueLv3Trap` / `trap_sm120.cpp`)**:
       - Uses `CU_ETID_ToolsTrapHandler` slot 20 (`GetInfoBlackwell`) to locate `.text.do_pause` (`slot20_pc`) and the live `.text.traphandler` (`live_base = slot20_pc - 0x4400`), patching `live_base + 0x390` (`reason == 1` handler) with SM120 `CALL.ABS.NOINC` (`0x...7943`) to the pre-existing utility vaspace code cave at `slot20_pc + 0xA00` (avoiding `cuXtraInstrMemBlockAlloc` which triggers context cancellation under `NV83DE`).
       - Inside the cave, warps inspect `c[0x0][0x170..0x188]` (`preempt_buffer`, `kernel_idx`, `killable`), set `global_exit_flag = 1` / `preempt_idx = kernel_idx` / `block_restore_flag = 1`, and exit hardware trap mode cleanly via `IADD R12, R12, 0x10; RET.ABS.NODEC R12 0x20` (with return PC `(R12, R13)` redirected to a normal-mode `EXIT` stub or returning to the kernel exit path) so the SM trap state machine completes cleanly without context corruption (`cudaSuccess`), followed by in-process `CudaQueueLv2::Reactivate(log)` CTA-granular replay.
       - Hardware trap is triggered in hundreds of microseconds via direct RM `NV83DE` (`GT200_DEBUGGER`) control calls (`0x83de0317` `SUSPEND_CONTEXT` / `0x83de0322` `SET_SINGLE_SM_STOP_TRIGGER` with `0x83de0329` `SET_DROP_DEFERRED_RC` and `0x83de0318` `RESUME_CONTEXT` / `0x83de0310` `CLEAR_ALL_SM_ERROR_STATES`).
    2. **SM120 Non-Abort Zero-Lost-Work Resumable Preemption (`resumable_sm120.cpp`)**:
       - Preempts in-flight GPU kernels in hundreds of microseconds and resumes them in place without aborting the process, destroying the context, or losing intra-CTA work, using:
         - Direct RM `NV83DE` (`GT200_DEBUGGER`) in-place SM warp suspend/stop (`0x83de0317` / `0x83de0322`) and resume/run (`0x83de0318` / `0x83de0323`) without `cuda-gdb` spawn overhead, and/or
         - Hardware Compute Instruction-Level / CTA Preemption (`NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE` = `0x20801210` with `COMPUTE_CILP = 2` / `COMPUTE_CTA = 1` + `NVA06C_CTRL_CMD_PREEMPT` = `0xa06c0105` / BAR0 `NV_RUNLIST_PREEMPT`).

## Feature Inventory
Every feature from the Survey phase and `ORIGINAL_REQUEST.md` appears here with its assigned milestone.
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | XSched Paper L3 Model & Critical Verification of `compendium` | Analyze OSDI'25 paper L1/L2/L3 architecture and empirically verify/debunk `compendium` claims on RTX 5060: prove `pGpu+0x911` is `IS_GSP_CLIENT(pGpu)==1` (not a `0` debug gate), demonstrate `0x20800122` working with fixed 56-byte `NV2080_CTRL_GPU_REG_OP` (`regOffset` at `+12`), prove `0x419e84` fails due to `gpuValidateRegOffset_IMPL` (non-root) and Blackwell PRI relocation (`0xbadf1100`), and prove why `trap_sm120.cpp` caused context cancellation (`cuXtraInstrMemBlockAlloc` + raw `EXIT` inside trap mode + broken SM86 `JMP`). | M1 | ORIGINAL_REQUEST §R1, Survey 1/2/3 |
| F2 | SM120 Level 2 Baseline Integration & Build Verification | Integrate verified `origin/release/sm120-level2` (`GuardianSM120`, `c[0x0][0x170..0x188]`, `extract_sass.py`) into `level3opencode/xsched`, fix `TarpHandlerSM120` SM86 `JMP` bug, and verify Level 2 preemption on live RTX 5060. | M1 | ORIGINAL_REQUEST §R1/R2, Survey 1 |
| F3 | Sub-Millisecond (~100–500 µs) Non-Abort Level 3 Trap Preemption on RTX 5060 | Implement and verify hardware trap preemption on SM120 (`live_base + 0x390` → `slot20_pc + 0xA00` cave) triggered via direct RM `NV83DE` (`0x83de0317` / `0x83de0322`) that cleanly exits trap mode via `RET.ABS.NODEC` without destroying the CUDA context or aborting the process, supporting multi-round preemption and CTA-granular replay (`CudaQueueLv2::Reactivate`). | M2 | ORIGINAL_REQUEST §R2, Survey 1/2/3 |
| F4 | Superior Non-Abort Zero-Lost-Work Resumable Preemption on RTX 5060 | Develop and verify non-abort, state-preserving Level 3 preemption on SM120 (direct RM `NV83DE` SM suspend/resume `0x83de0317`/`0x83de0318`/`0x83de0322`/`0x83de0323` and/or `0x20801210` `GR_SET_CTXSW_PREEMPTION_MODE` CILP/CTA + `0xa06c0105` / `NV_RUNLIST_PREEMPT`) achieving sub-millisecond preemption and exact resumption with zero lost work. | M3 | ORIGINAL_REQUEST §R2, Survey 2/3 |
| F5 | End-to-End Hardware Verification, Latency Benchmarking, & System Health Check | Pass 100% of E2E tests (Tiers 1–4) + Tier 5 adversarial coverage hardening on live RTX 5060 with objective preemption evidence and microsecond-level latency metrics, while ensuring dual-boot, system files, and GPU driver remain 100% healthy. | M4 | ORIGINAL_REQUEST §R2/R3 & Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | `M1_critical_verify_and_l2_base` | Integrate `origin/release/sm120-level2` & L3 scaffolding into `level3opencode/xsched`, build and execute live RTX 5060 empirical verification probes debunking false `compendium` claims and proving true root causes of SM120 trap failure, and write comprehensive analysis report `xsched/docs/sm120_level3_analysis_and_verification.md`. | none | IN_PROGRESS (`teamwork_worker_r2_1` `1f03756c`) |
| M2 | `M2_sm120_trap_preempt` | Implement and verify sub-millisecond (~100–500 µs) non-abort Level 3 trap preemption on RTX 5060 (`sm120.cpp` + `trap_sm120.cpp` + `xsched/test/sm120_l3_trap/`) using `.text.traphandler` cave (`slot20_pc + 0xA00`), clean `RET.ABS.NODEC` trap-mode exit, direct RM `NV83DE` trigger, and in-process CTA-granular `Reactivate` replay without context destruction. | M1 | IN_PROGRESS (`teamwork_worker_r2_1` `1f03756c`) |
| M3 | `M3_resumable_preempt` | Implement and verify superior non-abort, zero-lost-work resumable preemption on RTX 5060 (`resumable_sm120.cpp` + `xsched/test/sm120_l3_resumable/`) using direct RM `NV83DE` in-place SM suspend/resume (`0x83de0317`/`0x83de0318`/`0x83de0322`/`0x83de0323`) and/or `0x20801210` CILP/CTA + `0xa06c0105` preemption. | M1 | IN_PROGRESS (`teamwork_worker_r2_1` `1f03756c`) |
| M4 | `M4_final_e2e_and_hardening` | Execute Final Milestone Phase 1 (pass 100% of E2E test suite Tiers 1–4 from `TEST_READY.md`) and Phase 2 (Tier 5 white-box adversarial coverage hardening), and verify system/dual-boot/GPU driver integrity. | M2, M3, E2E Track | IN_PROGRESS (`teamwork_worker_r2_1` `1f03756c` Phase 1 E2E verification) |

## Interface Contracts

### `M1` ↔ `M2` & `M3` (XSched CUDA HAL Level 2/3 Base)
- `xsched::cuda::GuardianSM120` (`platforms/cuda/hal/include/xsched/cuda/hal/arch/sm120.h`, `src/arch/sm120.cpp`):
  - Provides SM120 Level 2 `GetGuardianInstructions` and `GetResumeInstructions` using `c[0x0][0x170..0x188]` (`DebuggerParams`) and `c[0x0][0x374..0x378]` (`gridDim`).
- `xsched::cuda::TarpHandlerSM120` (`platforms/cuda/hal/include/xsched/cuda/hal/arch/sm120.h`, `src/arch/sm120.cpp`):
  - Resolves `CU_ETID_ToolsTrapHandler` slot 20 (`slot20_pc`) and live `.text.traphandler` (`live_base = slot20_pc - 0x4400`).
- `xsched::cuda::CudaQueueLv3Trap` & `xsched::cuda::CudaQueueLv3Resumable` (`platforms/cuda/hal/include/xsched/cuda/hal/level3/`):
  - Inherit from `CudaQueueLv2` (`HwQueue`), implementing `Deactivate()`, `Interrupt()`, `Restore(log)`, and `Reactivate(log)`.
  - Mode selection in `platforms/cuda/hal/src/level3/cuda_queue.cpp` via `XSCHED_SM120_L3_MODE`:
    - `"trap"` (default for `CudaQueueLv3Trap` on SM120): Sub-millisecond non-abort trap preemption (`M2`).
    - `"resumable"`: Zero-lost-work non-abort resumable preemption (`M3`).
    - `"tsg"`: Legacy TSG timeslice preemption.

### `E2E Testing Track` ↔ `M1`, `M2`, `M3`, `M4`
- Verification binaries & CLI contracts located in `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/build_sm120/bin/` (or invoked via E2E runner `xsched/test/sm120_l3_e2e/run_e2e.sh`):
  1. `sm120_verify_compendium`: Runs empirical root-cause checks (`pGpu+0x911` `IS_GSP_CLIENT`, `0x20800122` 56-byte `RegOp` on `0x9400` PTIMER vs `0x419e84` `0xbadf1100`, trap handler layout at `slot20_pc - 0x4400`); exits `0` on pass and prints structured `VERIFY_PASS` metrics.
  2. `sm120_l3_trap_bench`: Runs in-process SM120 Level 3 trap preemption (`CudaQueueLv3Trap` / direct RM `NV83DE` + `.text.traphandler` cave + clean `RET.ABS.NODEC` exit + CTA replay) across configurable kernel durations, iterations, and streams; verifies zero context destruction (`cudaSuccess`), result correctness (`mismatch == 0`), and reports preemption latency in microseconds (`P50_US`, `P99_US`, `MIN_US`, `MAX_US`).
  3. `sm120_l3_resumable_bench`: Runs SM120 non-abort zero-lost-work resumable preemption (`M3`) on long-running / stateful / non-idempotent kernels while launching high-priority kernels during the preempted window; verifies exact completion without re-execution or process abort (`cudaSuccess`, `mismatch == 0`), and reports suspend/resume/turnaround latency in microseconds.

## Code Layout
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/`
  - `docs/sm120_level3_analysis_and_verification.md` — Paper Level 3 analysis & critical empirical verification of `compendium` (owned by `M1`)
  - `platforms/cuda/hal/include/xsched/cuda/hal/arch/sm120.h` — SM120 Guardian & TrapHandler declarations (initialized in `M1`, updated in `M2`)
  - `platforms/cuda/hal/src/arch/sm120.cpp` — SM120 Guardian & TrapHandler implementation (L2 base in `M1`, L3 trap handler in `M2`)
  - `platforms/cuda/hal/include/xsched/cuda/hal/level3/` — Level 3 headers (initialized in `M1`)
  - `platforms/cuda/hal/src/level3/trap_sm120.cpp` — SM120 sub-millisecond non-abort trap preemption implementation (owned exclusively by `M2`)
  - `platforms/cuda/hal/src/level3/resumable_sm120.cpp` — SM120 non-abort zero-lost-work resumable preemption implementation (owned exclusively by `M3`)
  - `platforms/cuda/hal/src/level3/cuda_queue.cpp` — Level 3 queue dispatch (initialized in `M1`, finalized in `M2`/`M3`)
  - `test/sm120_l3_verify/` — Empirical root-cause verification probes (`sm120_verify_compendium`, owned by `M1`)
  - `test/sm120_l3_trap/` — SM120 Level 3 trap preemption tests & latency benchmarks (`sm120_l3_trap_bench`, owned by `M2`)
  - `test/sm120_l3_resumable/` — SM120 non-abort resumable preemption tests & latency benchmarks (`sm120_l3_resumable_bench`, owned by `M3`)
  - `test/sm120_l3_e2e/` — Opaque-box 4-tier E2E test suite & runner (`run_e2e.sh`, owned exclusively by `E2E Testing Orchestrator`)
