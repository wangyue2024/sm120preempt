# TEST_INFRA — XSched SM120 Level 3 E2E Test Infrastructure Specification

**Project**: XSched Blackwell (RTX 5060 / SM120) Level 3 Trap & Resumable Preemption  
**Document**: Requirement-Driven 4-Tier End-to-End Test Infrastructure Specification  
**Version**: 1.0.0  
**Target Architecture**: NVIDIA GeForce RTX 5060 (GB206 / SM120 Compute Capability 12.0) on Ubuntu 26.04.1 LTS (Linux 7.0.0-38-generic) with NVIDIA Open Driver & GSP Firmware 595.99.02.

---

## 1. Overview & Architectural Principles

The XSched Blackwell E2E test infrastructure verifies the hierarchical preemption model, hardware trap mechanisms, zero-lost-work resumability, and system integrity across 5 primary features (`F1`–`F5`) defined in `PROJECT.md § Feature Inventory`.

### 1.1 Four-Tier Testing Hierarchy

```
+-----------------------------------------------------------------------------------+
|                        Tier 4: Real-World Scenarios (5)                           |
|       (LLM Attention, GEMM, Stencil PDE, Multi-Tenant SLA, Heavy Stress)          |
+-----------------------------------------------------------------------------------+
|                   Tier 3: Cross-Feature Combinations (6)                          |
|         (L1/L2 + L3 Trap, L2 + Resumable, Trap vs Resumable Latency, etc.)        |
+-----------------------------------------------------------------------------------+
|                    Tier 2: Boundary & Corner Cases (25)                           |
|    (Struct alignment, Rapid triggers, Max SM occupancy, Non-killable, Edge args) |
+-----------------------------------------------------------------------------------+
|                     Tier 1: Feature Coverage (25)                                 |
|            (5 tests/feature covering F1, F2, F3, F4, F5 core behavior)            |
+-----------------------------------------------------------------------------------+
```

### 1.2 Authoritative Expected Output Derivation
Every test in this suite derives its expected output from explicit authoritative sources:
1. **Reference Specifications**: `xsched-osdi25.pdf` (§3, §4, §6, §7), `PROJECT.md`, `ORIGINAL_REQUEST.md`.
2. **Hardware & Driver Realities**: Verified behavior of NVIDIA Open Driver `595.99.02` (`open-gpu-kernel-modules-595.99.02`), GSP-RM routing contracts, SM120 SASS opcodes (`CALL.ABS.NOINC`, `RET.ABS.NODEC`), and BAR0 register mappings.
3. **Formal Invariants**:
   - `exit_code == 0`: Test process completes successfully without unhandled signals or crashes.
   - `VERIFY_PASS`: Empirical validation probes match true driver architecture (`pGpu+0x911 == 1`, `0x20800122` functional).
   - `cudaSuccess`: GPU driver context remains clean and fully operational across all preemption cycles.
   - `mismatch == 0`: Zero computational deviation or corrupted output tensors after preemption/replay.
   - Sub-millisecond preemption latency: `P50_US < 1000` (target ~100–500 µs), `P99_US < 2000` µs.

---

## 2. Feature Mapping & Test Matrix

| Feature ID | Feature Name | Tier 1 (Core) | Tier 2 (Boundary) | Tier 3 (Cross) | Tier 4 (Scenario) | Total Tests |
|---|---|---|---|---|---|---|
| **F1** | Paper L3 Model & Critical `compendium` Verification | 5 | 5 | 1 | - | 11 |
| **F2** | SM120 Level 2 Baseline Integration & Contracts | 5 | 5 | 2 | - | 12 |
| **F3** | Sub-Millisecond Non-Abort L3 Trap Preemption | 5 | 5 | 3 | 2 | 15 |
| **F4** | Superior Non-Abort Zero-Lost-Work Resumable Preemption | 5 | 5 | 3 | 2 | 15 |
| **F5** | Hardware Health, Latency Benchmarking & System Safety | 5 | 5 | 2 | 1 | 13 |
| **Total** | | **25** | **25** | **6** | **5** | **61** |

---

## 3. Interface & Binary Execution Contracts

The test suite invokes the following binaries and system tools defined in `PROJECT.md § Interface Contracts`:

1. **`xsched/build_sm120/bin/sm120_verify_compendium`**:
   - Validates `pGpu+0x911` is `IS_GSP_CLIENT(pGpu)==1`.
   - Validates `0x20800122` working with corrected 56-byte `NV2080_CTRL_GPU_REG_OP` on `0x9400` PTIMER.
   - Validates `0x419e84` failure mode (`gpuValidateRegOffset_IMPL` non-root rejection and Blackwell PRI relocation `0xbadf1100`).
   - Validates `.text.traphandler` layout at `slot20_pc - 0x4400` and code cave at `slot20_pc + 0xA00`.
   - Structured Output: `[VERIFY_PASS] ...` and exit code `0`.

2. **`xsched/build_sm120/bin/sm120_l3_trap_bench`**:
   - Executes SM120 Level 3 trap preemption (`CudaQueueLv3Trap`).
   - CLI flags: `--duration_ms <int>`, `--iters <int>`, `--streams <int>`, `--grid <int>`, `--threads <int>`, `--killable <0|1>`, `--rapid <0|1>`.
   - Structured Output: `cudaSuccess`, `mismatch == 0`, `P50_US: <num>`, `P99_US: <num>`, `MIN_US: <num>`, `MAX_US: <num>`.

3. **`xsched/build_sm120/bin/sm120_l3_resumable_bench`**:
   - Executes SM120 non-abort resumable preemption (`CudaQueueLv3Resumable`).
   - CLI flags: `--duration_ms <int>`, `--iters <int>`, `--hp_workload <name>`, `--pause_pct <int>`, `--multi_pause <int>`.
   - Structured Output: `cudaSuccess`, `mismatch == 0`, `SUSPEND_P50_US: <num>`, `RESUME_P50_US: <num>`, `TURNAROUND_P50_US: <num>`, `P50_US: <num>`, `P99_US: <num>`.

4. **System Environment Probes**:
   - `nvidia-smi` (GPU status, memory, process table).
   - `/proc/driver/nvidia/version` (driver version match `595.99.02`).
   - `/boot` and `/boot/efi` filesystem integrity checks.

---

## 4. Test Case Catalog

### 4.1 Tier 1: Feature Coverage (25 Tests)

#### Feature F1: XSched Paper L3 Model & Critical Verification of `compendium`
- **`T1_F1_01` — Verification Analysis Document Completeness**:
  - *Objective*: Confirm `xsched/docs/sm120_level3_analysis_and_verification.md` exists and contains authoritative technical derivations for OSDI'25 L1/L2/L3 preemption, empirical proof of `pGpu+0x911`, `0x20800122` struct alignment, `0xbadf1100` PRI poison, and `.text.traphandler` cave layout.
  - *Input*: File presence and header/content inspection.
  - *Authoritative Source*: `PROJECT.md § Milestones M1`, `ORIGINAL_REQUEST.md §R1`.
  - *Verification*: Exit code 0, document contains all required topic sections.
- **`T1_F1_02` — Empirical Compendium Verification Binary Execution**:
  - *Objective*: Execute `sm120_verify_compendium` probe binary and verify successful run.
  - *Input*: `xsched/build_sm120/bin/sm120_verify_compendium --all`.
  - *Authoritative Source*: `PROJECT.md § Interface Contracts`.
  - *Verification*: Exit code 0, stdout contains `[VERIFY_PASS]`.
- **`T1_F1_03` — `pGpu+0x911` IS_GSP_CLIENT Field Empirical Proof**:
  - *Objective*: Verify that `pGpu+0x911` is `isGspClient == 1` on Blackwell RTX 5060, debunking the false "debug gate == 0" claim in prior compendium.
  - *Input*: `sm120_verify_compendium --probe gsp_client`.
  - *Authoritative Source*: `open-gpu-kernel-modules-595.99.02/src/nvidia/src/kernel/gpu/subdevice/subdevice_ctrl_gpu_regops.c:205`.
  - *Verification*: Exit code 0, output reports `IS_GSP_CLIENT: 1` and `GSP_DEBUG_GATE_DEBUNKED: PASS`.
- **`T1_F1_04` — RegOps 56-Byte Struct Alignment & PTIMER MMIO Read**:
  - *Objective*: Verify that `0x20800122` (`NV2080_CTRL_CMD_GPU_EXEC_REG_OPS`) with correct 56-byte layout (regOffset at +12) successfully reads hardware PTIMER `0x9400`.
  - *Input*: `sm120_verify_compendium --probe regops_ptimer`.
  - *Authoritative Source*: Survey 2 & 3 handoffs, `refs/open-gpu-kernel-modules/src/nvidia/src/kernel/gpu/subdevice/subdevice_ctrl_gpu_regops.c`.
  - *Verification*: Exit code 0, `PTIMER_READ: SUCCESS`, `val_lo != 0`.
- **`T1_F1_05` — Register `0x419e84` PRI Relocation & Trap Handler Layout**:
  - *Objective*: Validate root causes of `0x419e84` failure (`0xbadf1100` PRI bus error) and locate live `.text.traphandler` at `slot20_pc - 0x4400`.
  - *Input*: `sm120_verify_compendium --probe reg_419e84_and_traphandler`.
  - *Authoritative Source*: Survey 1 & 2 handoffs, `trap_sm120.cpp:414-465`.
  - *Verification*: Exit code 0, `PRI_RELOCATION_POISON: 0xbadf1100`, `LIVE_TRAP_HANDLER_BASE_OFFSET: -0x4400`.

#### Feature F2: SM120 Level 2 Baseline Integration & Build Verification
- **`T1_F2_01` — SM120 Level 2 HAL Library Compilation & Symbol Verification**:
  - *Objective*: Confirm `libxsched.so` builds cleanly and exports SM120 Level 2 symbols (`GuardianSM120`).
  - *Input*: Inspection of build artifacts and `nm -C` symbol lookup.
  - *Authoritative Source*: `PROJECT.md § M1`, `origin/release/sm120-level2`.
  - *Verification*: Exit code 0, symbols `GuardianSM120` and `c[0x0]` parameter bindings exist.
- **`T1_F2_02` — SM120 Constant Memory Bank 0 Mapping Verification**:
  - *Objective*: Verify SM120 constant memory offsets: `DebuggerParams` at `c[0x0][0x170..0x188]` and `gridDim` at `c[0x0][0x374..0x378]`.
  - *Input*: SASS disassembly validation or unit probe.
  - *Authoritative Source*: `PROJECT.md § Architecture`, `origin/release/sm120-level2`.
  - *Verification*: Exit code 0, offsets match `0x170` (preempt_buffer) and `0x374` (gridDim.x).
- **`T1_F2_03` — Progressive Command Launching Buffer Management**:
  - *Objective*: Verify `LaunchWorker` enforces `threshold_` and `batch_size_` bounds during kernel submission.
  - *Input*: Submit 32 test commands through `AsyncXQueue` with `threshold = 8, batch_size = 4`.
  - *Authoritative Source*: `xsched-osdi25.pdf` §5.2, `launch_worker.cpp`.
  - *Verification*: Exit code 0, in-flight command count never exceeds 8.
- **`T1_F2_04` — Level 2 `Deactivate()` Global Exit Flag Propagation**:
  - *Objective*: Verify that calling `Deactivate()` sets `global_exit_flag = 1` and causes unlaunched CTAs to exit cleanly.
  - *Input*: Launch multi-wave kernel (1000 CTAs), trigger `Deactivate()` after 10 ms.
  - *Authoritative Source*: `xsched-osdi25.pdf` §6.2, `guardian.h`.
  - *Verification*: Exit code 0, kernel returns in < 1 ms after `Deactivate()`, unlaunched CTAs skipped.
- **`T1_F2_05` — Level 2 `Reactivate()` CTA-Granular Replay Correctness**:
  - *Objective*: Verify that replaying an interrupted kernel via `Reactivate(log)` yields `mismatch == 0`.
  - *Input*: Run deactivated vector add kernel, reactivate queue, compare device output with CPU reference.
  - *Authoritative Source*: `xsched-osdi25.pdf` §6.2, `inject.cu:73-83`.
  - *Verification*: Exit code 0, `mismatch == 0`, final array elements strictly match expected sums.

#### Feature F3: Sub-Millisecond Non-Abort Level 3 Trap Preemption on RTX 5060
- **`T1_F3_01` — SM120 L3 Trap Preemption Execution & Exit Status**:
  - *Objective*: Execute `sm120_l3_trap_bench` on live RTX 5060 and confirm zero driver error.
  - *Input*: `xsched/build_sm120/bin/sm120_l3_trap_bench --duration_ms 100 --iters 5`.
  - *Authoritative Source*: `PROJECT.md § Interface Contracts`, `trap_sm120.cpp`.
  - *Verification*: Exit code 0, reports `cudaSuccess`.
- **`T1_F3_02` — Sub-Millisecond Trap Preemption Latency Target**:
  - *Objective*: Validate measured Level 3 trap preemption P50 latency is under 1000 µs (hundreds of microseconds).
  - *Input*: `sm120_l3_trap_bench --duration_ms 100 --iters 10`.
  - *Authoritative Source*: `ORIGINAL_REQUEST.md §R2`, `PROJECT.md § M2`.
  - *Verification*: `P50_US < 1000`, reported `P50_US` in range [100, 950] µs.
- **`T1_F3_03` — Non-Abort Context Preservation Across Multiple Trap Rounds**:
  - *Objective*: Verify CUDA context is not destroyed (`CUDA_ERROR_CONTEXT_IS_DESTROYED` avoided) across 5 consecutive trap rounds.
  - *Input*: `sm120_l3_trap_bench --iters 5 --rapid 0`.
  - *Authoritative Source*: `PROJECT.md § Architecture`, `trap_sm120.cpp`.
  - *Verification*: Exit code 0, 0 context resets, `cudaSuccess` on all rounds.
- **`T1_F3_04` — CTA-Granular Trap Replay Computational Correctness**:
  - *Objective*: Verify computational integrity (`mismatch == 0`) after in-place trap preemption and `Reactivate` replay.
  - *Input*: `sm120_l3_trap_bench --duration_ms 200 --check_correctness 1`.
  - *Authoritative Source*: `xsched-osdi25.pdf` §6.3, `PROJECT.md § M2`.
  - *Verification*: Exit code 0, `mismatch == 0`.
- **`T1_F3_05` — Multi-Stream Trap Selectivity**:
  - *Objective*: Preempt target stream via hardware trap while verifying concurrent stream on same context continues execution without interruption.
  - *Input*: `sm120_l3_trap_bench --streams 2 --preempt_stream 0`.
  - *Authoritative Source*: `xsched-osdi25.pdf` §6.3, Survey 1 handoff §1.1.4.
  - *Verification*: Exit code 0, stream 1 completes unperturbed, stream 0 successfully preempted and reactivated.

#### Feature F4: Superior Non-Abort Zero-Lost-Work Resumable Preemption on RTX 5060
- **`T1_F4_01` — SM120 L3 Resumable Preemption Execution**:
  - *Objective*: Run `sm120_l3_resumable_bench` and confirm clean execution without process abort.
  - *Input*: `xsched/build_sm120/bin/sm120_l3_resumable_bench --duration_ms 100 --iters 3`.
  - *Authoritative Source*: `PROJECT.md § Interface Contracts`, `resumable_sm120.cpp`.
  - *Verification*: Exit code 0, reports `cudaSuccess`.
- **`T1_F4_02` — Zero-Lost-Work Resumption Verification**:
  - *Objective*: Prove that paused kernel resumes from its exact execution point without resetting completed iterations or lost progress.
  - *Input*: `sm120_l3_resumable_bench --workload counter --check_zero_lost_work 1`.
  - *Authoritative Source*: `ORIGINAL_REQUEST.md §R2`, `PROJECT.md § M3`.
  - *Verification*: Exit code 0, `ZERO_LOST_WORK: PASS`, counter advances monotonically without duplicate or lost increments.
- **`T1_F4_03` — High-Priority Workload Execution During Preemption Window**:
  - *Objective*: Launch high-priority kernel while low-priority kernel is suspended; verify high-priority kernel finishes promptly.
  - *Input*: `sm120_l3_resumable_bench --hp_workload gemm_small --pause_duration_ms 50`.
  - *Authoritative Source*: `PROJECT.md § Interface Contracts`, `resumable_sm120.cpp`.
  - *Verification*: Exit code 0, high-priority task executes to completion within pause window, `HP_SUCCESS: 1`.
- **`T1_F4_04` — Resumable Latency Microsecond Reporting**:
  - *Objective*: Parse and verify suspend latency, resume latency, and turnaround latency in microseconds.
  - *Input*: `sm120_l3_resumable_bench --iters 5`.
  - *Authoritative Source*: `PROJECT.md § Interface Contracts`.
  - *Verification*: Output contains `SUSPEND_P50_US`, `RESUME_P50_US`, `TURNAROUND_P50_US`, all positive integer values < 2000 µs.
- **`T1_F4_05` — Multi-Round Pause-and-Resume Stability**:
  - *Objective*: Perform 5 successive pause-and-resume cycles on a single long-running kernel.
  - *Input*: `sm120_l3_resumable_bench --duration_ms 500 --multi_pause 5`.
  - *Authoritative Source*: `PROJECT.md § M3`.
  - *Verification*: Exit code 0, 5/5 pause-resume cycles succeed, final result verified `mismatch == 0`.

#### Feature F5: End-to-End Hardware Verification, Latency Benchmarking & System Health Check
- **`T1_F5_01` — Live GPU Hardware Detection & Status**:
  - *Objective*: Verify NVIDIA GeForce RTX 5060 (GB206) is online, healthy, and responsive.
  - *Input*: `nvidia-smi --query-gpu=name,driver_version,pstate,temperature.gpu,memory.total --format=csv,noheader`.
  - *Authoritative Source*: Live system environment inspection.
  - *Verification*: Exit code 0, GPU name contains `RTX 5060`, memory reports ~8151 MiB.
- **`T1_F5_02` — NVIDIA Driver & Module Integrity**:
  - *Objective*: Verify kernel driver version `595.99.02` is active and open kernel modules are loaded.
  - *Input*: `cat /proc/driver/nvidia/version` and `lsmod | grep nvidia`.
  - *Authoritative Source*: `PROJECT.md § Architecture`.
  - *Verification*: Exit code 0, version matches `595.99.02`, modules `nvidia` and `nvidia_uvm` loaded.
- **`T1_F5_03` — Dual-Boot & Root Filesystem Integrity**:
  - *Objective*: Confirm `/boot`, `/boot/efi`, and kernel grub configs are untouched and uncorrupted.
  - *Input*: `ls /boot/vmlinuz* /boot/initrd.img* /boot/efi/EFI`.
  - *Authoritative Source*: `ORIGINAL_REQUEST.md §R3`, `PROJECT.md § M4`.
  - *Verification*: Exit code 0, boot images and EFI bootloader directory intact.
- **`T1_F5_04` — Clean CUDA Context Creation Post-Preemption**:
  - *Objective*: Verify that new CUDA contexts can be initialized, executed, and destroyed cleanly following preemption runs.
  - *Input*: Standalone CUDA initialization test post-preemption.
  - *Authoritative Source*: `PROJECT.md § Acceptance Criteria`.
  - *Verification*: Exit code 0, `cuInit(0)` and `cuCtxCreate` return `CUDA_SUCCESS`.
- **`T1_F5_05` — End-to-End Metrics Aggregation & Reporting**:
  - *Objective*: Verify E2E runner aggregates all test results into structured JSON and table format.
  - *Input*: `run_e2e.sh --tier 1 --json`.
  - *Authoritative Source*: E2E Testing Track deliverables.
  - *Verification*: Exit code 0, JSON summary contains `total_tests`, `passed`, `failed`, `latency_summary`.

---

### 4.2 Tier 2: Boundary & Corner Cases (25 Tests)

#### Feature F1: Boundary & Corner Cases
- **`T2_F1_01` — Struct Alignment Boundary on `NV2080_CTRL_GPU_REG_OP`**:
  - *Objective*: Validate compiler padding and byte offsets of `NV2080_CTRL_GPU_REG_OP` (assert `offsetof(regOffset) == 12`, not 10).
  - *Input*: Struct alignment probe test.
  - *Authoritative Source*: Survey 2 handoff §1.2.
  - *Verification*: Exit code 0, struct size is 56 bytes, regOffset offset is 12 bytes.
- **`T2_F1_02` — Privilege Boundary on Hardware RegOps (Non-Root vs Root)**:
  - *Objective*: Verify that non-root execution of `0x20800122` on privileged register `0x419e84` returns `NV_ERR_INVALID_ARGUMENT (0x1f)` via `gpuValidateRegOffset_IMPL`.
  - *Input*: Non-root RegOp call targeting `0x419e84`.
  - *Authoritative Source*: `gpu_access.c:1195`, Survey 2 handoff §1.2.
  - *Verification*: Exit code 0, status is `0x1f` (`NV_ERR_INVALID_ARGUMENT`), matching expected security model.
- **`T2_F1_03` — Blackwell PRI Register Relocation Boundary & Poison Value**:
  - *Objective*: Verify that reading relocated PRI registers on SM120 returns hardware poison `0xbadf1100`.
  - *Input*: Root MMIO / RegOp read on `0x419e84`.
  - *Authoritative Source*: Survey 2 handoff §1.2, `report/07:142-167`.
  - *Verification*: Read value equals `0xbadf1100`, confirming PRI error bus response.
- **`T2_F1_04` — Export Table Slot Boundary: Slot 19 Rejection vs Slot 20 Acceptance**:
  - *Objective*: Verify `CU_ETID_ToolsTrapHandler` slot 19 returns `CUDA_ERROR_INVALID_DEVICE (101)` on SM120 while slot 20 returns valid 40B metadata.
  - *Input*: Unit test querying slot 19 and slot 20 export tables.
  - *Authoritative Source*: Survey 1 handoff §1.2.3, `sm120.cpp:261-327`.
  - *Verification*: Exit code 0, slot 19 returns 101, slot 20 returns 0 with struct_size `0x28`.
- **`T2_F1_05` — Utility Vaspace Code Cave Address Boundary**:
  - *Objective*: Verify that code cave at `slot20_pc + 0xA00` lies within the same 2MB utility page as `live_base + 0x390`.
  - *Input*: Vaspace boundary check probe.
  - *Authoritative Source*: Survey 1 & 2 handoffs, `trap_sm120.cpp:400-405`.
  - *Verification*: Exit code 0, page difference == 0, distance < 2 MB, accessible without TLB fault.

#### Feature F2: Boundary & Corner Cases
- **`T2_F2_01` — Ultra-Short Kernel Boundary (<10 µs Execution)**:
  - *Objective*: Verify preemption handling when kernel completes before `Deactivate()` signal is received.
  - *Input*: Kernel launching 1 CTA with 32 threads performing 1 add; trigger `Deactivate()` concurrently.
  - *Authoritative Source*: `xsched-osdi25.pdf` §6.2.
  - *Verification*: Exit code 0, queue drains cleanly, no crash, no unnecessary replay.
- **`T2_F2_02` — Maximum Grid Dimension Boundary (65,535 CTAs)**:
  - *Objective*: Verify `preempt_buffer` bounds when handling maximum grid dimensions under SM120.
  - *Input*: Grid size 65535 blocks, preemption triggered mid-grid.
  - *Authoritative Source*: `instrument.h:22-35`.
  - *Verification*: Exit code 0, no out-of-bounds write in `preempt_buffer`, replay completes accurately.
- **`T2_F2_03` — Zero-CTA / Empty Launch Edge Case**:
  - *Objective*: Verify `GuardianSM120` does not crash on empty kernel launches (gridDim = 0 or null launch).
  - *Input*: Launch 0-block or empty grid.
  - *Authoritative Source*: CUDA driver specification.
  - *Verification*: Handled gracefully without segmentation fault.
- **`T2_F2_04` — Rapid Back-to-Back Deactivate / Reactivate Cycles**:
  - *Objective*: Stress test queue state machine by calling `Deactivate()` immediately followed by `Reactivate()`.
  - *Input*: 50 rapid deactive/reactive cycles without intervening kernels.
  - *Authoritative Source*: `AsyncXQueue` lifecycle invariants.
  - *Verification*: Exit code 0, queue remains in valid state, subsequent launches succeed.
- **`T2_F2_05` — Non-Killable Kernel Boundary (`killable == 0`)**:
  - *Objective*: Verify that kernels marked `killable == 0` in `DebuggerParams` ignore preemption signals and run to completion.
  - *Input*: Launch kernel with `killable = 0`, trigger preemption.
  - *Authoritative Source*: `inject.cu:98-100`, `instrument.h:79`.
  - *Verification*: Exit code 0, kernel completes 100% of CTAs before queue yields.

#### Feature F3: Boundary & Corner Cases
- **`T2_F3_01` — Back-to-Back Rapid Trap Triggering (<1 ms Interval)**:
  - *Objective*: Fire multiple hardware traps in rapid succession to verify SM trap sequencer does not deadlock.
  - *Input*: `sm120_l3_trap_bench --rapid 1 --iters 10`.
  - *Authoritative Source*: SM trap architecture specifications.
  - *Verification*: Exit code 0, all 10 traps handled cleanly without GPU hang.
- **`T2_F3_02` — Trap Arrival During Kernel Epilogue Boundary**:
  - *Objective*: Trap fires at the exact moment warps are executing kernel epilogue / exit instructions.
  - *Input*: Variable-delay trap trigger targeting kernel final 5% duration.
  - *Authoritative Source*: `trap_sm120.cpp`.
  - *Verification*: Exit code 0, clean return via `RET.ABS.NODEC`, no orphan warps.
- **`T2_F3_03` — Maximum SM Occupancy Stress (30 SMs × 1536 Threads = 46,080 Threads)**:
  - *Objective*: Saturate all 30 SMs to 100% thread occupancy and trigger hardware trap simultaneously.
  - *Input*: Grid 90 blocks × 512 threads = 46,080 threads on SM120.
  - *Authoritative Source*: `PROJECT.md § Architecture`, RTX 5060 specs.
  - *Verification*: Exit code 0, all warps cleanly exit trap mode, `cudaSuccess`.
- **`T2_F3_04` — Trap Preemption with Active Shared Memory & Barriers (`__syncthreads()`)**:
  - *Objective*: Preempt kernel stalled at intra-block barrier `__syncthreads()`.
  - *Input*: Kernel with divergent warps waiting on shared memory barrier.
  - *Authoritative Source*: SM barrier architecture.
  - *Verification*: Exit code 0, warps successfully enter trap handler, clean exit.
- **`T2_F3_05` — Multi-Queue Concurrent Trap Interrupts Across 4 Streams**:
  - *Objective*: 4 concurrent CUDA streams executing independent kernels, preempting stream 0 and 2 simultaneously.
  - *Input*: `sm120_l3_trap_bench --streams 4 --preempt_mask 0x5`.
  - *Authoritative Source*: Multi-stream preemption requirements.
  - *Verification*: Exit code 0, streams 0 & 2 preempted, streams 1 & 3 complete without error.

#### Feature F4: Boundary & Corner Cases
- **`T2_F4_01` — Resumable Pause at 0% Progress (Kernel Entry Boundary)**:
  - *Objective*: Preempt kernel immediately upon launch before warp loop starts.
  - *Input*: `sm120_l3_resumable_bench --pause_pct 0`.
  - *Authoritative Source*: `resumable_sm120.cpp`.
  - *Verification*: Exit code 0, suspend succeeds, subsequent resume completes 100% work.
- **`T2_F4_02` — Resumable Pause at >95% Progress (Near-Completion Boundary)**:
  - *Objective*: Preempt kernel when warps have completed almost all iterations.
  - *Input*: `sm120_l3_resumable_bench --pause_pct 95`.
  - *Authoritative Source*: `resumable_sm120.cpp`.
  - *Verification*: Exit code 0, no race condition with kernel completion, `mismatch == 0`.
- **`T2_F4_03` — Consecutive Repeated Pause/Resume Stress (10 Cycles on Single Kernel)**:
  - *Objective*: Pause and resume the same in-flight kernel 10 times consecutively.
  - *Input*: `sm120_l3_resumable_bench --multi_pause 10`.
  - *Authoritative Source*: Non-abort preemption endurance testing.
  - *Verification*: Exit code 0, 10/10 pause-resumes succeed, `mismatch == 0`.
- **`T2_F4_04` — Non-Idempotent Memory Accumulator Integrity**:
  - *Objective*: Test preemption on an atomic accumulator kernel where duplicate execution would corrupt results.
  - *Input*: Kernel performing `atomicAdd` across shared buffer; verify exact final sum.
  - *Authoritative Source*: Zero-lost-work definition (`ORIGINAL_REQUEST.md §R2`).
  - *Verification*: Exit code 0, accumulator equals theoretical exact sum (no duplicates, no missed adds).
- **`T2_F4_05` — High-Priority Kernel Timeout & Priority Inversion Prevention**:
  - *Objective*: Validate system recovers gracefully if high-priority kernel takes longer than expected during pause window.
  - *Input*: Preempt low-priority kernel, launch HP kernel with configurable timeout.
  - *Authoritative Source*: Real-time scheduling safety invariants.
  - *Verification*: Exit code 0, HP kernel completes or times out cleanly, LP kernel resumes safely.

#### Feature F5: Boundary & Corner Cases
- **`T2_F5_01` — Heavy VRAM Allocation Boundary (>90% VRAM Occupied)**:
  - *Objective*: Trigger preemption when GPU VRAM is 90% full (7.2 GB of 8 GB).
  - *Input*: Allocate 7.2 GB buffer, run preemption benchmark in remaining memory.
  - *Authoritative Source*: Memory stress specifications.
  - *Verification*: Exit code 0, preemption succeeds without out-of-memory crash.
- **`T2_F5_02` — Invalid Kernel Pointer Error Recovery Without Leaking Debug Sessions**:
  - *Objective*: Pass intentionally invalid device pointer, verify error handling leaves RM debug sessions cleanly released.
  - *Input*: Launch faulty kernel, assert cleanup.
  - *Authoritative Source*: Resource Manager cleanup specifications.
  - *Verification*: Context error caught, debug session closed, next run unaffected.
- **`T2_F5_03` — Process SIGINT / SIGTERM Interruption Recovery**:
  - *Objective*: Send `SIGINT` to test process while preemption is active; verify GPU driver recovers and releases lock.
  - *Input*: Spawn preemption bench, kill with `SIGINT`, check driver status.
  - *Authoritative Source*: System resilience requirements.
  - *Verification*: Process exits, `nvidia-smi` shows 0 lingering zombie GPU processes.
- **`T2_F5_04` — File Descriptor & Device Node Leak Check**:
  - *Objective*: Verify `/dev/nvidia*` file descriptors do not leak across 20 preemption runs.
  - *Input*: Compare `/proc/$PID/fd` before and after test loop.
  - *Authoritative Source*: Linux OS resource hygiene.
  - *Verification*: Open file descriptor count returns to baseline.
- **`T2_F5_05` — GPU Thermal & Power Throttling Stability Under Preemption Stress**:
  - *Objective*: Monitor GPU temperature and performance state under continuous preemption workload.
  - *Input*: 60-second preemption loop; sample GPU temperature and throttle flags via `nvidia-smi`.
  - *Authoritative Source*: Hardware safety constraints (`ORIGINAL_REQUEST.md §R3`).
  - *Verification*: Temperature remains < 80°C, no HW slowdown/shutdown events.

---

### 4.3 Tier 3: Cross-Feature Combinations (6 Tests)

- **`T3_COMB_01` — F1 Empirical Verification Probes + F2 Level 2 Baseline**:
  - *Objective*: Verify that export table and RegOps sanity checks pass before initializing `GuardianSM120` Level 2 queues.
  - *Input*: Run F1 verification followed immediately by F2 Level 2 queue pipeline.
  - *Authoritative Source*: Integration contract between M1 and M2.
  - *Verification*: Both phases exit 0, queue initializes without regression.
- **`T3_COMB_02` — F2 Level 2 Draining + F3 Level 3 Trap Preemption on In-Flight Grid**:
  - *Objective*: Test hierarchical preemption where `Deactivate()` drains pending commands while `Interrupt()` traps executing warps, followed by selective CTA `Reactivate(log)`.
  - *Input*: Multi-kernel queue, trigger Level 3 preemption during kernel 2.
  - *Authoritative Source*: `xsched-osdi25.pdf` §6.3, `async_xqueue.cpp:109-112`.
  - *Verification*: Exit code 0, kernel 2 preempted and replayed, kernel 3 queued, final output `mismatch == 0`.
- **`T3_COMB_03` — F2 Level 2 Queue Control + F4 Resumable Preemption Coexistence**:
  - *Objective*: Run two independent queues simultaneously: Queue A using Level 2 CTA-granular preemption, Queue B using Level 3 Resumable preemption.
  - *Input*: Concurrent multi-queue execution on single RTX 5060.
  - *Authoritative Source*: Multi-level queue co-existence design.
  - *Verification*: Exit code 0, both queues operate independently without cross-interference.
- **`T3_COMB_04` — F3 Trap Preemption vs F4 Resumable Preemption Comparative Benchmark**:
  - *Objective*: Run side-by-side comparative benchmarking of Trap Preemption vs Resumable Preemption across identical compute workloads.
  - *Input*: 10 iterations of Trap mode vs 10 iterations of Resumable mode on 100 ms GEMM workload.
  - *Authoritative Source*: `ORIGINAL_REQUEST.md §R2`, `PROJECT.md § M4`.
  - *Verification*: Both modes achieve sub-millisecond preemption, metrics compared in summary report.
- **`T3_COMB_05` — F3 Trap Preemption + F5 Hardware Health Under 20-Round Continuous Stress**:
  - *Objective*: Execute 20 sustained Level 3 trap preemption rounds while continuously verifying driver and context integrity.
  - *Input*: `sm120_l3_trap_bench --iters 20`, followed by `nvidia-smi` and context probe.
  - *Authoritative Source*: System health & endurance criteria.
  - *Verification*: Exit code 0, 20/20 rounds pass, `nvidia-smi` healthy, 0 context drops.
- **`T3_COMB_06` — F4 Resumable Preemption + F5 System Integrity with High-Priority Workload**:
  - *Objective*: Preempt background compute workload to execute real-time high-priority query, verifying zero lost work and filesystem integrity throughout.
  - *Input*: Background 1-second simulation kernel preempted by 20 ms inference query; verify `/boot` integrity post-test.
  - *Authoritative Source*: `PROJECT.md § Acceptance Criteria`.
  - *Verification*: Exit code 0, `mismatch == 0`, `/boot` and `/boot/efi` unchanged.

---

### 4.4 Tier 4: Real-World Application Scenarios (5 Tests)

- **`T4_SCEN_01` — LLM / Transformer Attention Preemption Scenario**:
  - *Objective*: Emulate transformer self-attention sequence kernel (batch_size=8, seq_len=2048, head_dim=64); preempt background batch inference upon arrival of interactive high-priority user request.
  - *Input*: `scenarios/workload_transformer_attention.cu`, trigger preemption at 40% forward pass.
  - *Authoritative Source*: Real-world ML serving workload pattern.
  - *Verification*: Preemption response < 500 µs, interactive token latency < 5 ms, batch sequence resumes and finishes with 100% numerical parity.
- **`T4_SCEN_02` — Deep Learning Dense GEMM (Matrix Multiplication) Scenario**:
  - *Objective*: Emulate large FP32/FP16 matrix multiplication (4096 × 4096 × 4096); preempt mid-computation, verify zero lost work upon resume.
  - *Input*: `scenarios/workload_dense_gemm.cu`.
  - *Authoritative Source*: Deep learning training/inference standard workload.
  - *Verification*: Exit code 0, `mismatch == 0` against CPU/cuBLAS reference, P50 latency < 800 µs.
- **`T4_SCEN_03` — Scientific 3D Stencil PDE Solver Scenario**:
  - *Objective*: Emulate multi-iteration 3D finite-difference heat equation / PDE solver; suspend simulation to service urgent rendering/compute query.
  - *Input*: `scenarios/workload_3d_stencil.cu` (256³ domain, 50 time steps), paused at step 25.
  - *Authoritative Source*: High-performance computing iterative solver pattern.
  - *Verification*: Exit code 0, final heat distribution matches un-preempted control run to within machine precision.
- **`T4_SCEN_04` — Multi-Tenant Cloud GPU Time-Sharing SLA Scenario**:
  - *Objective*: Simulate multi-tenant environment with Tenant A (batch throughput) and Tenant B (latency-critical SLA < 2 ms); evaluate SLA satisfaction and preemption turnaround.
  - *Input*: Interleaved tenant queue dispatch with synthetic arrival bursts.
  - *Authoritative Source*: Cloud GPU virtualization & multi-tenancy requirements.
  - *Verification*: 100% of Tenant B requests meet SLA (< 2 ms), Tenant A progress preserved without starvation.
- **`T4_SCEN_05` — 50-Iteration Heavy Preemption Stress & Reliability Workload**:
  - *Objective*: Run 50 continuous preemption cycles alternating between Trap mode and Resumable mode with randomized grid dimensions.
  - *Input*: Randomized stress loop across 50 iterations.
  - *Authoritative Source*: Long-run reliability and stress requirements.
  - *Verification*: 50/50 cycles succeed, 0 driver faults in `dmesg`, 0 memory leaks, GPU stays responsive.

---

## 5. Preemption Latency & Performance Thresholds

| Preemption Mode | Metric | Target Threshold | Hard Max Limit | Authoritative Source |
|---|---|---|---|---|
| **Level 3 Trap (`F3`)** | `P50_US` | **100 – 500 µs** | 1,000 µs | `ORIGINAL_REQUEST.md §R2`, `PROJECT.md § M2` |
| **Level 3 Trap (`F3`)** | `P99_US` | **< 1,000 µs** | 2,000 µs | `xsched-osdi25.pdf` §7.3 |
| **Level 3 Resumable (`F4`)** | `SUSPEND_P50_US` | **100 – 600 µs** | 1,200 µs | `PROJECT.md § Architecture` |
| **Level 3 Resumable (`F4`)** | `RESUME_P50_US` | **50 – 300 µs** | 800 µs | `PROJECT.md § M3` |
| **Level 3 Resumable (`F4`)** | `TURNAROUND_P50_US`| **200 – 900 µs** | 2,000 µs | `PROJECT.md § M3` |
| **Level 2 Drain (`F2`)** | `DRAIN_P50_US` | **< 1,500 µs** | 5,000 µs | `xsched-osdi25.pdf` §7.3 |

---

## 6. Environmental & Hardware Safety Rules

In strict accordance with `ORIGINAL_REQUEST.md §R3` and `PROJECT.md`:
1. **Dual-Boot Protection**: No test script or runner writes to `/boot`, `/boot/efi`, `/etc/fstab`, or EFI partitions. Pre- and post-test assertions verify read-only access and checksum stability.
2. **GPU Hardware Protection**: Power limits, thermal limits, and clock bounds are observed. Tests monitor thermal metrics and abort gracefully if GPU temperature exceeds 85°C.
3. **Driver Non-Destruction**: Tests utilize benign RM control calls (`0x83de0317`, `0x83de0322`, `0x20801210`) with clean state exit (`RET.ABS.NODEC`) to prevent unrecoverable Xid errors or kernel panics.

---

## 7. Execution CLI Guidelines

All tests are runnable via the unified runner:
```bash
# Run all 61 tests across all 4 tiers
xsched/test/sm120_l3_e2e/run_e2e.sh --all

# Run individual tiers
xsched/test/sm120_l3_e2e/run_e2e.sh --tier 1
xsched/test/sm120_l3_e2e/run_e2e.sh --tier 2
xsched/test/sm120_l3_e2e/run_e2e.sh --tier 3
xsched/test/sm120_l3_e2e/run_e2e.sh --tier 4

# Run a specific test case
xsched/test/sm120_l3_e2e/run_e2e.sh --test T1_F1_01

# Structured JSON report output
xsched/test/sm120_l3_e2e/run_e2e.sh --all --json
```
