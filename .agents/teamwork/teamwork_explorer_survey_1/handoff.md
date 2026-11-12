# XSched Preemption Model & Repository Survey — Handoff Report

**Core Summary**: XSched (OSDI'25) implements a hierarchical three-level preemption model over hardware queues (`XQueue`), where **Level 1** blocks pending user-space command launches (`LaunchWorker`), **Level 2** drains queued in-flight GPU kernels and aborts unlaunched thread blocks (CTAs) via dynamic binary instrumentation (`check_preempt` / `restore_exec` using undocumented `cuxtra` driver export tables), and **Level 3** preempts currently executing GPU kernels either across contexts via TSG timeslice zeroing (`0xa06c0104`) or per-stream within a single context by patching the CUDA trap handler (`check_preempt_trap`) and triggering a hardware trap (`0x20800122` on GPU register `0x419e84`). Across git branches in this repository, **SM120 (Blackwell) Level 2** is complete and hardware-verified (`origin/release/sm120-level2` @ `eebe930`), whereas **SM120 Level 3** (`origin/sm120-level3-probe` @ `2eb2e63` and `origin/dev/Linuxlevel3` @ `aa69ca7`) identified that `CU_ETID_ToolsTrapHandler` moved from slot 19 to slot 20 and that NVIDIA Driver 595 disables the in-band `0x20800122` register-ops gate (`pGpu+0x911 == 0`) on SM120, motivating three working/experimental SM120 Level 3 paths (TSG timeslice preemption, `cuda-gdb` MI attach/detach, and external debug-session `0x83de0317` trap + code-cave patching + `XQueue` replay).

---

## 1. Observation

### 1.1 XSched Paper & Documentation (`xsched/docs/`)

All paper and presentation files in `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/docs/` were inspected:
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/docs/xsched-osdi25.pdf` (*XSched: Preemptible Scheduling for Diverse XPUs*, OSDI 2025, 13 pages)
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/docs/xsched-osdi25-talk.pdf` (36 presentation slides)
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/docs/xsched-intro-2025-en.md` and `xsched-intro-2025-zh.md`

#### 1.1.1 Core Abstraction (`XQueue`) and Progressive Command Launching (§3, §5.2, Fig. 6)
- **XQueue APIs (`xsched-osdi25.pdf` §3.2, Table 1)**:
  - `submit(xq, xcmd)` — Enqueues an `XCommand` into `xq`'s user-space command buffer instead of submitting directly to the underlying hardware queue (`hwQueue`, e.g., `CUstream`).
  - `wait(xq, xcmd)` — Blocks until `xcmd` completes execution on the XPU.
  - `suspend(xq, flags)` — Preempts `xq` using the configured preemption level (`Lv1`, `Lv2`, or `Lv3`).
  - `resume(xq, flags)` — Resumes `xq` and restores/replays any deactivated or interrupted commands.
- **Progressive Command Launching (`xsched-osdi25.pdf` §5.2, Fig. 6; `preempt/src/xqueue/launch_worker.cpp:164-263`)**:
  - A background `LaunchWorker` thread pulls commands from `CommandBuffer` (`buf_`) and submits them to `kHwQueue` while maintaining at most `threshold_` (`in_flight_threshold`) commands in flight on the hardware queue (`launch_worker.cpp:200-209`).
  - Every `batch_size_` (`command_batch_size`) commands, `LaunchWorker` inserts a synchronization marker (`XQueueSyncCommand` wrapping a `CUevent`) into `sync_cmd_log_` (`launch_worker.cpp:251-262`).
  - In `xsched-osdi25.pdf` §7.2 (Fig. 10), progressive command launching introduces an average runtime overhead of **1.5%** (maximum **3.4%**) across 7 XPUs when `threshold = 8, batch_size = 4` (or `16 / 8`), while bounding the number of in-flight commands queued inside the hardware/driver queue.

#### 1.1.2 Three-Level Preemption Model (`xsched-osdi25.pdf` §4, §6, Table 2, Fig. 3, Fig. 6–8, Fig. 11)

| Preemption Level | Enum (`xPreemptLevel`) | `HwQueue` HAL Interfaces | What It Preempts | Preemption Latency Bound ($T$ = single kernel runtime) | GV100 P99 Preemption Latency (`xsched-osdi25.pdf` §7.3 Fig. 11, $T = 100~\mu\text{s} / 1\text{ ms} / 10\text{ ms} / 100\text{ ms}$) |
|---|---|---|---|---|---|
| **Level 1** | `kPreemptLevelBlock` (`1`) | `Launch(cmd)`, `Synchronize()` | Blocks launching pending commands from `CommandBuffer` (`LaunchWorker::Pause()`); already-launched in-flight commands run to completion | $\approx \text{THRESHOLD} \times T$ | $590~\mu\text{s}~/~4.7\text{ ms}~/~44\text{ ms}~/~441\text{ ms}$ |
| **Level 2** | `kPreemptLevelDeactivate` (`2`) | Lv1 + `Deactivate()`, `Reactivate(log)` | Drains/squashes queued in-flight commands in `hwQueue` (~$7.2~\mu\text{s}$ per flushed kernel) AND aborts unlaunched CTAs (thread blocks) of the running kernel | $\le 1 \times T$ (down to CTA duration if multi-wave grid) | $175~\mu\text{s}~/~1.07\text{ ms}~/~10.1\text{ ms}~/~100\text{ ms}$ |
| **Level 3** | `kPreemptLevelInterrupt` (`3`) | Lv2 + `Interrupt()`, `Restore(log)` | Immediately interrupts Warps/CTAs currently executing on GPU SMs | Constant $\mu\text{s}$-scale (independent of $T$) | $32~\mu\text{s}~/~32~\mu\text{s}~/~32~\mu\text{s}~/~32~\mu\text{s}$ |

- **Hierarchical Execution in `AsyncXQueue` (`preempt/src/xqueue/async_xqueue.cpp:91-150`)**:
  - `AsyncXQueue::Suspend()` (`async_xqueue.cpp:109-112`):
    ```cpp
    launch_worker_->Pause();
    if (level_ >= kPreemptLevelDeactivate) kHwQueue->Deactivate();
    if (level_ >= kPreemptLevelInterrupt) kHwQueue->Interrupt();
    ```
  - `AsyncXQueue::Resume()` (`async_xqueue.cpp:131-134`):
    ```cpp
    const auto log = launch_worker_->GetCommandLog();
    if (level_ >= kPreemptLevelInterrupt) kHwQueue->Restore(log);
    if (level_ >= kPreemptLevelDeactivate) kHwQueue->Reactivate(log);
    launch_worker_->Resume();
    ```
  - Crucially, `LaunchWorker::Pause()` (`launch_worker.cpp:91-118`) checks `bool wait_deactivatable = level_ >= kPreemptLevelDeactivate;` (`line 104`) and waits until no non-deactivatable command is in flight before returning to `Suspend()`.

#### 1.1.3 NVIDIA GPU Level 2 Implementation Details (`xsched-osdi25.pdf` §6.2, Fig. 7, Fig. 8)
- Observed in:
  - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/platforms/cuda/hal/include/xsched/cuda/hal/level2/{cuda_queue.h,guardian.h,instrument.h,mm.h}`
  - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/platforms/cuda/hal/src/level2/{cuda_queue.cpp,instrument.cpp,mm.cpp}`
  - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/platforms/cuda/hal/inject/inject.cu` (lines 1–84)
  - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/ContextFeature/xsched/3rdparty/cuxtra/include/cuxtra/cuxtra.h`
- **Mechanism**:
  1. **Undocumented CUDA Driver Export Tables (`cuxtra`)**:
     - `CU_ETID_ToolsModule` (`6e163fbe-b958-444d-835c-e182aff1991e`): reads relocated kernel SASS (`cuXtraGetBinary`), sets custom entry points (`cuXtraSetEntryPoint`), sets per-launch debugger parameters in constant memory Bank 0 (`cuXtraSetDebuggerParams`), enforces minimum register count (`cuXtraSetLocalRegsPerThread(func, 32)`) and barrier count (`cuXtraSetBarrierCnt(func, 1)`).
     - `CU_ETID_ToolsMemory` (`bfdb432d-bf3c-5a4a-945e-b34029e81e75`): allocates GPU instruction memory blocks (`cuXtraInstrMemBlockAlloc`) and copies patched SASS HtoD/DtoH (`cuXtraInstrMemcpyHtoD` / `cuXtraInstrMemcpyDtoH`).
     - `CU_ETID_ToolsPushBufferHal` (`a6b1ff99-ecc4-c94f-92f9-1928663d5585`): flushes instruction and constant caches (`cuXtraInvalInstrCache`, `cuXtraInvalConstCache`).
  2. **Binary Instrumentation (`InstrumentManager::Instrument` at `instrument.cpp:105-156`)**:
     - For every `CUfunction`, XSched allocates a new instruction memory region containing `guardian` instructions (`check_preempt` in `inject.cu:50-71`) immediately followed by a full copy of the original kernel's SASS instructions (`entry_point_original = entry_point_guardian + guardian_size`).
     - During `InstrumentManager` initialization (`instrument.cpp:193-199`), XSched also loads a shared `restore_exec` stub (`GetResumeInstructions()`, `inject.cu:73-83`) at `entry_point_resume_`.
  3. **Per-Queue `preempt_buffer` Layout (`instrument.h:22-35`, `inject.cu:25-39`)**:
     - Allocated in mapped pinned memory (`cuMemHostAlloc` with `CU_MEMHOSTALLOC_DEVICEMAP` on `main`, or `cuMemAlloc` with async `op_stream_` writes):
       - Bytes `0..3`: `uint32_t global_exit_flag` (`0` = normal, `1` = deactivated)
       - Bytes `4..7`: `uint32_t reserved`
       - Bytes `8..15`: `uint64_t preempt_idx` (command `idx` of the first kernel that observed `global_exit_flag == 1`)
       - Bytes `16 + 8 * block_id .. 16 + 8 * block_id + 3`: `uint32_t block_exit_flag` for CTA `block_id`
       - Bytes `16 + 8 * block_id + 4 .. 16 + 8 * block_id + 7`: `uint32_t block_restore_flag` for CTA `block_id` (`1` = CTA was aborted/interrupted and must re-execute on resume; `0` = CTA completed normally before preemption)
  4. **Per-Launch Constant Memory `DebuggerParams` (`instrument.h:47-53`, `instrument.cpp:208-246`)**:
     - Before launching a kernel, `InstrumentManager::Launch` writes a 28-byte `DebuggerParams` struct via `cuXtraSetDebuggerParams(func, &params, sizeof(params))`:
       ```cpp
       struct DebuggerParams {
           CUdeviceptr preempt_buffer;        // 8B
           CUdeviceptr entry_point_guardian;  // 8B
           uint64_t    kernel_idx;            // 8B
           uint32_t    killable;              // 4B (1 if Lv3 trap preemption enabled & kernel is killable)
       };
       ```
     - On Volta (`sm_70`) and Ampere (`sm_86`), the driver places `DebuggerParams` at `c[0x0][0x1880..0x1898]`.
  5. **CTA-Granular Preemption (`check_preempt` in `inject.cu:50-71`) and Selective CTA Replay (`restore_exec` in `inject.cu:73-83`)**:
     - When a CTA starts at `entry_point_guardian` (`check_preempt`), thread 0 (`(tid.x | tid.y | tid.z) == 0`) reads `global_exit_flag` from `preempt_buffer` bypassing L1 cache (`ld.global.cg.u32`). If `global_exit_flag != 0`, thread 0 sets its CTA's `block_exit_flag = 1`, records `preempt_idx = kernel_idx` if `preempt_idx == 0`, and if `preempt_idx == kernel_idx` sets `block_restore_flag = 1`. After `MEMBAR.SC.CTA` and `BAR.SYNC 0x0`, all threads in the CTA read `block_exit_flag` and, if non-zero, immediately execute `EXIT` (`return;`), skipping the kernel body.
     - On `CudaQueueLv2::Reactivate(log)` (`level2/cuda_queue.cpp:69-105`, `instrument.cpp:255-282`), XSched waits for `kStream` to drain (`Driver::StreamSynchronize(kStream)`), reads `preempt_idx` from `preempt_buf_ptr_ + 8`, zeroes the 16-byte header, and replays all commands in `log` with `idx >= preempt_idx`.
     - The first replayed kernel (`idx == preempt_idx`) is launched with entry point `entry_point_resume_` (`restore_exec`), where each CTA checks its `block_restore_flag`:
       - If `block_restore_flag == 0` (the CTA already finished execution before `Deactivate()`), the CTA immediately executes `EXIT`.
       - If `block_restore_flag != 0` (the CTA was aborted by `check_preempt` or killed mid-execution by Lv3 `check_preempt_trap`), thread 0 clears `block_restore_flag = 0` and all threads jump via `RET.ABS.NODEC R20 0x0` to `entry_point_guardian` to re-execute that CTA.

#### 1.1.4 NVIDIA GPU Level 3 Implementation Details (`xsched-osdi25.pdf` §6.3, Fig. 3, Fig. 11)
The OSDI'25 paper (§6.3, p. 8) and codebase (`platforms/cuda/hal/src/level3/`) present **two distinct Level 3 mechanisms** for NVIDIA GPUs:

1. **TSG-based Preemption (`CudaQueueLv3Tsg`)**:
   - **Files**: `platforms/cuda/hal/include/xsched/cuda/hal/level3/tsg.h`, `platforms/cuda/hal/src/level3/tsg.cpp`.
   - **Mechanism**:
     - Uses undocumented NVIDIA Resource Manager (RM) ioctl controls (`class 0xa06c`, command `0xa06c0103` GET timeslice / `0xa06c0104` SET timeslice) wrapped in `cuXtraGetTimeslice(kCtx, &timeslice_)` and `cuXtraSetTimeslice(kCtx, 0)`.
     - `CudaQueueLv3Tsg::Interrupt()` (`tsg.cpp:24-27`) sets the context's TSG timeslice to `0`, forcing the GPU hardware/driver to immediately context-switch out the TSG (using Pascal+ Compute Instruction-Level Preemption / CILP) and save all SM register/shared-memory state automatically.
     - `CudaQueueLv3Tsg::Restore()` (`tsg.cpp:29-32`) restores the original timeslice via `cuXtraSetTimeslice(kCtx, timeslice_)`.
   - **Trade-off (`xsched-osdi25.pdf` §6.3)**: Preempts the **entire CUDA context (TSG)** rather than an individual `CUstream` (`hwQueue`). Thus, it supports inter-context / inter-process Level 3 preemption without binary instrumentation or idempotency requirements, but cannot selectively preempt one `CUstream` while keeping another `CUstream` in the same `CUcontext` running.

2. **Queue-based / Trap-based Preemption (`CudaQueueLv3Trap`)**:
   - **Files**:
     - `platforms/cuda/hal/include/xsched/cuda/hal/level3/{cuda_queue.h,interrupt.h,trap.h}`
     - `platforms/cuda/hal/src/level3/{cuda_queue.cpp,interrupt.cpp}`
     - `platforms/cuda/hal/src/arch/{sm70.cpp,sm86.cpp}`
     - `platforms/cuda/hal/inject/{inject.cu,inject_sm86.asm,trap_handler_sm86.asm}`
   - **Class Hierarchy**: `CudaQueueLv3Trap` inherits directly from `CudaQueueLv2` (`include/xsched/cuda/hal/level3/cuda_queue.h:15`).
   - **Step 1 — Trap Handler Patching (`InterruptContext::InstrumentTrapHandler`, `interrupt.cpp:29-61`, `sm86.cpp:305-331`, `sm70.cpp:283-309`)**:
     - Queries the stock CUDA trap handler device address and size via `cuXtraGetTrapHandlerInfo(kCtx, &trap_handler_dev_, &trap_handler_size_)` (backed by `CU_ETID_ToolsTrapHandler` = `cc529e94-5c4e-9d46-837c-9625868334e4`, slot 19 on SM70/SM86).
     - Allocates a GPU instruction memory block `inject_dev` via `InstrMemAllocator::Alloc`.
     - Patches 16 bytes (1 SASS instruction: `IADD3 R1, R1, -0x10, RZ`) inside the stock trap handler right after `trap_entry_bootstrap` (`offset 0x260` on SM70, `offset 0x3e0` on SM86) with an absolute jump (`JMP inject_dev`) to `inject_dev`.
     - Copies `trap_inject_instrs` (`check_preempt_trap`) into `inject_dev`, followed by the displaced `IADD3 R1, R1, -0x10, RZ` instruction and an absolute `JMP` back to `trap_handler_dev + offset + 16`, then invalidates the instruction cache (`cuXtraInvalInstrCache`).
   - **Step 2 — Firing the Hardware Trap (`CudaQueueLv3Trap::Interrupt`, `level3/cuda_queue.cpp:13-20`, `interrupt.cpp:99-106`)**:
     - During `AsyncXQueue::Suspend()`, `kHwQueue->Deactivate()` runs first, setting `global_exit_flag = 1` in the target stream's `preempt_buffer`.
     - Next, `CudaQueueLv3Trap::Interrupt()` synchronizes `op_stream_` (`Driver::StreamSynchronize(instrument_manager_->OpStream())`) to guarantee `global_exit_flag = 1` is visible in device memory, and calls `InterruptContext::Interrupt()`, which invokes `cuXtraTriggerTrap(kCtx)`.
     - `cuXtraTriggerTrap(kCtx)` issues RM control `0x20800122` (`NV2080_CTRL_CMD_GPU_EXEC_REG_OPS`) on the GPU subdevice to write bit 31 (`0x80000000`) to GPU register `0x419e84`, immediately trapping **all** active warps across the GPU into the trap handler in normal kernel context.
   - **Step 3 — Per-Stream Selective Warp Abort Inside the Trap Handler (`check_preempt_trap`, `inject.cu:85-122`, `inject_sm86.asm:72-154`)**:
     - Every trapped warp enters the trap handler, executes `trap_entry_bootstrap`, and jumps to `check_preempt_trap` in `inject_dev`:
       1. Saves `R0..R9` to thread-local memory `[0xfffe00..0xfffe24]`.
       2. Loads `preempt_buffer`, `kernel_idx`, and `idempotent`/`killable` from the warp's own kernel constant memory (`c[0x0][0x1880..0x1898]`).
       3. Executes `MEMBAR.SC.GPU; ERRBAR; CCTL.IVALL;` (`inject_sm86.asm:103-105`) to invalidate the SM L1 data cache so host updates to `preempt_buffer` are visible.
       4. Checks `if (idempotent == 0 || *global_exit_flag == 0)` (`inject.cu:98-100`).
          - **Non-target stream or non-killable kernel**: If the warp belongs to another `CUstream` (which was NOT deactivated, so its `preempt_buffer->global_exit_flag == 0`) or `killable == 0`, the warp branches to `out:`, restores `R0..R9` from `[0xfffe00..0xfffe24]`, executes `IADD3 R1, R1, -0x10, RZ`, jumps back to `trap_handler_dev + offset + 16`, clears the trap, and **continues executing uninterrupted**.
          - **Target deactivated stream (`idempotent != 0 && *global_exit_flag != 0`)**: The warp writes `preempt_idx = kernel_idx` (if `preempt_idx == 0`), sets `block_restore_flag = 1` for its CTA (`SR_CTAID.X/Y/Z`), executes `MEMBAR.SC.GPU; ERRBAR;`, restores `R0..R9`, and executes **`EXIT`** (`inject_sm86.asm:142`) directly inside the trap handler!
     - Because `Deactivate()` already set `global_exit_flag = 1`, any remaining unlaunched CTAs of the interrupted kernel and any subsequent in-flight kernels on the target `CUstream` immediately `EXIT` via `check_preempt` (`Lv2`), emptying the target `CUstream` in microseconds.
   - **Step 4 — Resume & CTA-Granular Re-execution (`CudaQueueLv3Trap::Restore` + `CudaQueueLv2::Reactivate`)**:
     - `CudaQueueLv3Trap::Restore(log)` (`level3/cuda_queue.cpp:22-25`) is a no-op because `CudaQueueLv2::Reactivate(log)` handles recovery: it relaunches the interrupted kernel (`idx == preempt_idx`) via `entry_point_resume_` (`restore_exec`), which re-executes **only** the CTAs whose `block_restore_flag == 1` (those killed mid-execution by `check_preempt_trap` or drained by `check_preempt`), while CTAs that had already completed before the trap (`block_restore_flag == 0`) immediately `EXIT`.

---

### 1.2 Repository & Git Branch Analysis

#### 1.2.1 Git Branches, Remotes, Tags, and Workspace Directories
In `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched`:
- **Remote**: `origin` → `https://github.com/OrderLab/xsched.git` (local fetch includes project branches).
- **Current `HEAD`**: `main` (`f49289fa050f9595c21f5e713972c4ca1f9f9ef7`, `"feat(lg200): add Level-Zero queue implementation and patch ze_api.h"`), working tree clean, no git stashes.
- **All Git Branches**:

| Branch / Ref | Commit Hash | Date | Summary / Scope |
|---|---|---|---|
| `main` / `origin/main` | `f49289fa050f` | 2025-11-28 | Upstream OSDI'25 codebase (`sm70` + `sm86` Lv1/Lv2/Lv3 support; no `sm120`) |
| `origin/release/sm120-level2` | `eebe9301e6fc` | 2026-03-31 | Clean release branch for **Blackwell (`sm_120`) Level 2** (`GuardianSM120` + `extract_sass.py`) |
| `origin/sm120-level2` | `6c07b2e33501` | 2026-03-31 | Development branch for `sm_120` Level 2 |
| `origin/evidence/sm120-level2` | `bbf1c1327b2d` | 2026-03-31 | Full MVE (48/48) & integration test (26/26) evidence logs for `sm_120` Level 2 |
| `origin/docs/sm120-release-audit` | `e83a89d6502a` | 2026-03-31 | Audit report (`docs/sm120-release-diff-audit.md`) of `sm_120` Level 2 release diff |
| `origin/sm120-level3-probe` | `2eb2e635de0c` | 2026-03-19 | Initial `sm_120` Level 3 probe: discovered `CU_ETID_ToolsTrapHandler` slot 20 (`TarpHandlerSM120`) and `check_preempt_trap` SM120 SASS |
| `origin/dev/Linuxlevel3` | `aa69ca770fd2` | 2026-10-19 | Comprehensive Linux `sm_120` Level 3 reverse-engineering, root-cause proof (`pGpu+0x911`), and 3 working/experimental backends (`TSG`, `GDB`, `0x83de0317` Trap) |

- **Additional Directories in `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/`**:
  - `open-gpu-kernel-modules-550.163.01/` — NVIDIA open GPU kernel modules source for driver `550.163.01`.
  - `open-gpu-kernel-modules-595.99.02/` — NVIDIA open GPU kernel modules source for driver `595.99.02`.
- **Sibling Workspace Directories under `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/`**:
  - `ContextFeature/xsched/3rdparty/cuxtra/` — Full source and headers of `libcuxtra.so` (`include/cuxtra/cuxtra.h`).
  - `Level3forLinux/xsched/` — Checked out at `aa69ca770fd2` (`origin/dev/Linuxlevel3`).

#### 1.2.2 Completed SM120 (Blackwell) Level 2 Implementation (`origin/release/sm120-level2` @ `eebe930`)
Inspected via `git show origin/release/sm120-level2:...` and `origin/docs/sm120-release-audit:docs/sm120-release-diff-audit.md`:
- **Modified/Added Files vs. `main` (`65af4bf`)**:
  1. `platforms/cuda/hal/include/xsched/cuda/hal/arch/sm120.h` (new, 26 lines): Declares `GuardianSM120` with `RequiredRegs() = 32`, `RequiredBarriers() = 1`.
  2. `platforms/cuda/hal/src/arch/sm120.cpp` (new, 189 lines): Contains `GuardianSM120::GetGuardianInstructions` (50 SASS instructions, 800 bytes) and `GuardianSM120::GetResumeInstructions` (32 SASS instructions, 512 bytes) for `sm_120`.
  3. `platforms/cuda/hal/src/arch/arch.cpp` (modified, +4 lines): Maps `case 120:` in `GetGuardian()` to `GuardianSM120` and in `CreateCudaQueue()` to `CudaQueueLv1` / `CudaQueueLv2`.
  4. `platforms/cuda/hal/include/xsched/cuda/hal/level2/guardian.h` & `platforms/cuda/hal/src/level2/instrument.cpp`: Adds virtual `RequiredRegs()` and `RequiredBarriers()` on `Guardian` so architecture-specific floors are passed to `cuXtraSetLocalRegsPerThread` and `cuXtraSetBarrierCnt`.
  5. `platforms/cuda/hal/inject/inject.cu` & `platforms/cuda/hal/inject/Makefile` & `tools/instrument/{extract_sass.py,README.md}`: Automated SASS extraction and patching pipeline for `sm_120`.
- **Key SM120 Architectural Changes Discovered and Solved in Level 2**:
  1. **Constant Memory `c[0x0]` Relocation**:
     - On `sm_70` / `sm_86`, `cuXtraSetDebuggerParams` writes the 28-byte `DebuggerParams` struct to `c[0x0][0x1880..0x1898]`, and `gridDim.x/y` live at `c[0x0][0x0]` and `c[0x0][0x4]`.
     - On `sm_120` (Blackwell), `cuXtraSetDebuggerParams` writes `DebuggerParams` to **`c[0x0][0x170..0x188]`**:
       - `c[0x0][0x170]` / `c[0x0][0x174]`: `preempt_buffer` (low/high 32 bits)
       - `c[0x0][0x178]` / `c[0x0][0x17c]`: `entry_point_guardian` (low/high 32 bits)
       - `c[0x0][0x180]` / `c[0x0][0x184]`: `kernel_idx` (low/high 32 bits)
       - `c[0x0][0x188]`: `killable` / `idempotent` flag (32 bits)
     - `gridDim.x` and `gridDim.y` on `sm_120` are located at **`c[0x0][0x374]`** and **`c[0x0][0x378]`**.
  2. **SM120 `LDC` Instruction Encoding & Control Flow (`tools/instrument/extract_sass.py`)**:
     - `inject.cu` is compiled with `nvcc 12.9 -cubin --keep-device-functions -Xptxas -astoolspatch -arch=sm_120`.
     - `nop()` (`asm volatile("trap;")`, encoded as `BPT.TRAP` in SASS) placeholders are replaced 1:1 with SM120 `LDC` (`ULDC`/scalar constant load) instructions:
       - `word0 = 0xff007b82 | (reg << 16) | ((offset >> 2) << 40)`
       - `word1 = 0x000fc00000000800`
     - For `guardian` (`check_preempt`), the trailing `RET.ABS.NODEC` is stripped so execution falls straight through `BSYNC.RECONVERGENT B0` into the relocated kernel instructions appended immediately after `guardian`.
     - For `resume` (`restore_exec`), `RET.ABS.NODEC R20 0x0` is preserved to jump to `entry_point_guardian` loaded in `(R20, R21)` from `c[0x0][0x178..0x17c]`.

#### 1.2.3 SM120 (Blackwell) Level 3 Investigation & Branches (`origin/sm120-level3-probe` @ `2eb2e63` and `origin/dev/Linuxlevel3` @ `aa69ca7`)
Inspected across `origin/sm120-level3-probe` and `origin/dev/Linuxlevel3` (`test/sm120_l3_probe/L3_FINAL_SUMMARY_ZH.md`, `test/sm120_l3_trap_v2/report/00-总索引与结论全表.md`, `01-成果汇报.md`, `02-中断入口.md`, `03-核心结论.md`, `kb/ABI.md`, `L3-linux-next-steps-20261019.md`, and `platforms/cuda/hal/src/level3/`):

1. **Discovery 1 — `CU_ETID_ToolsTrapHandler` Moved from Slot 19 to Slot 20 on SM120 (`platforms/cuda/hal/src/arch/sm120.cpp:261-327`)**:
   - In `libcuxtra.so`, `cuXtraGetTrapHandlerInfo` calls slot 19 (`offset 0x98`) of `CU_ETID_ToolsTrapHandler` (`cc529e94-5c4e-9d46-837c-9625868334e4`), which is the Pascal/Volta/Ampere getter (`sub_353E50` in `libcuda.so`). On SM120, slot 19 checks `dev+956 > 2` and returns `CUDA_ERROR_INVALID_DEVICE (101)`.
   - On SM120, the table length increased from `0xA0` to `0xA8`, adding **slot 20 (`offset 0xA0`, `sub_353EC0`)**.
   - Calling slot 20 with a 40-byte struct (`info[0] = 0x2800000000ULL`) returns `{flags, struct_size, mem_obj_handle, stub_offset = 0x880, handler_size = 0x940, reserved}`.
   - Passing `mem_obj_handle` (`info[1]`) to `CU_ETID_ToolsMemory` slot 24 (`ObjGetPc`) and slot 25 (`ObjGetSize`) yields the tools trap-handler GPU virtual address (`base + 0x15f5300`) and size (`0x1300`).
   - Furthermore (`report/kb/ABI.md`), on Driver 595.99.02 / 595.71.05, the **live** hardware trap handler `.text.traphandler` actually executed by trapped SM120 warps resides at **`base + 0x15f0f00`** (where `base = trap_reason_table - 0xc38300` decoded from `slot20_img[0x130..0x158]`), while `slot20_pc + 0xA00..0x12FF` provides an unused executable code cave in the same 24 MB instruction segment.

2. **Discovery 2 — Why the Paper's In-Band `cuXtraTriggerTrap` (`0x20800122`) Is Blocked on SM120 + Driver 595 (`report/02-中断入口.md`, `report/03-核心结论.md`)**:
   - `cuXtraTriggerTrap(ctx)` invokes `NV2080_CTRL_CMD_GPU_EXEC_REG_OPS` (`0x20800122`, handled in `nvidia.ko` by `subdeviceCtrlCmdGpuExecRegOps_cmn@0x2dcfc0`) to write `0x80000000` to GPU register `0x419e84`.
   - Binary disassembly of `nvidia.ko` (`595.99.02` on RTX 5060 and `595.71.05` on RTX 5090) and `open-gpu-kernel-modules-595.99.02` revealed a software capability byte gate at **`pGpu + 0x911`**:
     - `subdeviceCtrlCmdGpuExecRegOps_cmn` (`+0x2dd122`): `cmpb $0x0, 0x911(%r12)`
     - `ksmdbgssnCtrlCmdDebugExecRegOps_IMPL` (`0x83de031d`, `+0x21a9cc`): `cmpb $0x0, 0x911(%rbx)`
   - On SM120 under Driver 595, `__nvoc_init_dataField_OBJGPU` initializes `pGpu + 0x911 = 0` and never sets it to `1`. When `pGpu + 0x911 == 0`, `0x20800122` skips the hardware register execution loop (`nv2080CtrlCmdGpuExecRegOps_IMPL`) and returns `status = 0` without touching GPU hardware registers (even reading `PMC_BOOT_0` or `PTIMER` returns `val_lo = 0, val_hi = 0`).
   - By contrast, on RTX 3090 (`sm_86`, Driver `570.124.04`), `pGpu + 0x911 == 1`, and the exact same XSched Level 3 trap code achieves **229–241 $\mu\text{s}$ P50 preemption latency** and `bad = 0` across 200 preemptions (`report/03-核心结论.md`).

3. **Discovery 3 — Three Working/Experimental SM120 Level 3 Backends Implemented on `origin/dev/Linuxlevel3` (`platforms/cuda/hal/src/level3/`)**:
   - Controlled via environment variable `XSCHED_SM120_L3_MODE` in `platforms/cuda/hal/src/level3/cuda_queue.cpp`:
     1. **TSG Timeslice Preemption (`CudaQueueLv3Tsg`, `tsg.cpp`)**:
        - Uses `cuXtraSetTimeslice(kCtx, 0)` (`0xa06c0104`). Works out-of-the-box on SM120 Linux without `pGpu+0x911` for inter-context / inter-process Level 3 preemption.
     2. **GDB MI Backend (`GdbInterruptContext`, `gdb_interrupt.cpp`, `XSCHED_SM120_L3_MODE=gdb`)**:
        - Spawns a persistent `cuda-gdb --interpreter=mi2 --quiet` co-process, attaches to the target PID on `Interrupt()` (triggering hardware CILP context freeze in **~106–116 ms**), and detaches on `Restore()` (resuming bit-exact kernel execution in **~8–10 ms**, with zero SASS instrumentation overhead and `bad = 0` bit-exact accuracy).
     3. **External Debug Session `0x83de0317` Trap + Code Cave Patch + `XQueue` Replay (`TrapSm120Context`, `trap_sm120.cpp`, `XSCHED_SM120_L3_MODE=trap`)**:
        - Bypasses `pGpu+0x911` by opening `/dev/nvidiactl` as a separate RM client (`NV01_ROOT = 0x0`, `hClient2`), sharing the target device/subdevice/context with `RS_ACCESS_DEBUG` (`share_policy.access = 0xd04`), allocating `GT200_DEBUGGER` (`class 0x83de`, `CLASS_SMDBGSSN`), and issuing RM control **`0x83de0317`** (`Suspend` trap).
        - `0x83de0317` fires an authentic GSP/hardware SM trap in **0.204 ms** that diverts all running warps into `live + 0x390` (`CALL.ABS` to the patched code cave at `slot20_pc + 0xA00`), where warps check `preempt_buffer` (`c[0x0][0x170]`) and execute `EXIT`.
        - Caveat on `0x83de0317`: GSP/driver treats `0x83de0317` + warp `EXIT` as a fatal sticky error (`CUDA_ERROR_LAUNCH_FAILED = 719`) on the current `CUcontext` after ~27–30 ms of driver teardown; `test/sm120_l3_trap_v2/tests/p5_xqueue_e2e.cu` works around this in user space by tearing down the faulted context (`cuDevicePrimaryCtxReset`) and replaying the uncompleted command log from `XQueue`, completing end-to-end interrupt + high-priority kernel + replay in **0.234 s** (`mismatch = 0`).

---

## 2. Logic Chain

1. **Why XSched Needs All Three Preemption Levels**:
   - From `xsched-osdi25.pdf` §4 & `preempt/src/xqueue/async_xqueue.cpp:109-112`, XSched's preemption model is strictly cumulative (`Lv3` calls `Pause()` → `Deactivate()` → `Interrupt()`).
   - **Level 1 (`Pause()`)** stops user-space command submission from `LaunchWorker`, bounding queued work to `threshold_` commands, but cannot stop commands already submitted to `CUstream`.
   - **Level 2 (`Deactivate()` + `Reactivate()`)** instruments every kernel with `check_preempt` (`guardian`) and `restore_exec` (`resume`). Setting `global_exit_flag = 1` causes both subsequent queued kernels in `CUstream` and not-yet-dispatched CTAs of the running kernel to immediately `EXIT` (~7.2 $\mu\text{s}$), recording `preempt_idx` and per-CTA `block_restore_flag = 1`. However, CTAs/warps already executing on SMs (e.g., long-running or single-wave persistent kernels) will not hit `check_preempt` again.
   - **Level 3 (`Interrupt()` + `Restore()`)** closes this gap by forcing currently executing warps off the SMs.

2. **How Level 3 Achieves Per-Stream (`CUstream`) Preemption Inside a Shared `CUcontext`**:
   - A hardware GPU trap (`cuXtraTriggerTrap`, register `0x419e84` bit 31) or TSG timeslice zeroing (`cuXtraSetTimeslice(ctx, 0)`) affects the entire GPU or `CUcontext`, not a single `CUstream`.
   - XSched's `CudaQueueLv3Trap` turns a global hardware trap into a **per-stream (`hwQueue`) interrupt** by combining Level 2's per-stream `preempt_buffer` (stored in each kernel's constant memory `c[0x0]`) with a patched trap handler (`check_preempt_trap` at `inject.cu:85-122`):
     - When `Interrupt()` fires the global trap, warps from **all** streams enter `check_preempt_trap`.
     - Each warp invalidates L1 cache (`CCTL.IVALL`) and reads its own stream's `*global_exit_flag` via `c[0x0]`.
     - Warps belonging to non-preempted streams see `*global_exit_flag == 0`, restore `R0..R9`, return to the stock trap handler, and resume execution.
     - Warps belonging to the preempted stream see `*global_exit_flag == 1` and `idempotent == 1`, set their CTA's `block_restore_flag = 1` and `preempt_idx = kernel_idx`, and execute `EXIT` inside the trap handler.
     - On `Resume()`, `CudaQueueLv2::Reactivate(log)` relaunches the interrupted kernel via `restore_exec` (`entry_point_resume_`), which skips already-completed CTAs (`block_restore_flag == 0`) and re-executes only the CTAs killed by `check_preempt_trap` or `check_preempt` (`block_restore_flag == 1`).

3. **Why SM120 Level 2 Succeeded Cleanly While SM120 Level 3 Required New Mechanisms**:
   - SM120 Level 2 (`origin/release/sm120-level2`) only depends on `CU_ETID_ToolsModule`, `CU_ETID_ToolsMemory`, and `CU_ETID_ToolsPushBufferHal`, which remain functional on SM120 once the constant memory offsets (`c[0x0][0x170..0x188]` for `DebuggerParams`, `c[0x0][0x374..0x378]` for `gridDim`) and SM120 `LDC` encodings (`extract_sass.py`) are updated.
   - SM120 Level 3 (`CudaQueueLv3Trap`) depends on two additional driver capabilities:
     1. Locating the trap handler (`CU_ETID_ToolsTrapHandler`), which moved from slot 19 to slot 20 on SM120 (solved in `TarpHandlerSM120::GetTrapHandlerInfo`, plus locating the live `.text.traphandler` at `base + 0x15f0f00`).
     2. Firing an in-band non-fatal hardware trap (`cuXtraTriggerTrap` → `0x20800122` on register `0x419e84`), which is blocked in Driver 595 on SM120 because `pGpu + 0x911 == 0` in `nvidia.ko`.
   - Consequently, on unmodified Driver 595, SM120 Level 3 must either:
     - Use **TSG preemption (`CudaQueueLv3Tsg`)** (`0xa06c0104`) when scheduling across separate CUDA contexts/processes,
     - Use **`GdbInterruptContext`** (`cuda-gdb` attach/detach) or **`TrapSm120Context`** (`0x83de0317` + context reset/replay) in pure user space, or
     - Enable `pGpu + 0x911 = 1` in the open GPU kernel module (`open-gpu-kernel-modules-595.99.02` is already present in the workspace root!) or verify whether `0x20800122` works once `pGpu + 0x911` is enabled.

---

## 3. Caveats

1. **Kernel Module (`open-gpu-kernel-modules-595.99.02`) Modification vs. Pure User-Space Constraint**:
   - In `origin/dev/Linuxlevel3` (`test/sm120_l3_trap_v2/report/03-核心结论.md`), all experiments were conducted under a strict "no kernel module modification / no root" constraint (`pGpu + 0x911 == 0`). However, the current workspace (`/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/`) contains the full source trees of `open-gpu-kernel-modules-550.163.01` and `open-gpu-kernel-modules-595.99.02`, which were not yet inspected in detail in this survey. If kernel module modification or `0x20800122` / `0x83de` analysis inside `open-gpu-kernel-modules-595.99.02` is in scope for the next milestone, those source trees are immediately available locally.
2. **Branch Divergence**:
   - The local `xsched` checkout is currently on `main` (`f49289f`), which does **not** have `origin/release/sm120-level2` (`eebe930`) or `origin/dev/Linuxlevel3` (`aa69ca7`) merged yet. Any SM120 implementation work should start from or cherry-pick `origin/release/sm120-level2` (`eebe930`) and/or `origin/dev/Linuxlevel3` (`aa69ca7`).

---

## 4. Conclusion

1. **XSched Preemption Architecture (OSDI'25 Paper & `main`)**:
   - XSched's three-level preemption model is implemented in `AsyncXQueue::Suspend()` / `Resume()` (`preempt/src/xqueue/async_xqueue.cpp:91-150`) on top of progressive command launching (`LaunchWorker`, `preempt/src/xqueue/launch_worker.cpp:164-263`).
   - **Level 1 (`kPreemptLevelBlock`)** pauses user-space command launching from `CommandBuffer`.
   - **Level 2 (`kPreemptLevelDeactivate`)** uses `cuxtra` dynamic binary instrumentation (`platforms/cuda/hal/src/level2/instrument.cpp`) to prepend `check_preempt` (`guardian`) and install `restore_exec` (`resume`), using a per-stream `preempt_buffer` and per-launch constant memory `DebuggerParams` (`c[0x0][0x1880..0x1898]` on SM70/SM86; `c[0x0][0x170..0x188]` on SM120) to drain queued kernels and abort unlaunched CTAs, then selectively replay only uncompleted CTAs on `Reactivate()`.
   - **Level 3 (`kPreemptLevelInterrupt`)** interrupts running GPU kernels via two mechanisms on NVIDIA GPUs:
     - **TSG Preemption (`CudaQueueLv3Tsg`, `tsg.cpp`)**: zeroes TSG timeslice (`0xa06c0104`) for context-wide hardware CILP preemption.
     - **Trap Preemption (`CudaQueueLv3Trap`, `interrupt.cpp` + `inject.cu:85-122`)**: patches the CUDA trap handler with `check_preempt_trap` and pulses GPU register `0x419e84` bit 31 via `cuXtraTriggerTrap` (`0x20800122`), causing warps of the deactivated stream (`*global_exit_flag == 1 && idempotent == 1`) to mark their CTA's `block_restore_flag = 1` and execute `EXIT` inside the trap handler while warps of other streams resume normally.
2. **Current Repository State for SM120 (Blackwell)**:
   - **SM120 Level 2 (`origin/release/sm120-level2` @ `eebe930`)** is complete, clean (+425/-5 lines across 8 files), and verified (48/48 MVE tests, 26/26 integration tests).
   - **SM120 Level 3 (`origin/sm120-level3-probe` @ `2eb2e63`, `origin/dev/Linuxlevel3` @ `aa69ca7`)** resolved trap-handler discovery via `CU_ETID_ToolsTrapHandler` slot 20 (`TarpHandlerSM120::GetTrapHandlerInfo`) and live `.text.traphandler` patching (`base + 0x15f0f00`), identified the Driver 595 `pGpu + 0x911 == 0` gate blocking `0x20800122`, and implemented three SM120 Level 3 backends in `platforms/cuda/hal/src/level3/`: `CudaQueueLv3Tsg`, `GdbInterruptContext` (`gdb_interrupt.cpp`), and `TrapSm120Context` (`trap_sm120.cpp`).

---

## 5. Verification Method

All observations and conclusions in this report can be independently verified with the following commands in `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched`:

1. **Verify XSched Paper & Core Preemption Flow (`main`)**:
   - Inspect `preempt/src/xqueue/async_xqueue.cpp` lines 91–150 (`Suspend` / `Resume`) and `preempt/src/xqueue/launch_worker.cpp` lines 164–263 (`LaunchWorker`).
   - Inspect `platforms/cuda/hal/src/level2/instrument.cpp` lines 105–282 and `platforms/cuda/hal/inject/inject.cu` lines 50–122 (`check_preempt`, `restore_exec`, `check_preempt_trap`).
   - Inspect `platforms/cuda/hal/src/level3/{cuda_queue.cpp,interrupt.cpp,tsg.cpp}`.

2. **Verify SM120 Level 2 Implementation (`origin/release/sm120-level2`)**:
   ```bash
   git -C /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched diff 65af4bf..origin/release/sm120-level2 --stat
   git -C /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched show origin/docs/sm120-release-audit:docs/sm120-release-diff-audit.md
   git -C /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched show origin/release/sm120-level2:platforms/cuda/hal/src/arch/sm120.cpp
   ```

3. **Verify SM120 Level 3 Investigation & Backends (`origin/sm120-level3-probe` & `origin/dev/Linuxlevel3`)**:
   ```bash
   git -C /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched show origin/dev/Linuxlevel3:test/sm120_l3_trap_v2/report/00-总索引与结论全表.md
   git -C /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched show origin/dev/Linuxlevel3:platforms/cuda/hal/src/arch/sm120.cpp
   git -C /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched show origin/dev/Linuxlevel3:platforms/cuda/hal/src/level3/trap_sm120.cpp
   git -C /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched show origin/dev/Linuxlevel3:platforms/cuda/hal/src/level3/gdb_interrupt.cpp
   ```
