# Changelog — XSched Blackwell (`sm_120`) Level-3 Preemption

All notable changes to the Blackwell (`sm_120`) Level-3 transparent preemption and scheduling system are documented in this file.

## [v2.2.1-sm120] - 2026-11-24

### Dual-GPU Closed-Loop Verification (`RTX 5080 GB203` + `RTX 5060 GB206`)
- **RTX 5080 (`GB203`, 84 SMs, 16 GiB, Driver `580.95.05`, CUDA `12.9`, PyTorch `2.11+cu128` / cuDNN `9.19` / Triton `3.6.0`) Full Verification**:
  - Zero-warning `C++20 -Wall -Wextra -Werror` build, `79/79` SASS disasm, `11 Caves / 1,703 slots`, **`11 / 11 E2E ALL_PASS`**, **`P50 = 420 µs` CILP** (`ZERO_LOST_WORK`).
  - **`sm120_v2_bench` VRAM Speedups (`v2.0.0` → `v2.2.1`, RTX 5080)**: Entry Guardian `30.78 µs → 0.99 µs` (**31.1×**); Mode 1 `512 CTAs` Evict `971.6 → 110.1 µs` (**8.8×**); Mode 5 `84 CTAs 0K/32K` Evict **`50.9 / 55.4 µs`** (**6.5×–6.7×**), Restore **`30.5 / 41.4 µs`** (**20.1×–25.0×**); Mode 5 `512 CTAs 48K` Evict **`119.9 µs`** (**10.5×**), Restore **`292.0 µs`** (**6.4×**); compact `u_arena` **`139.7×`** (`29.31 MiB` vs `4,096 MiB`).
  - **Suite 1 (`15 / 15 PASS`) + `DEF-03/06` (`PASS`)**: `cuBLAS` `80–137 µs`, `cuFFT` `140 µs`, `cuSPARSE` `113 µs`, `CUB` `55 µs`, `mma.sync` `34 µs`, `cp.async` `114 µs`, **single-wave Blackwell CTA Cluster (`blocks=60 <= sm=84`) dynamically routed to Mode 5 (`111 µs`, `0/7,680`)**, `DEF-03` `34 µs`.
  - **Suite 2**: LLM Serving `HP P50 = 0.270 ms`, `SUSP P50 = 32 µs`, `LP_ret = 102.4%`; 100% SM Saturated Heavy QoS `HP P50 = 0.060 ms` (**102.6×** lower than Lv1 `6.155 ms`), `SUSP P50 = 45 µs` (**136.5×** faster than Lv1), `SLO ≤ 2 ms = 100%`, `LP_ret = 103.4%` (`+27.3 pp` over Mode 1 `76.1%`); Cross-Context BAR0 Bystander **`0.011 ms → 0.011 ms` (`0.000 ms` overhead even at P99/Max `0.012 ms`)**.
  - **Suite 3 (`11 / 11 PyTorch PASS`)**: ResNet-50 `70 µs`, ResNet-18 AdamW training `31 µs` (`7.45e-9`), 4× Llama SDPA `23–991 µs` (`0.000e+00`), Triton 3.6.0 `126 µs` (`0.000e+00`), `PT-06` `empty_cache` `30 µs` (`0.000e+00`), `PT-05-QOS` `HP P50 = 0.290 ms` (`SLO ≤ 2 ms = 100%`, `SUSP P50 = 224 µs`, `LP_ret = 105.3%`).

### Additional 84-SM Edge-Case Fixes Discovered on RTX 5080
- **SMEM Hardware Quantum Fix (`sm120_trap_engine.cpp`)**: Compute `smem_area = AlignUp(total_user_smem + 0x400, 4096)` (when `total_user_smem > 0`) to account for the `0x400 B` driver-reserved region before Blackwell's 4 KiB SMEM allocation quantum (prevents 3 KiB tail-slot overrun on kernels like `CUB Onesweep` with `static_smem = 36,352 B -> SR_SMEMSZ = 40,960 B`).
- **Stream-Ordered `is_restore_wave` Clear (`sm120_trap_engine.cpp`)**: `sm120_cond_clear_hdr` now unconditionally clears `is_restore_wave` (`+0x1c`) upon completion of the restored kernel while keeping `ctrl_dev_[0..27]` conditional on `cmd == 6 || cmd == 2`.
- **Entry Guardian Drain `active_kernel_idx` Tracking (`sm120_guardian.cpp`)**: Replaced 3-instruction `active != 0` check with 1-instruction `lop3_or3_p0(10, 11, kRegZero)` (zero slot shift, `bar_idx` stays at exact slot 39 / `0x270`) and recorded `active_kernel_idx = my_idx` on Entry Guardian State C drains (`st_e_64(4, 0x08, 8, kP0)`).
- **Dynamic SM-Count Delay Scaling (`test_libs_and_features.cu`, `test_pytorch_llm_dnn.py`)**: Automatically scales mid-execution preemption trigger delays by `30 / max(30, sm_count)` across 30-SM RTX 5060 and 84-SM RTX 5080.

## [v2.2.0-sm120] - 2026-11-24

### Performance Breakthroughs (10× – 548× Speedup on RTX 5060 `GB206`)
- **DEF-01 & DEF-08 — 100% GDDR7 VRAM Control Plane + High-Priority Copy Engine DMA (`sm120_trap_engine.h/.cpp`)**:
  - Migrated `Sm120ControlHeader` + `Sm120CtaEntry[65536]` (`ctrl_dev_`) from PCIe Zero-Copy pinned host RAM into GPU GDDR7 VRAM, eliminating PCIe Gen5 round-trip latency (~1.6–2.0 µs/atomic) in the Entry Guardian and Trap Handler hot paths.
  - Added dedicated high-priority `ctrl_op_stream_` using pure Copy Engine `Driver::MemcpyHtoDAsync_v2` / `MemcpyDtoHAsync_v2` (never blocked by 100% SM saturation, unlike compute-engine `cuMemsetD32Async`).
  - Added GPU-native 1-thread PTX helper `sm120_cond_clear_hdr` (~2 µs, strictly ordered on `kStream`) replacing host callback `cuLaunchHostFunc` (`PostRestoreClearHostCb`).
  - **Verified Before/After Impact (`v2.1.0` → `v2.2.0`, RTX 5060)**:
    - Entry Guardian Delta: `43.25 µs` → **`0.43–0.97 µs` (45×–100× faster)**
    - Mode 1 Evict P50 (`512 CTAs, 48K SMEM`): `901.2 µs` → **`38.0 µs` (23.7× faster)**
    - Mode 5 Evict P50 (`512 CTAs, 48K SMEM`): `1,076.2 µs` → **`48.0 µs` (22.4× faster)**
    - `cuBLAS` / `cuSPARSE` Suspend: `66.1 ms` / `50.8 ms` → **`0.856 ms` / `0.109 ms` (77× / 466× faster)**
    - LLM Serving (`15 ms` QoS): `HP P50 32.16 ms → 0.475 ms (67.7×)`, `SUSP P50 31.32 ms → 0.057 ms (548×)`, `SLO ≤ 2 ms: 0% → 100%`, `LP_ret: 19.3% → 98.2%`
    - 100% SM Saturated Heavy QoS (`15 ms`): `HP P50 0.083 ms` (85× lower than Lv1 `7.091 ms`), `SUSP P50 0.066 ms`, `SLO ≤ 2 ms: 100%`, `LP_ret: 100.7%`
    - Real PyTorch `PT-05-QOS` (`m5`): `HP P50 73.76 ms → 0.552 ms (133.6×)`, `SUSP P50 73.45 ms → 0.454 ms (161.8×)`, `SLO ≤ 2 ms: 0% → 100%`, `LP_ret: 0.7% → 101.9%`

### Production Hardening & Universal Compatibility (`DEF-02` .. `DEF-07` + Race Fix)
- **DEF-02 & Phase 2 Adaptive Kernel/Mode Router (`sm120_guardian.h/.cpp`, `cuda_queue.h/.cpp`, `xcommand.h`)**:
  - Static SM120 SASS scanner (`BinaryHasHwMbarrier`) detects `SYNCS` hardware `mbarrier` opcodes (`op9 = 0x1a7 / 0x1b2`) during one-time instrumentation.
  - `CudaQueueLv3Trap::OnHwCommandSubmit` adaptively routes `SYNCS/mbarrier`, Cooperative kernels, multi-wave CTA Clusters (`ClusterSize > 1 && BlockCnt > sm_count`), and CUDA Graphs cleanly to Level-1 boundary (`killable = false`, `ClearProps(kCommandPropertyDeactivatable)`), bringing Suite 1 to **15 / 15 PASS**.
- **DEF-03 — Memory-Safe Deferred Free Queue (`xqueue.h/.cpp`, `async_xqueue.h`, `shim.cpp`)**:
  - `XMemFree_v2` detects any suspended `XQueue` via `XQueueManager::HasAnySuspended()` and defers freeing into `g_deferred_frees` instead of blocking on `WaitAllActive()` (eliminating mid-suspend `cudaFree` / `torch.cuda.empty_cache()` deadlock while guaranteeing suspended LP kernels' buffers stay live until resume). Verified in `test_def03_def06.cu` and PyTorch `PT-06`.
- **DEF-04 — Intercept `cuLaunchCooperativeKernel[_ptsz]` (`cuda_command.h`, `shim.h/.cpp`, `intercept.cpp`)**:
  - Added `CudaCooperativeKernelLaunchCommand` and `XLaunchCooperativeKernel[_ptsz]` interception so cooperative kernels enter XQueue tracking and cleanly route to Level-1 boundary (`HW-07`).
- **DEF-05 — PyTorch ResNet-18 Training `cuDNN` Async SMEM Race & `LaunchEx` Deep Copy (`cuda_command.h/.cpp`)**:
  - Fixed `CUDA_ERROR_INVALID_VALUE (1)` in `PT-02` caused by `cuDNN` updating `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` on the main thread before async `LaunchWorker` launched an earlier larger-SMEM BatchNorm kernel (`EnsureDynamicSmem` zero-hot-path retry).
  - Fixed `CudaKernelLaunchExCommand` attribute deep copy when `param_cnt_ == 0` and max-extent `buffer_size` copy for `params_`. Verified `PT-02` PASS (`max_weight_err = 5.472e-09`, `avg_susp = 0.023 ms`).
- **DEF-06 — Context-Free Worker Thread Automatic `StreamGetCtx` Fallback (`shim.h/.cpp`)**:
  - Added RAII `ScopedStreamContext` so threads with `cuCtxGetCurrent() == nullptr` seamlessly resolve and bind the stream's CUDA context. Verified in `test_def03_def06.cu`.
- **DEF-07 — `max_slots` Occupancy Slack & Restore-Wait Cap (`sm120_trap_engine.cpp`)**:
  - Raised `max_slots` slack to `std::max(64u, sm_count * 2)` and capped prior-restore wait target to `std::min(evicted_slots, max_slots)`.
- **Upstream `CudaQueueLv3Trap` Concurrency Fix (`cuda_queue.h/.cpp`)**:
  - Guarded `last_kernel_` with `last_kernel_mtx_`, removed unsynchronized `OnHwCommandSubmit` write, and prevented trailing `CudaEventRecordCommand`/memory commands from clobbering `last_kernel_`. Eliminated `PT-05-QOS` `launch_worker.cpp:98` crash.
- **Test Harness Hygiene (`test_pytorch_llm_dnn.py`, `test_def03_def06.cu`, `run_all_eval.sh`)**:
  - Added `test_def03_def06.cu` and PyTorch `PT-06`; added missing `torch.cuda.synchronize()` after default-stream tensor creation before non-blocking stream consumption in `test_pytorch_llm_dnn.py`, achieving **11 / 11 PyTorch PASS** (`0.000e+00` bit-exact across ResNet-50, 4 Llama SDPA backends, Triton 3.6.0, and `PT-06`).

## [v2.1.0-sm120] - 2026-11-24

### Added
- **Comprehensive 5-Dimension Evaluation Suite** (`xsched/test/sm120_v2/eval/`):
  - `test_libs_and_features.cu`: 8 closed-source NVIDIA CUDA libraries (`cuBLAS`, `cuBLAS GemmEx FP16/BF16`, `cuBLASLt`, `cuFFT`, `cuSPARSE`, `CUB`) + 7 Blackwell HW boundary tests (`mma.sync`, `cp.async`, `mbarrier`, `CUDA Graph`, Idle Suspend, CTA Cluster + DSMEM, Cooperative Groups).
  - `bench_dnn_llm_scheduling.cu` & `bench_heavy_qos.cu`: LLM Serving (LP Prefill vs HP Decode), Non-Idempotent DNN Training (in-place AdamW vs HP Inference), 100% SM-saturated QoS, and 5/10/20/50 ms frequency sweep across Native / Level-1 / Mode 1 / Mode 5.
  - `bench_bar0_cross_ctx.cu` & `bystander_pure.cu`: Cross-context BAR0 broadcast collateral interference benchmark.
  - `test_pytorch_llm_dnn.py`: Unmodified PyTorch 2.11 + `torchvision` ResNet-50/18 + `cuDNN 9.19` + 4 Llama SDPA backends (`math`/`flash`/`efficient`/`cudnn`) + `Triton 3.6.0` preemption & QoS suite.
  - `run_all_eval.sh`: One-command reproducible runner with built-in `nvidia-smi` GPU idle safety check and MPS isolation.
- **Authoritative Technical Documentation (`doc/`)**:
  - `doc/SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md`: Full real-world library, DNN/LLM, PyTorch/Triton evaluation & defect analysis.
  - `doc/SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md`: Milestone status report, 10-item P0/P1/P2 problem ledger, Phase 1–3 roadmap, and 5080 sync guide.
  - `doc/SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md`: Cross-chip (`GB206` vs `GB203`) and cross-driver (`595.99.02` vs `580.95.05`) adaptation & difference analysis.

### Fixed (6 Deep HAL Bugs Uncovered Under Closed-Source Libraries)
- **FIX-1 (`cuda_queue.cpp`)**: Prevent idle-queue `Suspend/Resume` from poisoning `ctrl_host_->cmd = 6` (`kCmdMode5Restore`) via `has_preempted` check and `ResetControlSync()`.
- **FIX-2 (`sm120_trap_engine.h/.cpp`)**: Replace unconditional `EnqueuePostRestoreClear` DMA zeroing with `PostRestoreClearState` (dedicated micro-mutex + monotonic `epoch`) `Driver::LaunchHostFunc`, eliminating stale wipe on double preemption (`NP >= 2`).
- **FIX-3 (`sm120_trap_engine.cpp`)**: Guard `EmitDynamicSmemLoop` with `SR_SMEMSZ >= 0x401` before subtracting `0x400` (fixes unsigned underflow `cuda error 717` on 0-SMEM kernels) and align `smem_area` to `4096 B`.
- **FIX-4 (`sm120_trap_engine.h/.cpp`)**: Track exact per-kernel arena VA via `kernel_arena_map_[kernel_idx]` (fixes multi-kernel growth `cuda error 700`), expand `layout_stage_` to 256 slots (16 KiB), and raise initial `ctrl_cta_cap_` to 65,536 CTAs with stream sync before growth.
- **FIX-5 (`sm120_guardian.h/.cpp`)**: Route Entry Guardian uniformly by wave-invariant `is_restore_wave` (`+0x1c`) instead of mutable `cmd`, eliminating same-CTA warp split deadlock on `cmd: 6 -> 5` transitions.
- **FIX-6 (`sm120_trap_engine.cpp`)**: Upgrade Cave 0 to 5-interval deadlock-free consensus distinguishing `entry_dec == 2` (Guardian-decided drain: bypass `bar_sync_trap()`) from `entry_dec == 4` (Trap-decided drain: safe `bar_sync_trap()`), shrinking Cave 0 to `139/152` slots (`1,045/1,390` total).

## [v2.0.0-sm120] - 2026-11-23

### Changed
- Completed Blackwell `sm_120` Level-3 v2 clean architecture refactor: simplified to Mode 1 (`kEarlyExit`) + Mode 5 (`kTrapFrame`) + decoupled Context CILP (`ResumableSm120Context`).
- Implemented static 9-Cave engine, permanent `0x500` B 3D Entry Guardian, user-space `BPT.TRAP 0x1` 0-pulse restore, compact `slot_id` pooled `u_arena` (160.6×–436.9× VRAM reduction), D10 `NV_PTIMER_TIME_0` BAR0 pulse safety, and `C++20 -Wall -Wextra -Werror` zero-warning build.
- Verified `11/11 ALL_PASS` × 3 runs on RTX 5080 (`580.95.05`), 79/79 `nvdisasm -b SM120` encoding checks, and cross-driver cave discovery on RTX 5060 (`595.99.02`).
- Removed legacy v1 drafts, vendored NVIDIA driver source trees, and temporary build artifacts.
