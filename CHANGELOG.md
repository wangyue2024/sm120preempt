# Changelog — XSched Blackwell (`sm_120`) Level-3 Preemption

All notable changes to the Blackwell (`sm_120`) Level-3 transparent preemption and scheduling system are documented in this file.

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
