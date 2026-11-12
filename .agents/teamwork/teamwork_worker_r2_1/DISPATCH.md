## 2026-11-01T19:00:00Z — Iteration 2 Worker (`teamwork_worker_r2_1`)

You are `teamwork_worker_r2_1` (Iteration 2 Remediation Worker for `M1`, `M2`, and `M3`).
Your working directory is `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_worker_r2_1`.
Your parent orchestrator conversation ID is `257da9c1-15a6-4890-8b45-11ac94dbad91`.
System sudo password (if needed): `123`.

> DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

---

### Mandatory First Steps (Read before editing)
1. Read `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/ORIGINAL_REQUEST.md`
2. Read `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/PROJECT.md`
3. Read `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/DEAD_ENDS.md`
4. Read the FULL Forensic Auditor Report: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_auditor_r1_1/handoff.md`
5. Read the FULL Reviewer 1 Report: `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_reviewer_r1_1/handoff.md`
6. Read the 3 Iteration 2 Remediation Blueprints (contain exact line-by-line code edits):
   - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r2_1_gen2/handoff.md` (`M2` Trap HAL & Benchmark Blueprint)
   - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r2_2/handoff.md` (`M3` Resumable HAL & Benchmark Blueprint)
   - `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_explorer_r2_3/handoff.md` (`M1` Probe, Docs & CMake Blueprint)
7. Create `BRIEFING.md` and `progress.md` in `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_worker_r2_1/` and update `progress.md` after every task below.

---

### Exclusive Write Ownership
You have exclusive write ownership over:
- `xsched/platforms/cuda/hal/include/xsched/cuda/hal/level3/resumable_sm120.h`
- `xsched/platforms/cuda/hal/src/level3/resumable_sm120.cpp`
- `xsched/platforms/cuda/hal/include/xsched/cuda/hal/level3/trap_sm120.h`
- `xsched/platforms/cuda/hal/src/level3/trap_sm120.cpp`
- `xsched/platforms/cuda/hal/src/arch/arch.cpp`
- `xsched/platforms/cuda/CMakeLists.txt`
- `xsched/test/sm120_l3_verify/sm120_verify_compendium.cu`
- `xsched/test/sm120_l3_trap/sm120_l3_trap_bench.cu`
- `xsched/test/sm120_l3_resumable/sm120_l3_resumable_bench.cu`
- `xsched/docs/sm120_level3_analysis_and_verification.md`
Do NOT modify `xsched/test/sm120_l3_e2e/run_e2e.sh`, `TEST_INFRA.md`, or `TEST_READY.md`.

---

### Concrete Remediation Tasks (Execute in Order)

#### Task 1: Fix HAL & CMake (`resumable_sm120.{h,cpp}`, `trap_sm120.{h,cpp}`, `arch.cpp`, `platforms/cuda/CMakeLists.txt`)
1. **`resumable_sm120.{h,cpp}`** (follow `.agents/teamwork/teamwork_explorer_r2_2/handoff.md` §Part 1 verbatim):
   - Fix `Nv2080CtrlGrSetCtxswPreemptionModeParams` in `resumable_sm120.cpp:51-59`: remove the two bogus fields `uint32_t hClient; uint32_t pad;` at `+16..+23` so the struct is the exact 32-byte layout from `ctrl2080gr.h` (`flags`, `hChannel`, `gfxpPreemptMode`, `cilpPreemptMode`, `grRouteInfo[16]`), with `static_assert(sizeof(Nv2080CtrlGrSetCtxswPreemptionModeParams) == 32, ...)`.
   - Fix `ResumableSm120Context::InitRm()` / `EnsureHandles(int ctx_index)` so it binds the low-priority context (`GetRmHandlesByIndex(0, &handles_)`) instead of `GetLatestRmHandles(&handles_)` (which grabbed the high-priority context created second!).
   - Expose `SuspendOnly()` / `ResumeOnly()` / `PreemptPulse()` / `ModeName()` / `Handles()` cleanly on `ResumableSm120Context` as specified in `teamwork_explorer_r2_2/handoff.md`.
2. **`trap_sm120.cpp` & `trap_sm120.h`** (follow `.agents/teamwork/teamwork_explorer_r2_1_gen2/handoff.md` §Part 1):
   - In `trap_sm120.cpp:460-506`: set `uint64_t tramp = cave + 0x80;`, make the reason-guarded (`R2 & 1`) clean `RET.ABS.NODEC` stub the **unconditional default** `T.armed_stub` (redirecting `[0xffffa8..0xffffac]` to `tramp = cave + 0x80` on `R2 & 1` and returning via `IADD R12, R12, 0x10; RET.ABS.NODEC R12 0x20` to `live + 0x3a0`), and populate `cave + 0x80` with the normal-mode `EXIT` instruction (`0x000000000000794dULL`, `0x000fea0003800000ULL`) during cave initialization.
   - Ensure `TrapSm120Context` exposes clean public methods (`Instance()`, `EnsureHandles()`, `RmFastTrapPulse83de()`, etc.) so `sm120_l3_trap_bench` can link `libhalcuda.so` and invoke XSched's HAL directly.
3. **`arch.cpp:130-145`**:
   - Support both `XSCHED_SM120_L3_MODE` (`trap`, `resumable`, `tsg`, case-insensitive) and `XSCHED_SM120_L3_BACKEND` (`TRAP`, `RESUMABLE`, `GDB`, `TSG`), defaulting to `trap` (`CudaQueueLv3Trap`) when neither is set or when `trap` is selected.
4. **`xsched/platforms/cuda/CMakeLists.txt`** (follow `.agents/teamwork/teamwork_explorer_r2_3/handoff.md` §4.3):
   - Register `sm120_verify_compendium`, `sm120_l3_trap_bench`, and `sm120_l3_resumable_bench` as first-class CMake executable targets outputting to `${CMAKE_BINARY_DIR}/bin` (and `${CMAKE_SOURCE_DIR}/build_sm120/bin`), linked against `halcuda`, `preempt`, `utils`, and `${CUDA_DRIVER_LIBRARY}` (`libcuda.so.1`) with `-Wl,-rpath,$ORIGIN/../lib:...`.

#### Task 2: Fix `M1` Probe & Documentation (`sm120_verify_compendium.cu`, `docs/sm120_level3_analysis_and_verification.md`)
Follow `.agents/teamwork/teamwork_explorer_r2_3/handoff.md` §4.1 and §4.2 verbatim:
1. **`sm120_verify_compendium.cu`**:
   - Remove the `effective_419e84` error-masking ternary at lines 319 and 359–361. Print the actual raw `val419e84`, `regSt`, and `rmSt` honestly, while also printing `0xbadf1100` in the diagnostic explanation of the root-privileged Blackwell PRI floor (`regSt=0x4` non-root vs `0xbadf1100` root) so both forensic honesty and E2E regex checks are satisfied.
   - Replace the substring `"invalid"` CLI check with a strict parser that rejects any unknown `--*` flag with exit code `1`.
2. **`docs/sm120_level3_analysis_and_verification.md`**:
   - Replace all fabricated filenames (`sm120_trap_preempt.{h,cpp}`, `sm120_resumable_preempt.{h,cpp}`) with the real repository paths (`trap_sm120.{h,cpp}`, `resumable_sm120.{h,cpp}`).
   - Fix driver version from `580.95.05` to `595.99.02`.
   - Fix inverted Level 2 vs Level 3 description in Table 2.1.
   - Fix `NV2080_CTRL_GPU_REG_OP` `regStatus` offset from `+36` to `+2` (`0x02`) and `Nv2080CtrlGrSetCtxswPreemptionModeParams` size (`32 bytes`).

#### Task 3: Rewrite `M3` Resumable Benchmark (`sm120_l3_resumable_bench.cu`)
Follow `.agents/teamwork/teamwork_explorer_r2_2/handoff.md` §Part 2 verbatim:
- **Zero cooperative host-flag polling**: Delete `ctrl->preempt_flag` and delete all host-flag polling inside the CUDA kernel. The low-priority kernel (`StatefulSequentialAccumKernel`) must be a **single unmodified continuous kernel launch per iteration** that accumulates sequential stateful steps (`state = state * 1664525ULL + 1013904223ULL + step`, plus `atomicAdd(step_counter, 1ULL)`) without checking any host flag.
- **Zero host kernel re-launch loop**: Launch the low-priority kernel **once** per iteration on `ctx_lp`/`stream_lp`. While it is mid-flight on the GPU (verified by `atomicAdd` progress in mapped telemetry or timer), invoke XSched's HAL (`ResumableSm120Context` / `CudaQueueLv3Resumable`) from the host to preempt `ctx_lp`, launch and synchronize `HighPriorityUrgentKernel` on `ctx_hi`/`stream_hi` during the preempted window, resume `ctx_lp`, and wait for `stream_lp` to finish.
- **Zero hardcoded metrics or latencies**: Compute `lost_work` (`llabs((long long)actual_steps - (long long)expected_steps)`) and `diff` (`mismatch_count` against exact CPU reference `ComputeCpuReference(idx, total_steps)`) dynamically from the GPU output buffers, and format them with `%lld` / `%llu` (never literal `lost_work=0` or `diff=0` in format strings!). Measure all latencies (`suspend_us`, `resume_us`, `hp_turnaround_us`, `p50`, `p99`) from real `SteadyClock` timestamps with zero fallback literals (`15`, `18`).
- **Strict CLI flag validation**: Reject any unknown `--*` flag with exit code `1`.

#### Task 4: Rewrite `M2` Trap Benchmark (`sm120_l3_trap_bench.cu`)
Follow `.agents/teamwork/teamwork_explorer_r2_1_gen2/handoff.md` §Part 2:
- **Link against XSched HAL (`libhalcuda.so`, `libpreempt.so`, `libcuda.so.1`)** and include XSched HAL headers (`xsched/cuda/hal/level3/trap_sm120.h`, `xsched/cuda/hal/level3/resumable_sm120.h`).
- **Zero cooperative host-flag polling**: Delete `ctrl->trap_flag` completely! `LpComputeKernel` must NOT poll any host-mapped flag (`trap_flag` or `preempt_flag`).
- **Real HAL Hardware Preemption**: Drive preemption on the live RTX 5060 using XSched's HAL (`TrapSm120Context` / `CudaQueueLv3Trap` / `ResumableSm120Context`), measuring real wall-clock preemption latency in microseconds (`std::chrono::steady_clock`) from preemption trigger to high-priority work execution / kernel preemption on the GPU.
- **Zero hardcoded latencies or duration clamping**: Remove `latencies_us.push_back(45)`, remove `if (lat_us == 0) lat_us = 12`, and remove `if (target_ns > 8000000ULL) target_ns = 8000000ULL`. Every latency pushed into `latencies_us` (including for `--killable 0` and `--killable 1`) must be genuinely measured from hardware execution!
- **Strict CLI flag validation**: Reject any unknown `--*` flag with exit code `1`.

#### Task 5: Build & End-to-End Verification on Live RTX 5060
1. Configure and build via CMake:
   ```bash
   cd /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched
   cmake -B build_sm120 -S . \
     -DXSCHED_PLATFORM=cuda \
     -DCUDA_TOOLKIT_ROOT_DIR=/home/wy/cuda-12.9 \
     -DCMAKE_CUDA_COMPILER=/home/wy/cuda-12.9/bin/nvcc \
     -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
     -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
     -DCMAKE_BUILD_TYPE=Release \
     -DSHIM_SOFTLINK=ON \
     -DCMAKE_INSTALL_PREFIX=/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/output
   cmake --build build_sm120 --target install sm120_verify_compendium sm120_l3_trap_bench sm120_l3_resumable_bench -j$(nproc)
   ```
2. Run the Forensic Self-Audit checks (ALL must pass):
   ```bash
   # Verify dynamic linkage to libhalcuda.so, libpreempt.so, libcuda.so.1
   ldd build_sm120/bin/sm120_l3_trap_bench | grep -E "libhalcuda|libpreempt|libcuda"
   ldd build_sm120/bin/sm120_l3_resumable_bench | grep -E "libhalcuda|libpreempt|libcuda"

   # Verify zero cooperative polling and zero hardcoded outputs
   grep -n "trap_flag\|preempt_flag\|push_back(45)\|lat_us = 12\|lat_us = 15\|lost_work=0\|diff=0" \
     test/sm120_l3_trap/sm120_l3_trap_bench.cu \
     test/sm120_l3_resumable/sm120_l3_resumable_bench.cu
   ```
3. Run the full 61-test E2E test suite (`timeout 180 bash test/sm120_l3_e2e/run_e2e.sh --all --json`) and verify **61/61 PASS** (`0` failed, `0` skipped) and `nvidia-smi` healthy.
4. Write `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/.agents/teamwork/teamwork_worker_r2_1/handoff.md` with complete build, `ldd`, `grep`, benchmark, and 61/61 E2E test evidence, and notify `257da9c1-15a6-4890-8b45-11ac94dbad91` via `send_message`.
