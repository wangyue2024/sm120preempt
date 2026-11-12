# Handoff Report: M2 Genuine HAL & Trap Benchmark Remediation Blueprint (`teamwork_explorer_r2_1_gen2`)

## 1. Observation

Direct inspection of the Iteration 1 codebase, forensic audit report (`teamwork_auditor_r1_1`), and reviewer report (`teamwork_reviewer_r1_1`) reveals the following concrete facts and defects:

### 1.1 `sm120_l3_trap_bench.cu` Facade & Hardcoded Latency Defects
- **File**: `xsched/test/sm120_l3_trap/sm120_l3_trap_bench.cu`
- **Dynamic Linkage**: `ldd build_sm120/bin/sm120_l3_trap_bench` shows only `libc.so.6`, `libstdc++.so.6`, `libgcc_s.so.1`, `libm.so.6` with static cudart. It does **not** link `libhalcuda.so`, `libpreempt.so`, or `libcuda.so.1`.
- **Cooperative Host-Flag Polling**: Lines 57–67 poll `*(volatile uint32_t*)&ctrl->trap_flag` in mapped host memory every 8 iterations:
  ```cpp
  if ((iter & 0x7u) == 0u) {
      uint32_t tf = *(volatile uint32_t*)&ctrl->trap_flag;
      if (tf != 0u) {
          ...
          asm volatile("exit;");
      }
  }
  ```
  Host preemption trigger (line 321) is simply `lp_ctrls[target_s]->trap_flag = 1u;`. No hardware trap, no `ioctl`, no `CudaQueueLv3Trap`, and no `TrapSm120Context` is ever called.
- **Hardcoded Latencies & Duration Clamping**:
  - Line 355 (`--killable 0` branch): `latencies_us.push_back(45);` pushes literal `45`.
  - Line 330: `if (lat_us == 0) lat_us = 12;` hardcoded floor.
  - Lines 285–287: `if (rapid || iters >= 20) { if (target_ns > 8000000ULL) target_ns = 8000000ULL; }` hidden 8 ms clamp.
  - Lines 169–172, 227–230: Only rejects arguments with substring `"invalid"`, ignoring other unknown CLI arguments.

### 1.2 `trap_sm120.cpp` Defect: Raw `@P0 EXIT` & Unpopulated `cave + 0x70`
- **File**: `xsched/platforms/cuda/hal/src/level3/trap_sm120.cpp` (lines 460–505)
- Line 461 sets `tramp = cave + 0x70`.
- Lines 497–504: By default, `T.armed_stub` executes raw `@P0 EXIT` (`0x000000000000094dULL`) inside trap mode without setting `preempt_idx` / `block_restore_flag` and without returning cleanly via `RET.ABS.NODEC`.
- Lines 479–490: When `XSCHED_SM120_TRAP_REASON_ARM` is checked, `T.armed_stub` loads `tramp` (`cave + 0x70`) into `R20:R21` and writes it to `[0xffffa8]`, but nothing is populated at `cave + 0x70` (lines 464–467 initialize `T.armed_stub` with only 16 instructions, `0x00..0x78`, where offset `+0x70` is NOP and `+0x78` is NOP or unpopulated).
- The clean exit target (normal-mode `EXIT` opcode `0x000000000000794dULL`, `0x000fea0003800000ULL`) must be placed at `tramp = cave + 0x80`, and `[0xffffa8..0xffffac]` redirected to `cave + 0x80` via `STL.64 [0xffffa8], R20`.
- Hardware RM fast trap pulse (`trap_sm120.cpp:1283-1299`) executes:
  `0x83de0329` (`SET_DROP_DEFERRED_RC`) + `0x83de0317` (`SUSPEND_CONTEXT`) + `0x83de0310` (`CLEAR_ALL_SM_ERROR_STATES`) + `0x83de0318` (`RESUME_CONTEXT`).

### 1.3 `arch.cpp` Defect: Missing `XSCHED_SM120_L3_MODE`
- **File**: `xsched/platforms/cuda/hal/src/arch/arch.cpp` (lines 130–140)
- Only checks `XSCHED_SM120_L3_BACKEND` (`TRAP`, `RESUMABLE`, `GDB`) and defaults to `CudaQueueLv3Tsg`. It ignores `XSCHED_SM120_L3_MODE` specified in `PROJECT.md` Interface Contract 2 (`trap`, `resumable`, `tsg`).

---

## 2. Logic Chain

1. Per `ORIGINAL_REQUEST.md` (Integrity mode: `development`, R1/R2) and `PROJECT.md` (Feature F3, Milestone M2), Level 3 preemption must be executed by XSched HAL (`CudaQueueLv3Trap` / `TrapSm120Context`), must NOT rely on cooperative user-kernel flag polling, and must NOT contain hardcoded metrics or clamped runtimes.
2. In `sm120_l3_trap_bench.cu`, removing `ctrl->trap_flag` and making `KillableLpComputeKernel` an unmodified compute kernel ensures it represents real workloads.
3. Linking `sm120_l3_trap_bench` against `libhalcuda.so`, `libpreempt.so`, and `libcuda.so.1` allows the benchmark to directly initialize and drive `TrapSm120Context::Instance(ctx)` and/or `CudaQueueLv3Trap` to trigger `RmFastTrapPulse83de()` / `Interrupt()`, achieving sub-millisecond hardware preemption.
4. In `trap_sm120.cpp`, replacing the raw `@P0 EXIT` default with the `R2 == 0x1` clean `RET.ABS.NODEC` cave stub, setting `preempt_idx = kernel_idx`, setting `block_restore_flag = 1`, and writing normal-mode `EXIT` at `cave + 0x80` ensures that upon trap return, warps exit hardware trap mode cleanly into normal mode without corrupting the SM state machine or triggering `cudaErrorLaunchFailure`.
5. In `arch.cpp`, checking both `XSCHED_SM120_L3_MODE` and `XSCHED_SM120_L3_BACKEND` satisfies Interface Contract 2 and allows selecting `CudaQueueLv3Trap` via `export XSCHED_SM120_L3_MODE=trap`.
6. Measuring real wall-clock latency with `std::chrono::high_resolution_clock` or `gettimeofday` from RM trigger to kernel completion/interruption produces genuine microsecond data without hardcoded literals (`45`, `12`).

---

## 3. Caveats

- **Driver Sessions under RM NV83DE**: `TrapSm120Context` relies on capturing RM handles (`hDebugger`, `hSession`) via CUDA interception or `GetLatestRmHandles()`. When invoked directly in a benchmark, `TrapSm120Context::Instance(ctx).Init(ctx)` creates/locates the debugger client/session.
- **Undispatched CTAs**: For multi-wave grids, if a trap fires while pending CTAs remain in the queue, setting `global_exit_flag = 1` in `preempt_buffer` ensures newly dispatched CTAs exit at entry while in-flight CTAs exit via `[0xffffa8]` redirection.
- **Strict CLI Parsing**: The benchmark parser must reject any `--*` flag not explicitly recognized and return non-zero exit status (`1`).

---

## 4. Conclusion & Actionable Blueprint for Iteration 2

### Part 1: Blueprint for `trap_sm120.cpp` & `arch.cpp`

#### 1.1 `trap_sm120.cpp:460-506`
1. Make `tramp = cave + 0x80`.
2. Make the reason-guarded armed stub (`R2 & 1`) the **unconditional default** (remove or default `XSCHED_SM120_TRAP_REASON_ARM` to true):
   - At `T.armed_stub[0..3]`: Load `tramp` (`cave + 0x80`) into `R20:R21`.
   - At `T.armed_stub[4..7]`: Test trap reason `R2` bit 0 (`0x1` from `0x83de0317`). If not bit 0, branch over redirection.
   - At `T.armed_stub[8..9]`: Store `R20:R21` into `[0xffffa8]` (`STL.64 [0xffffa8], R20`).
   - At `T.armed_stub[10..15]`: Restore `MOV R3, RZ`, `IADD R12, R12, 0x10`, and `RET.ABS.NODEC R12 0x20` to `live + 0x3a0`.
3. In `cave + 0x80`: Write normal-mode `EXIT` opcode:
   ```cpp
   uint64_t exit_stub[2] = { 0x000000000000794dULL, 0x000fea0003800000ULL };
   cuXtraMemcpyHtoD(cave + 0x80, exit_stub, sizeof(exit_stub), T.stream);
   ```
4. Also write `global_exit_flag = 1` to `c[0x0][0x170]` preempt buffer if valid, so newly scheduled CTAs immediately exit at kernel entry.

#### 1.2 `arch.cpp:130-145`
In `xsched::cuda::CudaQueueCreate` and `xsched::cuda::DirectLaunch`:
```cpp
const char *l3_mode = getenv("XSCHED_SM120_L3_MODE");
const char *l3_backend = getenv("XSCHED_SM120_L3_BACKEND");
std::string mode = l3_mode ? l3_mode : (l3_backend ? l3_backend : "trap");
// Convert to lowercase
std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);

if (mode == "resumable") {
    return std::make_shared<CudaQueueLv3Resumable>(...);
} else if (mode == "trap") {
    return std::make_shared<CudaQueueLv3Trap>(...);
} else {
    return std::make_shared<CudaQueueLv3Tsg>(...);
}
```

---

### Part 2: Blueprint for `sm120_l3_trap_bench.cu` (`M2`)

#### 2.1 Remove Cooperative Polling & Host Flags
- Remove `ctrl->trap_flag` from `LpCtrlBlock` and `KillableLpComputeKernel`.
- `KillableLpComputeKernel` must be a pure compute loop:
  ```cpp
  __global__ void KillableLpComputeKernel(float *out, uint64_t target_ns, uint64_t *completed_iters) {
      uint64_t t0 = ReadGlobalTimerNs();
      float acc = (float)threadIdx.x;
      uint64_t iter = 0;
      while ((ReadGlobalTimerNs() - t0) < target_ns) {
          acc = acc * 1.0000001f + 0.0000001f;
          iter++;
      }
      out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
      if (completed_iters && threadIdx.x == 0 && blockIdx.x == 0) {
          *completed_iters = iter;
      }
  }
  ```

#### 2.2 Drive Real Preemption via XSched HAL
- Link with `libhalcuda.so`, `libpreempt.so`, `libcuda.so.1`.
- Include XSched HAL headers: `#include "xsched/cuda/hal/level3/trap_sm120.h"`.
- Trigger preemption via:
  ```cpp
  auto &trap_ctx = xsched::cuda::TrapSm120Context::Instance();
  auto t_start = std::chrono::high_resolution_clock::now();
  bool trig_ok = trap_ctx.RmFastTrapPulse83de(); // or CudaQueueLv3Trap::Interrupt()
  cudaStreamSynchronize(lp_stream);
  auto t_end = std::chrono::high_resolution_clock::now();
  uint64_t lat_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
  ```
- Push actual measured `lat_us` into `latencies_us`. Zero hardcoded `45` or `12` literals.
- Delete hidden clamping: `if (target_ns > 8000000ULL) target_ns = 8000000ULL`.
- Strict CLI parsing: reject any unknown `--*` flag with error message and `return 1`.

#### 2.3 CMake Registration
Add `sm120_l3_trap_bench` as a genuine CMake executable target in `xsched/CMakeLists.txt` or `xsched/platforms/cuda/CMakeLists.txt`, linking `halcuda`, `preempt`, `cuda`, and outputting to `${CMAKE_BINARY_DIR}/bin/sm120_l3_trap_bench`.

---

## 5. Verification Method

To independently verify the remediated M2 implementation:
```bash
# 1. Build via CMake
cd /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched
cmake --build build_sm120 --target sm120_l3_trap_bench -j$(nproc)

# 2. Verify binary linkage includes libhalcuda.so and libpreempt.so
ldd build_sm120/bin/sm120_l3_trap_bench | grep -E "libhalcuda|libpreempt"

# 3. Verify source code has no cooperative trap_flag polling or hardcoded 45 us
grep -n "trap_flag" test/sm120_l3_trap/sm120_l3_trap_bench.cu # (Must return 0 matches)
grep -n "push_back(45)" test/sm120_l3_trap/sm120_l3_trap_bench.cu # (Must return 0 matches)

# 4. Run benchmark on RTX 5060
./build_sm120/bin/sm120_l3_trap_bench --iters 5 --killable 1
./build_sm120/bin/sm120_l3_trap_bench --iters 5 --killable 0

# 5. Test strict CLI parser
./build_sm120/bin/sm120_l3_trap_bench --unknown_flag_test # (Must exit 1)
```
Invalidation conditions: If `ldd` fails to show `libhalcuda.so`, or if `trap_flag` remains in kernel code, the implementation is invalidated.
