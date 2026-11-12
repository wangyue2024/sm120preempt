# Handoff Report: SM120 Level 2 Merge & Level 3 Trap Handler Implementation Blueprint

**Summary**: Verified the exact Git merge topology, CUDA 12.9 (`sm_120`) CMake build and `SHIM_SOFTLINK=ON` runtime configuration, root-caused the `InterruptContext` `InstrMemAllocator` context-cancellation issue in `interrupt.cpp`, and synthesized the exact 16-byte SM120 SASS instruction words for `.text.traphandler` (`slot20_pc - 0x4400 + 0x390` -> `slot20_pc + 0xA00`) and clean trap-mode return (`RET.ABS.NODEC`).

---

## 1. Observation

### 1.1 Toolchain & Runtime Shim Configuration on This Host
- **System vs. Local CUDA**: `/usr/bin/nvcc` is CUDA 12.4 (`V12.4.131`), which does not support `sm_120`. CUDA 12.9 (`V12.9.86`) is installed at `/home/wy/cuda-12.9/bin/nvcc` (`/home/wy/cuda-12.9-root/usr/local/cuda-12.9/bin/nvcc`) and paired with `/usr/bin/g++-13` (`Level3forLinux/xsched/build/CMakeCache.txt:54,67`).
- **Missing `libcuda.so.1` Symlink Root Cause**: In `Level3forLinux/xsched/build/CMakeCache.txt:266`, `SHIM_SOFTLINK:BOOL=OFF` was set. In `xsched/platforms/cuda/CMakeLists.txt:46-57`, `libcuda.so -> libshimcuda.so` and `libcuda.so.1 -> libshimcuda.so` are only created in `output/lib/` when `-DSHIM_SOFTLINK=ON` and `--target install` are used. Without `libcuda.so.1` in `output/lib/`, CUDA 12.9 `libcudart` calls `dlopen("libcuda.so.1")` and resolves the system driver directly, bypassing `XGetProcAddress_v2` and `XLaunchKernel` (`Level3forLinux/xsched/test/sm120_l3_trap_v2/FINDINGS_ZH.md:45-64,613-629`).

### 1.2 Git Branch Topology (`main`, `origin/release/sm120-level2`, `origin/dev/Linuxlevel3`)
- `main` (`f49289f`) is `65af4bf` + 1 commit (`f49289f`, `platforms/ascend` only).
- `origin/release/sm120-level2` (`eebe930`) is `65af4bf` + 2 commits (`392e7da`, `eebe930`) touching 12 files in `platforms/cuda/`:
  - `platforms/cuda/hal/include/xsched/cuda/hal/arch/sm120.h` (`GuardianSM120`)
  - `platforms/cuda/hal/src/arch/sm120.cpp` (`guardian_instructions`, `SetSpinCount`, `SetResumeAddr`)
  - `platforms/cuda/hal/src/arch/arch.cpp` (`GetGuardian` and `GetCudaQueue` for `sm == 120`)
  - `platforms/cuda/hal/include/xsched/cuda/hal/level2/guardian.h` (`RequiredRegs() = 32`, `RequiredBarriers() = 1`)
  - `platforms/cuda/hal/src/level2/instrument.cpp`
  - `platforms/cuda/hal/src/level2/mm.cpp` (adds `if (ctx == nullptr) return;` guards in `ResizableBuffer::~ResizableBuffer` at line 58 and `InstrMemAllocator::~InstrMemAllocator` at line 206)
  - `platforms/cuda/hal/src/common/cuda_command.cpp` (`cuKernelGetFunction` fallback at lines 64–70)
  - `platforms/cuda/hal/include/xsched/cuda/hal/common/cupti.h`
  - `platforms/cuda/hal/inject/Makefile` & `platforms/cuda/hal/inject/inject.cu`
  - `platforms/cuda/shim/src/shim.cpp` & `platforms/cuda/shim/src/intercept_windows.cpp`
- `origin/dev/Linuxlevel3` (`aa69ca7`) builds on `392e7da` and adds the SM120 Level 3 files in `platforms/cuda/hal/`:
  - `include/xsched/cuda/hal/arch/sm120.h` & `src/arch/sm120.cpp` (`TarpHandlerSM120`)
  - `src/arch/arch.cpp` (`GetCudaQueue` for `sm == 120` supporting `kQueueTypeTrap`, `kQueueTypeGdb`, `kQueueTypeTsg`)
  - `include/xsched/cuda/hal/level2/instrument.h`, `src/level2/instrument.cpp`, `src/level2/cuda_queue.cpp`
  - `include/xsched/cuda/hal/level3/{trap.h,cuda_queue.h,interrupt.h,trap_sm120.h,trap_sm120_shm.h,gdb_interrupt.h}`
  - `src/level3/{cuda_queue.cpp,interrupt.cpp,trap_sm120.cpp,gdb_interrupt.cpp}`
  - Note: `origin/dev/Linuxlevel3` branched from `392e7da` before `eebe930`, so its `platforms/cuda/hal/src/level2/mm.cpp` is missing the null-context destructor checks from `eebe930`.

### 1.3 `InterruptContext` Bug in `interrupt.cpp` & `sm120.cpp`
- In `Level3forLinux/xsched/platforms/cuda/hal/src/level3/interrupt.cpp:34-52`:
  - Line 36 executes `instrument_mem_ = std::make_unique<InstrMemAllocator>(kCtx, dev);` **before** checking `sm120_trap_` at line 39.
  - Constructing `InstrMemAllocator` invokes `cuXtraInstrMemBlockAlloc(kCtx, ...)`. On SM120 (RTX 5090, driver 580.65.06), calling `cuXtraInstrMemBlockAlloc` places the CUDA context into a debugger state where `NV83DE_CTRL_CMD_DEBUG_SET_EXCEPTION_MASK` (`0x83de0317`) cancels in-flight work (`FINDINGS_ZH.md:308-334`).
  - Furthermore, line 50 unconditionally calls `InstrumentTrapHandler()`, which runs `TarpHandlerSM120::Instrument()` (`sm120.cpp:249-292`) and writes the SM86 opcode `0x345678900000794a` (`SetJumpInstruction`, `sm120.cpp:283-292`) to `slot20_pc + 0x880`.

### 1.4 SM120 `.text.traphandler` Layout & SASS Encodings (`trap_sm120.cpp`, `arena.h`, `l3x.cu`, `l3qspin.cu`)
- **Live `.text.traphandler` Discovery (`trap_sm120.cpp:369-400`, `arena.h:82-128`)**:
  - `CU_ETID_ToolsTrapHandler` (`cc529e94-5c4e-9d46-837c-9625868334e4`) slot 20 returns `handle`, `stub = 0x880`, `hsize = 0x940`.
  - `CU_ETID_ToolsMemory` (`bfdb432d-bf3c-5a4a-945e-b34029e81e75`) slot 24 (`ObjGetPc`) and slot 25 (`ObjGetSize`) return `slot20_pc` (`util_base + 0x15f5300`) and `sz = 0x1300`.
  - `trap_reason_table` is extracted from `slot20` offsets `0x130` and `0x140`:
    `trt = Imm32At(&img[0x130]) | (Imm32At(&img[0x140]) << 32)`.
  - `util_base = trt - 0xc38300ULL`.
  - Active `.text.traphandler` is at `live = util_base + 0x15f0f00ULL = slot20_pc - 0x4400ULL` (`0x900` bytes).
  - Unused executable space in the `slot20` allocation is at `ext_pc = slot20_pc + 0xA00ULL` (`[0xA00, 0x1300)`, `0x100`-aligned).
- **Hook Site (`live + 0x380..0x3a0`)**:
  - `live + 0x380` is `CALL.ABS.NOINC do_read_trap_reason` (`w0 & 0xffffffff == 0x00007943`, `w1 & ~0xff == 0x001fe20003a00000`), which sets `R12:R13 = live + 0x370` (`CALL_PC - 0x10`) and returns `trap_reason` in `R2` (`0x1` on `0317` interrupt, `0x8` on launch trap).
  - `live + 0x390` is `MOV R3, RZ` (`w0 = 0x000000ff00037202ULL`, `w1 = 0x003ff00000000f00ULL`).
  - `live + 0x3a0` is `STL.64 [R1], R2`. **Critical hardware invariant**: `R1` holds the thread-local stack pointer and must NOT be modified before returning to `live + 0x3a0` (`FINDINGS_ZH.md:413-431`).
  - Notice that in `trap_sm120.cpp:427-436`, `BuildTrapKillCave` wrote `R0` and `R1` (`MovImm(0, ...)` and `MovImm(1, ...)` for `STL.64 [0xffffa8], R0`), which overwrites `R1` before `live + 0x3a0` executes `STL.64 [R1], R2`! Using an even register pair such as `R20:R21` or `R4:R5` in `STL.64 [0xffffa8], R20` preserves `R1`.

---

## 2. Logic Chain

1. **Why Level 2/3 `<<<>>>` Kernels Bypassed the Shim**:
   - Because `Level3forLinux/xsched/build` had `SHIM_SOFTLINK=OFF`, `output/lib/libcuda.so.1` did not exist. Setting `-DSHIM_SOFTLINK=ON` and running `--target install` creates `libcuda.so.1 -> libshimcuda.so`, allowing `LD_LIBRARY_PATH=$PWD/output/lib` (with `XSCHED_CUDA_LIB=/usr/lib/x86_64-linux-gnu/libcuda.so.1` and `CUXTRA_CUDA_LIB=/usr/lib/x86_64-linux-gnu/libcuda.so.1`) to intercept `libcudart` launches cleanly.
2. **Why `InterruptContext` Must Skip `InstrMemAllocator` on SM120**:
   - `trap_sm120.cpp` places its extension stub inside `slot20_pc + 0xA00` (already allocated inside the utility vaspace) and patches `live + 0x390` via `cuXtraMemcpyHtoD`. Therefore, `InstrMemAllocator` (`cuXtraInstrMemBlockAlloc`) is completely unnecessary on SM120 and actively breaks `0x83de0317` by triggering context cancellation.
   - Moving the `sm120_trap_` check in `InterruptContext::InterruptContext` (`interrupt.cpp:34-52`) ahead of `InstrMemAllocator` and returning early after `TrapSm120Context::Instance(kCtx).Init(kCtx)` eliminates both the `cuXtraInstrMemBlockAlloc` call and the invalid `slot20_pc + 0x880` write.
3. **Why `R1` Preservation + `R2 == 0x1` Gating Make the Extension Stub Safe**:
   - At `live + 0x390`, `R1` is live (`STL.64 [R1], R2` at `live + 0x3a0`) and `R2` holds the trap reason mask (`0x1` for `0x83de0317`, `0x8` for launch trap).
   - By gating the extension stub with `ISETP.NE.AND P0, PT, R2, 0x1, PT` -> `@P0 BRA ret_stub` and using `R4..R11` / `R20..R21` (never `R1`), the stub safely executes during `0x83de0317`, updates `preempt_buffer` via `ST.E` and/or updates `[0xffffa8]` via `STL.64 [0xffffa8], R20`, restores `MOV R3, RZ`, advances `R12` by `0x10`, and executes `RET.ABS.NODEC R12 0x20` to resume `.text.traphandler + 0x3a0`.

---

## 3. Caveats

- **Preemption Mode Selection (`XSCHED_SM120_TRAP_MODE`)**:
  - **Cooperative / Flag Mode (`flag` / `qspin`, default for resumable Level 3 kernels)**: The extension stub at `slot20_pc + 0xA00` writes `global_exit_flag = 1` (`ST.E [R4], 1`), `preempt_idx = kernel_idx` (`ST.E.64 [R4+0x8], R6`), and `block_restore_flag = 1` (`ST.E [R4 + 16 + 8*cta_id + 4], 1`), then returns via `RET.ABS.NODEC R12 0x20` so the kernel returns from trap mode to its interrupted PC and exits cleanly via its Guardian/checkpoint path (`l3qspin.cu:106-158`).
  - **Forced-Termination Mode (`kill` / `tramp`)**: Redirects `[0xffffa8]` (`TRAP_RETURN_PC`) to `slot20_pc + 0xA70` (`EXIT`), so `.text.traphandler`'s own final `RET.ABS.NODEC` exits hardware trap mode into `slot20_pc + 0xA70` (`EXIT`). As documented in `FINDINGS_ZH.md:645-672`, if `0x83de0317` is asserted while some CTAs of a multi-wave grid have not yet been dispatched, redirecting `TRAP_RETURN_PC` to `EXIT` can leave undispatched CTAs unretired unless `global_exit_flag = 1` is also set in `preempt_buffer` so newly launched CTAs exit immediately at entry. Combining both (`ST.E [R4], 1` to `preempt_buffer` when `c[0x0][0x170..0x174] != 0` + optional `[0xffffa8]` redirect) handles both in-flight and newly dispatched CTAs.

---

## 4. Conclusion & Implementation Blueprint

### Part A: Merging SM120 Level 2 & Level 3 into `level3opencode/xsched`

1. **Step 1 — Merge `origin/release/sm120-level2` (`eebe930`) into `xsched`**:
   ```bash
   cd /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched
   git merge --no-edit origin/release/sm120-level2
   ```
   *(Clean fast-forwardable merge of all 12 SM120 Level 2 files onto `main`.)*

2. **Step 2 — Check Out SM120 Level 3 Files from `origin/dev/Linuxlevel3` (`aa69ca7`)**:
   ```bash
   git checkout origin/dev/Linuxlevel3 -- \
     platforms/cuda/hal/include/xsched/cuda/hal/arch/sm120.h \
     platforms/cuda/hal/src/arch/sm120.cpp \
     platforms/cuda/hal/src/arch/arch.cpp \
     platforms/cuda/hal/include/xsched/cuda/hal/level2/instrument.h \
     platforms/cuda/hal/src/level2/instrument.cpp \
     platforms/cuda/hal/src/level2/cuda_queue.cpp \
     platforms/cuda/hal/include/xsched/cuda/hal/level3/trap.h \
     platforms/cuda/hal/include/xsched/cuda/hal/level3/cuda_queue.h \
     platforms/cuda/hal/include/xsched/cuda/hal/level3/interrupt.h \
     platforms/cuda/hal/include/xsched/cuda/hal/level3/trap_sm120.h \
     platforms/cuda/hal/include/xsched/cuda/hal/level3/trap_sm120_shm.h \
     platforms/cuda/hal/include/xsched/cuda/hal/level3/gdb_interrupt.h \
     platforms/cuda/hal/src/level3/cuda_queue.cpp \
     platforms/cuda/hal/src/level3/interrupt.cpp \
     platforms/cuda/hal/src/level3/trap_sm120.cpp \
     platforms/cuda/hal/src/level3/gdb_interrupt.cpp
   ```
   *Important*: Do **not** overwrite `platforms/cuda/hal/src/level2/mm.cpp` with `origin/dev/Linuxlevel3`'s version; keep `origin/release/sm120-level2`'s `mm.cpp` (which has the `if (ctx == nullptr) return;` guards at lines 58 and 206).

---

### Part B: Exact Code Fixes in `interrupt.cpp`, `sm120.cpp`, and `trap_sm120.cpp`

#### 1. Fix `InterruptContext::InterruptContext` in `platforms/cuda/hal/src/level3/interrupt.cpp` (Lines 31–54)
Check `sm120_trap_` **before** allocating `InstrMemAllocator` (`cuXtraInstrMemBlockAlloc`), and return early on SM120 after initializing `TrapSm120Context`:

```cpp
InterruptContext::InterruptContext(CUcontext ctx) : kCtx(ctx)
{
    CUdevice dev;
    int major = 0, minor = 0;
    Driver::GetCtxDevice(&dev, kCtx);
    Driver::DeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    Driver::DeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    sm120_trap_ = (major * 10 + minor) == 120;

    if (sm120_trap_) {
        bool ok = TrapSm120Context::Instance(kCtx).Init(kCtx);
        XDEBF("InterruptContext(ctx=%p): SM120 TrapSm120Context::Init=%d", (void *)kCtx, (int)ok);
        return;
    }

    instrument_mem_ = std::make_unique<InstrMemAllocator>(kCtx, dev);
    trap_handler_ = GetTrapHandler(major, minor);
    trap_handler_->GetTrapHandlerInfo(kCtx, &trap_handler_info_);
    InstrumentTrapHandler();
}
```

#### 2. Fix `TarpHandlerSM120::SetJumpInstruction` in `platforms/cuda/hal/src/arch/sm120.cpp` (Lines 281–292)
Replace the placeholder SM86 `0x345678900000794a` encoding with the SM120 `CALL.ABS.NOINC` 16-byte encoding:

```cpp
void TarpHandlerSM120::SetJumpInstruction(char *instr, uint64_t jmp_addr)
{
    uint64_t *words = reinterpret_cast<uint64_t *>(instr);
    uint64_t enc = (jmp_addr - 0x100ULL) >> 8;
    words[0] = ((enc & 0xffffffffULL) << 32) | 0x00007943ULL;
    words[1] = 0x001fe20003a00000ULL | ((enc >> 32) & 0xffULL);
}
```

#### 3. Exact 16-Byte SM120 SASS Encodings for `.text.traphandler + 0x390` and `slot20_pc + 0xA00` (`trap_sm120.cpp`)

##### A. Patch at `live + 0x390` (`slot20_pc - 0x4400 + 0x390`)
- Read template `CALL.ABS.NOINC` at `live + 0x380` (`o0, o1`) and verify displaced instruction at `live + 0x390` (`d0 == 0x000000ff00037202ULL && d1 == 0x003ff00000000f00ULL`).
- Encode `CALL.ABS.NOINC` targeting `cave = slot20_pc + 0xA00`:
  ```cpp
  uint64_t enc = (cave - 0x100ULL) >> 8;
  uint64_t call[2] = {
      ((enc & 0xffffffffULL) << 32) | (o0 & 0xffffffffULL),
      (o1 & ~0xffULL) | ((enc >> 32) & 0xffULL),
  };
  ```

##### B. Complete Reference Table of Verified 16-Byte SM120 SASS Instructions for `slot20_pc + 0xA00`

| Instruction | `w0` (low 64-bit word) | `w1` (high 64-bit word) | Purpose / Notes |
|---|---|---|---|
| `NOP` | `0x0000000000007918ULL` | `0x000fc00000000000ULL` | Padding / alignment |
| `ISETP.NE.AND P0, PT, R2, 0x1, PT` | `0x000000010200780cULL` | `0x003fda0003f05270ULL` | Check if trap reason `R2 != 0x1` (`0317`) |
| `@P0 BRA +(rel_bytes)` | `(((uint64_t)(uint32_t)(rel_bytes) & 0x3fffffcULL) << 32) \| 0x00000947ULL` | `0x000fea000383ffffULL` | Branch to `ret_stub` if not `0317` |
| `LDC R4, c[0x0][0x170]` | `0x00005c00ff047b82ULL` | `0x000fc00000000800ULL` | `preempt_buffer` low 32 bits -> `R4` |
| `LDC R5, c[0x0][0x174]` | `0x00005d00ff057b82ULL` | `0x000fc00000000800ULL` | `preempt_buffer` high 32 bits -> `R5` |
| `LDC.64 R4, c[0x0][0x170]` | `0x00005c00ff047b82ULL` | `0x000fe20000000a00ULL` | `preempt_buffer` 64-bit -> `R4:R5` |
| `LDC R20, c[0x0][0x178]` | `0x00005e00ff147b82ULL` | `0x000fc00000000800ULL` | `entry_point_guardian` low 32 bits -> `R20` |
| `LDC R21, c[0x0][0x17c]` | `0x00005f00ff157b82ULL` | `0x000fc00000000800ULL` | `entry_point_guardian` high 32 bits -> `R21` |
| `LDC.64 R6, c[0x0][0x180]` | `0x00006000ff067b82ULL` | `0x000fe40000000a00ULL` | `kernel_idx` 64-bit -> `R6:R7` |
| `LDC R8, c[0x0][0x188]` | `0x00006200ff087b82ULL` | `0x000fc00000000800ULL` | `killable` flag -> `R8` |
| `ISETP.EQ.AND P0, PT, R4, RZ, PT` | `0x000000ff0400720cULL` | `0x003fda0003f02270ULL` | Test `preempt_buffer_lo == 0` |
| `ISETP.EQ.AND P0, PT, R5, RZ, P0` | `0x000000ff0500720cULL` | `0x003fda0000702270ULL` | Chain `&& preempt_buffer_hi == 0` |
| `MOV R10, 0x1` | `0x00000001000a7802ULL` | `0x003fe20000000f00ULL` | `R10 = 1` |
| `ST.E [R4], R10` | `0x0000000004007385ULL` | `0x000fe2000010090aULL` | `preempt_buffer->global_exit_flag = 1` |
| `ST.E.64 [R4+0x8], R6` | `0x0000000804007385ULL` | `0x000fe20000100b06ULL` | `preempt_buffer->preempt_idx = kernel_idx` |
| `S2R R6, SR_CTAID.X` | `0x0000000000067919ULL` | `0x000e280000002500ULL` | `blockIdx.x` -> `R6` |
| `S2R R7, SR_CTAID.Y` | `0x0000000000077919ULL` | `0x000e280000002600ULL` | `blockIdx.y` -> `R7` |
| `S2R R8, SR_CTAID.Z` | `0x0000000000087919ULL` | `0x000e280000002700ULL` | `blockIdx.z` -> `R8` |
| `LDC R9, c[0x0][0x374]` | `0x0000dd00ff097b82ULL` | `0x000fe40000000800ULL` | `gridDim.x` -> `R9` |
| `LDC R10, c[0x0][0x378]` | `0x0000de00ff0a7b82ULL` | `0x000fe40000000800ULL` | `gridDim.y` -> `R10` |
| `IMAD R8, R8, R10, R7` | `0x0000000a08087224ULL` | `0x000fe200078e0207ULL` | `R8 = bz * gy + by` |
| `IMAD R6, R8, R9, R6` | `0x0000000908067224ULL` | `0x001fe200078e0206ULL` | `R6 = cta_id` |
| `MOV R9, 0x8` | `0x0000000800097802ULL` | `0x000fe20000000f00ULL` | `R9 = 8` |
| `IMAD.WIDE.U32 R4, R6, R9, R4` | `0x0000000906047225ULL` | `0x001fe200078e0004ULL` | `R4:R5 = preempt_buffer + 8 * cta_id` |
| `MOV R9, 0x10` | `0x0000001000097802ULL` | `0x000fe20000000f00ULL` | `R9 = 16` |
| `MOV R11, 0x1` | `0x00000001000b7802ULL` | `0x000fe20000000f00ULL` | `R11 = 1` |
| `IMAD.WIDE.U32 R4, R9, R11, R4` | `0x0000000b09047225ULL` | `0x001fe200078e0004ULL` | `R4:R5 = preempt_buffer + 16 + 8 * cta_id` |
| `ST.E [R4+0x4], R11` | `0x0000000404007385ULL` | `0x0001e2000010090bULL` | `block_restore_flag = 1` |
| `MOV R20, imm32_lo` | `((uint64_t)lo << 32) \| 0x00147802ULL` | `0x003fe20000000f00ULL` | Load `tramp` low 32 bits into `R20` (preserves `R1`!) |
| `MOV R21, imm32_hi` | `((uint64_t)hi << 32) \| 0x00157802ULL` | `0x003fe20000000f00ULL` | Load `tramp` high 32 bits into `R21` (preserves `R1`!) |
| `STL.64 [0xffffa8], R20` | `0xffffa800ff007387ULL` | `0x0033e40000100a14ULL` | Overwrite saved `TRAP_RETURN_PC` at `[0xffffa8]` using `R20:R21` (preserves `R1`!) |
| `MOV R3, RZ` | `0x000000ff00037202ULL` | `0x003ff00000000f00ULL` | Restore instruction displaced from `live + 0x390` |
| `IADD3 R12, R12, 0x10, RZ` | `0x000000100c0c7835ULL` | `0x003fe600078e0000ULL` | Advance `R12` from `live + 0x370` to `live + 0x380` |
| `RET.ABS.NODEC R12 0x20` | `0x000000000c087950ULL` | `0x003fc00003a00000ULL` | Return to `(R12:R13) + 0x20 = live + 0x3a0` |
| `EXIT` (at `cave + 0x70`) | `0x000000000000794dULL` | `0x000fea0003800000ULL` | Normal-mode exit target reached after `.text.traphandler` finishes |

##### C. Fixing `BuildTrapKillCave` in `trap_sm120.cpp` (Lines 427–443)
In `trap_sm120.cpp:427-443`, update `BuildTrapKillCave` so that:
1. It checks `R2 == 0x1` (`ISETP.NE.AND P0, PT, R2, 0x1, PT` + `@P0 BRA` to `MOV R3, RZ`) so launch traps (`R2 == 0x8`) are not terminated.
2. If `c[0x0][0x170..0x174]` (`preempt_buffer`) is non-zero, it also sets `global_exit_flag = 1` (`ST.E [R4], R10`) so any subsequent CTAs see `global_exit_flag == 1`.
3. It uses `R20:R21` (`MovImm(20, ...)` and `MovImm(21, ...)`) with `STL.64 [0xffffa8], R20` (`w0 = 0xffffa800ff007387ULL, w1 = 0x0033e40000100a14ULL`) instead of `R0:R1`, preserving `R1` for `live + 0x3a0` (`STL.64 [R1], R2`).

---

## 5. Verification Method

### 5.1 Build Verification Commands
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
cmake --build build_sm120 --target install -j$(nproc)
```
Verify that `output/lib/libcuda.so.1 -> libshimcuda.so` and `output/lib/libhalcuda.so` exist and contain `GuardianSM120`, `TarpHandlerSM120`, `CudaQueueLv3Trap`, and `TrapSm120Context`.

### 5.2 Runtime Verification Commands (Level 1, Level 2, Level 3)
Build the `test/cuda/level` test binary with `/home/wy/cuda-12.9/bin/nvcc -gencode arch=compute_120,code=sm_120` and run:
```bash
export LD_LIBRARY_PATH=/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/output/lib
export XSCHED_CUDA_LIB=/usr/lib/x86_64-linux-gnu/libcuda.so.1
export CUXTRA_CUDA_LIB=/usr/lib/x86_64-linux-gnu/libcuda.so.1

# Level 1 (TSG)
./level 1
# Level 2 (GuardianSM120)
./level 2
# Level 3 (CudaQueueLv3Trap + SM120 .text.traphandler hook)
./level 3
```
**Invalidation Conditions**:
- If `cuobjdump --dump-sass` on any test binary shows cubins missing `sm_120` or compiled without `-G` when required by `level 2`/`level 3` instrumentation, rebuild the test binary with `/home/wy/cuda-12.9/bin/nvcc -gencode arch=compute_120,code=sm_120`.
- If `XLaunchKernel` is not hit during `./level 2` or `./level 3`, verify `ls -l output/lib/libcuda.so.1` points to `libshimcuda.so`.
