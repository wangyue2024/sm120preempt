# Critical Evaluation of Prior SM120 Level 3 Trap Preemption Research (`Level3forLinux` & `sm120_l3_trap_v2/compendium`)

**Executive Summary**: Prior research in `Level3forLinux` (`sm120_l3_probe` v1 and `sm120_l3_trap_v2` v2) successfully reverse-engineered the SM120 (RTX 5060 Blackwell) `.text.traphandler` layout (`slot20 - 0x4400`), SM120 SASS `CALL.ABS`/`RET.ABS` encodings, and the `NV83DE` (`0x83de0317`) benign trap entry. However, **`sm120_l3_trap_v2/compendium/` (frozen 2026-10-26) is built around a false reverse-engineering thesis**—that `pGpu+0x911 == 0` is a "boot-time debug gate" blocking `0x20800122` (`GPU_EXEC_REG_OPS`) and `0x83de031d` (`DebugExecRegOps`)—which was caused by two bugs in their own probe (`regops.cpp`) and a misread `je` branch in `nvidia.ko` (`pGpu+0x911` is actually `IS_GSP_CLIENT(pGpu) == 1`, as later admitted in `report/07-源码复核-TRAP触发根因.md` on 2026-10-27). Furthermore, `compendium/` prematurely abandoned a working **in-band trap-assisted L2 flag exit (`6–89 ms`, `cudaSuccess`, zero context corruption)** demonstrated in `FINDINGS_ZH.md` §26 (v1.8) because later HAL integration introduced `cuXtraInstrMemBlockAlloc` (which causes driver context cancellation under `0317{1,0}`) and raw `EXIT` inside the trap handler.

---

## 1. Observation

All file paths below are relative to `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/Level3forLinux` unless an absolute path is given.

### 1.1 Repository & Compendium Inventory
- **Phase 1 (`sm120_l3_probe`, v1)**:
  - `xsched/test/sm120_l3_probe/L3_FINAL_SUMMARY_ZH.md` (lines 1–289): Documents initial SM120 export table discovery (`CU_ETID_ToolsTrapHandler` slot 20), SASS patching at `slot20 + 0x880`, and `cuXtraTriggerTrap` (`0x419e84`) tests.
- **Phase 2 (`sm120_l3_trap_v2`, v2)**:
  - `xsched/test/sm120_l3_trap_v2/compendium/` (frozen 2026-10-26):
    - `README.md` (lines 1–88), `01-研究过程.md` (lines 1–133), `02-路径与方法.md` (lines 1–171), `03-机制与逆向.md` (lines 1–280), `04-通道矩阵.md` (lines 1–150), `05-工具与复现.md` (lines 1–156), `06-结论与下一步.md` (lines 1–145), `07-A-B-完整实测.md` (lines 1–244).
  - `xsched/test/sm120_l3_trap_v2/report/` (updated through 2026-10-27):
    - `00-总索引与结论全表.md`, `02-中断入口.md`, `03-核心结论.md`, `04-替代形态.md`, `05-下一步论点.md`, `06-问答.md`, **`07-源码复核-TRAP触发根因.md` (lines 1–239)**, `10-开放问题与假设清单.md`, `kb/ABI.md`, `kb/Methods.md`.
  - `xsched/test/sm120_l3_trap_v2/FINDINGS_ZH.md` (lines 1–3386, v0.1 through §55).
  - `xsched/test/sm120_l3_trap_v2/RM_REVERSE_ZH.md` (lines 1–331) and `L3_TRAP_SM120_REPORT_ZH.md` (lines 1–371).
- **HAL Implementation & Probes**:
  - `xsched/platforms/cuda/hal/src/level3/trap_sm120.cpp` (lines 1–1562): `CudaQueueLv3Trap` implementation for SM120.
  - `xsched/platforms/cuda/hal/src/arch/sm120.cpp` (lines 1–387): `TarpHandlerSM120` implementation.
  - `xsched/platforms/cuda/hal/src/arch/arch.cpp` (lines 1–185): `Get TrapHandler` factory (`sm_major == 12 && sm_minor == 0` -> `TarpHandlerSM120`).
  - `xsched/test/sm120_l3_trap_v2/probe/regops.cpp` (lines 1–466), `probe/victim_spin.cu` (lines 1–91), `tools/trigger_root_test.sh` (lines 1–52), `tools/local_root_battery.sh` (lines 1–74), `tools/nvidia_dbg/nvidia_dbg.c` (lines 1–61).
- **Kernel Driver Source (`595.99.02`, commit `e394a404`)**:
  - `refs/open-gpu-kernel-modules/src/nvidia/src/kernel/gpu/subdevice/subdevice_ctrl_gpu_regops.c` (lines 58–260).
  - `refs/open-gpu-kernel-modules/src/nvidia/src/kernel/gpu/gpu_access.c` (lines 1176–1205).
  - `refs/open-gpu-kernel-modules/src/nvidia/src/kernel/gpu/gr/kernel_sm_debugger_session_ctrl.c` (lines 629–662).

### 1.2 Exact Methods Attempted to Trigger a Trap on SM120 (RTX 5060)

Across `sm120_l3_probe` (v1) and `sm120_l3_trap_v2` (v2), **16 distinct trigger mechanisms** across 5 families were attempted:

| # | Family | Method / Command / Entry | Target / Mechanism | Observed Result & Source Evidence |
|---|---|---|---|---|
| 1 | **v1 Export Table + `cuXtraTriggerTrap`** | `CU_ETID_ToolsTrapHandler` slot 20 (`GetInfoBlackwell`) + patch `slot20 + 0x880` with SM86 48-bit `JMP` + `cuXtraTriggerTrap(ctx)` | Write `0x419e84` bit 31 via `0x20800122`; jump from `slot20 + 0x880` to 40-instr `trap_inject_instrs` | **Failed (`0%` trap hits)**: kernel ran to natural timeout (`3000 ms`). (`sm120_l3_probe/L3_FINAL_SUMMARY_ZH.md:94-169`, `sm120.cpp:281-360`) |
| 2 | **RM Subdevice RegOps (Non-root, buggy struct)** | `NV2080_CTRL_CMD_GPU_EXEC_REG_OPS` (`0x20800122`) in early `probe/regops.cpp` | Passed `hClientTarget != 0, hChannelTarget == 0`; `RegOp32` had `regOffset` at `+10` instead of `+12` | **Failed (`0x3a` / `0x1f`, `regStatus=0x04`)**: Rejected by `(hClientTarget==0)!=(hChannelTarget==0)` (`subdevice_ctrl_gpu_regops.c:143`) or read `regOffset=0`. (`report/07-源码复核-TRAP触发根因.md:62-97`) |
| 3 | **RM Subdevice RegOps (Non-root, fixed struct)** | `0x20800122` with corrected 56-byte `NV2080_CTRL_GPU_REG_OP` (`regOffset` at `+12`) | Read `0x9400` (`PTIMER`), `0x200510`, `0x420000`; Read/Write `0x419e84` (`SM_DBGR_CONTROL0`) | `0x9400` (`0xeba36080` -> `0xf4512960`) and `0x420000` (`0x87fc029f`) **succeeded (`status=0`)**; `0x419e84` rejected with **`0x1f` (`NV_ERR_INVALID_ARGUMENT`)** by `gpuValidateRegOffset_IMPL`. (`report/07:108-140`, `FINDINGS_ZH.md:3350-3368`) |
| 4 | **RM Subdevice RegOps (Root / `CAP_SYS_ADMIN`, fixed struct)** | `sudo` + `0x20800122` on `0x419e84` (bits `0, 1, 2, 31, 0xffffffff`), `0x419e00..0x419ea4`, `0x420000`, `0x4208b0`, `0x424684` | `osIsAdministrator()` bypasses user register access map (`gpu_access.c:1195`); RPC sent to GSP-RM | `status=0, regStatus=0`, but reading `0x419e84` returned **`0xbadf1100`** (PRI error poison) and writing bit 31 produced **no trap** on `victim_spin`. (`report/07:142-167`, `FINDINGS_ZH.md:3374-3385`) |
| 5 | **Direct PCI BAR0 MMIO (`root`)** | `probe/bar0poke` (`mmap` `/sys/bus/pci/devices/0000:01:00.0/resource0`) | Direct CPU MMIO read/write to BAR0 `0x0`, `0x9400`, `0x419e84` (`0x80000000` then `0x0`) | `0x0` (`PMC_BOOT_0 = 0x1b2000a1`) and `0x9400` (`PTIMER`) readable; `0x419e84` read **`0xbadf1100`**; writing `0x80000000` did not stall `victim_spin`. (`report/07:176-184`, `tools/local_root_battery.sh:43-66`) |
| 6 | **Pushbuffer In-Band Method Write** | `ToolsPushBufferHal` (`38477145-...` slot `0x12`) and internal `libcuda+0x4551e0` / `libcuda+0x18ecd0` | Submits pushbuffer methods `0x20032d00` + `0x20012144` targeting `0x419e84` on user compute channel | Pushbuffer written (`+24B`), but `cuCtxSynchronize()` failed with **`CUDA_ERROR_ILLEGAL_INSTRUCTION (715)`** or **`CUDA_ERROR_ILLEGAL_ADDRESS (700)`**. (`probe/regops.cpp:365-434`, `report/07:169-175`) |
| 7 | **RM Debugger Session RegOps (`0x83de031d`)** | `NV83DE_CTRL_CMD_DEBUG_EXEC_REG_OPS` (`0x83de031d`) on `GT200_DEBUGGER` (`0x83de`) handle | Write `0x419e84` through `ksmdbgssnCtrlCmdDebugExecRegOps_IMPL` (`kernel_sm_debugger_session_ctrl.c:629`) | Tested **only with the buggy `RegOp32` struct** (returning `0x1f`); **never re-tested as root after fixing `regOffset` at `+12`**. (`compendium/04-通道矩阵.md:29`, `report/07:204-213`) |
| 8 | **RM Preemption Mode Control (`0x20801201`)** | `NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE` (`0x20801201`) | Request CTA/CILP/WFI context-switch preemption on SM120 compute context | Returned **`0x1f`** (`NV_ERR_INVALID_ARGUMENT`) across tested parameter permutations. (`compendium/04-通道矩阵.md:30-31`) |
| 9 | **RM Channel / TSG Preempt & Reset (`0x906f0105`, `0xa06c0105`, `0x2080110b`)** | `FIFO_PREEMPT` (`0x906f0105` / `0xa06c0105`), `DISABLE_CHANNELS` (`0x20801106`), `RESET_CHANNEL` (`0x2080110b`) | Hardware runlist/TSG timeslice yield (`PREEMPT`) or forced channel reset (`RESET_CHANNEL`) | `FIFO_PREEMPT` succeeds (`status=0`, `70 µs`) as a **TSG timeslice yield** (does not enter `.text.traphandler` or evict a single-context spin kernel); `RESET_CHANNEL` kills the context (`err=700/715`). (`compendium/04-通道矩阵.md:54-68`, `FINDINGS_ZH.md:3092-3228`) |
| 10 | **In-Process `NV83DE` (`0x83de`) Allocation** | `NvRmAlloc(hClient, hChannel/hObject, hDbg, 0x83de, &params)` inside the same process | Allocate `GT200_DEBUGGER` session without external debugger handshake | Returned **`0x56` (`NV_ERR_NOT_SUPPORTED`)** or **`0x1f`** unless the channel has `0xd04` (`ToolsDebugger` slot 1) enabled and internal GR object parent. (`compendium/04-通道矩阵.md:32-34`, `FINDINGS_ZH.md:835-910`) |
| 11 | **External `cuda-gdb` + `NV83DE` (`0x83de0317{1,0}`) Suspend** | Target calls `CU_ETID_ToolsDebugger` slot 1 (`+0xd04 = 1`); helper `cuda-gdb` attaches and sends `0x83de0317{1,0}` | Enters `.text.traphandler` (`base + 0x15f0f00`) with `reason = 1` (`BPT_TRAP`) | **Trap fires and freezes GPU in `116 ms` (`65 ms` post-warm)**. Cave at `slot20 + 0xA00` executes! (`compendium/02-路径与方法.md:35-82`, `07-A-B-完整实测.md:19-46`) |
| 12 | **External `NV83DE` (`0x83de0317{0,0}`) Benign Pulse + Cave L2 Flag Write (v1.8)** | Helper steals `(fd, hClient, hDebugger)` from `cuda-gdb`, sends `0x83de0317{0,0}` to plain context (NO `cuXtraInstrMemBlockAlloc`); cave executes `LDC R4, c[0x0][0x170]; ST.E [R4], R10; RET.ABS` | Cave writes `1` to XSched's `global_exit_flag` (`c[0x0][0x170]`) and returns cleanly from trap handler; L2 guardian loop reads flag and exits | **Succeeded (`cudaSuccess`, `6–89 ms` kernel exit, zero context corruption, multi-round sustainable)**! (`FINDINGS_ZH.md:1389-1549`, `evidence/live/exp_guard_devflag_kill.log`) |
| 13 | **External `NV83DE` (`0x83de0317{1,0}`) + `CudaQueueLv3Trap` (`cuXtraInstrMemBlockAlloc` / Raw `EXIT`)** | `CudaQueueLv3Trap::InterruptContext` allocates `InstrMemAllocator` (`cuXtraInstrMemBlockAlloc`) or patches raw `EXIT` into cave, then triggers `0317` | Attempted pure L3 trap exit without L2 guardian loop | **Failed / Context Killed**: With `cuXtraInstrMemBlockAlloc` + `0317{1,0}`, driver cancels context in `27 ms` (`cudaErrorContextIsDestroyed`); with raw `EXIT` inside `0317` trap handler, warp exits without completing trap return state machine, leaving context dirty (`err=715`). (`FINDINGS_ZH.md:1456-1549`, `3288-3331`, `trap_sm120.cpp:1244-1255`) |
| 14 | **In-Kernel SASS `BPT.TRAP` / `BPT.PAUSE`** | `asm volatile("bpt.trap 0x1;")` (`0x000000010000739a`) or `bpt.pause` | Software breakpoint instruction inside kernel | Enters `.text.traphandler`; with `0xd04=1` and `RET.ABS` at `+0x390`, returns cleanly in `154 µs` (4-warp test). (`FINDINGS_ZH.md:451-540`, `compendium/04-通道矩阵.md:92-104`) |
| 15 | **CUPTI PC Sampling / Profiling Interrupts** | `cuptiPcSamplingEnable` / `cuptiActivityEnable` | Hardware PMU / PC sampling periodic interrupt | Does **not** enter `.text.traphandler` (`sentinel` unchanged). (`compendium/04-通道矩阵.md:70-75`) |
| 16 | **CUDA Green Contexts (`cuGreenCtxCreate`)** | `cuGreenCtxCreate` SM partitioning (`6/28 SMs` vs `22/28 SMs`) | Spatial SM isolation instead of temporal preemption | **Succeeded (`0.095 ms` H-task launch latency)** while L-task spins on remaining SMs. (`compendium/07-A-B-完整实测.md:107-118`) |

---

### 1.3 Verified Hardware & Driver Facts on SM120 (`595.99.02`)

1. **Dual Trap Image Layout in Utility Vaspace (`FINDINGS_ZH.md:541-620`, `compendium/03-机制与逆向.md:11-57`, `trap_sm120.cpp:414-465`)**:
   - `cuGetExportTable(CU_ETID_ToolsTrapHandler)` slot 20 (`GetInfoBlackwell`) does **not** return the live `.text.traphandler` entry! It returns the relocated `.text.do_pause` sub-image at `util_base + 0x15f5300` (with `+0x10 = memObjHandle`, `+0x18 = 0x880`, `+0x20 = 0x940`).
   - The **live `.text.traphandler` entry** is located at **`slot20_pc - 0x4400`** (`util_base + 0x15f0f00`, size `0x7c0`).
   - When a trap with `reason == 1` (`BPT_TRAP` / `0x83de0317`) arrives, hardware enters `live_base + 0x000`, dispatches `reason == 1` to `live_base + 0x390` (`0x15f1290`), and returns via `live_base + 0x5e0` (`IADD R12, R12, 0x10; RET.ABS.NODEC R12 0x20`).
   - The region `slot20_pc + 0xA00` (`live_base + 0x4e00`) is a zero-filled executable code cave inside the same 2MB utility page (`0x1400000..0x1600000`), reachable by a single 16B `CALL.ABS.NOINC` instruction (`lo = 0x0000000000007943 | ((target & 0x3ffff) << 46)`, `hi = 0x000fea0003800000 | (target >> 18)`) patched at `live_base + 0x390` (`trap_sm120.cpp:400-405`).

2. **SM120 SASS Jump Encoding Breakage in `sm120.cpp` (`platforms/cuda/hal/src/arch/sm120.cpp:281-292`)**:
   - In `TarpHandlerSM120::SetJumpInstruction` (`sm120.cpp:281-292`), the code still emits SM86's 48-bit absolute `JMP` opcode:
     ```cpp
     uint64_t lo = 0x345678900000794aULL;
     uint64_t hi = 0x000fea0003807f12ULL;
     ```
   - `FINDINGS_ZH.md` (§11, lines 505–540) and `compendium/03-机制与逆向.md` (lines 70–95) proved that opcode `0x94a` (`JMP.ABS` on SM86) is **invalid on SM120 SASS** and immediately faults with `CUDA_ERROR_ILLEGAL_ADDRESS (700)`. Worse, `trap_sm120.cpp:1286-1290` **still calls `trap_handler_->Instrument()`** (which writes this illegal opcode to `slot20 + 0x880`) whenever `XSCHED_SM120_TRAP_CAVE` is not set!

3. **Kernel RM Source Code Verification of `0x20800122` and `0x83de031d` (`refs/open-gpu-kernel-modules`, tag `595.99.02`)**:
   - In `subdevice_ctrl_gpu_regops.c:105-235` (`subdeviceCtrlCmdGpuExecRegOps_cmn`):
     - Line 143: `if ((hClientTarget == 0) != (hChannelTarget == 0)) return NV_ERR_INVALID_PARAM_STRUCT;` (`0x3a`).
     - Line 163: `if (IS_VIRTUAL(pGpu))` (`pGpu->isVirtual` at offset `+0x910`).
     - Line 198: `status = gpuValidateRegOps(pGpu, pRegOps, regOpCount, bNonTransactional, isClientGspPlugin, NV_FALSE);`
     - Line 205: `if (IS_GSP_CLIENT(pGpu))` (`pGpu->isGspClient` at offset **`+0x911`**) -> calls **`NV_RM_RPC_GPU_EXEC_REG_OPS(pGpu, ..., status)`** and returns `status`!
   - In `gpu_access.c:1177-1203` (`gpuValidateRegOffset_IMPL`):
     - Line 1195: `if (!bSkipPermissionValidation && !osIsAdministrator() && !gpuGetUserRegisterAccessPermissions(pGpu, offset))` -> logs `"User does not have permission to access register offset 0x%x\n"` and returns `NV_ERR_INVALID_ARGUMENT` (`0x1f` via `gpuValidateRegOps`).
   - In `kernel_sm_debugger_session_ctrl.c:629-662` (`ksmdbgssnCtrlCmdDebugExecRegOps_IMPL`, `0x83de031d`):
     - Calls `gpuValidateRegOps(pGpu, pParams->regOps, ..., NV_FALSE)` and, when `IS_GSP_CLIENT(pGpu)` is true (`+0x911 == 1`), forwards the `0x83de031d` control call directly to GSP-RM via `pRmApi->Control(...)`!

---

## 2. Logic Chain

### 2.1 Why `compendium/`'s Central Thesis (`pGpu+0x911 == 0` "Boot-Time Debug Gate") Is False

1. **Observation**: Every document in `compendium/` (`README.md:45-47`, `01-研究过程.md:62-75`, `03-机制与逆向.md:148-198`, `04-通道矩阵.md:17-29`, `06-结论与下一步.md:18-24`, `07-A-B-完整实测.md:169-174`) states as a verified root cause that `nvidia.ko` checks a boot-time debug gate `*(uint8_t*)(pGpu + 0x911) == 0` in `subdeviceCtrlCmdGpuExecRegOps_cmn` and `ksmdbgssnCtrlCmdDebugExecRegOps_IMPL`, early-returning `NV_OK (0)` or `0x1f` without executing register ops on RTX 5060 (`595.99.02`).
2. **Evidence from `refs/open-gpu-kernel-modules` (`subdevice_ctrl_gpu_regops.c:163-232`) and `report/07-源码复核-TRAP触发根因.md:20-97`**:
   - In `OBJGPU` (`g_gpu_nvoc.h`), offset `+0x910` is `isVirtual` and offset `+0x911` is **`isGspClient`** (`IS_GSP_CLIENT(pGpu)`).
   - On RTX 5060 (Blackwell, GSP-RM mandatory), `pGpu->isGspClient` is **`1`**, not `0`. When `isGspClient == 1`, both `subdeviceCtrlCmdGpuExecRegOps_cmn` (`line 205`) and `ksmdbgssnCtrlCmdDebugExecRegOps_IMPL` (`line 647`) **forward the register operations to GSP-RM via RPC**!
   - Why did their early `probe/regops.cpp` see `0x3a` / `0x1f` / no-op? Because of **two bugs in their own probe** (`report/07:62-97`):
     1. They passed `hClientTarget != 0` with `hChannelTarget == 0`, hitting line 143 (`NV_ERR_INVALID_PARAM_STRUCT = 0x3a`).
     2. Their custom `RegOp32` struct placed `regOffset` at byte offset **`+10`** instead of **`+12`** (`NV2080_CTRL_GPU_REG_OP` has 3 `NvU8` fields + 1 byte compiler padding before `NvU32 regGroup` at `+4`, `NvU32 regSubGroup` at `+8`, `NvU32 regOffset` at `+12`), so every register op actually requested **`regOffset = 0`**!
3. **Deduction**: The entire "`pGpu+0x911 == 0` debug gate" narrative in `compendium/` is an artifact of misreading `objdump` disassembly combined with two struct/parameter bugs in `regops.cpp`. `0x20800122` is **active and functional** on SM120 (`595.99.02`), as proven by live reads of Blackwell `PTIMER` (`0x9400`).

### 2.2 Why `0x419e84` (`cuXtraTriggerTrap`)Actually Failed in Hardware & Driver

1. **Non-root failure (`status = 0x1f`)**:
   - `cuXtraTriggerTrap` issues `0x20800122` with `hClientTarget=0, hChannelTarget=0, regOpCount=2` targeting `regOffset = 0x419e84` (op 0: write bit 31 = `0x80000000`; op 1: clear bit 31 = `0x0`).
   - For non-root users (`!osIsAdministrator()`), `gpuValidateRegOffset_IMPL` (`gpu_access.c:1195`) checks `gpuGetUserRegisterAccessPermissions(pGpu, 0x419e84)`. Because `0x419e84` is not in the unprivileged user register access allowlist, CPU-RM rejects it with `NV_ERR_INVALID_ARGUMENT (0x1f)`.
2. **Root / `CAP_SYS_ADMIN` failure (`status = 0`, read value = `0xbadf1100`, no trap)**:
   - Under `sudo` (`osIsAdministrator() == true`), `gpuValidateRegOffset_IMPL` returns `NV_OK (0)` (verified via `tools/nvidia_dbg/nvidia_dbg.c` kretprobe in `report/07:142-151`), and CPU-RM forwards the `0x20800122` RPC to GSP-RM, which returns `status = 0, regStatus = 0`.
   - However, both `0x20800122` (via GSP-RM) and direct PCIe BAR0 `mmap` (`bar0poke`) read **`0xbadf1100`** at `0x419e84`!
   - What is `0xbadf1100`? In NVIDIA GPU hardware architecture, `0xbadfxxxx` (`NV_PPRIV_SYS_PRI_ERROR_CODE`) is the hardware PRI ring error poison value: specifically, `0xbadf1100` indicates a **PRI address decode / floorplan error or GR context routing error** (`BADF11xx` = unmapped/unrouted PRI offset in PGRAPH/GPC/TPC).
   - Why did `0x419e84` return `0xbadf1100` on SM120 (GB206 / RTX 5060)?
     - `0x419e84` was `NV_PGRAPH_PRI_GPCS_TPCS_SM_DBGR_CONTROL0` on Volta/Turing/Ampere/Ada (`SM70`–`SM89`). On Hopper (`SM90`) and Blackwell (`SM100`/`SM120`), the GPC/TPC/SM PRI broadcast aperture (`NV_PGRAPH_PRI_GPCS_TPCS_SM_*`) was **relocated in the BAR0 PRI address map** (or requires `grRouteInfo` / `NV2080_CTRL_GPU_REG_OP_TYPE_GR_CTX` context routing)!
     - Notice that in `report/07:134`, `0x200510` (the old PTIMER offset) also returned `0xbadf1002` on SM120 while the actual Blackwell PTIMER was at `0x9400`! Yet for `SM_DBGR_CONTROL0`, the prior team only probed a handful of hardcoded Ampere neighbors (`0x419e00..0x419ea4`, `0x420000`, `0x4208b0`, `0x424684`) and concluded "GSP accepts the write but produces no hardware effect."
3. **Additional Experimental Flaw in `tools/trigger_root_test.sh` (`lines 32-46`)**:
   - Even if a register write had triggered a single benign trap (`reason = 1`), `trigger_root_test.sh` tested against `bin/victim_spin` (`probe/victim_spin.cu`)—an **uninstrumented kernel** whose `.text.traphandler` is the default driver handler that executes `RET.ABS` in `< 1 µs` on `reason = 1`! Without a cave patched into `.text.traphandler` (to write a sentinel or `EXIT`), a single benign trap pulse (`--reps 1`) is completely invisible to `victim_spin`!
   - Furthermore, `trigger_root_test.sh` ran `bin/regops` as a **separate process** whose `hChannelTarget` was `regops`'s own idle channel (`probe/regops.cpp:194-200`), not `victim_spin`'s channel!

### 2.3 Why `compendium/` Claimed "In-Band Cave Cannot Preempt (`LD.E=0`, Context Cancelled)" Despite Demonstrating a Working `6–89 ms` Clean Exit in `FINDINGS_ZH.md` §26 (v1.8)

This is the most critical internal contradiction in the prior work:
1. **Observation A (`FINDINGS_ZH.md` §26–§28, lines 1389–1549, v1.8–v2.0)**:
   - On 2026-10-25, the team tested a plain CUDA context with:
     - `ToolsDebugger` slot 1 (`+0xd04 = 1`) enabled;
     - `live_traphandler + 0x390` patched with `CALL.ABS` to `slot20 + 0xA00` (inside the existing utility vaspace—**NO `cuXtraInstrMemBlockAlloc` called**);
     - The 8-instruction cave at `slot20 + 0xA00` executing:
       ```sass
       LDC.64 R4, c[0x0][0x170]   // Read XSched preempt_buffer ptr from warp constant bank
       MOV R10, 0x1
       ST.E [R4], R10             // Set global_exit_flag = 1 in device memory!
       IADD R12, R12, 0x10
       RET.ABS.NODEC R12 0x20     // Return cleanly from trap handler!
       ```
     - Triggered via stolen `(fd, hClient, hDebugger)` calling **`0x83de0317` with `{0, 0}`** (benign trap pulse, **not** `{1, 0}` full suspend).
   - **Result (`FINDINGS_ZH.md:1405-1455`, `exp_guard_devflag_kill.log`)**:
     - The trap fired, the cave executed `ST.E [R4], 1` and returned cleanly via `RET.ABS` without corrupting the context;
     - The running kernel's L2 guardian loop read `global_exit_flag == 1` and **exited cleanly (`cudaSuccess`) in `6–89 ms` across multiple consecutive rounds**!
2. **Observation B (`FINDINGS_ZH.md:1456-1549`, `3288-3331`, and `trap_sm120.cpp:447-460, 1244-1290`)**:
   - Why did this stop working when integrated into `CudaQueueLv3Trap` (`trap_sm120.cpp`), leading `compendium/` (`03-机制与逆向.md:200-260`, `06-结论与下一步.md:30-55`) to claim that `0317` always cancels the context in `27 ms` or that `LD.E` returns `0`?
   - Because `CudaQueueLv3Trap` made **three fatal changes** relative to the working v1.8 experiment:
     1. **Called `cuXtraInstrMemBlockAlloc` in `InterruptContext` (`trap_sm120.cpp:1249`)**: As documented in `FINDINGS_ZH.md:1456-1495` (§27), calling `cuXtraInstrMemBlockAlloc` puts the context into `InstrMemAllocator` state; when `cuda-gdb` attaches or sends `0317{1,0}` on a context with `InstrMemAllocator` active, `libcuda` immediately cancels the entire context in `~27 ms` (`cudaErrorContextIsDestroyed`)! Yet `slot20 + 0xA00` is **already allocated** by the driver's utility vaspace and does **not** need `cuXtraInstrMemBlockAlloc` at all!
     2. **Changed the cave from `ST.E [R4], 1; RET.ABS` (writing the exit flag and returning cleanly) to `LD.E R10, [R4]; @P0 EXIT` (`trap_sm120.cpp:450-457`)**:
        - Inside the `0317` trap handler context, `LD.E` from global memory returns `0` (`FINDINGS_ZH.md:1500-1540`), whereas **`ST.E` to global memory (`[R4]`) and `RED.E.ADD` to mapped host memory (`[R2]`) DO work**!
        - Worse, executing a raw `EXIT` inside the `.text.traphandler` cave (instead of `RET.ABS` back to the kernel so the kernel executes `EXIT` in normal mode) aborts the hardware trap handler state machine mid-trap, leaving the SM/context in a dirty trap state (`CUDA_ERROR_ILLEGAL_INSTRUCTION 715`).
     3. **Switched from `0317{0,0}` (benign trap pulse) to `0317{1,0}` (debugger suspend)**: `0317{1,0}` instructs RM/GSP to hold the SM in debugger-stopped state until `0x83de0318` (Resume) is called, whereas `0317{0,0}` (with `RET.ABS` in the cave) allowed the kernel to continue and exit via its L2 flag check!

---

## 3. Caveats

1. **Hardware Execution Boundary**: Per role constraints (`teamwork_explorer`, read-only investigation), no new root kernel probes, BAR0 scans, or code modifications were executed during this survey; all verification was performed by cross-checking the repository's source code (`trap_sm120.cpp`, `sm120.cpp`, `regops.cpp`, `refs/open-gpu-kernel-modules`) against its experimental logs and reports.
2. **Closed-Source GSP Firmware**: While the CPU-RM side of `595.99.02` is fully visible in `refs/open-gpu-kernel-modules`, the GSP-RM handlers for `NV_RM_RPC_GPU_EXEC_REG_OPS` and `0x83de031d` reside inside `gsp_ga10x.bin` / `gsp_tu10x.bin`. However, `libcuda.so.595.99.02` (which constructs the register ops for `cuXtraTriggerTrap` and `ToolsDebugger`) and `open-gpu-kernel-modules` headers/sources are locally inspectable.

---

## 4. Conclusion

### 4.1 Verdict Table: Verified vs. Flawed Claims in `compendium/`

| Claim in `compendium/` (`01`–`07`) | Status | Critical Assessment & Evidence |
|---|---|---|
| **`pGpu+0x911 == 0` is a boot-time debug gate blocking `0x20800122` and `0x83de031d` on RTX 5060** | ❌ **FALSE (Debunked)** | `pGpu+0x911` is `IS_GSP_CLIENT(pGpu)` (`= 1` on RTX 5060), which routes regops to GSP-RM via RPC (`subdevice_ctrl_gpu_regops.c:205`). Early failures were caused by `regops.cpp` struct alignment (`regOffset` at `+10` vs `+12`) and `hClientTarget!=0, hChannelTarget==0` (`report/07:20-97`). |
| **`0x20800122` (`GPU_EXEC_REG_OPS`) is completely dead on SM120 / `595.99.02`** | ❌ **FALSE (Debunked)** | With the fixed 56-byte `NV2080_CTRL_GPU_REG_OP` struct, `0x20800122` works for non-root on allowed offsets (`0x9400` PTIMER, `0x420000`) and for root on all offsets (`report/07:108-151`). |
| **Writing `0x419e84` (`cuXtraTriggerTrap`) as root succeeds in GSP but does not trigger a trap on SM120** | ⚠️ **PARTIALLY VERIFIED, MISDIAGNOSED ROOT CAUSE** | Root bypasses the CPU-RM user access map (`gpu_access.c:1195`), and GSP returns `status=0`, but reading `0x419e84` returns **`0xbadf1100`** (PRI decode/routing error poison). On Blackwell (`SM120`), `SM_DBGR_CONTROL0` is no longer at direct BAR0 offset `0x419e84` (or requires `grRouteInfo` / `0x83de031d` debugger session routing). Also, `trigger_root_test.sh` tested against an uninstrumented kernel (`victim_spin`) from a separate process's idle channel. |
| **Live `.text.traphandler` on SM120 is at `slot20_pc - 0x4400` (`util_base + 0x15f0f00`), while `slot20_pc` (`+0x15f5300`) is `.text.do_pause`** | ✅ **VERIFIED** | Confirmed by memory dumps, instruction binary matching across 4 processes, and live sentinel hit counts (`compendium/03-机制与逆向.md:11-57`, `FINDINGS_ZH.md:541-620`). |
| **SM86's 48-bit `JMP` (`0x345678900000794a`) is illegal on SM120 SASS; SM120 requires `CALL.ABS.NOINC` (`0x...7943`) / `RET.ABS.NODEC` (`0x...7950`)** | ✅ **VERIFIED** | Confirmed by `nvdisasm -b SM120` and hardware execution (`compendium/03-机制与逆向.md:70-128`, `trap_sm120.cpp:400-460`). Yet `sm120.cpp:281-292` still contains the broken SM86 `JMP`! |
| **`0x83de0317` trap cave cannot be used for non-destructive kernel exit because `0317` always cancels the context in `27 ms` and `LD.E` returns `0`** | ❌ **FLAWED / INCOMPLETE** | In `FINDINGS_ZH.md` §26 (v1.8, lines 1389–1455), `0317{0,0}` on a plain context (without `cuXtraInstrMemBlockAlloc`) executed `LDC R4, c[0x0][0x170]; ST.E [R4], 1; RET.ABS` in the cave, and the kernel exited cleanly via its L2 guardian loop in `6–89 ms` (`cudaSuccess`)! The `27 ms` context cancel only occurred because `CudaQueueLv3Trap` called `cuXtraInstrMemBlockAlloc` and used `0317{1,0}` + raw `EXIT`. |
| **Route A (`cuda-gdb` + `0x83de0317{1,0}/0318` suspend/resume) achieves `~116 ms` (`65 ms` warm) pause and `< 2 ms` resume** | ✅ **VERIFIED** | Confirmed by `compendium/07-A-B-完整实测.md:19-46` and `FINDINGS_ZH.md:3010-3090`. |

---

### 4.2 Promising Avenues Missed, Misconfigured, or Prematurely Ruled Out

1. **Avenue 1 (Immediate, Proven in v1.8): Non-Destructive In-Band Trap-Assisted L2 Exit (`0x83de0317{0,0}` + Cave `ST.E [c[0x0][0x170]], 1` + `RET.ABS`, WITHOUT `cuXtraInstrMemBlockAlloc`)**:
   - **Why it was missed/abandoned**: `trap_sm120.cpp` unconditionally calls `cuXtraInstrMemBlockAlloc` in `InterruptContext` (`line 1249`) and uses `LD.E` + raw `EXIT` inside the cave (`lines 450-457`).
   - **How to fix**:
     - Do **not** call `cuXtraInstrMemBlockAlloc` on SM120 (use the pre-existing `slot20 + 0xA00` cave directly).
     - Instead of `LD.E R10, [R4]; @P0 EXIT` inside the trap handler (where `LD.E` returns 0 and raw `EXIT` corrupts the trap state machine), have the cave execute `LDC.64 R4, c[0x0][0x170]; MOV R10, 1; ST.E [R4], R10; IADD R12, R12, 0x10; RET.ABS.NODEC R12 0x20` (writing `1` to `global_exit_flag` in XSched's `preempt_buffer` from inside the trap handler, or modifying the return PC at `[R12]`/`R13` to jump to the kernel's clean exit epilogue before `RET.ABS`), and trigger with `0x83de0317{0,0}`!

2. **Avenue 2 (Misconfigured / Never Re-tested): `0x83de031d` (`ksmdbgssnCtrlCmdDebugExecRegOps_IMPL`) with Fixed 56-Byte `NV2080_CTRL_GPU_REG_OP` (`regOffset` at `+12`)**:
   - **Why it was missed**: `0x83de031d` was only tested with the broken `RegOp32` struct (`regOffset` at `+10`) before 2026-10-27 (`report/07:204-213`).
   - **Why it matters**: In `kernel_sm_debugger_session_ctrl.c:629-660`, `0x83de031d` is bound to a specific `KernelSMDebuggerSession` (which already knows the target `hChannel` and graphics context) and forwards directly to GSP-RM via `pRmApi->Control`. Moreover, when `cuda-gdb` executes `0x83de0317`, GSP-RM itself writes the hardware SM debug control register—tracing or disassembling the `0x83de031d` / `0x83de0317` handler in `gsp_ga10x.bin` (or `libcuda.so.595.99.02` where `0x419e84` is selected per-architecture) will reveal the exact Blackwell (`GB20x`) PRI offset or `grRouteInfo` that replaced `0x419e84`!

3. **Avenue 3 (Prematurely Ruled Out): Locating the True Blackwell (`GB20x` / `SM120`) PRI Offset for `SM_DBGR_CONTROL0` & Fixing the Root Trigger Test**:
   - **Why it failed previously**:
     1. `0x419e84` returned `0xbadf1100` on SM120 because `0x419e84` is the pre-Blackwell (`Volta`–`Ada`) PRI offset, just as `0x200510` (`0xbadf1002`) was the old PTIMER offset replaced by `0x9400`.
     2. In `libcuda.so.595.99.02`, `cuXtraTriggerTrap` (`libcuda+0x4551e0`) or `ToolsTrapHandler` has architecture-specific tables or `grRouteInfo` / `regType` (`GR_CTX`) logic; furthermore, `gsp_ga10x.bin` (which executes `0x83de0317`) contains the exact register write that triggers `BPT_TRAP` (`reason = 1`) on SM120!
     3. Any root register trigger test **must** run against a victim kernel with the **`slot20 + 0xA00` sentinel cave patched** (not the uninstrumented `victim_spin` where a benign trap returns in `< 1 µs` invisibly) and must pass the **victim's `(hClientTarget, hChannelTarget)`** (or run in-process inside the victim).

4. **Avenue 4 (Code Bug in `sm120.cpp` + `trap_sm120.cpp`)**:
   - `TarpHandlerSM120::SetJumpInstruction` in `platforms/cuda/hal/src/arch/sm120.cpp:281-292` still writes the illegal SM86 `JMP` instruction (`0x345678900000794a`), and `trap_sm120.cpp:1286-1290` still invokes `trap_handler_->Instrument()` when `XSCHED_SM120_TRAP_CAVE` is unset. This fallback must be disabled or replaced with the SM120 `CALL.ABS.NOINC` patch on `live_traphandler + 0x390`.

---

## 5. Verification Method

To independently verify every finding in this report:

1. **Verify the `pGpu+0x911` (`IS_GSP_CLIENT`) source code vs. `compendium/`'s false debug-gate claim**:
   - Inspect `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/Level3forLinux/refs/open-gpu-kernel-modules/src/nvidia/src/kernel/gpu/subdevice/subdevice_ctrl_gpu_regops.c` lines 143–235 and `src/nvidia/src/kernel/gpu/gpu_access.c` lines 1177–1203.
   - Compare with `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/Level3forLinux/xsched/test/sm120_l3_trap_v2/report/07-源码复核-TRAP触发根因.md` lines 1–185 and `compendium/03-机制与逆向.md` lines 148–198.
2. **Verify the v1.8 working in-band trap-assisted L2 exit (`0317{0,0}` + cave `ST.E` + `RET.ABS`)**:
   - Inspect `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/Level3forLinux/xsched/test/sm120_l3_trap_v2/FINDINGS_ZH.md` lines 1389–1549 (§26–§28) and `evidence/live/exp_guard_devflag_kill.log`.
3. **Verify the HAL bugs in `trap_sm120.cpp` and `sm120.cpp`**:
   - Inspect `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/Level3forLinux/xsched/platforms/cuda/hal/src/arch/sm120.cpp` lines 281–360 (SM86 `JMP` opcode `0x345678900000794a` and `slot20 + 0x880` patch target).
   - Inspect `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/Level3forLinux/xsched/platforms/cuda/hal/src/level3/trap_sm120.cpp` lines 447–460 (`LD.E` + `@P0 EXIT` in `armed_stub`) and lines 1244–1290 (`cuXtraInstrMemBlockAlloc` + fallback to `trap_handler_->Instrument()`).
4. **Invalidation Conditions**:
   - These conclusions would only be invalidated if tested on a non-GSP driver or a pre-Blackwell GPU (`SM80`/`SM86`/`SM89`) where `0x419e84` is still the valid PRI offset for `SM_DBGR_CONTROL0` and SM86 `JMP` encoding is valid.
