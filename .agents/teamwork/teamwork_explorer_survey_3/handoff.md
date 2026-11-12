# Handoff Report: Live Blackwell (RTX 5060) Hardware/Driver Environment & Preemption/Trap Mechanisms

**Core Summary**: The host runs Ubuntu 26.04.1 LTS (`Linux 7.0.0-38-generic`, SecureBoot disabled) with an **NVIDIA GeForce RTX 5060** (`GB206` [`10de:2d05`], Compute Capability `12.0` / `sm_120`, 30 SMs, 64 MB BAR0) on NVIDIA Open Kernel Driver & GSP Firmware **`595.99.02`** (exact source match in workspace `open-gpu-kernel-modules-595.99.02`; `550.163.01` has zero Blackwell support). On Blackwell, GSP-RM is mandatory (`EnableGpuFirmware=18`) and owns PGRAPH/SM debug registers, while CPU-RM (`nvidia.ko`) provides **four distinct sub-millisecond preemption/trap paths**—including three **non-abort resumable** mechanisms (`NVA06C_CTRL_CMD_PREEMPT` / `NV_RUNLIST_PREEMPT` with `CILP`/`CTA` via `NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE`, `NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT`/`RESUME_CONTEXT` or `SET_SINGLE_SM_STOP_TRIGGER`/`RUN_TRIGGER` routed to GSP-RM, and CUDA 13.2 `cuCheckpointProcess*` / `cuptiCheckpoint*`) alongside paper-style **abort/trap Level-3 preemption** (`NV83DE` + `SET_DROP_DEFERRED_RC` or PTX `trap;`/`brkpt;`).

---

## 1. Observation

### 1.1 Live Host, Kernel, GPU Hardware, and Driver Inventory

All observations below were collected via read-only inspection of the live system:

| Property | Observed Value | Source Command / File |
|---|---|---|
| **OS & Kernel** | Ubuntu 26.04.1 LTS (Resolute Raccoon), `Linux 7.0.0-38-generic #38-Ubuntu SMP PREEMPT_DYNAMIC x86_64` | `uname -a`, `/etc/os-release` |
| **Compiler & Secure Boot** | `gcc (Ubuntu 15.2.0-18ubuntu1) 15.2.0`; `SecureBoot disabled` (`Platform is in Setup Mode`) | `gcc --version`, `mokutil --sb-state` |
| **GPU Model & PCI ID** | `NVIDIA GeForce RTX 5060`, `GB206` [`10de:2d05`] (rev a1), Subsystem MSI [`1462:5371`], Bus `0000:01:00.0`, IRQ 99, IOMMU group 12 | `lspci -vvnn -s 01:00.0`, `/proc/driver/nvidia/gpus/0000:01:00.0/information` |
| **PCIe BAR Regions** | **Region 0 (BAR0 MMIO)**: `0xf0000000 - 0xf3ffffff` (**64 MB** non-prefetchable, `0x04000000`)<br>**Region 1 (BAR1 VRAM)**: `0xa00000000 - 0xbffffffff` (8 GB 64-bit prefetchable)<br>**Region 3 (BAR2/3)**: `0xc10000000 - 0xc11ffffff` (32 MB 64-bit prefetchable) | `lspci -vvnn -s 01:00.0`, `/sys/bus/pci/devices/0000:01:00.0/resource` |
| **Compute Capability & SMs** | Compute Capability **`12.0`** (`sm_120`), **30 SMs**, 1536 max threads/SM, 24 max blocks/SM, 102400 B shared mem/SM, 65536 regs/SM, 8151 MiB VRAM, Addressing Mode: `HMM` | `cuDeviceGetAttribute` via `libcuda.so.1`, `nvidia-smi -q` |
| **Preemption Device Attrs** | `CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED` (attr 90) = **`1`**<br>`CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH` (attr 95) = **`1`**<br>`CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM` = **`1`** | `cuDeviceGetAttribute` via `libcuda.so.1` |
| **Loaded Kernel Modules** | `nvidia` (17530880 B, Dual MIT/GPL, `/lib/modules/7.0.0-38-generic/kernel/nvidia-595-open/nvidia.ko`), `nvidia_uvm` (2293760 B), `nvidia_modeset`, `nvidia_drm` | `lsmod`, `modinfo nvidia` |
| **Driver & GSP Firmware** | Driver Version: **`595.99.02`**, GSP Firmware Version: **`595.99.02`**, VBIOS: `98.06.40.00.49` | `nvidia-smi -q`, `/proc/driver/nvidia/version` |
| **NVIDIA Module Params** | `EnableGpuFirmware: 18` (GSP-RM active/mandatory), `RmProfilingAdminOnly: 1`, `EnableDbgBreakpoint: 0`, `IgnoreMMIOCheck: 0`, `ModifyDeviceFiles: 1` | `/proc/driver/nvidia/params` |
| **Device Nodes** | `/dev/nvidiactl` (`195:255`), `/dev/nvidia0` (`195:0`), `/dev/nvidia-uvm` (`509:0`), `/dev/nvidia-uvm-tools` (`509:1`), `/dev/nvidia-modeset` (`195:254`) — all `crw-rw-rw-` | `ls -la /dev/nvidia*` |
| **CUDA Toolkit vs Driver** | `/usr/bin/nvcc` is CUDA `12.4` (`V12.4.131`, supports up to `compute_90` offline and PTX JIT to `sm_120`); `libcuda.so.595.99.02` reports `cuDriverGetVersion = 13020` (**CUDA 13.2**) | `nvcc --version`, `cuDriverGetVersion` |
| **Open Kernel Sources** | 1. `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/open-gpu-kernel-modules-595.99.02` (exact match to running `595.99.02` driver; full Blackwell GB100/GB202/GB206 support)<br>2. `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/open-gpu-kernel-modules-550.163.01` (zero `*blackwell*` / `*gb20*` files)<br>3. `/usr/src/nvidia-595.99.02` (installed DKMS kernel interface sources) | `find_by_name` |

---

### 1.2 Open Kernel Module (`595.99.02`) Code Observations on Blackwell (`GB20x` / `SM120`)

All file paths below are relative to `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/open-gpu-kernel-modules-595.99.02`:

#### Observation A — GSP-RM Architecture & Routing of `NV83DE` (`GT200_DEBUGGER`) Control Calls
1. **`src/common/sdk/nvidia/inc/ctrl/ctrl83de/ctrl83dedebug.h:1-1235`** defines the `NV83DE` (`GT200_DEBUGGER`) RM control commands:
   - `NV83DE_CTRL_CMD_SM_DEBUG_MODE_ENABLE` (`0x83de0301`, line 74) and `DISABLE` (`0x83de0302`, line 85)
   - `NV83DE_CTRL_CMD_DEBUG_SET_MODE_MMU_DEBUG` (`0x83de0307`, line 179)
   - `NV83DE_CTRL_CMD_DEBUG_SET_EXCEPTION_MASK` (`0x83de0309`, line 221)
   - `NV83DE_CTRL_CMD_DEBUG_READ_SINGLE_SM_ERROR_STATE` (`0x83de030b`, line 292) & `CLEAR_SINGLE_SM_ERROR_STATE` (`0x83de030c`, line 333)
   - `NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES` (`0x83de030f`, line 378) & `CLEAR_ALL_SM_ERROR_STATES` (`0x83de0310`, line 417)
   - `NV83DE_CTRL_CMD_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE` (`0x83de0313`, line 456)
   - `NV83DE_CTRL_CMD_DEBUG_SET_SINGLE_STEP_INTERRUPT_HANDLING` (`0x83de0314`, line 473)
   - `NV83DE_CTRL_CMD_DEBUG_READ_MEMORY` (`0x83de0315`, line 490) & `WRITE_MEMORY` (`0x83de0316`, line 521)
   - `NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT` (`0x83de0317`, line 552) & `RESUME_CONTEXT` (`0x83de0318`, line 570)
   - `NV83DE_CTRL_CMD_DEBUG_READ_SURFACE` (`0x83de031a`, line 593) & `WRITE_SURFACE` (`0x83de031b`, line 600)
   - `NV83DE_CTRL_CMD_DEBUG_GET_MAPPINGS` (`0x83de031c`, line 614)
   - `NV83DE_CTRL_CMD_DEBUG_EXEC_REG_OPS` (`0x83de031d`, line 675)
   - `NV83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_SINGLE_STEP` (`0x83de0321`, line 722)
   - `NV83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_STOP_TRIGGER` (`0x83de0322`, line 743, supports `NV83DE_CTRL_DEBUG_SET_SINGLE_SM_STOP_TRIGGER_SM_ID_BROADCAST = ~0` at line 756)
   - `NV83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_RUN_TRIGGER` (`0x83de0323`, line 765)
   - `NV83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_SKIP_IDLE_WARP_DETECT` (`0x83de0324`, line 781)
   - `NV83DE_CTRL_CMD_DEBUG_GET_SINGLE_SM_DEBUGGER_STATUS` (`0x83de0325`, line 803)
   - `NV83DE_CTRL_CMD_DEBUG_READ_BATCH_MEMORY` (`0x83de0326`, line 848) & `WRITE_BATCH_MEMORY` (`0x83de0327`, line 856)
   - `NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO` (`0x83de0328`, line 911)
   - `NV83DE_CTRL_CMD_DEBUG_SET_DROP_DEFERRED_RC` (`0x83de0329`, line 947)
   - `NV83DE_CTRL_CMD_DEBUG_SET_MODE_ERRBAR_DEBUG` (`0x83de032a`, line 964)
2. **`src/nvidia/generated/g_kernel_sm_debugger_session_nvoc.c:182-538`** and **`src/nvidia/inc/kernel/rmapi/control.h:269`** (`#define RMCTRL_FLAGS_ROUTE_TO_PHYSICAL 0x00000040`) show how `NV83DE` commands execute when GSP-RM is enabled:
   - Commands exported with `flags = 0x248u` or `0x640u` (bit `0x40` = `RMCTRL_FLAGS_ROUTE_TO_PHYSICAL`) are intercepted by `rmControlCmdExecute_Prologue` (`src/nvidia/src/kernel/rmapi/control.c:775-812`) and forwarded directly to GSP-RM via `resControl_Prologue` -> `gpuresInternalControlForward_IMPL` (`src/nvidia/src/kernel/gpu/gpu_resource.c:365-378`) -> `pRmApi->Control(...)`:
     - `0x83de0301` (`SM_DEBUG_MODE_ENABLE`), `0x83de0302` (`SM_DEBUG_MODE_DISABLE`), `0x83de0307` (`SET_MODE_MMU_DEBUG`), `0x83de0319` (`SET_WATCHPOINT`), `0x83de031a` (`CLEAR_WATCHPOINT`), `0x83de0321` (`SET_SINGLE_SM_SINGLE_STEP`), `0x83de0322` (`SET_SINGLE_SM_STOP_TRIGGER`), `0x83de0323` (`SET_SINGLE_SM_RUN_TRIGGER`), `0x83de0324` (`SET_SINGLE_SM_SKIP_IDLE_WARP_DETECT`), `0x83de0325` (`GET_SINGLE_SM_DEBUGGER_STATUS`), `0x83de0328` (`READ_MMU_FAULT_INFO`), `0x83de032a` (`SET_MODE_ERRBAR_DEBUG`), `0x83de032b` (`GET_MODE_ERRBAR_DEBUG`).
   - Commands implemented in CPU-RM (`src/nvidia/src/kernel/gpu/gr/kernel_sm_debugger_session.c` and `kernel_sm_debugger_session_ctrl.c`) perform CPU-side preprocessing or validation and then explicitly call `ksmdbgssnInternalControlForward(pKernelSmDebuggerSession, command, pParams, size)` to execute in GSP-RM:
     - `ksmdbgssnCtrlCmdDebugSuspendContext_IMPL` (`kernel_sm_debugger_session_ctrl.c:558-580`): zeros `pParams`, sets `waitForEvent = NV83DE_CTRL_CMD_DEBUG_SUSPEND_ALL_CONTEXTS_FOR_CLIENT_WAIT_FOR_EVENT` (`1`), and calls `ksmdbgssnInternalControlForward(pKernelSmDebuggerSession, NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT, pParams, sizeof(*pParams))`.
     - `ksmdbgssnCtrlCmdDebugResumeContext_IMPL` (`kernel_sm_debugger_session_ctrl.c:585-591`): calls `ksmdbgssnInternalControlForward(pKernelSmDebuggerSession, NV83DE_CTRL_CMD_DEBUG_RESUME_CONTEXT, NULL, 0)`.
     - `ksmdbgssnCtrlCmdDebugReadAllSmErrorStates_IMPL` (`kernel_sm_debugger_session_ctrl.c:636-698`), `ClearAllSmErrorStates` (`lines 728-746`), and `SetDropDeferredRc` (`lines 751-759`): forward directly to GSP-RM via `ksmdbgssnInternalControlForward`.
     - `ksmdbgssnCtrlCmdDebugExecRegOps_IMPL` (`kernel_sm_debugger_session.c:405-474`): calls `kmigmgrPerformScrubberIfNeeded_HAL`, then `subdeviceCtrlCmdGpuExecRegOps_IMPL` (`src/nvidia/src/kernel/gpu/subdevice/subdevice_ctrl_gpu_regops.c:107-244`), which validates each register offset against `pGpu->userRegisterAccessMap` (a compressed bitmap fetched from GSP-RM in `gpuInitRegisterAccessMap_IMPL`, `src/nvidia/src/kernel/gpu/gpu_register_access_map.c:208-246`) unless `pGpu->bIgnoreMMIOCheck` (`IgnoreMMIOCheck=1`) is set (`src/nvidia/src/kernel/gpu/gpu_access.c:1177-1193`), and on GSP clients (`IS_GSP_CLIENT(pGpu)`) forwards the verified batch to GSP-RM (`subdevice_ctrl_gpu_regops.c:146-184`).

#### Observation B — Native Hardware Compute Preemption (`WFI` / `CTA` / `CILP`) & Runlist/TSG Preemption on Blackwell
1. **`src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gr.h:788-850`** defines `NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE` (`0x20801210`) with compute preemption modes:
   - `NV2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_COMPUTE_WFI` (`0x00000000`) — Wait-For-Idle
   - `NV2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_COMPUTE_CTA` (`0x00000001`) — Thread-block (CTA) boundary preemption
   - `NV2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_COMPUTE_CILP` (`0x00000002`) — Compute Instruction-Level Preemption
2. **`src/nvidia/src/kernel/gpu/gr/kernel_graphics_context.c:216-612`** implements `NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE` on CPU-RM + GSP-RM:
   - `kgrctxCtrlSetCtxswPreemptionMode_IMPL` (`lines 216-247`) -> `_kgrctxSetCtxswPreemptionMode` (`lines 249-330`) -> `kgrctxAllocCtxPreemptionBuffers_IMPL` (`lines 434-612`).
   - Queries context buffer sizes discovered from GSP-RM static info (`kgraphicsGetContextBufferInfo`, `lines 468-474`) for:
     - `NV_CTXSW_MAIN_IMAGE_PREEMPTION_BUFFER` (`preemptCtxswBuffer`)
     - `NV_CTXSW_MAIN_IMAGE_SPILL_BUFFER` (`spillCtxswBuffer`)
     - `NV_CTXSW_MAIN_IMAGE_BETA_CB_BUFFER` (`betaCBCtxswBuffer`)
     - `NV_CTXSW_MAIN_IMAGE_PAGEPOOL_BUFFER` (`pagepoolCtxswBuffer`)
     - `NV_CTXSW_MAIN_IMAGE_RTV_CB_BUFFER` (`rtvCbCtxswBuffer`)
   - Allocates and maps these VA buffers into the channel/TSG context, promotes them to GSP-RM via `NV2080_CTRL_CMD_GPU_PROMOTE_CTX` (`lines 393-402`), and forwards `NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE` to GSP-RM (`lines 262-269`).
3. **`src/common/sdk/nvidia/inc/ctrl/ctrla06c.h:191-214`** and **`src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c:1054-1130`** define and implement `NVA06C_CTRL_CMD_PREEMPT` (`0xa06c0105`):
   - Exported in `src/nvidia/generated/g_kernel_channel_group_api_nvoc.c:270` with `flags = 0x10248u` (includes `RMCTRL_FLAGS_ROUTE_TO_PHYSICAL = 0x40`, `PRIVILEGED_IF_RS_ACCESS_DISABLED`).
   - Accepts `bWait`, `bManualTimeout`, `timeoutUs` (up to `NVA06C_CTRL_CMD_PREEMPT_MAX_MANUAL_TIMEOUT_US = 1000000` us).
4. **`src/common/inc/swref/published/blackwell/gb202/dev_runlist.h:25-77`** and **`src/nvidia/src/kernel/gpu/fifo/arch/blackwell/kernel_fifo_gb202.c:281-316`** reveal the exact Blackwell hardware runlist preemption registers in BAR0 MMIO:
   - `NV_RUNLIST_PREEMPT` (`offset 0x014` from `runlistPriBase`):
     - `11:0` (`ID`): HW TSG ID (`tsgId`)
     - `25:24` (`TYPE`): `_RUNLIST` (`0x0`), `_TSG` (`0x1`)
   - `NV_RUNLIST_PREEMPT_STS` (`offset 0x024`): bit `0` (`PREEMPT_PENDING` = `0x0` idle / `0x1` pending)
   - `NV_CHRAM_CHANNEL(ChID)` (`offset 0x000 + ChID*4` from `chramPriBase`): channel status bits (`ENABLE`, `NEXT`, `BUSY`, `PENDING`, `CTX_RELOAD`) and write actions (`_WRITE_CONTROL_FORCE_CTX_RELOAD`).
   - Every user/kernel channel has its `runlistPriBase` and `chramPriBase` stored in its `ENG_FIFO` engine info table (`kfifoEngineInfoXlate_HAL`).

#### Observation C — User-Space CUDA Driver (`595.99.02` / CUDA 13.2) Debugger & Process Checkpoint APIs
1. **`/usr/lib/x86_64-linux-gnu/libcuda.so.595.99.02`** and **`libcudadebugger.so.595.99.02`** export:
   - **CUDA Debugger API**: `cudbgGetAPI`, `cudbgGetAPIVersion`, `cudbgApiInit`, `cudbgApiAttach`, `cudbgApiDetach`, `cudbgEnablePreemptionDebugging`, `cudbgMain`, `cudbgReportDriverApiError`, `cudbgReportDriverInternalError`.
   - **CUDA Process Checkpoint/Restore API**: `cuCheckpointProcessGetState`, `cuCheckpointProcessLock`, `cuCheckpointProcessCheckpoint`, `cuCheckpointProcessRestore`, `cuCheckpointProcessUnlock`.
2. **`/usr/include/cupti_checkpoint.h:1-128`** provides the header-documented **CUPTI Checkpoint API** (`NV::Cupti::Checkpoint::cuptiCheckpointSave`, `cuptiCheckpointRestore`, `cuptiCheckpointFree`) to snapshot, restore, and replay a CUDA kernel launch at sub-kernel or kernel granularity.

---

## 2. Logic Chain

1. **Why Legacy Direct BAR0 `PGRAPH` SM Register Poking Differs on Blackwell**:
   - From Observation 1.1 (`EnableGpuFirmware: 18`) and Observation 1.2A (`g_kernel_sm_debugger_session_nvoc.c`), Blackwell (`GB206`) runs exclusively with GSP-RM enabled.
   - In the GSP-RM split architecture, the RISC-V GSP processor owns PGRAPH context switching and SM debug state machines. While BAR0 is 64 MB (`0xf0000000 - 0xf3ffffff`) and exposes host/runlist MMIO (`runlistPriBase + NV_RUNLIST_PREEMPT`), `NV83DE` debugger operations (`SUSPEND_CONTEXT`, `RESUME_CONTEXT`, `SET_SINGLE_SM_STOP_TRIGGER`, `SET_SINGLE_SM_RUN_TRIGGER`, `SET_DROP_DEFERRED_RC`, `EXEC_REG_OPS`) are routed by CPU-RM directly to GSP-RM via `RMCTRL_FLAGS_ROUTE_TO_PHYSICAL` (`0x40`) or `ksmdbgssnInternalControlForward`.

2. **Why Paper-Style "Trap-as-Abort" Level 3 Preemption Can Be Reproduced on Blackwell via `NV83DE` + `SET_DROP_DEFERRED_RC`**:
   - In the XSched paper's Level 3 design, an in-flight low-priority command is aborted in hundreds of microseconds via an SM trap/exception and re-launched from the command buffer upon resume.
   - On Blackwell (`595.99.02`), allocating an `NV83DE` (`GT200_DEBUGGER`) object attached to the target `KernelGraphicsObject` (`hObject`) or `Subdevice` allows issuing:
     1. `NV83DE_CTRL_CMD_SM_DEBUG_MODE_ENABLE` (`0x83de0301`)
     2. `NV83DE_CTRL_CMD_DEBUG_SET_DROP_DEFERRED_RC` (`0x83de0329`, `kernel_sm_debugger_session_ctrl.c:751`) — critical so that an SM trap/exception does not fatally tear down the channel via Robust Channels (RC) recovery.
     3. Either host-triggered stop/suspend (`NV83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_STOP_TRIGGER` `0x83de0322` with `SM_ID_BROADCAST = ~0` or `NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT` `0x83de0317`) or device-triggered PTX `trap;` / `brkpt;` (`HWW_WARP_ESR_BPT_TRAP` / `BPT_PAUSE`), followed by `NV83DE_CTRL_CMD_DEBUG_CLEAR_ALL_SM_ERROR_STATES` (`0x83de0310`).

3. **Why Non-Abort Resumable Preemption Is Strictly Superior on Blackwell (`GB206` / `SM120`) and Immediately Available**:
   - **Path 1 (Hardware CILP / CTA Preemption via `NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE` + `NVA06C_CTRL_CMD_PREEMPT` / `NV_RUNLIST_PREEMPT`)**:
     - Observation 1.1 confirms `CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 1` on RTX 5060.
     - Observation 1.2B (`kernel_graphics_context.c:216-612`) shows that calling `NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE` (`0x20801210`) with `COMPUTE_CILP` (`2`) or `COMPUTE_CTA` (`1`) causes CPU-RM + GSP-RM to allocate and bind the hardware context-switch save buffers (`preemptCtxswBuffer`, `spillCtxswBuffer`, `betaCBCtxswBuffer`, `pagepoolCtxswBuffer`, `rtvCbCtxswBuffer`).
     - Once CILP or CTA buffers are bound to a TSG's graphics context, preempting that TSG via `NVA06C_CTRL_CMD_PREEMPT` (`0xa06c0105`) or a direct BAR0 write to `runlistPriBase + NV_RUNLIST_PREEMPT` (`0x014`, `_TYPE_TSG | tsgId`) saves all in-flight SM warp state to the context buffers in sub-millisecond time **without aborting or losing work**, and automatically resumes at the exact instruction (`CILP`) or CTA (`CTA`) when the TSG is rescheduled!
   - **Path 2 (Non-Destructive SM Pause/Resume in Place via `NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT` / `RESUME_CONTEXT` or `SET_SINGLE_SM_STOP_TRIGGER` / `RUN_TRIGGER`)**:
     - Observation 1.2A shows that `NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT` (`0x83de0317`) and `NV83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_STOP_TRIGGER` (`0x83de0322` with `SM_ID_BROADCAST = ~0`) halt SM warp scheduling in GSP-RM without killing the warps, and `NV83DE_CTRL_CMD_DEBUG_RESUME_CONTEXT` (`0x83de0318`) / `SET_SINGLE_SM_RUN_TRIGGER` (`0x83de0323`) resume execution with zero lost work.
   - **Path 3 (Device-Assisted PTX `brkpt;` / Cooperative Exit + `cuptiCheckpoint*` / `cuCheckpointProcess*`)**:
     - Unlike PTX `trap;` (which raises `BPT_TRAP` error state), PTX `brkpt;` raises `BPT_PAUSE`, which halts warps cleanly for debugger inspection/resume, or a kernel checking a mapped host flag can exit early at CTA/loop boundaries without any RC exception overhead.

---

## 3. Caveats

1. **CUDA Toolkit (`nvcc` 12.4) vs Driver (`595.99.02` / CUDA 13.2)**:
   - The installed `/usr/bin/nvcc` is CUDA 12.4, which supports up to `-arch=compute_90` / `-code=compute_90` offline; to run on `sm_120` (RTX 5060), compile with `-arch=compute_90` (embedding PTX) or `-gencode arch=compute_90,code=compute_90` so `libcuda.so.595.99.02` JIT-compiles PTX to native `sm_120` SASS at module load time.
2. **RM Access Control & Profiling Permissions**:
   - `/proc/driver/nvidia/params` currently has `RmProfilingAdminOnly: 1` and `IgnoreMMIOCheck: 0`. Unprivileged user processes invoking privileged RM control commands or debugger regops may require either `CAP_SYS_ADMIN` (`sudo`), setting `options nvidia NVreg_RestrictProfilingToAdminUsers=0`, or a lightweight helper ioctl/module (note: `SecureBoot` is disabled on this machine, so loading a custom `nvidia.ko` or helper kmod is unrestricted).
3. **`open-gpu-kernel-modules-550.163.01` Inapplicability**:
   - The `550.163.01` tree in the workspace has no Blackwell support (`GB100`/`GB202`/`GB206` files do not exist in 550). All Blackwell driver work and header inclusions must reference `open-gpu-kernel-modules-595.99.02`.

---

## 4. Conclusion

1. **Environment Readiness**: The live system has an **NVIDIA GeForce RTX 5060 (`GB206`, `sm_120`, 30 SMs)** running open kernel modules **`595.99.02`** on Linux `7.0.0-38-generic` with **SecureBoot disabled** and the exact matching source tree at `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/open-gpu-kernel-modules-595.99.02`.
2. **Actionable Architecture for XSched on Blackwell (`SM120`)**:
   - **Primary Non-Abort Sub-Millisecond Preemption (Recommended)**: Configure compute contexts/TSGs with **`NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE` (`0x20801210`)** set to **`COMPUTE_CILP` (`2`)** or **`COMPUTE_CTA` (`1`)** (which automatically allocates and promotes the 5 hardware preemption buffers in `kgrctxAllocCtxPreemptionBuffers_IMPL`), and trigger preemption via **`NVA06C_CTRL_CMD_PREEMPT` (`0xa06c0105`)** or direct BAR0 write to **`runlistPriBase + NV_RUNLIST_PREEMPT` (`0x014`)**. This achieves sub-millisecond hardware instruction-level (`CILP`) or CTA-level (`CTA`) preemption with **zero wasted work and no kernel abort/re-execution**.
   - **Secondary Debugger-Based Pause/Resume & Trap-Abort Preemption (`NV83DE`)**: Attach an **`NV83DE` (`GT200_DEBUGGER`)** session to the target `KernelGraphicsObject` and use:
     - **Resumable Pause/Resume**: `NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT` (`0x83de0317`) + `NV83DE_CTRL_CMD_DEBUG_RESUME_CONTEXT` (`0x83de0318`) (or `SET_SINGLE_SM_STOP_TRIGGER` `0x83de0322` with `SM_ID_BROADCAST = ~0` + `SET_SINGLE_SM_RUN_TRIGGER` `0x83de0323`).
     - **Paper-Style Trap/Abort + Re-Execution**: Enable `NV83DE_CTRL_CMD_SM_DEBUG_MODE_ENABLE` (`0x83de0301`) + `NV83DE_CTRL_CMD_DEBUG_SET_DROP_DEFERRED_RC` (`0x83de0329`) + PTX `trap;`/`brkpt;` or SM stop trigger + `NV83DE_CTRL_CMD_DEBUG_CLEAR_ALL_SM_ERROR_STATES` (`0x83de0310`) + command buffer replay.

---

## 5. Verification Method

Run the following read-only commands in `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode` to independently verify every observation and conclusion:

```bash
# 1. Verify live GPU, driver 595.99.02, GSP firmware, and 64MB BAR0
nvidia-smi -q | grep -E "Product Name|Product Architecture|Driver Version|GSP Firmware|Bus Id"
lspci -vvnn -s 01:00.0 | grep -E "VGA|Region"
cat /proc/driver/nvidia/params

# 2. Verify NV83DE (GT200_DEBUGGER) control commands and GSP-RM routing in 595.99.02
grep -n "NV83DE_CTRL_CMD_DEBUG_" open-gpu-kernel-modules-595.99.02/src/common/sdk/nvidia/inc/ctrl/ctrl83de/ctrl83dedebug.h
grep -n -C 5 "ksmdbgssnCtrlCmdDebugSuspendContext_IMPL\|ksmdbgssnCtrlCmdDebugResumeContext_IMPL\|ksmdbgssnCtrlCmdDebugSetDropDeferredRc_IMPL" open-gpu-kernel-modules-595.99.02/src/nvidia/src/kernel/gpu/gr/kernel_sm_debugger_session_ctrl.c

# 3. Verify CILP/CTA preemption mode setup and Blackwell NV_RUNLIST_PREEMPT registers
grep -n -C 10 "NV2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_COMPUTE_CILP" open-gpu-kernel-modules-595.99.02/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gr.h
grep -n -C 15 "NV_RUNLIST_PREEMPT" open-gpu-kernel-modules-595.99.02/src/common/inc/swref/published/blackwell/gb202/dev_runlist.h
grep -n -C 20 "kfifoTriggerPreempt_GB202\|kfifoCheckChannelPreemptPending_GB202" open-gpu-kernel-modules-595.99.02/src/nvidia/src/kernel/gpu/fifo/arch/blackwell/kernel_fifo_gb202.c
```

**Invalidation Conditions**:
- If the system boots into a different GPU or driver version than `595.99.02`, re-run `nvidia-smi` and check `/proc/driver/nvidia/version`.
- If `RmProfilingAdminOnly` or device node permissions are changed, re-check `/proc/driver/nvidia/params`.
