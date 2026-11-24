# NVIDIA Blackwell (`sm_120`) 用户态透明非幂等 GPU 硬件抢占与调度系统完全技术手册

**文档版本**：`v2.2.1-sm120`（双卡 RTX 5080 `GB203` + RTX 5060 `GB206` 生产级全线闭环权威版）  
**更新日期**：2026-11-24  
**研究合作**：NVIDIA × 上海交通大学 IPADS 实验室（基于 XSched OSDI'25 框架扩展，仅用于学术研究）  
**代码基线**：GitHub `wangyue2024/sm120preempt`（`main` / Tag `v2.2.1-sm120`）  
**验证硬件**：
- **RTX 5080 (`GB203`, `sm_120`)**：84 SMs、`100 KiB` SMEM/SM、16 GiB GDDR7、Driver `580.95.05`、CUDA `12.9`、`g++-13`、PyTorch `2.11.0+cu128` / cuDNN `9.19.0` / Triton `3.6.0`（无头计算服务器）
- **RTX 5060 (`GB206`, `sm_120`)**：30 SMs、`100 KiB` SMEM/SM、8 GiB GDDR7、Driver `595.99.02`、CUDA `12.9`、`g++-13`、PyTorch `2.11.0+cu128` / cuDNN `9.19.0` / Triton `3.6.0`（Ubuntu 26.04 桌面独显环境）

---

## 目录

1. **第 1 章：研究背景、核心挑战与极简双引擎总体架构**
2. **第 2 章：Blackwell (`sm_120`) ToolsMemory、`.text.traphandler` 与代码洞（Code Cave）全解**
3. **第 3 章：SM120 128-bit SASS 编码学与 79 条手写指令全解析**
4. **第 4 章：永久 `0x500` B（80 槽位）Entry Guardian 逐槽位指令逻辑详解**
5. **第 5 章：静态 9-Cave 引擎（1,045 / 1,390 槽位）逐洞指令流与无死锁协同机制**
6. **第 6 章：数据承载、状态保存、GDDR7 VRAM 控制面与零脉冲原位恢复机制**
7. **第 7 章：BAR0 D10 硬件安全法则、自适应路由器、生产级 Shim/HAL 加固与 Context CILP**
8. **第 8 章：研发历程中的 15 项关键 Blackwell 微架构发现与根因复盘**
9. **第 9 章：RTX 5080 与 RTX 5060 双卡全方位实测结果与 `v2.0 → v2.2.1` 演进对比**
10. **第 10 章：能力边界、已知局限与 Phase 3 演进与论文路线图**

---

## 第 1 章：研究背景、核心挑战与极简双引擎总体架构

### 1.1 为什么非幂等 GPU 细粒度抢占极具挑战？

现代 AI 基础设施普遍采用单卡混部模式（如 **LLM 长上下文 Prefill 与低延迟 Token Decode 混部**、**后台 DNN 训练与在线实时推理混部**）。现有 GPU 抢占方案存在显著的“不可能三角”：

| 既有方案 | 颗粒度 | 是否依赖源码修改 | 是否支持非幂等内核 | 典型延迟与局限 |
| :--- | :---: | :---: | :---: | :--- |
| **Native CUDA Streams（流优先级）** | 非抢占式（仅 CTA 发射优先） | 否 | 是 | 已驻留 SM 的重载长内核无法被逐出，100% SM 饱和时 HP 尾延迟飙升至数毫秒~数十毫秒 |
| **XSched Level-1 (`Block`)** | 内核边界（Kernel Boundary） | 否 | 是 | 需等待当前在途长内核自然执行完毕，重载训练下 HP P50 高达 `6.16–17.55 ms`，`2 ms SLO = 0%` |
| **XSched Level-2 / Mode 1（幂等丢弃重放）** | 内核内早退（CTA 边界） | 否 | **否（会静默损坏！）** | 丢弃已执行部分工作并重算：在非幂等残差累加（`beta=1.0`）、原地 `AdamW`、`cuFFT`、`atomicAdd` 上造成 **+21.3% 重复计数或 `max_err = 0.5–2.0` 严重数值损坏**，同时浪费 `10%–27%` GPU 算力 |
| **硬件 Context CILP（上下文切换）** | 整个 `CUcontext` / 进程 | 否 | 是 | 无法隔离**同一进程 / 同一 `CUcontext` 内的不同 CUDA Stream**（如 PyTorch 单进程内的高优推理流与低优训练流），且切换 P50 约 `420 µs` |
| **编译器插桩 / 源码改写方案** | 指令/循环级 | **是（需重编译源码）** | 部分支持 | 无法应用于闭源商业库（`cuBLAS`、`cuBLASLt`、`cuDNN`、`cuFFT`、`cuSPARSE`）与黑盒 PyTorch/Triton 二进制 |
| **本文 Blackwell SM120 Level-3 v2 (`Mode 5 kTrapFrame`)** | **单 Stream 内任意 SASS 指令级** | **否（纯用户态透明）** | **100% 支持（比特级零误差）** | 零修改闭源二进制、单上下文双流精确隔离；Evict **`48–55 µs`**、0 脉冲 Restore **`30.5–73.9 µs`**、Entry Guardian **`< 1 µs`** |

### 1.2 从 v1 探索原型到 v2 极简通用架构的演进

在早期 v1 探索阶段，系统曾并存 Mode 1/2/3/4/5、`cmd=7`、运行时 `func_pc` 指令热补丁、高达 2,000 次盲目 BAR0 轮询以及按总 Grid 大小预分配的 `4 GiB` `u_arena`。经工程重构，v2 彻底裁剪一切过渡冗余，收敛为**正交互补的双流级模式 + 独立上下文级引擎**：

| 模式 / 引擎 | 适用场景 | 硬件触发机制 | `u_arena` 显存占用 | 恢复机制 |
| :--- | :--- | :--- | :---: | :--- |
| **Mode 1 (`kEarlyExit`)** | 明确可重算的**幂等内核** | 单次 BAR0 `0x00424684` 脉冲早退 + Entry Guardian 拦截 `State C` | **0 B** | 恢复波次按 `cta[b].state` 跳过已完成 `State A`，重算 `State B/C` |
| **Mode 5 (`kTrapFrame`)** | **通用非幂等内核**（默认 Level-3 主力，零源码修改） | 单次 BAR0 `0x00424684` 脉冲触发静态 9-Cave 硬件 Trap，原位保存全部在途 `State B` 硬件状态并释放 SM | **按物理 SM 并发上限紧凑池化**（`16,384 CTAs` 缩减 **139.7×–436.9×**） | 恢复波次由 Entry Guardian 对 `State B` 主动执行用户态 **`BPT.TRAP 0x1`（0 次 Host 脉冲）** 原位恢复并继续执行 |
| **Context CILP (`ResumableSm120Context`)** | 跨进程 / 跨 `CUcontext` 隔离、全图抢占 | NVIDIA RM ioctl (`0x20801210` / `0x83de0317` / `0x83de0318` / `0xa06c0105`) | 硬件驱动上下文槽托管 | 硬件上下文切换原位恢复（与 Trap 引擎完全解耦、零干扰） |

### 1.3 全栈软件架构与核心模块映射

系统无缝集成在 XSched 的 Shim → Preempt (`AsyncXQueue`) → CUDA HAL (`CudaQueueLv3Trap`) 三层框架中，支持通过 `LD_LIBRARY_PATH`（`libcuda.so.1 -> libshimcuda.so`）对任意未修改 CUDA/PyTorch 应用零入侵透明生效：

```text
+-------------------------------------------------------------------------------+
|  未修改用户应用：PyTorch 2.11 / cuDNN 9.19 / Triton 3.6.0 / cuBLAS / cuFFT    |
+--------------------------------------+----------------------------------------+
                                       | (标准 CUDA Driver API 调用)
                                       v
+-------------------------------------------------------------------------------+
|  CUDA Shim 层 (xsched/platforms/cuda/shim/src/{intercept.cpp, shim.cpp})      |
|  - 拦截 cuLaunchKernel / cuLaunchKernelEx / cuLaunchCooperativeKernel[_ptsz]  |
|  - XMemFree_v2() 挂起感知延迟释放队列 (DEF-03 零死锁)                          |
|  - ScopedStreamContext 无上下文工作线程自动 StreamGetCtx 绑定 (DEF-06)        |
+--------------------------------------+----------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|  XSched Preempt 核心 (xsched/preempt/src/xqueue/{xqueue.cpp, async_xqueue.cpp,|
|                       launch_worker.cpp})                                     |
|  - AsyncXQueue 管理 FIFO CommandLog、LaunchWorker 异步发射线程                |
|  - XQueueManager::HasAnySuspended() 全局挂起状态感知                          |
+--------------------------------------+----------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|  Blackwell SM120 CUDA HAL Level-3 (xsched/platforms/cuda/hal/src/level3/)     |
|  +-----------------------------------+   +----------------------------------+ |
|  | CudaQueueLv3Trap (cuda_queue.cpp) |   | ResumableSm120Context            | |
|  | - 自适应内核/模式路由器           |   | (resumable_sm120.cpp)            | |
|  |   (SYNCS/Cluster/Coop/Graph->Lv1) |   | - 独立 RM ioctl Context CILP     | |
|  | - last_kernel_mtx_ 并发互斥       |   | - 0x20801210 / 0xa06c0105        | |
|  | - DirectLaunch 高优零开销通道     |   +----------------------------------+ |
|  +-----------------+-----------------+                                        |
|                    v                                                          |
|  +--------------------------------------------------------------------------+ |
|  | Sm120StreamContext (Per-Stream, sm120_trap_engine.cpp)                   | |
|  | - 100% GDDR7 VRAM 控制面: ctrl_dev_ (Header 80B + CtaEntry[65536])       | |
|  | - 高优 Copy Engine DMA 流: ctrl_op_stream_ (MemcpyHtoD/DtoHAsync_v2)     | |
|  | - 256 槽位流序布局中转: layout_stage_ (16 KiB)                           | |
|  | - 紧凑池化状态池: arena_dev_ + kernel_arena_map_[kernel_idx]             | |
|  +-----------------+-----------------------------+--------------------------+ |
|                    v                             v                            |
|  +---------------------------------+  +-------------------------------------+ |
|  | Sm120GuardianRegistry           |  | Sm120TrapEngine (Per-Context)       | |
|  | (sm120_guardian.cpp)            |  | (sm120_trap_engine.cpp)             | |
|  | - 永久 0x500 B Entry Guardian   |  | - 静态 9-Cave 引擎 (1,045/1,390 槽) | |
|  | - 静态扫描 SYNCS / CALL 指令    |  | - 1 线程 PTX sm120_cond_clear_hdr   | |
|  | - 不可变 Sm120KernelDesc (32B)  |  | - Sm120Arena (sm120_arena.cpp):     | |
|  +---------------------------------+  |   ToolsMemory 探测 + BAR0 0x424684  | |
|                                       | - sm120_sass.h (79 条 SM120 编码器) | |
|                                       +-------------------------------------+ |
+-------------------------------------------------------------------------------+
```

核心文件清单（全目标 `C++20 -fPIC -Wall -Wextra -Werror` 零警告）：

| 模块 | 头文件 / 源文件 | 核心职责 |
| :--- | :--- | :--- |
| **SASS 编码库** | `hal/include/xsched/cuda/hal/level3/sm120_sass.h` | 79 条经 `nvdisasm -b SM120` 验证的 128-bit Blackwell 指令构造器 |
| **代码洞与 BAR0** | `hal/include/.../sm120_arena.h`, `hal/src/level3/sm120_arena.cpp` | 导出表解析、`.text.traphandler` 指纹定位、11 代码洞动态发现、`+0x390` `CALL.ABS` 挂钩、D10 安全 BAR0 脉冲 |
| **Entry Guardian** | `hal/include/.../sm120_guardian.h`, `hal/src/level3/sm120_guardian.cpp` | 80 槽位（`0x500` B）入口守护代码生成、首次发射二进制前置、`SYNCS`/`CALL` 静态扫描、`Sm120KernelDesc` 管理 |
| **Trap 引擎与流上下文** | `hal/include/.../sm120_trap_engine.h`, `hal/src/level3/sm120_trap_engine.cpp` | 静态 9-Cave 指令生成与跨洞链接、GDDR7 VRAM 控制面、Copy Engine DMA、紧凑 `u_arena` 池、PTX `sm120_cond_clear_hdr` |
| **HAL 队列与路由** | `hal/include/.../cuda_queue.h`, `hal/src/level3/cuda_queue.cpp` | `CudaQueueLv3Trap` 生命周期对接、自适应内核/模式路由器、`DirectLaunch`、`last_kernel_mtx_` 并发保护 |
| **命令封装与自愈** | `hal/include/.../cuda_command.h`, `hal/src/common/cuda_command.cpp` | `CudaCooperativeKernelLaunchCommand`、`EnsureDynamicSmem` 竞态自愈、`LaunchEx` 属性与参数深拷贝 |
| **CUDA Shim** | `shim/include/.../shim.h`, `shim/src/{intercept.cpp, shim.cpp}` | 驱动 API 拦截、`XMemFree_v2` 延迟释放队列、`ScopedStreamContext` 上下文回退 |
| **Context CILP** | `hal/include/.../resumable_sm120.h`, `hal/src/level3/resumable_sm120.cpp` | 独立 RM ioctl 上下文级 CILP 抢占模块与 RM 句柄捕获（`sm120_rm_handles.cpp`） |

---

## 第 2 章：Blackwell (`sm_120`) ToolsMemory、`.text.traphandler` 与代码洞（Code Cave）全解

Blackwell 架构上，每当硬件触发调试/软件 Trap（无论是外部 BAR0 `STOP_TRIGGER` 还是用户态 `BPT.TRAP 0x1`），硬件会将当前 Warp 切换至特权 Trap 模式，保存 112 字节硬件 Trap Frame 至线程局部内存顶端 `local[0xffff90..0xfffffc]`，并跳转至当前 `CUcontext` 专属的 **`.text.traphandler`** 代码段执行。

### 2.1 动态定位 `.text.traphandler` 与跨驱动偏移自适应

不同版本 NVIDIA 驱动的 ToolsMemory 内部布局存在显著偏移漂移——例如 **Driver `580.95.05` (RTX 5080)** 与 **Driver `595.99.02` (RTX 5060)** 之间，`.text.traphandler` 整体相对 `arena_base_va` 发生了 **`+0x5900` 字节漂移**。因此 `Sm120Arena::Discover()`（`sm120_arena.cpp`）彻底抛弃硬编码偏移，采用四步动态自发现算法：

1. **查询 CUDA 内部导出表**：
   - 通过 `cuGetExportTable` 获取两个内部表：
     - Tools 表 UUID：`cc529e94-5c4e-9d46-837c-9625868334e4`（调用 `tt[20](ctx, &ti)` 取得当前上下文的 ToolsMemory 句柄 `ti.handle`）；
     - Memory 表 UUID：`bfdb432d-bf3c-5a4a-945e-b34029e81e75`（调用 `mt[24](ctx, ti.handle, &tools_pc)` 取得 `tools_pc` 设备虚拟地址，调用 `mt[25](ti.handle, &tools_sz)` 取得对象大小，验证 `tools_sz >= 0x1300`）。
2. **推导 `arena_base_va` 与 `handler_va`**：
   - 经高优非阻塞 `op_stream_` 读出 `tools_pc` 镜像；从偏移 `+0x130` 与 `+0x140` 两条立即数指令提取 64-bit `trap_reason_tbl`，计算 `arena_base_va = trap_reason_tbl - 0xc38300ULL`。
   - 利用跨驱动恒等不变量 **`tools_pc - handler_va == 0x4400ULL`** 优先计算 `handler_va = tools_pc - 0x4400ULL`（若失败再回退 `arena_base_va + 0x15f0f00ULL`）。
3. **10 指令二进制指纹严格校验**：
   - 校验 `handler_va` 起始 10 条指令的低 32 位机器码（无重定位字段）与 `kTrapHandlerFp[10]` 逐字匹配：
     `{0x0000795c, 0x1a007956, 0xff007387, 0xff007387, 0xff007387, 0x00047202, 0x00057202, 0xff067803, 0x17077355, 0xff007387}`。
4. **唯一 Hook 挂钩点（`handler_va + 0x390`）验证**：
   - 在官方 `.text.traphandler` 中，偏移 `+0x380` 处为原生 `CALL.ABS`（低 16 位 `0x7943`），紧邻其后的 **`+0x390`** 处为一条固定的指令（`w0 = 0x000000ff00037202`, `w1 = 0x003ff00000000f00`）。
   - 安装静态 9-Cave 引擎时，仅需将 `+0x390` 这一条指令替换为指向 **Cave 0** 的 `CALL.ABS`（使用 `EncodeCallAbs(cave0_va, call_380_, patched)` 以 `+0x380` 指令为模板编码目标地址 `target_va >> 2`）。析构或卸载时原样恢复备份的 `orig_390_[2]`，保证零残留。

> **官方 `.text.traphandler` Hook 点入口与返回 ABI 契约**：
> - **进入 Cave 0 时的寄存器状态**：
>   - `R2`：Trap 原因位掩码（`R2 & 0x11 != 0` 表示硬件上下文切换 CILP/GfxP，**必须直接放行**；`R2` Bit 3 为调试/软件 Trap 标志，托管路径清除 `R2 &= ~0x8`）；
>   - `R12:R13`：64-bit 链接返回地址（指向 `handler_va + 0x3a0`，严禁在任意 Cave 中破坏！）；
>   - 线程局部内存 `local[0xffff90..0xfffffc]`：已由官方序言保存完整的 112 字节硬件 Trap Frame（包含 `TRAP_RETURN_PC`、原用户 `R0..R13`、谓词 `P0..P6`、`B0`、活跃掩码等）。
> - **返回官方 `.text.traphandler` 尾声的契约（`c0_done_ctxsw_idx`）**：
>   - 托管路径返回前将 `R2 = 0`（通知官方尾声无须继续上报调试器）；
>   - 设置 `R1 = 0xffff80`、`R3 = 0`、`R12 += 0x10`（跳过紧随 `+0x390` 的默认分发跳转），执行 `RET.ABS R12` 返回官方尾声恢复 `local[0xffff90..0xfffffc]` 并执行硬件 `RTT`（Return From Trap）。

### 2.2 什么是代码洞（Code Cave）？为什么必须使用代码洞？

在 Blackwell 上，硬件 Trap Handler 的指令取指受硬件安全/内存属性约束，只能直接跳转执行位于 **ToolsMemory 区域内** 的代码对象（普通的 `cuMemAlloc` 显存不可作为 Trap Handler 代码直接执行）。而官方 `.text.traphandler` 自身的主函数体十分紧凑，但 NVIDIA 驱动在 `[handler_va, handler_va + 0x8200)` 窗口内为每个子工具函数分配了按 `0x100`–`0x1300` 对齐的独立 `CUtoolsMemObjHandle_st` 内存对象——**每个子对象末尾在结束指令 `BRA self`（机器码 `w0 = 0xfffffffc00fc7947ULL`）之后，都留有大量未使用的 `0x900–0x9f0` 字节对齐填充区（约 144–159 条 128-bit 指令槽位）**，这就是“代码洞（Code Cave）”。

`Sm120Arena::Discover()` 通过 `cuxtra::cuda::FindMemObjHandle()` 遍历 `[handler_va, handler_va + 0x8200)` 内的每一个子对象，读取其二进制镜像并定位首个 `BRA self`（`0xfffffffc00fc7947`），将 `bra_off + 0x10` 作为该洞的起始地址（对 `handler_va` 所在首对象额外保底 `start_off >= 0x860`，以保护官方尾声 `+0x830..+0x850` 安全区），由此自动发掘全部可用代码洞。

### 2.3 双卡实测代码洞分布全表（RTX 5080 vs RTX 5060）

| Cave 序号 | RTX 5080 (`580.95.05`) `obj_sz` / 可用槽位 | RTX 5060 (`595.99.02`) `obj_sz` / 可用槽位 | 静态 9-Cave 分配映射（`SelectWorkingCaves(9, 144)`） |
| :---: | :---: | :---: | :--- |
| `cave[0]` (`handler_va` 首对象) | `0x1200` / **154 槽** (`2,464 B`) | `0x1200` / **154 槽** (`2,464 B`) | **Cave 1**（Restore 主控 + 分发器 + `clean_exit` + `exit_stub`，占 `119/154`） |
| `cave[1]` | `0x1200` / **153 槽** (`2,448 B`) | `0x1200` / **153 槽** (`2,448 B`) | **Cave 2**（`R14..R253` GPR 保存级联，占 `136/153`） |
| `cave[2]` | `0x0b00` / **159 槽** (`2,544 B`) | `0x0b00` / **159 槽** (`2,544 B`) | **Cave 3**（0~15 号命名屏障 + `B2R.WARP` 保存/清理/恢复，占 `99/159`） |
| `cave[3]` | `0x0a00` / **158 槽** (`2,528 B`) | `0x0a00` / **158 槽** (`2,528 B`) | **Cave 4**（`UR0..UR63 + UPR + B0..B15` 保存，占 `113/158`） |
| `cave[4]` | `0x0b00` / **156 槽** (`2,496 B`) | `0x0b00` / **156 槽** (`2,496 B`) | **Cave 5**（`R14..R253` GPR 恢复级联，占 `136/156`） |
| `cave[5]` (`tools_pc` 对象) | `0x1300` / **152 槽** (`2,432 B`) | `0x1300` / **152 槽** (`2,432 B`) | **Cave 0**（Trap 入口过滤 + 五区间边界 + Mode 1/5 Evict 主体，占 `139/152`） |
| `cave[6]` | `0x0b00` / **158 槽** (`2,528 B`) | `0x0b00` / **158 槽** (`2,528 B`) | **Cave 6**（`UR0..UR63 + UPR + B1..B15` 恢复，占 `112/158`） |
| `cave[7]` | `0x0a00` / **144 槽** (`2,304 B`) | `0x0a00` / **144 槽** (`2,304 B`) | **Cave 7**（动态栈溢出 + `__noinline__` 递归 LMEM 保存/恢复，占 `78/144`） |
| `cave[8]` | `0x0b00` / **156 槽** (`2,496 B`) | `0x0b00` / **156 槽** (`2,496 B`) | **Cave 8**（动态 `SR_SMEMSZ` SMEM 环路 + 112B 硬件 Trap Frame，占 `113/156`） |
| `cave[9]`（备用余量） | `0x0a00` / **154 槽** (`2,464 B`) | `0x0a00` / **154 槽** (`2,464 B`) | 未使用冗余储备 |
| `cave[10]`（备用余量） | `0x1000` / **159 槽** (`2,544 B`) | `0x2b00` / **151 槽** (`2,416 B`) | 未使用冗余储备 |
| **总计** | **11 Caves / 1,703 槽位** | **11 Caves / 1,695 槽位** | **静态 9-Cave 共占 `1,045 / 1,390` 槽位（75.2%），每洞余量 ≥ 13 槽！** |

> **关键设计优点**：全部 9 个工作代码洞在 `Sm120TrapEngine::Init()` 时通过**两遍汇编（Pass 1 构建 Cave 0/1 骨架导出跨洞返回地址 → Pass 2 构建 Cave 2..8 并回填精确 64-bit 相对 `BRA` 位移）**一次性静态写入，进程运行期间 **0 次指令热补丁、0 次 I-Cache 刷新**，天然支持同一 `CUcontext` 内多条异构流并发抢占。

---

## 第 3 章：SM120 128-bit SASS 编码学与 79 条手写指令全解析

Blackwell (`sm_120`) 采用固定 **128-bit（16 字节）指令字** `struct Ins { uint64_t w0; uint64_t w1; }`。由于 NVIDIA 未公开 SM120 SASS 汇编器，本项目通过系统逆向与 `nvdisasm -b SM120` 逐条交叉比对，在 `sm120_sass.h` 中实现了 **79 条 100% 经过 `sm120_sass_disasm` 反汇编验证** 的无依赖 `constexpr` 编码原语。

### 3.1 128-bit SM120 指令字通用位域结构

```text
w0 (64-bit, 低 8 字节):
  [11: 0]  主操作码 (Opcode, 低 9 位 op9 = w0 & 0x1ff 标识指令大类)
  [15:12]  执行谓词 (Predicate: 0..6 = P0..P6, 7 = PT 恒真, 8..14 = !P0..!P6, 15 = !PT)
  [23:16]  目的寄存器 Rd (0..253 为通用寄存器, 0xff = RZ 零寄存器)
  [31:24]  源寄存器 Ra (8-bit)
  [63:32]  源寄存器 Rb / 32-bit 立即数 / 常量银行偏移 / 内存偏移低位 / 分支位移高位

w1 (64-bit, 高 8 字节):
  [31: 0]  扩展修饰符 (缓存一致性 .E/.STRONG/.CG、比较条件 .EQ/.NE/.LT/.GE、寻址模式、分支位移最高位)
  [40:32]  寄存器/谓词目的扩展字段 (如 ISETP 目的谓词 Pd、源谓词 Ps、BAR/UR 编号)
  [57:52]  控制字 Wait Barrier 掩码 (req_mask: 6-bit 位图, 等待计分板 0..5 完成)
  [63:41]  调度控制字 (Stall count、Yield 标志、Read/Write Barrier 分配，保守无冒险控制字如 0x000fda00... / 0x003fda00...)
```

### 3.2 关键硬件编码陷阱与特殊寄存器索引

开发过程中实测验证了多项 Blackwell 特有编码规律（均已固化至 `sm120_sass.h` 并由 `sm120_sass_disasm` 回归守护）：

1. **特殊寄存器（`S2R`）索引表**：
   - `SR_LANEID = 0x00`，`SR_TID.X/Y/Z = 0x21/0x22/0x23`，`SR_CTAID.X/Y/Z = 0x25/0x26/0x27`；
   - **`SR_SMEMSZ = 0x32`**：返回硬件为当前 CTA 分配的共享内存总字节数（含驱动保留的 `0x400` B，按 4 KiB 量子对齐）；
   - **`SR_REGALLOC = 0x3d`**：返回硬件为当前 Warp 每线程实际分配的寄存器槽总数（8 寄存器对齐，最高 `256`；其中 `R254:R255` 为硬件 Trap 专用寄存器，用户可见最高为 `R253`）；
   - `SR_EQMASK = 0x38`、`SR_GLOBAL_ERROR_STATUS = 0x40`、`SR_WARP_ERROR_STATUS = 0x42`。
2. **相对分支 `BRA` 分段位移编码**：
   - 位移单位为 `off = n_instrs * 4`（每条 16 字节指令占 4 个 4 字节步长），**分散打包在三处位域**：`lo8 = off[7:0]` 置于 `w0[23:16]`，`mid30 = off[37:8]` 置于 `w0[63:34]`，`hi18 = off[55:38]` 置于 `w1[17:0]`。跨代码洞跳转 `bra_to(from_va, to_va)` 统一计算 `n_instrs = (to_va - (from_va + 16)) >> 4`。
3. **`BRX Rn` 消费 64-bit 寄存器对 `Rn:Rn+1`**：
   - SM120 的间接跳转 `BRX R10`（`w0 = (10 << 24) | 0x7949`）会将 `R10:R11` 拼接为 64-bit 有符号字节偏移！若此前指令在 `R11` 残留了 `0xffffffff`（如恢复 112B 硬件帧后），会造成 `-4 GiB` 非法负跳。因此 **Cave 2 / Cave 5 在 `BRX R10` 前必须显式执行 `mov_imm(11, 0)`**。
4. **`BAR.SYNC.DEFER_BLOCKING 0x0` vs `BAR.SYNC.TRAP`**：
   - 用户态 Entry Guardian 使用 `bar_sync0()`（`w0=0x7b1d, w1=0x000fec0000010000`），同步普通 0 号 CTA 屏障；
   - Trap Handler 内绝不能污染用户命名屏障 `0..15`，必须使用硬件专属 Trap 屏障 `bar_sync_trap()`（`w0=0x7b1d, w1=0x000fc00000008000`）。**重要硬件约束**：在 SM120 上，只要同 CTA 内任意兄弟 Warp 已经执行 `EXIT` 退出，其余 Warp 调用 `bar_sync_trap()` 就会永久挂死（这也是 Cave 0 五区间无死锁协同算法的根本动因）。
5. **硬件 `mbarrier`（`SYNCS`）静态识别编码**：
   - 在 `sm_120` 上，PTX `mbarrier.init / arrive / try_wait` 编译为 `SYNCS.ARRIVE / PHASECHK / CCTL`（`op9 = w0 & 0x1ff == 0x1a7`）与 `SYNCS.EXCH.64`（`op9 == 0x1b2`）。经核验，这两条 `op9` 编码在 SM120 指令集中 100% 专属于硬件 `SYNCS` 指令，无任何误报。

### 3.3 79 条手写 SASS 指令分类总览

| 功能类别 | 代表指令（`sm120_sass.h`） | 核心用途 |
| :--- | :--- | :--- |
| **控制流与陷阱生命期** | `NOP`, `EXIT`, `BPT.TRAP 0x1`, `RET.ABS R12`, `BRA`, `BRX`, `BAR.SYNC 0`, `BAR.SYNC.TRAP` | Entry Guardian 跳转、0 脉冲用户态自陷、Cave 2/5 动态寄存器级联跳转表、Trap 屏障同步与返回官方尾声 |
| **通用算术与 64-bit 寻址** | `MOV`, `IADD`, `IADD3`, `IADD.64`, `IMAD`, `IMAD.WIDE.U32`, `LEA` (左移), `SHF.R.U32`, `SHF.L` | 3D `tid`/`cta_id` 线性化、`u_arena` 64-bit 槽位基址计算、动态 SMEM/LMEM 步长推进、`pc_off` 64-bit 差值计算 |
| **位运算与谓词逻辑** | `LOP3` (OR3 / AND / TST), `ISETP.{EQ,NE,LT,GE}.{U32,U64}` (含 `AND P0` 组合谓词) | 零分支 `tid==0 && cmd!=0` 判定、`warp_mask` 位测试、五区间无符号 64-bit `pc_off` 边界判定 |
| **全局/常量/共享/局部访存** | `LDG.E.{32,64,128}`, `STG.E.{32,64,128}`, `LDC.{32,64}`, `LDS.U32`, `STS.U32`, `LDL.{32,64,128}`, `STL.{32,64,128}` | 读取 `c[0x0][0x170..0x187]` 与 `0x360..0x37c`、GDDR7 VRAM 控制面与 `u_arena` 读写、SMEM/LMEM 与 112B 硬件帧读写 |
| **原子与 Warp 选举** | `ELECT P0`, `ATOMG.E.ADD.U32`, `ATOMG.E.OR.U32`, `REDG.E.ADD.U32` | 动态 CTA Leader 选举（首个存活 Warp 胜出）、紧凑 `slot_id` 原子申请、`warp_mask` 位图合并、`trap_inflight`/`restored_slots` 计数 |
| **屏障与均匀寄存器状态** | `B2R.BAR`, `R2B.BAR`, `B2R.WARP`, `R2B.WARP`, `MOV R, UR`, `R2UR`, `UP2UR UR7`, `UR2UP UR7`, `BMOV.CLEAR`, `BMOV` | 完整保存/清理/恢复 16 个 CTA 命名屏障、每 Warp 屏障参与态、64 个均匀寄存器 `UR0..UR63`、均匀谓词 `UPR` 与 16 个收敛屏障 `B0..B15` |

---

## 第 4 章：永久 `0x500` B（80 槽位）Entry Guardian 逐槽位指令逻辑详解

### 4.1 首次发射一次性插桩与 `DebuggerParams` 常量窗口

在 `Sm120GuardianRegistry::EnsureInstrumented(CUfunction func)`（`sm120_guardian.cpp`）中，每个目标内核仅在**进程生命期内首次发射前**完成以下操作：
1. 调用 `cuXtraGetBinary()` 提取原始 `sm_120` SASS 镜像；
2. 静态扫描 `BinaryHasHwMbarrier()`（`op9 == 0x1a7 / 0x1b2`）与 `BinaryHasDeviceCalls()`（`CALL.REL/ABS`）：若含硬件 `mbarrier`，标记 `has_hw_mbarrier = true` 直接返回（交由自适应路由器降级 Level-1，零改动二进制）；
3. 通过 `InstrMemAllocator` 分配 128 字节对齐的可执行显存 `ep_inst`（大小 `0x500 + orig_bin_sz`），将固定的 **80 槽位（`0x500` B）Entry Guardian** 拷贝至 `ep_inst + 0x000`，将**原内核完整二进制原封不动**拷贝至 `ep_inst + 0x500`；
4. 调用 `cuXtraSetEntryPoint(func, ep_inst)`，按需将寄存器数钳位至 `>= 16`、屏障数钳位至 `>= 1`，并刷新指令缓存 `cuXtraInvalInstrCache(ctx)`；
5. 在设备 VRAM 分配 32 字节不可变 `Sm120KernelDesc`（记录 `ep_inst_va`, `kGuardianBarOff=0x270`, `lmem_static_bytes`, `lmem_cap_bytes`, `bar_cnt`, `flags`, `magic=0x534d3132`）。

每次发射该内核前，`PrepareLaunch()` 通过 `cuXtraSetDebuggerParams(func, &dbg, 24)` 将 24 字节运行时上下文写入 `c[0x0][0x170..0x187]`（**不修改指令本身，无需再刷 I-Cache**）：

| 常量银行偏移 | 字段名 | 大小 | 说明 |
| :---: | :--- | :---: | :--- |
| `c[0x0][0x170..0x177]` | `u_control_va` | 8 B | 本流 GDDR7 VRAM `Sm120ControlHeader` 虚拟地址；**为 `0` 表示未托管 / `DirectLaunch` 高优零开销快通** |
| `c[0x0][0x178..0x17f]` | `kdesc_va` | 8 B | 本函数不可变 `Sm120KernelDesc` 设备虚拟地址 |
| `c[0x0][0x180..0x187]` | `kernel_idx` | 8 B | XSched 单调递增命令序号 `HwCommand::GetIdx()`，用于 FIFO 多内核精确隔离 |
| `c[0x0][0x360..0x36b]` | `blockDim.{x,y,z}` | 12 B | CUDA 驱动标准内核网格常量（用于 3D `tid` 与 `threads_per_cta` 计算） |
| `c[0x0][0x370..0x377]` | `gridDim.{x,y}` | 8 B | CUDA 驱动标准内核网格常量（用于 3D `cta_id` 线性化） |
| `c[0x0][0x37c..0x37f]` | `stack_top` | 4 B | 线程局部内存软件栈顶（通常 `0xfffdc0`，供 Cave 7 LMEM 区间计算） |

### 4.2 80 槽位（`0x000..0x4f0`）逐区间执行逻辑

Entry Guardian 被精确划分为 **5 个硬件地址区间**（与 Cave 0 的五区间 `pc_off` 判定一一对应，`orig_kernel` 恒位于 `+0x500`）：

| 槽位范围 | 字节偏移 | 区间名称 | 逐指令核心逻辑与并发不变量 |
| :---: | :---: | :--- | :--- |
| **Slots 0–3** | `0x000..0x030` | **未托管快通门** | 读 `R4:R5 = c[0][0x170..0x177]`；`lop3_or3_p0(4, 5, RZ)` 判断 `u_control_va == 0` 时直接 `BRA +0x500` 进入原内核。**高优流 `DirectLaunch` 清零 `DebuggerParams` 后仅多 4 条指令（`< 20 ns`）**。 |
| **Slots 4–11** | `0x040..0x0b0` | **3D 坐标线性化** | 读取 `SR_CTAID.{Z,Y,X}` 与 `gridDim.{Y,X}`，计算 `R3 = (ctaZ * gridY + ctaY) * gridX + ctaX`；再以 `IMAD.WIDE.U32` 计算 `R6:R7 = u_control_va + 16 * cta_id`（指向 `cta[b]`，注意访存时加偏移 `0x50` 即为 `Sm120ControlHeader` 之后的 `Sm120CtaEntry[b]`）。 |
| **Slots 12–14** | `0x0c0..0x0e0` | **波次不变路由门** | 读取 `R1 = u_ctrl->is_restore_wave (+0x1c)`。若 `!= 0`，**全体 Warp 无条件一致跳转至 `restore_path` (`+0x2c0`)**！因为 `is_restore_wave` 在整个恢复波次内绝不被 `Preempt #2` 清除，从根源上消灭了 `cmd: 6 -> 5` 瞬间同一 CTA 内部分 Warp 走 `restore_path`、部分走 `norm_gate` 的共享 `BAR.SYNC 0` 死锁（FIX-5）。 |
| **Slots 15–23** | `0x0f0..0x170` | **正常/驱逐 FIFO 隔离门** | 读 `cmd (+0x00)`；若 `cmd == 0` 跳 `norm_gate`；若 `cmd != 0`，读 `active_kernel_idx (+0x08)`，经单指令 `lop3_or3_p0(10, 11, RZ)` 判断 `active != 0` 且 `my_idx != active` 时立即 `@P0 EXIT`——确保后续排队内核 `K2` 绝不污染正在被抢占的 `K1` 的 `cta[]` 状态表。 |
| **Slots 24–39** | `0x180..0x270` | **`norm_gate` 无分支共识 + Slot 39 `BAR.SYNC 0`** | 全程无分支：算 3D `tid`，`P0 = (tid == 0 && cmd != 0)`；若 `cmd == 1` 置 `R2 = 1`（Mode 1 重放），否则置 `R2 = 2`（Mode 5 `State C`）；`@P0` 写入 `cta[b].state = R2`、`cta[b].entry_dec = R2`、**`u_ctrl->active_kernel_idx = my_idx`（`v2.2.1` 补齐纯 Drain 内核归属）** 并 `drained_cnt++`；**Slot 39 (`0x270 = kGuardianBarOff`) 执行 `BAR.SYNC 0`** 达成全 CTA 一致性栅栏。 |
| **Slots 40–43** | `0x280..0x2b0` | **Post-`BAR.SYNC 0` 放行/早退** | 所有 Warp 统一读 `cta[b].entry_dec (+0x54)`：非零则 `EXIT` 干净早退（零用户代码执行，无泄漏 `State C`），为零则 `BRA +0x500` 执行原内核正文。 |
| **Slots 44–58** | `0x2c0..0x3a0` | **`restore_path` 三态派发** | ① 多内核校验：若 `active != 0 && my_idx != active` 则 `EXIT`；② 读 `R3 = cta[b].state (+0x50)`：若 `state == 0`（**State A 已完成**）立即 `EXIT`；若 `state == 5`（**State B 已换出**）在 **Slot 57 执行用户态 `BPT.TRAP 0x1`** 自陷进入 Cave 1 原位恢复（0 Host 脉冲！）；否则（**State C `state == 1/2`**）保留 `R3` 跳转至 `replay_state_c` (`+0x3b0`)。 |
| **Slots 59–79** | `0x3b0..0x4f0` | **`replay_state_c` 奇偶二次抢占共识** | 先执行 `BAR.SYNC 0`（Slot 59）确保同 CTA 全部 Warp 已读完 `state`；`tid == 0` 读取 `cmd` 并取最低位 **`cmd & 1`**（**协议巧妙之处：所有 Evict 命令 `1/5` 为奇数，所有 Restore 命令 `2/6` 为偶数！**）：若为偶数（正常恢复）则清零 `cta[b].state = 0, entry_dec = 0`；若为奇数（恢复波次中途遭遇 `Preempt #2`！）则保留 `state`、置 `entry_dec = 2` 并 `drained_cnt++`；**Slot 76 (`0x4c0 = kGuardianReplayBarOff`) 执行共识 `BAR.SYNC 0`**，Slots 77–79 统一读 `entry_dec != 0` 则一致 `EXIT`，否则落入 `+0x500` 首次执行原内核。 |

---

## 第 5 章：静态 9-Cave 引擎（1,045 / 1,390 槽位）逐洞指令流与无死锁协同机制

### 5.1 线程局部内存（Local Memory）硬件帧与草稿区精确映射

当任意 Warp 陷入 Trap Handler 后，线程局部内存顶端 `0xffff6c..0xfffffc` 的完整布局如下（**严格只读/只写规定区段，绝不侵入用户栈 `<= stack_top`！**）：

| Local 偏移 | 大小 | 归属 | 存储内容与生命周期 |
| :---: | :---: | :---: | :--- |
| `0xffff6c` | 4 B | 引擎草稿 | Evict 阶段存 **`is_leader`**（动态选举出的 CTA Leader 标志）；Restore 阶段存 **`is_live_warp`**（该 Warp 在 Evict 时是否存活） |
| `0xffff70` | 4 B | 引擎草稿 | 3D 线性化后的 `cta_id`（`stl_128` 一次性打包写入 `0xffff70..0xffff7f`） |
| `0xffff74` | 4 B | 引擎草稿 | 3D 线性化后的线程号 `tid` |
| `0xffff78` | 4 B | 引擎草稿 | 3D 线程块总线程数 `threads_per_cta` |
| `0xffff7c` | 4 B | 引擎草稿 | Mode 5 State B 分配/读取的紧凑 **`slot_id`** |
| `0xffff80..0xffff8f` | 16 B | 官方保留 | 官方 `.text.traphandler` 尾声栈指针基准（返回前必置 `R1 = 0xffff80`） |
| **`0xffff90..0xfffffc`** | **112 B** | **硬件 Trap Frame** | 官方序言自动保存的 7×16B 硬件上下文：含 **`[0xffffa8]` 64-bit `TRAP_RETURN_PC`**、**`[0xffffbc]` 收敛屏障 `B0`**、**`[0xffffc4]` 用户栈指针 `user R1`**、原用户 `R0..R13`、谓词 `P0..P6` 与 `[0xfffffc]` 活跃线程掩码 |

### 5.2 Cave 0（`139 / 152` 槽位）：入口过滤、五区间无死锁边界与 Mode 1/5 Evict 主控

1. **快速过滤与硬件 CILP 放行**：
   - 检查 `R2 & 0x11`：若非零（硬件上下文切换 CILP/GfxP），**立即跳转 `c0_done_ctxsw_idx` 原样放行**（确保 Trap 引擎与硬件 CILP 零冲突）；
   - 清除 `R2` Bit 3（`R2 &= ~0x8`）；读 `R4:R5 = u_control_va`，若为 `0`（未托管流）放行；
   - 读 `cmd = u_ctrl->cmd`，若为 `0` 则清 `R2 = 0` 走 NO-OP RTT 返回；若 `cmd != 0`，**统一跳转至 Cave 1 的重入防护分发器 `c1_dispatch_guard_idx`**。
2. **多内核 FIFO 归属认领（`coords_idx`）**：
   - 计算 3D `{cta_id, tid, threads_per_cta}` 存入 `local[0xffff70..0xffff7f]`；
   - 读取 `my_idx = c[0][0x180]` 与 `active_kernel_idx`：若 `active == 0` 则由 `ELECT P0` 选出单活跃 Lane 认领 `active_kernel_idx = my_idx`；若 `my_idx != active` 则直接跳转 `c1_clean_exit_idx` 干净退出后续排队内核 `K2`。
3. **Mode 1 (`cmd == 1`) 零 `bar_sync_trap()` 极速早退**：
   - 当 `cmd == 1` 时，由于同 CTA 兄弟 Warp 可能已在 Entry Guardian Slot 42 执行 `EXIT`，绝不调用 `bar_sync_trap()`：`tid == 0 && entry_dec == 0` 原子标记 `cta[b].state = 1` 并 `mode1_exit_cnt++`，随后全 Warp 跳转 Cave 3 `clear_exit_idx` 清屏障后直接 `clean_exit`。
4. **Mode 5 (`cmd == 5`) 五区间无死锁协同边界（核心算法，FIX-6）**：
   - 计算无符号 64-bit `pc_off = TRAP_RETURN_PC (local[0xffffa8]) - kdesc->ep_inst_va`。
   - **为什么必须区分 `entry_dec == 2` 与 `entry_dec == 4`？**
     - `entry_dec == 2` 表示 **Entry Guardian 侧已裁决 Drain**：此时同 CTA 兄弟 Warp **极可能已在 Slot 42 或 Slot 79 执行 `EXIT`**，任何残留 Warp 绝不能执行 `bar_sync_trap()`，必须直接绕过同步逃逸至 `clear_bars_exit_va`；
     - `entry_dec == 4` 表示 **Trap Handler（Cave 0）侧首次裁决 Drain**：由于 `pc_off` 位于栅栏前且此前 `entry_dec != 2`，全 CTA 所有兄弟 Warp 必然都存活并已随 BAR0 脉冲同时陷入 Trap，可安全在 `bar_sync_trap()` 汇合达成全 CTA 一致的 `State C` 裁决！
   - 五个精确无符号区间如下：
     - **区间 1 `[0x000, 0x270)`（严格在 `norm_gate BAR.SYNC 0` 前）**：若 `is_restore_wave && state == 0`（State A 兄弟可能已退出）或 `entry_dec == 2` 则无同步逃逸；否则置 `entry_dec = 4` 并进入 `post_gate_sync_idx` 同步为 `State C`。
     - **区间 2 `[0x270, 0x2c0)`（含 Slot 39 `BAR.SYNC 0` 本身至 `EXIT`）**：读 `entry_dec`，若 `== 2` 则无同步逃逸；否则（`0` 或 `4`）安全进入 `post_gate_sync_idx` 同步（兄弟写了 `4` 则一致走 `State C`，均为 `0` 则一致走 `State B`）。
     - **区间 3 `[0x2c0, 0x3b0)`（`restore_path`）**：若 `is_restore_wave && state == 0` 直接逃逸保留 `State A`；否则检查 `entry_dec == 2` 逃逸或置 `4` 同步。
     - **区间 4 `[0x3b0, 0x4c0)`（`replay_state_c` 栅栏前）**：同上检查 `entry_dec == 2` 逃逸或置 `4` 同步。
     - **区间 5 `[0x4c0, 0x500)`（`replay_state_c` 栅栏及之后）**：复用区间 2 逻辑；而 `pc_off >= 0x500` 确认为原内核正文，安全汇合 `bar_sync_trap()` 进入完整 `State B` 保存。
5. **动态 CTA Leader 选举与紧凑 `slot_id` 申请（解决部分已退出 CTA）**：
   - 在真实内核中（如带早退分支的算子），`Warp 0`（`tid == 0`）可能已先执行 `@P0 EXIT` 退出，仅剩 `Warp 1..N` 仍在运行！若死板依赖 `tid == 0` 做 Leader，会导致整个 CTA 无人申请 `slot_id` 且无人保存屏障。
   - **动态选举算法**：每个存活 Warp 通过 `ELECT P0` 选出唯一活跃 Lane，执行 `R0 = ATOMG.E.OR(&cta[b].warp_mask, 1u << warp_id)`；**原子返回旧值 `R0 == 0` 的那个 Warp（第一个抢到标志的存活 Warp）胜出成为动态 CTA Leader**（`local[0xffff6c] = 1`），由该 Leader 原子申请 `slot_id = ATOMG.E.ADD(&u_ctrl->evicted_slots, 1)` 并写入 `cta[b].slot_id`。经 `bar_sync_trap()` 广播至全 CTA 并校验 `slot_id < max_slots`（防越界保护）后，依次跳跃执行状态保存链：
     **Cave 3（命名屏障）→ Cave 4（Uniform/B0..B15）→ Cave 8（SMEM）→ Cave 7（LMEM）→ Cave 8（112B 硬件帧）→ Cave 2（R14..R253 GPR）**，最后 `bar_sync_trap()` 由 Leader 原子发布 `cta[b].state = 5` 并跳转 `c1_clean_exit_idx` 释放 SM。

### 5.3 Cave 1（`119 / 154` 槽位）：重入防护分发、0 脉冲 Restore 主控与干净退出桩

1. **`c1_dispatch_guard_idx` 重入防护分发器**：
   - 任何 `cmd != 0` 的入陷 Warp 必经此处。首条检查 **`local[0xffffa8] (TRAP_RETURN_PC) == exit_stub_va`**——若该 Warp 已被前一次 Trap 重定向至 `exit_stub_va`（正准备在用户态执行 `EXIT`）时又被伴随 `BPT.TRAP 0x1` 或恢复脉冲二次命中，**强制走 NO-OP RTT 直接放行**，100% 杜绝重复计数或重入 `bar_sync_trap()`。
   - 随后分发：`cmd == 6` → Cave 1 Restore 主入口，`cmd == 1/5` → Cave 0 `coords_idx`，其余 → NO-OP RTT。
2. **Mode 5 Restore 主控（0 Host 脉冲，由 Entry Guardian Slot 57 `BPT.TRAP 0x1` 触发）**：
   - **原子在途追踪**：入口 `ELECT P0` 执行 `ATOMG.E.ADD(&u_ctrl->trap_inflight, +1)`，所有退出路径（正常恢复、死 Warp 退出、伴随入陷 NO-OP）返回前原子 `-1`；确保主机 `PreemptStream` 绝不会在还有 Warp 驻留 Cave 1 时发射下一轮 BAR0 脉冲。
   - **同 SM 伴随入陷（Collateral Trap）过滤与重检**：`BPT.TRAP 0x1` 是 SM 级微架构事件，同一 SM 上驻留的其他 CTA（`State A`/`State C`/已恢复的 `State B`/另一内核）会被伴随拉入 Trap。Cave 1 检查 `my_idx == active_kernel_idx && cta[b].state == 5`，非 `State B` 的伴随 Warp 转入 `c1_noop_recheck_idx`：在 `trap_inflight--` 前重读 `cmd`，若 Host 已切换为 `cmd == 5/1` 则直接无缝派发至 Cave 0 Evict（避免该 Warp 逃回用户态错过单脉冲！），否则安全 NO-OP RTT。
   - **部分已退出 CTA 精确重建（`is_live_warp`）**：读取 `cta[b].warp_mask`，计算 `is_live_warp = ((warp_mask & (1u << warp_id)) != 0)` 存入 `local[0xffff6c]`。按严格逆序执行恢复链：
     **Cave 8（SMEM）→ Cave 7（LMEM，仅活 Warp）→ Cave 3（命名屏障 `0..15`）→ Cave 6（`UR0..63/UPR/B1..15`，仅活 Warp）→ Cave 8（112B 硬件帧）→ Cave 5（`R14..R253` GPR，仅活 Warp）**。
   - **最终同步与死/活分流**：预载本 Warp 的 `B2R.WARP` 状态至 `R8` → `tid == 0` 清零 `cta[b].{state, entry_dec, warp_mask}` → `bar_sync_trap()` → `tid == 0` 执行 `REDG.E.ADD(&u_ctrl->restored_slots, 1)`：
     - **活 Warp（`is_live_warp == 1`）**：执行 `R2B.WARP R8` 恢复该 Warp 的屏障参与状态（必须在最后一次 `bar_sync_trap()` 之后！），`trap_inflight--`，跳转 `c0_done_clear_r2` 经官方尾声 `RTT` **精确回到被抢占瞬间的 `TRAP_RETURN_PC` 继续执行**！
     - **死 Warp（`is_live_warp == 0`，即 Evict 前已自然 `EXIT` 的 Warp）**：`trap_inflight--`，跳转 `c1_clean_exit_idx` 干净退出，绝不错误复活已退出线程！
3. **`c1_clean_exit_idx` 与 `exit_stub_va`**：
   - 将 `local[0xffff90..0xffffbf]` 重置为安全全活掩码与清零屏障状态，将 `local[0xfffffc]` 置 `0xffffffff`，并将 **`local[0xffffa8]`（`TRAP_RETURN_PC`）改写为 `exit_stub_va`**（Cave 1 内一条永久 `EXIT` 指令的地址），返回官方尾声执行 `RTT` 后立即在用户态无副作用退出并释放 SM 资源。

### 5.4 Cave 2 & Cave 5（各 `136` 槽位）：`SR_REGALLOC + BRX` 全量 `R14..R253` 寄存器级联

- 通用寄存器 `R0..R13` 已由硬件保存在 `local[0xffff90..0xfffffc]`（112B 硬件帧，由 Cave 8 处理），剩余全部高寄存器 `R14..R253`（最多 240 个 32-bit 寄存器）由 Cave 2（保存）与 Cave 5（恢复）采用**零循环、单次 `BRX` 级联落穿表**完成：
  1. 读 `R4 = SR_REGALLOC - 3`（扣去硬件保留的 `R254:R255` 及 0-based 顶界）；若 `< 14` 直接跳过；
  2. 若 `R4 >= 253`（即 `REG:255` 内核、`SR_REGALLOC == 256`），显式以谓词 `@P0` 在 `[thread_base + 0x434]` 单独存取 `R253`；
  3. 计算 `r_start = min(R4, 252) & ~1`，计算跳转偏移 `R10 = ((252 - r_start) / 2) * 16`（`lea_shl(10, 10, 3)`），**显式清零 `R11 = 0`**（防止 SM120 64-bit `BRX R10:R11` 高位脏值），执行 `BRX R10`；
  4. 落穿表第 0 项以 32-bit 存取 `R252`（`+0x430`），第 1..119 项以 64-bit 向量 `STG.E.64` / `LDG.E.64` 依次落穿存取 `R250:R251` 直至 `R14:R15`（`+0x78 + (r-14)*4`）；
  5. 末尾重新加载 `R4:R5 = u_control_va` 保持全局 ABI 完整，跳回 Cave 0 / Cave 1。
- **性能优势**：对 32-reg 轻量内核仅执行 9 条 `STG.E.64`，对 255-reg 重型 cuBLAS/FlashAttention 内核零分支满速覆盖全部 240 个寄存器。

### 5.5 Cave 3（`99 / 159` 槽位）：0~15 号命名屏障与 `B2R.WARP`

- **Save 段 (`save_idx`)**：每个 Warp 先读取自身 `B2R.WARP` 存入 `warp_base + 0x104` → `bar_sync_trap()` → 动态 CTA Leader 经 4 组×4 条 `B2R.BAR` + `STG.E.128` 将 16 个命名屏障硬件计数/到达状态保存至 `[slot_base + 0x00..0x3f]` → 将全部 16 个命名屏障和 `R2B.WARP` 清零释放 → `bar_sync_trap()` 返回。
- **Clear+Exit 段 (`clear_exit_idx`)**：专供 Mode 1 / State C / State A 逃逸路径使用，**严禁执行 `bar_sync_trap()`**，仅清理 16 个命名屏障与 `R2B.WARP` 后直接跳转 `c1_clean_exit_idx`。
- **Restore 段 (`rest_idx`)**：`tid == 0` 从 `[slot_base + 0x00..0x3f]` 经 4 组 `LDG.E.128` + `R2B.BAR` 恢复 16 个命名屏障 → `bar_sync_trap()` 返回 Cave 1（每 Warp 各自的 `R2B.WARP` 延迟至 Cave 1 最后一次 `bar_sync_trap()` 之后恢复）。

### 5.6 Cave 4 & Cave 6（`113 / 158` & `112 / 158` 槽位）：`UR0..UR63`、`UPR` 与 `B0..B15`

- 均匀寄存器与收敛屏障属于 Warp 级共享状态，因此用 `ELECT P0` 仅选单活跃 Lane 操作，访存量减少 32×：
  - **Save（Cave 4）**：16 组×4 条 `MOV R, UR` + `STG.E.128` 保存 `UR0..UR63` 至 `warp_base + [0x000..0x0ff]`；随后安全借用已保存的 `UR7` 执行 `UP2UR UR7` 保存 `UPR` 至 `warp_base + 0x100`；读取 `local[0xffffbc]`（`B0`）及 `BMOV.CLEAR` 保存并清空 `B1..B15` 至 `warp_base + [0x110..0x14f]`。
  - **Restore（Cave 6）**：**必须严格先恢复 `UPR`**（经临时 `UR7` + `UR2UP UR7`），再恢复 `UR0..UR63`（避免覆盖 `UR7` 真实值），最后用 `BMOV` 恢复 `B1..B15`（`B0` 位于 `local[0xffffbc]`，已由 Cave 8 112B 硬件帧恢复并由官方尾声写回硬件）。

### 5.7 Cave 7（`78 / 144` 槽位）：栈溢出与 `__noinline__` 递归局部内存（LMEM）

- 支持静态栈溢出（`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES > 0`）与动态 `__noinline__` 深度递归调用栈（`BinaryHasDeviceCalls() == true`）：
  1. 若 `kdesc->lmem_cap_bytes == 0`，3 条指令直接跳过（零开销）；
  2. 读 `stack_top = c[0][0x37c]`，计算 `static_lo = stack_top - kdesc->lmem_static_bytes`；
  3. 若 `kdesc->flags & 2`（含设备函数调用），从 `local[0xffffc4]` 读取被抢占瞬间的用户栈指针 `user_R1`（向下 16 字节对齐），以全无分支谓词裁剪计算：
     $$\text{save\_lo} = \min\Big(\text{static\_lo},\ \max(\text{user\_R1} \ \&\ \sim 15,\ \text{stack\_top} - \text{lmem\_cap})\Big)$$
  4. 将实际 `save_lo` 存入 `[thread_base + 0x70]`，以 128-bit 向量循环（`LDL.128` / `STG.E.128`）仅拷贝 `[save_lo, stack_top)` 有效栈区至 `lmem_base`；Restore 时读取 `[thread_base + 0x70]` 精确原位写回。全程严格保护 `R12:R13` 链接寄存器。

### 5.8 Cave 8（`113 / 156` 槽位）：动态 `SR_SMEMSZ`（含 Partial-CTA 回退）与 112B 硬件帧

1. **动态共享内存环路（`EmitDynamicSmemLoop`）**：
   - 读取 `R8 = SR_SMEMSZ`；**先执行 `ISETP.GE.U32 R8, 0x401` 防护**（FIX-3：防止 0-SMEM 内核 `SR_SMEMSZ == 0` 或 `0x400` 减 `0x400` 无符号下溢为 `0xFFFFFC00` 引发 `cuda error 717`），再计算用户有效字节数 `R8 = SR_SMEMSZ - 0x400`。
   - **Partial-CTA 32-lane 合并回退**：保存阶段比对 `cta[b].warp_mask == full_mask`（全员存活时全 CTA 并行拷贝步长 `threads_per_cta * 4`）；若部分 Warp 已提前退出，则通过 `R2 = warp_mask & (-warp_mask)` 选出**最低编号存活 Warp**，以 `lane_id * 4` 起始、`32 * 4 = 128 B` 合并步长独自完整保存整块用户共享内存 `[0x400, SR_SMEMSZ)`，杜绝共享内存空洞！
2. **112B 硬件 Trap Frame 存取与尾链跳转**：
   - 7 次 `LDL.128 / STG.E.128` 将 `local[0xffff90..0xfffffc]` 完整保存至 `[thread_base + 0x00..0x6f]`（Restore 反向写回），随后直接尾链跳转至 Cave 2（保存 `R14..R253`）或 Cave 5（恢复 `R14..R253`），省去一次往返跳转。

---

## 第 6 章：数据承载、状态保存、GDDR7 VRAM 控制面与零脉冲原位恢复机制

### 6.1 100% GDDR7 显存控制面（`v2.2.0` 核心架构飞跃）

在 `v2.0/v2.1` 阶段，`Sm120ControlHeader` 与 `Sm120CtaEntry[65536]` 位于 `CU_MEMHOSTALLOC_DEVICEMAP` 固定宿主内存（PCIe Zero-Copy），导致 Entry Guardian 每 Warp 访问均触发跨 PCIe Gen5 总线的未缓存往返（`~1.6–2.0 µs/op`），在大网格 `cuBLAS SGEMM`（4.3× 膨胀）与未融合 PyTorch（`LP_ret = 0.7%`）上形成严重瓶颈（`DEF-01`）。

**`v2.2.0/v2.2.1` 彻底重构为 100% GDDR7 显存控制面 + 独立高优先级 Copy Engine DMA 通道**：
1. **`ctrl_dev_` 全量驻留 GPU GDDR7 VRAM**：
   - `Sm120ControlHeader`（80 B）+ `Sm120CtaEntry[65536]`（`1 MiB`，动态可扩容，扩容前同步 `last_stream_` 并将旧块存入 `retired_ctrls_` 防止在途引用悬垂）全部分配在设备显存，享受 GPU L2 Cache 与 384–960 GB/s GDDR7 带宽；
   - 结果：Entry Guardian 空载开销骤降至 **`0.43–0.99 µs`**（提速 **31×–100×**），Mode 5 Evict/Restore 降至 **`30.5–55.4 µs`**，PyTorch `PT-05-QOS` 提速 **133.6×**、`LP_ret` 从 `0.7%` 跃升至 **`101.9%–105.3%`**！
2. **高优先级 Copy Engine DMA 通道（`ctrl_op_stream_`）**：
   - 为每条 `Sm120StreamContext` 创建最高优先级非阻塞流 `ctrl_op_stream_` 与 80B 固定主机中转缓冲 `ctrl_host_`；
   - **关键工程法则（严禁 Compute Engine 清零）**：在 `PreemptStream` 与 `PrepareRestore` 中，**绝对禁止调用 `Driver::MemsetD32Async`**（CUDA 驱动将其提交至 Compute Engine，当 GPU SM 被 100% 占满时会被在途计算内核完全阻塞！），而必须全部使用纯硬件 Copy Engine 的 **`Driver::MemcpyHtoDAsync_v2` / `MemcpyDtoHAsync_v2`**——即便在 100% SM 饱和重载下，硬件 Copy Engine DMA 往返仍保持 **`~12.6 µs`** 恒定超低延迟！
3. **256 槽位流序异步布局更新（`layout_stage_`）**：
   - `PrepareLaunch` 从 `256 × 64 B = 16 KiB` 固定主机环形缓冲 `layout_stage_` 发起单次非阻塞 `MemcpyHtoDAsync_v2`，严格按目标 `CUstream` 队列顺序将 36 字节 `Layout36` 写入 `ctrl_dev_ + 0x20`，杜绝深队列（CUB/PyTorch）回绕覆盖且无需双跳中转。
4. **GPU 原生 1 线程条件清理核（`sm120_cond_clear_hdr`）**：
   - 取代旧版 `cuLaunchHostFunc` 宿主机回调，初始化时由 `Sm120TrapEngine::LoadHelperModule()` 编译加载 1 线程 PTX 辅助核（执行仅 `~2 µs`，严格按流序排在恢复内核之后、下一排队内核之前）：
     ```ptx
     .visible .entry sm120_cond_clear_hdr(.param .u64 p_ctrl) {
         ld.param.u64 %rd1, [p_ctrl];
         mov.u32 %r0, 0;
         st.global.cg.u32 [%rd1 + 28], %r0;   // 无条件清零 is_restore_wave (+0x1c)
         ld.global.cg.u32 %r1, [%rd1];        // 读取 cmd (+0x00)
         setp.eq.u32 %p0, %r1, 6;             // kCmdMode5Restore?
         setp.eq.u32 %p1, %r1, 2;             // kCmdMode1Replay?
         or.pred %p0, %p0, %p1;
         @!%p0 bra DONE;                      // 若 Preempt #2 已将 cmd 翻为 5/1 则保留！
         mov.u64 %rd0, 0;
         st.global.cg.u64 [%rd1 + 0], %rd0;   // 清零 cmd, evicted_slots
         st.global.cg.u64 [%rd1 + 8], %rd0;   // 清零 active_kernel_idx
         st.global.cg.u64 [%rd1 + 16], %rd0;  // 清零 restored_slots, drained_cnt
         st.global.cg.u32 [%rd1 + 24], %r0;   // 清零 mode1_exit_cnt
     DONE:
         ret;
     }
     ```
   - 这一设计同时保障了两大不变量：① 恢复内核刚完成即自动清零 `is_restore_wave`，绝不污染后续内核；② 若恢复中途触发二次抢占将 `cmd` 翻为 `5/1`，则自动跳过清零 `[0..27]`，绝不抹除二次抢占计数。

### 6.2 核心数据结构逐字节 ABI 布局

#### (1) `Sm120ControlHeader`（80 字节 = `0x50`，`alignas(16)`）

| 字节偏移 | 字段名 | 类型 | 职责与访问者 |
| :---: | :--- | :---: | :--- |
| `+0x00` | `cmd` | `uint32_t` | 命令字：`0=Normal`, `1=Mode1Evict`, `2=Mode1Replay`, `5=Mode5Evict`, `6=Mode5Restore`（奇数=Evict，偶数=Restore） |
| `+0x04` | `evicted_slots` | `uint32_t` | Mode 5 State B 动态 Leader `ATOMG.E.ADD` 分配紧凑 `slot_id` 计数 |
| `+0x08` | `active_kernel_idx` | `uint64_t` | 当前被抢占/正在恢复的 64-bit `HwCommand::GetIdx()`，用于 FIFO 多内核精确隔离 |
| `+0x10` | `restored_slots` | `uint32_t` | Cave 1 恢复结束 `REDG.E.ADD` 计数，供 `PreemptStream` 确认前序恢复已收口 |
| `+0x14` | `drained_cnt` | `uint32_t` | Entry Guardian / Cave 0 拦截的 `State C` CTA 计数 |
| `+0x18` | `mode1_exit_cnt` | `uint32_t` | Mode 1 提前退出的 CTA 计数 |
| `+0x1c` | `is_restore_wave` | `uint32_t` | 波次不变路由标志：`1` 表示当前波次为恢复波次（全 Warp 统一走 `restore_path`） |
| `+0x20` | `arena_va` | `uint64_t` | 本内核绑定的紧凑 `u_arena` 显存基址（由 `Layout36` 随流异步写入） |
| `+0x28` | `slot_stride_bytes` | `uint32_t` | `u_arena` 单个紧凑 CTA 槽位的字节步长（128B 对齐） |
| `+0x2c` | `max_slots` | `uint32_t` | `u_arena` 最大容纳槽位数（`min(num_ctas, sm_count * max_blocks_sm + max(64, sm*2))`） |
| `+0x30` | `thread_area_off` | `uint32_t` | 槽位内 Per-Thread 区起始偏移 |
| `+0x34` | `thread_frame_stride` | `uint32_t` | 每线程帧步长（`AlignUp(0x78 + gpr_tail_regs * 4, 16)`） |
| `+0x38` | `smem_area_off` | `uint32_t` | 槽位内共享内存保存区起始偏移 |
| `+0x3c` | `lmem_area_off` | `uint32_t` | 槽位内局部内存保存区起始偏移 |
| `+0x40` | `lmem_cap_bytes` | `uint32_t` | 每线程动态 LMEM 容量上限 |
| `+0x44` | `max_ctas` | `uint32_t` | 当前已分配的 `Sm120CtaEntry[]` 容量（初始 `65,536`） |
| `+0x48` | `overflow_err` | `uint32_t` | 若 `slot_id >= max_slots` 置 `1` 告警并安全早退防越界 |
| `+0x4c` | `trap_inflight` | `uint32_t` | 当前正驻留 Cave 1 的在途 Warp 原子计数（防止 Restore→Evict 交叠竞态） |

#### (2) `Sm120CtaEntry`（16 字节，紧随 `ctrl_dev_ + 0x50 + 16 * cta_id`）

| 相对偏移 | 字段名 | 语义 |
| :---: | :--- | :--- |
| `+0x00` (`+0x50`) | `state` | `0 = State A`（已完成/空闲）、`1 = Mode 1 Replay`、`2 = State C`（未入正文已排空）、`5 = State B`（全状态已保存至 `slot_id`） |
| `+0x04` (`+0x54`) | `entry_dec` | 入口共识决策：`0 = 放行`、`1 = Mode 1 排空`、`2 = Guardian 侧裁决 Drain（兄弟可能已 EXIT）`、`4 = Trap 侧裁决 Drain（全员均存活）` |
| `+0x08` (`+0x58`) | `slot_id` | State B 分配的紧凑槽位号 `0 .. max_slots - 1` |
| `+0x0c` (`+0x5c`) | `warp_mask` | 位图：Bit `w == 1` 当且仅当第 `w` 号 Warp 在 Mode 5 Evict 瞬间仍然存活（用于部分已退出 CTA 精确重建） |

### 6.3 紧凑池化 `u_arena` 单槽精确内存布局与硬件粒度公式

单个被换出 `State B` CTA 在 `u_arena` 中占据 `slot_base = arena_va + slot_id * slot_stride_bytes` 开始的连续显存块，按 128 字节严格对齐划分为 **5 大区域**：

```text
slot_base + 0x000 ──────────────────────────────────────────────────────
  [CTA 命名屏障区, 64 B]
    +0x00..+0x3f : Named Barriers 0..15 (16 × 4 B, B2R.BAR / R2B.BAR)

slot_base + 0x040 ──────────────────────────────────────────────────────
  [Per-Warp 区, warps × 0x160 B, 每个 Warp 352 字节]
    warp_base + 0x000..0x0ff : UR0..UR63 (64 × 4 B, 256 B)
    warp_base + 0x100..0x103 : UPR 均匀谓词寄存器 (4 B)
    warp_base + 0x104..0x107 : B2R.WARP 每 Warp 屏障参与态 (4 B)
    warp_base + 0x108..0x10f : 对齐保留 (8 B)
    warp_base + 0x110..0x14f : 收敛屏障 B0..B15 (16 × 4 B, 64 B)
    warp_base + 0x150..0x15f : 对齐保留 (16 B)

slot_base + thread_area_off (128B 对齐) ────────────────────────────────
  [Per-Thread 帧区, threads_per_cta × thread_frame_stride]
    thread_base + 0x00..0x6f : 112 B 硬件 Trap Frame (local[0xffff90..0xfffffc])
    thread_base + 0x70..0x73 : saved_lmem_lo (Cave 7 动态栈底指针, 4 B)
    thread_base + 0x74..0x77 : 保留 (4 B)
    thread_base + 0x78..     : R14..R251 (64-bit 成对), R252 (+0x430), R253 (+0x434)
                               按 reg_ceil = min(AlignUp(clamped_regs + 16, 16), 256) 动态裁剪!

slot_base + smem_area_off (128B 对齐) ──────────────────────────────────
  [动态共享内存区, smem_area B]
    用户共享内存 [0x400, SR_SMEMSZ) 原始镜像
    smem_area = (total_user_smem > 0) ? AlignUp(total_user_smem + 0x400, 4096) : 0
    (v2.2.1 精确匹配 Blackwell "+0x400 保留区后再按 4 KiB 量子取整" 的硬件分配公式!)

slot_base + lmem_area_off ──────────────────────────────────────────────
  [动态局部内存区, AlignUp(threads_per_cta × lmem_cap_bytes, 128)]
    每线程有效栈帧 [save_lo, stack_top) 原始镜像 (仅当 lmem_cap > 0 时非零)
────────────────────────────────────────────────────────────────────────
```

- **显存压缩比**：`max_slots = min(num_ctas, sm_count * max_blocks_sm + max(64, sm_count * 2))`，使 `u_arena` 占用上限仅与物理 SM 数成正比而与 Grid 总大小无关（`16,384 CTAs` 相比 v1 `4,096 MiB` 缩减至 RTX 5080 **`29.31 MiB` (`139.7×`)** 与 RTX 5060 **`9.38 MiB` (`436.9×`)**）；
- **多内核扩容防错配**：`kernel_arena_map_[kernel_idx]` 精确记录每个内核发射时实际使用的 `arena_va`，`EnsureArenaPool` 扩容时将旧 `arena` 移入 `retired_arenas_` 延迟释放，`PrepareRestore` 按 `active_kernel_idx` 查表还原原 `arena_va`，杜绝多内核背靠背扩容引发 `cuda error 700`。

### 6.4 端到端 Evict → HP 并发执行 → 0 脉冲 Restore 完整时序

```text
Host (调度器 / HP 线程)                    GPU (LP Stream / SM120 硬件)                 HP Stream
─────────────────────────                  ─────────────────────────────                ──────────
1. XQueueSuspend(lp_xq):
   a) LaunchWorker::Pause()
   b) PreemptStream(kTrapFrame):
      - 若前序 cmd==6: 等待 restored_slots >= min(evicted, max_slots) && trap_inflight == 0
      - 经 ctrl_op_stream_ 纯 CE DMA 写 zeros[0x04..0x1b], overflow=0, cmd=5 (~12.6 µs)
      - FireEvictPulse(): BAR0 0x424684=0x80001c01 -> 读 0x9400 PTIMER -> 写 0x00001c01 (~1 µs)
                                           2. BAR0 广播中断命中全部 SM (~10 µs):
                                              - 已完成 CTA: 早已自然退出 (State A, state=0)
                                              - 在途 CTA: 陷进 Cave 0 -> 五区间判定 ->
                                                Leader 选举 -> 申请 slot_id ->
                                                Cave 3/4/8/7/8/2 全态保存至 u_arena ->
                                                state=5 -> Cave 1 clean_exit 释放 SM!
                                              - 未发射 CTA: Entry Guardian 见 cmd=5 ->
                                                state=2, active_kernel_idx=my_idx, EXIT!
                                              - 后续排队 K2: 见 my_idx != active -> EXIT!
   c) StreamQuery(lp_stream) == SUCCESS!   <-- SM 100% 释放，耗时仅 ~45–66 µs!

3. HP 独占满速执行:                                                                     4. HP Kernel
                                                                                          (DirectLaunch,
                                                                                           u_control=0,
                                                                                           无争抢全速完成!)

5. XQueueResume(lp_xq):
   a) Reactivate(log):
      - 检查 evicted/drained/mode1 > 0 (否则 ResetControlSync 安全空转)
      - PrepareRestore(): 查 kernel_arena_map_[active_idx], 经 CE DMA 写
        restored_slots=0, is_restore_wave=1, cmd=6 (~12 µs)
      - 重发 log 中 >= active_idx 的内核:
        * 对 active_idx 内核重新 Launch -> 入队 sm120_cond_clear_hdr (~2 µs)
        * 继续入队后续排队内核 K2, K3...
   b) LaunchWorker::Resume()
                                           6. 恢复波次零脉冲原位续算 (0 Host 脉冲!):
                                              - State A (state==0): Entry Guardian 秒退
                                              - State B (state==5): Slot 57 BPT.TRAP 0x1
                                                自陷 Cave 1 -> Cave 8/7/3/6/8/5 逆序恢复 ->
                                                清 state=0 -> restored_slots++ ->
                                                活 Warp RTT 精确回到原 PC 继续算; 死 Warp 退!
                                              - State C (state==2): replay_state_c 清 state=0
                                                进入 +0x500 首次执行原内核
                                              - 末尾 sm120_cond_clear_hdr 自动清零控制头!
```

---

## 第 7 章：BAR0 D10 硬件安全法则、自适应路由器、生产级 Shim/HAL 加固与 Context CILP

### 7.1 BAR0 MMIO 硬件安全法则（D9 / D10 铁律）

- **唯一允许访问的两个 BAR0 偏移**：
  1. `0x00424684` (`NV_PGRAPH_PRI_GPCS_TPCS_SM_SFE_DBGR_CONTROL0`)：全芯片 GPC/TPC/SM 调试 Trap 广播触发寄存器；
  2. `0x00009400` (`NV_PTIMER_TIME_0`)：非 PGRAPH PRI 时钟低 32 位寄存器（仅作只读刷新）。
- **D10 刷新铁律**：写入 `bar0[0x424684/4] = 0x80001c01u`（拉高 `STOP_TRIGGER`）后，**严禁在置位期读取 `0x424684` 或任意 PGRAPH PRI 寄存器**（否则会截断跨 GPC 广播或造成总线死锁），必须执行一次非 PRI 读 `(void)bar0[0x9400/4]` 刷新首个 PCIe Posted Write 并获得干净 `~1 µs` 脉冲宽度，随后立即写入 `0x00001c01u` 撤销 `STOP_TRIGGER`（同时绝不可人为拉长置位脉宽至 8 µs，否则会误伤紧邻恢复的短内核）。
- **共享 GPU 运维红线**：触发 BAR0 前必须以 `nvidia-smi --query-compute-apps` 确认无他人计算作业运行，并设置 `CUDA_MPS_PIPE_DIRECTORY=/tmp/no_mps_<user>` 隔离 MPS。

### 7.2 自适应内核/模式路由器（`DEF-02` + `DEF-04` + Phase 2）

并非所有 Blackwell 硬件特性都能被纯软件字节拷贝完整重建（例如硬件 `mbarrier` 含隐式不可见事务相位寄存器、Cooperative Kernel 依赖全网格同时驻留 SM、多波次 CTA Cluster 依赖集群内成员同时在线以支持 DSMEM 互访）。`CudaQueueLv3Trap::OnHwCommandSubmit()` 实现了**零用户感知的自动安全路由**：

| 内核 / 命令类型 | 自动检测机制 | 路由策略 | 实测验证 |
| :--- | :--- | :---: | :--- |
| **硬件 `mbarrier` (`SYNCS`) 内核** | `Sm120GuardianRegistry::BinaryHasHwMbarrier()` 静态扫描 `op9 == 0x1a7 / 0x1b2` | 自动降级 **Level-1 边界**（`killable=false`, 清除 `Deactivatable`） | `HW-03` 双卡 **PASS (`0/15,360`)** |
| **Cooperative Kernel (`grid.sync`)** | `intercept.cpp` 完整拦截 `cuLaunchCooperativeKernel[_ptsz]` → `CudaCooperativeKernelLaunchCommand` | 自动降级 **Level-1 边界**（保护全网格常驻语义） | `HW-07` 双卡 **PASS (`0/3,840`)** |
| **Blackwell CTA Cluster + DSMEM** | 运行期比对 `ClusterSize() > 1 && BlockCnt() > sm_count_` | **单波次自动走 Mode 5**（5080 `60 ≤ 84` 实测 `111 µs` PASS）；**多波次自动降级 Level-1**（5060 `60 > 30` PASS） | `HW-06` 双卡 **PASS (`ce=0, 0/7,680`)** |
| **CUDA Graph** | 识别 `CudaGraphCommand` | 自动降级 **Level-1 图边界**（跨进程场景可配合 Context CILP） | `HW-04` 双卡 **PASS (`0.000e+00`)** |
| **幂等内核（标记 `kCommandPropertyIdempotent`）** | `SelectSm120Mode()` | **Mode 1 (`kEarlyExit`)**（0 B `u_arena`） | `LIB-01`, `CASE 01` **PASS** |
| **其余所有通用/非幂等内核** | 默认主力路径 | **Mode 5 (`kTrapFrame`)**（全态保存与 0 脉冲原位恢复） | 全部闭源库 + PyTorch + Triton **100% PASS** |

### 7.3 生产级 Shim / HAL 并发加固（`DEF-03` / `DEF-05` / `DEF-06` / 竞态修复）

1. **`DEF-03` 内存安全 `XMemFree_v2` 延迟释放队列（`shim.cpp`, `xqueue.cpp`）**：
   - 原版 XSched 的 `XMemFree_v2()` 无条件调用 `XQueueManager::ForEachWaitAll()`，在 LP 挂起期间一旦 HP 线程或 PyTorch `torch.cuda.empty_cache()` / cuBLAS workspace 触发 `cudaFree()` 便永久死锁。
   - **修复方案**：新增 `XQueue::IsSuspended()` 与 `XQueueManager::HasAnySuspended()`。当存在挂起队列时，`XMemFree_v2()` 将待释放 `CUdeviceptr` 压入受互斥锁保护的 `g_deferred_frees` 立即返回 `CUDA_SUCCESS`——**既 100% 消除死锁，又保证被挂起 LP 内核正在使用的显存不被提前释放**；待后续无挂起状态的 `XMemFree_v2` / `XStreamSynchronize` / `XCtxSynchronize` 安全点调用 `DrainDeferredFreesIfSafe()` 统一释放。
2. **`DEF-05` PyTorch ResNet-18 训练自愈（`cuda_command.cpp`）**：
   - 根因：`cuDNN`（如 `bn_fw_tr_1C11_singleread`）在主线程为后续较小通道 BatchNorm 层调用 `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, 8192)`，抢先覆盖了异步 `LaunchWorker` 尚未发射的前序 `shm = 26,624 B` BatchNorm 层的动态 SMEM 上限，导致 `CUDA_ERROR_INVALID_VALUE (1)`。
   - **修复方案**：`CudaKernelCommand::EnsureDynamicSmem(shm)` 仅在遭遇 `CUDA_ERROR_INVALID_VALUE` 时动态上调 `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` 并重试（热路径 0 开销）；同时修复 `CudaKernelLaunchExCommand` 在 `param_cnt_ == 0` 时的属性数组深拷贝及变长参数缓冲区最大跨度深拷贝。
3. **`DEF-06` 无上下文工作线程回退（`shim.h`, `shim.cpp`）**：
   - RAII 守卫 `ScopedStreamContext` 在 `cuCtxGetCurrent() == nullptr` 时自动通过 `Driver::StreamGetCtx(stream, &sctx)` 解析并 Push/Pop 目标流所属上下文，彻底消除多线程库调用断言崩溃。
4. **`CudaQueueLv3Trap::last_kernel_` 多线程竞态修复（`cuda_queue.cpp`）**：
   - 此前提交线程（`OnHwCommandSubmit`）与 `LaunchWorker` 线程无锁并发读写 `std::shared_ptr<HwCommand> last_kernel_` 导致引用计数损坏崩溃（`launch_worker.cpp:98`），且 `CudaEventRecordCommand` 覆写 `last_kernel_` 导致 `SelectSm120Mode` 退化。改为受 `last_kernel_mtx_` 保护且仅在 `Launch()` 中对 `CudaKernelCommand` / `CudaGraphCommand` 更新。

### 7.4 独立解耦的硬件 Context CILP 引擎（`ResumableSm120Context`）

针对跨进程多租户隔离、或含 `mbarrier` / CUDA Graph 的上下文级抢占，系统在 `resumable_sm120.cpp` 与 `sm120_rm_handles.cpp` 中独立实现了基于 NVIDIA RM ioctl（`NV_ESC_RM_CONTROL = 0xc020462a`）的硬件 CILP 引擎：
- 通过 `sm120_rm_handles` 透明捕获 `hClient`、`hSubdevice`、`hTsg`、`hDebugObject` 与 `/dev/nvidiactl` 文件描述符；
- 调用 **`NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE (0x20801210)`** 将目标 TSG 的计算抢占模式配置为 `CILP`（`cilpPreemptMode = 2`）；
- 暂停时调用 **`NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT (0x83de0317)`** 与 **`NVA06C_CTRL_CMD_PREEMPT (0xa06c0105)`** 触发硬件上下文换出（RTX 5080 实测 `P50 = 420 µs, P99 = 476 µs, lost_work = 0, diff = 0`）；恢复时调用 **`NV83DE_CTRL_CMD_DEBUG_RESUME_CONTEXT (0x83de0318)`** 与 `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE (0xa06c0101)` 原位续算。

---

## 第 8 章：研发历程中的 15 项关键 Blackwell 微架构发现与根因复盘

本节沉淀本项目在从 v1 推进至 `v2.2.1-sm120` 过程中，通过双卡真机调试积累的 **15 项一线底层知识与避坑法则**：

| 序号 | 编号 / 阶段 | 故障现象 | 底层微架构 / 系统根因 | 最终根治方案 |
| :---: | :---: | :--- | :--- | :--- |
| 1 | `v2.0` | `580.95.05` 与 `595.99.02` 切换即初始化失败 | 跨主版本驱动 ToolsMemory 发生 `+0x5900` 偏移漂移，但 `tools_pc - handler_va == 0x4400` 恒定 | 10 指令二进制指纹动态定位 + `BRA self` 动态扫洞 |
| 2 | `v2.0` | 带早退分支内核恢复报 `Error 700` | 部分 Warp 已 `EXIT` 后 `Warp 0` 缺失且传统全 CTA SMEM 步长出现数据空洞 | `warp_mask` + `@P0 ATOMG.E.OR` 动态 Leader + Partial-CTA 32-lane 回退 + 死 Warp `clean_exit` |
| 3 | `v2.0` | `Cases 04–09` 在 Cave 2/5 随机崩溃 | SM120 `BRX R10` 消费 64-bit `R10:R11`，`R11` 残留 `0xffffffff` 触发 `-4 GiB` 负跳 | `BRX R10` 前显式 `mov_imm(11, 0)` |
| 4 | `v2.0` | `REG:255` 内核最高寄存器校验错 | Blackwell `R254:R255` 为硬件 Trap 寄存器，用户最高为 `R253`（`SR_REGALLOC==256`） | Cave 2/5 增 `SR_REGALLOC - 3 >= 253` 显式谓词保护 + `reg_ceil` 封顶 `256` |
| 5 | `v2.0` | BAR0 置位期偶发跨 GPC 中断丢失（D10） | `0x424684` 置 `0x80001c01` 期间读 PRI 寄存器打断广播 | 改读非 PRI `NV_PTIMER_TIME_0 (0x9400)` 刷新首个 Posted Write |
| 6 | `FIX-1` | 空闲队列挂起恢复后后续输出全 0 | `Reactivate` 无条件将 `cmd` 置 `6`，无 `State B/C` 消费导致后续 CTA 秒退 | 仅当 `evicted/drained/mode1 > 0` 才恢复，否则 `ResetControlSync()` |
| 7 | `FIX-2` | `NP >= 2` 二次抢占丢失 70%–96% CTA | 第一次恢复末尾无条件清零覆盖了第二次抢占计数 | `v2.2.0` 升级为 1 线程 GPU 原生条件清理核 `sm120_cond_clear_hdr` |
| 8 | `FIX-3` | 0-SMEM 闭源核报 `cuda error 717` | `SR_SMEMSZ == 0` 减 `0x400` 无符号下溢为 `0xFFFFFC00` | 先判 `SR_SMEMSZ >= 0x401` 再减 `0x400` |
| 9 | `FIX-4` | 多内核扩容偶发 `cuda error 700` | 背靠背内核切换大 `arena` 导致恢复用回小 `arena` 且 8 槽 `layout_stage` 回绕 | `kernel_arena_map_[kernel_idx]` 精确绑定 + 256 槽 `layout_stage_` + 初始 65,536 CTA |
| 10 | `FIX-5` | `cuBLAS` 二次抢占触发 4s 看门狗（`719`） | `cmd: 6 -> 5` 瞬间同 CTA Warp 分裂进入 `restore_path` 与 `norm_gate` 共抢 Barrier 0 | 改按波次不变 `is_restore_wave` 统一路由 + `replay_state_c` 奇偶 `(cmd & 1)` 共识 |
| 11 | `FIX-6` | `cuBLAS` 首次抢占 `ev=12 dr=2087` 后死锁 | SM120 上兄弟 Warp `EXIT` 后残余 Warp 调 `bar_sync_trap()` 永死 | 严格区分 `entry_dec=2`（Guardian 裁决，无同步逃逸）与 `4`（Trap 裁决，安全同步） |
| 12 | `DEF-01/08` | `cuBLAS SGEMM` 慢 4.3×、PyTorch `LP_ret=0.7%` | 零拷贝宿主控制面 PCIe 串行读 + 100% SM 饱和阻塞 Compute Engine `cuMemsetD32Async` | **100% GDDR7 VRAM 控制面 + 纯 Copy Engine DMA `ctrl_op_stream_`**（提速 **10×–548×**） |
| 13 | `DEF-03` | 挂起期 `cudaFree` / `empty_cache()` 死锁 | `XMemFree_v2` 无条件等待已挂起 LP 队列 | `HasAnySuspended()` 感知 + 内存安全 `g_deferred_frees` 延迟释放队列 |
| 14 | `DEF-05` | PyTorch ResNet-18 训练报 `cuda error 1` | `cuDNN` 主线程提前调低 `MAX_DYNAMIC_SHARED_SIZE_BYTES` 与异步 `LaunchWorker` 竞态 | 零热路径开销 `EnsureDynamicSmem` 重试 + `LaunchEx` 属性/参数深拷贝 |
| 15 | `v2.2.1` | 84-SM 5080 `CUB Onesweep` 末槽 `error 700` | Blackwell 硬件先加 `0x400 B` 保留区再按 4 KiB 对齐（`36,352 B -> 40,960 B`） | `smem_area = AlignUp(total_user_smem + 0x400, 4096)` + `sm120_cond_clear_hdr` 清 `is_restore_wave` + Guardian Drain 记 `active_kernel_idx` |

---

## 第 9 章：RTX 5080 与 RTX 5060 双卡全方位实测结果与 `v2.0 → v2.2.1` 演进对比

以下全部数据均来自真实硬件运行（零造假、零模拟），覆盖 **5 项标准回归 + Suite 1（15 项闭源库/硬件边界 + `DEF-03/06`）+ Suite 2（LLM/DNN/100% SM 饱和 QoS + 频扫 + BAR0 跨上下文）+ Suite 3（11 项真实 PyTorch/cuDNN/Triton）**。

### 9.1 底层微基准与 11 项 E2E / CILP 双卡对照（`sm120_v2_bench`, `sm120_v2_e2e_suite`, `sm120_l3_resumable_bench`）

| 测试项 | RTX 5080 (`v2.0` PCIe) | **RTX 5080 (`v2.2.1` VRAM)** | RTX 5060 (`v2.1` PCIe) | **RTX 5060 (`v2.2.1` VRAM)** |
| :--- | :---: | :---: | :---: | :---: |
| **`sm120_sass_disasm` (`nvdisasm -b SM120`)** | 79 / 79 PASS | **79 / 79 PASS** | 79 / 79 PASS | **79 / 79 PASS** |
| **`.text.traphandler` 代码洞探测** | 11 Caves / 1,703 | **11 Caves / 1,703** | 11 Caves / 1,695 | **11 Caves / 1,695** |
| **11 项合成 E2E (`sm120_v2_e2e_suite`)** | 11 / 11 ALL_PASS | **11 / 11 ALL_PASS** | `CASE 02` 停滞 | **`CASE 01..10` PASS** |
| **Entry Guardian 零抢占开销 (`84×128`)** | `+30.78 µs` | **`+0.99 µs` (31.1×↑)** | `+43.25 µs` | **`+0.43–0.97 µs` (45×–100×↑)** |
| **Mode 1 (`512 CTAs, 48K SMEM`) Evict P50** | `971.6 µs` | **`110.1 µs` (8.8×↑)** | `901.2 µs` | **`38.0 µs` (23.7×↑)** |
| **Mode 5 (`84 CTAs, 0K SMEM`) Evict / Restore** | `332.3 / 763.6 µs` | **`50.9 / 30.5 µs` (6.5× / 25.0×↑)** | `399.7 / 772.2 µs` | **`50.8 / 62.9 µs` (7.9× / 12.3×↑)** |
| **Mode 5 (`84 CTAs, 32K SMEM`) Evict / Restore** | `372.8 / 831.8 µs` | **`55.4 / 41.4 µs` (6.7× / 20.1×↑)** | `447.0 / 806.4 µs` | **`55.2 / 73.9 µs` (8.1× / 10.9×↑)** |
| **Mode 5 (`512 CTAs, 48K SMEM`) Evict / Restore** | `1,260.8 / 1,859.0 µs` | **`119.9 / 292.0 µs` (10.5× / 6.4×↑)** | `1,076.2 / 917.4 µs` | **`48.0 / 325.8 µs` (22.4× / 2.8×↑)** |
| **紧凑 `u_arena` 缩放 (`16,384 CTAs`)** | `25.5 MiB` (`160.6×`) | **`29.31 MiB` (`139.7×`)** | `9.38 MiB` (`436.9×`) | **`9.38 MiB` (`436.9×`)** |
| **Context CILP (`sm120_l3_resumable_bench`)** | `425 µs` (`diff=0`) | **`420 µs` (`P99=476 µs, diff=0`)** | `431 µs` (`diff=0`) | **PASS (`ZERO_LOST_WORK`)** |

### 9.2 Suite 1 — 8 大闭源 CUDA 库 + 7 项 Blackwell 硬件特性 + `DEF-03/06`（双卡均 **15 / 15 PASS**）

| 编号 | 测试负载（零源码修改闭源二进制 / 微架构原语） | RTX 5080 (`v2.2.1`) 挂起 / 结果 | RTX 5060 (`v2.1 → v2.2.1`) 挂起 / 结果 | 数值精度 |
| :---: | :--- | :---: | :---: | :---: |
| **LIB-01** | `cuBLAS` FP32 SGEMM `3072×3072` (`beta=0.0`) | **`0.137 ms`** PASS | `66.20 → 0.856 ms` (**77×**) PASS | `0.000e+00` (`0/9.4M`) |
| **LIB-02** | `cuBLAS` FP32 SGEMM `3072×3072` (`beta=1.0` 非幂等) | **`0.080 ms`** PASS | `65.52 → 0.862 ms` (**76×**) PASS | `0.000e+00` (`0/9.4M`) |
| **LIB-03** | `cuBLAS GemmEx` FP16 Tensor Core `4096×4096` (`beta=1`) | **`0.134 ms`** PASS | `9.26 → 0.284 ms` (**33×**) PASS | `0.000e+00` (`0/16.8M`) |
| **LIB-04** | `cuBLAS GemmEx` BF16 Tensor Core `4096×4096` (`beta=1`) | **`0.135 ms`** PASS | `9.14 → 0.295 ms` (**31×**) PASS | `0.000e+00` (`0/16.8M`) |
| **LIB-05** | `cuBLASLt` FP16 Tensor Core `4096×4096` (`beta=1`) | **`0.130 ms`** PASS | `9.25 → 0.282 ms` (**33×**) PASS | `0.000e+00` (`0/16.8M`) |
| **LIB-06** | `cuFFT` 原地 1D Batched C2C (`65536 × 256`) | **`0.140 ms`** PASS | `15.67 → 0.188 ms` (**83×**) PASS | `0.000e+00` (`0/16.8M`) |
| **LIB-07** | `cuSPARSE` CSR SpMM (`4096×512, nnz=524K, beta=1`) | **`0.113 ms`** PASS | `50.82 → 0.109 ms` (**466×**) PASS | `0 / 2,097,152` |
| **LIB-08** | `CUB` `DeviceRadixSort::SortPairs` (`8M`, `111r/36.3K`) | **`0.055 ms`** PASS\* | `1.75 → 0.079 ms` (**22×**) PASS | `0 / 8,388,608` |
| **HW-01** | PTX Tensor Core `mma.sync` + `atomicAdd` 非幂等 | **`0.034 ms`** PASS | **`0.055 ms`** PASS | `cnt=450,240/450,240` |
| **HW-02** | PTX `cp.async` (`LDGSTS`) 异步拷贝流水线 | **`0.114 ms`** PASS | **`0.168 ms`** PASS | `0 / 15,360` |
| **HW-03** | PTX SMEM `mbarrier`（静态识别 `SYNCS` → 自动 Lv1） | **PASS**（自动 Lv1） | **PASS**（自动 Lv1） | `0 / 15,360` |
| **HW-04** | CUDA Graph 4 节点（自动回退 Lv1 边界） | **PASS** (`8.57 ms`) | **PASS** (`19.87 ms`) | `0.000e+00` |
| **HW-05** | 空闲队列 `Suspend/Resume` 安全性 | **PASS** | **PASS** | `0 / 4,096` |
| **HW-06** | Blackwell CTA Cluster `2×1×1` + DSMEM (`60 blocks`) | **Mode 5 `0.111 ms` PASS** (`60 ≤ 84`) | **自动 Lv1 PASS** (`60 > 30`) | `ce=0, 0 / 7,680` |
| **HW-07** | Cooperative Groups `grid.sync`（Shim 拦截 → Lv1） | **PASS** (`17.82 ms`) | **PASS** (`19.42 ms`) | `0 / 3,840` |
| **DEF-03** | 挂起期 `8×(cudaMalloc + cudaFree 4 MiB)` 零死锁 | **`0.034 ms`** PASS | **`0.068 ms`** PASS | `0 / 7,680` |
| **DEF-06** | 无上下文工作线程 `cuCtxGetCurrent()==NULL` | **PASS** | **PASS** | 零断言崩溃 |

### 9.3 Suite 2 — DNN / LLM 多模式能力调度与 BAR0 跨上下文干扰（`v2.2.1` 双卡实测）

| 调度场景 | 平台 | Level-1 HP P50 / SLO≤2ms | **Mode 5 HP P50 / P99** | **Mode 5 SUSP P50** | **Mode 5 SLO≤2ms** | **Mode 5 LP 吞吐保留** | 数值误差 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM Serving (`15 ms`)** | RTX 5080 | `0.358 ms` / `100%` | **`0.270 / 0.490 ms`** | **`32 µs`** | **`100.0%`** | **`102.4%`** | `0.00e+00` |
| **LLM Serving (`15 ms`)** | RTX 5060 | `4.217 ms` / `15.0%` | **`0.475 / 0.610 ms`** (**8.9×↑**) | **`57 µs`** (**548×↑**) | **`100.0%`** | **`98.2%`** | `0.00e+00` |
| **DNN Training (`15 ms`)** | RTX 5080 | `0.072 ms` / `100%` | **`0.071 / 0.105 ms`** | **`31 µs`** | **`100.0%`** | **`90.6%`** | `0.00e+00` (`918/918`) |
| **DNN Training (`15 ms`)** | RTX 5060 | `0.634 ms` / `100%` | **`0.246 / 0.306 ms`** (**2.6×↑**) | **`70 µs`** | **`100.0%`** | **`84.3%`** | `0.00e+00` (`579/579`) |
| **100% SM 饱和 Heavy (`15 ms`)** | **RTX 5080** | `6.155 ms` / **`0.0%`** | **`0.060 / 0.079 ms`** (**102.6×↑**) | **`45 µs`** (**136.5×↑**) | **`100.0%`** | **`103.4%`** (vs M1 `76.1%`) | `0.00e+00` |
| **100% SM 饱和 Heavy (`15 ms`)** | **RTX 5060** | `7.091 ms` / **`0.0%`** | **`0.083 / 0.106 ms`** (**85.4×↑**) | **`66 µs`** | **`100.0%`** | **`100.7%`** | `0.00e+00` |
| **Heavy 频率扫描 (`5/10/20/50 ms`)** | RTX 5080 | `0.0%` SLO≤2ms | **`0.060–0.072 ms`** | **`45–56 µs`** | **`100.0%`** | **`99.9%–120.3%`** | `0.00e+00` |
| **Heavy 频率扫描 (`5/10/20/50 ms`)** | RTX 5060 | `0.0%–6.7%` SLO≤2ms | **`0.084–0.088 ms`** | **`66–70 µs`** | **`100.0%`** | **`98.7%–105.0%`** | `0.00e+00` |
| **BAR0 跨上下文旁观者 (60 脉冲)** | RTX 5080 | 独立 `0.011 / 0.012 ms` | 并发 **`0.011 / 0.012 ms` (Max `0.012`)** | — | — | **0.000 ms 零干扰** | `err=0, mis=0` |
| **BAR0 跨上下文旁观者 (60 脉冲)** | RTX 5060 | 独立 `0.015 ms` | 并发 **`0.015 ms`** | — | — | **0.000 ms 零干扰** | `err=0, mis=0` |

### 9.4 Suite 3 — 真实未修改 PyTorch 2.11 + cuDNN 9.19 + Triton 3.6.0（双卡均 **11 / 11 PASS**）

| 编号 | 真实框架工作负载 | **RTX 5080 (`v2.2.1`) 挂起 / 误差** | **RTX 5060 (`v2.2.1`) 挂起 / 误差** | 结论 |
| :---: | :--- | :---: | :---: | :---: |
| **PT-01** | `torchvision.models.resnet50` FP16 推理 (`cuDNN 9.19`) | **`0.070 ms`** / `0.000e+00` | **`0.291 ms`** (较 v2.1 快 **267×**) / `0.000e+00` | **双卡 PASS** |
| **PT-02** | `torchvision.models.resnet18` 非幂等训练 (`Fwd+Bwd+AdamW`) | **`0.031 ms`** / `7.451e-09` | **`0.023 ms`** (修复 `DEF-05`) / `5.472e-09` | **双卡 PASS** |
| **PT-03-MATH** | 4 层 Llama Prefill (`SDPA = MATH`) | **`0.991 ms`** / `0.000e+00` | **`0.539 ms`** (**71×**) / `0.000e+00` | **双卡 PASS** |
| **PT-03-FLASH** | 4 层 Llama Prefill (`SDPA = FlashAttention-2`) | **`0.176 ms`** / `0.000e+00` | **`0.313 ms`** (**30×**) / `0.000e+00` | **双卡 PASS** |
| **PT-03-EFF** | 4 层 Llama Prefill (`SDPA = xFormers Memory-Efficient`) | **`0.179 ms`** / `0.000e+00` | **`0.239 ms`** (**43×**) / `0.000e+00` | **双卡 PASS** |
| **PT-03-CUDNN** | 4 层 Llama Prefill (`SDPA = cuDNN 9.19 Fused Attention`) | **`0.023 ms` (`23 µs`!)** / `0.000e+00` | **`0.033 ms` (`33 µs`!)** / `0.000e+00` | **双卡 PASS** |
| **PT-04** | 自定义 **Triton 3.6.0** `sm_120` 非幂等核（含 LMEM 栈溢出） | **`0.126 ms`** / `0.000e+00` | **`0.268 ms`** / `0.000e+00` | **双卡 PASS** |
| **PT-06** | 挂起期间 `torch.cuda.empty_cache()` + HP 分配/释放 (`DEF-03`) | **`0.030 ms`** / `0.000e+00` | **`0.139 ms`** / `0.000e+00` | **双卡 PASS** |
| **PT-05-QOS (`m5`)** | 3 层 Llama (`MATH`) + Triton 混部 QoS (对比 Lv1) | **`HP P50 = 0.290 ms`** (Lv1 `0.453`), `SUSP = 224 µs`, `SLO≤2ms = 100%`, **`LP_ret = 105.3%`** | **`HP P50 = 0.552 ms`** (**133.6×↑**, Lv1 `1.866 / SLO 65%`), `SUSP = 454 µs`, `SLO≤2ms = 100%`, **`LP_ret = 101.9%`** | **双卡 PASS** |

---

## 第 10 章：能力边界、已知局限与 Phase 3 演进与论文路线图

### 10.1 坦诚边界与已知局限

1. **`DEF-09`（P2 — CUDA Graph 图内检查点）**：当前 `CudaGraphCommand` 自动安全降级至 Level-1 图边界（`HW-04` 双卡 PASS）。对超长 CUDA Graph（如 vLLM Decode / `torch.compile(reduce-overhead)`），短期可切换至硬件 Context CILP，Phase 3 将探索 `cuGraphInstantiate` 节点预插桩与图内中断检查点。
2. **`DEF-10`（P2 — BAR0 特权剥离）**：当前直接写 `/sys/bus/pci/.../resource0` 需要 `CAP_SYS_ADMIN`（root）权限。Phase 3 将把单次 D10 BAR0 脉冲下沉至带频率限流与进程白名单校验的独立特权 `xserver` 守护进程或极简字符设备模块，普通非 root 用户进程经 IPC 安全请求。
3. **84-SM 多 Pass Decoupled Look-Back 极高频二次抢占边界**：`CUB DeviceRadixSort`（`LIB-08`）在双卡 `Suite 1 all`（15/15）、`NP=0`（3/3）及 `NP=1`（5/5）均 100% PASS，在 84-SM RTX 5080 `178 µs` 极高频 `NP=2` 连续压测下仍有约 3/10 概率触发多 Pass `cuda error 700`，列为 Phase 3 多 Pass Look-Back 专项排查项。
4. **桌面显示 GPU 看门狗 vs 无头计算服务器**：在运行 Wayland 桌面合成器的消费级 RTX 5060（Driver `595.99.02`）上，`__nanosleep(200)` + 48K SMEM 极端合成循环偶发驱动级 `Xid 109 CTX SWITCH TIMEOUT`，而无头服务器 RTX 5080（Driver `580.95.05`）稳定 11/11 ALL_PASS。另运行直接调用 HAL 的二进制（`sm120_v2_bench`、`sm120_v2_e2e_suite`）时，`LD_LIBRARY_PATH` 中须前置系统原生 `libcuda.so.1`，避免软链 `libshimcuda.so` 劫持未托管流。

### 10.2 Phase 3 演进与顶会论文实验规划

1. **系统去特权化与图内抢占**：落地特权 `xserver` BAR0 代理（`DEF-10`）与 CUDA Graph 节点预插桩原型（`DEF-09`）；
2. **真实大模型服务引擎接入**：零源码修改接入 **vLLM / SGLang**（PagedAttention + Continuous Batching），实测长上下文 Prefill（`8K–32K` tokens）与低延迟 Decode（TTFT / TPOT P99）在单卡混部下的 SLO 收益；
3. **顶会论文全景多基线对比**：在真实 Llama-8B LoRA/全量微调 + 在线推理混部下，系统对比 **Native CUDA Streams**、**NVIDIA MPS**、**CUDA Green Contexts**、**XSched Level-1**、**Mode 1（幂等）**、**本文 Mode 5（非幂等 VRAM 9-Cave Trap Frame）** 与 **Context CILP**，形成面向 SOSP / OSDI / ASPLOS / EuroSys 的完整论文评估体系。
