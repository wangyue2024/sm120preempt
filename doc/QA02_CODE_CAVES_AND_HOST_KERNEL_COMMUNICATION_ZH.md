# Q&A 专题二：可编辑代码洞（Code Caves）定位、主机-设备双向通信、Target ID 过滤与 Kernel 元数据获取全解

> **对应问题**：  
> *“可编辑的代码洞是什么？在哪里？这些代码洞如何与主机端信息交换？如何指定抢占的 target ID？如何写回状态和反馈信息？可以调用全局内存吗？代码洞如何获取 kernel 端信息？例如 kernel ID 和其他 kernel 参数？”*

---

## 1. 什么是“可编辑的代码洞（Code Caves）”？为什么必须使用代码洞？

### 1.1 定义
**可编辑代码洞（Executable Code Caves）** 是指：在 NVIDIA CUDA 驱动为每个 `CUcontext` 预分配并已通过硬件/驱动签名校验的**官方特权 Trap Handler 可执行代码段（Utility Code Segments）尾部**，天然存在的、由编译器按页/段对齐填充的大片连续 **`NOP`（空操作）指令区域**。

### 1.2 为什么不能自己分配一块新内存放 Trap 代码？
在旧架构（Volta/Ampere）上，XSched 调用 `cuXtraInstrMemBlockAlloc` 在 GPU 虚拟地址空间动态申请一块新的指令内存页来存放自定义 Trap 代码。但在 **Blackwell (`SM120`) + Driver `595.99.02`** 上：
- 一旦激活硬件 Trap 调试状态机，驱动内部会校验当前 Context 所有挂接指令页的合法性；
- 凡是检测到跳出官方 Utility Code Segment 进入外部动态分配的未签名指令块，驱动会立即判定发生指令流篡改，直接触发 **Context Cancel**，导致后续所有 Kernel 发射报错 `CUDA_ERROR_LAUNCH_FAILED (719)`。
- **核心破局**：我们将自定义抢占与恢复机器码直接写入**官方 Trap Handler 自身代码段尾部的天然 `NOP` 代码洞**中（使用 `cuXtraMemcpyHtoD` 覆写并调用 `cuXtraInvalInstrCache(ctx)` 刷新 SM L1 指令缓存）。因为这些代码洞与官方 `.text.traphandler` 处于**同一个合法的代码段映射页内**，驱动和硬件 100% 信任其合法性，彻底根除了 `Error 719`！

---

## 2. 代码洞具体在哪里？（三级代码洞精确地址地图与容量盘点）

### 2.1 如何定位官方 Trap Handler 的基地址？
在 [`trap_sm120.cpp`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/platforms/cuda/hal/src/level3/trap_sm120.cpp) 中，我们通过 `cuGetExportTable` 调用 `libcuda.so` 内部的两个未公开导出表来解析代码段基址：
1. 调用 **`ToolsTrapHandler` 导出表的 Slot 20** 获取当前 Context 的 Trap Handler 内部句柄 `ti.handle`；
2. 调用 **`ToolsMemory` 导出表的 Slot 24 与 Slot 25**，由 `ti.handle` 解析出 `.text.do_pause` 段的 GPU 虚拟基地址 **`tools_pc`** 以及段大小 **`tools_sz = 0x1300`（`4,864` 字节）**；
3. 在 `.text.traphandler` 的 `+0x0390` 处有一条固定调用指令 `CALL.ABS.NOINC tools_pc`，由此确定主入口段基址为 **`handler_va = tools_pc - 0x1000`**。

### 2.2 全芯片 7 大 Trap 代码段扫描与三级代码洞（Level 1 ~ Level 3）精确地图
我们对 `handler_va` 周边的 `64 KiB` 指令虚拟地址空间进行了完整二进制转储与 `nvdisasm -b SM120` 反汇编（详见 [`mmio/spill/live_handler.sass`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/mmio/spill/live_handler.sass)），在 7 个官方子例程段中开辟并串联了 **三级可执行代码洞**：

| 代码洞级别 | 符号名 | 所属官方代码段 | 相对 `handler_va` / `tools_pc` 精确偏移 | 可用容量（字节 / 128-bit 指令数） | 当前已使用指令数 | 承载的核心功能模块 |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **入口跳板** | `do_pause` | `.text.do_pause` 首部 | `tools_pc + 0x0000` (`handler_va + 0x1000`) | `16 B` (`1` 条指令) | `1` 条 | 注入 `BRA (tools_pc + 0x0980)`，将官方 `+0x0390` 的调用无损劫持进一级代码洞 |
| **第一级代码洞 (L1)** | **`ext_code`** | `.text.do_pause` 尾部 | **`tools_pc + 0x0980 .. +0x12ff`**<br>(`handler_va + 0x1980 .. +0x22ff`) | **`2,432 B`** (`152` 条指令槽位) | `138` 条 (`idx 0..137`) + 尾部 `exit_stub` / `pause_tramp` | - `R2 & 0x8` 原因过滤与 `c[0x0][0x170]` Target 签名校验<br>- Mode 1/2/3/4/5 命令字 (`R1`) 六路分发<br>- Mode 5 全 CTA 并行 `64 KiB` SMEM 保存、`112B` 硬件 Trap 栈帧保存、`UR4..UR15` 与 `B2R.WARP` 保存 |
| **第二级代码洞 (L2)** | **`ext2_code`**<br>+ **`entry_tramp`** | `.text.traphandler` 尾部 | **`handler_va + 0x0860 .. +0x11ff`**<br>(`ext2_va = handler_va + 0x0860`) | **`2,464 B`** (`154` 条指令槽位) | `63` 条 (`idx 0..62`) + `15` 条 (`idx 64..78` `entry_tramp`) | - `path_a_restore` (`idx 0..40`)：协同恢复 `64 KiB` SMEM、`112B` 栈帧、`UR4..UR15`<br>- `path_a_notify` (`idx 41..62`)：多 Warp 原子计数、`2*W` 恢复屏障、`BAR.SYNC.DEFER_BLOCKING 0x0` 重建<br>- `entry_tramp` (`idx 64..78`, `handler_va + 0x0c60`)：普通态三态条件入口蹦床 (`State A/B/C`) |
| **第三级代码洞 (L3)** | **`ext3_code`** | `.text.do_preempt_restore` 尾部 | **`handler_va + 0x2520 .. +0x2eff`**<br>(`ext3_va = handler_va + 0x2520`) | **`2,528 B`** (`158` 条指令槽位) | `140` 条 (`idx 0..139`) | - `clean_exit_redirect` (`idx 0..11`)：SIMT 分支分化退场中和器<br>- `ext3_save_regs` (`idx 12..75`)：无分支零间隙保存 `R14..R[num_regs-3]`（最高 `R252` 即 `255` 满配寄存器）<br>- `ext3_restore_regs` (`idx 76..139`)：无分支零间隙恢复 `R14..R[num_regs-3]` |
| **备用第四/五级洞** | *(预留扩展)* | `.text.do_read_trap_reason` / `.text.do_preempt_save` 尾部 | `handler_va + 0x1b00 .. +0x23ff`<br>`handler_va + 0x5900 .. +0x61ff` | 各 **`2,304 B`** (各 `144` 条指令槽位) | `0` 条（空闲备用） | 可随时通过单条 `BRA` 串联，提供额外 `288` 条指令空间 |

> **关键发现**：在 `.text.do_pause` 中，`tools_pc + 0x0100` 到 `tools_pc + 0x0960`（`RET.ABS R12 0x20`）是驱动在进入 `do_pause` 前后调用的活跃辅助函数 `.text.do_read_trap_reason` 的一部分，**绝不能覆盖 `tools_pc + 0x0960` 之前的区域**！真正的安全代码洞严格从 **`tools_pc + 0x0980`** 开始。

---

## 3. 代码洞如何调用全局显存？如何与主机端双向交换信息？

### 3.1 代码洞内可以调用全局显存（Global Memory）吗？
**完全可以，且支持 100% 完整的 64-bit 虚拟地址全局显存读（`LD.E`）、写（`ST.E`）和原子归约操作（`REDG.E`）！**
- 在 SM120 的 `trap_mode = 1` 下，局部内存（`LDL/STL`）通过 `SETLMEMBASE` 映射到了 Trap 专属窗口，但**全局内存地址空间（Global Memory Virtual Address Space）与普通用户态 Kernel 100% 相同且完全互通**。
- 我们在代码洞中使用强一致性 GPU 作用域修饰符（`.E.STRONG.GPU`）：
  - **读取主机下发的控制命令与快照数据**：`LD.E.STRONG.GPU R1, [R8.64]` 或 128-bit 向量读 `LD.E.128 R8, [R6.64 + off]`；
  - **向主机或显存池写回状态与寄存器快照**：`ST.E.STRONG.GPU [R8.64], R0` 或 128-bit 向量写 `ST.E.128 [R6.64 + off], R8`；
  - **多 Warp / 多 SM 并发原子计数与置位**：`@P0 REDG.E.ADD.STRONG.GPU [R8.64 + 0x4], R0`（原子加 1）与 `@P0 REDG.E.OR.STRONG.GPU [R8.64], R0`（原子按位或）。

### 3.2 代码洞如何与主机端实时交换控制命令与反馈状态？
我们通过一块由 `cuMemHostAlloc(..., CU_MEMHOSTALLOC_DEVICEMAP)` 分配的 **零拷贝主机映射固定内存（Zero-Copy Mapped Pinned Memory）或统一内存缓冲区 `control_buf`**（GPU 侧 64-bit 虚拟地址为 **`control_va`**）作为主机（CPU）与代码洞（GPU Trap Handler）之间的高速共享信箱：

```
 主机端 (CPU 调度器线程)                     PCIe / NVLink 一致性映射                    GPU 代码洞 (ext_code / ext2_code)
┌────────────────────────────┐                                                       ┌───────────────────────────────────┐
│ 1. 写 control[0] = cmd     │ ────────────── Zero-Copy / Unified VA ──────────────► │ 1. LDC R8:R9, c[0x0][0x170]       │
│    (1/2/3/4/5/6/7)         │                                                       │ 2. LD.E.STRONG.GPU R1, [R8.64]    │
│ 2. 写 BAR0[0x424684] 脉冲  │ ────────────── PCIe BAR0 MMIO (< 2 µs) ─────────────► │ 3. 按 R1 分发执行保存/恢复/休眠   │
│ 3. 轮询 control[4+2*b+0/1] │ ◄───────────── REDG.E.ADD / ST.E.STRONG.GPU ───────── │ 4. 写回 cta_w[b]=W, cta_ack[b]+=1 │
└────────────────────────────┘                                                       └───────────────────────────────────┘
```

#### `control_va` 共享控制区内存布局（32-bit 字数组 `uint32_t control[]`）：
| 32-bit 字下标 | 字节偏移 (`+hex`) | 写入方 $\rightarrow$ 读取方 | 字段名称与精确协议语义 |
| :---: | :---: | :---: | :--- |
| **`control[0]`** | `+0x00` | **Host $\rightarrow$ GPU Cave** | **全局命令字 (`cmd`)**：<br>- `0`：正常放行 / 唤醒休眠的 Warp (`Resume`)<br>- `1`：Mode 1 整流早期退出 (`Early-Exit`)<br>- `2`：Mode 2 整流進入 `NANOSLEEP` 休眠 (`Per-Stream Pause`)<br>- `3`：Mode 3 按 CTA 列表选择性休眠 (`Per-CTA Pause`)<br>- `4`：Mode 4 触发安全点显存请求置位 (`Checkpoint Signal`)<br>- `5`：Mode 5 全状态换出并退出 (`kTrapFrameEvictCommand`)<br>- `6`：Mode 5 全状态恢复并 `RTT` 瞬移 (`kTrapFrameRestoreCommand`)<br>- `7`：Mode 5 多波次未启动 CTA 快速排空 (`kTrapFrameDrainCommand`) |
| **`control[1]`** | `+0x04` | **GPU Cave $\rightarrow$ Host** | **Trap 触发确认字 (`ack_word`)**：Mode 1/2/3 下进入代码洞的受管 Warp 写入 `1`，向主机确认已收到中断 |
| **`control[2]`** | `+0x08` | **GPU Cave $\rightarrow$ Host** | **调试/遥测状态字**：记录最后进入 Trap 的 `cta_id` 或状态诊断码 |
| **`control[3]`** | `+0x0c` | — | 对齐保留字 |
| **`control[4 + 2*b + 0]`** | `+0x10 + 8*b` | **Dual (Mode 3 Host $\rightarrow$ GPU / Mode 5 GPU $\rightarrow$ Host)** | - **在 Mode 3 下**：Host 写入 `1` 表示允许 `cta_id == b` 继续运行，写 `0` 表示暂停该 CTA；<br>- **在 Mode 5 下 (`cta_w[b]`)**：被 `cmd=5` 换出的活跃 CTA 写入该块的 Warp 总数 **`W`**；被 `cmd=7` 排空的未启动 CTA 保持为 **`0`**。 |
| **`control[4 + 2*b + 1]`** | `+0x14 + 8*b` | **GPU Cave $\rightarrow$ Host & Entry Tramp** | **每 CTA Warp 完成原子计数器 (`cta_ack[b]`)**：<br>- 每个完成换出（`cmd=5`）或排空（`cmd=7`）的 Warp 由 Lane 0 执行 `REDG.E.ADD [R8+4], 1`；当 **`cta_ack[b] == W`** 时，Host 确知该 CTA 所有 Warp 已 100% 完成保存并退场！<br>- 在 `cmd=6` 恢复阶段，每个完成 SMEM 写入的 Warp 再次原子加 1；当 **`cta_ack[b] == 2 * W`** 时，全 CTA 跨过恢复屏障执行 `RTT.FALLTHROUGH`！ |

---

## 4. 如何指定抢占的 Target ID（只抢占特定 Stream/Kernel，不影响其他流）？

当主机向 BAR0 `0x00424684` 发送硬件脉冲时，该脉冲是一个**全芯片广播信号**——此时 GPU 上正在运行的所有 Stream（包括高优先级 Stream 和无关任务）的活跃 Warp 都会在 `< 1 µs` 内进入 `.text.traphandler`。

那么，**代码洞是如何精确识别“谁是要被抢占的 Target Stream/Kernel，谁是应当立即无损放行的高优先级任务”的？**

### 4.1 核心机制：`cuXtraSetDebuggerParams` 与常量缓存 `c[0x0][0x170..0x18f]` 签名绑定
在 CUDA 架构中，每个发射的 `CUfunction` 都有自己独立的 **Constant Bank 0 (`c[0x0]`)**。其中偏移 **`0x170 .. 0x18f`（共 32 字节）** 是驱动预留给调试器/工具链的私有槽位（在普通 Kernel 中全为 `0`）。

1. **主机端绑定 Target (`BindKernelControl` / `BindKernelTrapFrameEvict`)**：
   我们对受管低优先级 Stream 即将发射的 `CUfunction` 调用 `cuXtraSetDebuggerParams(func, &params, 28)`，将 28 字节的专属控制结构体注入该函数的 `c[0x0][0x170..0x18b]` 中：
   - **`c[0x0][0x170..0x177]`**：绑定该 Stream/Kernel 专属的 **`control_va`**（若为每个 Stream 分配独立的 `control_buf`，即可实现 **100% 独立的 Per-Stream Target ID 控制**！主机想抢占哪个 Stream，就只改写哪个 Stream 对应的 `control_va[0]`！）；
   - **`c[0x0][0x184..0x187]`**：写入 32-bit 魔数签名 **`magic`**（Mode 1/2/3 为 `0x31323046` `"F021"`，Mode 4 为 `0x80000000`，Mode 5 为 `0x40000000`）。
2. **代码洞内极速过滤与放行（`ext_code idx 5..14`）**：
   当全芯片所有 Warp 进入 `ext_code` 后：
   - **未绑定的无关/高优先级 Kernel**：其 `c[0x0][0x170]` 为 `0`（或 `magic` 不匹配），代码洞在 `idx 10` (`ISETP.EQ P0, R8, RZ`) 检测到 `R8 == 0` 后，直接跳至 `done_r1` (`idx 57`)，清除 `R2` 的 `0x8` 位并在 `< 0.5 µs` 内通过 `RTT.FALLTHROUGH` 原地返回继续执行，**受到的干扰为 0**！
   - **已绑定但非本次抢占目标的受管 Stream**：虽然 `c[0x0][0x170]` 非零，但主机并未修改该 Stream 的 `control_va[0]`（仍为 `0`），代码洞读取 `[control_va] == 0` 后同样在 `< 0.5 µs` 内立即返回！
   - **本次抢占的目标 Target Stream**：其 `control_va[0]` 被主机设为了非零命令字（如 `5`），立即命中对应分支执行换出！

---

## 5. 代码洞如何获取 Kernel 端信息（如 Kernel ID、线程块维度、寄存器数与用户参数）？

在代码洞（`trap_mode = 1` 或入口蹦床 `trap_mode = 0`）中，我们无需修改用户 Kernel 传参，即可通过 **Constant Bank 0 (`c[0x0]`) 系统固定偏移** 与 **硬件特殊寄存器（Special Registers, `SR_*`）** 读取当前被中断 Kernel 的全部元数据：

### 5.1 从 Constant Bank 0 (`c[0x0]`) 读取的 Kernel 元数据与注入参数

在 SM120 上，通过 `LDC.U32 Rd, c[0x0][offset]` 或 `LDC.64 Rd, c[0x0][offset]` 可直接读取以下关键信息：

| 常量缓存地址 | 位宽 | 来源 | 存放的信息内容与代码洞中的用途 |
| :--- | :---: | :--- | :--- |
| **`c[0x0][0x170 .. 0x177]`** | 64-bit | `cuXtraSetDebuggerParams` 注入 | **`control_va`**：当前 Kernel/Stream 绑定的主机-设备共享控制区虚拟地址（同时充当 Stream/Kernel Target 标识） |
| **`c[0x0][0x178 .. 0x17f]`** | 64-bit | `cuXtraSetDebuggerParams` 注入 | **`arena_va`（Mode 5）或 `request_va`（Mode 4）**：当前 Kernel 绑定的显存全状态快照池基地址 |
| **`c[0x0][0x180 .. 0x183]`** | 32-bit | `cuXtraSetDebuggerParams` 注入 | **`num_regs` (`SR_REGALLOC`)**：主机通过 `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)` 查询并注入的每线程硬件分配寄存器总数（`1 .. 255`） |
| **`c[0x0][0x184 .. 0x187]`** | 32-bit | `cuXtraSetDebuggerParams` 注入 | **`magic` 模式签名**：标识当前 Kernel 启用的是 Mode 1/2/3 (`0x31323046`)、Mode 4 (`0x80000000`) 还是 Mode 5 (`0x40000000`) |
| **`c[0x0][0x188 .. 0x18b]`** | 32-bit | `cuXtraSetDebuggerParams` 注入 | 自定义扩展字 / `Kernel ID`（可由调度器填入任意 32-bit `kernel_id` 或动态 SMEM 大小） |
| **`c[0x0][0x358 .. 0x35f]`** | 64-bit | **SM120 硬件/驱动自动填入** | 运行时全局状态描述符指针（用户态序言常加载至 `UR6:UR7`） |
| **`c[0x0][0x360 .. 0x363]`** | 32-bit | **SM120 硬件/驱动自动填入** | **`blockDim.x`**：线程块 X 维度大小（`ext_code idx 78` 读取用于计算线性 `tid` 与 `threads_per_cta`） |
| **`c[0x0][0x364 .. 0x367]`** | 32-bit | **SM120 硬件/驱动自动填入** | **`blockDim.y`**：线程块 Y 维度大小（`ext_code idx 79` 读取，支持 2D 多 Warp 线程块如 `(32, 4)`） |
| **`c[0x0][0x368 .. 0x36b]`** | 32-bit | **SM120 硬件/驱动自动填入** | **`blockDim.z`**：线程块 Z 维度大小 |
| **`c[0x0][0x36c .. 0x377]`** | 3×32-bit | **SM120 硬件/驱动自动填入** | **`gridDim.x`、`gridDim.y`、`gridDim.z`**：网格维度大小 |
| **`c[0x0][0x378 .. ]`** | 变长 | **SM120 硬件/驱动自动填入** | **用户 Kernel 的原始形式参数列表（Kernel Arguments）**：按 C++ 参数声明顺序依次存放用户传入的设备指针与标量参数！ |

### 5.2 从硬件特殊寄存器 (`S2R Rd, SR_*`) 读取的实时线程/CTA 拓扑身份

在代码洞中，通过 `S2R`（Special Register to Register）指令直接读取当前正在执行 Trap 的线程身份：

| 特殊寄存器名称 | SR 编号 (`imm`) | 对应 CUDA 内置变量 | 在代码洞中的精确用途 |
| :--- | :---: | :--- | :--- |
| **`SR_LANEID`** | `0x00` (`0`) | `%laneid` (`0 .. 31`) | 在 `ext_code` / `ext2_code` 中选出每个 Warp 的 **Lane 0 (`lane_id == 0`)** 执行单次原子计数 `REDG.E.ADD` |
| **`SR_TID.X`** | `0x21` (`33`) | `threadIdx.x` | 读取块内 X 线程号（`ext_code idx 76`） |
| **`SR_TID.Y`** | `0x22` (`34`) | `threadIdx.y` | 读取块内 Y 线程号（`ext_code idx 77`），结合 `blockDim.x` 计算块内一维线程号 `tid = tid.y * blockDim.x + tid.x` |
| **`SR_CTAID.X`** | `0x25` (`37`) | `blockIdx.x` (`cta_id`) | 读取当前 CTA 在网格中的块索引 `cta_id`（`ext_code idx 15` 与 `entry_tramp idx 2`），用于索引 `control[4 + 2*cta_id]` 及计算全局线程号 `global_tid = cta_id * threads_per_cta + tid` |
| **`SR_GLOBALERRORSTATUS`** | — | 全局错误/Trap 状态 | 官方 `.text.traphandler +0x0210` 读取以判断是否由调试 Trap 触发 |
