# Q&A 专题四：抢占与恢复状态机更新逻辑、端到端调用方式与分步耗时/效率定量拆解

> **对应问题**：  
> *“恢复流程和状态机的更新逻辑，抢占和恢复的逻辑和调用方式分别是什么？详细介绍这个调用逻辑，并介绍每一步花费的时间和效率？”*

---

## 1. 四层核心状态机（State Machines）的定义与更新逻辑

在 Universal Mode 5 (`trapframe`) 的抢占（Evict）与恢复（Restore）全生命周期中，系统通过 **4 个相互咬合的硬件-软件协同状态机** 保证任意多 Warp、任意多波次网格在并发中断下的绝对一致性：

### 1.1 状态机一：全局控制命令状态机（`u_control[0]`，由 Host 驱动）

```
  [0: 正常原速运行]
         │
         │ 1. 高优任务到达，Host 写 u_control[0] = 5 并触发 BAR0 脉冲
         ▼
  [5: kTrapFrameEvictCommand]  (驻留的活跃波次 State B 保存 64 KiB SMEM + 全寄存器 + 112B 栈帧后 EXIT)
         │
         │ 2. 当驻留的 active_ctas (如 30 个 CTA) 的 cta_ack[b] == W 全部确认后
         ▼
  [7: kTrapFrameDrainCommand]  (若 total_ctas > active_ctas，快速排空后续被 CWD 推上 SM 的未启动波次 State C)
         │
         │ 3. 旧低优 Stream 100% 退役，高优任务独享全部 SM 空间配额运行完毕；
         │    Host 热补丁入口 func_pc -> BRA entry_tramp，重发原 Kernel，写 u_control[0] = 6 并触发脉冲
         ▼
  [6: kTrapFrameRestoreCommand] (State B 从 u_arena 装回全部状态并经 2*W 屏障与 BAR.SYNC 重建后执行 RTT.FALLTHROUGH)
         │
         │ 4. 当所有 State B CTA 的 cta_ack[b] >= 2*W 确认恢复完毕，Host 还原 func_pc 原入口指令
         ▼
  [0: 恢复正常原速运行] (State B 从中断 PC 继续跑完，State C 走 entry_tramp 快车道从头跑完)
```

---

### 1.2 状态机二：每线程幂等防重入状态机（`u_arena` 中 `[R6 + 0x70]` 的 `status` 字）

由于多 Warp / 多波次 CTA 在排空或恢复期间可能收到多次 BAR0 Trap 脉冲（间隔 `15 µs`），每个线程利用其栈帧偏移 `+0x70` 处的 `status` 字防止重复保存或重复恢复：

| 当前 `status` (`[R6+0x70]`) | 收到 `cmd = 5` (Evict) 时的动作与状态转移 | 收到 `cmd = 7` (Drain) 时的动作与状态转移 | 收到 `cmd = 6` (Restore) 时的动作与状态转移 |
| :---: | :--- | :--- | :--- |
| **`0`（初始态）** | **执行完整保存**：写入 `status = 5`，转储 SMEM、112B 栈帧、`UR4..15`、`B2R`、`R14..252`，写 `cta_w = W`、`cta_ack += 1`，经 `clean_exit_redirect` 执行 `EXIT` | **执行快速排空**：写入 `status = 7`，**跳过全部 SMEM 与寄存器转储**，保持 `cta_w == 0`，执行 `cta_ack += 1` 后直接 `EXIT` | 防御性跳至 `clean_exit_redirect` 执行 `EXIT` |
| **`5`（已换出态）** | 幂等跳过：检测到 `status != 0`（`ext_code idx 95`），直接 `BRA done_r1` 返回 | 幂等跳过：直接 `BRA done_r1` 返回 | **执行完整恢复**：将 `status` 更新为 **`6`**（`ext2_code idx 4`），恢复 SMEM、112B 栈帧、`UR4..15`、`R14..252`，`cta_ack += 1`，等待 `2*W` 屏障后执行 `RTT.FALLTHROUGH`！ |
| **`6`（已恢复态）** | — | — | **幂等保护（关键！）**：某 Warp 恢复后已回到用户循环继续运行，若此时又收到针对其他慢速 CTA 的 `cmd=6` 脉冲，在 `ext2_code idx 2` 检测到 `status (6) != 5`，**直接跳至 `done_r1` 原地返回用户代码继续运行**，绝不二次覆盖寄存器！ |

---

### 1.3 状态机三：多波次网格 CTA 三态分流状态机（`u_control[4 + 2*b + 0/1]` 的 `(w, ack)` 二元组）

在一个超配多波次网格（例如 `60` 个 CTA 在 `30` 个 SM 上分多波次运行）中，抢占发生时网格内的 CTA 必然处于以下 **三种状态（State A / State B / State C）** 之一。在恢复阶段重发网格时，`func_pc + 0x00` 处的 `BRA entry_tramp` 将所有 CTA 引入 15 条指令的 **`entry_tramp`**（`handler_va + 0x0c60`），按 `(w, ack)` 精确分流：

| CTA 类别 | 物理含义 | 换出阶段结束时的 `(cta_w[b], cta_ack[b])` | 恢复阶段进入 `entry_tramp` 时的精确分流路径 |
| :--- | :--- | :---: | :--- |
| **State A**<br>*(Pre-Completed)* | **抢占发生前就已经自然执行完 `EXIT` 的早期波次 CTA** | **`w == 0, ack == 0`**<br>*(从未进入过换出/排空 Trap)* | `entry_tramp idx 4..7`：读取 `ack` 发现 **`ack == 0`** $\rightarrow$ **在 `idx 7` 直接执行单条 `EXIT` 瞬间退场**（耗时 `< 0.2 µs`，0 重复计算、0 非幂等副作用污染！） |
| **State B**<br>*(Evicted Mid-Run)* | **抢占发生时正在 SM 上运行、被 `cmd = 5` 保存了完整快照的活跃 CTA** | **`w == W (>0), ack == W (>0)`** | `entry_tramp idx 8..12`：发现 `ack != 0 && w != 0` $\rightarrow$ **进入 `NANOSLEEP 0x40; BRA -2` 自旋等待**，随后捕获 `cmd = 6` BAR0 Trap，装回快照并通过 `RTT.FALLTHROUGH` **直接瞬移回被打断的内层循环 PC**！ |
| **State C**<br>*(Drained Unstarted)* | **抢占时在硬件 CWD 队列排队、在换出阶段被 `cmd = 7` 快速排空的未启动 CTA** | **`w == 0, ack == W (>0)`** | `entry_tramp idx 8..14`：发现 **`ack != 0 && w == 0`** $\rightarrow$ **跳过 `NANOSLEEP` 自旋**，直接在 `idx 13` 原地执行被覆盖的原入口首指令 `orig_entry_inst`，并在 `idx 14` 执行 **`BRA (func_pc + 0x10)` 从头全速原生运行**！ |

---

### 1.4 状态机四：多 Warp CTA 内双阶段协同屏障（`cta_ack[b]: W -> 2*W`）与硬件 `BAR.SYNC` 重建状态机

当一个 CTA 包含多个 Warp（如 `W = 4` 个 Warp = 128 线程）时：
1. **第一阶段（换出完成计数 `0 -> W`）**：每个完成保存的 Warp 对 `cta_ack[b]` 原子加 1，到达 `W` 时标志全块保存完毕。
2. **第二阶段（恢复期跨 Warp SMEM 写入栅栏 `W -> 2*W`，`ext2_code idx 50..58`）**：
   - 在恢复阶段，4 个 Warp 并行向同一个 `64 KiB` Shared Memory 执行 `STS` 写入；
   - 如果 Warp 0 先写完自己的 1/4 SMEM 就立即执行 `RTT.FALLTHROUGH` 返回用户代码，它可能立刻在用户代码里读取或修改属于 Warp 1..3 **尚未从显存恢复完**的 SMEM 数据！
   - 因此，每个写完 SMEM 和寄存器的 Warp 必须对 `cta_ack[b]` 再次原子加 1，并在 `ext2_code idx 56..58` **自旋等待 `cta_ack[b] >= 2 * W`**，确保全块所有 Warp 的 `STS` 100% 落盘后才放行！
3. **第三阶段（跨 Warp 硬件命名屏障 0 重建，`ext2_code idx 59..61`）**：
   - 若被抢占瞬间，Warp 0 已经执行了用户循环里的 `__syncthreads()` (`BAR.SYNC 0`) 正在等待尚未到达的 Warp 1..3，则 Warp 0 保存的 `[R6 + 0x74]` (`B2R.WARP`) 值满足 `>= 0x1000`；
   - 在跨过 `2 * W` 栅栏后、跳转官方尾声前，所有 `[R6 + 0x74] >= 0x1000` 的 Warp 在 `trap_mode = 1` 下执行 **`@P0 BAR.SYNC.DEFER_BLOCKING 0x0`**，向新 SM 的硬件屏障单元重新注册“本 Warp 已到达 0 号屏障”并将阻塞生效时机推迟到 `RTT.FALLTHROUGH` 之后，从而与稍后运行到用户态 `BAR.SYNC 0` 的兄弟 Warp 完美汇合！

---

## 2. 抢占与恢复的完整调用方式（C++ API 签名与调用代码范式）

所有底层细节均已封装在 [`TrapSm120Context`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/xsched/platforms/cuda/hal/src/level3/trap_sm120.cpp) 类中，上层调度器或应用程序只需调用以下 **5 个简洁的 C++ 接口**：

### 2.1 核心 API 列表
```cpp
// 1. [一次性初始化] 初始化 SM120 硬件 Trap 上下文（钉住 NV83DE、映射 BAR0、注入三级代码洞）
bool InitContext();

// 2. [发射前绑定] 将目标 CUfunction 绑定到 Mode 5 空间抢占引擎（自动查询 num_regs 生成 ext3_code 并注入 c[0x0][0x170..0x187]）
bool BindKernelTrapFrameEvict(CUfunction kernel, uint32_t *host_control_words,
                              uint64_t control_gpu_va, uint64_t arena_gpu_va);

// 3. [抢占换出] 触发硬件 Trap 将正在运行的低优 Stream 全状态换出至 u_arena 并彻底退出释放 100% SM 空间配额
bool RequestStreamTrapFrameEvict(uint32_t *host_control_words,
                                 uint32_t active_ctas,   // 当前物理驻留 CTA 数 (如 30)
                                 uint32_t total_ctas,    // 网格总 CTA 数 (如单波次 30 或多波次 60)
                                 uint32_t timeout_us,
                                 double *mmio_us_out = nullptr,
                                 double *evict_us_out = nullptr);

// 4. [恢复前准备] 在重发原 Kernel 前，将其入口首指令热补丁为 BRA entry_tramp 并刷新 I-Cache
bool PrepareKernelTrapFrameRestore(CUfunction kernel, uint64_t *saved_inst0_lo, uint64_t *saved_inst0_hi,
                                   uint64_t *saved_inst1_lo, uint64_t *saved_inst1_hi);

// 5. [硬件 RTT 恢复 + 还原入口] 向重发的 Stream 发送 cmd=6 恢复脉冲，等待全部被换出 CTA 通过 RTT.FALLTHROUGH 瞬移回原中断 PC，随后还原入口指令
bool RequestStreamTrapFrameRestore(uint32_t *host_control_words, uint32_t expected_ctas,
                                   uint32_t timeout_us, double *mmio_us_out = nullptr,
                                   double *restore_us_out = nullptr);
bool FinishKernelTrapFrameRestore(CUfunction kernel, uint64_t saved_inst0_lo, uint64_t saved_inst0_hi,
                                  uint64_t saved_inst1_lo, uint64_t saved_inst1_hi);
```

### 2.2 标准端到端调用时序范式（以抢占低优流 `low`、运行高优流 `high`、再恢复 `low` 为例）

```cpp
// ==================== 阶段 0：初始化与绑定 ====================
trap_ctx.InitContext();
trap_ctx.BindKernelTrapFrameEvict(low_func, u_control,
                                  (uint64_t)(uintptr_t)u_control,
                                  (uint64_t)(uintptr_t)u_arena);

// 低优任务在 stream low 上正常启动（0 插桩，100% 原生速度运行）
universal_highreg_kernel<<<30, dim3(32, 4), 65536, low>>>(d_out, kSteps);

// ==================== 阶段 1：高优任务到达，发起空间抢占换出 ====================
// 也可以先把高优任务 enqueue 到 stream high（硬件 CWD 会在首个低优 CTA 退出后的 ~12 us 内立即调度高优 CTA 上 SM！）
high_priority_kernel<<<30, 32, 65536, high>>>(...);

double mmio_evict_us = 0, evict_us = 0;
trap_ctx.RequestStreamTrapFrameEvict(u_control, /*active_ctas=*/30, /*total_ctas=*/30,
                                     /*timeout_us=*/100000, &mmio_evict_us, &evict_us);
cudaStreamSynchronize(low);  // 此时 low 已 100% 退场，寄存器/SMEM/Warp 占用为 0！
cudaStreamSynchronize(high); // 高优任务无阻塞极速完成！

// ==================== 阶段 2：高优完成，无损瞬移恢复低优任务 ====================
uint64_t s0_lo = 0, s0_hi = 0, s1_lo = 0, s1_hi = 0;
trap_ctx.PrepareKernelTrapFrameRestore(low_func, &s0_lo, &s0_hi, &s1_lo, &s1_hi);

// 用完全相同的参数重新发射原低优 Kernel
universal_highreg_kernel<<<30, dim3(32, 4), 65536, low>>>(d_out, kSteps);

double mmio_rest_us = 0, rest_us = 0;
trap_ctx.RequestStreamTrapFrameRestore(u_control, /*expected_ctas=*/30,
                                       /*timeout_us=*/100000, &mmio_rest_us, &rest_us);
trap_ctx.FinishKernelTrapFrameRestore(low_func, s0_lo, s0_hi, s1_lo, s1_hi);

cudaStreamSynchronize(low); // 低优任务从被打断的 SASS 指令处精确跑完剩余迭代，0 bit error！
```

---

## 3. 每一步花费的时间与效率定量拆解（RTX 5060 `SM120` 实测数据）

我们在 RTX 5060（`30` 个物理 SM，GDDR7 显存理论峰值带宽 `448 GB/s`）上，对 Universal Mode 5 在不同负载下的**每一个微操作阶段**进行了微秒级计时拆解：

### 3.1 抢占换出阶段（Evict Phase）分步耗时与带宽效率拆解

以 4 Warps/CTA (`128` 线程/块)、30 个物理驻留 CTA 占满全芯片 `30 × 64 KiB = 1.92 MiB` Shared Memory + `30 × 128 × 1152 B = 4.22 MiB` 线程栈帧与满配寄存器为例：

| 步骤序号 | 子阶段名称 | 实测耗时 (`µs`) | 底层发生的微架构动作与效率分析 |
| :---: | :--- | :---: | :--- |
| **Step E1** | **Host 写 `cmd=5` + BAR0 `0x00424684` 双写脉冲** | **`1.89 ~ 2.40 µs`** | 2 次 PCIe BAR0 32-bit 写 + 1 次同步读刷新写缓冲 + GPC PRI 广播至 30 个 SM |
| **Step E2** | **首个 SM 完成单 CTA 状态转储并执行 `EXIT` $\rightarrow$ 首个高优 CTA 抢占入场 (`High CTA Admission`)** | **`12.26 ~ 22.91 µs`**<br>*(35 regs: `12.26 µs`)*<br>*(255 regs: `22.91 µs`)* | **核心调度效率亮点**：高优流无需等待全部 30 个低优 CTA 退场！只要最快的 1 个物理 SM 完成 `64 KiB` SMEM + 寄存器转储并执行 `EXIT`（仅需 **`~10 µs`**），硬件 CWD 立即将高优流的首个 CTA 调度进该 SM 开始计算！ |
| **Step E3** | **全芯片 30 个 CTA 完成 `64 KiB` SMEM + 全寄存器转储 (`Evict Signal`)** | **`70.59 ~ 73.09 µs`**<br>*(35 regs: `70.59 µs`)*<br>*(255 regs: `73.09 µs`)* | - **驚人的寄存器扩展效率**：从保存 `35` 个寄存器增加到保存架构极限的 **`255` 个满配寄存器**（每线程多搬运 `880 B`，全芯片多搬运 `3.38 MiB`），由于 `ext3_code` 采用了零分支 16B 对齐 `ST.E.128` 向量突发写入，耗时仅从 `70.59 µs` 增加到 **`73.09 µs`（仅增加 `+2.50 µs`）**！ |
| **Step E4** | *(仅多波次网格)* **`cmd=7` 未启动波次快速排空 (`Fast Drain`)** | **`+54.17 µs`**<br>*(60 CTAs 总计 `127.26 µs`)* | 当驻留的 30 个 State B CTA 退出后，等待在 CWD 队列中的 20 个未启动 State C CTA 被推上 SM，捕获 `cmd=7` Trap 后**零显存写入直接 `EXIT`**（相比优化前傻存 64 KiB SMEM 的 `439.74 µs` **提速 `3.46x`**） |
| **Step E5** | **低优 CUDA Stream 彻底退役 (`Stream Drain`)** | **`+0.0 ~ 1.1 µs`**<br>*(与 Step E3/E4 几乎同时)* | 所有 CTA 执行完 `exit_stub_va` 处的 `EXIT` 后，`cudaStreamQuery(low)` 立即变为 `cudaSuccess` |

---

### 3.2 恢复阶段（Restore Phase）分步耗时与效率拆解

| 步骤序号 | 子阶段名称 | 实测耗时 (`µs`) | 底层发生的微架构动作与效率分析 |
| :---: | :--- | :---: | :--- |
| **Step R1** | **`PrepareKernelTrapFrameRestore`（热补丁入口 + 刷 I-Cache）** | **`~15 µs`** | 通过 `cuXtraMemcpyHtoD` 将 `func_pc + 0x00` 的 16 字节替换为 `BRA entry_tramp` 并调用 `cuXtraInvalInstrCache` |
| **Step R2** | **重发原 Kernel + 三态 `entry_tramp` 普通态分流** | **`~10 ~ 25 µs`** | - **State A（已完成 CTA）**：检测到 `ack == 0` 在 `< 0.2 µs` 内直接 `EXIT`；<br>- **State C（未启动 CTA）**：检测到 `w == 0` 直接跳回 `func_pc + 0x10` 原生执行；<br>- **State B（被换出 CTA）**：检测到 `w > 0` 进入 `NANOSLEEP` 等待恢复脉冲（使多波次恢复延迟从 `5,538.7 µs` 骤降至 **`126.08 µs`，提速 `43.9x`**！） |
| **Step R3** | **发送 `cmd=6` BAR0 脉冲 + 全芯片装回 `64 KiB` SMEM 与 `1..255` 寄存器 + `2*W` 屏障 + `BAR.SYNC` 重建 + `RTT.FALLTHROUGH` 瞬移** | **`126.08 ~ 147.97 µs`**<br>*(255 regs: `144.90 µs`)* | 全芯片 30 个 CTA 并行从 `u_arena` 读回 `1.92 MiB` SMEM 写入片上 `[0x400..0x103ff]`，装回 `112B` 栈帧、`UR4..15` 与 `R14..R252`，跨过 `2*W` 屏障并重建 0 号硬件屏障后，执行硬件 `RTT.FALLTHROUGH` 直接瞬移回中断时的内层 SASS 指令继续运行！ |
| **Step R4** | **`FinishKernelTrapFrameRestore`（还原 `func_pc` 原入口指令）** | **`~15 µs`** | 将 `func_pc + 0x00` 还原为原始首条指令并刷新 I-Cache，后续新发射的 Kernel 恢复 100% 零修改形态 |

---

## 4. 全项目各抢占模式横向效率总览表

| 模式名称 | 正常无抢占运行开销 (`O1`) | Trap 脉冲触发延迟 | 高优首 CTA 抢占入场延迟 (`P50`) | 低优全芯片排空延迟 (`P50`) | 全芯片恢复延迟 (`P50`) | 是否释放 100% 空间配额？ | 是否零用户源码修改？ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode 2 (`NANOSLEEP` 休眠)** | `0.00%` | `2.02 µs` | *(空间满载时阻塞)* | `2.85 µs` *(仅停 ALU)* | `26.66 µs` | ❌ 否 (仅让出 ALU) | ✅ **是** |
| **Mode 4 (`Checkpointing` 安全点)** | `+0.02% ~ +0.09%` | `1.95 µs` | **`10.50 ~ 12.00 µs`** | **`24.83 ~ 27.36 µs`** | **`6.78 ~ 9.73 µs`** | ✅ **是 (100% 释放)** | ❌ 否 (需轻量安全点) |
| **Mode 5 (`trapframe` 单 Warp `14 regs`)** | **`0.00%`** | `1.92 µs` | **`23.52 µs`** | **`40.51 µs`** | **`127.30 µs`** | ✅ **是 (100% 释放)** | ✅ **是 (100% 零修改)** |
| **Mode 5 (`trapframe-univ-1w` `35 regs` 4-Warp)** | **`0.00%`** | `2.46 µs` | **`12.26 µs`** | **`70.59 µs`** | **`147.97 µs`** | ✅ **是 (100% 释放)** | ✅ **是 (100% 零修改)** |
| **Mode 5 (`trapframe-univ-2w` 三态多波次)** | **`0.00%`** | `2.10 µs` | **`12.80 µs`** | **`127.26 µs`** | **`126.08 µs`** | ✅ **是 (100% 释放)** | ✅ **是 (100% 零修改)** |
| **Mode 5 (`trapframe-univ-hireg` `255 regs` 满配)** | **`0.00%`** | `2.38 µs` | **`22.91 µs`** | **`73.09 µs`** | **`144.90 µs`** | ✅ **是 (100% 释放)** | ✅ **是 (100% 零修改)** |
| **跨 Context 硬件 CILP (`Resumable`)** | `0.00%` | — | `~559 µs` | `~559 µs` | *(随 Context 切换恢复)* | ✅ **是 (跨 Context)** | ✅ **是 (支持闭源库)** |
