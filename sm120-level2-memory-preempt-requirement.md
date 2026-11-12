
1. Your current Mode 2 frees only the ALUs, as your advisor said. The paused CTAs keep running NANOSLEEP, so they keep their registers, shared memory and warp slots. Mode 1 does free everything, but it loses the CTA's work.
2. The trap handler already saves part of each warp's state. On entry it writes R0–R13, the predicates, the return PC, the active mask, the convergence state and the reconvergence PC into a fixed area at 0xffff90..0xffffff in each thread's local memory. On exit it reloads all of that. So if a resumed warp re-enters the trap with that area rewritten, the handler puts back R0–R13, PC and masks in one step. This is the hard part of the review doc's Path A, and it confirms that path can work.
3. The driver has its own hardware context-save path. When the trap reason has bit 0x10 set, the handler calls a save routine and then runs EXIT.PREEMPTED. The save buffer is 30 × 0x630b0 bytes, one slice per SM. From the parts of that routine that disassembled normally, a complete save needs:
   - all allocated registers (count from SR_REGALLOC)
   - uniform registers and uniform predicates
   - convergence barriers B0–B15
   - the per-lane reconvergence PC and the thread-state masks
   - named-barrier state
   - shared memory (size from SR_SMEMSZ), copied out by one elected warp per CTA

   This tells me what my own save has to include. I won't copy the driver's code.

Proposed design: save to a global-memory arena, EXIT, then relaunch to restore

- Evict (new mode 4): the BAR0 broadcast trap fires. In trap mode, each victim warp copies its saved trap area into a global arena indexed by CTA id, warp and lane, so it is not tied to a physical SM slot. It then returns to a normal-mode stub. That stub saves R14 up to the register limit, the uniform registers, B1–B15, its share of shared memory (after a CTA-wide sync) and any local memory, then executes a plain EXIT. At that point the SM gets back all registers, shared memory and warp slots.
- CTA status tracking: an entry guard, installed with cuXtraSetEntryPoint, marks each CTA as "not started" if it is dispatched during an eviction, "saved" if it was evicted mid-run, or "done".
- Restore: relaunch the same grid with the entry point redirected to a restore stub. CTAs marked done exit at once. Not-started CTAs run from the original entry. Saved CTAs reload shared memory and the higher registers, write the saved trap area back, and trap themselves. The handler's normal exit then restores R0–R13, PC and the divergence masks, and the warp continues where it stopped. Repeated preemptions work the same way.
- Expected cost: a full SM is about 256 KB of registers plus up to about 100 KB of shared memory. Across 30 SMs that is roughly 10 MB, which should stay in the 32 MB L2 and take about tens of µs, plus relaunch time. That is well inside your "hundreds of µs" target.
- Steps, each tested on its own:
  a. Evict and restore a kernel that uses only registers (non-idempotent checksum check).
  b. Add shared memory and __syncthreads, testing the case where a warp is waiting at a barrier when the trap hits.
  c. Add divergent warps and multi-wave grids.
  d. A benchmark where the low kernel fills the SMs' registers and shared memory, so the high-priority kernel cannot launch until the low one is evicted. It measures eviction time, how long the high kernel waits to start, and restore time, compared with Mode 2.
  e. Integrate into TrapSm120Context.


Shall I go ahead with this, starting with step 1? 
# 导师评价深度解读与 SM120 寄存器/共享内存状态化抢占设计方案

结合您提供的 `preempt_rigorous_bench.cu` 源码与导师的评价——**“即使没有计算，这部分寄存器和 shm 没有让出，其他 kernel 还是发射不了，目前主要受限是寄存器和 shared memory 大小”**，可以非常清晰地定位当前系统所处的技术阶段、核心瓶颈以及下一步通往顶会/生产级系统的突破方向。

---

## 一、 导师的评价说明了什么？（结合 `preempt_rigorous_bench.cu` 诊断）

导师一针见血地指出了当前抢占实现中 **“让出计算流水线（Compute Yield）”** 与 **“让出片上存储配额（Storage Quota Eviction）”** 之间的根本矛盾。对照您的 `preempt_rigorous_bench.cu` 代码，具体说明了以下三个事实：

### 1. 当前的 Mode 2（原地暂停）只让出了“动态算力”，没有让出“静态不动产”
在您的 `ext_code`（第 $240 \sim 255$ 行）中，当 `cmd == 2`（Mode 2 In-Place Pause）时：
*   Trap Handler 将原始返回地址 `PC_orig` 从 Local Memory 栈帧 `[0xffffa8]` 备份到 `[0xffff88]`，并将 `[0xffffa8]` 改写为 `pause_tramp_va`（即执行 `NANOSLEEP 0x100; BRA -2` 的蹦床指令）。
*   **硬件后果：** 当执行 `RET.ABS R12 0x20` 退出 Trap 后，被抢占的 $30,720$ 个线程虽然停止了 FP32 FMA 计算（因此没有计算流水线争抢），但**这些线程块（CTA）并没有执行 `EXIT` 指令**。
*   在 GPU 硬件工作分发器（Compute Work Distributor, CWD）眼中，这些正在 `NANOSLEEP` 的 CTA 依然是**活跃驻留（Active Resident）**状态，它们在启动时锁定的 **物理寄存器（Register File）和共享内存（Shared Memory）配额一字节都没有退还给 SM**。

### 2. 为什么你的 `BENCH 1` 和 `BENCH 3` 能跑通，但导师却说“其他 Kernel 发射不了”？
观察 `preempt_rigorous_bench.cu` 中的测试负载配置可以发现：
*   在 `BENCH 1` 中，`heavy_non_idempotent_low_kernel` 声明了 `__launch_bounds__(768, 2)`，编译器将其每线程寄存器压得很低（约 $32 \sim 40$ 个），且代码第 $74$ 行的 `__shared__ float s_pad[512]` 仅占 $2\text{ KB}$ Shared Memory。同时它只启动了 $60 \times 512 = 30,720$ 个线程（占 RTX 5060 全卡 $46,080$ 最大线程槽位的 $66.7\%$）。因此，每个 SM 上恰好还剩约 $33\%$ 的空闲寄存器和大量空闲 Shared Memory，这才让 `high_pri_compute_burst_kernel` 得以侥幸塞进同一个 SM 并行启动。
*   **但在真实世界的高性能算子中（导师所指的场景）：**
    诸如 CUTLASS 矩阵乘法、FlashAttention、cuBLAS 或复杂科学计算算子，为了极致性能，往往单 CTA 就会吃满 **每线程 $128 \sim 255$ 个寄存器** 以及 **每 SM $64\text{ KB} \sim 100\text{ KB}$ 的 Shared Memory**。
    此时，即便低优先级任务只占了 SM 一半的线程槽位且全部处于 `NANOSLEEP` 状态（$0\%$ 计算占用），它锁死的 $128\text{ KB}$ 寄存器或 $64\text{ KB}$ Shared Memory 也会导致高优先级 Kernel 因**凑不够单个 CTA 所需的寄存器或 Shared Memory 硬性配额**，被硬件 CWD 直接拒之门外——**连 $1$ 个线程块都发射（Issue/Launch）不进 SM，高优任务完全卡死在硬件队列中！**

### 3. 当前的 Mode 1 / Mode 3 虽然能通过 `EXIT` 腾出空间，但属于“有损丢弃（Lossy Kill）”
在 `BENCH 2`（$100\%$ 槽位占满测试）中，为了解决高优任务发射不了的问题，您的代码对 CTA $0 \sim 29$ 下发了 `cmd = 1`（第 $257 \sim 266$ 行），将其 PC 重定向到 `exit_stub_va`（`EXIT; EXIT;`）。
*   这虽然强行杀死了 CTA 并释放了寄存器、Shared Memory 和线程槽位，但**那 $30$ 个 CTA 算了一半的寄存器状态（如非幂等的 `state`、`a`、`b`）和 Shared Memory 数据随着 `EXIT` 灰飞烟灭了**。如果这是一个不能重头再来的非幂等算子或已经跑了 $100\text{ ms}$ 的长算子，直接 `EXIT` 意味着之前所有的计算功亏一篑。

---

## 二、 按照导师的观点，理想的抢占状态是什么？

按照导师的评价，一个真正解决 GPU 痛点的**理想抢占状态（的无损空间级抢占，Lossless Spatial Preemption）**应当具备以下四个特征：

1.  **物理空间 $100\%$ 腾退（True Hardware Quota Release）：**
    被抢占的低优先级线程块（CTA）必须**真正执行 `EXIT` 指令退出 SM**，将其占用的 **物理寄存器堆（Register File）、共享内存（Shared Memory）以及 Warp/CTA 硬件槽位** 彻底归还给 SM 硬件分配器。高优先级 Kernel 到达时，看到的是一个近乎完全空载的 SM，无论它需要 $255$ 个寄存器/线程还是 $96\text{ KB}$ Shared Memory，都能在微秒级内立即发射进 SM。
2.  **状态无损冻结（Stateful Snapshot Before Exit）：**
    在低优 CTA 执行 `EXIT` 之前，必须将其片上的所有存活状态——包括 **通用寄存器（`R0..R_max`）、统一寄存器（`UR0..UR_max`）、谓词寄存器（`P0..P6`）、片上共享内存（Shared Memory）以及精确的恢复指令地址（`PC_orig`）**——完整导出保存。
3.  **按需定量换出（Demand-Driven Partial CTA Eviction）：**
    结合您代码里已经实现的 `Mode 3` 按 CTA 精确寻址能力（`pbuf_low[4 + 2 * cta_id]`）：高优任务需要多少寄存器和 Shared Memory，就只把对应数量的低优 CTA 执行“保存状态 $+$ `EXIT`”，其余不阻碍高优任务发射的低优 CTA 依然可以使用零拷贝开销的 `Mode 2`（原地 `NANOSLEEP`）。
4.  **原位装填恢复（Stateful Relaunch & Resume）：**
    当高优任务结束后，被驱逐的低优 CTA 能够重新发射上卡，在入口处自动将保存的 Shared Memory 和寄存器数据装填回 SM 片上硬件，并直接跳转回中断前的 `PC_orig` 继续执行，最终通过您的非幂等校验（`checksum_errs == 0`）。

---

## 三、 这部分寄存器和 Shared Memory 可能保存在哪里？

当一个 CTA 要执行 `EXIT` 让出 SM 片上的寄存器堆（每 SM $256\text{ KB}$）和 Shared Memory（每 SM 最高 $128\text{ KB} \sim 228\text{ KB}$）时，状态数据只能向下一级存储层次转移。我们对比三个候选位置：

| 候选保存位置 | 硬件本质与寻址方式 | 能否在 `EXIT` 后保留并跨 SM 恢复？ | 带宽与延迟表现 | 适用结论 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 线程私有局部内存**<br>（Local Memory / LMEM，如 `[0xffff88]`） | 位于全局显存但按**物理硬件槽位 `(SM_ID, Warp_ID, Lane_ID)`** 映射的私有栈空间 | **不能！** 一旦执行 `EXIT`，高优任务进入同一物理 SM 槽位会覆写该 LMEM；且低优 CTA 重启时会被分发到不同的物理 SM 槽位，无法读回原数据 | 经由 L1/L2 缓存加速，寻址无需计算全局偏移 | 仅适合 **不执行 `EXIT`** 的原地暂存（如你目前保存 `PC_orig` 的 `[0xffff88]`） |
| **2. 预分配的全局显存状态池 + L2 Cache 驻留**<br>（Global Memory Spill Arena in VRAM / L2） | 通过 `cudaMalloc` 分配、按**逻辑 `(cta_id, thread_id)`** 线性寻址的显存缓冲区 | **完全可以！** 与物理 SM 槽位解耦，无论恢复时该 CTA 被调度到哪个 SM，都能按逻辑 `blockIdx` 和 `threadIdx` 精准找回状态 | 单个 CTA 状态约 $64\text{ KB} \sim 160\text{ KB}$，全卡 $30$ 个 SM 仅需数 MB，**可 $100\%$ 命中片上 L2 Cache**（数微秒内完成读写） | **【最理想主力存储位置】** |
| **3. 主机端零拷贝内存**<br>（Host Pinned / Mapped Memory） | 通过 PCIe 总线直连的 CPU 内存（如你代码中的 `pbuf_low`） | 可以，但受限于 PCIe 带宽（$\sim 32\text{ GB/s}$） | 搬移数 MB 寄存器/Shm 需数百微秒，延迟过高 | 仅适合存放控制标志 `pbuf_low`，不适合存放海量寄存器快照 |

### 关键定量推算：为什么全局显存状态池（配合 L2 Cache）会非常快？
在您的 RTX 5060（Blackwell SM120，共 $30$ 个 SM）上：
*   假设每个 SM 上需要驱逐 $1$ 个重度 CTA（共 $30$ 个 CTA）：每个 CTA 含 $512$ 线程，每线程使用 $64$ 个寄存器（$256\text{ B}$/线程，共 $128\text{ KB}$），外加 $32\text{ KB}$ Shared Memory。
*   单个被驱逐 CTA 的总状态大小 $= 128\text{ KB} + 32\text{ KB} = 160\text{ KB}$。
*   全卡 $30$ 个 SM 合计转储大小 $= 30 \times 160\text{ KB} = 4.8\text{ MB}$。
*   **硬件红利：** Blackwell SM120 拥有高达数十 MB 的片上二级缓存（L2 Cache）！这意味着当 $30$ 个 SM 并发通过 `ST.E.128` 将这 $4.8\text{ MB}$ 寄存器与 Shared Memory 写入全局显存状态池时，**数据实际上直接落在了片上超高速 L2 Cache 中**（甚至可用 `cudaAccessPolicyWindow` 锁定在 L2），转储耗时仅需 **$5\text{ }\mu\text{s} \sim 15\text{ }\mu\text{s}$**！

---

## 四、 有办法做到让出 / 保存 / 恢复 寄存器和 Shared Memory 吗？

**完全有办法！而且您目前在 `preempt_rigorous_bench.cu` 中搭建的 SM120 Trap Handler 架构，恰好攻克了其中最难的一个硬件痛点。**

在传统的纯编译器插桩方案中，“恢复全部寄存器并跳回原 PC”是一个著名的死结：因为执行最终跳转指令 `BRX R_pc` 和加载指令 `LDG` 必须占用至少 $2 \sim 4$ 个通用寄存器作为基址指针和跳转目标，导致最后这几个寄存器无法恢复。

**但在您的 SM120 Trap Handler 架构中：**
当 Trap Handler 执行最后一条 `RET.ABS R12 0x20` 指令时，**SM120 硬件微码会自动从 Local Memory 的 Trap 栈帧（`[0xffff80..0xffffa8]`）中弹出恢复 `R0..R15` 以及 `PC` 指针**！这意味着您可以利用正常模式桩代码（Stub）配合 `R0..R15` 无障碍地搬移 `R16..R254` 和 Shared Memory，最后把 `R0..R15` 和 `PC_orig` 交给 Trap 栈帧由硬件原子恢复！

下面为您提供 **两条具体的工程实现路径**（路径 A 为纯二进制 Trap 桩全自动方案，路径 B 为 Trap 信号协同的轻量快照方案）：

---

### 路径 A：基于现有 SM120 Trap Handler 扩展“状态转储桩（Dump Stub）”与“恢复桩（Restore Stub）”

您目前在 `info.tools_pc` 中注入了 `ext_va (+0xa00)`、`exit_stub_va (+0xe00)` 和 `pause_tramp_va (+0xe20)`。只需将 `exit_stub_va` 升级为 **三段式状态化换出与恢复流水线**：

#### 第一步：数据结构设计（在全局显存分配 `PreemptStateArena`）
将状态池基地址存入 `KernelDbgParams`（例如占用 `guardian` 字段 `c[0x0][0x178]`），按你代码第 $7 \sim 15$ 行已经算出的线性 `cta_id` 划分固定大小的槽位（例如每 CTA 分配 $512\text{ KB}$ 槽位）：
*   `Offset 0x00000 .. 0x3FFFF` ($256\text{ KB}$)：存放本 CTA 最多 $1024$ 个线程的通用寄存器 `R0..R254`、谓词寄存器 `P0..P6` 与每个 Warp 的 `PC_orig`、`UR0..UR63`。
*   `Offset 0x40000 .. 0x5FFFF` ($128\text{ KB}$)：存放本 CTA 的 Shared Memory 镜像。
*   `Offset 0x60000` ($64\text{ B}$)：CTA 内部 Warp 同步计数器 `warp_arrive_cnt`。

#### 第二步：保存并让出（Stateful Evict，`cmd == 1` 时的动作）
当 Trap Handler 检测到 `cmd == 1` 时，不再直接跳往裸 `EXIT`，而是分两阶段执行：

1.  **阶段 1（Trap Mode 内，位于 `ext_code`）：保存 Trap 栈帧中的 `R0..R15` 与 `PC_orig`**
    *   因为进入 Trap Handler 时，用户态的 `R0..R15` 和 `PC_orig` 已经被硬件保存在 LMEM `[0xffff80..0xffffb0]` 中，此时在 `ext_code` 里直接用 `LDL.64` 读出它们，并用 `ST.E.64` 写入全局显存 `Arena[cta_id].thread[tid].r0_r15` 和 `pc_orig`。
    *   将 `[0xffffa8]`（Trap 返回地址）修改为正常模式的转储桩地址 `dump_stub_va`（如 `info.tools_pc + 0x1000`），然后执行 `RET.ABS R12 0x20` 退出 Trap Mode。
2.  **阶段 2（Normal Mode 内，位于 `dump_stub_va`）：保存 `R16..R_max`、Shared Memory 并执行 `EXIT`**
    *   此时硬件已回到正常执行模式，且用户态原有的 `R0..R15` 已经在阶段 1 安全落盘，因此**现在可以自由使用 `R0..R15` 作为指针和临时变量**，而 `R16..R254` 和 Shared Memory 依然完好无损地保留在片上！
    *   **转储谓词与高号寄存器：**
        执行 `P2R R2, PR, RZ, 0x7f` 读取谓词 `P0..P6` 存入显存；随后使用向量化 128 位存储指令连续写出高号寄存器：
        `ST.E.128 [R4 + 0x40], R16;`
        `ST.E.128 [R4 + 0x50], R20;`
        $\dots$（直至该 Kernel 实际使用的最大寄存器号）。
    *   **转储 Shared Memory（无死锁同步法）：**
        为了防止部分 Warp 还在半途、Shared Memory 尚未稳定，每个 Warp 的 `lane 0` 对显存中的 `Arena[cta_id].warp_arrive_cnt` 执行 `ATOM.E.ADD`（或执行 `BAR.SYNC 0`）。待本 CTA 的所有活跃 Warp 均到达 `dump_stub_va` 后，各线程按 `tid` 并行执行：
        `LDS.128 R8, [R_shm_offset];`
        `ST.E.128 [R_arena_shm + R_shm_offset], R8;`
        循环数次即可将整个 CTA 的 Shared Memory 搬入全局显存（L2 Cache）。
    *   **彻底释放物理空间：**
        转储完毕后，执行内存屏障 `MEMBAR.GPU`，将 `pbuf_low` 中该 CTA 的状态标记为 `EVICTED_SAVED (2u)`，最后执行 **`EXIT;`**！
        **此时该 CTA 在 SM 上占用的所有寄存器和 Shared Memory 被硬件当场清零释放，高优先级 Kernel 立即发射进 SM！**

#### 第三步：恢复并继续计算（Stateful Restore，`cmd == 4` 时的动作）
当高优先级 Kernel 执行完毕后，主机端将 `pbuf_low` 设为恢复模式，重新启动被驱逐的 CTA（并向硬件发一次 Trap 脉冲或在入口处拦截）：

1.  **阶段 1（Normal Mode 内，位于 `restore_stub_va`）：还原 Shared Memory 与 `R16..R_max`**
    *   利用 `R0..R15` 作为工作寄存器，根据 `cta_id` 和 `tid` 定位到 `Arena[cta_id]`。
    *   **还原 Shared Memory：** 并行执行 `LD.E.128 R8, [R_arena_shm + off]` 和 `STS.128 [off], R8`，随后执行 `BAR.SYNC 0` 确保全 CTA 的 Shared Memory 完全就绪。
    *   **还原高号寄存器与谓词：** 连续执行 `LD.E.128 R16, [R4 + 0x40]` ... 还原 `R16..R_max`，并用 `R2P PR, R2, 0x7f` 还原谓词寄存器 `P0..P6`。
2.  **阶段 2（通过 Trap 栈帧原子还原 `R0..R15` 并跳回 `PC_orig`）：**
    *   当 `R16..R_max` 和 Shared Memory 全部就位后，触发自陷（或由主机发送 `pulse_trap_broadcast()`）进入 Trap Handler 的恢复分支：
    *   Trap Handler 从 `Arena[cta_id].thread[tid]` 中读出保存的原始 `R0..R15` 和 `PC_orig`，直接通过 `STL.64` 写回 Local Memory 的 Trap 栈帧 `[0xffff80..0xffffa8]`。
    *   最后执行 `RET.ABS R12 0x20`——硬件微码自动将 `R0..R15` 装回通用寄存器堆，并将程序计数器精准恢复到中断前的 `PC_orig`，算子无缝继续执行！

---

### 路径 B：Trap 标志触发 $+$ 算子循环边界轻量转储（Hybrid Trap-Signaled Checkpointing）

如果导师允许对被抢占的低优先级 Kernel 源码（或 PTX/LLVM IR）做极轻量的宏包装，还有一种工程复杂度低得多、且转储数据量小 $10$ 倍的方案：

1.  **为什么随机指令点转储很难，而循环边界转储极简？**
    *   在任意一条 SASS 指令处强行转储，一个线程可能正有 $120$ 个临时寄存器处于活跃态，且不同线程可能处于 `if/else` 分支发散（Warp Divergence）状态。
    *   但在算子的主循环边界（例如 `tiled_sgemm_2d_kernel` 的外层 Tile 循环末尾，或 `heavy_non_idempotent_low_kernel` 的每次迭代末尾），所有线程刚刚做完 `__syncthreads()`，**Shared Memory 中的分块数据已经消费完毕无需保存**，且真正跨迭代存活的变量只有极少数几个（例如 SGEMM 的累加器 `acc` 和循环步数 `t, bx, by`，或者 `low_kernel` 的 `it, state, a, b, c` 共 $5$ 个寄存器）！
2.  **如何与你的零开销 Trap 机制完美结合？**
    *   **正常运行时（0% 轮询开销）：** 不在循环里用慢速的 `ld.global` 轮询内存！而是利用一个专用的谓词寄存器（如 `P6`）或特殊寄存器作为抢占标志，正常时只做一条 $1$ 周期的寄存器判断。
    *   **抢占触发时：** 主机端依然通过你的 `pulse_trap_broadcast()` 发送硬件 Trap 脉冲。Trap Handler 不修改 PC 去死循环，而是直接修改被中断线程的标志寄存器（或将 PC 前推到当前循环末尾的轻量转储块）。
    *   **极速保存与 `EXIT`：** 线程块将这仅有的几个核心存活变量（及必要的 Shared Memory）写入显存 `Arena[cta_id]`，立即执行 `EXIT` 释放 $100\%$ 的寄存器与 Shared Memory！

---

## 五、 总结：如何向导师汇报并推进下一步实验？

您可以用以下清晰的逻辑向导师回应，展示您对底层硬件限制的透彻理解与下一步方案：

1.  **承认并定位现状（Validating the Critique）：**
    *   目前代码中的 `Mode 2`（Trampoline `NANOSLEEP`）实现了 **Level 1（计算流水线级让出）**，解决了 ALU 争抢问题；但在高寄存器/高 Shared Memory 负载下，由于未执行 `EXIT`，硬件 CWD 会因 SM 静态配额不足而无法发射新 Kernel。
    *   目前代码中的 `Mode 1 / Mode 3` 通过跳往 `exit_stub_va`（`EXIT`）实现了 **SM 空间的物理腾退**，证明了只要执行 `EXIT`，高优 Kernel 的排队延迟（Queue-to-SM-Start Delay）就能从数千微秒骤降至十微秒级（见 `BENCH 2C/2D`），但目前缺少 `EXIT` 前的状态保存与恢复闭环。
2.  **明确存储介质（Where to Save）：**
    *   由于 `EXIT` 后物理槽位会被高优任务覆写，状态不能留在按物理槽位索引的 Local Memory 中，而应保存在**按逻辑 `cta_id` 索引的全局显存预分配状态池（VRAM Spill Arena）**中。
    *   得益于 SM120 数十 MB 的 L2 Cache，全卡 $30$ 个 SM 的寄存器与 Shared Memory 快照（约 $3 \sim 8\text{ MB}$）在写出时可直接命中片上 L2 Cache，转储耗时可控在 $10\text{ }\mu\text{s}$ 量级。
3.  **给出落地路线（How to Implement）：**
    *   将现有的 `exit_stub_va` 扩展为 **`dump_stub_va`（利用 Trap 栈帧保存 `R0..R15` 与 `PC`，在正常模式桩代码中通过 `ST.E.128` 和 `LDS.128` 导出 `R16..R_max` 与 Shared Memory 后执行 `EXIT`）** $+$ **`restore_stub_va`（重发射后装填 Shared Memory 与高号寄存器，再借由 Trap Handler 的 `RET.ABS` 硬件机制原子恢复 `R0..R15` 与 `PC_orig`）**，即可彻底打通“无损保存 $\rightarrow$ `EXIT` 让出全部寄存器与 Shared Memory $\rightarrow$ 原断点恢复”的完整闭环。