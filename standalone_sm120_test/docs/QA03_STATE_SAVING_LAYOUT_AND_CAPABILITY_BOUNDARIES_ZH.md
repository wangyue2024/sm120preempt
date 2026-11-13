# Q&A 专题三：全状态保存机制、显存地址布局、可保存状态全集与不可保存边界深度剖析

> **对应问题**：  
> *“抢占模式如何完整地保存寄存器和其他信息？一共有哪些信息？在代码洞里面如何保存这些信息？存储方式和地址是什么？哪些信息是目前可以存储的？哪些是目前还做不到的？”*

---

## 1. 一个运行中的 CUDA Kernel 一共有哪些架构状态信息？

当一个通用 CUDA Kernel 在 Blackwell (`SM120`) SM 上运行到任意一条 SASS 指令被硬件 Trap 打断时，决定其后续能否 100% 无损恢复继续执行的**片上架构与微架构状态一共有 8 大类**：

| 序号 | 状态类别 | 具体硬件单元 / 寄存器名称 | 单线程 / 单 CTA 最大容量 | 当前 Universal Mode 5 是否支持保存与恢复？ |
| :---: | :--- | :--- | :---: | :---: |
| **1** | **低位通用寄存器 (`GPRs R0..R13`) 与谓词寄存器 (`PR`)** | `R0 .. R13`（14 个 32-bit 寄存器）+ `P0 .. P6`（7 个 1-bit 条件谓词寄存器） | `60 B / thread` | ✅ **100% 支持**（随 112B 硬件 Trap 栈帧自动压栈并向量转储） |
| **2** | **高位通用寄存器 (`GPRs R14..R252`)** | `R14 .. R[num_regs - 3]`（SM120 满配 `num_regs = 255` 时最高为 `R252`，`R253..R254` 为 Trap 保留，`R255 = RZ`） | `956 B / thread`<br>(`239 × 4 B`) | ✅ **100% 支持**（支持 `1..255` 任意整数个寄存器零间隙保存/恢复） |
| **3** | **统一寄存器 (`Uniform Registers`)** | `UR4 .. UR15`（含动态 Shared Memory 偏移基址 `UR4 = 0x400`、全局描述符指针 `UR6:UR7` 及循环不变量） | `48 B / thread`<br>(`12 × 4 B`) | ✅ **100% 支持**（`MOV R, UR` 保存 + `R2UR` 原位恢复） |
| **4** | **指令指针 (`PC`) 与 SIMT 分支分化重收敛栈** | `TRAP_RETURN_PC` (64-bit)、`TRAP_RETURN_MASK` (32-bit 活跃掩码)、`Rpc` (重收敛 PC)、`OPT_STACK`、`B0`、`SR_THREAD_STATE_ENUM.0..4` | `52 B / thread`<br>(位于 112B 栈帧内) | ✅ **100% 支持**（支持任意 `if/else` 奇偶线程分化中断与退场中和） |
| **5** | **静态与动态共享内存 (`Shared Memory`)** | 每个 CTA 独立的片上高速共享内存窗口 `[0x0400 .. 0x103ff]`（跳过驱动保留的 `0x0000..0x03ff` 头部） | **`64 KiB / CTA`** | ✅ **100% 支持**（全 CTA 多 Warp 合并搬运 + 双阶段 `2*W` 屏障） |
| **6** | **跨 Warp 硬件同步屏障 0 (`__syncthreads()`)** | CTA 0 号硬件命名屏障（`BAR.SYNC 0`）的已到达阻塞状态（`B2R.WARP` Bit 12 = `0x1000`） | `4 B / thread` | ✅ **100% 支持**（`B2R.WARP R8, 0x0` 探测 + `@P0 BAR.SYNC.DEFER_BLOCKING 0x0` 重建） |
| **7** | **多波次网格 CTA 生命周期进度状态** | 网格内各 CTA 处于“已自然完成 (`State A`)”、“运行中被换出 (`State B`)”还是“未启动被排空 (`State C`)” | `8 B / CTA`<br>(`cta_w` + `cta_ack`) | ✅ **100% 支持**（三态 `entry_tramp` 精确分流，绝不重跑已完成块） |
| **8** | **特例硬件状态（栈溢出 Local Memory、1~15 号命名屏障、`cp.async`/TMA 在途事务）** | 用户态 `STL/LDL` 局部内存栈、`bar.sync 1..15`、`mbarrier`、异步拷贝在途队列 | 变长 | ❌ **Mode 5 当前不支持**<br>*(需使用 Mode 4 安全点或跨 Context CILP)* |

---

## 2. 在三级代码洞中是如何一步步保存这些信息的？

当主机下发 `cmd = 5` (`kTrapFrameEvictCommand`) 并触发 BAR0 `0x00424684` 脉冲后，被中断的 Warp 进入 `.text.traphandler`，随后跳入我们的三级代码洞（`ext_code` $\rightarrow$ `ext3_code` $\rightarrow$ `ext2_code`），按以下 **5 个严密步骤** 保存全部片上状态：

### 步骤一：官方 `.text.traphandler` 序言自动构建 `112B` 硬件 Trap 栈帧（保存 `R0..R13, PR, PC, SIMT 栈`）
在进入我们的代码洞之前，硬件微码与官方 `.text.traphandler` 序言（`+0x0000 .. +0x0380`）已经执行了 `SETLMEMBASE R4`，将局部内存切换到当前物理 `(SM, Warp)` 专属的隔离窗口，并在局部内存顶端 **`local[0xffff90 .. 0xffffff]`（共 `112` 字节 = `0x70` 字节）** 中保存了以下关键现场：
- `local[0xffff90 .. 0xffffa3]`（20 B）：`SR_THREAD_STATE_ENUM.0..4`（5 个特殊寄存器，记录每个线程是否处于活跃、`WARPSYNC` 挂起或 `YIELD` 状态）；
- `local[0xffffa8 .. 0xffffaf]`（8 B）：**`TRAP_RETURN_PC`**（被打断的用户 SASS 指令 64-bit 虚拟地址）；
- `local[0xffffb0 .. 0xffffbf]`（16 B）：**`Rpc.LO`、`Rpc.HI`、`OPT_STACK`、`B0`**（SIMT 分支分化重收敛栈状态）；
- `local[0xffffc0 .. 0xffffdf]`（32 B）：被打断时的通用寄存器 **`R0, R1, R2, R3, R4, R5`** 以及打包后的谓词寄存器 **`PR` (`P0..P6`)**；
- `local[0xffffe0 .. 0xfffff7]`（24 B）：被打断时的通用寄存器 **`R8, R9, R10, R11, R12, R13`**（注：`R6, R7` 保存在 `0xffffd4` 与 `0xffffdc`）；
- `local[0xfffffc .. 0xffffffff]`（4 B）：**`TRAP_RETURN_MASK`**（被打断瞬间的 32-bit 活跃线程掩码）。

> **重点**：正因为 `R0 .. R13` 与 `PR` 已经被官方序言保存在了 `local[0xffffc0 .. 0xfffff7]` 中，所以在我们的代码洞里，**`R0 .. R13` 和 `P0` 可以完全自由地作为临时工作寄存器使用**，而不会丢失用户的 `R0 .. R13`！

### 步骤二：在 `ext_code` (`idx 100..105`) 中全 CTA 并行保存 `64 KiB` Shared Memory
1. **为什么要跳过前 `1,024 B` (`0x400`)？**
   逆向发现，SM120 驱动和硬件固定占用每个 CTA Shared Memory 起始的 `[0x0000 .. 0x03ff]`（`1,024 B`）存放系统元数据；编译器生成的用户代码在入口序言把统一寄存器 `UR4` 设为 `0x400`，所有用户态 `shared[i]` 实际访问的是物理偏移 **`[0x0400 .. 0x103ff]`**。
2. **多 Warp 合并搬运循环**：
   - 计算当前线程在块内的起始字节偏移 `R4 = tid * 4`，以及全 CTA 单步跨度 `R5 = threads_per_cta * 4`（例如 128 线程块每次循环跨越 `512 B`）；
   - 计算当前 CTA 在全局显存 `u_arena` 中的 SMEM 镜像起始地址：`R8:R9 = arena_va + (cta_id + 224) * 65536 + tid * 4`；
   - 执行 6 条指令的高速循环（`idx 100..105`）：
     ```text
     LDS.U32          R0, [R4 + 0x400]       // 从用户 SMEM 窗口读取 4 字节
     ST.E.STRONG.GPU  [R8.64], R0            // 写入全局显存 u_arena
     IMAD             R4, R4, 1, R5          // R4 += threads_per_cta * 4
     IMAD.WIDE.U32    R8, R5, 1, R8          // R8:R9 += threads_per_cta * 4
     ISETP.GE.U32.AND P0, PT, R4, 0x10000, PT // 检查是否已满 64 KiB (65,536 B)
     @!P0 BRA         -6                     // 未满则继续循环
     ```

### 步骤三：在 `ext_code` (`idx 106..136`) 中向量转储 `112B` 硬件 Trap 栈帧、`UR4..UR15` 与 `B2R.WARP`
1. **转储 `112B` 硬件 Trap 栈帧（`idx 106..119`）**：
   使用 7 组 128-bit（16 字节）局部内存加载与全局显存写入指令（`LDL.128 R8, [0xffff90 + 16*k]` $\rightarrow$ `ST.E.128 [R6.64 + 16*k], R8`，`k = 0..6`），仅用 14 条指令就将 `local[0xffff90..0xffffff]` 的 112 字节完整搬入当前线程的显存栈帧 `[R6 + 0x000 .. 0x06f]`。
2. **转储统一寄存器 `UR4..UR15`（`idx 120..134`）**：
   分 3 组（每组 4 个 UR），先用 `MOV R8..R11, UR[4..7 / 8..11 / 12..15]` 将统一寄存器读入临时通用寄存器 `R8..R11`，再用 3 条 128-bit 向量写指令 `ST.E.128 [R6.64 + 0x440 / 0x450 / 0x460], R8` 存入 `[R6 + 0x440 .. 0x46f]`。
3. **探测并保存跨 Warp 硬件屏障 0 状态（`idx 135..136`）**：
   执行特权指令 **`B2R.WARP R8, 0x0`** 读取当前 Warp 在 0 号命名屏障（`__syncthreads()`）上的硬件状态字（若该 Warp 在被打断前已到达 `BAR.SYNC 0` 并正在等待其他兄弟 Warp，则 `R8` 的 Bit 12 即 `0x1000` 为 `1`），并通过 `ST.E.32 [R6.64 + 0x074], R8` 存入 `[R6 + 0x074]`。

### 步骤四：跳入 `ext3_code` (`idx 12..75`) 零间隙保存全部通用寄存器 `R14 .. R[num_regs - 3]`（最高 `R252`）
1. **SM120 `SR_REGALLOC - 2` 硬件保留寄存器铁律**：
   在 SM120 上，若内核分配了 `N = num_regs` 个寄存器（`1 <= N <= 255`），则最高 2 个寄存器 `R[N-2]` 与 `R[N-1]` 被硬件保留给 Trap 使用，真实用户寄存器上限为 **`max_reg = min(N - 3, 252)`**。一旦某条 128-bit 宽指令（如 `ST.E.128`）覆盖的 4 个寄存器中有任何一个触碰到 `>= N - 2`，硬件寄存器堆会整条拒绝该指令！
2. **确定性动态代码生成（`BuildFineGrainedExt3Code`）**：
   在绑定 Kernel 时，主机根据该 Kernel 的 `max_reg` 在 `ext3_code` (`idx 12..75`) 中自动生成无分支、零间隙的保存序列：
   - **首部对齐（`R14..R15`）**：发射 1 条 64-bit `ST.E.64 [R6 + 0x078], R14`，使后续显存偏移对齐到 16 字节边界 `+0x080`；
   - **主干 128-bit 向量喷射（`R16 .. R251`）**：只要 `r + 3 <= max_reg`，连续发射 `ST.E.128 [R6 + 0x040 + 4*r], R[r]`（最多 59 条指令即可覆盖到 `R251`）；
   - **尾部精确余数收尾**：对末尾剩余的 `1 ~ 3` 个寄存器，精确发射 `ST.E.64` 和/或 `ST.E.32`（例如 `N = 35, max_reg = 32` 时精确发射 `ST.E.32 [R6 + 0x0c0], R32`；`N = 255, max_reg = 252` 时精确发射 `ST.E.32 [R6 + 0x430], R252`），**既不漏掉任何一个非 8 倍数寄存器，也绝不触碰 `R[N-2..N-1]`**！

---

## 3. 显存存储方式与精确地址计算公式（`u_arena` 全景内存布局）

在 Universal Mode 5 中，所有换出状态均存储在设备全局显存池 **`u_arena`**（基地址 `arena_va` 存放在 `c[0x0][0x178..0x17f]`，默认预分配 `22 MiB`）中，分为**前半区的每线程栈帧区（Thread Frame Region）**与**后半区的每 CTA 共享内存镜像区（Per-CTA SMEM Region）**：

### 3.1 地址计算公式
设：
- 块内一维线程号：$\text{tid} = \text{tid.y} \times \text{blockDim.x} + \text{tid.x}$
- 每 CTA 线程总数：$\text{threads\_per\_cta} = \text{blockDim.x} \times \text{blockDim.y}$
- 全网格全局线性线程号：$\text{global\_tid} = \text{cta\_id} \times \text{threads\_per\_cta} + \text{tid}$

则：
1. **任意线程的 `1,152 B` (`0x480`) 专属栈帧基地址 (`R6:R7`)**：
   $$\text{my\_frame\_va} = \text{arena\_va} + \text{global\_tid} \times 1152 \quad (\text{即 } \text{global\_tid} \times \texttt{0x480})$$
   *(前 `14 MiB` 空间可容纳最多 $14,680,064 / 1152 = 12,743$ 个并发线程的满配 255 寄存器栈帧)*
2. **任意 CTA 的 `64 KiB` (`0x10000`) Shared Memory 镜像基地址 (`R8:R9`)**：
   $$\text{my\_smem\_va} = \text{arena\_va} + (\text{cta\_id} + 224) \times 65536 \quad (\text{从偏移 } \texttt{0x0e00000} = 14\text{ MiB 起})$$

### 3.2 单线程 `1,152 B` (`0x480`) 栈帧内部逐字节偏移表

| 帧内字节偏移 (`[R6 + off]`) | 大小 (Bytes) | 存放的架构状态字段 | 写入指令 (`Save`) | 读取指令 (`Restore`) |
| :--- | :---: | :--- | :--- | :--- |
| **`+0x000 .. +0x013`** | `20 B` | `SR_THREAD_STATE_ENUM.0..4`（线程状态枚举） | `LDL.128` + `ST.E.128` | `LD.E.128` + `STL.128` |
| **`+0x014 .. +0x017`** | `4 B` | 硬件栈帧对齐保留字 (`local[0xffffa4]`) | 同上 | 同上 |
| **`+0x018 .. +0x01f`** | `8 B` | **`TRAP_RETURN_PC`**（被打断的 64-bit 指令地址） | 同上 | 同上 |
| **`+0x020 .. +0x02f`** | `16 B` | **`Rpc.LO`, `Rpc.HI`, `OPT_STACK`, `B0`**（SIMT 分化栈） | 同上 | 同上 |
| **`+0x030 .. +0x06b`** | `60 B` | 通用寄存器 **`R0 .. R13`** + 谓词寄存器 **`PR` (`P0..P6`)** | 同上 | 同上 |
| **`+0x06c .. +0x06f`** | `4 B` | **`TRAP_RETURN_MASK`**（32-bit 活跃线程掩码） | 同上 | 同上 |
| **`+0x070 .. +0x073`** | `4 B` | **`status` 线程生命周期状态字** (`0`=初始, `5`=已换出, `6`=已恢复) | `ST.E.STRONG.GPU [R6+0x70], R1` | `LD.E.STRONG.GPU R0, [R6+0x70]` |
| **`+0x074 .. +0x077`** | `4 B` | **`B2R.WARP` 状态字**（Bit 12 `0x1000` = 已到达 `BAR.SYNC 0`） | `ST.E.32 [R6+0x74], R8` | `LD.E.STRONG.GPU R0, [R6+0x74]` |
| **`+0x078 .. +0x07f`** | `8 B` | 通用寄存器 **`R14 .. R15`** | `ST.E.64 [R6+0x78], R14` | `LD.E.64 R14, [R6+0x78]` |
| **`+0x080 .. +0x42f`** | `944 B` | 通用寄存器 **`R16 .. R251`**（59 组 16B 对齐四元组） | `59 × ST.E.128` | `59 × LD.E.128` |
| **`+0x430 .. +0x433`** | `4 B` | 最高用户通用寄存器 **`R252`**（对应 `num_regs = 255`） | `ST.E.32 [R6+0x430], R252` | `LD.E.32 R252, [R6+0x430]` |
| **`+0x434 .. +0x43f`** | `12 B` | 16B 对齐填充区 | — | — |
| **`+0x440 .. +0x46f`** | `48 B` | 统一寄存器 **`UR4 .. UR15`**（3 组 16B） | `3 × ST.E.128 [R6+0x440..0x460]` | `3 × LD.E.128` + `12 × R2UR` |
| **`+0x470 .. +0x47f`** | `16 B` | 预留扩展槽位（对齐至 `0x480 = 1,152 B`） | — | — |

---

## 4. 哪些信息目前还做不到在 Mode 5 中直接存储？为什么？

在明确了上述已 100% 实机攻克的 7 大类状态后，以下 **3 类底层特例状态** 是当前零源码修改 Mode 5 **尚不支持直接保存**的（需在架构选型时加以界定）：

1. **线程私有局部内存栈溢出（User Local Memory Stack Spill，`STL/LDL`）**：
   - **为什么目前没存**：在 `trap_mode = 1` 下，官方序言执行了 `SETLMEMBASE R4` 将局部内存基址切换到了 Trap 专用区。如果要转储用户态的 Local Memory（当 `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES) > 0` 时），需要在代码洞中临时执行 `SETLMEMBASE RZ` 切回用户态 LMEM，按栈指针 `R1` 遍历拷贝完后再切回 `SETLMEMBASE R4`。由于绝大多数高性能算子（以及我们的全部 8 个基准，包括 `255 regs/thread` 满配算子）的 Local Memory Spill 均为 `0 Bytes`，当前未开启用户态 LMEM 转储。
2. **1 ~ 15 号非默认硬件命名屏障（`Named Barriers 1..15`）与 `mbarrier` 对象**：
   - **为什么目前没存**：CUDA 标准的 `__syncthreads()` 固定编译为 `BAR.SYNC 0x0`（0 号屏障），当前 `ext_code idx 135` 针对性地执行了 `B2R.WARP R8, 0x0`。若算子通过内联 PTX 手工调用了 `bar.sync 1..15`，需要对 1..15 号屏障循环执行 `B2R.WARP`；而若使用了 Shared Memory 中的异步事务屏障 `mbarrier`（常配合 TMA 使用），其状态与硬件异步拷贝引擎绑定，不能在事务未完成时直接 `EXIT`。
3. **正在飞行中的硬件异步拷贝事务（In-flight `cp.async` / TMA `cp.async.bulk`）**：
   - **为什么做不到在任意 PC 强退**：当 TMA 或异步拷贝单元正在后台将全局显存数据直接 DMA 写入某块物理 SM 的 Shared Memory 时，如果此时直接让该 CTA 执行 `EXIT` 并把该物理 SM 让给高优先级 CTA，后台仍在飞行的 DMA 事务会覆写高优先级 CTA 的 Shared Memory！因此含在途 DMA 事务的算子必须在 `cp.async.wait_all` 之后的 Tile 安全点换出（即使用 **Mode 4**）或使用硬件级 **CILP**。
