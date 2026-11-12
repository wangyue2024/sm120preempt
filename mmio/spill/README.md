# SM120 CTA 空间配额换出与恢复实验 (`mmio/spill`)

本原型仅用于学术研究，禁止商用。

**完整实现原理、Mode 4 与 Mode 5 的详细对比、三级代码洞（`ext_code`/`ext2_code`/`ext3_code`/`entry_tramp`）逐行注释、调用触发结构、数值分析以及 SM120 底层逆向工程百科**请参见：
- 👉 **[《Blackwell SM120 通用零源码修改细粒度空间抢占与硬件恢复架构白皮书 (Universal Mode 5 / TrapFrame Engine)》](UNIVERSAL_TRAPFRAME_ARCHITECTURE_ZH.md)**
- 👉 **[《Blackwell SM120 Level 3 空间配额抢占（Mode 4）完整实现原理与 Mode 5 探索详解》](MODE4_IMPLEMENTATION_ZH.md)**

---

## 研发记录

- **2026-11-09（Mode 4 基础架构）**：复核 Mode 2 的 `NANOSLEEP` 蹦床与 Mode 1 的 `EXIT` 路径。采用硬件 Trap 原子置位显存请求字（`< 2 µs`）、受管 CTA 在同步安全点边界压缩保存跨步活状态后执行 `EXIT` 退出的设计（`Spill-and-Exit + Restore-and-Continue`）。旧 CUDA Stream 退役后，硬件 CWD 100% 回收寄存器、Shared Memory 与 Warp 槽位；恢复时按原逻辑 CTA 重发，从单波次显存快照池（Single-Wave Arena）装回存活寄存器与 Shared Memory 并从 `next` 步无缝继续。
- **2026-11-09（三类空间配额极限隔离验证）**：在 RTX 5060 / 595.99.02 上完成寄存器（`238 regs/thread`）、Shared Memory（`64 KB/CTA`）与 Warp 槽位（`6 CTAs/SM = 48 Warps/SM`）三类资源饱和准入测试，并在 `TrapSm120Context` 中正式封装 Mode 4 API（`BindKernelCheckpointRequest`、`RequestStreamCheckpoint`）。
- **2026-11-09（单波次显存池与多流隔离）**：换出池按单波次物理驻留 CTA 容量预分配，三波次逻辑网格的镜像空间降至 `4,350 KiB`、`2,040 KiB` 和 `360 KiB`。验证同一 CUDA Context 内 Armed、Unarmed 和 Unbound 三个 Stream 100% 隔离。
- **2026-11-09（1,206 次换出压力复测）**：`all 100`（606 次换出）与 `all 200`（1,206 次换出）压力测试 100% 通过，非幂等副作用精确一次，逐线程摘要零误差。
- **2026-11-09（真实 2D Tiled FP32 SGEMM 算子集成与 `O1` 开销评测）**：在 `checkpoint_bench.cu` 中新增 `[sgemm]` 评测套件（`BM=128, BN=128, BK=16`，`80 regs/thread`，`64 KB SMEM/CTA`）。利用安全点状态压缩，每个 CTA 仅需换出每线程 16 个 FP32 累加器（无需写出已消费完的 `As/Bs` SMEM 缓存，30 个驻留 CTA 总 Arena 仅 `120 KiB`）。实测正常无抢占执行开销 `O1` 仅为 **`-0.23% ~ +0.02%`**，Trap 到高优入场 P50 **`14.75 µs`**，低优排空 P50 **`25.66 µs`**，重发到全部恢复 P50 **`7.42 µs`**，连续空间换出恢复后与 CPU 双精度参考对比 **`0` FP32 bit error**。
- **2026-11-09（跨 Context 硬件 CILP 修复与验证）**：核对 Open-RM `595.99.02` 头文件 `ctrl2080gr.h`，将 `resumable_sm120.cpp` 中 `0x20801210` 的 `flags` 修复为 `0x1u` (`CILP_SET`)，并在 `trap_sm120.cpp` 中精准锁定 Compute Engine TSG (`engineType == 1`, class `0xcec0`)。实机运行 `sm120_l3_resumable_bench` 达成 **`P50 = 559 µs`, `P99 = 729 µs`, `lost_work = 0` (`1,843,200 / 1,843,200`)**。
- **2026-11-09（Mode 5 `trapframe` / Path A 零源码修改全状态空间换出与 `RTT.FALLTHROUGH` 硬件恢复实机验证）**：完整逆向 `live_handler.sass` 与 `tools_image.bin`（定位 `tools_pc + 0x0960` 的 `RET.ABS` 边界，将 `ext_code` 部署于 `tools_pc + 0x0980..0x12ff`），实现 **100% 零用户源码修改**（纯 C++ CUDA Kernel，无内联 PTX、无 C++ 抢占序言/尾声、无安全点轮询）的任意 PC 抢占与恢复：
  - **换出 (`RequestStreamTrapFrameEvict`, `cmd = 5`)**：全部在 Trap Handler (`ext_code`) 内完成每 CTA **`64 KiB Shared Memory`**（通过 `LDS R0, [R4+0x400]` 跨过 SM120 驱动保留的 `1024B` SMEM 头部，保存用户完整 `[0x0400..0x103ff]` 窗口）、**`UR4` DSMEM 基址**、以及 **`112B` 硬件 Trap 栈帧** (`local[0xffff90..0xffffff]`：`R0..R13`、`PR`、`Rpc/OPT_STACK/B0`、`TRAP_RETURN_PC`、`TRAP_RETURN_MASK`) 的显存换出，随后重定向至 `exit_stub_va` (`EXIT`) 100% 释放 SM 空间配额；
  - **恢复 (`PrepareKernelTrapFrameRestore` + `RequestStreamTrapFrameRestore` + `FinishKernelTrapFrameRestore`, `cmd = 6`)**：通过 `cuXtraGetEntryPoint(func)` 在原 Kernel 入口 (`func_pc + 0x00`) 动态热补丁 2 条纯 ALU 自旋桩 (`IADD3 R0, PT, PT, R0, 0, RZ; BRA -2`)，重发未修改的原 Kernel 后触发单次 BAR0 Trap 脉冲，在 `ext_code` 内由 `STS [R4+0x400], R0` 恢复 `64 KiB` Shared Memory、恢复 `UR4` 并重载 `UR6:UR7` (`c[0][0x358]`)、覆写 `local[0xffff90..0xffffff]` 的 112B 硬件 Trap 栈帧，最后由官方 `.text.traphandler` 尾声执行硬件 `RTT.FALLTHROUGH` 直接瞬移回中断时的任意内层循环指令 `TRAP_RETURN_PC`。
  - **实测指标（RTX 5060，30 CTAs × 64 KiB SMEM = 1.92 MiB 总状态）**：高优 CTA 入场 P50 **`22.91 µs`**，全量 30 CTA 64 KiB SMEM + 112B 栈帧换出完成 P50 **`39.01 µs`**，低优 Stream 退役 P50 **`41.79 µs`**，硬件 `RTT` 全量恢复 P50 **`126.94 µs`**，连续 20 轮零误差 (`0` bit errors)。
- **2026-11-11（通用化 Mode 5 `trapframe-univ-1w` / `trapframe-univ-2w` 全 Kernel 状态保存与恢复引擎）**：将 Mode 5 零源码修改抢占与恢复架构推广至任意通用 CUDA Kernel（同时覆盖高寄存器压力、Uniform 寄存器、多 Warp 2D/3D CTA、SIMT 分支分化、`__syncthreads()` 跨 Warp 同步屏障与多波次超配网格）：
  1. **Step 1 — 高通用寄存器 (`R14..R75`)、Uniform 寄存器 (`UR4..UR15`) 与 `SR_REGALLOC - 2` 硬件保留边界**：逆向 `.text.cilp_traphandler_save` 证实 SM120 硬件始终将最高 2 个已分配寄存器 (`R[num_regs - 2]` 与 `R[num_regs - 1]`) 保留给 Trap Handler，任何触碰保留寄存器的宽指令（如 `ST.E.128` / `ST.E.64`）会被硬件静默丢弃；在 `ext_code` (`tools_pc + 0x0980`) 与 `ext2_code` (`handler_va + 0x0860`) 双段架构中按 `c[0][0x180]` (`num_regs`) 精确门控并保存/恢复 `R14..R75` 及 `UR4..UR15`。
  2. **Step 2 — 多 Warp CTA (`blockDim = (32, 4)` = 128 线程/CTA = 4 Warps/CTA) 并行协同 SMEM 换出与两阶段恢复栅栏**：根据 `c[0][0x360..0x364]` 动态计算 `tid = tid.y * blockDim.x + tid.x` 与 `threads_per_cta = blockDim.x * blockDim.y`，全 CTA 线程以 `stride = threads_per_cta * 4` 并行协同搬运 `64 KiB` SMEM；通过 `@P0 REDG.E.ADD.STRONG.GPU` 逐 Warp 原子计数（换出等待 `ack == W`，恢复在 `ack == 2 * W` 处设置跨 Warp 恢复栅栏），彻底消除多 Warp 间的 `STS` / `LDS` 读写竞争。
  3. **Step 3 — 多波次超配网格 (`[trapframe-univ-2w]`, `grid = 60 CTAs = 2 waves`)**：利用每线程 `status` 字 (`[R6 + 0x70]`: `0` 未执行/已完成、`5` 已换出、`6` 已恢复) 实现多波次逐波换出与按序恢复。
  4. **Step 4 — SIMT 分支分化 (`TRAP_RETURN_MASK` / `OPT_STACK` / `B0`) 与跨 Warp `__syncthreads()` (`B2R.WARP` + `@P0 BAR.SYNC.DEFER_BLOCKING 0x0`) 硬件屏障重构**：在 Trap 换出时执行 `B2R.WARP R8`（捕获到达 CTA 硬件屏障 0 的递延阻塞位 `0x1000`）并保存至 `[R6 + 0x74]`；在恢复阶段跨过 `2 * W` 协同 SMEM 栅栏后、执行 `RTT.FALLTHROUGH` 前，对 `B2R.WARP >= 0x1000` 的 Warp 重新执行 `@P0 BAR.SYNC.DEFER_BLOCKING 0x0`，在新 SM 上精确重建跨 Warp 硬件屏障到达计数与阻塞状态。
  - **实测指标（RTX 5060，`universal_trapframe_kernel`，`32 regs/thread`，`128 threads/CTA = 4 Warps/CTA`，`64 KiB SMEM/CTA`，奇偶线程全步 SIMT 分化 + 每步双 `__syncthreads()`）**：
    - **`[trapframe-univ-1w]` (1 波次, 30 CTAs)**：全量换出 P50 **`71.55 µs`**，高优 CTA 入场 P50 **`12.67 µs`**，`RTT` 全量恢复 P50 **`144.29 µs`**，10 轮 `0` 误差；
    - **`[trapframe-univ-2w]` (2 波次, 60 CTAs，多波次快速排空与条件入口蹦床优化后)**：全量 60 CTA 换出 P50 **`135.65 µs`**（提速 `3.2x`），高优 CTA 入场 P50 **`17.28 µs`**，双波次 `RTT` 全量恢复完成 P50 **`138.72 µs`**（提速 `40x`），10 轮 `0` 误差；
    - **`[trapframe-univ-hireg]` (1 波次, 30 CTAs，高寄存器 `93 regs/thread` + 第三级代码洞 `ext3_code`)**：全量换出 P50 **`75.97 µs`**，高优 CTA 入场 P50 **`19.07 µs`**，`RTT` 全量恢复 P50 **`142.02 µs`**，10 轮 `0` 误差。
- **2026-11-11（第三级代码洞 `ext3_code` 任意寄存器扩展 + 多波次快速排空 `cmd=7` 与条件入口蹦床优化）**：
  1. **多波次未启动 Wave 1+ 快速排空 (`kTrapFrameDrainCommand = 7`) 与普通态条件入口蹦床 (`entry_tramp` @ `handler_va + 0x0c60`)**：
     - **换出阶段**：当 Wave 0 的 30 个驻留 CTA 完成状态封存并 `EXIT` 后，主机侧将全局命令切换为 `cmd = 7` (Drain)。被硬件 CWD 随后调度的未启动 Wave 1+ CTA 捕获 Trap 后跳过 `64 KiB` SMEM 与寄存器转储，保持 `control[4+2*cta_id] == 0` 并通过 `clean_exit_redirect` 直接 `EXIT` 排空，使 2 波次 60 CTA 换出延迟降至 **`127.26 µs`**；
     - **恢复阶段**：重发网格时，`func_pc + 0x00` 处的分支跳入 `entry_tramp` (`handler_va + 0x0c60`)，使 2 波次恢复延迟从 `5,538.72 µs` 骤降至 **`126.08 µs`**（**`44x` 加速**）。
  2. **第三级代码洞 (`ext3_code` @ `.text.do_preempt_restore`, `handler_va + 0x2520`) 与全架构 `1..255 regs/thread` 零间隙覆盖 + 多波次三态（State A/B/C）精确恢复（2026-11-12）**：
     - **全寄存器覆盖（Angle 2，`1 .. 255 regs/thread`）**：将所有通用寄存器 `R14 .. R[num_regs - 3]`（最高 `R252`，对应 SM120 架构极限 `num_regs = 255`）统一整合进 `BuildFineGrainedExt3Code`（`140` 条指令：`clean_exit_redirect` `idx 0..11`，保存 `idx 12..75`，恢复 `idx 76..139`），将每线程栈帧步长扩展为 **`1,152 B` (`0x480`)**，消除任意非 8 倍数寄存器数（如 `35`、`93`、`255`）的门控间隙，并在 **`255 regs/thread`** 满配算子上实机验证通过；
     - **多波次三态精确恢复（Angle 3，`State A / B / C`）**：`entry_tramp`（15 条指令）同时检查 `ack = control[4+2*cta_id+1]` 与 `w = control[4+2*cta_id+0]`：
       - **State A（`ack == 0`）**：抢占发生前就已经自然执行完 `EXIT` 的早期波次 CTA -> 在入口直接执行 `EXIT` 秒退（0 重复执行、0 非幂等累加污染）；
       - **State B（`ack > 0 && w > 0`）**：运行中途被 `cmd=5` 换出的 CTA -> 进入 `NANOSLEEP` 等待 `cmd=6` `RTT.FALLTHROUGH` 恢复；
       - **State C（`ack > 0 && w == 0`）**：换出阶段被 `cmd=7` 直接排空的未启动波次 CTA -> 原地执行 `orig_entry_inst` 并跳回 `func_pc + 0x10` 从头原生运行。

---

## 一键构建与复现

从仓库根目录 `level3opencode` 运行：

```bash
# 1. 构建 XSched HAL 与 CUDA Shim
cmake --build xsched/build_sm120 --target halcuda shimcuda sm120_l3_resumable_bench -j 8

# 2. 构建 checkpoint_bench
make -C mmio spill/checkpoint_bench

# 3. 运行全部套件（registers, shared, warps, trapframe 4 子项, sgemm）
./mmio/spill/checkpoint_bench all 10

# 4. 单独运行零源码修改 Mode 5 (trapframe 全部 4 子项) 10 轮验证
./mmio/spill/checkpoint_bench trapframe 10

# 5. 运行跨 Context 硬件 CILP 零丢失抢占基准
sudo ./xsched/build_sm120/bin/sm120_l3_resumable_bench
```

首个参数可指定为：
- `all`：依次运行 `registers`、`shared`、`warps`、`trapframe`（含 `trapframe`、`trapframe-univ-1w`、`trapframe-univ-2w`、`trapframe-univ-hireg`）、`sgemm` 全部套件；
- `registers` / `shared` / `warps` / `sgemm`：单独运行对应 Mode 4 套件；
- `trapframe`：单独运行 **Mode 5 (`Path A`) 零源码修改任意 PC 硬件 Trap 栈帧 + 64 KiB SMEM 空间换出与 `RTT.FALLTHROUGH` 恢复套件（包含单 Warp、通用 4-Warp 1 波次、通用 4-Warp 三态多波次、通用架构极限满配 `255 regs/thread` 四个子套件）**。

---

## 实验如何区分负载与资源瓶颈

| 组别 | 模式 | 每个 SM 低优驻留 CTA | 低优 / 高优寄存器每线程 | 低优 Shared 每 CTA | 准入瓶颈与设计目的 | 单波次 Arena |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **`[registers]`** | Mode 4 | `1 × 256` 线程 | `238 / 238` | `2,064 B` | 两个 CTA 的寄存器申请超过 SM 的 65,536 个 32 位寄存器配额 | `4,350 KiB` |
| **`[shared]`** | Mode 4 | `1 × 256` 线程 | `40 / 40` | `66,576 B` | 两个 CTA 的 Shared Memory 申请超过 SM 配额 | `2,040 KiB` |
| **`[warps]`** | Mode 4 | `6 × 256` 线程 | `40 / 8` | `2,064 B` | 低优占满 `48 / 48` 个 Warp 槽位，高优请求 8 个 Warp | `360 KiB` |
| **`[sgemm]`** | Mode 4 | `1 × 256` 线程 | `80 / 40` | `66,576 B` | **真实 2D Tiled FP32 SGEMM (`BM=128, BN=128, BK=16`)**，同时高占寄存器与 64 KB SMEM | `120 KiB` |
| **`[trapframe]`** | **Mode 5 (零源码修改)** | `1 × 32` 线程 | `14 / 14` | `66,560 B` (`64 KiB` 用户 + `1 KiB` 保留) | **纯 C++ 无安全点轮询单 Warp Kernel**，Trap 内直接换出 `64 KiB` SMEM + `112B` 栈帧并通过 `RTT.FALLTHROUGH` 恢复至任意 PC | `16,256 KiB` |
| **`[trapframe-univ-1w]`** | **Mode 5 (通用零源码修改)** | `1 × 128` 线程 (`32×4`) | `35 / 14` | `66,560 B` | **通用寄存器 (`35 regs`) + Uniform 寄存器 + 4 Warps/CTA + SIMT 分支分化 + `__syncthreads()`**（1 波次 30 CTAs） | `16,256 KiB` |
| **`[trapframe-univ-2w]`** | **Mode 5 (通用三态多波次)** | `1 × 128` 线程 (`32×4`, 60 CTAs) | `35 / 14` | `66,560 B` | **三态超配多波次 (`10 State A + 30 State B + 20 State C`) + 快速排空 (`cmd=7`) + 三态条件入口蹦床 (`entry_tramp`)** | `18,176 KiB` |
| **`[trapframe-univ-hireg]`** | **Mode 5 (通用 255 满配寄存器)** | `1 × 128` 线程 (`32×4`) | **`255 / 14`** | `66,560 B` | **SM120 架构极限满配寄存器 (`255 regs/thread`, 用户 `R0..R252`) + 三级代码洞 (`ext3_code`) + 4 Warps/CTA + SIMT 分化 + `__syncthreads()`** | `16,256 KiB` |

---

## 实测结果汇总（RTX 5060 / Driver 595.99.02 / CUDA 12.9，10 轮评测）

| 指标（单位：`µs`，P50 / P95） | `[registers]` (`238 regs`) | `[shared]` (`64 KB SMEM`) | `[warps]` (`48 Warps/SM`) | `[sgemm]` (2D Tiled SGEMM) | **`[trapframe]` (Mode 5 单 Warp)** | **`[trapframe-univ-1w]` (Mode 5 `35 regs`)** | **`[trapframe-univ-2w]` (Mode 5 三态多波次)** | **`[trapframe-univ-hireg]` (Mode 5 `255 regs`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`O1` 正常无抢占运行开销** | — | — | — | **`+0.09%`** (`1135.6 µs` vs `1136.6 µs`) | **`0.00%`**（零插桩） | **`0.00%`**（零插桩） | **`0.00%`**（零插桩） | **`0.00%`**（零插桩） |
| **首次请求 → 状态保存/安全点确认** | `7.36 / 10.66` | `9.28 / 12.83` | `30.82 / 36.61` | `11.14 / 12.74` | **`40.51 / 46.85`** | **`70.59 / 76.70`** | **`127.26 / 135.84`** | **`73.09 / 78.69`** |
| **首次请求 → 首个高优 CTA 入场** | `10.50 / 13.76` | `10.75 / 15.20` | `31.68 / 37.38` | **`12.00 / 13.79`** | **`23.52 / 29.73`** | **`12.26 / 18.69`** | **`12.80 / 19.33`** | **`22.91 / 25.09`** |
| **首次请求 → 低优 Stream 退役** | `24.83 / 28.00` | `27.36 / 30.82` | `99.97 / 105.70` | `26.37 / 28.74` | **`41.66 / 48.03`** | **`70.59 / 76.70`** | **`127.26 / 135.84`** | **`73.09 / 78.69`** |
| **重发/脉冲 → 全部 CTA 恢复运行** | `9.73 / 11.78` | `9.54 / 10.62` | `6.78 / 11.17` | **`7.01 / 7.90`** | **`127.30 / 136.22`** | **`147.97 / 156.99`** | **`126.08 / 134.91`** | **`144.90 / 152.99`** |
| **非幂等副作用与精度校验** | `0` 错误 (精确一次) | `0` 错误 (精确一次) | `0` 错误 (精确一次) | **`0` FP32 bit error** | **`0` bit error (100% 一致)** | **`0` bit error (100% 一致)** | **`0` bit error (100% 一致)** | **`0` bit error (100% 一致)** |
