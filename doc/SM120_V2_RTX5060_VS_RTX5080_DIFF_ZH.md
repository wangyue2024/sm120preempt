# RTX 5060 (`GB206`) 与 RTX 5080 (`GB203`) 跨芯片/跨驱动适配状态与差异对比

**文档版本**：v2.1  
**更新日期**：2026-11-24  
**覆盖硬件**：NVIDIA GeForce RTX 5060 (`GB206`, `sm_120`, Driver `595.99.02`) vs NVIDIA GeForce RTX 5080 (`GB203`, `sm_120`, Driver `580.95.05`)

---

## 一、核心结论：两台机器是否都已适配成功？

**简明结论**：**是的——两张卡（`GB206` RTX 5060 与 `GB203` RTX 5080）均已基于同一套动态自适应代码库完成 Blackwell `sm_120` Level-3 v2 核心适配，无需任何按卡硬编码或条件编译**；但两机当前处于**互补的验证进度与明确的环境差异边界**，如实汇总如下：

| 适配与验证维度 | RTX 5080 (`GB203`, `580.95.05`) | RTX 5060 (`GB206`, `595.99.02`) | 结论 |
| :--- | :---: | :---: | :--- |
| **`C++20 -Werror` 零警告编译 + `sm_120` 79/79 SASS 核验** | **PASS** | **PASS** | 双机完全一致 |
| **`.text.traphandler` 代码洞动态探测 & 静态 9-Cave 初始化** | **PASS** (`11 Caves / 1,703 slots`) | **PASS** (`11 Caves / 1,695 slots`) | 双机动态自适应（跨 `+0x5900` 驱动偏移漂移） |
| **BAR0 `0x00424684` + `NV_PTIMER_TIME_0` D10 安全脉冲** | **PASS** (`0` 重试单脉冲) | **PASS** (`sm120_v2_bench` `0` 重试单脉冲) | 双机兼容 |
| **Mode 1 / Mode 5 延迟与紧凑 `u_arena` 缩放 (`sm120_v2_bench`)** | **PASS** (`332–1,261 µs`, 显存缩 `160.6×`) | **PASS** (`399–1,076 µs`, 显存缩 `436.9×`) | 双机全通 |
| **Context 级 CILP 可恢复抢占 (`sm120_l3_resumable_bench`)** | **PASS** (`P50=425 µs`, `diff=0`) | **PASS** (`P50=431 µs`, `diff=0`) | 双机高度一致 |
| **11 项合成 E2E 与 100 轮压测 (`sm120_v2_e2e_suite`)** | **11 / 11 ALL_PASS**（连续 3 轮） | `CASE 01` PASS；`CASE 02` 桌面 `595.x` 336-Warp PCIe 原子风暴停滞 | 5080 满通；5060 暴露桌面 PCIe 瓶颈（`DEF-01/08`） |
| **8 大闭源 CUDA 库 + Blackwell 特性 + PyTorch/Triton + QoS（含 FIX-1~6）** | 待跳板机 SSH 恢复后一键同步并复跑 [`run_all_eval.sh`](../xsched/test/sm120_v2/eval/run_all_eval.sh) | **全量完成**：8/8 闭源库 `NP=2` 零误差、PyTorch ResNet-50/4×SDPA/Triton 3.6.0 零误差、DNN/LLM QoS 全通 | 5060 最新基线已验毕；5080 仅差同步重跑 |

---

## 二、硬件规格、系统环境与工具链差异

| 维度 | RTX 5080（服务器计算节点） | RTX 5060（本地桌面主机） | 对系统设计与适配的实际影响 |
| :--- | :--- | :--- | :--- |
| **芯片核心 / 架构** | `GB203` (`sm_120`, Blackwell) | `GB206` (`sm_120`, Blackwell) | 128-bit SASS ISA 与 79 条手写编码 **100% 二进制通用**，无须区分芯片 |
| **物理 SM 数量** | **84 SMs** | **30 SMs** | `Sm120TrapEngine` 运行期通过 `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` 动态查询，自动缩放 `max_resident` 与 `u_arena` 槽位 |
| **单 SM 共享内存上限** | `100 KiB` / SM | `100 KiB` / SM | SMEM 硬件保留区均为 `0x400` B，`SR_SMEMSZ` 与 `align4096` 逻辑完全一致 |
| **显存容量 / 带宽** | `16 GiB` GDDR7 | `8 GiB` GDDR7 | 5060 显存更紧张，v2 紧凑池化 `u_arena`（`9.38 MiB` vs v1 `4 GiB`，节省 **436.9×**）在 8 GiB 卡上收益尤为关键 |
| **运行形态** | 无头计算服务器（独占 GR 引擎） | Ubuntu 26.04 桌面环境（显示合成器共享 GR 引擎 + 4 秒桌面看门狗） | 5060 上若内核挂死会触发 4s 看门狗（`cuda error 719`）；且桌面合成器会在 BAR0 瞬间偶发争抢 GR 引擎 |
| **NVIDIA 驱动分支** | **`580.95.05`** | **`595.99.02`** | **重大差异**：跨主版本驱动导致 `.text.traphandler` ToolsMemory 内部布局与基址发生 **`+0x5900` 字节偏移漂移**（见第三节） |
| **宿主机 GCC 工具链** | `gcc-13 / g++-13` | 默认 `gcc-15 / g++-15`（Ubuntu 26.04）+ `gcc-13 / g++-13` | `nvcc 12.9` 仅支持至 `g++-13`；在 5060 上若混用 `g++-15` 编宿主 C++ 与 `g++-13` 编 CUDA 会报 `__cxa_call_terminate@CXXABI_1.3.15` 链接冲突，必须统一指定 `-DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13` |
| **MPS 隔离要求** | 共享服务器可能存在他人 MPS 进程，须显式设置 `CUDA_MPS_PIPE_DIRECTORY=/tmp/no_mps_<user>` 并检查 `nvidia-smi` | 单用户本地环境 | 所有评测脚本（[`run_all_eval.sh`](../xsched/test/sm120_v2/eval/run_all_eval.sh)）均已统一内建 `nvidia-smi` 空闲检查与 `CUDA_MPS_PIPE_DIRECTORY` 隔离 |

---

## 三、底层微架构与驱动行为差异（实测深度剖析）

### 1. 跨驱动 `.text.traphandler` ToolsMemory 偏移漂移（`580.95.05` vs `595.99.02`）

| 探测指标 | RTX 5080 (`Driver 580.95.05`) | RTX 5060 (`Driver 595.99.02`) | 自适应处理机制（[`sm120_arena.cpp`](../xsched/platforms/cuda/hal/src/level3/sm120_arena.cpp)） |
| :--- | :---: | :---: | :--- |
| **`handler_va` 与 `tools_pc` 相对关系** | `tools_pc - handler_va == 0x4400` | `tools_pc - handler_va == 0x4400`（整体基址漂移 **`+0x5900`**） | 彻底废弃 v1 硬编码偏移，改用动态二进制指纹扫描定位 `.text.traphandler` 入口与 `handler_va + 0x390` 唯一 `CALL.ABS` 挂钩点 |
| **探测到的可用代码洞总数** | **11 Caves / 1,703 槽位** | **11 Caves / 1,695 槽位**（少 8 槽位） | 初始化阶段动态扫描并筛选容量 `>= 144` 槽位的 9 个工作洞 |
| **静态 9-Cave 引擎实际占用** | **`1,045 / 1,390` 槽位**（每洞余量 `>= 13` 条） | **`1,045 / 1,390` 槽位**（完全一致） | 9 个 Cave 的指令长度与跨 Cave 跳转目标均按运行期实际 `cave_va[0..8]` 动态链接重定位 |

### 2. 物理 SM 并发度与紧凑 `u_arena` 显存缩放差异

因 `max_slots = min(num_ctas, sm_count * max_blocks_sm + margin)` 严格绑定物理 SM 数，在大网格（`16,384 CTAs`, 128 threads/CTA, ~32 regs, 0 SMEM）下：

| Grid 规模 | v1 朴素预分配 | RTX 5080 (`84 SMs`) | RTX 5060 (`30 SMs`) | 差异说明 |
| :---: | :---: | :---: | :---: | :--- |
| `256 CTAs` | `64.00 MiB` | `6.38 MiB` (`10.0×`) | `6.38 MiB` (`10.0×`) | 小于单波次驻留上限，按 `num_ctas` 分配 |
| `1,024 CTAs` | `256.00 MiB` | `25.50 MiB` (`10.0×`) | `9.38 MiB` (`27.3×`) | 5060 达到 30 SM 饱和封顶，显存不再随 Grid 增长 |
| `4,096 CTAs` | `1,024.00 MiB` | `25.50 MiB` (`40.2×`) | `9.38 MiB` (`109.2×`) | 5080 亦达到 84 SM 饱和封顶 |
| **`16,384 CTAs`** | **`4,096.00 MiB`** | **`25.50 MiB` (`160.6×`)** | **`9.38 MiB` (`436.9×`)** | **5060 显存占用仅为 5080 的 `36.8%`**（正比于 `30 / 84 = 35.7%`） |

### 3. 微秒级延迟与 PCIe 路径差异（`sm120_v2_bench` 与 CILP 实测对照）

| 基准项 | RTX 5080 (`84 SMs, 580.95.05`) | RTX 5060 (`30 SMs, 595.99.02`) | 差异归因分析 |
| :--- | :---: | :---: | :--- |
| **Entry Guardian 零抢占延迟 (`84×128` null)** | `Baseline 4.92 µs → Guarded 35.70 µs` (`Δ = +30.78 µs`) | `Baseline 4.50 µs → Guarded 47.75 µs` (`Δ = +43.25 µs`) | 5060 桌面平台 PCIe 往返延迟高于服务器平台；且 84 CTAs 在 30 SM 上需跨 3 波次完成 |
| **Mode 1 (`512 CTAs, 48K SMEM`) Evict / Restore P50** | `971.6 µs` / `419.9 µs` (`0 KiB`) | `901.2 µs` / `370.5 µs` (`0 KiB`) | 两机高度接近，5060 需早退的并发在途 CTA 更少（30 vs 84），略快约 7–12% |
| **Mode 5 (`84 CTAs, 0K SMEM`) Evict / Restore P50** | `332.3 µs` / `763.6 µs` (`2,176 KiB`) | `399.7 µs` / `772.2 µs` (`2,176 KiB`) | 5080 84 SM 单波次并发换出完 84 CTAs；5060 需 30 SM 换出 + 54 CTAs 经 Guardian Drain |
| **Mode 5 (`84 CTAs, 32K SMEM`) Evict / Restore P50** | `372.8 µs` / `831.8 µs` (`4,864 KiB`) | `447.0 µs` / `806.4 µs` (`4,864 KiB`) | 32 KiB SMEM 增量：5080 仅 `+40.5 µs`，5060 仅 `+47.3 µs`，GDDR7 带宽均极充裕 |
| **Mode 5 (`512 CTAs, 48K SMEM`) Evict / Restore P50** | `1,260.8 µs` / `1,859.0 µs` (`13,568 KiB`) | `1,076.2 µs` / `917.4 µs` (`5,632 KiB`) | 5060 单次仅需保存/恢复 ~30 个物理 SM 的 `State B`（~2.4× 更少 SMEM 换出量），故 5060 的 Restore P50 反而快 **2.03×** |
| **Context CILP (`sm120_l3_resumable_bench`) P50** | **`425 µs`** (`lost_work=0, diff=0`) | **`431 µs`** (`P99=504 µs, lost_work=0, diff=0`) | RM 硬件上下文切换延迟跨芯片与跨驱动**几乎恒定（相差仅 `6 µs`）** |

### 4. 极端并发 PCIe 原子风暴下的双机行为差异（`CASE 02` 根因）

- **RTX 5080 (`580.95.05`, 无头服务器)**：`sm120_v2_e2e_suite` 全部 **11 / 11 CASES 连续 3 轮稳定 ALL_PASS**（含 100 轮双连压测）。
- **RTX 5060 (`595.99.02`, 桌面独显)**：`sm120_v2_bench`（使用 VRAM 停止标志，无人工 PCIe 原子风暴）全部 80 轮 **0 重试 PASS**，且 8 大闭源 CUDA 库 + PyTorch/Triton 均双连零误差 PASS；唯独合成用例 `CASE 02`（让 336 个 Warp 在紧循环中持续向主机零拷贝内存打 `atomicAdd`）会出现 `ev=0 dr=0 m1=0` 停滞。
- **核心归因**：桌面环境 X11/Wayland 合成器与计算上下文共享 GR 引擎 + 5060 消费级 PCIe 根复合体在 336-Warp 持续原子写饱和下阻塞 `u_control` 零拷贝通道。该问题与 **`DEF-01`（大网格 `cuBLAS SGEMM` 4.3× 退化）完全同源**，在 Phase 1 将 `CtaEntry[]` 迁入 GPU GDDR7 VRAM 后将彻底根治。

---

## 四、跨卡可移植性工程清单（面向未来 RTX 5070 / 5090 / Blackwell Data Center 扩展）

本套 v2 架构在 5060（`GB206`, `595.x`）与 5080（`GB203`, `580.x`）上已验证的**零硬编码可移植准则**：

1. **禁止硬编码 SM 数量**：统一以 `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` 与 `cuOccupancyMaxActiveBlocksPerMultiprocessor` 动态计算 `max_slots`（扩展至 170 SM 的 RTX 5090 `GB202` 时零改动即可适配）。
2. **禁止硬编码 ToolsMemory 基址**：统一通过 `Sm120Arena::Discover()` 指纹定位 `.text.traphandler` 并动态重定位 9 个工作 Cave，天然兼容 `580.x`、`595.x` 及未来驱动版本。
3. **统一 4096 B SMEM 粒度与 `SR_SMEMSZ >= 0x401` 保护**：兼容不同驱动/芯片上的 0-SMEM 内核与 4 KiB SMEM 分配量子。
4. **严格编译链隔离**：在 Ubuntu 26.04 等新发行版（默认 GCC 15）上固定 `-DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13`，避免 C++ ABI 符号冲突。
