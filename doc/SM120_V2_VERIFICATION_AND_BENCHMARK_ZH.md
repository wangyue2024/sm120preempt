# Blackwell (`sm_120`) Level-3 v2 实机验证与性能基准报告

**文档版本**：`v2.2.1-sm120`（双卡 GDDR7 VRAM 控制面全线实机闭环）  
**验证日期**：2026-11-24  
**构建标准**：`C++20 -fPIC -Wall -Wextra -Werror`（双机全目标 0 错误、0 警告）  
**验证平台**：
- **主平台 (`5080`)**：NVIDIA GeForce RTX 5080（`GB203`, `sm_120`, 84 SMs, `100 KiB` SMEM/SM, 16 GiB, Driver `580.95.05`, CUDA `12.9`, `g++-13`, PyTorch `2.11+cu128`）
- **交叉验证平台 (`5060`)**：NVIDIA GeForce RTX 5060（`GB206`, `sm_120`, 30 SMs, `100 KiB` SMEM/SM, 8 GiB, Driver `595.99.02`, CUDA `12.9`, `g++-13`, PyTorch `2.11+cu128`, Ubuntu 26.04 桌面环境）

---

## 1. 基础正确性与工具链核验（双卡全部 PASS）

| 回归项 | 验证平台 | 关键实测指标（`v2.2.1-sm120`） | 结果 |
| :--- | :--- | :--- | :---: |
| **零缓存全量构建** | RTX 5080 & RTX 5060 | `C++20 -Wall -Wextra -Werror`，全部核心库 + 5 项标准目标 + 6 项 Eval 目标 **0 警告** | **PASS** |
| **`sm120_sass_disasm` + `nvdisasm -b SM120`** | RTX 5080 & RTX 5060 | **79 / 79** 条手写 128-bit SM120 SASS 指令反汇编逐条核验通过 | **PASS** |
| **`sm120_cave_probe`（代码洞与引擎探测）** | RTX 5080 (`580.95.05`) | `total_caves=11 total_slots=1703`, `select_9_ge_144=PASS`, `c0=139/152 ...` (`1,045/1,390`), `bar0=MAPPED` | **PASS** |
| **`sm120_cave_probe`（跨驱动交叉探测）** | RTX 5060 (`595.99.02`) | `total_caves=11 total_slots=1695`, `select_9_ge_144=PASS`, `c0=139/152 ...` (`1,045/1,390`), `bar0=MAPPED` | **PASS** |
| **`sm120_l3_resumable_bench`（Context CILP）** | RTX 5080 (`580.95.05`) | `--iters 5 --multi_pause 3 --check_exact_sum 1`：**`P50=420 µs, P99=476 µs`**, `lost_work=0`, `diff=0`, `hp_errors=0` | **PASS** |
| **`sm120_l3_resumable_bench`（Context CILP）** | RTX 5060 (`595.99.02`) | `ZERO_LOST_WORK`, `diff=0`, `hp_errors=0` | **PASS** |

---

## 2. 端到端正确性与压力测试（`sm120_v2_e2e_suite` — RTX 5080 `v2.2.1`）

在空闲 RTX 5080 上执行 `v2.2.1-sm120` `sm120_v2_e2e_suite`，全部达成 **`11 / 11 CASES PASSED (ALL_PASS)`**（6 秒全通，0 重试，0 警告）：

| Case | 测试场景与覆盖维度 | RTX 5080 (`v2.2.1`) 实测输出摘要 | 结果 |
| :---: | :--- | :--- | :---: |
| **01** | Mode 1 多波次幂等早退（`512 CTAs, 48 KiB SMEM`） | `mode1_exit=168 replay_marked=512 leaks=0 errs=0` | **PASS** |
| **02** | Mode 5 单波次非幂等 `atomicAdd`（`84 CTAs`） | `evicted=84 restored=84 drained=0 arena=2176 KiB`, `got=107520/107520` | **PASS** |
| **03** | Mode 5 多波次 `State A + B + C` 零泄漏（`512 CTAs, 48 KiB`） | `A=40 B=133 C=339 (sum=512/512) ev=133 rest=133 cta_errs=0 ovf=0 leaks=0 total=524288/524288` | **PASS** |
| **04** | 全 3D 网格与线程块坐标 `Grid(4,3,2) × Block(8,4,2)` | `evicted=24 restored=24 arena=448 KiB`, `errs=0/1536` | **PASS** |
| **05** | 极限高寄存器级联 `R0..R253`（`num_regs=255`）+ `UR/UPR` | `CASE05-A` & `CASE05-B`：`evicted=16 restored=16 arena=5248 KiB`, `errs=0/1024` | **PASS** |
| **06** | 静态 + 动态共享内存（`36 KiB = 4K + 32K`） | `evicted=32 restored=32 arena=2240 KiB`, `errs=0/4096` | **PASS** |
| **07** | 栈溢出 + `__noinline__` 递归局部内存（`LMEM`） | `evicted=32 restored=32 arena=8896 KiB`, `static_lmem=128 B`, `errs=0/2048` | **PASS** |
| **08** | 0~15 号命名屏障（`bar.sync 0,1,5,7,11,15 + bar.arrive 11`） | `evicted=8 restored=8 arena=448 KiB`, `errs=0/2048` | **PASS** |
| **09** | 深层嵌套分支分化收敛屏障（`B1..B15`） | `evicted=16 restored=16 arena=448 KiB`, `errs=0/2048` | **PASS** |
| **10** | XSched `AsyncXQueue` FIFO `[K1->K2]` + 同函数 `DirectLaunch` HP | `k2_mid=0 errs(k1=0, k2=0, hp=0)` | **PASS** |
| **11** | **100 轮多波次连续压测（含每 10 轮双连抢占 `State A` 保留）** | `errs=0`（全部 100 轮位精确一致） | **PASS** |

---

## 3. 性能基准测试（`sm120_v2_bench` — 双卡 `v2.0/v2.1 → v2.2.1` 对比）

`sm120_v2_bench` 在未修改的连续计算循环内核上（设备 VRAM 停止标志）实测全部 80 轮次 **0 次恢复脉冲重试（100% 单脉冲 Evict + 0 脉冲自陷 Restore）**。

### 3.1 Entry Guardian 常驻空载发射开销（`84 × 128` 空内核，`N = 100`）

| 平台 | `v2.0/v2.1` PCIe 控制面 Delta | **`v2.2.1` GDDR7 VRAM 控制面（Baseline → Guarded `cmd=0`）** | **净增量 Delta** | 较 PCIe 加速 |
| :--- | :---: | :---: | :---: | :---: |
| **RTX 5080** (`84 SMs, 580.95.05`) | `+30.78 µs` | `4.86 µs` (`P95=5.22`) → **`5.84 µs` (`P95=6.21`)** | **`+0.99 µs`** | **31.1×** |
| **RTX 5060** (`30 SMs, 595.99.02`) | `+43.25~53.14 µs` | `4.50 µs` → **`4.93–5.47 µs`** | **`+0.43–0.97 µs`** | **45×–100×** |

> **开销来源说明**：`v2.2.1` 将 `Sm120ControlHeader` 与 `Sm120CtaEntry[65536]` 全部迁入 GPU GDDR7 显存后，`cmd == 0` 路径彻底脱离 PCIe 往返并命中 GPU L2 Cache，双卡常驻空载开销均压缩至 **`< 1 µs`**。

### 3.2 Mode 1 vs Mode 5 抢占（Evict）与 0 脉冲自陷恢复（Restore）延迟（`N = 20`）

| 负载配置 | 平台 | `v2.0/v2.1` Evict / Restore P50 (µs) | **`v2.2.1` Evict P50 / P95 (µs)** | **`v2.2.1` Restore (0-Pulse) P50 / P95 (µs)** | `u_arena` 占用 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Mode 1 (`512 CTAs, 48K SMEM`)** | RTX 5080 | `971.6 / 419.9` | **`110.1` / `115.3`** (**8.8×↑**) | **`387.6` / `401.4`** | **`0 KiB`** |
| **Mode 5 (`84 CTAs, 0K SMEM`)** | RTX 5080 | `332.3 / 763.6` | **`50.9` / `52.1`** (**6.5×↑**) | **`30.5` / `31.2`** (**25.0×↑**) | `2,176 KiB` |
| **Mode 5 (`84 CTAs, 32K SMEM`)** | RTX 5080 | `372.8 / 831.8` | **`55.4` / `56.4`** (**6.7×↑**) | **`41.4` / `42.3`** (**20.1×↑**) | `4,864 KiB` |
| **Mode 5 (`512 CTAs, 48K SMEM`)** | RTX 5080 | `1,260.8 / 1,859.0` | **`119.9` / `121.1`** (**10.5×↑**) | **`292.0` / `293.1`** (**6.4×↑**) | `24,704 KiB` |
| **Mode 1 (`512 CTAs, 48K SMEM`)** | RTX 5060 | `882.0 / 298.5` | **`38.0`** (**23.2×↑**) | **`~200`** | **`0 KiB`** |
| **Mode 5 (`84 CTAs, 0K SMEM`)** | RTX 5060 | `374.5 / 754.2` | **`50.8`** (**7.4×↑**) | **`62.9`** (**12.0×↑**) | `2,176 KiB` |
| **Mode 5 (`84 CTAs, 32K SMEM`)** | RTX 5060 | `427.5 / 794.0` | **`55.2`** (**7.7×↑**) | **`73.9`** (**10.7×↑**) | `4,864 KiB` |
| **Mode 5 (`512 CTAs, 48K SMEM`)** | RTX 5060 | `1,059.3 / 842.0` | **`48.0`** (**22.1×↑**) | **`325.8`** (**2.6×↑**) | `5,632 KiB` |

> **关键观察**：在 RTX 5080 满载 84 SMs 下，为 84 个 CTA 额外保存 `32 KiB` 共享内存（总计 `2.625 MiB`）仅使 Evict P50 从 `50.9 µs` 增至 `55.4 µs`（**`+4.5 µs`**），0 脉冲 Restore P50 仅从 `30.5 µs` 增至 `41.4 µs`（**`+10.9 µs`**），充分释放 Blackwell GDDR7 显存带宽与零脉冲用户态自陷恢复的极限性能。

### 3.3 紧凑 `slot_id` 池化 `u_arena` 显存缩放（128 threads/CTA, ~32 regs, 0 SMEM）

| Grid 规模 | v1 朴素按 Grid 预分配 (64K 固定 SMEM) | `v2.2.1` 紧凑池化（RTX 5080, 84 SMs） | `v2.2.1` 紧凑池化（RTX 5060, 30 SMs） |
| :---: | :---: | :---: | :---: |
| `256 CTAs` | `64.00 MiB` | `6.38 MiB` (**10.0×** 缩减) | `6.38 MiB` (**10.0×** 缩减) |
| `1,024 CTAs` | `256.00 MiB` | `25.50 MiB` (**10.0×** 缩减) | `9.38 MiB` (**27.3×** 缩减) |
| `4,096 CTAs` | `1,024.00 MiB` | `29.31 MiB` (**34.9×** 缩减) | `9.38 MiB` (**109.2×** 缩减) |
| **`16,384 CTAs`** | **`4,096.00 MiB`** | **`29.31 MiB` (`139.7×` 缩减)** | **`9.38 MiB` (`436.9×` 缩减)** |

---

## 4. 开发期攻克的 Blackwell 微架构与系统根因实录（摘要，完整 15 项详解见完整技术手册）

1. **驱动版本间 `.text.traphandler` 基址漂移（`580.95.05` vs `595.99.02`）**：不同驱动的 ToolsMemory 内部偏移存在 `+0x5900` 漂移，但 `tools_pc - handler_va == 0x4400` 恒定；动态指纹扫描与通用 `CALL.ABS` 编码实现跨驱动零硬编码。
2. **部分已退出 CTA（Partially-Exited CTA）**：引入 `cta[b].warp_mask` + `@P0 ATOMG.E.OR` 动态 CTA Leader 选举 + `Cave 8` Partial-CTA 32-lane 合并 SMEM 保存 + `Cave 1` 死 Warp `c1_clean_exit_idx` 清理退出。
3. **SM120 `BRX Rn` 消费 64-bit 寄存器对 `Rn:Rn+1`**：在 `Cave 2/5` 的 `BRX R10` 前显式 `mov_imm(11, 0)` 消除高位残留 `0xffffffff` 负跳。
4. **`REG:255`（`SR_REGALLOC == 256`）最高寄存器 `R253` 边界**：`R254:R255` 为硬件 Trap 寄存器，在 `Cave 2/5` 增加 `SR_REGALLOC - 3 >= 253` 显式谓词保护。
5. **100% GDDR7 VRAM 控制面 + 纯 Copy Engine DMA（`v2.2.0`）**：消除 PCIe Zero-Copy 原子风暴与大网格 4.3× 膨胀，且严禁在抢占路径调用 Compute Engine 的 `cuMemsetD32Async`。
6. **BAR0 `0x00424684` D10 法则**：写入 `0x80001c01` 后仅读取非 PRI `NV_PTIMER_TIME_0 (0x00009400)` 刷新首个 Posted Write 再写 `0x00001c01`。
7. **SMEM `+0x400` 硬件 4 KiB 分配量子（`v2.2.1`）**：硬件先加 `0x400 B` 驱动保留区再按 4 KiB 对齐，因此 `smem_area = AlignUp(total_user_smem + 0x400, 4096)`（消除 `CUB Onesweep` `36,352 B -> 40,960 B` 末槽越界）。
8. **完整 15 项根因、9-Cave 逐指令逻辑与双卡 Suite 1–3 全量数据**：请参阅 [`SM120_V2_COMPLETE_TECHNICAL_MANUAL_ZH.md`](SM120_V2_COMPLETE_TECHNICAL_MANUAL_ZH.md) 与 Word 报告 [`Blackwell_SM120_Level3_Preemption_Technical_Manual_v2.2.1_ZH.docx`](Blackwell_SM120_Level3_Preemption_Technical_Manual_v2.2.1_ZH.docx)。
