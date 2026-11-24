# Blackwell (`sm_120`) Level-3 v2 实机验证与性能基准报告

**验证日期**：2026-11-23  
**构建标准**：`C++20 -fPIC -Wall -Wextra -Werror`（全目标 0 错误、0 警告）  
**验证平台**：
- **主平台 (`5080`)**：NVIDIA GeForce RTX 5080（`GB203`, `sm_120`, 84 SMs, `100 KiB` SMEM/SM, Driver `580.95.05`, CUDA `12.9`, `g++-13`）
- **交叉验证平台 (`5060`)**：NVIDIA GeForce RTX 5060（`GB206`, `sm_120`, 30 SMs, `100 KiB` SMEM/SM, Driver `595.99.02`, Ubuntu 26.04 桌面环境）

---

## 1. 基础正确性与工具链核验（全部 PASS）

| 回归项 | 验证平台 | 关键实测指标 | 结果 |
| :--- | :--- | :--- | :---: |
| **零缓存全量构建** (`build_sm120`) | RTX 5080 (`CUDA 12.9`) | `C++20 -Wall -Wextra -Werror`，全部核心库 + 5 项测试目标 **0 警告** | **PASS** |
| **`sm120_sass_disasm` + `nvdisasm -b SM120`** | RTX 5080 (`CUDA 12.9`) | **79 / 79** 条手写 128-bit SM120 SASS 指令反汇编逐条核验通过 | **PASS** |
| **`sm120_cave_probe`（代码洞与引擎探测）** | RTX 5080 (`580.95.05`) | `total_caves=11 total_slots=1703`, `select_9_ge_144=PASS`, `c0=143/152 c1=119/154 ...` (`1,049/1,390`), `engine_init=PASS` | **PASS** |
| **`sm120_cave_probe`（跨驱动交叉探测）** | RTX 5060 (`595.99.02`) | `total_caves=11 total_slots=1695`, `select_9_ge_144=PASS`, `c0=143/152 c1=119/154 ...` (`1,049/1,390`), `engine_init=PASS`, `bar0=MAPPED` | **PASS** |
| **`sm120_l3_resumable_bench`（Context CILP）** | RTX 5080 (`580.95.05`) | `--iters 5 --multi_pause 3 --check_exact_sum 1`：`P50=425 µs`（多次运行稳定 `423–427 µs`）, `lost_work=0`, `diff=0`, `hp_errors=0` | **PASS** |

---

## 2. 端到端正确性与压力测试（`sm120_v2_e2e_suite` — RTX 5080）

在空闲 RTX 5080 上连续两次独立执行 `sm120_v2_e2e_suite`，均稳定达成 **`11 / 11 CASES PASSED (ALL_PASS)`**（exit code `0`，且 Cases 01–10 与 Case 11 全部 100 轮常规抢占均为严格单脉冲 0 重试）：

| Case | 测试场景与覆盖维度 | RTX 5080 实测输出摘要 | 结果 |
| :---: | :--- | :--- | :---: |
| **01** | Mode 1 多波次幂等早退（`512 CTAs, 48 KiB SMEM`） | `mode1_exit=168 replay_marked=512 leaks=0 errs=0` | **PASS** |
| **02** | Mode 5 单波次非幂等 `atomicAdd`（`84 CTAs`） | `evicted=84 restored=84 drained=0 arena=2176 KiB`, `got=107520/107520` | **PASS** |
| **03** | Mode 5 多波次 `State A + B + C` 零泄漏（`512 CTAs, 48 KiB`） | `A=40 B=151..154 C=321..318 (sum=512/512) cta_errs=0 ovf=0 leaks=0 total=524288/524288` | **PASS** |
| **04** | 全 3D 网格与线程块坐标 `Grid(4,3,2) × Block(8,4,2)` | `evicted=24 restored=24 arena=448 KiB`, `errs=0/1536` | **PASS** |
| **05** | 极限高寄存器级联 `R0..R253`（`num_regs=255`）+ `UR/UPR` | `CASE05-A (<96>)` & `CASE05-B (<180>)`：`evicted=16 restored=16 arena=5248 KiB`, `errs=0/1024` | **PASS** |
| **06** | 静态 + 动态共享内存（`36 KiB = 4K + 32K`） | `evicted=32 restored=32 arena=2240 KiB`, `errs=0/4096` | **PASS** |
| **07** | 栈溢出 + `__noinline__` 递归局部内存（`LMEM`） | `evicted=32 restored=32 arena=8896 KiB`, `static_lmem=128 B`, `errs=0/2048` | **PASS** |
| **08** | 0~15 号命名屏障（`bar.sync 0,1,5,7,11,15 + bar.arrive 11`） | `evicted=8 restored=8 arena=448 KiB`, `errs=0/2048` | **PASS** |
| **09** | 深层嵌套分支分化收敛屏障（`B1..B15`） | `evicted=16 restored=16 arena=448 KiB`, `errs=0/2048` | **PASS** |
| **10** | XSched `AsyncXQueue` FIFO `[K1->K2]` + 同函数 `DirectLaunch` HP | `k2_mid=0 errs(k1=0, k2=0, hp=0)` | **PASS** |
| **11** | **100 轮多波次连续压测（含每 10 轮双连抢占 `State A` 保留）** | `errs=0`（全部 100 轮 × `336 CTAs × 128 × 4` 位精确一致，含极端 PCIe 原子风暴下的双连抢占） | **PASS** |

---

## 3. 性能基准测试（`sm120_v2_bench` — RTX 5080 & RTX 5060）

`sm120_v2_bench` 在未修改的连续计算循环内核上（设备 VRAM 停止标志，无人工 PCIe 原子风暴干扰）实测全部 80 轮次 **0 次恢复脉冲重试（100% 单脉冲 Evict + 0 脉冲自陷 Restore）**。

### 3.1 Entry Guardian 常驻空载发射开销（`84 × 128` 空内核，`N = 100`）

| 平台 | 未插桩 Baseline P50 (P95) | Guarded `cmd=0` P50 (P95) | 净增量 Delta |
| :--- | :---: | :---: | :---: |
| **RTX 5080** (`84 SMs, 580.95.05`) | `4.92 µs` (`5.24 µs`) | `35.70 µs` (`36.34 µs`) | `+30.78 µs` |
| **RTX 5060** (`30 SMs, 595.99.02`) | `4.62 µs` (`4.92 µs`) | `57.76 µs` (`66.60 µs`) | `+53.14 µs` |

> **开销来源说明**：`cmd == 0` 路径上每个 CTA 需经 PCIe 读取一次主机映射的 `ctrl_host_->cmd` 并执行入口 `BAR.SYNC 0`。对于毫秒级真实 GPU 计算内核，数十微秒的一次性入口同步开销占比 `< 1%`。

### 3.2 Mode 1 vs Mode 5 抢占（Evict）与 0 脉冲自陷恢复（Restore）延迟（`N = 20`）

| 负载配置 | 平台 | Evict P50 / P95 / Max (µs) | Restore (0-Pulse) P50 / P95 / Max (µs) | `u_arena` 占用 |
| :--- | :---: | :---: | :---: | :---: |
| **Mode 1 (`512 CTAs, 48K SMEM`)** | RTX 5080 | **`971.6`** / `984.5` / `984.5` | **`419.9`** / `424.9` / `424.9` | **`0 KiB`** |
| **Mode 5 (`84 CTAs, 0K SMEM`)** | RTX 5080 | **`332.3`** / `343.2` / `343.2` | **`763.6`** / `837.0` / `837.0` | `2,176 KiB` |
| **Mode 5 (`84 CTAs, 32K SMEM`)** | RTX 5080 | **`372.8`** / `385.7` / `385.7` | **`831.8`** / `865.3` / `865.3` | `4,864 KiB` |
| **Mode 5 (`512 CTAs, 48K SMEM`)** | RTX 5080 | **`1,260.8`** / `1,269.6` / `1,269.6` | **`1,859.0`** / `1,903.9` / `1,903.9` | `13,568 KiB` |
| **Mode 1 (`512 CTAs, 48K SMEM`)** | RTX 5060 | **`882.0`** / `926.0` / `926.0` | **`298.5`** / `397.7` / `397.7` | **`0 KiB`** |
| **Mode 5 (`84 CTAs, 0K SMEM`)** | RTX 5060 | **`374.5`** / `415.8` / `415.8` | **`754.2`** / `863.1` / `863.1` | `2,176 KiB` |
| **Mode 5 (`84 CTAs, 32K SMEM`)** | RTX 5060 | **`427.5`** / `483.4` / `483.4` | **`794.0`** / `925.4` / `925.4` | `4,864 KiB` |
| **Mode 5 (`512 CTAs, 48K SMEM`)** | RTX 5060 | **`1,059.3`** / `1,115.9` / `1,115.9` | **`842.0`** / `1,081.8` / `1,081.8` | `5,632 KiB` |

> **关键观察**：在 RTX 5080 满载 84 SMs 下，为 84 个 CTA 额外保存 `32 KiB` 共享内存（总计 `2.625 MiB`）仅使 Evict P50 从 `332.3 µs` 增至 `372.8 µs`（`+40.5 µs`），0 脉冲 Restore P50 从 `763.6 µs` 增至 `831.8 µs`（`+68.2 µs`），充分体现了 Blackwell GDDR7 显存带宽与零脉冲用户态自陷恢复的高效性。

### 3.3 紧凑 `slot_id` 池化 `u_arena` 显存缩放（128 threads/CTA, ~32 regs, 0 SMEM）

| Grid 规模 | v1 朴素按 Grid 预分配 (64K 固定 SMEM) | v2 紧凑池化（RTX 5080, 84 SMs） | v2 紧凑池化（RTX 5060, 30 SMs） |
| :---: | :---: | :---: | :---: |
| `256 CTAs` | `64.00 MiB` | `6.38 MiB` (**10.0×** 缩减) | `6.38 MiB` (**10.0×** 缩减) |
| `1,024 CTAs` | `256.00 MiB` | `25.50 MiB` (**10.0×** 缩减) | `9.38 MiB` (**27.3×** 缩减) |
| `4,096 CTAs` | `1,024.00 MiB` | `25.50 MiB` (**40.2×** 缩减) | `9.38 MiB` (**109.2×** 缩减) |
| **`16,384 CTAs`** | **`4,096.00 MiB`** | **`25.50 MiB` (`160.6×` 缩减)** | **`9.38 MiB` (`436.9×` 缩减)** |

---

## 4. 开发期攻克的 8 项关键 Blackwell 微架构根因实录

1. **驱动版本间 `.text.traphandler` 基址漂移（`580.95.05` vs `595.99.02`）**：不同驱动的 ToolsMemory 内部偏移存在 `+0x5900` 漂移，但 `tools_pc - handler_va == 0x4400` 保持恒定。通过动态指纹扫描与通用 `CALL.ABS` 编码彻底消除硬编码依赖。
2. **部分已退出 CTA（Partially-Exited CTA）导致恢复崩溃**：快路径退出 CTA 会先执行 `@P0 EXIT` 释放 Warp 1..3，仅留 Warp 0 完成尾声原子操作。引入 `cta[b].warp_mask` + `@P0 ATOMG.E.OR` 动态 CTA Leader 选举 + `Cave 8` Partial-CTA 32-lane 合并 SMEM 保存 + `Cave 1` 死 Warp `c1_clean_exit_idx` 清理退出，彻底杜绝 `Error 700`。
3. **SM120 `BRX Rn` 消费 64-bit 寄存器对 `Rn:Rn+1`**：`Cave 8` 恢复硬件帧后在 `R11` 残留 `0xffffffff`，导致 `Cave 2/5` `BRX R10` 负跳 `-4 GiB`。在 `BRX R10` 前补齐 `mov_imm(11, 0)` 后 Cases 04–09 全部零误码通过。
4. **`REG:255`（`SR_REGALLOC == 256`）最高寄存器 `R253` 边界**：Blackwell 上 `R254:R255` 为硬件 Trap 寄存器，`R0..R253` 为用户可用寄存器。在 `Cave 2/5` 增加 `SR_REGALLOC - 3 >= 253` 显式谓词保护，并调整 `reg_ceil` 上限为 `256`。
5. **`cuMemcpyHtoDAsync_v2` 对 Host-Mapped 内存同步阻塞死锁**：在多内核 FIFO 队列场景下导致 `LaunchWorker` 持锁阻塞 `Suspend()`。改用 576B Device VRAM `layout_stage_dev_` 异步双跳中转并缓存 `saved_arena_dev_`，Case 10 立即在 `< 10 ms` 内通过。
6. **BAR0 `0x00424684` 置位期读取引发跨 GPC 广播中断（D10）**：在 `0x80001c01` 与 `0x00001c01` 之间读取 `0x00424684` 会打断跨 GPC PRI 广播。改用非 PRI 的 `NV_PTIMER_TIME_0 (0x00009400)` 刷新首个 Posted Write，兼顾 D10 安全与干净 `~1 µs` 脉冲宽度。
7. **多轮双连抢占与伴随入陷竞态（Case 11）**：高并发 608-Warp PCIe 原子风暴下，上一轮 Restore 伴随入陷 Warp 与新一轮 Evict 存在微秒级交叠窗口，导致 1~3 个 SM（2/4/6 CTAs）在首脉冲下未能完成换出。引入 `trap_inflight` 原子在途计数、`c1_noop_recheck_idx` 伴随重检、`c1_dispatch_guard_idx` 的 `TRAP_RETURN_PC == exit_stub_va` 重入防护以及 `15 ms` 进度感知有界恢复确认后，连续两次 100 轮压测均 `11 / 11 ALL_PASS`。
8. **基准内核 PCIe 原子序列化失真**：若测试内核在 `while` 等待循环中使用全线程 `atomicAdd(host_mapped, 0)`，65,536 线程串行竞争单条 PCIe 缓存行会人为引入 `~47 ms` 延迟。将 `sm120_v2_bench` 的停止标志置于设备 VRAM 配合真实 FMA 计算循环后，准确还原了亚毫秒至低毫秒级的真实硬件抢占与 0 脉冲恢复性能。
