# XSched Blackwell (`sm_120`) Level-3 Hardware Trap Preemption (v2)

> **📌 当前状态（2026-11-25，Phase 3 生产负载测试）**：在 29 个 PyTorch 生产算子、真实模型服务与新增合成 Trap 测试中发现 **DEF-20（P0）：`v2.2.1` 的 Trap 引擎漏存每个内核最高 1–3 个寄存器**（下文“`R0..R253` 全量寄存器级联”在 `v2.2.1` 中并不成立），它是 dtype 转换/CUB/topk 恢复崩溃、Triton RMSNorm 静默错误及 LIB-08/LLM-07 偶发错误的共同根因；修复已随本次提交合入并在两卡验证（Suite 7 Lv3 29/29）。开放项：DEF-21（恢复中途再抢占）、DEF-19（Cluster+DSMEM）、DEF-17（Guardian 每 CTA 开销）。
> **📌 Phase 3.2 / 3.3 完成（2026-11-26）**：DEF-21 关闭、DEF-19 缓解、DEF-22/23/24 修复；DEF-17 达标（Guardian 去冗余重读 + 短 CTA 路由 + `BAR.RED` 屏障 + 门控镜像：5080 `transpose_copy` +82% → −0.0%、`rope_apply` +101% → +2.2%、`sort` +25% → +3.3%，29 个生产算子 Lv3 中位数两卡均 +0.3%、最高 +5.7%）；每调用开销达标（自适应直接提交 + 调用方 `WaitAll` + 布局缓存：单内核调用 Lv1 +1.1–1.2 µs、Lv3 +2.2–2.4 µs）；PERF-2 修复（HP 透传发射，5080 decode TPOT 1.23× → ≈1.0×）；MEM-1 arena 空闲回收；调度策略在低负载下 LP > 80%（TPOT P50 ≤ 1.07×）、原负载下时间/空间共享均已量化；一键回归流水线（含 3.2 验收组）。最终构建两卡全量回归 **265/265 + 265/265**，详见 [Phase 3.2/3.3 报告](doc/SM120_V2_PHASE3_2_3_3_REPORT_ZH.md)。
> 请先阅读：[Phase 3.2/3.3 报告](doc/SM120_V2_PHASE3_2_3_3_REPORT_ZH.md) · [阶段总结与规划](doc/SM120_V2_PHASE3_SUMMARY_AND_ROADMAP_ZH.md) · [缺陷台账](doc/SM120_V2_PHASE3_DEFECT_LEDGER_ZH.md) · [Phase 3 测试报告](doc/SM120_V2_PHASE3_PRODUCTION_OPS_GROUNDED_TESTS_ZH.md)

面向 **NVIDIA Blackwell (`sm_120`)** 架构的用户端透明细粒度硬件抢占与调度系统（基于上海交通大学 IPADS 实验室 [XSched](xsched/docs/xsched-osdi25.pdf) 框架扩展，仅用于学术研究）。

本仓库已全面完成 **v2 极简通用架构重构**，在**不修改任何 CUDA 内核源码**、**不依赖幂等性假设**的前提下，实现了跨 `RTX 5080`（`GB203`, 84 SMs, Driver `580.95.05`, CUDA 12.9）与 `RTX 5060`（`GB206`, 30 SMs, Driver `595.99.02`）的硬件 Trap 级空间抢占、零脉冲原位状态恢复与 Context 级 CILP 抢占。

---

## 核心特性（v2 架构）

- **极简双模式设计**：
  - **Mode 1 (`kEarlyExit`)**：面向幂等内核的零显存开销早退与无泄漏 `State C` 拦截。
  - **Mode 5 (`kTrapFrame`)**：面向非幂等内核的全架构硬件状态保存与恢复——覆盖全 3D 网格/线程块、`R0..R253` 全量寄存器级联（`SR_REGALLOC + BRX`）、`UR0..UR63 + UPR + B0..B15`、`0..15` 号命名屏障（`B2R`/`R2B`）、动态 `SR_SMEMSZ` 共享内存、栈溢出及 `__noinline__` 递归局部内存（LMEM）、部分已退出 CTA（`warp_mask` + 动态 Leader 选举）。
- **静态 9-Cave 引擎 + 100% GDDR7 显存控制面 (`v2.2.0`)**：进程初始化时一次性在 `.text.traphandler` ToolsMemory 中静态部署全部 9 个代码洞（`1,045 / 1,390` 工作槽位，全 `C++20 -Wall -Wextra -Werror` 零警告），零运行期热补丁；控制头与 `Sm120CtaEntry[65536]` 驻留 GPU GDDR7 显存（配合高优先级 Copy Engine DMA 通道与 1 线程 GPU 原生条件清理 `sm120_cond_clear_hdr`），彻底消除 PCIe Zero-Copy 瓶颈，Entry Guardian 开销低至 **`0.43–0.97 µs`**、Mode 5 抢占延迟低至 **`48–70 µs`**。
- **自适应内核/模式路由器 + 生产级运行时加固**：静态扫描 SM120 `SYNCS`（`mbarrier`）指令、完整拦截 `cuLaunchCooperativeKernel[_ptsz]`、自动识别多波次 Cluster 与 CUDA Graph 并安全降级至 Level-1 边界；支持 `XMemFree_v2` 延迟释放队列（彻底消除抢占挂起期间 `cudaFree` / `torch.cuda.empty_cache()` 死锁）、无上下文工作线程 `ScopedStreamContext` 回退与 `cuDNN` 动态 SMEM 竞态自愈。
- **永久 Entry Guardian + `BPT.TRAP 0x1` 0 脉冲自陷恢复**：首次发射前注入 `0x500` B（80 槽位）入口守护序言；Evict 阶段单次 BAR0 `0x00424684` 广播脉冲拦截并换出全部在途 CTA，Restore 阶段由恢复波次自主执行用户态 `BPT.TRAP 0x1` 自陷恢复（0 Host 脉冲）。
- **紧凑 `slot_id` 池化 `u_arena`**：按物理 SM 并发上限动态池化显存，16,384 CTAs 网格下较朴素按 Grid 分配方案节省 **160.6×**（RTX 5080）/ **436.9×**（RTX 5060）显存。
- **XSched HAL 闭环 + 独立 Context CILP**：`CudaQueueLv3Trap` 完整对接 `AsyncXQueue` 多内核 FIFO 队列与高优 `DirectLaunch` 零开销绕过；同时独立保留 `ResumableSm120Context`（`0x20801210` / `0xa06c0105` 上下文级 CILP 抢占）。

### 核心实测亮点（`v2.2.1-sm120`，RTX 5080 `GB203` + RTX 5060 `GB206` 双卡闭环实测）

| 评估维度 | RTX 5080 (`GB203`, 84 SMs, `580.95.05`) | RTX 5060 (`GB206`, 30 SMs, `595.99.02`) |
| :--- | :--- | :--- |
| **微基准（Guardian / Mode 5 `84 CTAs` Evict & Restore）** | **`0.99 µs`** / **`50.9 µs`** & **`30.5 µs`**（`11/11 E2E ALL_PASS`） | **`0.43–0.97 µs`** / **`50.8 µs`** & **`62.9 µs`**（较 `v2.1.0` 快 **22×–100×**） |
| **Suite 1 闭源 CUDA 库 + Blackwell 特性** | **15 / 15 PASS**（`cuBLAS 80–137 µs`, `CUB 55 µs`, 单波次 Cluster Mode 5 `111 µs`）+ `DEF-03/06` PASS (`34 µs`) | **15 / 15 PASS**（`cuBLAS 0.86 ms`, `cuSPARSE 0.11 ms`, 多波次 Cluster 自动 Lv1）+ `DEF-03/06` PASS (`68 µs`) |
| **Suite 2 LLM Serving (`15 ms`, Mode 5)** | `HP P50 = 0.270 ms`, **`SUSP P50 = 32 µs`**, `SLO≤2ms = 100%`, **`LP_ret = 102.4%`** | `HP P50 = 0.475 ms`, **`SUSP P50 = 57 µs`**, `SLO≤2ms = 100%`, **`LP_ret = 98.2%`** |
| **Suite 2 100% SM 饱和 Heavy QoS (`15 ms`, Mode 5)** | **`HP P50 = 0.060 ms`**（较 Lv1 低 **102.6×**）, **`SUSP P50 = 45 µs`**, `SLO≤2ms = 100%`, **`LP_ret = 103.4%`** | **`HP P50 = 0.083 ms`**（较 Lv1 低 **85×**）, **`SUSP P50 = 66 µs`**, `SLO≤2ms = 100%`, **`LP_ret = 100.7%`** |
| **Suite 3 真实未修改 PyTorch 2.11 + cuDNN 9.19 + Triton 3.6.0** | **11 / 11 PASS**（ResNet-50 `70 µs`, ResNet-18 训练 `31 µs`, `cuDNN` SDPA **`23 µs`**, Triton `126 µs`, `PT-05-QOS HP P50 = 0.290 ms / LP_ret = 105.3%`） | **11 / 11 PASS**（ResNet-50 `291 µs`, ResNet-18 训练 `23 µs`, `cuDNN` SDPA **`33 µs`**, Triton `268 µs`, `PT-05-QOS HP P50 = 0.552 ms / LP_ret = 101.9%`） |
| **跨进程 BAR0 旁观者干扰 (`bystander_pure`)** | **`0.011 ms → 0.011 ms`**（P99/Max 亦 `0.012 → 0.012 ms`, 0 错误 0 错配） | **`0.015 ms → 0.015 ms`**（0 错误 0 错配） |

---

## 快速构建与验证（CUDA 12.9+ / `sm_120`）

```bash
cd xsched
cmake -S . -B build_sm120 \
  -DCMAKE_BUILD_TYPE=Release \
  -DPLATFORM_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/home/wangyue/.local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build_sm120 -j$(nproc)
```

### 5 项标准回归套件

> **安全提示**：BAR0 MMIO (`0x00424684`) 触发需要 `root` 权限；在共享服务器上运行前务必通过 `nvidia-smi` 确认无其他用户的计算任务在运行，并设置 `CUDA_MPS_PIPE_DIRECTORY=/tmp/no_mps_wangyue` 以避免 MPS 干扰。

```bash
cd xsched/build_sm120
export CUDA_LIB=/home/wangyue/.local/cuda-12.9/lib64

# 1. 79 条 SM120 128-bit SASS 编码反汇编核验
./bin/sm120_sass_disasm && nvdisasm -b SM120 /tmp/sm120_sass_check.bin > /dev/null && rm -f /tmp/sm120_sass_check.bin

# 2. .text.traphandler 代码洞探测与静态 9-Cave 引擎初始化验证
LD_LIBRARY_PATH="$CUDA_LIB" CUDA_MPS_PIPE_DIRECTORY=/tmp/no_mps_wangyue ./bin/sm120_cave_probe

# 3. 11 项端到端正确性与 100 轮双连抢占压测（11 / 11 ALL_PASS）
sudo env LD_LIBRARY_PATH="$CUDA_LIB" CUDA_MPS_PIPE_DIRECTORY=/tmp/no_mps_wangyue timeout 90s ./bin/sm120_v2_e2e_suite

# 4. Entry Guardian / Mode 1 vs Mode 5 延迟与紧凑 u_arena 显存缩放基准
sudo env LD_LIBRARY_PATH="$CUDA_LIB" CUDA_MPS_PIPE_DIRECTORY=/tmp/no_mps_wangyue timeout 90s ./bin/sm120_v2_bench

# 5. Context 级 CILP 可恢复抢占回归验证
sudo env LD_LIBRARY_PATH="$CUDA_LIB" CUDA_MPS_PIPE_DIRECTORY=/tmp/no_mps_wangyue timeout 60s \
  ./bin/sm120_l3_resumable_bench --iters 5 --multi_pause 3 --check_exact_sum 1
```

---

## 文档导航

| 文档 | 内容说明 |
| :--- | :--- |
| **[`doc/Blackwell_SM120_Level3_Preemption_Technical_Manual_v2.2.1_ZH.docx`](doc/Blackwell_SM120_Level3_Preemption_Technical_Manual_v2.2.1_ZH.docx)** | **【推荐·35 页排版 Word 完全技术手册（`v2.2.1-sm120`）】** 10 章全景覆盖总体架构、ToolsMemory 11 代码洞、SM120 128-bit SASS 编码与 79 指令、80 槽 Entry Guardian、静态 9-Cave 指令流、GDDR7 控制面与 `u_arena` 承载恢复、BAR0 D10 与安全路由、15 项微架构发现、RTX 5080 & 5060 双卡实测与 Phase 3 规划 |
| **[`doc/SM120_V2_COMPLETE_TECHNICAL_MANUAL_ZH.md`](doc/SM120_V2_COMPLETE_TECHNICAL_MANUAL_ZH.md)** | **【完整技术手册 Markdown 源文（`v2.2.1-sm120`）】** 与上述 `.docx` 完全同步的 10 章详尽技术手册（适合 GitHub 在线阅读） |
| [`PROJECT.md`](PROJECT.md) | 研究背景、目录结构、核心 HAL 模块清单与双端开发/安全规范 |
| [`doc/README.md`](doc/README.md) | 技术文档总览与阅读索引 |
| [`doc/SM120_V2_ARCHITECTURE_ZH.md`](doc/SM120_V2_ARCHITECTURE_ZH.md) | Blackwell SM120 v2 静态 9-Cave 引擎、Entry Guardian、状态机、内存布局与硬件安全法则详解 |
| [`doc/SM120_V2_VERIFICATION_AND_BENCHMARK_ZH.md`](doc/SM120_V2_VERIFICATION_AND_BENCHMARK_ZH.md) | RTX 5080 与 RTX 5060 全量实机正确性验证、微秒级延迟/显存基准与微架构根因实录 |
| [`doc/SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md`](doc/SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md) | 闭源 CUDA 库、Blackwell 硬件边界、PyTorch/cuDNN/Triton、DNN/LLM 调度 QoS 全方位评测、测试方法论与缺陷全景分析 |
| [`doc/SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md`](doc/SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md) | **项目阶段总汇报**：当前状态、已达成成果、10 项现存问题台账（P0/P1/P2）、Phase 1–3 演进与论文规划及 5080 同步手册 |
| [`doc/SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md`](doc/SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md) | **RTX 5060 vs RTX 5080 适配与差异对比**：`GB206 (595.x)` vs `GB203 (580.x)` 硬件/驱动/工具链/延迟/显存差异全解 |
| [`CHANGELOG.md`](CHANGELOG.md) | 版本发布与修复变更日志（`v2.0.0-sm120` → `v2.1.0-sm120` → `v2.2.0-sm120` → `v2.2.1-sm120`） |
| [`xsched/docs/xsched-osdi25.pdf`](xsched/docs/xsched-osdi25.pdf) | XSched OSDI'25 原始论文 |
