# XSched Blackwell (`sm_120`) Level-3 Hardware Trap Preemption (v2)

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

### 核心实测亮点（`v2.2.0-sm120`, RTX 5060 `GB206`）

| 评估维度 | 关键指标（`v2.2.0` 实测） | 较 `v2.1.0` 提升 / 对比 |
| :--- | :--- | :--- |
| **微基准（Entry Guardian / Mode 5 Evict）** | `0.43–0.97 µs` / `48.0 µs`（512 CTAs, 48K SMEM） | **45×–100×** / **22.4×** 加速 |
| **8 大闭源 CUDA 库 + 7 项 Blackwell 硬件特性** | **15 / 15 PASS**（`cuBLAS/Lt/cuFFT/cuSPARSE/CUB` + `mma/cp.async/mbarrier/Graph/Cluster/Coop`） | `cuBLAS` `0.86 ms` (**77×**), `cuSPARSE` `0.11 ms` (**466×**) |
| **LLM Serving (`15 ms` QoS, Mode 5)** | `HP P50 = 0.475 ms`, `SUSP P50 = 57 µs`, `SLO ≤ 2 ms = 100%`, `LP_ret = 98.2%` | **67.7×** HP 加速 / **548×** 抢占加速 / `err = 0.00e+00` |
| **100% SM 饱和 Heavy QoS (`5–50 ms` 全频段)** | `HP P50 = 0.083–0.088 ms`, `SUSP P50 = 66 µs`, `SLO ≤ 2 ms = 100%`, `LP_ret = 98.7%–105%` | 较 Lv1 延迟低 **85×**，`err = 0.00e+00` |
| **真实未修改 PyTorch 2.11 + cuDNN 9.19 + Triton 3.6.0** | **11 / 11 PASS**（含 ResNet-18 AdamW 训练 `5.5e-9`、4 种 Llama SDPA、Triton、`PT-06` `empty_cache` 零死锁） | `PT-05-QOS`: `HP P50 = 0.552 ms` (**133.6×**), `SLO ≤ 2 ms = 100%`, `LP_ret = 101.9%` |

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
| [`PROJECT.md`](PROJECT.md) | 研究背景、目录结构、核心 HAL 模块清单与双端开发/安全规范 |
| [`doc/README.md`](doc/README.md) | 技术文档总览与阅读索引 |
| [`doc/SM120_V2_ARCHITECTURE_ZH.md`](doc/SM120_V2_ARCHITECTURE_ZH.md) | Blackwell SM120 v2 静态 9-Cave 引擎、Entry Guardian、状态机、内存布局与硬件安全法则详解 |
| [`doc/SM120_V2_VERIFICATION_AND_BENCHMARK_ZH.md`](doc/SM120_V2_VERIFICATION_AND_BENCHMARK_ZH.md) | RTX 5080 与 RTX 5060 全量实机正确性验证、微秒级延迟/显存基准与微架构根因实录 |
| [`doc/SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md`](doc/SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md) | 闭源 CUDA 库、Blackwell 硬件边界、PyTorch/cuDNN/Triton、DNN/LLM 调度 QoS 全方位评测、测试方法论与缺陷全景分析 |
| [`doc/SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md`](doc/SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md) | **项目阶段总汇报**：当前状态、已达成成果、10 项现存问题台账（P0/P1/P2）、Phase 1–3 演进与论文规划及 5080 同步手册 |
| [`doc/SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md`](doc/SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md) | **RTX 5060 vs RTX 5080 适配与差异对比**：`GB206 (595.x)` vs `GB203 (580.x)` 硬件/驱动/工具链/延迟/显存差异全解 |
| [`CHANGELOG.md`](CHANGELOG.md) | 版本发布与修复变更日志（`v2.0.0-sm120` → `v2.1.0-sm120` → `v2.2.0-sm120`） |
| [`xsched/docs/xsched-osdi25.pdf`](xsched/docs/xsched-osdi25.pdf) | XSched OSDI'25 原始论文 |
