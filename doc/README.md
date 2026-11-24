# Blackwell (`sm_120`) Level-3 v2 技术文档中心

本目录仅保留与当前 **Blackwell SM120 v2** 代码基线完全同步的权威技术文档（历史探索草稿与 v1 过渡文档已归档清理）。

## 文档索引

| 序号 | 文档 | 适用读者与核心内容 |
| :---: | :--- | :--- |
| 1 | [`../README.md`](../README.md) | 项目总览、核心特性摘要、一键构建命令与 5 项标准回归入口 |
| 2 | [`../PROJECT.md`](../PROJECT.md) | 研究背景、双机硬件环境、精简目录树、HAL 模块清单与硬件安全规范 |
| 3 | [`SM120_V2_ARCHITECTURE_ZH.md`](SM120_V2_ARCHITECTURE_ZH.md) | **核心架构设计（`v2.2.0`）**：双模式极简设计、静态 9-Cave 代码洞布局（`1,045 / 1,390` 槽位）、100% GDDR7 显存控制面 + 高优 Copy Engine DMA + GPU 原生 `sm120_cond_clear_hdr`、`0x500` B 永久 Entry Guardian（`is_restore_wave` 统一路由 + 五区间无死锁协同）、自适应内核/模式路由器（`SYNCS`/Cluster/Coop/Graph）、`XMemFree_v2` 延迟释放队列与 XSched HAL 集成 |
| 4 | [`SM120_V2_VERIFICATION_AND_BENCHMARK_ZH.md`](SM120_V2_VERIFICATION_AND_BENCHMARK_ZH.md) | **实机验证与性能评测**：`C++20 -Werror` 零警告构建、`nvdisasm -b SM120` 79/79 编码核验、RTX 5080 & RTX 5060 代码洞探测、11/11 端到端正确性与 100 轮双连压测、Mode 1 vs Mode 5 延迟与显存缩放基准、Context CILP 回归及关键微架构根因复盘 |
| 5 | [`SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md`](SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md) | **全方位能力评测与 `v2.1.0 → v2.2.0` 优化前后对比**：五维评测方法论、8 大闭源 CUDA 库 + 7 项 Blackwell 硬件特性（15/15 PASS）、真实未修改 PyTorch 2.11 + cuDNN 9.19 + FlashAttention + Triton 3.6.0（11/11 PASS，含 ResNet-18 训练与 `empty_cache` 零死锁）、DNN/LLM/100% SM 饱和 QoS（`57–70 µs` 抢占 / `100%` SLO≤2ms）与 BAR0 跨上下文 0 干扰验证 |
| 6 | [`SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md`](SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md) | **阶段总汇报、问题闭环台账与未来规划**：`v2.2.0` 全部 Phase 1–2 成果汇总、`DEF-01..08` + 并发竞态全部解决验证闭环、剩余 2 项架构级 P2 项（`DEF-09` 图内断点、`DEF-10` 非 root BAR0 守护进程）、Phase 3 演进与论文规划及 5080 一键同步复跑手册 |
| 7 | [`SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md`](SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md) | **RTX 5060 vs RTX 5080 跨芯片/跨驱动差异对比**：`GB206 (595.99.02)` 与 `GB203 (580.95.05)` 的适配状态、ToolsMemory `+0x5900` 偏移漂移、SM 并发与显存缩放、延迟/桌面看门狗/GCC 工具链差异及跨卡可移植清单 |
| 8 | [`../CHANGELOG.md`](../CHANGELOG.md) | 版本演进记录（`v2.0.0-sm120` → `v2.1.0-sm120` → `v2.2.0-sm120`） |
| 9 | [`../xsched/docs/xsched-osdi25.pdf`](../xsched/docs/xsched-osdi25.pdf) | XSched 原始 OSDI'25 学术论文 |
