# Blackwell (`sm_120`) Level-3 v2 技术文档中心

本目录仅保留与当前 **Blackwell SM120 v2** 代码基线完全同步的权威技术文档（历史探索草稿与 v1 过渡文档已归档清理）。

## 文档索引

| 序号 | 文档 | 适用读者与核心内容 |
| :---: | :--- | :--- |
| 1 | [`../README.md`](../README.md) | 项目总览、核心特性摘要、一键构建命令与 5 项标准回归入口 |
| 2 | [`../PROJECT.md`](../PROJECT.md) | 研究背景、双机硬件环境、精简目录树、HAL 模块清单与硬件安全规范 |
| 3 | [`SM120_V2_ARCHITECTURE_ZH.md`](SM120_V2_ARCHITECTURE_ZH.md) | **核心架构设计**：双模式极简设计、静态 9-Cave 代码洞布局（`1,045 / 1,390` 槽位）、`0x500` B 永久 Entry Guardian（`is_restore_wave` 统一路由 + 五区间无死锁协同）、用户态 `BPT.TRAP 0x1` 0 脉冲自陷恢复、紧凑 `slot_id` 池化 `u_arena`、256 槽位 VRAM 双跳中转、Epoch 保护恢复清理及 XSched HAL 集成 |
| 4 | [`SM120_V2_VERIFICATION_AND_BENCHMARK_ZH.md`](SM120_V2_VERIFICATION_AND_BENCHMARK_ZH.md) | **实机验证与性能评测**：`C++20 -Werror` 零警告构建、`nvdisasm -b SM120` 79/79 编码核验、RTX 5080 & RTX 5060 代码洞探测、11/11 端到端正确性与 100 轮双连压测、Mode 1 vs Mode 5 延迟与显存缩放基准、Context CILP 回归及关键微架构根因复盘 |
| 5 | [`SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md`](SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md) | **全方位能力评测与缺陷全景分析**：五维评测方法论设计、8 大闭源 CUDA 计算库（`cuBLAS/cuBLASLt/cuFFT/cuSPARSE/CUB`）双连抢占实测、Blackwell 硬件特性边界矩阵（`mma.sync`/`cp.async`/`mbarrier`/Cluster/Coop/Graph）、真实未修改 PyTorch 2.11 + cuDNN 9.19 + FlashAttention + Triton 3.6.0 验证、DNN/LLM 多模式 QoS 与频率扫描、BAR0 跨上下文干扰及 6 项已修复 / 10 项现存缺陷深度剖析 |
| 6 | [`SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md`](SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md) | **阶段总汇报、问题台账与未来规划**：面向课题组/合作方的当前状态总览、全部阶段性成果汇总、双机同步核对、10 项现存问题（P0/P1/P2）精确代码定位与修复方案、Phase 1–3 演进与论文发表路线图及 5080 一键同步复跑手册 |
| 7 | [`SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md`](SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md) | **RTX 5060 vs RTX 5080 跨芯片/跨驱动差异对比**：`GB206 (595.99.02)` 与 `GB203 (580.95.05)` 的适配状态、ToolsMemory `+0x5900` 偏移漂移、SM 并发与显存缩放、延迟/桌面看门狗/GCC 工具链差异及跨卡可移植清单 |
| 8 | [`../CHANGELOG.md`](../CHANGELOG.md) | 版本演进记录（`v2.0.0-sm120` → `v2.1.0-sm120`） |
| 9 | [`../xsched/docs/xsched-osdi25.pdf`](../xsched/docs/xsched-osdi25.pdf) | XSched 原始 OSDI'25 学术论文 |
