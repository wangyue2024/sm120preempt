# Blackwell (`sm_120`) Level-3 v2 技术文档中心

本目录仅保留与当前 **Blackwell SM120 v2** 代码基线完全同步的权威技术文档（历史探索草稿与 v1 过渡文档已归档清理）。

## 文档索引

| 序号 | 文档 | 适用读者与核心内容 |
| :---: | :--- | :--- |
| **★** | **[`Blackwell_SM120_Level3_Preemption_Technical_Manual_v2.2.1_ZH.docx`](Blackwell_SM120_Level3_Preemption_Technical_Manual_v2.2.1_ZH.docx)** | **【推荐·完整排版 Word 权威版（39 页 A4；2026-11-26 新增第 11 章：Phase 3 勘误、缺陷修复与 3.2/3.3 结果）】** 全景 10 章完全技术手册（含封面、导航大纲、27 张双卡数据与 ABI 表格、架构/时序图、PTX 与代码洞剖析）：覆盖总体架构、ToolsMemory 11 代码洞、SM120 128-bit SASS 编码与 79 指令、80 槽位 Entry Guardian 逐槽逻辑、静态 9-Cave（1,045/1,390 槽）指令流、GDDR7 控制面/DMA/`u_arena` 数据承载与 0 脉冲恢复、BAR0 D10 与自适应路由/Shim 加固、15 项微架构发现、RTX 5080 & 5060 全量实测及 Phase 3 规划 |
| **★** | **[`SM120_V2_COMPLETE_TECHNICAL_MANUAL_ZH.md`](SM120_V2_COMPLETE_TECHNICAL_MANUAL_ZH.md)** | **【完整技术手册 Markdown 权威源文（`v2.2.1-sm120`）】** 与上述 `.docx` 完全一一对应的 10 章完整技术文档，便于在 GitHub / IDE 中直接在线查阅、检索与版本 Diff |
| **★ 最新** | **[`SM120_V2_PHASE3_4_MULTIPROCESS_ZH.md`](SM120_V2_PHASE3_4_MULTIPROCESS_ZH.md)** | **【Phase 3.4-2：HP / LP 多进程服务测试与跨进程调度（2026-11-26）】** 6 种模式两卡结果（按请求挂起 TPOT 1.00×，证实 OBS-7）；DEF-26 / 28 修复、DEF-27（Xid 109）缓解、DEF-25 撤回、OBS-8（显示 GPU 上 Xid 31） |
| **★ 新** | **[`SM120_V2_PHASE3_4_DEF10_BAR0_PROXY_ZH.md`](SM120_V2_PHASE3_4_DEF10_BAR0_PROXY_ZH.md)** | **【Phase 3.4-1：DEF-10 非 root Level-3 抢占（2026-11-26）】** `xsched-bar0d` 脉冲代理的设计、安全模型（两页 BAR0、降权 `nobody`、组授权、限流、防接错 GPU）、非 root 两卡验证（37/37、50/50）、故障注入与开销（+2–15 µs） |
| **★ 新** | **[`SM120_V2_PHASE3_2_3_3_REPORT_ZH.md`](SM120_V2_PHASE3_2_3_3_REPORT_ZH.md)** | **【Phase 3.2 / 3.3 正确性收尾与性能优化报告（2026-11-26）】** DEF-21 关闭（测试竞态 T-11 + 约 3,000 次压测抢占）、DEF-19 缓解、DEF-22/23/24 修复；DEF-17：Guardian 去冗余重读、实验 v2 的 LLM-08 二分（11/61 vs 0/78）、短 CTA 路由（SASS 回边分析）；每调用开销：直接提交、调用方 `WaitAll`、布局缓存；调度策略与 `1 − HP 占空比` 上限分析、低负载验证；一键回归流水线（含 3.2 验收组）与两卡全量回归；Suite 7/8 最终数据。证据见 [`evidence/phase3_2_3_3_20261125/`](evidence/phase3_2_3_3_20261125/)（Guardian A/B、LLM-08 二分）与 [`evidence/phase3_2_3_3_20261126/`](evidence/phase3_2_3_3_20261126/)（最终回归、短 CTA 路由、每调用开销、PERF-2、低负载服务） |
| **★ 新·先读** | **[`SM120_V2_PHASE3_SUMMARY_AND_ROADMAP_ZH.md`](SM120_V2_PHASE3_SUMMARY_AND_ROADMAP_ZH.md)** | **【阶段成果总结与下一步规划（2026-11-25）】** 当前状态一句话、技术基线、Phase 3 测试体系（Suite 1–8 + 合成测试 + 复现工具）、修复后两卡关键结果、已修复/开放缺陷、交付物清单、Phase 3.2–3.5 规划（含验收标准）与风险 |
| **★ 新** | **[`SM120_V2_PHASE3_DEFECT_LEDGER_ZH.md`](SM120_V2_PHASE3_DEFECT_LEDGER_ZH.md)** | **【Phase 3 缺陷台账与调查全记录（2026-11-25）】** DEF-20 根因（Trap 引擎漏存最高 1–3 个寄存器，`SR_REGALLOC` 实测等于寄存器数）及其修复验证；DEF-14..22 逐项记录（现象/范围/触发/复现/证据/根因/修复/验证/剩余风险/难度）；OBS-1..12 观察；测试方法自身问题 T-1..10；环境问题 E-1..7；调查时间线；工作区改动清单。原始证据见 [`evidence/phase3_20261125/`](evidence/phase3_20261125/) |
| **★ 新** | **[`SM120_V2_PHASE3_PRODUCTION_OPS_GROUNDED_TESTS_ZH.md`](SM120_V2_PHASE3_PRODUCTION_OPS_GROUNDED_TESTS_ZH.md)** | **【Phase 3 生产算子性能与真实负载接地测试（2026-11-25）】** 遗留 Suite 1/4/5/6 在 RTX 5080 补测；Suite 7（29 个生产算子 × native/shim/Lv1/Lv3 × 双卡：稳态开销、Lv3 覆盖率、挂起延迟、逐调用比特级正确性、原生高优先级流基线）；Suite 8（真实 Qwen2.5-0.5B 权重在线服务 + 离线评测共置：TTFT/TPOT/LP 吞吐）；新增缺陷台账 DEF-14..19（cast / CUB / topk 恢复崩溃、Triton RMSNorm 静默错误、Entry Guardian 每 CTA 开销、DEF-12 改动的回归风险 p=0.019）与诊断工具 |
| 1 | [`../README.md`](../README.md) | 项目总览、核心特性摘要、一键构建命令与 5 项标准回归入口 |
| 2 | [`../PROJECT.md`](../PROJECT.md) | 研究背景、双机硬件环境、精简目录树、HAL 模块清单与硬件安全规范 |
| 3 | [`SM120_V2_ARCHITECTURE_ZH.md`](SM120_V2_ARCHITECTURE_ZH.md) | **核心架构专题（`v2.2.0/v2.2.1`）**：双模式极简设计、静态 9-Cave 代码洞布局（`1,045 / 1,390` 槽位）、100% GDDR7 显存控制面 + 高优 Copy Engine DMA + GPU 原生 `sm120_cond_clear_hdr`、`0x500` B 永久 Entry Guardian（`is_restore_wave` 统一路由 + 五区间无死锁协同）、自适应内核/模式路由器（`SYNCS`/Cluster/Coop/Graph）、`XMemFree_v2` 延迟释放队列与 XSched HAL 集成 |
| 4 | [`SM120_V2_VERIFICATION_AND_BENCHMARK_ZH.md`](SM120_V2_VERIFICATION_AND_BENCHMARK_ZH.md) | **实机验证与微基准评测（已同步 `v2.2.1`）**：`C++20 -Werror` 零警告构建、`nvdisasm -b SM120` 79/79 编码核验、RTX 5080 & RTX 5060 代码洞探测、11/11 端到端正确性与 100 轮双连压测、Mode 1 vs Mode 5 延迟与显存缩放基准、Context CILP 回归及关键微架构根因复盘 |
| 5 | [`SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md`](SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md) | **全方位能力评测与 `v2.1.0 → v2.2.1` 优化前后对比**：五维评测方法论、8 大闭源 CUDA 库 + 7 项 Blackwell 硬件特性（双卡 15/15 PASS）、真实未修改 PyTorch 2.11 + cuDNN 9.19 + FlashAttention + Triton 3.6.0（双卡 11/11 PASS）、DNN/LLM/100% SM 饱和 QoS（`32–70 µs` 抢占 / `100%` SLO≤2ms）与 BAR0 跨上下文 0 干扰验证 |
| 6 | [`SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md`](SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md) | **阶段总汇报、问题闭环台账与未来规划**：`v2.2.1` 全部 Phase 1–2 成果汇总、`DEF-01..08` + 并发竞态全部解决验证闭环、剩余 2 项架构级 P2 项（`DEF-09` 图内断点、`DEF-10` 非 root BAR0 守护进程）、Phase 3 演进与论文规划及双机同步复跑手册 |
| 7 | [`SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md`](SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md) | **RTX 5060 vs RTX 5080 跨芯片/跨驱动差异对比**：`GB206 (595.99.02)` 与 `GB203 (580.95.05)` 的双卡闭环适配状态、ToolsMemory `+0x5900` 偏移漂移、SM 并发与显存缩放、延迟/桌面看门狗/GCC 工具链差异及跨卡可移植清单 |
| 8 | [`../CHANGELOG.md`](../CHANGELOG.md) | 版本演进记录（`v2.0.0-sm120` → `v2.1.0-sm120` → `v2.2.0-sm120` → `v2.2.1-sm120`） |
| 9 | [`../xsched/docs/xsched-osdi25.pdf`](../xsched/docs/xsched-osdi25.pdf) | XSched 原始 OSDI'25 学术论文 |
