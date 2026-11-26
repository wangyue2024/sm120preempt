# Phase 3.5 证据（2026-11-26 / 27）

报告：[`../../SM120_V2_PHASE3_5_TIER1_ZH.md`](../../SM120_V2_PHASE3_5_TIER1_ZH.md)。所有文件已脱敏（主机名、用户名、私有路径）；日志去掉了逐内核插桩行与进度条。

## Level-3 价值实验（§2）

| 文件 | 内容 |
|---|---|
| `value_5060.jsonl` / `value_5080.jsonl` | 4 类 LP × 5–6 个规模 × native / lv1 / lv1-sync / lv3 / lv3-strict × 小 / 大 HP，每卡 210 组（`lv3` 为 DEF-30 修复后的默认配置） |
| `value_5060_lv3_waveonly.jsonl` / `value_5080_lv3_waveonly.jsonl` | 修复前（只按波数路由）的 `lv3` 行（归档） |
| `value_table.md` / `value_p99.png` | 完整表格与 HP P99 对数坐标图 |
| `compact_5060.md` / `compact_5080.md` | 报告中的精简表 |
| `value_lv3_strict_sweep_5080_log.txt` | 5080 `lv3-strict` 扫描日志 |
| `lv3_tail_ab_5060.jsonl` | DEF-30 定位：默认 / 关闭路由 / 加大间隔的 A/B |
| `lv3_tail_after_def30_5060.jsonl` | 修复后的尾延迟（路由统计） |
| `router_ab_ops_5060_{native,old,new,off}.jsonl` | 路由目标算子：native / 只按波数 / 按时长 / 关闭路由 |

## CUDA Graph 图内抢占（§3）

| 文件 | 内容 |
|---|---|
| `graph_preempt_flatten_ab_5060.jsonl` | 整网捕获训练步：展平 / 不展平 |
| `graph_relaunch_event_bridge_5080_log.txt` | 5080：DEF-31 复现用例、跨流等待测试、图抢占测试 |

## vLLM（§4）

| 文件 | 内容 |
|---|---|
| `vllm_final_matrix_5080.jsonl` / `vllm_table.md` | 最终矩阵（0.5B / 7B-AWQ decode / 7B-AWQ prefill / 3B prefill × native / Level-1 / Level-3 不展平 / 展平） |
| `vllm_first_lv3_blocking_wait_5080_log.txt`、`vllm_05b_matrix_blocking_wait_5080_log.txt`、`vllm_05b_blocking_wait_5080.jsonl` | DEF-29 修复前（阻塞等待）的首轮结果 |
| `vllm_xqueue_knobs_5080*.{txt,jsonl}` | 在飞阈值 / 自旋 / 直接提交对吞吐无影响 |
| `vllm_cprofile_*` | cProfile：native `Event.synchronize` 0.040 s vs Level-1 `Event.wait` 0.231 s（128 步） |
| `vllm_flag_bridge_deadlock_5080_log.txt`、`vllm_flag_bridge_deadlock_gdb_threads_5080.txt` | 被否决的“GPU 等主机标志”设计：启动阶段死锁与线程栈 |
| `vllm_promotion_matrix_stream_bug_5080_log.txt` | 提升方案首轮；0.5B 不展平 0/32（DEF-32 暴露）、7B 绕过 XQueue（OBS-18） |
| `vllm_matrix_def31_only_5080_log.txt` | 修复 OBS-18 与 DEF-31 后，不展平仍 0/32 |
| `vllm_noflat_diag_trace_def32_5080_log.txt`、`vllm_def32_reactivate_trace_excerpt_5080.txt` | 阻塞等待 / 关闭路由 / 跟踪；恢复下标 `active_idx=4231`、`drained=35832` |
| `vllm_def32_capture_clear_rejected_5080_log.txt` | “捕获开始时清零”仍 0/32（假设被否定） |
| `vllm_def32_fixed_5080_log.txt`、`vllm_def32_ab_5080.jsonl` | 整图发射前清零后 32/32（0.5B ×3、7B、跟踪） |
| `post_regression_vllm_final_value_lib12_5080_log.txt` | 最终矩阵、5080 价值扫描 `lv3`、LIB-12 ×20 的运行日志 |

## 收尾（§5）

| 文件 | 内容 |
|---|---|
| `def33_lib12_bisect_5080_log.txt` | LIB-12：关闭各新功能仍失败 |
| `def33_lib12_head_libs_5080_log.txt`、`def33_lib12_head_binary_5080_log.txt` | 提交版本 f29aab3 的库 / 测试程序同样失败 |
| `def33_cufft_native_shim_5080_log.txt` | 纯驱动与 shim 透传的 cuFFT 正常 |
| `def33_lib12_gdb_backtrace_5080_log.txt` | 中止栈：静态析构 → `RestoreOriginal` → `cuXtraMemcpyHtoD` |
| `def33_lib12_fixed_5080_log.txt` | 修复后 0/20、0/10，Suite 4 全过 |
| `obs20_ab_probe_*_log.txt` | Level-3 进程退出后立即启动的确定性探针（纯驱动 / 经 shim），0/54 |
| `regress_full_final_*` | 最终构建两卡全量回归 |
