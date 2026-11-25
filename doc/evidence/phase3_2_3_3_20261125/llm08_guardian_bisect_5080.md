# LLM-08 二分记录（RTX 5080，2026-11-25）

用例：`test_real_llm_prod_p3.py llm08`，同一进程内两条 Lv3 LP 队列 + 一条高优先级 HP 流，每次运行是一个独立进程（失败表现为 `e2=nan` 或 `e2` 远大于容差）。
全部运行经 `p3_guarded_run.sh`（每条前检查 GPU 上无其他用户进程）。

触发：Phase 3.2 首轮全量回归（Guardian v2 为默认）中 `S6-llm08` 失败 1/1：
`p1=2 susp1=[0.044, 0.171] | p2=1 susp2=[0.236] | e1=0.000e+00 e2=nan`。

## 第 1 轮（每种配置 6 个进程）

| 配置 | 环境变量 | 失败 |
|---|---|---:|
| cur（v2 默认） | — | 1/6 |
| gv1 | `XSCHED_SM120_GUARDIAN_V1=1` | 0/6 |
| spin0 | `XSCHED_SPIN_US=0` | 1/6 |
| noabs | `XSCHED_SM120_DETECT_CALL_ABS=0` | 3/6 |
| cap4k | `XSCHED_SM120_LMEM_CAP_BYTES=4096` | 0/6 |
| alloff | 以上四项全部 | 0/6 |
| prevbuild | 上一轮已提交构建（DEF-20 修复 + v1 Guardian） | 0/6 |

## 第 2 轮（每种 12 个进程，交错运行）

| 配置 | 失败 |
|---|---:|
| cur（v2） | 2/12 |
| gv1 | 0/12 |
| cap4k（v2） | 1/12 |
| gv1 + cap4k | 0/12 |

结论：与 Guardian 版本相关，与 LMEM 上限 / 自旋 / `CALL.ABS` 检测无关。

## 第 3 轮：v2 子特性（每种 12 个进程，交错运行）

| 配置 | 失败 |
|---|---:|
| v2 全部 | 2/12 |
| v2 + 恢复 slot 28 重读（`_V2_RELOAD`） | 1/12 |
| v2 + 串行控制头读取（`_V2_SERIAL_HDR`） | 0/12 |
| v2 + 串行 `S2R/LDC`（`_V2_SERIAL_SREG`） | 0/12 |

结论：失败需要异步发射窗口；去掉重读本身不是原因。

## 第 4 轮：新默认（串行发射 + 去重读）

| 配置 | 失败 |
|---|---:|
| 新默认 | 0/24 |
| 原始 v1（`XSCHED_SM120_GUARDIAN_V1=1`） | 0/12 |

## 合计

| 类别 | 失败 |
|---|---:|
| 含 v2 全部异步特性的运行（第 1 轮 cur/spin0/noabs/cap4k + 第 2 轮 cur/cap4k + 第 3 轮 v2 + 首轮回归） | **11/61** |
| 串行发射（v1 各组合 54 + 新默认 24） | **0/78** |

新默认随后在两卡全量回归中通过 184/184（含 `S6-llm08`），见 `regress_full_5080_summary.md`。
