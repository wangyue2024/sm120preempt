# SM120 v2 Phase 3：生产算子性能与真实负载接地测试报告（RTX 5080 + RTX 5060）

> **测试日期**：2026-11-25　|　**代码基线**：`v2.2.1-sm120`（commit `c44da4f`）+ 工作区未提交的 `DEF-11/12/13` 修复 + 本轮新增诊断代码
> **硬件**：RTX 5080（`GB203`，84 SM，16 GiB，驱动 `580.95.05`，无头，**多用户共享服务器**）/ RTX 5060（`GB206`，30 SM，8 GiB，驱动 `595.99.02`，Wayland 桌面卡）
> **软件**：CUDA 12.9 / PyTorch `2.11.0+cu128` / cuDNN 9.19 / Triton 3.6.0 / transformers 5.24 / peft 0.22
> **原则**：所有数字均来自本轮两台实机的原始日志（`jsonl` 由脚本自动汇总生成表格，未手工改写）；失败、崩溃与不确定结论如实列出。

> **⚠ 2026-11-25 后续更新（以 [`SM120_V2_PHASE3_DEFECT_LEDGER_ZH.md`](SM120_V2_PHASE3_DEFECT_LEDGER_ZH.md) 为准）**：
> DEF-14 / DEF-15 / DEF-16 以及长期“偶发”的 LIB-08、LLM-07 m5 的**共同根因已确认**为 **DEF-20：Trap 引擎从不保存每个内核最高的 1–3 个通用寄存器**（`SR_REGALLOC` 实测等于寄存器数，原实现按 `−3` 计算；奇数寄存器时 64 位寄存器对越界）。
> 修复（`SR_REGALLOC−1` + 插桩寄存器数取偶，共 3 行，**未提交**）后：Suite 7 Lv3 **两卡 × 两种构建均 29/29 比特一致**，5080 上 LIB-08 60/60、LLM-07 m5 10/10、Suite 1/4/5/6 全过（WT 构建）。
> DEF-18 结论已修订（DEF-12 改动应保留）；新增 DEF-21（恢复波次再抢占）、DEF-22（`CALL.ABS` 漏检）；DEF-19（Cluster+DSMEM）仍开放。下文 §0–§5 保留修复前的原始测量与判断，供对照。

---

## 0. 结论摘要

1. **遗留 Phase 3 套件在 RTX 5080 上补测完成**：Suite 4（6 个闭源库）6/6、Suite 5（8 个真实 DNN 场景）8/8、Suite 6 LLM 场景与真实预训练权重 `LLM-09` 均比特级通过；但 **Suite 1 `HW-06`、`LIB-08`（NP=2）与 `LLM-07 m5` 存在偶发失败**（见 §5）。
2. **新增 Suite 7「生产算子」基准（29 个算子 × 4 种模式 × 2 张卡，每个 (算子, 模式) 独立进程）**：
   - 稳态开销：Shim 拦截 ≈ 0%，Level-1 ≈ 0.1%，**Level-3 中位数 +1.7%（5080）/ +0.6%（5060）**；但对 **CTA 数量极多、单 CTA 工作量极小** 的访存型算子开销显著（5080：`rope_apply` **+101%**、`transpose_copy` **+82%**、`embedding` +23%、`gelu` +18%），Entry Guardian 成本按 CTA 数线性累积，而非此前文档所述的“每内核固定 ~1 µs”。
   - Level-3 覆盖率：所有能跑通的算子 **100% 内核被插桩为 Lv3 可抢占**（无内核被路由回 Lv1）。
   - 抢占延迟：Lv3 挂起 P50 中位数 **0.27 ms**（5080，范围 0.06–1.22 ms），Lv1 为 **1.98 ms**（0.21–7.23 ms），Lv3 较 Lv1 快约 7×（5060 上约 17×）。
   - 正确性：Lv3 下 **24/29（5080）、23/29（5060）个算子在 15 次中途抢占下逐调用比特级一致**；其余算子出现 **崩溃（cuda error 700/717）或静默数据错误**（§5 DEF-14/15/16）。
   - **诚实的基线对比**：在单算子微基准中，仅使用 **原生高优先级 CUDA Stream（无调度器）** 时 HP 探针延迟中位数 P50/P99 = 0.11/0.42 ms（5080），**反而优于 Lv3 的 0.33/0.58 ms**——因为这些 LP 算子由大量短 CTA 组成，高优先级流可在 CTA 粒度插队。此前评测仅与 Lv1 对比，未覆盖该基线。
3. **新增 Suite 8「真实模型服务接地测试」**（真实 `Qwen2.5-0.5B-Instruct` 权重、真实聊天 Prompt、Poisson 到达、贪心解码 + KV Cache；LP 为同一权重上的离线困惑度评测 / Prefill 批处理）：**两卡 × 2 种 LP × 4 种模式共 16 组全部比特级一致（HP 生成 token 20/20 完全一致，LP 0 次错误迭代）**；5080 上 Lv3 将 HP 的 TTFT/TPOT P50 相对“无调度共置”分别降低 20%/18%，P99 尾延迟最紧，但 LP 吞吐降至独占的 ~52%（Lv1 ~74%，无调度 ~98%）。
4. **本轮定位的缺陷**（全部有可复现命令与证据，详见 §5）：
   - **DEF-14（P0，崩溃）**：PyTorch 任意 dtype 转换核 `unrolled_elementwise_kernel<direct_copy_kernel_cuda>`（如 `bool→int64`）在 **首次 State-B 驱逐恢复** 时确定性崩溃（700/717），双卡复现，最小复现脚本 < 40 行。
   - **DEF-15（P0，崩溃）**：CUB Onesweep 基数排序（`torch.sort`）、CUB 扫描（整型 `cumsum`）、`torch.topk`（`mbtopk`）在 Lv3 抢占下崩溃；`LIB-08` 长期存在的“P2 偶发 700”即属此类。
   - **DEF-16（P0，静默错误）**：Triton 融合 RMSNorm（`tl.sum` 跨 Warp 归约）在 Lv3 抢占下 **不崩溃但输出错误**（每 5 轮仅 1–2 轮正确），是最危险的一类。
   - **DEF-17（P1，性能）**：Entry Guardian 每 CTA 开销导致短 CTA 访存算子最高 +101%。
   - **DEF-18（P1，回归风险）**：工作区未提交的 `DEF-12` 修复使 `LIB-08`（NP=2）失败率由 **15%（HEAD，6/40）升至 38%（工作区，19/50）**，Fisher 精确检验 **p = 0.019**——**该改动在根因确认前不应提交**。
   - `HW-06`（Cluster+DSMEM）与 `LLM-07 m5` 的偶发失败在 HEAD 与工作区上失败率相同（各 1/3），属 **既有问题而非回归**。
5. **新增诊断能力**：内核级启动追踪、驱逐 CTA 的 Trap PC 反查（精确到原始 SASS 偏移）、原始 SASS 导出、自适应路由统计导出 API、共享服务器安全守护运行器，以及针对部分退出/发散 Warp 的合成测试（已排除 7 类根因假设）。

---

## 1. 测试环境、安全规范与方法

### 1.1 共享 GPU 安全规范（RTX 5080）

- 5080 为多用户共享服务器（本轮期间有其他用户登录）。所有 GPU 测试通过新增的守护运行器 `p3_guarded_run.sh` 执行：**每一个测试进程启动前**都检查 `nvidia-smi --query-compute-apps`，若有他人计算进程则等待（最多 600 s）后跳过并记录 `SKIPPED_GPU_BUSY`，**绝不在他人作业驻留时触发 BAR0 脉冲**。本轮全部计划执行期间未检测到他人 GPU 作业（0 次等待）。
- Level-3 需要 root 映射 BAR0。本轮改为 **经 ssh 标准输入把 sudo 口令传给 `sudo -S`**，口令不再出现在远端进程命令行（其他用户可通过 `/proc/*/cmdline` 读取），修正了之前会话的口令暴露隐患。
- 5060 为单用户本地桌面卡，桌面进程（gnome-shell 等）常驻，使用 `SKIP_GUARD=1`。

### 1.2 Suite 7：生产算子基准方法（`xsched/test/sm120_v2/eval/bench_prod_ops_p3.py`）

| 项 | 设计 |
|---|---|
| 隔离 | **每个 (算子, 模式) 一个独立进程**：单个算子的致命 CUDA 错误不会污染其他测量；`native` 模式严格不加载 XSched（进程内校验 `libshimcuda` 未映射） |
| 模式 | `native`（无 XSched）/ `shim`（仅 Driver API 拦截，无 XQueue）/ `lv1`（XQueue Level-1）/ `lv3`（SM120 Trap 引擎 Mode 5 + 自适应路由） |
| 稳态延迟 | 批量背靠背调用的每调用墙钟时间（中位数，含主机侧提交路径）+ 单次调用（提交 1 次 + 同步，反映解码类小算子的启动开销） |
| 路由覆盖 | 新增导出 `XSm120GetRouteStats`：统计每调用内核被插桩为 Lv3 可抢占 / 因 mbarrier、Cooperative、多波 Cluster、CUDA Graph、插桩失败被路由到 Lv1 的数量 |
| 抢占测试 | LP 作业 = 该算子重复至 ~60 ms；在 20%/45%/70% 处挂起 → 运行 HP 探针（512×1024×1024 BF16 GEMM，高优先级流）→ 恢复；5 轮 × 3 次 = 15 次中途抢占 |
| 正确性 | **非原地算子：每一次调用的输出都在 GPU 上与 Golden 逐元素比较**（无类型转换的 `ne().any()` + `logical_or` 累积，避免把被测的 cast 核注入校验路径）；**原地非幂等算子（Fused AdamW、BN 运行统计、INT32 原子 `index_add_`、梯度累加）比较整条作业后的最终状态**；同时记录 Golden SHA-256 摘要用于跨模式/跨进程比对 |
| 共置基线 | `native`/`shim` 模式下同一时间点把 HP 探针直接投到高优先级流（不挂起 LP），即“仅靠硬件流优先级”的基线 |

### 1.3 Suite 8：真实模型服务方法（`xsched/test/sm120_v2/eval/bench_real_serving_p3.py`）

- 模型：**未修改的 HuggingFace `Qwen2.5-0.5B-Instruct` 真实预训练权重**（离线加载），SDPA 注意力，BF16。
- HP（在线）：20 个真实聊天请求（chat template），Poisson 到达（均值间隔 500 ms），手写贪心解码循环 + KV Cache（逐 token 同步以精确测量 TTFT 与每 token TPOT），生成 16 token；**每个请求的 token 序列必须与独占运行的 Golden 完全一致**。
- LP（离线，同一份权重）：`ppl` = 真实困惑度评测（`labels=` 触发 HF 把 logits 上转 FP32 并计算交叉熵，2×512 token）；`prefill` = 仅保留最后位置 logits 的批量 Prefill（4×1024 token）。**每次迭代的 loss/logits 与 Golden 比特级比较**。
- 模式：`solo`（只有 HP）/ `native`（共置、仅流优先级）/ `lv1` / `lv3`；`lv1/lv3` 采用 **请求粒度策略**：HP 请求到达即挂起 LP 队列，请求完成后恢复。
- 指标：TTFT、TPOT、请求端到端延迟（P50/P99）、挂起延迟、LP 吞吐占独占吞吐百分比、LP 对 HP 空闲时间的利用率。

---

## 2. 遗留 Phase 3 套件在 RTX 5080 上的补测（Level-3 / Mode 5）

| 套件 | 用例 | 结果 |
|---|---|---|
| Suite 1 | LIB-01..07、HW-01..05、HW-07 | 13/13 PASS（比特级） |
| Suite 1 | **HW-06 CTA Cluster 2×1×1 + DSMEM** | **偶发 FAIL**（`mismatch=256/7680`）：工作区 2/4、HEAD 2/3 通过（§5 DEF-19） |
| Suite 1 | **LIB-08 CUB DeviceRadixSort 8M，NP=2** | **工作区 31/50、HEAD 34/40 通过**（`cuda error 700`，§5 DEF-15/18）；NP=1 3/3 PASS |
| Suite 4 | LIB-09 cuSOLVER LU、LIB-10 Cholesky、LIB-11 cuRAND 有状态 PRNG、LIB-12 cuFFT 2D、LIB-13 CUB Look-Back 扫描、LIB-14 cuBLASLt FP8 | **6/6 PASS**（挂起 0.09–0.39 ms，`0.000e+00`） |
| Suite 5 | DNN-01..08（ViT-B/16、Swin-B+ConvNeXt、DeepLabV3+RoIAlign+NMS、AMP+梯度累积+Fused AdamW、20 步收敛 31 次抢占、`torch.compile` default / reduce-overhead、cuDNN BiLSTM + Diffusers UNet） | **8/8 PASS**（`0.000e+00`；DNN-06 `DEF-12` 场景 3/3 通过） |
| Suite 6 | LLM-01..05、LLM-08、LLM-06 native/lv1/m5、LLM-07 native/lv1 | **全部 PASS**（`0.000e+00`） |
| Suite 6 | **LLM-07 m5（LoRA 训练 + 在线解码）** | **偶发崩溃**（`cuda error 700`）：工作区 2/4、HEAD 2/3 通过（§5 DEF-15） |
| Suite 6 | **LLM-09 真实预训练 Qwen2.5-0.5B 生成（6 次抢占）** | **PASS**（`exact_match=True`） |

---

## 3. Suite 7：生产算子性能与抢占（自动生成表格）

> 表 A 中 Δ% 为相对 `native` 批量每调用延迟的变化；`kernels/call` 与覆盖率来自 `XSm120GetRouteStats`；`CRASH` 表示该进程因 CUDA 致命错误异常退出（无 JSON 结果）。
> 表 B 中 “co-run HP” 为 `native` 模式下 LP 持续运行、HP 探针仅靠高优先级流插队时的端到端延迟；Lv1/Lv3 的 HP 延迟包含挂起时间。

### NVIDIA GeForce RTX 5080 (84 SMs)

#### 表 A：稳态开销（无抢占）与 Level-3 路由覆盖

| op | category | native batch µs | shim Δ% | Lv1 Δ% | Lv3 Δ% | native single µs | Lv3 single µs | kernels/call | Lv3 coverage | routed→Lv1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gemm_bf16_4k` | gemm | 1275.2 | +0.0 | +0.6 | +1.4 | 1287.2 | 1307.9 | 1 | 100% | - |
| `gemm_bf16_decode` | gemm | 135.2 | -0.0 | +0.1 | +1.6 | 146.3 | 157.2 | 1 | 100% | - |
| `linear_bias_ffn` | gemm | 1657.7 | +0.0 | +0.0 | +0.7 | 1660.8 | 1680.6 | 1 | 100% | - |
| `gemm_fp8_e4m3` | gemm | 581.1 | -0.1 | +0.0 | +1.0 | 591.5 | 607.5 | 1 | 100% | - |
| `moe_bmm_8x` | gemm | 396.9 | +0.0 | +0.1 | +1.1 | 407.7 | 421.8 | 1 | 100% | - |
| `int8_mm` | gemm | 631.8 | +0.0 | +0.1 | +0.6 | 641.0 | 653.8 | 1 | 100% | - |
| `sdpa_prefill_causal` | attention | 412.0 | +0.0 | +0.1 | +1.7 | 422.9 | 438.1 | 1 | 100% | - |
| `sdpa_decode_kv4k` | attention | 823.1 | -0.1 | +0.1 | +1.0 | 832.7 | 848.9 | 1 | 100% | - |
| `rope_apply` | attention | 361.9 | +0.0 | +0.1 | +101.2 | 365.0 | 749.1 | 5 | 100% | - |
| `rmsnorm` | norm_act | 162.8 | -0.0 | +0.1 | +2.2 | 171.0 | 185.2 | 1 | 100% | - |
| `layernorm` | norm_act | 164.5 | +0.1 | +0.1 | +2.6 | 175.1 | 188.2 | 1 | 100% | - |
| `softmax` | norm_act | 347.9 | +0.0 | +0.1 | -0.0 | 357.0 | 364.7 | 1 | 100% | - |
| `swiglu` | norm_act | 1078.1 | +0.1 | +0.2 | +6.9 | 1086.1 | 1169.0 | 2 | 100% | - |
| `gelu_tanh` | norm_act | 441.4 | -0.1 | +0.0 | +18.3 | 446.0 | 539.5 | 1 | 100% | - |
| `cross_entropy` | norm_act | 1305.6 | +0.0 | +0.0 | +0.6 | 1316.1 | 1331.8 | 2 | 100% | - |
| `triton_rmsnorm` | norm_act | 162.7 | +0.2 | +0.1 | +1.7 | 174.5 | 187.9 | 1 | 100% | - |
| `conv3x3_nhwc` | conv | 1043.0 | -0.0 | +0.1 | +1.5 | 1048.9 | 1072.8 | 1 | 100% | - |
| `conv1x1_nchw_fp32` | conv | 558.3 | +0.0 | +0.0 | +4.3 | 567.7 | 600.2 | 1 | 100% | - |
| `batchnorm_train` | conv | 448.1 | -0.1 | -0.0 | +7.0 | 458.8 | 500.8 | 3 | 100% | - |
| `reduce_sum` | reduce_sort | 301.1 | -0.0 | +0.1 | +2.5 | 311.0 | 327.6 | 1 | 100% | - |
| `cumsum_64m` | reduce_sort | 2377.1 | +0.1 | -0.1 | CRASH | 2384.8 | - | - | - | - |
| `sort_16m` | reduce_sort | 2402.3 | -0.0 | +0.0 | CRASH | 2413.1 | - | - | - | - |
| `topk_sampling` | reduce_sort | 1064.2 | +0.0 | +0.1 | CRASH | 1074.3 | - | - | - | - |
| `embedding` | gather_scatter | 162.3 | +0.1 | +0.1 | +22.8 | 171.5 | 220.2 | 1 | 100% | - |
| `index_add_int32` | gather_scatter | 2755.0 | -0.0 | -0.1 | +0.2 | 2762.6 | 2775.6 | 2 | 100% | - |
| `cast_bool_to_i64` | gather_scatter | 414.9 | +0.0 | -0.3 | CRASH | 424.3 | - | - | - | - |
| `transpose_copy` | gather_scatter | 881.6 | +0.0 | +0.1 | +81.8 | 895.7 | 1621.0 | 1 | 100% | - |
| `adamw_fused_32m` | training | 1149.9 | -0.0 | +0.1 | +1.1 | 1174.9 | 1203.7 | 6 | 100% | - |
| `linear_bwd_accum` | training | 1462.0 | +0.0 | +0.1 | +3.5 | 1472.6 | 1538.8 | 4 | 100% | - |

#### 表 B：抢占（HP 探针延迟、挂起延迟、LP 正确性）

| op | LP job (reps) | HP solo ms | co-run HP P50/P99 ms (native) | Lv1 susp P50/max ms | Lv1 HP P50/P99 ms | Lv3 susp P50/max ms | Lv3 HP P50/P99 ms | Lv3 job infl. | bit-exact native/Lv1/Lv3 | golden digest match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `gemm_bf16_4k` | 63.8 ms (47) | 0.032 | 0.25 / 0.60 | 2.776 / 3.732 | 2.83 / 3.83 | 0.162 / 0.190 | 0.23 / 0.49 | +1.4% | ✅ / ✅ / ✅ | ✅ |
| `gemm_bf16_decode` | 65.5 ms (444) | 0.033 | 0.09 / 0.42 | 0.209 / 0.565 | 0.32 / 0.92 | 0.059 / 0.070 | 0.12 / 0.34 | +1.0% | ✅ / ✅ / ✅ | ✅ |
| `linear_bias_ffn` | 63.8 ms (36) | 0.032 | 0.35 / 0.52 | 4.884 / 6.552 | 4.95 / 6.66 | 0.205 / 0.232 | 0.26 / 0.53 | +1.5% | ✅ / ✅ / ✅ | ✅ |
| `gemm_fp8_e4m3` | 67.7 ms (103) | 0.032 | 0.09 / 0.41 | 1.433 / 2.437 | 1.56 / 2.49 | 0.161 / 0.175 | 0.22 / 0.47 | +1.4% | ✅ / ✅ / ✅ | ✅ |
| `moe_bmm_8x` | 65.4 ms (151) | 0.033 | 0.09 / 0.34 | 1.050 / 1.605 | 1.19 / 1.66 | 0.090 / 0.127 | 0.14 / 0.43 | +1.0% | ✅ / ✅ / ✅ | ✅ |
| `int8_mm` | 66.0 ms (95) | 0.033 | 0.15 / 0.38 | 1.849 / 2.645 | 1.94 / 2.94 | 0.122 / 0.191 | 0.24 / 0.41 | +1.2% | ✅ / ✅ / ✅ | ✅ |
| `sdpa_prefill_causal` | 65.7 ms (146) | 0.033 | 0.08 / 0.40 | 1.088 / 1.601 | 1.25 / 1.65 | 0.108 / 0.128 | 0.16 / 0.43 | +1.1% | ✅ / ✅ / ✅ | ✅ |
| `sdpa_decode_kv4k` | 60.9 ms (73) | 0.033 | 0.20 / 0.42 | 2.551 / 3.217 | 2.61 / 3.27 | 0.113 / 0.152 | 0.18 / 0.45 | +1.3% | ✅ / ✅ / ✅ | ✅ |
| `rope_apply` | 69.8 ms (166) | 0.033 | 0.07 / 0.50 | 0.488 / 0.783 | 0.54 / 0.92 | 0.403 / 0.786 | 0.48 / 0.91 | +2.7% | ✅ / ✅ / ✅ | ✅ |
| `rmsnorm` | 117.9 ms (369) | 0.032 | 0.13 / 1.12 | 0.842 / 1.098 | 0.90 / 1.28 | 0.294 / 0.540 | 0.38 / 0.66 | +0.9% | ✅ / ✅ / ✅ | ✅ |
| `layernorm` | 118.5 ms (365) | 0.033 | 0.11 / 0.89 | 0.658 / 1.011 | 0.82 / 1.07 | 0.284 / 0.550 | 0.35 / 0.66 | +1.1% | ✅ / ✅ / ✅ | ✅ |
| `softmax` | 136.3 ms (172) | 0.032 | 0.13 / 0.39 | 1.981 / 2.572 | 2.04 / 2.62 | 0.487 / 0.717 | 0.54 / 1.00 | +1.4% | ✅ / ✅ / ✅ | ✅ |
| `swiglu` | 93.8 ms (56) | 0.033 | 0.14 / 0.43 | 3.181 / 4.613 | 3.25 / 4.71 | 0.988 / 1.424 | 1.07 / 1.47 | +3.9% | ✅ / ✅ / ✅ | ✅ |
| `gelu_tanh` | 141.1 ms (136) | 0.033 | 0.14 / 0.44 | 2.648 / 3.336 | 2.73 / 3.39 | 0.670 / 1.310 | 0.73 / 1.58 | +2.0% | ✅ / ✅ / ✅ | ✅ |
| `cross_entropy` | 60.3 ms (46) | 0.032 | 0.09 / 0.40 | 3.574 / 4.955 | 3.65 / 5.03 | 0.256 / 0.345 | 0.31 / 0.60 | +2.0% | ✅ / ✅ / ✅ | ✅ |
| `triton_rmsnorm` | 117.2 ms (369) | 0.033 | 0.11 / 0.89 | 0.742 / 1.012 | 0.80 / 1.25 | 0.261 / 0.345 | 0.31 / 0.62 | +0.9% | ✅ / ✅ / ❌ | ✅ |
| `conv3x3_nhwc` | 68.2 ms (58) | 0.034 | 0.08 / 0.34 | 2.860 / 3.929 | 2.91 / 4.01 | 0.202 / 0.222 | 0.26 / 0.52 | +1.4% | ✅ / ✅ / ✅ | ✅ |
| `conv1x1_nchw_fp32` | 87.0 ms (107) | 0.033 | 0.07 / 0.33 | 2.136 / 2.615 | 2.19 / 2.69 | 0.206 / 0.369 | 0.27 / 0.43 | +1.1% | ✅ / ✅ / ✅ | ✅ |
| `batchnorm_train` | 60.0 ms (134) | 0.033 | 0.08 / 0.36 | 1.671 / 2.265 | 1.72 / 2.38 | 0.772 / 1.120 | 0.98 / 1.21 | +5.1% | ✅ / ✅ / ✅ | ❌ job_golden_digest={'native': 'bff5e69516343ae7', 'shim': 'bff5e69516343ae7', 'lv1': 'bff5e69516343ae7', 'lv3': '263c1ac43eefc469'} |
| `reduce_sum` | 61.7 ms (199) | 0.033 | 0.10 / 0.42 | 0.848 / 1.196 | 0.96 / 1.27 | 0.336 / 0.368 | 0.40 / 0.65 | +2.5% | ✅ / ✅ / ✅ | ✅ |
| `cumsum_64m` | 93.0 ms (25) | 0.032 | 0.12 / 0.37 | 7.234 / 8.332 | 7.32 / 8.39 | - / - | - / - | - | ✅ / ✅ / CRASH | ✅ |
| `sort_16m` | 63.8 ms (25) | 0.032 | 0.10 / 0.35 | 1.954 / 2.538 | 2.01 / 2.86 | - / - | - / - | - | ✅ / ✅ / CRASH | ✅ |
| `topk_sampling` | 60.1 ms (56) | 0.032 | 0.10 / 1.13 | 0.540 / 0.743 | 0.62 / 0.88 | - / - | - / - | - | ✅ / ✅ / CRASH | ✅ |
| `embedding` | 120.6 ms (370) | 0.033 | 0.10 / 0.78 | 0.750 / 0.991 | 0.82 / 1.12 | 0.305 / 0.446 | 0.39 / 0.56 | +0.9% | ✅ / ✅ / ✅ | ✅ |
| `index_add_int32` | 60.6 ms (22) | 0.034 | 1.45 / 1.66 | 5.462 / 42.754 | 5.51 / 43.02 | 0.183 / 0.293 | 0.24 / 0.53 | +1.6% | ✅ / ✅ / ✅ | ✅ |
| `cast_bool_to_i64` | 156.7 ms (145) | 0.033 | 0.29 / 0.36 | 2.663 / 3.722 | 2.72 / 3.77 | - / - | - / - | - | ✅ / ✅ / CRASH | ✅ |
| `transpose_copy` | 90.3 ms (68) | 0.034 | 0.10 / 0.42 | 3.068 / 3.870 | 3.16 / 4.23 | 1.216 / 1.816 | 1.28 / 1.88 | +6.3% | ✅ / ✅ / ✅ | ✅ |
| `adamw_fused_32m` | 59.8 ms (52) | 0.033 | 0.53 / 0.64 | 5.139 / 6.256 | 5.19 / 6.38 | 0.319 / 0.371 | 0.39 / 0.66 | +2.6% | ✅ / ✅ / ✅ | ✅ |
| `linear_bwd_accum` | 59.9 ms (41) | 0.033 | 0.18 / 0.30 | 6.021 / 7.631 | 6.08 / 7.68 | 0.480 / 0.586 | 0.54 / 0.68 | +3.1% | ✅ / ✅ / ✅ | ❌ job_golden_digest={'native': '5acd9f5118253de4', 'shim': '5acd9f5118253de4', 'lv1': '5acd9f5118253de4', 'lv3': '67a205e6ae670723'} |

#### 非比特一致 / 错误记录

- `triton_rmsnorm` [lv3]:  ['trial1: some LP call output != golden', 'trial2: some LP call output != golden', 'trial3: some LP call output != golden', 'trial4: some LP call output != golden']

### NVIDIA GeForce RTX 5060 (30 SMs)

#### 表 A：稳态开销（无抢占）与 Level-3 路由覆盖

| op | category | native batch µs | shim Δ% | Lv1 Δ% | Lv3 Δ% | native single µs | Lv3 single µs | kernels/call | Lv3 coverage | routed→Lv1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gemm_bf16_4k` | gemm | 3555.6 | -0.2 | -0.3 | -0.2 | 3566.0 | 3623.6 | 1 | 100% | - |
| `gemm_bf16_decode` | gemm | 318.0 | +1.1 | +0.3 | +6.3 | 323.2 | 352.3 | 2 | 100% | - |
| `linear_bias_ffn` | gemm | 4658.8 | +0.1 | +0.5 | +0.0 | 4627.9 | 4760.9 | 1 | 100% | - |
| `gemm_fp8_e4m3` | gemm | 1763.4 | +0.1 | +0.2 | +0.4 | 1708.9 | 1771.4 | 1 | 100% | - |
| `moe_bmm_8x` | gemm | 1226.1 | +0.5 | +1.4 | +1.3 | 1162.4 | 1197.8 | 1 | 100% | - |
| `int8_mm` | gemm | 1763.8 | -0.0 | -0.2 | +0.6 | 1706.2 | 1764.5 | 1 | 100% | - |
| `sdpa_prefill_causal` | attention | 1127.3 | +0.3 | +0.2 | +1.3 | 1090.2 | 1141.0 | 1 | 100% | - |
| `sdpa_decode_kv4k` | attention | 2026.3 | -0.4 | -1.0 | +1.1 | 1965.8 | 2031.9 | 1 | 100% | - |
| `rope_apply` | attention | 1135.1 | +2.5 | +0.7 | +19.4 | 1094.5 | 1332.2 | 5 | 100% | - |
| `rmsnorm` | norm_act | 370.4 | -2.0 | -2.7 | -2.3 | 520.9 | 378.3 | 1 | 100% | - |
| `layernorm` | norm_act | 365.8 | -1.0 | +0.0 | +0.5 | 360.6 | 380.2 | 1 | 100% | - |
| `softmax` | norm_act | 842.8 | +0.3 | -0.2 | +6.0 | 822.4 | 896.1 | 1 | 100% | - |
| `swiglu` | norm_act | 2409.7 | -0.6 | -0.1 | +0.4 | 2393.1 | 2359.9 | 2 | 100% | - |
| `gelu_tanh` | norm_act | 969.3 | +2.1 | -0.2 | -0.1 | 940.5 | 968.4 | 1 | 100% | - |
| `cross_entropy` | norm_act | 2942.2 | -0.1 | -0.8 | -1.4 | 2801.8 | 2803.5 | 2 | 100% | - |
| `triton_rmsnorm` | norm_act | 366.0 | +0.1 | +0.2 | +0.2 | 357.5 | 379.9 | 1 | 100% | - |
| `conv3x3_nhwc` | conv | 3221.0 | -0.5 | -2.1 | -0.3 | 3193.3 | 3114.9 | 1 | 100% | - |
| `conv1x1_nchw_fp32` | conv | 1668.4 | +0.0 | +0.5 | +3.9 | 1614.3 | 1717.3 | 1 | 100% | - |
| `batchnorm_train` | conv | 860.8 | +0.3 | +0.2 | +6.1 | 834.9 | 907.7 | 3 | 100% | - |
| `reduce_sum` | reduce_sort | 673.3 | +0.1 | -0.6 | +0.0 | 648.9 | 678.7 | 1 | 100% | - |
| `cumsum_64m` | reduce_sort | 5283.5 | +0.5 | -0.1 | CRASH | 5329.3 | - | - | - | - |
| `sort_16m` | reduce_sort | 5431.8 | +0.1 | +0.1 | CRASH | 5458.7 | - | - | - | - |
| `topk_sampling` | reduce_sort | 2551.5 | -0.7 | -1.0 | CRASH | 2485.9 | - | - | - | - |
| `embedding` | gather_scatter | 359.6 | +0.6 | +0.6 | +2.5 | 355.6 | 381.1 | 1 | 100% | - |
| `index_add_int32` | gather_scatter | 8269.1 | -0.3 | +0.9 | CRASH | 8310.6 | - | - | - | - |
| `cast_bool_to_i64` | gather_scatter | 913.9 | +0.8 | -0.2 | CRASH | 883.2 | - | - | - | - |
| `transpose_copy` | gather_scatter | 2405.8 | +0.8 | +1.2 | +29.4 | 2303.8 | 3099.4 | 1 | 100% | - |
| `adamw_fused_32m` | training | 2560.4 | +0.5 | -0.1 | +0.7 | 2499.0 | 2548.8 | 6 | 100% | - |
| `linear_bwd_accum` | training | 3970.3 | +0.4 | -0.6 | -0.0 | 3850.0 | 3887.7 | 4 | 100% | - |

#### 表 B：抢占（HP 探针延迟、挂起延迟、LP 正确性）

| op | LP job (reps) | HP solo ms | co-run HP P50/P99 ms (native) | Lv1 susp P50/max ms | Lv1 HP P50/P99 ms | Lv3 susp P50/max ms | Lv3 HP P50/P99 ms | Lv3 job infl. | bit-exact native/Lv1/Lv3 | golden digest match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `gemm_bf16_4k` | 63.5 ms (17) | 0.073 | 0.19 / 0.48 | 9.317 / 13.253 | 9.52 / 13.50 | 0.181 / 0.417 | 0.35 / 0.59 | +2.2% | ✅ / ✅ / ✅ | ✅ |
| `gemm_bf16_decode` | 62.1 ms (189) | 0.066 | 0.14 / 0.44 | 0.611 / 0.980 | 0.90 / 1.21 | 0.105 / 0.124 | 0.26 / 0.46 | +2.3% | ✅ / ✅ / ✅ | ✅ |
| `linear_bias_ffn` | 64.4 ms (13) | 0.065 | 0.27 / 0.58 | 11.437 / 16.561 | 11.82 / 16.79 | 0.218 / 0.336 | 0.36 / 0.71 | +2.4% | ✅ / ✅ / ✅ | ✅ |
| `gemm_fp8_e4m3` | 66.7 ms (34) | 0.065 | 0.20 / 0.49 | 4.164 / 5.932 | 4.36 / 6.14 | 0.209 / 0.306 | 0.50 / 0.71 | +2.7% | ✅ / ✅ / ✅ | ✅ |
| `moe_bmm_8x` | 62.1 ms (49) | 0.066 | 0.15 / 0.45 | 3.168 / 4.501 | 3.27 / 4.63 | 0.131 / 0.264 | 0.33 / 0.68 | +3.9% | ✅ / ✅ / ✅ | ✅ |
| `int8_mm` | 65.5 ms (34) | 0.066 | 0.25 / 0.58 | 4.473 / 6.022 | 4.60 / 6.43 | 0.180 / 0.533 | 0.33 / 0.68 | +1.9% | ✅ / ✅ / ✅ | ✅ |
| `sdpa_prefill_causal` | 64.0 ms (53) | 0.067 | 0.15 / 0.46 | 2.994 / 4.946 | 3.18 / 5.09 | 0.140 / 0.240 | 0.33 / 0.57 | +2.1% | ✅ / ✅ / ✅ | ✅ |
| `sdpa_decode_kv4k` | 60.2 ms (30) | 0.066 | 0.28 / 0.54 | 6.752 / 8.567 | 6.86 / 8.80 | 0.148 / 0.161 | 0.32 / 0.62 | +3.1% | ✅ / ✅ / ✅ | ✅ |
| `rope_apply` | 72.3 ms (53) | 0.065 | 0.21 / 0.46 | 1.997 / 2.571 | 2.28 / 2.89 | 0.410 / 0.928 | 0.58 / 1.38 | +3.1% | ✅ / ✅ / ✅ | ✅ |
| `rmsnorm` | 141.2 ms (162) | 0.065 | 0.22 / 0.56 | 1.865 / 2.847 | 2.08 / 3.10 | 0.280 / 0.425 | 0.51 / 0.71 | +1.1% | ✅ / ✅ / ✅ | ✅ |
| `layernorm` | 143.7 ms (164) | 0.065 | 0.20 / 0.52 | 2.217 / 3.060 | 2.49 / 3.22 | 0.401 / 0.541 | 0.66 / 1.03 | +1.4% | ✅ / ✅ / ✅ | ✅ |
| `softmax` | 134.7 ms (71) | 0.065 | 0.20 / 0.51 | 4.660 / 6.261 | 4.90 / 6.44 | 0.602 / 1.082 | 0.80 / 1.67 | +1.7% | ✅ / ✅ / ✅ | ✅ |
| `swiglu` | 97.0 ms (25) | 0.065 | 0.21 / 0.51 | 8.166 / 11.240 | 8.54 / 11.44 | 1.462 / 2.343 | 1.61 / 2.55 | +3.3% | ✅ / ✅ / ✅ | ✅ |
| `gelu_tanh` | 145.1 ms (62) | 0.065 | 0.22 / 0.49 | 5.824 / 7.914 | 5.99 / 8.13 | 1.078 / 1.669 | 1.29 / 1.99 | +3.6% | ✅ / ✅ / ✅ | ✅ |
| `cross_entropy` | 58.2 ms (20) | 0.067 | 0.20 / 0.50 | 6.920 / 9.551 | 7.09 / 9.73 | 0.276 / 0.445 | 0.50 / 0.89 | +2.5% | ✅ / ✅ / ✅ | ✅ |
| `triton_rmsnorm` | 145.0 ms (164) | 0.072 | 0.22 / 0.46 | 1.998 / 2.832 | 2.29 / 3.01 | 0.361 / 0.575 | 0.61 / 1.05 | +1.4% | ✅ / ✅ / ❌ | ✅ |
| `conv3x3_nhwc` | 67.9 ms (19) | 0.066 | 0.17 / 0.46 | 10.317 / 12.535 | 10.44 / 12.67 | 0.267 / 0.363 | 0.45 / 0.81 | +2.2% | ✅ / ✅ / ✅ | ✅ |
| `conv1x1_nchw_fp32` | 84.5 ms (36) | 0.065 | 0.18 / 0.47 | 6.242 / 8.346 | 6.43 / 8.54 | 0.263 / 0.769 | 0.49 / 0.94 | +1.6% | ✅ / ✅ / ✅ | ✅ |
| `batchnorm_train` | 60.2 ms (70) | 0.065 | 0.16 / 0.50 | 3.207 / 5.249 | 3.48 / 5.78 | 0.825 / 1.060 | 0.99 / 1.44 | +4.6% | ✅ / ✅ / ✅ | ❌ job_golden_digest={'native': '840e840e1acdb440', 'shim': '01a8eefba3051a10', 'lv1': '840e840e1acdb440', 'lv3': '0410e9923ffac2b1'} (native nondeterministic) |
| `reduce_sum` | 59.9 ms (89) | 0.065 | 0.18 / 0.47 | 1.801 / 2.897 | 1.97 / 3.08 | 0.268 / 0.620 | 0.48 / 0.75 | +4.3% | ✅ / ✅ / ✅ | ✅ |
| `cumsum_64m` | 91.1 ms (11) | 0.066 | 0.24 / 0.52 | 14.663 / 19.514 | 14.93 / 19.68 | - / - | - / - | - | ✅ / ❌ / CRASH | ❌ golden_digest={'native': 'e05ec8e6e9e741b3', 'shim': 'e05ec8e6e9e741b3', 'lv1': 'e08217dd19350423', 'lv3': None} |
| `sort_16m` | 64.3 ms (11) | 0.065 | 0.21 / 0.53 | 4.595 / 5.693 | 4.81 / 6.14 | - / - | - / - | - | ✅ / ✅ / CRASH | ❌ golden_digest={'native': '500e9a6b662ba7ce', 'shim': '2be115963bdae234', 'lv1': '500e9a6b662ba7ce', 'lv3': None} |
| `topk_sampling` | 60.7 ms (24) | 0.069 | 0.18 / 0.48 | 1.156 / 2.093 | 1.36 / 2.44 | - / - | - / - | - | ✅ / ✅ / CRASH | ✅ |
| `embedding` | 146.2 ms (167) | 0.065 | 0.23 / 0.61 | 2.484 / 3.111 | 2.73 / 3.41 | 0.341 / 0.523 | 0.57 / 0.73 | +2.1% | ✅ / ✅ / ✅ | ✅ |
| `index_add_int32` | 59.4 ms (7) | 0.065 | 6.10 / 8.93 | 0.007 / 46.460 | 0.23 / 46.76 | - / - | - / - | - | ✅ / ✅ / CRASH | ✅ |
| `cast_bool_to_i64` | 162.9 ms (66) | 0.065 | 0.31 / 0.61 | 5.732 / 7.640 | 5.93 / 7.87 | - / - | - / - | - | ✅ / ✅ / CRASH | ✅ |
| `transpose_copy` | 86.4 ms (25) | 0.065 | 0.20 / 0.51 | 8.828 / 10.052 | 9.07 / 10.22 | 1.268 / 2.122 | 1.57 / 2.38 | +7.5% | ✅ / ✅ / ✅ | ✅ |
| `adamw_fused_32m` | 58.5 ms (23) | 0.065 | 0.41 / 0.84 | 10.020 / 14.572 | 10.15 / 14.72 | 0.292 / 0.339 | 0.47 / 0.62 | +3.7% | ✅ / ✅ / ✅ | ✅ |
| `linear_bwd_accum` | 58.3 ms (15) | 0.065 | 0.20 / 0.58 | 16.682 / 20.294 | 16.90 / 20.47 | 0.517 / 0.756 | 0.68 / 1.09 | +5.4% | ✅ / ✅ / ✅ | ✅ |

#### 非比特一致 / 错误记录

- `triton_rmsnorm` [lv3]:  ['trial0: some LP call output != golden', 'trial1: some LP call output != golden', 'trial3: some LP call output != golden', 'trial4: some LP call output != golden']
- `cumsum_64m` [lv1]:  ['trial0: some LP call output != golden', 'trial1: some LP call output != golden', 'trial2: some LP call output != golden', 'trial3: some LP call output != golden', 'trial4: some LP call output != golden']
- `sort_16m` [shim]:  ['trial0: some LP call output != golden', 'trial1: some LP call output != golden', 'trial2: some LP call output != golden', 'trial3: some LP call output != golden', 'trial4: some LP call output != golden']

### 3.1 结果解读

1. **Shim 与 Level-1 的稳态开销可忽略**：批量延迟 Δ 中位数 Shim ≈ 0%、Lv1 ≈ 0.1%；单次调用 Shim 仅增加 ~1–4 µs/调用，Lv1 因经过 XQueue 工作线程增加 ~8–17 µs/调用（5080），对解码类小算子不可忽略。
2. **Level-3 稳态开销取决于 CTA 数量（DEF-17）**：GEMM/注意力/卷积等“少量长 CTA”内核为 +0.6%–+3.5%；而 `transpose_copy`（131,072 个 CTA）+82%、`rope_apply`（5 个内核、大量短 CTA）+101%、`embedding` +23%、`gelu_tanh` +18%（5080）。Entry Guardian 在每个 CTA 入口都执行控制头读取与 `BAR.SYNC`，短 CTA 内核的相对开销随 CTA 数放大。5060 上同类算子开销较小（`transpose_copy` +29%、`rope_apply` +19%），与其 SM 少、单 CTA 相对耗时长一致。
3. **覆盖率 100%**：所有能运行的生产算子（cuBLAS/cuBLASLt FP8/INT8、FlashAttention SDPA、cuDNN 卷积/BN、Triton、Fused AdamW、autograd 反向）在本测试中 **均未触发 mbarrier/Cluster/Cooperative 路由**，全部内核被插桩为 Lv3 可抢占。这说明 RTX 50 系上 PyTorch 2.11 的主流算子尚未大量使用 TMA/mbarrier（与数据中心 Blackwell 不同），但也意味着 **Lv3 的正确性风险覆盖到了全部内核**。
4. **抢占延迟**：Lv3 挂起 P50 中位数 0.27 ms（两卡相近），Lv1 1.98 ms（5080）/ 4.60 ms（5060）。Lv3 抢占时延对作业类型不敏感（0.06–1.2 ms），Lv1 则随 in-flight 内核时长线性变化（`cumsum_64m` 7.2 ms、`linear_bwd_accum` 16.7 ms@5060）。
5. **与“仅流优先级”基线对比（重要修正）**：单算子微基准里 `native` 共置的 HP 探针延迟 P50/P99 中位数为 0.11/0.42 ms（5080）、0.20/0.51 ms（5060），**低于 Lv3 的 0.33/0.58 ms 与 0.50/0.75 ms**。原因：这些 LP 内核由大量短 CTA 组成，硬件按 CTA 粒度调度高优先级流即可快速插队，而 Lv3 需要先完成 BAR0 脉冲与驱逐。只有当 LP 中存在长驻 CTA（例如 `index_add_int32` 原子归约：native 共置 HP P50 1.45 ms vs Lv3 0.24 ms）或 HP 需要整卡资源时，Lv3 的优势才在单算子层面显现。论文评测必须把 “Native High-Priority Stream” 列为基线。
6. **Lv3 作业膨胀**（含 3 次驱逐/恢复）中位数 1.4%（5080）/ 2.5%（5060），恢复本身代价很低。

---

## 4. Suite 8：真实模型服务接地测试（自动生成表格）

#### NVIDIA GeForce RTX 5060 — LP = `ppl` (Qwen/Qwen2.5-0.5B-Instruct)

| mode | TTFT P50/P99 ms | TPOT P50/P99 ms | E2E P50/P99 ms | suspend P50/P99 ms | HP exact | LP iters (bad) | LP tput % of solo | LP use of HP-idle time | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| solo | 11.73 / 12.59 | 10.08 / 11.24 | 159.31 / 171.32 | - / - | 20/20 | 0 (0) | - | - | PASS |
| native | 13.92 / 19.66 | 12.41 / 15.93 | 189.78 / 209.37 | - / - | 20/20 | 195 (0) | 92.9% | 154.1% | PASS |
| lv1 | 12.02 / 23.32 | 10.34 / 11.53 | 168.31 / 184.20 | 0.018 / 11.430 | 20/20 | 140 (0) | 65.8% | 102.7% | PASS |
| lv3 | 12.37 / 17.01 | 10.31 / 11.50 | 170.09 / 177.04 | 0.214 / 1.871 | 20/20 | 106 (0) | 50.3% | 78.4% | PASS |

#### NVIDIA GeForce RTX 5060 — LP = `prefill` (Qwen/Qwen2.5-0.5B-Instruct)

| mode | TTFT P50/P99 ms | TPOT P50/P99 ms | E2E P50/P99 ms | suspend P50/P99 ms | HP exact | LP iters (bad) | LP tput % of solo | LP use of HP-idle time | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| solo | 10.86 / 12.55 | 9.23 / 10.89 | 153.89 / 169.56 | - / - | 20/20 | 0 (0) | - | - | PASS |
| native | 13.02 / 14.29 | 10.05 / 15.26 | 168.48 / 171.13 | - / - | 20/20 | 76 (0) | 93.0% | 144.3% | PASS |
| lv1 | 12.71 / 16.66 | 10.41 / 11.58 | 170.87 / 177.82 | 0.590 / 4.147 | 20/20 | 55 (0) | 67.1% | 104.9% | PASS |
| lv3 | 12.72 / 14.01 | 10.50 / 11.81 | 172.61 / 176.68 | 0.386 / 0.623 | 20/20 | 43 (0) | 52.4% | 82.2% | PASS |

#### NVIDIA GeForce RTX 5080 — LP = `ppl` (Qwen/Qwen2.5-0.5B-Instruct)

| mode | TTFT P50/P99 ms | TPOT P50/P99 ms | E2E P50/P99 ms | suspend P50/P99 ms | HP exact | LP iters (bad) | LP tput % of solo | LP use of HP-idle time | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| solo | 6.81 / 7.25 | 5.86 / 6.02 | 94.97 / 96.69 | - / - | 20/20 | 0 (0) | - | - | PASS |
| native | 10.59 / 12.12 | 8.95 / 10.58 | 145.73 / 150.37 | - / - | 20/20 | 468 (0) | 97.7% | 142.1% | PASS |
| lv1 | 9.26 / 15.05 | 7.86 / 7.97 | 127.54 / 133.56 | 0.092 / 5.999 | 20/20 | 347 (0) | 73.5% | 102.0% | PASS |
| lv3 | 8.49 / 11.96 | 7.34 / 7.45 | 119.33 / 122.96 | 0.159 / 0.275 | 20/20 | 250 (0) | 53.2% | 71.9% | PASS |

#### NVIDIA GeForce RTX 5080 — LP = `prefill` (Qwen/Qwen2.5-0.5B-Instruct)

| mode | TTFT P50/P99 ms | TPOT P50/P99 ms | E2E P50/P99 ms | suspend P50/P99 ms | HP exact | LP iters (bad) | LP tput % of solo | LP use of HP-idle time | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| solo | 7.34 / 7.75 | 6.28 / 6.41 | 101.77 / 103.40 | - / - | 20/20 | 0 (0) | - | - | PASS |
| native | 11.36 / 11.88 | 7.46 / 10.71 | 129.08 / 131.81 | - / - | 20/20 | 222 (0) | 98.5% | 136.7% | PASS |
| lv1 | 9.03 / 13.10 | 7.68 / 7.79 | 124.68 / 128.66 | 0.238 / 0.916 | 20/20 | 168 (0) | 74.8% | 102.6% | PASS |
| lv3 | 8.75 / 9.07 | 7.38 / 7.56 | 119.79 / 121.95 | 0.322 / 0.482 | 20/20 | 115 (0) | 51.4% | 69.5% | PASS |

### 4.1 结果解读与注意事项

1. **正确性**：两卡、两种 LP、四种模式共 16 组运行，HP 20/20 请求 token 完全一致，LP 每次迭代 loss/logits 比特级一致——包括 LP 在 **含大规模 FP32 上转换（cast）内核的困惑度评测** 中被 Lv3 抢占 ~20 次的场景。这说明 DEF-14 的触发需要驱逐恰好命中 cast 内核的 State-B CTA，在真实负载中属于概率事件，而非必然。
2. **延迟**（5080，`ppl`）：无调度共置使 TTFT P50 +55%、TPOT P50 +53%；Lv3 将其压至 +25%，P99 尾延迟最紧（TPOT P99 7.45 ms vs 共置 10.58 ms、Lv1 7.97 ms；TTFT P99 11.96 ms vs Lv1 15.05 ms）。Lv1 的 TTFT 尾部受限于等待 in-flight LP 内核（挂起 P99 6.0 ms）。
3. **吞吐代价**：请求粒度挂起策略下 LP 吞吐为独占的 ~52%（Lv3）/ ~74%（Lv1），无调度共置为 ~98%。Lv3 对 HP 空闲时间的利用率仅 ~70%，低于 Lv1 的 ~102%，与 DEF-17 的短 CTA 开销及每请求一次驱逐/恢复一致。**0.5B 小模型的解码受 CPU 启动开销主导（GPU 在 HP 请求期间大量空闲），请求粒度“整段挂起 LP”会浪费这部分 GPU 时间**，后续应评估 token 粒度或 SLO 感知的挂起策略。
4. **测试限制（如实说明）**：HP 与 LP 为同一 Python 进程内的两个线程，LP 线程即使 GPU 队列被挂起仍在执行 Python 前向代码并争用 GIL；Lv1/Lv3 下 HP 相对 solo 仍有 ~+25% 的残余膨胀，Shim 直通路径仅 ~1–4 µs/调用，无法解释，GIL 争用是主要嫌疑（未验证）。生产部署中 HP/LP 通常为独立进程，应以多进程方式复测。模型规模受 8 GiB 5060 限制为 0.5B。

---

## 5. 缺陷台账（本轮新增 / 更新）

| 编号 | 级别 | 现象 | 复现 | 范围 | 状态 |
|---|---|---|---|---|---|
| **DEF-14** | P0 崩溃 | `unrolled_elementwise_kernel<direct_copy_kernel_cuda>`（PyTorch 所有 dtype 转换，如 `bool→int64`、`int8→int64`）在首次 State-B 驱逐后恢复即 `cuda error 700/717` | `bench_prod_ops_p3.py run --op cast_bool_to_i64 --mode lv3`（双卡 100%）；最小脚本：Lv3 队列上循环 `x.to(torch.int64)` + 3 次挂起/恢复 | 双卡；HEAD 同样复现 | **根因 = DEF-20，已修复验证（待提交）** |
| **DEF-15** | P0 崩溃 | CUB Onesweep（`torch.sort` 16M）、CUB 扫描（整型 `cumsum` 64M）、`torch.topk`（`mbtopk::computeBlockDigitCounts`）、`index_add_int32`（仅 5060）在 Lv3 抢占下崩溃；`LIB-08` NP=2 偶发 700；`LLM-07 m5` 偶发 700 | `--op sort_16m / cumsum_64m / topk_sampling --mode lv3`；`TEST_NP=2 test_libs_and_features LIB-08` | 双卡；HEAD 同样复现 | **根因 = DEF-20，已修复验证（待提交）** |
| **DEF-16** | **P0 静默错误** | Triton 融合 RMSNorm（256 线程、32 B 动态 SMEM、`tl.sum` 跨 Warp 归约）在 Lv3 抢占下输出错误但不报错，每 5 轮仅 1–2 轮正确 | `--op triton_rmsnorm --mode lv3` | 双卡；HEAD 同样复现 | **根因 = DEF-20，已修复验证（待提交）** |
| **DEF-17** | P1 性能 | Entry Guardian 每 CTA 开销：短 CTA 访存算子 Lv3 稳态最高 +101% | 表 A | 双卡 | 待优化 |
| **DEF-18** | P1 回归风险 | 工作区 `DEF-12` 恢复路径改动使 `LIB-08`（NP=2）失败率 15% → 38%（p = 0.019） | 交错 A/B：`L8-CUR` vs `L8-HEAD` 各 30 次 + 之前 20/10 次 | 5080 | **结论已修订**：差异源于与 DEF-20 叠加；修复后两版本 LIB-08 均 30/30，DEF-12 改动应保留（见缺陷台账 §2.3） |
| DEF-19 | P2 偶发 | `HW-06` Cluster+DSMEM `mismatch=256/7680` 约 1/3；`LLM-07 m5` 约 1/3 崩溃 | `TEST_NP=2 test_libs_and_features HW-06` | 5080；HEAD 失败率相同（既有问题） | 待查 |
| OBS-1 | 观察 | 5080 上 `linear_bwd_accum` / `batchnorm_train` 的 Lv3 进程整条作业摘要与 native 进程不同（两次 Lv3 进程彼此一致）；**同进程 A/B 40/60 次连续调用逐次比特一致** | `lv3_drift.py` | 5080 | 判定为进程级库启发式（cuBLAS/cuDNN 算法选择）差异，非数据损坏；需固定算法后复核 |
| OBS-2 | 观察 | 5060 首轮出现 `cumsum@lv1`、`sort@shim` 摘要不一致 | 对照组各 2 进程 × 3 轮复测 0/6 复现 | 5060 | 未能复现，记录为瞬态 |

### 5.1 DEF-14/15/16 的诊断证据

- **触发条件**：所有崩溃/静默错误都发生在 **存在 State-B 驱逐（`evicted > 0`）的抢占** 之后；只有 State-C 排空（`evicted = 0`）的抢占全部正确恢复。
- **DEF-14 的 Trap 现场**：驱逐 CTA 的保存 PC 位于原始 SASS 的 `+0x49c0`（`@P0 BREAK.RELIABLE B6`）以及 `+0x3190`（等待 `LDG.E.U8 R2, [R2.64]` 计分板的 `ISETP`）。该内核含 23 个 `EXIT`、42 对 `BSSY/BSYNC`，使用收敛屏障 B0/B1/B6/B7；通过的 `vectorized_elementwise_kernel` 仅 3 对 `BSSY` 且最高为 B1/B3。
- **DEF-16 的 Trap 现场**：驱逐 CTA 位于 `+0x650..0x710`，即 `BAR.SYNC` 之后读取跨 Warp 归约结果的 `LDS` 附近，各 Warp PC 互不相同。
- **已排除的 7 类假设**（均有实验证据）：
  1. 代码重定位（`BRX/JMX` 跳转表、`CALL.ABS`）：失败内核无 `BRX/JMX`，`CALL.ABS` 使用 `LEPC` 运行时 PC 计算返回地址，可重定位；
  2. `BMOV.32(.CLEAR)` B0–B15 手写编码：`nvdisasm -b SM120` 逐条核验 32 条全部正确；
  3. HP 流干扰：不运行 HP 时同样崩溃；
  4. 部分退出 Warp（`if (lane&1) return`、逐 lane 错峰 `EXIT`）：新增合成测试 `test_partial_exit_p3.cu` 5/5 通过；
  5. Warp 内发散后再汇合：同上 5/5 通过；
  6. 在途全局加载：通过的 bf16 转换核同样在等待 `LDG` 计分板时被 Trap 且正确恢复；
  7. `SR_SMEMSZ − 0x400` 共享内存记账溢出：编译器以 `%total_smem_size = SR_SMEMSZ − c[0][0x16c]` 计算且用户 SMEM 起始于 `0x400`，引擎的 4 KiB 槽位面积覆盖实际拷贝长度。
- **后续结论（已确认）**：收敛屏障编号同样被合成测试排除（D1–D6 嵌套 B0–B5 全部通过）；寄存器压力扫描显示 **≤13 寄存器的内核从不出错、用到 R14+ 的内核都会出错**，`SR_REGALLOC` 探针证明其值等于内核寄存器数，原 `SR_REGALLOC − 3` 漏存最高 1–3 个寄存器 —— 即 **DEF-20**。上面的 Trap 现场（`direct_copy` 33 寄存器丢 R32、RMSNorm 35 寄存器丢 R34）与此一致。详见缺陷台账 §2.1 与 §6 调查时间线。

### 5.2 风险提示

**修复前的 v2.2.1 不应宣称对 PyTorch 生产负载“100% 比特级正确”**。DEF-20 修复后两卡 29/29 生产算子通过，但仍有 DEF-21（恢复波次再抢占）与 DEF-19（Cluster+DSMEM）开放，宣称正确性时需注明这两类边界。

---

## 6. 本轮新增的诊断与测试资产

| 资产 | 位置 | 用途 |
|---|---|---|
| `XSm120GetRouteStats()` | `platforms/cuda/hal/src/level3/cuda_queue.cpp` | 进程级自适应路由计数（Lv3 可抢占 / mbarrier / Cooperative / 多波 Cluster / Graph / 插桩失败），供评测统计覆盖率 |
| `XSCHED_SM120_TRACE_LAUNCH=1` | `cuda_queue.cpp`、`sm120_trap_engine.cpp` | 每次 Lv3 启动的内核名/网格/线程/动态 SMEM/路由；每次恢复的遥测（cmd/evicted/drained/active_idx/在途命令序列）；**驱逐 CTA 的逐 Warp Trap PC（换算为原始 SASS 偏移）** |
| `XSCHED_SM120_DUMP_DIR=<dir>` | `sm120_guardian.cpp` | 插桩时导出原始 SASS 二进制与内核名，`nvdisasm -b SM120` 反汇编 |
| 内核名 | `sm120_guardian.cpp` 插桩日志 | 插桩日志新增原始二进制大小与内核名 |
| Suite 7 | `test/sm120_v2/eval/bench_prod_ops_p3.py` | 29 个生产算子 × 4 模式；`run / list / report` |
| Suite 8 | `test/sm120_v2/eval/bench_real_serving_p3.py` | 真实权重在线服务 + 离线评测共置；`run / report` |
| 合成 Trap 测试 | `test/sm120_v2/eval/test_partial_exit_p3.cu` | 部分退出 / 发散 / 汇合 Warp 的 Mode 5 驱逐恢复 |

复现命令示例（5060 本地，需 root 映射 BAR0）：

```bash
XS=/path/to/xsched_install; cd xsched/test/sm120_v2/eval
# 单个算子单个模式（native 需保证 XSched 不在 LD_LIBRARY_PATH 中）
sudo env LD_LIBRARY_PATH=$XS/lib:$CUDA_HOME/lib64 python bench_prod_ops_p3.py run --op cast_bool_to_i64 --mode lv3 --out ops.jsonl
python bench_prod_ops_p3.py report ops.jsonl --md ops.md
# 诊断：追踪 + 驱逐现场 + SASS 导出
sudo env LD_LIBRARY_PATH=$XS/lib XSCHED_SM120_TRACE_LAUNCH=1 XSCHED_SM120_DUMP_DIR=/tmp/sass \
     python bench_prod_ops_p3.py run --op triton_rmsnorm --mode lv3 --out dbg.jsonl
# 真实模型服务
sudo env LD_LIBRARY_PATH=$XS/lib HF_HUB_OFFLINE=1 python bench_real_serving_p3.py run --mode lv3 --lp ppl --out serving.jsonl
```

---

## 7. 下一步建议（按优先级）

1. ~~根因修复 DEF-14/15/16~~ → 已定位为 DEF-20 并修复验证；**提交 DEF-20 + DEF-11/12/13 并全量回归**。
2. **DEF-21**：重新设计恢复波次再抢占状态机（DEF-12 改动保留）；**DEF-19**：先把所有 Cluster 内核路由到 Lv1。
3. **降低 Entry Guardian 每 CTA 开销（DEF-17）**：例如以 warp 0 单线程读取控制头、对“无挂起”快速路径去掉 `BAR.SYNC`。
4. **补充基线**：论文评测加入 “Native High-Priority Stream”、MPS、Green Context；服务场景改为 HP/LP 独立进程，并评估 token 粒度 / SLO 感知挂起策略以挽回 LP 吞吐。
5. 每次 Trap 引擎改动后回归 Suite 7（两卡约 15–25 分钟）+ Suite 1/4/5/6 + `LIB-08` 交错 A/B。
