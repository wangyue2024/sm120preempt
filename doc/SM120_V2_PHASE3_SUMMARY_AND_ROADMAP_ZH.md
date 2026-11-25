# SM120 v2 阶段成果总结与下一步规划（截至 2026-11-25）

> 适用代码：`main@c44da4f`（`v2.2.1-sm120`）+ 本次提交的 Phase 3 测试资产、DEF-11/12/13 与 **DEF-20** 修复、诊断工具。
> 详细材料：
> - 测试方法与性能数据：[`SM120_V2_PHASE3_PRODUCTION_OPS_GROUNDED_TESTS_ZH.md`](SM120_V2_PHASE3_PRODUCTION_OPS_GROUNDED_TESTS_ZH.md)
> - 缺陷台账与调查全记录：[`SM120_V2_PHASE3_DEFECT_LEDGER_ZH.md`](SM120_V2_PHASE3_DEFECT_LEDGER_ZH.md)
> - 原始证据（脱敏）：[`evidence/phase3_20261125/`](evidence/phase3_20261125/)
> - 架构与实现细节：[`SM120_V2_COMPLETE_TECHNICAL_MANUAL_ZH.md`](SM120_V2_COMPLETE_TECHNICAL_MANUAL_ZH.md)（v2.2.1，已加 DEF-20 勘误）

> **2026-11-25 晚更新：Phase 3.2 / 3.3 已完成**，详见 [`SM120_V2_PHASE3_2_3_3_REPORT_ZH.md`](SM120_V2_PHASE3_2_3_3_REPORT_ZH.md)：DEF-21 关闭（原合成失败为测试竞态）、DEF-19 缓解、DEF-22/23 修复、Entry Guardian 开销大幅下降（5080 最差 +101% → +56.5%，`embedding`/`gelu` 降至 ≈ 0；默认安全版本）、XQueue 有界自旋、真实服务调度策略 Pareto 前沿、一键两卡回归流水线；全部改动后两卡全量回归 **184/184 + 184/184 全部通过**（该报告 §6）。下文 §0–§4 为 3.2 之前的状态，保留供对照；§4 中 3.2/3.3 各项的完成情况见 §4.0。

---

## 0. 一句话现状

在 Blackwell `sm_120`（RTX 5060 / RTX 5080）上，**用户端透明、不依赖幂等性的内核内抢占（Level-3 Mode 5）已经可以对 PyTorch 2.11 主流生产算子做到逐调用比特级正确**（29/29 个算子 × 两卡），抢占挂起延迟中位数约 0.26–0.31 ms；本阶段通过生产负载测试发现并修复了一个影响所有内核的 P0 根因缺陷（DEF-20）。仍有 1 个 P1 正确性问题（恢复中途再抢占，DEF-21）和 1 个 P2 问题（Cluster+DSMEM，DEF-19）开放，Lv3 对“短 CTA 访存算子”的稳态开销偏高（DEF-17）。

---

## 1. 成果总览

### 1.1 技术基线（`v2.2.1-sm120`，此前已完成）

| 组件 | 内容 |
|---|---|
| 抢占模式 | Mode 1（幂等早退）/ **Mode 5**（非幂等，全状态 Trap 帧保存与 0 脉冲 `BPT.TRAP` 自陷恢复） |
| Trap 引擎 | 静态 9-Cave 代码洞（ToolsMemory），保存 GPR / UR / UPR / B0–B15 / 命名屏障 / 动态 SMEM / LMEM 栈 / 部分退出 CTA |
| Entry Guardian | 每个内核插桩 0x500 B 入口序言，负责多内核 FIFO、排空（State C）与恢复分派 |
| 控制面 | GDDR7 显存控制头 + 高优先级 Copy Engine DMA + GPU 原生条件清理 |
| 自适应路由 | mbarrier / Cooperative / 多波次 Cluster / CUDA Graph 自动降级 Level-1 |
| XSched 集成 | `CudaQueueLv3Trap` 接入 XQueue；Shim 透明拦截 CUDA Driver API；独立 Context CILP |

### 1.2 Phase 3 测试体系（本阶段建成）

| 套件 | 内容 | 文件 |
|---|---|---|
| Suite 1–3 | 闭源库 + Blackwell 硬件特性、LLM/DNN QoS、PyTorch 端到端（v2.2.1 已有） | `test/sm120_v2/eval/` |
| Suite 4 | cuSOLVER LU/Cholesky、cuRAND 有状态 PRNG、cuFFT 2D、CUB Look-Back、cuBLASLt FP8 | `test_real_cuda_libs_p3.cu` |
| Suite 5 | ViT / Swin / ConvNeXt / DeepLabV3 / AMP+梯度累积+Fused AdamW / `torch.compile` / cuDNN LSTM / Diffusers | `test_real_dnn_prod_p3.py` |
| Suite 6 | HF Qwen2 Prefill/Decode、PEFT LoRA、FP8/INT8/W4A16、PagedAttention、共置 QoS、真实权重 | `test_real_llm_prod_p3.py` |
| **Suite 7** | **29 个生产算子 × native/shim/Lv1/Lv3，逐调用比特级校验、路由覆盖率、挂起/HP 延迟、原生高优先级流基线** | `bench_prod_ops_p3.py` |
| **Suite 8** | **真实 Qwen2.5-0.5B 权重在线服务（TTFT/TPOT）+ 离线评测共置** | `bench_real_serving_p3.py` |
| 合成 Trap 测试 | 部分退出/发散 warp；嵌套收敛屏障；寄存器压力扫描；逐寄存器归因 | `test_partial_exit_p3.cu`、`test_convbar_p3.cu` |
| 复现与探针 | DEF-14 最小复现、`SR_REGALLOC` 二进制改写探针、SMEM 大小探针、同进程漂移检查 | `repro/` |
| 安全运行器 | 共享服务器上每个测试前检查 GPU 占用；sudo 口令经 stdin 传递 | `p3_guarded_run.sh` |

### 1.3 关键实测结果（DEF-20 修复后，工作区构建）

| 指标 | RTX 5080（84 SM） | RTX 5060（30 SM） |
|---|---|---|
| Suite 7 Lv3 逐调用比特级正确（29 算子 × 15 次中途抢占） | **29/29** | **29/29** |
| Lv3 挂起延迟 P50 中位数（范围） | **0.263 ms**（0.058–1.215） | **0.308 ms**（0.099–1.791） |
| Lv1 挂起延迟 P50 中位数（对照） | 1.98 ms | 4.60 ms |
| HP 探针端到端 P50 / P99 中位数：Lv3 | 0.35 / 0.61 ms | 0.45 / 0.82 ms |
| HP 探针端到端 P50 / P99 中位数：仅高优先级流（无调度） | 0.11 / 0.42 ms | 0.20 / 0.51 ms |
| Lv3 稳态开销中位数 / 最大（修复前测量；5080 上已核对修复前后差异 ≤ 0.1%） | +1.7% / +101%（`rope_apply`） | +0.6% / +29% |
| Lv3 作业膨胀（含 3 次驱逐恢复）中位数 | 1.5% | 2.5% |
| Lv3 内核覆盖率（未被路由回 Lv1） | 100% | 100% |
| Suite 1 / 4 / 5 / 6（Lv3） | 13/13、6/6、8/8、7/7（HW-06 见 DEF-19） | — |
| LIB-08 CUB RadixSort NP=2 / LLM-07 m5 | 30/30 / 5/5 | — |
| 真实模型服务（Qwen2.5-0.5B，Lv3，20 请求） | HP 20/20 token 一致、LP 0 错（修复前测量）；TTFT/TPOT P50 较无调度共置 −20%/−18%；LP 吞吐为独占的 ~53% | 修复后 HP 20/20、LP 0 错 |

### 1.4 本阶段定位/修复的缺陷

| 编号 | 内容 | 状态 |
|---|---|---|
| DEF-11 | Shim 漏拦截 `cuMemcpyAsync` / `cuMemAllocFromPoolAsync`（cuDNN LSTM 竞态） | 已修复（本次提交） |
| DEF-12 | 恢复中途再抢占丢失 CTA（`torch.compile` inductor） | 已修复主路径（本次提交），剩余问题见 DEF-21 |
| DEF-13 | cuDNN 多内部流复用同一 `CUevent` 导致等待快照丢失 | 已修复（本次提交） |
| **DEF-20** | **Trap 引擎漏存每个内核最高 1–3 个寄存器**（`SR_REGALLOC` 实测等于寄存器数）——DEF-14/15/16、LIB-08、LLM-07 的共同根因 | **已修复并两卡验证（本次提交）** |

### 1.5 诊断与工程能力

- `XSCHED_SM120_TRACE_LAUNCH=1`：逐内核启动追踪、恢复遥测、**驱逐 CTA 的逐 warp Trap PC（原始 SASS 偏移）**；
- `XSCHED_SM120_DUMP_DIR`：导出插桩前原始 SASS 供 `nvdisasm -b SM120`；
- `XSm120GetRouteStats()`：自适应路由覆盖率统计；
- 跨机器安全运行规范：共享 5080 每测试前 GPU 空闲检查、口令不进命令行。

---

## 2. 当前已知问题（开放项）

| 编号 | 级别 | 问题 | 现状 |
|---|---|---|---|
| DEF-21 | P1 | 恢复波次进行中再次抢占 | **已关闭**（3.2：合成失败为测试竞态 T-11；DEF-12 逻辑经约 3,000 次压测抢占验证） |
| DEF-19 | P2 | CTA Cluster + DSMEM（HW-06）偶发 1–2 个 CTA 结果错误（5080 约 10%） | **已缓解**（3.2：Cluster 内核默认走 Level-1） |
| DEF-17 | P1（性能） | Entry Guardian 每 CTA 开销：短 CTA 访存算子 Lv3 稳态最高 +101% | **改善**（3.3：去冗余重读，5080 `rope_apply` +101% → +56.5%、`embedding` +23% → +1.9%；5060 `transpose_copy` +28.5% → +18.1%）；更激进版本为实验开关 |
| DEF-22 | P2（潜在） | `CALL.ABS` 未被识别为设备调用，被调函数栈帧不保存 | **已修复**（3.2） |
| DEF-23 | P2（显存） | 含调用内核固定预留 4 KiB/线程栈保存区 | **已修复**（3.2：按栈上限精确预留） |
| EXP-1 | — | 实验 Guardian v2 在双 Lv3 队列 + HP 下偶发错误（11/61） | 仅实验开关 |
| DEF-20b | P3 | 255 寄存器内核的 R254 无法保存 | 生产内核未见 |
| DEF-09 | P2 | CUDA Graph 仅在图边界抢占（Level-1） | 架构项 |
| DEF-10 | P2 | BAR0 需要 root | 架构项 |
| 遗留 | — | `shim/context_registry.{h,cpp}`：context-isolation 合并中断留下的未接入代码 | 待决定完成或删除 |
| 评测缺口 | — | 缺“原生高优先级流 / MPS / Green Context”基线；服务测试为单进程双线程（GIL 干扰）；模型仅 0.5B | 待补 |

---

## 3. 交付物清单

| 类别 | 路径 |
|---|---|
| 本总结 | `doc/SM120_V2_PHASE3_SUMMARY_AND_ROADMAP_ZH.md` |
| 测试报告 | `doc/SM120_V2_PHASE3_PRODUCTION_OPS_GROUNDED_TESTS_ZH.md` |
| 缺陷台账 | `doc/SM120_V2_PHASE3_DEFECT_LEDGER_ZH.md` |
| 原始证据 | `doc/evidence/phase3_20261125/`（Suite 7/8 JSONL 与报表、修复前后对比、Trace 与 SASS 摘录、探针输出、二分统计） |
| 测试与工具 | `xsched/test/sm120_v2/eval/`（Suite 4–8、合成测试、`repro/`、`p3_guarded_run.sh`、`run_phase3_eval.sh`） |
| 代码修复 | `xsched/platforms/cuda/hal/src/level3/{sm120_trap_engine,sm120_guardian,cuda_queue}.cpp`、`shim/*`、`hal/common/*` |

---

## 4. 下一步规划

### 4.0 Phase 3.2 / 3.3 完成情况（2026-11-25）

| 计划项 | 状态 | 说明 |
|---|---|---|
| 3.2-1 DEF-21 | ✅ 关闭 | 合成失败为测试竞态；压测通过 |
| 3.2-2 DEF-19 缓解 | ✅ | Cluster → Level-1 |
| 3.2-3 DEF-22 | ✅ | 并发现、修复 DEF-23（LMEM 过度预留） |
| 3.2-4 回归流水线 | ✅ | `regress_p3.py` / `build_regress_bins.sh` / `p3_guarded_run.sh` |
| 3.2-5 文档 | ✅（`.docx` 未重生成） | 新增 3.2/3.3 报告，更新台账、总结、手册勘误 |
| 3.2-5 `context_registry` 清理 | ⏳ | 保留为 WIP 提交，待 context-isolation 方案决策 |
| 3.3-1 DEF-17 | ◐ 部分 | 默认版本：5080 最差算子 +101% → +56.5%，多数短 CTA 算子开销基本消失；实验 v2 / `BAR.RED` 需硬件表征后再启用 |
| 3.3-2 每调用开销 | ◐ 部分 | 有界自旋 −3~5 µs（Lv3）；剩余需改 XQueue 同步模型 |
| 3.3-3 调度策略 | ✅（结论为 Pareto 前沿） | `prefill/token/slo` 策略；无同时满足两目标的策略 |

### 4.1 Phase 3.2 — 正确性收尾（优先，约 1–2 周）

| # | 任务 | 验收标准 |
|---|---|---|
| 1 | **DEF-21**：重新设计“恢复波次中再抢占”的状态机（候选：恢复波次期间先等待恢复完成再驱逐；或为二次驱逐使用独立计数与槽位），用 Trace 工具逐步核对 State A/B/C 与 `active_kernel_idx` | `test_convbar_p3 C1 10 2/3` 两卡 10/10；`dnn06` 5/5；`bench_real_serving_p3 --mode lv3` 两卡 0 错误；12 内核顺序 3 次抢占全过 |
| 2 | **DEF-19 缓解**：所有 `ClusterSize > 1` 的内核路由到 Lv1 | HW-06 两卡 20/20 |
| 3 | **DEF-22**：`BinaryHasDeviceCalls` 增加 `CALL.ABS`（`0x343`），评估 Cave 7 栈保存带来的开销 | Suite 7 全过且开销中位数变化 < 1% |
| 4 | **回归流水线**：把 Suite 1/4/5/6/7 + 合成测试 + LIB-08×30 固化为一键脚本（基于 `p3_guarded_run.sh` 计划文件），每次 Trap 引擎改动后两卡必跑 | 一条命令产出两卡汇总表 |
| 5 | 文档：更新 `.docx` 技术手册（DEF-20 勘误、Phase 3 结果），清理 `context_registry` 遗留 | — |

### 4.2 Phase 3.3 — 性能（约 1–2 周）

| # | 任务 | 目标 |
|---|---|---|
| 1 | **DEF-17**：Entry Guardian 快速路径（无抢占时减少每 CTA 的控制头读取与 `BAR.SYNC`），或对“CTA 极多、单 CTA 极短”的内核采用 Lv1 路由 | `transpose_copy` / `rope_apply` / `embedding` 开销降至 < 10% |
| 2 | Lv1 / Lv3 每调用提交开销（8–17 µs/调用）分析与优化 | 解码类小算子单次调用开销 < 5 µs |
| 3 | 调度策略：从“请求粒度整段挂起 LP”改为 token 粒度 / SLO 感知挂起，利用 HP 解码期间的 GPU 空闲 | 真实服务 LP 吞吐由 ~53% 提升到 > 80%，HP TPOT P99 不劣化 |

### 4.3 Phase 3.4 — 系统化（约 2–4 周）

| # | 任务 |
|---|---|
| 1 | **DEF-10**：非 root BAR0 代理（特权守护进程 / 字符设备 + 进程校验 + 频率限流 + 多进程互斥） |
| 2 | 多进程/多租户：HP 与 LP 独立进程的服务测试（消除 GIL 干扰），跨进程抢占仲裁 |
| 3 | **DEF-09**：CUDA Graph 图内抢占（`cuGraphInstantiate` 节点预插桩）或与 Context CILP 的混合路由，覆盖 `torch.compile(reduce-overhead)` 与 vLLM Decode |
| 4 | 决定 xsched `release/context-isolation` 合并方案（`context_registry` 接入或放弃） |

### 4.4 Phase 3.5 — 论文评测（与 3.3/3.4 并行）

| # | 任务 |
|---|---|
| 1 | 统一负载上的基线对比：原生 CUDA Stream、**原生高优先级流**、NVIDIA MPS、CUDA Green Context、XSched Lv1、Mode 1、Mode 5、Context CILP |
| 2 | 更大模型与真实服务框架：vLLM / SGLang（7B–8B，长上下文 Prefill + 实时 Decode；LoRA 训练 + 在线推理） |
| 3 | 消融：GDDR7 控制面 vs PCIe、0 脉冲恢复、紧凑 slot 池化、Guardian 快速路径 |
| 4 | 正确性论证：Suite 7 + 合成测试作为“非幂等内核全状态恢复”的系统性证据，报告 DEF-20 类问题的检测方法（寄存器压力扫描 + `SR_REGALLOC` 探针） |

### 4.5 风险

- Trap 引擎依赖逆向得到的硬件语义（`SR_REGALLOC`、ToolsMemory 布局、`BAR.SYNC.TRAP`），驱动版本变化可能改变行为：**每次升级驱动需先跑合成测试与 `repro/` 探针**。
- 5080 为共享服务器：测试需避开他人作业（已有守护运行器），大规模评测需提前协调时间。

---

## 5. 复现与回归命令（摘要）

```bash
# 构建（两卡相同）
cmake -S xsched -B build_sm120 -DCMAKE_BUILD_TYPE=Release -DPLATFORM_CUDA=ON -DSHIM_SOFTLINK=ON \
      -DCMAKE_CUDA_COMPILER=<cuda-12.9>/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build_sm120 -j && cmake --install build_sm120 --prefix <XS>

cd xsched/test/sm120_v2/eval
# Suite 7：单个算子单个模式（native 需保证 XSched 不在 LD_LIBRARY_PATH）
sudo env LD_LIBRARY_PATH=<XS>/lib:<cuda>/lib64 python bench_prod_ops_p3.py run --op sort_16m --mode lv3 --out ops.jsonl
python bench_prod_ops_p3.py report ops.jsonl --md ops.md
# 合成 Trap 测试
nvcc -arch=sm_120 -O3 -I<XS>/include test_convbar_p3.cu -o test_convbar_p3 -L<XS>/lib -lpreempt -lhalcuda -lcuda
sudo env LD_LIBRARY_PATH=<XS>/lib ./test_convbar_p3 all 3 3
# 真实模型服务
sudo env LD_LIBRARY_PATH=<XS>/lib HF_HUB_OFFLINE=1 python bench_real_serving_p3.py run --mode lv3 --lp ppl --out serving.jsonl
# 共享服务器上按计划文件安全运行（每个测试前检查 GPU 空闲）
printf '%s\n' "$PW" | ssh <host> "sudo -S -p '' bash p3_guarded_run.sh plan.txt run.log"
```
