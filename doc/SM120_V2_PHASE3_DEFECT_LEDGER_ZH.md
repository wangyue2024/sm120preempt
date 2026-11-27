# SM120 v2 Phase 3 缺陷台账与问题调查全记录（2026-11-25）

> 配套文档：[`SM120_V2_PHASE3_PRODUCTION_OPS_GROUNDED_TESTS_ZH.md`](SM120_V2_PHASE3_PRODUCTION_OPS_GROUNDED_TESTS_ZH.md)（测试方法与性能数据）
> 原始证据：[`evidence/phase3_20261125/`](evidence/phase3_20261125/)（已脱敏的 JSONL 原始结果、Trace 摘录、SASS 现场、探针输出、合成测试与二分结果）
> 复现工具：[`../xsched/test/sm120_v2/eval/repro/`](../xsched/test/sm120_v2/eval/repro/)
> 硬件：RTX 5060（GB206，30 SM，驱动 595.99.02）/ RTX 5080（GB203，84 SM，驱动 580.95.05，共享服务器）
> 记录原则：所有结论都标注证据与样本量；“已确认”仅用于有对照实验支持的结论；推测明确标注为“假设/未验证”。

---

## 0. 一页结论

| 问题 | 一句话描述 | 原因 | 状态 | 修复难度 |
|---|---|---|---|---|
| **DEF-20**（P0，根因） | Trap 引擎从不保存/恢复每个内核**最高的 1–3 个寄存器** | `SR_REGALLOC` 实测等于内核寄存器数，引擎按“多含 2 个保留寄存器”处理（`−3`）；奇数寄存器时 64 位寄存器对还会越界 | **根因已确认；修复（2 处、共 3 行）已在两卡验证通过；未提交** | **容易**（已完成） |
| DEF-14 / 15 / 16 | dtype 转换核恢复即崩溃；CUB 排序/扫描、topk、int 原子崩溃；Triton RMSNorm 静默算错 | **均为 DEF-20 的表现** | 随 DEF-20 修复，两卡 29/29 算子比特一致 | — |
| LIB-08 / LLM-07 偶发 700 | 长期记为 “P2 偶发” 的 CUB 基数排序与 LoRA 训练崩溃 | **DEF-20** | 修复后 LIB-08 60/60、LLM-07 10/10 | — |
| **DEF-21**（P1） | 恢复波次（restore wave）进行中再次被抢占时出错 | HEAD 版本丢 CTA（真实缺陷，已由 DEF-12 逻辑修复）；“DEF-12 逻辑在合成用例上出错”系**测试工具竞态**（T-11） | **已关闭**（§9.1，约 3,000 次压测抢占全部比特一致） | — |
| **DEF-19**（P2） | Blackwell CTA Cluster + DSMEM 用例偶发错 1–2 个 CTA 的结果 | 推测：Cluster 级状态（集群屏障、分布式共享内存映射）不在 Trap 帧中（未验证） | **已缓解**：Cluster 内核默认走 Level-1（§9.2）；完整支持仍开放 | 缓解已完成；真正支持**难** |
| **DEF-17**（P1，性能） | CTA 数多且单 CTA 极短的访存算子 Lv3 稳态开销最高 +101% | Entry Guardian 每条指令串行等待 + 热点控制头的冗余重读 | **达标**：默认去冗余重读 + 短 CTA 路由（无回边多波次内核走 Lv1）：最终回归 5080 `transpose_copy` +82% → −0.0%、`rope_apply` +101% → +2.2%、`sort` +25% → +3.3%；5060 → +0.1% / +1.1% / +5.7%；29 个算子 Lv3 中位数两卡 +0.3%（§9.3、§10.1、§10.7、§10.8） | Guardian 本体剩余开销（有循环的短 CTA 内核）**中–难** |
| **DEF-22**（P2，潜在） | 设备函数调用检测漏掉 `CALL.ABS` | 只匹配 `0x943/0x944`，`CALL.ABS.NOINC` 编码为 `0x343` | **已修复**（§9.2） | — |
| **DEF-23**（P2，显存） | 含设备调用的内核每线程固定预留 4 KiB 栈保存区（LayerNorm arena 231 MiB） | 预留值固定为 `max(static, 4096)` | **已修复**：按“静态局部 + `CU_LIMIT_STACK_SIZE`”精确预留（§9.2） | — |
| **DEF-24**（P2，潜在） | 通过函数指针的间接调用未被识别为设备调用 | 间接调用编码为 `CALL.REL.NOINC Rx`（`0x344`），检测只匹配 `0x943/0x944/0x343` | **已修复**：增加 `0x344`，并以 `RET`（`0x950`）兜底（§10.2） | — |
| **PERF-1**（P2，性能） | 每次调用（提交 + 同步）比 native 多 8–20 µs | XQueue 两次线程交接 + Lv3 每次发射的驱动调用 | **已改善（达标）**：直接提交（含按调用长度自适应）、调用方 `WaitAll`、布局缓存；单内核调用两卡 Lv1 +1.1–1.2 µs、Lv3 +2.2–2.4 µs（§10.3、§10.6） | — |
| **MEM-1**（P3，显存） | Mode 5 arena 按流高水位常驻，增长替换下来的旧 arena 直到流销毁才释放 | 无空闲回收 | **已修复**：同步后的安全点回收（§10.6） | — |
| **PERF-2**（P1，性能） | 加载 XSched 后 HP（未受管流）decode TPOT 慢约 1.2×，与 LP 是否运行无关 | 每个透传内核命令构造时都调用 `cuXtraKernelGetFunction`（decode 步 8.4% CPU 样本），sm120 透传每次清零 Guardian 参数 | **已修复**：按需解析 `CUfunction`；仅在可能持有非零参数时清零；5060 进程内 decode TPOT 开销 +16% → +1.3–3.8%（§10.5） | — |
| **DEF-09**（P2→完成） | CUDA Graph 只能在图边界挂起（`reduce-overhead` 图挂起 10–31 ms，训练步图 289 / 407 ms） | 驱动整体发射图 | **已完成（Phase 3.5）**：Level-3 队列上按节点展平提交；训练步图挂起 P99 0.4–1.4 ms、比特一致（§10.12） | — |
| **DEF-29**（P1，性能） | 普通流等待 XQueue 事件时阻塞调用线程（vLLM Level-1 −22%） | Level-2/3 下事件可能因提前结束的内核而提前触发，原实现一律在主机上等完成 | **已修复**：Level-1 发射后原生等待；Level-2/3 把等待流提升为内部 Level-1 XQueue（§10.12） | — |
| **DEF-30**（P2，性能） | 短 CTA 路由放行毫秒级内核，5060 Lv3 HP P99 2.5–3.2 ms | 只按波数判定 | **已修复**：再加实测时长 ≤ 200 µs 条件（§10.12） | — |
| **DEF-31**（P1，正确性） | Lv3 恢复会重发非 killable 命令（整图执行两次） | “无驱逐”分支按只由 killable 内核更新的下标重发其后全部日志 | **已修复**：只重发 killable 内核与幂等命令（§10.12） | — |
| **DEF-33**（P1，崩溃） | 进程退出时静态析构把原 Trap 处理程序写回已拆除的上下文，libcuda 堆损坏（5080 LIB-12 退出时 18/20） | 每上下文单例只在静态析构时销毁 | **已修复**：单例表有意泄漏，退出时不调用驱动（§10.12） | — |
| **DEF-32**（P0，**静默数据错误**） | 整图发射的内核节点继承同函数上一次 Mode 5 发射的参数窗口，抢占后被驱逐 / 排空而不恢复（vLLM 不展平 0/32） | 参数窗口是每函数状态，图节点在发射时使用它 | **已修复**：整图发射前在 `LaunchMutex` 内清零图中函数的窗口（§10.12） | — |
| DEF-18（已修订） | 工作区 DEF-12 改动曾使 LIB-08 失败率升高（p=0.019） | 与 DEF-20 叠加所致；修复 DEF-20 后两版本 LIB-08 均 0 失败 | **结论修订**：DEF-12 改动在真实负载上优于 HEAD，应与 DEF-20 一同保留 | — |

---

## 1. 状态总表

| 编号 | 级别 | 类别 | 状态 | 影响 GPU | 首次发现 |
|---|---|---|---|---|---|
| DEF-14 | P0 | 崩溃 | 已由 DEF-20 修复（待提交） | 两卡 | Suite 7 |
| DEF-15 | P0 | 崩溃 | 已由 DEF-20 修复（待提交） | 两卡 | Suite 7 / Suite 1 |
| DEF-16 | P0 | **静默数据错误** | 已由 DEF-20 修复（待提交） | 两卡 | Suite 7 |
| DEF-17 | P1 | 性能 | 改善（默认去冗余重读） | 两卡（5080 更明显） | Suite 7 |
| DEF-18 | — | 回归风险判断 | 结论已修订（见 §2.3） | 5080 | LIB-08 二分 |
| DEF-19 | P2 | 偶发错误 | 已缓解（Cluster → Lv1） | 5080（5060 上该用例走 Lv1） | Suite 1 HW-06 |
| **DEF-20** | **P0** | **根因：寄存器保存缺失** | **已修复并验证，待提交** | 两卡 | 合成测试 + 探针 |
| DEF-20b | P3 | 残留边界 | 记录 | 两卡 | 代码审查 |
| DEF-21 | P1 | 恢复波次再抢占 | 已关闭（HEAD 缺陷由 DEF-12 修复；合成失败为 T-11） | 两卡 | 合成测试 / 服务测试 / DNN-06 |
| DEF-22 | P2 | 潜在（未触发） | 已修复 | 两卡 | SASS 审查 |
| DEF-23 | P2 | 显存过度预留 | 已修复 | 两卡 | arena 测量 |
| EXP-1 | — | 实验 Guardian v2 在多 Lv3 队列下偶发错误 | 仅实验开关，默认不启用 | 5080 | LLM-08 二分 |
| DEF-09 | P2 | CUDA Graph 图内抢占 | **已完成（Phase 3.5）** | 两卡 | HW-04 / DNN-07 |
| DEF-29 | P1 | 性能（跨流等待阻塞） | 已修复 | 两卡 | vLLM cProfile |
| DEF-30 | P2 | 性能（Lv3 尾延迟） | 已修复 | 5060 明显 | Lv3 价值扫描 |
| DEF-31 | P1 | 恢复时重复执行 | 已修复 | 两卡 | 代码审查 + vLLM 跟踪 |
| **DEF-32** | **P0** | **静默数据错误** | **已修复** | 两卡 | vLLM 不展平 |
| PERF-3 | P3 | 每次发射的占用率查询 | 已修复 | 两卡 | 代码审查 |
| DEF-33 | P1 | 退出时崩溃 | 已修复 | 5080 明显 | LIB-12 / gdb |
| OBS-1..20 | — | 观察，非缺陷 | 记录 | — | — |

---

## 2. 缺陷详细记录

### 2.1 DEF-20（P0）：Trap 引擎漏存每个内核最高的 1–3 个通用寄存器 —— DEF-14/15/16 的共同根因

**现象**
- 抢占（State-B 驱逐，即 CTA 正在执行内核正文时被换出）后恢复，内核结果错误或非法访存：
  - PyTorch `unrolled_elementwise_kernel<direct_copy_kernel_cuda>`（所有 dtype 转换，33 寄存器）：首次恢复即 `cuda error 700/717`，两卡 100% 复现（DEF-14）；
  - CUB Onesweep 排序（`torch.sort`，79 寄存器）、CUB 扫描（整型 `cumsum`）、`mbtopk::computeBlockDigitCounts`（`torch.topk`）、`index_add_int32`（5060）：崩溃（DEF-15）；
  - Triton 融合 RMSNorm（35 寄存器）：**不报错但输出错误**，5 轮仅 1–2 轮正确（DEF-16）；
  - `LIB-08`（NP=2）在 5080 上 15%–38% 偶发 700；`LLM-07 m5` 约 1/3 偶发 700。
- 合成内核（`test_convbar_p3.cu`）：**≤13 个寄存器的内核永远通过；用到 R14 及以上的内核都会失败**（15 寄存器 C1、16 寄存器 R1、26 寄存器 R2、55 寄存器 R3），单次抢占即可触发，错误线程在 lane / warp 间均匀分布、常见整 warp 全错（55 寄存器：364 个 warp 全错）。

**影响范围**：所有被 Lv3 插桩且寄存器数 ≥ 14 的内核（即几乎全部生产内核）。是否出错取决于被丢弃的最高寄存器在 Trap 时刻是否“活跃”，因此表现为随机、与算子相关、与抢占时机相关——这正是此前各种“P2 偶发”的来源。

**根因（已确认）**
1. Trap 引擎 Cave 2（保存）/ Cave 5（恢复）用 `S2R SR_REGALLOC` 决定要级联保存的最高寄存器：`r_top = SR_REGALLOC − 3`，`r_start = r_top & ~1`，然后以 64 位寄存器对从 `R(r_start):R(r_start+1)` 存到 `R14:R15`；`R0..R13` 由 112 B 硬件帧保存。原设计注释认为 `SR_REGALLOC` 额外包含硬件保留的 `R254:R255`。
2. **实测**（`repro/sr_regalloc_probe_*`：把 `S2R SR_TID.X` 二进制改写为 `S2R SR_REGALLOC`）：`SR_REGALLOC` **严格等于** `CU_FUNC_ATTRIBUTE_NUM_REGS`（16/20/25/45/69/125/206），普通上下文与 Lv3 下相同（证据：[`sr_regalloc_probe_5060.txt`](evidence/phase3_20261125/sr_regalloc_probe_5060.txt)、[`synthetic_and_bisect_results.md`](evidence/phase3_20261125/synthetic_and_bisect_results.md) §2）。
3. 因此原逻辑丢失：寄存器数 N 为偶数时 `R(N−2)、R(N−1)`；N 为奇数时 `R(N−1)`；N ≤ 16 时（插桩会把 <16 钳到 16）**R14、R15 全部丢失**。
4. 次级问题：若只改为 `SR_REGALLOC − 1`，N 为奇数时最后一对 `R(N−1):R(N)` 跨出分配边界，`R(N−1)` 仍出错（55、15 寄存器内核仍失败）。

**修复（工作区，未提交）**
- `sm120_trap_engine.cpp` Cave 2/5 头部：`SR_REGALLOC − 3` → **`SR_REGALLOC − 1`**；
- `sm120_guardian.cpp` 插桩时把寄存器数**向上取偶**（`<255` 时 `(max(raw,16)+1)&~1`），探针确认 `cuXtraSetLocalRegsPerThread` 会改变 `SR_REGALLOC`（25→26、45→46…），使寄存器对永不越界。

**验证**（证据：`evidence/phase3_20261125/` 中 `*after_fix*` 与 `validation_5080_after_fix_log.txt`）

| 测试 | 修复前 | 修复后（WT 与 VAR 两种构建） |
|---|---|---|
| Suite 7 Lv3（29 个生产算子 × 15 次中途抢占，逐调用比特级） | 5080 24/29，5060 23/29 | **两卡 × 两构建均 29/29** |
| LIB-08 CUB RadixSort NP=2（5080） | HEAD 34/40，WT 31/50 | **WT 30/30，VAR 30/30** |
| LLM-07 m5 LoRA 训练 + 解码（5080） | HEAD 2/3，WT 2/4 | **5/5，5/5** |
| Suite 1/4/6（5080） | — | 13/13、6/6、7/7（两构建） |
| 合成 R1/R2/R3/C1（单次抢占） | 失败 | 5/5–10/10 通过（两卡） |
| 稳态开销 | — | 修复前后差异 ≤ 0.1%（如 `gemm_bf16_4k` +1.4%→+1.4%） |

**剩余风险**：DEF-20b——255 寄存器内核无法取偶（每线程上限 255），其 `R254` 仍不会被保存；实测生产内核最高约 234 寄存器，未观察到 255 的情况。

**修复难度**：**容易**——已完成（3 行改动），需要：提交、全量回归、更新 v2.2.1 文档中“`SR_REGALLOC − 3` 扣除 R254:R255”的错误描述。

**为什么之前的测试没发现**：丢失的只是最高 1–3 个寄存器，只有当它们在 Trap 瞬间恰好活跃才出错；历史回归中的合成内核多为 <14 寄存器，生产内核测试中每个内核实例通常只被抢占 0–1 次，且很多内核最高寄存器只保存短生命周期临时值，因此长期表现为“偶发”。

---

### 2.2 DEF-21（P1，开放）：恢复波次进行中再次抢占（Preempt #2 mid-restore）

**现象**（DEF-20 修复后仍存在）

| 场景 | HEAD 恢复逻辑（VAR） | 工作区 DEF-12 逻辑（WT） |
|---|---|---|
| `DNN-06 torch.compile(inductor)`，5080 | **FAIL**（`max_err=0.875`，与上一会话 DEF-12 所修问题完全一致：丢失 CTA） | PASS |
| 真实模型服务 Lv3（5060，Qwen2.5-0.5B，20 次抢占） | **LP 迭代错误**：ppl 2/108、prefill 9/46 | PASS（0 错误） |
| 合成 C1（15 寄存器、60 CTA 单波次）2 次抢占 ×10，5060 | PASS 10/10 | **FAIL 0/10** |
| 合成 C1 2 次抢占 ×10，5080 | PASS 10/10 | PASS 10/10 |
| 12 个合成内核顺序执行、每个 3 次抢占（5060，2 轮） | 每轮 1 个早期内核失败（C0 / C1） | 每轮 1 个早期内核失败（C0 / C1） |
| 同上（5080） | 全部通过 | 全部通过 |

**触发条件**：同一内核实例在第一次恢复（`is_restore_wave=1`）尚未完成时再次收到抢占；Trace 显示第 2、3 次 reactivate 的 `restore_wave=1`。

**根因**：未定位。已知事实：HEAD 逻辑在恢复中途再抢占会清零 `active_kernel_idx` 并无条件清 `is_restore_wave`，导致后续排队内核篡夺活跃内核、恢复中的 State-C CTA 丢失（上一会话 DEF-12 已分析）；工作区 DEF-12 改动修复了该路径但在 5060（30 SM，60 CTA 恰好两轮驻留）上引入新的失败。假设（未验证）：WT 在“恢复中途再抢占 → 保留 `active_kernel_idx`”后，`Reactivate` 中 `!has_preempted && active_kernel_idx != 0` 时对后续命令的重发与恢复波次中 State-C/State-B 的二次分类存在竞态。

**复现**：`test_convbar_p3 C1 10 2`（5060，WT 构建）；`test_real_dnn_prod_p3.py dnn06`（5080，VAR 构建）；`bench_real_serving_p3.py run --mode lv3`（5060，VAR 构建）。诊断：`XSCHED_SM120_TRACE_LAUNCH=1` 可打印每次 reactivate 的 `cmd/evicted/drained/restore_wave/active_idx` 与在途命令序列。

**建议**：保留 WT 的 DEF-12 改动（在真实负载上明显优于 HEAD），把“恢复波次中的再抢占”作为独立状态（例如：恢复波次期间先等待恢复完成再驱逐，或对恢复中的 CTA 采用独立的二次驱逐计数与槽位）重新设计，并把 C1×2 抢占加入回归。

**修复难度**：**中–难**。需要理清多内核 FIFO、恢复波次、State A/B/C 与控制头字段在并发下的完整状态机；好在已有确定性复现用例与逐步 Trace 工具。

---

### 2.3 DEF-18（结论已修订）：工作区 DEF-12 改动是否应提交

- **修复 DEF-20 之前**：交错二分显示 LIB-08 失败率 HEAD 6/40（15%）vs 工作区 19/50（38%），Fisher 精确检验 p = 0.019，当时建议“暂不提交”。
- **修复 DEF-20 之后**：两版本 LIB-08 均 30/30；工作区版本在 DNN-06 与真实服务上优于 HEAD（见 DEF-21 表）。
- **修订结论**：LIB-08 的差异是 DEF-20（寄存器丢失）与恢复路径时序交互造成的；**DEF-12 改动应与 DEF-20 修复一同保留**，其剩余问题归入 DEF-21。

---

### 2.4 DEF-19（P2，开放）：CTA Cluster（2×1×1）+ DSMEM 偶发结果错误

- **现象**：Suite 1 `HW-06`（单波次 60 个 CTA ≤ 84 SM，因此在 5080 上被路由到 Mode 5）偶发 `mismatch=256/7680` 或 `512/7680`，即 1 个或 2 个 CTA（256 线程/CTA）的输出整体错误。
- **样本**：修复 DEF-20 前 HEAD 2/3、WT 2/4 通过；修复后 WT 9/10、VAR 6/10 通过。说明**与 DEF-20 无关**。5060 上 60 CTA > 30 SM 属多波次 Cluster，按设计路由到 Lv1，因此不触发。
- **根因**：未定位。假设（未验证）：Cluster 级状态——集群屏障（`barrier.cluster`）、DSMEM 对伙伴 CTA 共享内存的映射与可见性——不在 Trap 帧中；或恢复时同一 Cluster 的 CTA 未被保证协同调度。
- **缓解**：在自适应路由器中把 **所有** `ClusterSize > 1` 的内核路由到 Lv1 边界（目前只对多波次 Cluster 这样做），一行条件即可，代价是 Cluster 内核失去内核内抢占（RTX 50 上的 PyTorch 主流算子在本轮测试中未使用 Cluster）。
- **修复难度**：缓解**容易**；完整支持**难**（需逆向 Cluster 级硬件状态的保存/恢复）。

---

### 2.5 DEF-17（P1，性能）：Entry Guardian 每 CTA 开销

- **现象**（5080，无抢占稳态，相对 native）：`rope_apply` +101%、`transpose_copy` +82%、`sort_16m` +25%、`embedding` +23%、`gelu_tanh` +18%、`topk_sampling` +16%、`batchnorm_train` +7%、`swiglu` +7%；GEMM/注意力/卷积 +0.6%–3.5%；全部算子中位数 +1.7%。5060 上同类算子较轻（`transpose_copy` +29%、`rope_apply` +19%）。修复 DEF-20 前后不变。
- **原因**：Guardian 在**每个 CTA**入口读取控制头（`cmd`、`is_restore_wave`、`active_kernel_idx`）并执行 `BAR.SYNC 0` 做 CTA 级一致裁决；对 10⁵ 量级、每 CTA 只做几百纳秒工作的访存型内核，这部分固定开销被放大。单次调用延迟同样受影响（`rope_apply` 5 个内核/调用，+384 µs/调用）。
- **可选方向**：① Guardian 快速路径——无抢占时（`cmd==0 && !restore_wave`）由 warp 0 单次读取并以更轻量的方式广播；② 路由策略——对“CTA 数极多、单 CTA 极短”的内核直接走 Lv1（它们本身执行很快，Lv1 边界等待代价小）。
- **修复难度**：**中**。快速路径需保证 CTA 内各 warp 裁决一致（否则会出现部分 warp 运行、部分排空），必须配合合成测试与 Suite 7 验证。

---

### 2.6 DEF-22（P2，潜在）：`BinaryHasDeviceCalls` 漏检 `CALL.ABS`

- **现象**：`BinaryHasDeviceCalls` 只匹配低 12 位 `0x943/0x944`（`CALL.REL.NOINC` = `0x944`），而 `CALL.ABS.NOINC Rx` 编码为 `0x343`。含 `CALL.ABS` 的内核（`direct_copy` 8 处、`reduce_kernel` 4 处、`vectorized_gather_kernel` 1 处、`indexFuncLargeIndex` 1 处）插桩日志显示 `calls=0`、`lmem cap=0`，即不会为被调函数的栈帧预留保存区。
- **影响**：若 Trap 发生在 `CALL.ABS` 调用的函数内部且该函数使用了栈（`R1` 下移 + `STL`），恢复后栈内容丢失。当前观察到的 `CALL.ABS` 均位于断言失败路径（`LEPC` + `CALL.ABS.NOINC`），正常执行不会进入，因此**尚未触发**。
- **修复**：检测中加入 `0x343`（并审查其他调用形式）。**难度：代码容易**；但会让更多内核启用 Cave 7 栈保存（arena 变大、驱逐变慢），需要性能复测。

### 2.7 DEF-20b（已关闭，不可达）：255 寄存器内核的 R254

见 §2.1“剩余风险”。Phase 3.5 第二轮确认：ptxas 在 sm_120 上**从不分配 R253 / R254**，报告 255 寄存器的 272 个 PyTorch flash 内核与合成内核（`kernels/regpress255.cu`）的最高寄存器都是 `R252`。曾实现的 R254 保存已撤回。见 §10.13。

---

## 3. 观察记录（非缺陷，但影响结论或后续工作）

| 编号 | 观察 | 证据 / 说明 |
|---|---|---|
| OBS-1 | 5080 上 `linear_bwd_accum` / `batchnorm_train` 的 Lv3 进程整条作业摘要与 native 进程不同 | 两个 Lv3 进程彼此一致、DEF-20 修复前后一致（`263c1a…`、`67a205…`）；同进程 A/B 40/60 次连续调用逐次比特一致（`repro/lv3_drift_obs1.py`）。判定为进程级库启发式（cuBLAS/cuDNN 算法选择）差异，**不是数据损坏**；如需彻底排除可固定算法后复核 |
| OBS-2 | 5060 首轮出现 `cumsum@lv1`、`sort@shim` 摘要不一致 | 对照组 2 进程 × 3 轮复测 0/6 复现，记为瞬态 |
| OBS-3 | 单算子微基准中，**原生高优先级流**（无调度器）HP 延迟 P50/P99 中位数 0.11/0.42 ms，优于 Lv3 的 0.33/0.58 ms（5080） | 短 CTA 内核可被硬件按 CTA 粒度插队；此前评测只与 Lv1 对比。论文需加入该基线 |
| OBS-4 | Shim 直通开销 ~1–4 µs/调用；Lv1 ~8–17 µs/调用（工作线程转发） | Suite 7 单次调用延迟 |
| OBS-5 | PyTorch 2.11 在 RTX 50 上的 29 个主流算子 **100% 内核可被 Lv3 插桩**，无 mbarrier/Cluster/Cooperative 路由 | `XSm120GetRouteStats` |
| OBS-6 | 真实服务中请求粒度挂起使 LP 吞吐降至独占的 ~52%（Lv3）/ ~74%（Lv1） | 0.5B 小模型解码受 CPU 启动开销主导，整段挂起 LP 浪费 GPU 时间；应评估 token 粒度 / SLO 感知策略 |
| OBS-7 | Lv1/Lv3 下 HP 相对 solo 仍有约 +25% 延迟 | 测试为单进程双线程，LP 线程在 GPU 队列挂起时仍执行 Python 代码争用 GIL（假设，未验证）；应改为多进程复测 |
| OBS-8 | `SR_SMEMSZ` 包含 1 KB 系统保留区：编译器以 `%total_smem_size = SR_SMEMSZ − c[0][0x16c]` 计算，用户 SMEM 从 `0x400` 开始 | `repro/smem_size_probe.cu` + SASS；引擎 `SR_SMEMSZ − 0x400` 的拷贝长度正确 |
| OBS-9 | 控制头只有**一个**布局槽位，每次 `PrepareLaunch` 以流序 DMA 覆盖；恢复前排队内核会把它改写成自己的布局 | 正常恢复路径会重新 DMA 正确布局，不是缺陷；但任何在 Reactivate 时读取头部布局的诊断都会读错（本轮诊断工具已改为按内核记录重算） |
| OBS-10 | `CALL.ABS` 返回地址由 `LEPC`（运行时 PC）计算，内核搬移（插桩复制到 `ep_inst+0x500`）是安全的 | SASS 审查 |
| OBS-11 | Trap 屏障 `bar_sync_trap()`（`w1=0x000fc00000008000`）是硬件专用 `BAR.SYNC.TRAP`，`nvdisasm` 无法解码 | 不与用户命名屏障 0–15 冲突 |
| OBS-12 | 收敛屏障编号（B0–B5 嵌套）本身不是触发条件；`BMOV.32(.CLEAR) B0..B15` 手写编码 32 条全部正确 | `test_convbar_p3` D1–D6 全部通过；`nvdisasm` 核验 |
| OBS-13 | 用户内核的越界写可以破坏 XSched 的设备端控制结构（与同进程任何设备分配相同，属固有风险） | 自写微基准 `out[1024+b]` 越界（T-16，Phase 3.3） |
| OBS-14 | 同时驱动桌面的 GPU 上，Lv3 运行期间桌面图形进程（`ghostty` / `gnome-shell` / `Xwayland`）偶发 Xid 31 MMU 故障，进程均恢复；计算型旁观进程未受影响 | 内核日志与测试日志逐条对时（Phase 3.4-2）；Lv3 在 `KERNEL_EXEC_TIMEOUT != 0` 的 GPU 上启动告警 |
| OBS-15 | 暂停的 Level-1 发射线程在命令日志满、等待空位后不再检查暂停，仍发射一条命令（Level-3 下被 Guardian 立即排空） | 补做检查的改动经交替 A/B 无收益，已回退（Phase 3.4-2） |

---

## 4. 测试方法自身的问题与修正（避免误判，如实记录）

| # | 问题 | 后果 | 修正 |
|---|---|---|---|
| T-1 | 逐调用校验最初用 `torch.ne(o, gold).sum()`，对 bool 求和会先启动 bool→int64 的 `direct_copy` 转换核 | 5 个“算子崩溃”中有 4 个其实是校验代码里的转换核被驱逐（DEF-14），误归因到 `transpose/embedding/conv1x1/triton` | 改为无类型转换的 `ne().any()` + `logical_or`；转换核单独作为算子 `cast_bool_to_i64` 测试 |
| T-2 | FP32 CUB 前缀和（`cumsum`）不是逐位确定的（look-back 合并顺序依赖时序） | 所有模式包括 native 都“失败” | 改为 INT32 输入 |
| T-3 | 0 维张量（`cross_entropy` 输出）无法 `view(uint8)` 计算摘要 | 该算子所有模式报错 | `reshape(-1)` 后再计算 |
| T-4 | “独占作业耗时”包含首次运行的预热 | 作业膨胀率出现 −53% 的无意义值 | 增加一次不计时预热，取两次最小值 |
| T-5 | 服务测试 HP 负载过高（请求耗时 ~250 ms、到达间隔 150 ms，HP 占空 97%） | LP 保留率指标无意义（881% 等） | 改为 16 token、间隔 500 ms（占空 ~30%），LP 指标改为“占独占吞吐百分比” |
| T-6 | 困惑度 LP 用 4×1024 token，FP32 logits 达 2.3 GB | 8 GB 的 5060 上 OOM | 改为 2×512 |
| T-7 | 第一版驱逐现场诊断从控制头读取布局 | RMSNorm 的“PC”实为浮点数据（布局被排队内核覆盖，见 OBS-15） | 改为按活跃内核记录重算布局并使用抢占时保存的 arena 地址 |
| T-8 | `SR_REGALLOC` 探针第一版：编译器把被改写的 `S2R` 结果复用到 `threadIdx.x==0` 判断 | 读到全 0 | 只用 `blockIdx.x` 作写入条件 |
| T-9 | 本地 `smem_size_probe` 在 0 字节动态 SMEM 时写共享内存 | illegal address | 仅在 `dsz>0` 时写入 |
| T-10 | 跨进程摘要比较受库启发式影响（OBS-1） | 一度被怀疑为 Lv3 数据错误 | 增加 native-vs-native 对照与同进程 A/B |

---

## 5. 环境与流程问题

| # | 问题 | 处理 |
|---|---|---|
| E-1 | 之前会话把 sudo 口令写在远端命令行中，共享服务器其他用户可通过 `/proc/*/cmdline` 看到 | 改为经 ssh 标准输入传给 `sudo -S`（`p3_guarded_run.sh` 以 root 运行整份计划） |
| E-2 | 5080 为多用户共享服务器 | 守护运行器在**每个**测试前检查 GPU 计算进程，有他人作业则等待/跳过；本轮全部计划 0 次等待 |
| E-3 | 5080 的 CUDA 12.9 工具包缺少 cuBLAS 头文件 | 使用 pip `nvidia-*` 轮子中的头文件与库（`build_p3.sh`） |
| E-4 | 5080 HF 缓存中的 safetensors 是指向缓存外的符号链接，`rsync -a` 复制成悬空链接 | 改用 `rsync -L` 单独复制该 blob |
| E-5 | 本地重启后 `/tmp` 下的构建目录丢失 | 重新构建到会话临时目录 |
| E-6 | 5060 是桌面卡，桌面进程常驻于 compute-apps 列表 | 本地运行使用 `SKIP_GUARD=1` |
| E-7 | 安装目录被正在运行的测试进程动态链接 | 测试计划运行期间不重建/重装对应安装目录 |

---

## 6. 调查时间线（假设 → 实验 → 结论）

1. Suite 7 发现 Lv3 下多个算子崩溃 → 复现脚本单独运行 `torch.sort` 不抢占全部通过 → **问题需要抢占**。
2. `CUDA_LAUNCH_BLOCKING=1` 下通过 → **与时序有关**。
3. Reactivate 遥测：所有失败都伴随 `evicted>0`（State-B 驱逐）。
4. 怀疑“在途命令（memset）在驱逐与恢复之间执行” → 遥测显示活跃内核之后无在途命令 → **否定**。
5. 发现 4/5 崩溃的被驱逐内核是校验代码中的 `direct_copy` 转换核 → 修正校验方法（T-1）；最小复现：`x.to(int64)` 必崩，`x+1.0` 不崩。
6. 怀疑代码搬移（BRX/JMX/CALL.ABS）→ 无 BRX/JMX，CALL.ABS 用 LEPC → **否定**（但发现 DEF-22）。
7. 怀疑 BMOV B0–B15 编码 → nvdisasm 32 条全对 → **否定**。
8. 怀疑 HP 干扰 → 无 HP 同样崩溃 → **否定**。
9. 怀疑部分退出 / 发散 warp → `test_partial_exit_p3` 全过 → **否定**。
10. 怀疑在途加载 → 通过的 bf16 内核同样在等待 LDG 时被 Trap → **否定**。
11. 怀疑 SMEM 记账 → 实测 `%total_smem_size` 与编译器语义 → **否定**。
12. 怀疑高编号收敛屏障（Trap 现场在 `BREAK.RELIABLE B6`）→ D1–D6 嵌套深度扫描全过，但 C1 失败 → **否定屏障编号**，发现 C1 需多次抢占、寄存器 15 个。
13. 寄存器压力扫描：≤13 寄存器全过、≥14 均会失败 → 指向 R14+ 的 Cave 2/5 级联。
14. `SR_REGALLOC` 探针：等于寄存器数而非 +2/+3 → **根因 DEF-20 确认**。
15. `−3`→`−1`：偶数寄存器内核全过，奇数仍失败 → 发现寄存器对越界 → 寄存器数取偶 → 全部合成用例通过。
16. 真实算子：29/29 两卡两构建通过；LIB-08 60/60、LLM-07 10/10。
17. 剩余：C1 两次抢占（5060，WT）、DNN-06 与真实服务（VAR）→ 归入 DEF-21；HW-06 → DEF-19。

---

## 7. 工作区代码改动清单（均未提交）

| 文件 | 改动 | 性质 |
|---|---|---|
| `platforms/cuda/hal/src/level3/sm120_trap_engine.cpp` | Cave 2/5 `SR_REGALLOC − 3` → `− 1`（DEF-20）；新增 `DebugDumpEvicted()` | **修复** + 诊断 |
| `platforms/cuda/hal/src/level3/sm120_guardian.cpp` | 插桩寄存器数取偶（<255）（DEF-20）；插桩日志加内核名/二进制大小；`XSCHED_SM120_DUMP_DIR` 导出原始 SASS | **修复** + 诊断 |
| `platforms/cuda/hal/include/.../sm120_trap_engine.h` | `DebugDumpEvicted()` 声明 | 诊断 |
| `platforms/cuda/hal/src/level3/cuda_queue.cpp` | `XSm120GetRouteStats()` 导出与路由计数；`XSCHED_SM120_TRACE_LAUNCH` 启动追踪与 reactivate 遥测 | 诊断（默认关闭，计数为 relaxed 原子操作） |
| 上一会话遗留：`cuda_command.h`、`event_pool.*`、`cuda_queue.cpp`（Reactivate）、`sm120_trap_engine.cpp`（DEF-12）、`shim.*`、`intercept.cpp` | DEF-11/12/13 修复 | 建议与 DEF-20 一同提交（见 §2.3） |
| 上一会话遗留：`shim/.../context_registry.{h,cpp}` | 未被引用的半成品（context-isolation 合并中断） | 需决定完成或删除 |
| 新增测试 | `bench_prod_ops_p3.py`、`bench_real_serving_p3.py`、`test_partial_exit_p3.cu`、`test_convbar_p3.cu`、`repro/*`、`p3_guarded_run.sh` | 测试资产 |

两卡均以 `-Wall -Wextra -Werror` 零警告构建。

---

## 8. 未完成事项与建议顺序

1. **提交 DEF-20 修复 + DEF-11/12/13**，并在两卡全量回归（Suite 1/4/5/6/7 + 合成用例 + LIB-08 ×30），更新 v2.2.1 技术手册中关于 `SR_REGALLOC` 的错误描述（含 `.docx`）。
2. **DEF-21**：重新设计恢复波次再抢占状态机；回归用例 `test_convbar_p3 C1 10 2`（5060）、`dnn06`（5080）、`bench_real_serving_p3 --mode lv3`（5060）。
3. **DEF-19**：先把所有 Cluster 内核路由到 Lv1（缓解），再研究 Cluster 级状态。
4. **DEF-17**：Guardian 快速路径或“短 CTA 内核走 Lv1”的路由策略。
5. **DEF-22**：补充 `CALL.ABS` 检测并复测性能。
6. 评测补充：原生高优先级流 / MPS / Green Context 基线；HP/LP 多进程服务测试；token 粒度挂起策略。
7. 处理 `context_registry.{h,cpp}` 遗留文件。


---

## 9. Phase 3.2 / 3.3 更新（2026-11-25 晚）

完整设计、测量与回归数据见 [`SM120_V2_PHASE3_2_3_3_REPORT_ZH.md`](SM120_V2_PHASE3_2_3_3_REPORT_ZH.md)。此处记录与缺陷相关的结论。

### 9.1 DEF-21 关闭
- 重新审视：上一轮“工作区 DEF-12 逻辑在 5060 上 C1 两次抢占 0/10、HEAD 逻辑 10/10”的结论不成立。失败在进程内全有或全无、**零抢占的轮次同样失败**、错误总是网格尾部连续一段 CTA——这是合成测试自身的上传竞态（T-11），每种构建实际上只有 1 个进程样本。
- 修正测试后：36 个进程（np∈{0,2,3}）0 失败；专项压测（多波次网格、4–6 内核排队、300 µs–1 ms 间隔、每轮 8–10 次抢占）约 3,000 次抢占全部比特一致，约一半命中恢复波次，覆盖 120–147 个不同内核位置。
- HEAD 旧逻辑的真实缺陷（DNN-06 丢 CTA、真实服务 LP 错误）由已提交的 DEF-12 逻辑修复。

### 9.2 DEF-19 缓解、DEF-22 / DEF-23 修复
- DEF-19：`ClusterSize > 1` 的内核默认路由到 Level-1；`XSCHED_SM120_CLUSTER_MODE5=1` 可恢复实验性单波次 Mode 5。
- DEF-22：调用检测加入 `CALL.ABS.NOINC`（`0x343`）；`XSCHED_SM120_DETECT_CALL_ABS=0` 可关闭。
- DEF-23：LMEM 保存区 = 无调用时静态局部大小；有调用时静态局部 + `CU_LIMIT_STACK_SIZE`（默认 1024 B，栈超过此深度内核自身会出错）。LayerNorm / RMSNorm arena 230.9 → 102.6 MiB；含 `CALL.ABS` 的 `reduce_kernel` 等首次获得栈保存区（arena 25.6 → 102.6 MiB，为正确覆盖调用栈所需）。

### 9.3 DEF-17 与实验 Guardian v2（EXP-1）
- 对照实验：插桩后强制走 Guardian 旁路时开销为 0%，说明开销全部来自 Guardian 逻辑；原手写控制字使每条 `LDC/S2R/LD` 串行等待，且上万 CTA 争用同一控制头缓存行。
- **默认**：保持串行发射，去掉 slot 28 对 `cmd` 的冗余重读（读到 `cmd==0` 的 CTA 必在 `cmd` 发布前驻留，脉冲在发布后才触发）。5080：`rope_apply` +101% → +56.5%、`transpose_copy` +82% → +53.9%、`embedding` +22.8% → +1.9%、`gelu_tanh` +18.3% → +0.3%、`sort` +24.9% → +15.7%；5060：`transpose_copy` +28.5% → +18.1%、`rope_apply` +18.5% → +15.8%、`sort` +14.8% → +13.4%。5080 上主要成本是上万 CTA 争用同一控制头缓存行，去重读即拿到几乎全部收益。
- **EXP-1（实验 v2）**：再把控制头读取与 `S2R/LDC` 链改为异步发射（SB2–SB5），开销再降一些（`rope` +13.2%），但 LLM-08（两条 Lv3 队列 + HP）失败 11/61，串行发射 0/78；二分显示失败需要异步发射窗口（与去重读无关）。推测为 Trap 在 Guardian 异步指令未完成时送达、其目的寄存器 R0–R10 恰为 Trap 帧保存/暂存寄存器所致（未经硬件实验证实）。仅保留为 `XSCHED_SM120_GUARDIAN_V2=1`。
- 踩坑（T-12）：编辑控制字时把 `ISETP` 的 stall 从 15 改成 5，`@P0 BRA` 读到旧谓词，全部 CTA 误入恢复路径退出；改为“保留原控制字、只追加等待位”（`with_extra_wait()`）。

### 9.4 新增测试方法问题

| # | 问题 | 后果 | 修正 |
|---|---|---|---|
| T-11 | 合成测试用同步 `cudaMemcpy`（可分页内存）上传输入后在非阻塞流上启动内核，二者无序 | 网格尾部 CTA 读到旧输入，伪造出“DEF-21 失败” | `H2D()`：拷贝后 `cudaDeviceSynchronize()` |
| T-12 | 手改指令控制字时缩短了固定延迟指令的 stall | 分支读到旧谓词 | 只追加等待位，保留原 stall |
| T-13 | `rsync --exclude 'build*'` 也排除了 `build_regress_bins.sh` | 5080 首次回归缺少测试二进制 | 改为只排除目录 `build*/` |
| T-14 | 回归计划用 `^@register` 匹配算子，漏掉缩进在 `if HAVE_TRITON:` 内的 Triton RMSNorm | 首轮回归 S7 缺 1 个算子 | 正则允许前导空白；补跑该算子（两卡 4 模式均通过） |
| T-15 | 失败按进程“全有或全无”时，每种构建只跑 1 个进程就做 A/B | 得出错误的 A/B 结论 | 对按进程出现的失败，以进程为样本单位并扩大样本 |


---

## 10. Phase 3.3 收尾更新（2026-11-26）

完整数据见 [`SM120_V2_PHASE3_2_3_3_REPORT_ZH.md`](SM120_V2_PHASE3_2_3_3_REPORT_ZH.md) §3.6、§4、§5、§8。

### 10.1 DEF-17 短 CTA 路由
- 分析 Suite 7 全部内核的 SASS：开销最高的全是无回边的逐元素内核（`elementwise_kernel<128,4>`、`vectorized_elementwise_kernel`，1.6–8.8 万个 CTA）。
- 新规则：无回边、无间接跳转、无调用、网格 2–1024 波的内核走 Level-1 边界（`XSCHED_SM120_SHORT_CTA_LV1`，默认开）。`BRA` 偏移编码由 3,653 条分支拟合验证。
- 结果：5080 `transpose_copy` +53.6% → −0.1%、`rope_apply` +56.6% → +3.2%；5060 +17.9% → +0.3%、+17.6% → +1.9%；全部比特一致。代价：被路由内核的 HP 延迟上限为其自身时长（5060 `transpose_copy` HP P50 1.09 → 2.37 ms）。
- 机制套件（合成、Suite 1/4、验收组）关闭路由运行，以保持 Mode 5 覆盖（T-18）。

### 10.2 DEF-24 间接调用漏检
- 用含 `switch` 跳转表与函数指针的内核确认编码：`BRX` = `0x949`，`CALL.REL.NOINC Rx` = `0x344`，`RET` = `0x950`。原 `BinaryHasDeviceCalls` 漏掉 `0x344`，此类内核不会获得调用栈保存区（与 DEF-22 同类）。已修复，尚未在生产负载中触发。

### 10.3 PERF-1 每调用开销
- 直接提交（工作线程休眠时由提交线程发射，仅 CUDA 内核）、调用方 `WaitAll`（调用线程自行同步硬件流，挂起时回退）、后台批同步前自旋、控制头布局缓存（跳过重复的 36 B HtoD）。
- 顺带修正 `SyncAllWithLock`：原实现把同步后的**全部**在途命令标为完成，在直接提交下会误标同步窗口内新发射的命令，现只完成同步前已发射的命令。
- C++ 微基准（~300 µs 内核 + 50 µs 主机间隔）：5060 Lv1/Lv3 +10.4/+11.7 → +2.0/+3.1 µs；5080 +6.5/+7.8 → +1.9/+3.0 µs；Lv3 每次发射 4.10 → 2.1 µs。紧密循环下 5080 Lv3 仍为 +5.2 µs。
- 被否决：调用线程在 `WaitAll` 后立即发送 Idle 事件（调度器 IPC 进入关键路径，反而变慢），已回退。

### 10.4 新增观察与测试方法问题
- **OBS-13**：用户内核越界写可破坏 XSched 的设备端控制结构（控制头/CTA 表与用户分配同处显存），表现为后续 CTA 在 Guardian 中退出。属固有风险，排查“Lv3 下内核结果异常”时应先排除越界写。
- **T-16**：自写微基准的越界写（见 OBS-13）曾被误判为 Lv3 缺陷。
- **T-17**：首次发射的插桩时间（约 19 ms/内核）计入平均值，误判为“事件计时偏差”。
- **T-18**：生产路由策略会把机制测试的被测内核移出 Mode 5，机制套件因此关闭路由运行（首轮 `LIB-06` 失败起初归因于此，关闭路由后仍失败，实际原因为 T-19）。
- **T-19**：`LIB-06` 的 LP 工作只有一次 cuFFT（< 1 ms），而测试在 800 µs 后才抢占；提交路径加速后 LP 先结束，`preempts=0/2` 判失败。修正：LP 与参考均做 4 次原地前向 FFT、按位比较；该测试文件 23 处可分页 H2D 拷贝补设备同步（T-11 同类）。

### 10.5 PERF-2 透传发射开销
- 现象：Suite 8 中 `request` 策略在 HP 请求期间已完全挂起 LP，5080 上 HP TPOT 仍为无 XSched 时的 1.23×；同一进程内、LP 线程启动之前测得的 solo TPOT 已慢同样幅度。5060 上仅加载 shim（无任何 XQueue）decode 每步慢约 1.3 ms。
- 根因（`perf`）：`cuXtraKernelGetFunction` 占 decode 运行全部样本的 8.4%，在每个内核命令构造函数中执行（包括无 XQueue 的透传发射）；sm120 透传另有每次的 `cuXtraSetDebuggerParams` 清零。
- 修复：透传命令不再在构造时解析 `CUfunction` 与参数个数（`FuncHandle()` 按需解析）；按发射句柄登记哪些函数持有非零 Mode 5 参数，仅对这些函数清零（检查与发射仍在 `LaunchMutex` 内）；`DirectLaunch` 的上下文/架构查询按线程缓存。`XSCHED_SM120_ALWAYS_CLEAR_PARAMS=1` 可恢复每次清零。
- 结果：5060 进程内 decode TPOT 8.61 ms（无 XSched）→ Lv1 8.76 / Lv3 8.72 / 仅 shim 8.94 ms（+1.3–3.8%）；修复前仅 shim 9.77 vs 8.41 ms（+16%）。
- 教训（T-20）：单一小内核的微基准不能代表真实框架的发射开销——驱动对同一函数句柄的转换有缓存，而真实 decode 每步发射数百个不同内核；应以真实负载 `perf` 采样为准。

### 10.6 收尾补充（2026-11-26 下午）
- **自适应直接提交**：单内核调用的紧密循环中工作线程常醒，原规则（仅在工作线程休眠时直接提交）下 5080 Lv3 为 +5.2 µs。改为“上一次调用（两次 `WaitAll` 之间）只有 1 个命令时也直接提交”，5080 Lv1/Lv3 → +1.2 / +2.4 µs，5060 → +1.1 / +2.2 µs；多命令调用保持交接，`sync4` 不变。
- **MEM-1 arena 空闲回收**：`CudaQueueLv3Trap::Synchronize()` 后调用 `TrimArenas()`，在“同步期间无新发射且无待恢复驱逐状态”时释放增长替换下来的旧 arena，并在 64 次同步窗口内持续超大（> 2× 需求且 > 64 MiB，`XSCHED_SM120_ARENA_TRIM_MIN_MB`）时释放当前 arena。回归计划新增 1 MiB 阈值下的抢占测试以覆盖释放路径。
- **空间共享实验**（报告 §5.3）：用 CUDA green context 把 LP 限制在部分 SM 上，HP 独占其余 SM。原负载下 5060（LP 24/30 SM）LP 79–80%、TPOT P50 不变，但 P99 为 solo 的 1.44×；5080（LP 64–72/84 SM）LP 86–87%，但 P50 +27–43%（HP 可用 SM 不足）：green context 只划分 SM，不划分显存带宽（推测为 LP 的访存密集阶段拖慢 HP decode；GIL 切换间隔已排除）。
- **DEF-19 完整支持的方案**（未实现，Phase 3.4）：Cluster 状态（集群屏障相位、DSMEM 映射）无法从 Trap 帧保存与恢复，可行路线是“Cluster 级排空”——Guardian 以 `barrier.cluster` + DSMEM 广播 rank 0 的决定，使整个 Cluster 一致地排空或运行；已运行的 Cluster 不驱逐，抢占延迟上限为单个 Cluster 时长。
- **Lv1 / Lv3 的 LP 吞吐相对 `1 − 占空比` 上限**：Lv1 挂起只停止发射新命令，在途命令继续进入 HP 请求，因而略高于上限（5060 71–72% vs 65%）；Lv3 立即驱逐在途内核并承担驱逐/恢复成本，低于上限约 4–7 个百分点。属于挂起语义差异，不是缺陷。

### 10.7 `BAR.RED` norm_gate（2026-11-26 晚，默认开启）
- slot 39 改为 `BAR.RED.OR 0, P0`（`P0 = tid==0 && cmd!=0`）+ `B2R.RESULT` → `P1`，仅当 `P1` 为真时读取 `entry_dec`。正确性依据：`entry_dec` 只由本 Guardian（此时归约为真）或 Trap 处理程序写入，而 Trap 处理程序从不完成该屏障；State B 恢复后 `B2R` 值不确定但两条路径都正确。`restore_path` / `replay_state_c` 各后移一槽（0x2d0 / 0x3c0），锚点改为 Guardian 与 Trap 引擎共用的运行时函数；默认布局字节不变。
- 验证：5060 40/40、5080 62/62（`LLM-08` 36/36），全部比特一致。5060 `sort` +14.8% → +8.0%、`batchnorm` +7.0% → +3.8%；5080 `sort` +15.6% → +14.8%、`batchnorm` +4.5% → +2.1%。5080 剩余开销主要来自控制头首读的 L2 热点。
- 开关：`XSCHED_SM120_GUARDIAN_BARRED=0`；`XSCHED_SM120_GUARDIAN_V1=1` 时自动关闭。

### 10.8 门控镜像（gate mirror，2026-11-27，默认开启）
- 定位：测量探针（把 Guardian 的两次控制头读取换成常数，会使抢占失效）表明，5080 上剩余开销主要来自数千个 CTA 串行两次读取同一条控制头缓存行（纯 Mode 5 `rope_apply` +54% → +5%）。
- 实现：控制分配前缀 256 × 128 B 镜像行 `{cmd, is_restore_wave}`；Guardian 每个 CTA 读自己 SM 的行（一次 `LD.E.64`）；主机严格按 header 的写入者维护镜像（`PreemptStream` 只写 `cmd`；`PrepareRestore`、256 线程条件清零内核、`ResetControlSync`、重分配写全部字段）。
- **T-21**（开发中发现并修复）：最初 `PreemptStream` 按读到的旧 `is_restore_wave` 写镜像，条件清零内核随后清零 header 时镜像留下过期 1，压测 4/4 非法指令；一致性检查器定位后修复，6/6 + 5060 44/44 + 5080 68/68（`LLM-08` 36/36）通过。
- 结果（纯 Mode 5）：5080 `sort` +14.8% → +3.3%、`topk` +9.9% → +4.8%、`rope_apply` +54.1% → +7.3%、`transpose_copy` +53.4% → +1.1%；5060 `transpose_copy` +18.9% → +3.5%。关闭短 CTA 路由时纯 Mode 5 也满足 DEF-17 目标。

### 10.9 DEF-10 非 root Level-3（Phase 3.4-1，2026-11-26，已修复）
- 范围确认：非 root 进程中 Trap 处理程序补丁、Guardian 插桩、控制内存全部正常，只有 BAR0 驱逐脉冲失败（退化为等待内核结束，挂起 9–16 ms，结果仍正确）。
- 修复：`xsched-bar0d` 代理。root 启动，只映射 Trap 寄存器与刷新寄存器所在的两页 BAR0，建 `root:<group> 0660` 的 `SOCK_SEQPACKET` socket 后降权为 `nobody`（`CapEff=0`、`NoNewPrivs=1`）；只接受 `PULSE` / `HELLO`，每连接令牌桶限流，单线程串行保证多进程脉冲不交错。客户端（`Sm120Arena`）先尝试直接映射，失败再连代理，`HELLO` 校验 PCI BDF，断线后自动重连。
- 验证：非 root 5060 37/37、5080 50/50（`LLM-08` 24/24、双进程并发、压测、库与硬件特性、8 个生产算子比特一致）；故障注入（无代理、无权限、限流、接错 GPU、路径过长、运行中杀掉并重启代理）均按预期退化或恢复。开销：往返 4.5–12 µs，同会话 A/B 挂起延迟 +2–15 µs。
- 详见 [`SM120_V2_PHASE3_4_DEF10_BAR0_PROXY_ZH.md`](SM120_V2_PHASE3_4_DEF10_BAR0_PROXY_ZH.md)。

### 10.10 Phase 3.4-2 多进程测试（2026-11-26）
- **OBS-7 证实**：HP / LP 分为两个进程后，按请求挂起的 HP TPOT 两卡均为 solo 的 1.00×（单进程 1.02–1.07×），残余膨胀来自 GIL。
- **DEF-25（撤回）**：曾因同步后立即查询仍为 Ready 而判定“调用方 `WaitAll` 快速路径漏报 Idle”。实际由发射线程在约 21 µs（最大 41 µs）后完成转换。为它加的“调用线程直接置 Idle”打破了命令缓冲区“Idle 时最后命令是已完成 `WaitAll`”的约定，导致 `HW-05` 断言失败（两卡），5080 `S8-ppl-lv3-prefill` 还出现 1 次 LP 不一致；已回退，`HW-05` 与该 S8 配置复测通过（T-22）。回归保留 `SYN-xq-idle`（检查 50 ms 内变为 Idle）。
- **DEF-28（已修复）**：`xserver` 只在 SIGINT / SIGQUIT 时删除 IPC 消息队列（SysV 键 0x1）。root 回归运行后残留的队列使普通用户的 `xserver` 无法启动通道，客户端向满队列发送时阻塞，跨进程测试挂起。修复：处理 SIGTERM；测试脚本用 SIGINT 结束服务并预检残留队列（T-23）。
- **DEF-26（已修复）**：全局 / 本地调度器在其执行线程上挂起 / 恢复队列，该线程无 CUDA 上下文；Lv3 `Reactivate()` 重新发射内核时 `PrepareLaunch` 断言失败。修复：Lv3 的 `Deactivate/Interrupt/Reactivate` 临时压入队列上下文。
- **DEF-27（已缓解，开放）**：跨进程透明调度下 HP 在 LP 被驱逐期间继续发射，GPU 时间片切换若在 LP 线程束处于 Trap 处理程序时到达会超时（Xid 109 `CTX SWITCH TIMEOUT`，约 0.3%/次挂起），驱动恢复使两个进程都报 719。应用层“等驱逐确认再发射”（`req-lv3`）不触发。缓解：`XSCHED_SCHEDULER=GLB` 时默认走 Level-1（`XSCHED_SM120_GLB_LV3=1` 覆盖）。
- **OBS-14**：RTX 5060 同时驱动桌面，两次 Lv3 多波次压测开始后约 1 s 内桌面图形进程（`ghostty` / `gnome-shell`）出现 Xid 31 MMU 故障，进程未崩溃；早先的旁观测试只覆盖计算上下文。
- 详见 [`SM120_V2_PHASE3_4_MULTIPROCESS_ZH.md`](SM120_V2_PHASE3_4_MULTIPROCESS_ZH.md)。

### 10.11 DEF-27 根本修复：就绪门控（2026-11-26）
- 机制：GLB 下客户端为每个 Idle→Ready 转换编号，该命令走发射线程并等待服务端确认；`SchedOperation` 携带 `ready_ack`（服务端已处理的该进程 Ready 数）；`xserver` 每轮先发挂起操作，其中挂起 Level-3 队列的进程等其 `OperationCompleteEvent`（有界 100 ms）后再放行其他进程（Level-1/2 挂起不等待）。Lv3 的挂起返回前已等到驱逐完成，所以 HP 不会在 LP 处于 Trap 时提交工作。
- 默认：`XSCHED_SCHEDULER=GLB` 时客户端与 `xserver` 默认开启（`XSCHED_GLB_READY_GATE=0` 关闭）；开启时 Lv3 在 GLB 下不再回退到 Level-1。
- 验证：RTX 5080 无门控 3 次 TPOT P99 3.1–3.5×（P99 19–22 ms），有门控 10 次 P99 1.10–1.29×、0 Xid；RTX 5060 有门控 6 次 0 次 Xid 109（无门控时 4/4 次运行出现）、P99 1.06–1.21×；HP 门控等待均值 160–580 µs，0 次超时，HP 请求 380/380 一致、LP 0 错。RTX 5060 全量 288/288、最终构建 10/10；RTX 5080 全量 283/288（5 项为共享主机环境原因），最终构建复跑 15/15。
- 对 Level-1 无可测影响（`glb-lv1` 8×8 交替 A/B：P50 0.968× / 0.969×，P99 区间重叠）。
- OBS-15：暂停的 Level-1 发射线程在命令日志满、等待空位后不再检查暂停，仍会发射一条命令；补做检查的改动经交替 A/B 无收益（`request` 无差别，`glb-lv1` 略差），已回退，记录为观察项。
- OBS-14 第三、四次：5060 门控测试中 `Xwayland`、`gnome-shell` 各出现一次 Xid 31（进程均未重启）；Lv3 现在在显示 GPU（`KERNEL_EXEC_TIMEOUT`）上启动时告警。
- 测试安全（T-24）：5080 上另一用户运行着自己的 XSched `xserver`（同一 IPC 键）。root 启动的服务端会删除其队列并占用其端口；多进程测试脚本改为只要键 0x1 的队列属于他人就拒绝启动（包括 root），回归在 GPU 与 IPC 都空闲后才开始。


### 10.12 Phase 3.5：第一梯队（2026-11-26）

完整报告：[`SM120_V2_PHASE3_5_TIER1_ZH.md`](SM120_V2_PHASE3_5_TIER1_ZH.md)；原理：原理文档 §3.5（DEF-31）、§5（DEF-30）、§5.1（DEF-09）、§5.2（DEF-29）、§7.3（DEF-32）。

- **DEF-09（已完成）**：图展平（`shim/src/graph_flatten.cpp`）。实例化时拓扑排序、参数字节拷贝；修改即失效；`XSCHED_CUDA_GRAPH_FLATTEN`（0 / 1 默认 / 2）、`_MIN_US`（按首发时长）。整网捕获训练步 40 次重放 15 次挂起比特一致，挂起 P99 407 ms → 0.4–1.4 ms；DNN-07、HW-04 图内抢占。代价 OBS-17。
- **DEF-29（已修复）**：`cuStreamWaitEvent(普通流, XQueue 事件)` 不再阻塞调用线程（`shim/src/event_bridge.cpp`）。Level-1 录制队列：record 发射后原生等待；Level-2/3：等待流提升为内部 Level-1 XQueue（继承优先级提示，随流排空销毁）。`XSCHED_CUDA_ASYNC_EVENT_WAIT=2` 不安全模式在挂起下 30/30 次损坏、无挂起 0/20（证明主机确认的必要性）。被否决的“GPU 等主机标志”设计与 `cuModuleLoadData` 的驱动锁死锁（gdb 栈确认）。
- **DEF-30（已修复）**：短 CTA 路由加实测时长条件 `XSCHED_SM120_SHORT_CTA_MAX_US`（默认 200 µs）。5060 GEMM / SDPA 挂起 P99 1.94 / 2.53 → 0.22 / 0.25 ms。
- **DEF-31（已修复）**：`Reactivate()` 只重发 killable 内核与幂等命令（`RelaunchableAfterTrap()`）。由代码审查与 vLLM 跟踪发现，合成用例未能单独触发，作为防护保留并入回归（`SYN-graph-relaunch-*`）。
- **DEF-32（已修复，P0）**：整图发射的内核节点在发射时使用函数当前的 Guardian 参数窗口；同函数先前的 Mode 5 eager 发射留下的窗口让图节点被当作那次发射驱逐 / 排空且不再恢复。跟踪：恢复时 `active_idx=4231`（日志为 6157–6167）、`drained=35832`。修复后 vLLM 0.5B / 7B 不展平 32/32、排空 8 个 CTA。“捕获开始时清零”无效（节点不在捕获时固化窗口），已删除。
- **PERF-3（已修复）**：占用率查询按几何形状缓存。
- **OBS-16**：同进程 HP 走高优先级流时，LP 留有 SM 空位则 HP 共驻，native 的 HP P99 已低（0.7–1.8 ms）；LP 占满 SM 时 native 随 LP 内核时长线性增长（12.7 / 26.6 ms），Level-3 保持 0.8–2.6 ms。
- **OBS-17**：展平每节点约 5 µs CPU；0.5B decode（每步约 380 个微秒级内核）吞吐 −36%，7B decode −20%，prefill −10%–−16%。
- **OBS-18（测试方法）**：vLLM 自己维护线程局部的当前流；7B 模型的稳态 decode 曾在 vLLM 新建的流上运行、完全绕过 XQueue。测试脚本经 vLLM 替换后的 `set_stream` 设置流并断言 `main_stream`。
- **DEF-33（已修复）**：5080 `S4-LIB-12` 进程退出时 `malloc(): invalid size`（单独重复 18/20；提交版本 f29aab3 同样 5–6/8，Phase 3.4 回归中通过，属潜在缺陷）。gdb：`~Sm120TrapEngine` → `RestoreOriginal` → `RestoreAll` → `cuXtraMemcpyHtoD` → libcuda `calloc`，发生在静态析构中、上下文可能已被运行时 / cuFFT 的退出处理拆除之后。Trap 处理程序在每个上下文自己的 ToolsMemory 中，退出时无需恢复：引擎、Guardian 注册表、`LaunchMutex` 表与图计划表改为有意泄漏。修复后 0/20（路由关闭）、0/10（开启）。
- **OBS-19**：5060 全量回归中 `S7-linear_bwd_accum-lv3m5` 一次 `cuda error 719`（Xid 43，重跑 0/20），见 OBS-20。
- **OBS-20（开放）**：5060 两次全量回归中 3 个进程在启动时出错，都紧跟一个带抢占的 Level-3 进程：两次 native 金标准不确定（尚无 XQueue），一次 Xid 43 + 719。受控复现 0/54（`probe_native_det_p35.py`）。可能与 DEF-33（退出时写入已拆除的上下文）同源，未证实；DEF-33 修复后的 5060 全量回归 295/295 中未再出现。
- **OBS-21（开放）**：5080 最终回归 `S8-prefill-lv3-slo` 一次运行 156 个 LP 迭代中 1 个不一致（HP 20/20；该次 TTFT P99 50 ms 异常）；重跑 10/10 通过，不经过跨流等待提升路径；累计约 17 次运行、2,700 个 LP 迭代中 1 次。
- 最终构建两卡全量回归：5060 **295/295**，5080 **294/295**（OBS-21），均无新增 Xid。
- **T-25**：从未 record 的 CUDA 事件查询为“完成”，计时探针必须先确认事件已 record（DEF-30 与按时长展平的第一版都因此失效）。
- **T-26**：PyTorch 流池中的流是 non-blocking 的，在默认流上生成的输入必须先同步再在其他流上使用；否则 GPU 争用时金标准不确定，整进程“全部损坏”。

### 10.13 Phase 3.5 第二轮：透明部署流程（2026-11-27）

完整报告：[`SM120_V2_PHASE3_5_TIER1_ZH.md`](SM120_V2_PHASE3_5_TIER1_ZH.md) §8。

- **DEF-36（已修复）**：自动 XQueue（`XSCHED_AUTO_XQUEUE=1`）为每个创建的流建队列，PyTorch 每进程预建 128 条池流 → 128 个 XQueue；`xserver` 每次挂起 LP 的全部 Level-3 队列，空闲队列也发 BAR0 脉冲（`Interrupt()` 只在最后一个内核不可 Trap 时跳过）。透明 vLLM `glb-lv3`：LP 1/11 批错误、HP E2E P50 79.8 ms（比 native 差）、门控超时。修复：流第一次提交时才建 XQueue（`XSCHED_AUTO_XQUEUE_LAZY`，默认开），空闲队列（发射线程已暂停、`cuStreamQuery` 空闲）不发脉冲。
- **DEF-37（已修复）**：传统默认流上的工作永远绕过 XSched。shim 计数（新增导出 `XCudaShimStats`）显示 3B LP 的 46% 命令在默认流上，LP 挂起后仍与 HP 时间片轮转，HP 46 ms（独占 30.5）。修复：自动 XQueue 模式下默认流的异步工作转到每个上下文一个带 XQueue 的阻塞代理流（`XSCHED_AUTO_XQUEUE_LEGACY`，默认开）；57 个同步拷贝 / 置零 API 先排空阻塞 XQueue；应用一旦创建自己的阻塞流即退回原路径（传统流与阻塞流的隐式同步）。新测试 `test_auto_legacy_p35.py`（默认流 PyTorch：GEMM 链、整数 cumsum、`.item()`、pinned 拷贝、`wait_stream`、事件，挂起 / 恢复约 450 次，60 步全部一致）。修复后两个 vLLM 进程：HP E2E P50 29.9 ms（独占 30.5）、LP 吞吐 88–92%，两侧输出全部一致。
- **DEF-34（硬件限制，检测 + 可选路由）**：SM 的寄存器文件分配到最后一个寄存器时 Trap 不被接受（空闲 1,024 个寄存器时 45/59 驱逐，空闲 0 个时 0/61；与每线程寄存器数无关）。挂起退化为等内核结束，数据正确（108 次运行 0 错）。按几何形状检测并计入 `rf_full`；`XSCHED_SM120_RF_FULL=lv1` 路由到 Level-1。受影响：cutlass INT8 GEMM（Suite 7 `int8_mm`）、vLLM Marlin AWQ GEMM（255 × 8 线程束 × 1 CTA）、`_topk_topp_kernel`、`_penalties_kernel`。
- **DEF-35（已修复）**：循环检测漏掉统一谓词分支 `BRA.U`（`0x547`），这类循环被判为 loop-free（影响短 CTA 路由）。Suite 7 49 个内核分类不变。
- **DEF-20b 关闭**（见 §2.7）。
- **DEF-38（已修复）**：同一进程内多个 Level-3 队列各自挂起 / 恢复时 GPU 卡死（`test_multiqueue_lv3_p35.py` 4 队列、约 6,600 次脉冲 / 秒：5080 5/8 次挂死、GPU 100%、所有线程停在 `cuStreamSynchronize`；2 队列 0/8；5060 触发 Xid 109）。看门狗转储（`XSCHED_SM120_WATCHDOG_MS`）：三个队列停在 `PreemptStream` 第 4 阶段（`cmd=5` 已写入控制头，但门控镜像的跨步 memset 没有完成，脉冲尚未发出），其中两个 `evicted=0`；第四个停在 `ResetControlSync` 的 memset。机理：脉冲同时打断一个 SM 上的全部线程束，各线程束随即读 cmd；**队列 B 的脉冲恰在队列 A 的 cmd 写入落地时到达**，A 的同一 CTA 中一部分线程束读到 0（返回用户代码，停在用户 `BAR.SYNC`），另一部分读到 5（停在处理程序的 `BAR.SYNC.TRAP`），CTA 卡住、占着 SM。单队列时“先写 cmd、同步、再发脉冲”保证不会发生；A 自己的脉冲本可再次打断用户态线程束使 CTA 汇合，但它发不出去（DEF-39）。修复：引擎级脉冲互斥（`PreemptStream()` 整段与 `PrepareRestore()` 持有），发脉冲前等待同上下文其他队列的 Mode 5 恢复完成（`XSCHED_SM120_CTX_PULSE_GATE`，默认开）；跨进程用主机级 `flock`（`/dev/shm/xsched-sm120/<PCI>.pulse`）串行；新增停滞看门狗与导出 `XSm120DumpStreams()`。结果：门控开启 26/26 通过，关闭 13/23 挂死，只串行化 10/10 通过。门控关闭时的看门狗转储：一个队列 83 个 CTA 已保存、54 个已排空，正好少一个 SM 上的 CTA，8 次恢复脉冲后仍不结束——卡住的 CTA 不能靠脉冲修复，竞争必须预防。
- **DEF-39（已修复）**：控制流中的跨步 memset（门控镜像 `cuMemsetD2D32Async`）、32 KB / 1 MB memset 与设备内拷贝由**SM 上的内核**执行（`kernels/ce_vs_sm_probe.cu`：SM 全部被占时这三类操作阻塞，主机↔设备拷贝与小 memset 不阻塞）。因此 SM 被卡住时 `PreemptStream` 无法发出脉冲、也无法发恢复脉冲（DEF-38 的卡死因此是永久的）；平时每次挂起也要等某个 SM 有空位。修复：门控镜像改为固定内存的 2D 主机→设备拷贝（只写 cmd 字或 {cmd, is_restore_wave} 两字），`ResetControlSync` 改为主机→设备拷贝清零，且只清自上次复位以来发射过的 CTA 范围；自上次复位以来未发生脉冲 / 恢复时直接跳过（`XSCHED_SM120_CE_CONTROL`，默认开）。

- **DEF-40（已修复，P0 安全）**：BAR0 脉冲打断**发出时驻留在 GPU 上的任何上下文**。`test_foreign_ctx_p35.py`：同一 5080 上的原生 GEMM 进程在 Level-3 挂起压力下 2/2 次被杀死（CUDA 719，受害者通道 Xid 109），对照组无误。桌面 Xid 31（`faulted @ 0x0`，OBS-8 / OBS-14）、5060 桌面冻结与 13:13 共享 5080 上的 Xid 109 都与此一致：没有 XSched 处理程序的上下文被打断。修复：任何级别的 XQueue 都为其上下文安装处理程序并在 `/dev/shm/xsched-sm120/` 登记；NVML 每 100 ms 列出 GPU 上的进程，存在未登记进程时不发脉冲（挂起退化为 Level-1；`XSCHED_SM120_FOREIGN_CTX=allow` 放行）。5060 桌面上默认即禁止脉冲（列出 gnome-shell、Xwayland、nautilus、ghostty），回归环境显式放行。
- **DEF-41（已缓解，默认安全）**：同一 GPU 上**两个同时运行 Level-3 队列的进程**：`test_multiqueue_lv3_p35.py --procs 2 --queues 2`（两个进程同步开始各自的挂起压力）14/14 次运行都有一个进程收到 Xid 109 `CTX SWITCH TIMEOUT`、随后 CUDA 719 与 SIGSEGV；主机级脉冲锁关闭 6/6、开启 8/8，没有区别。这是 DEF-27 的机理：两个进程是两个分时的上下文，时间片切换恰在一方的线程束位于 Trap 处理程序中时到达便超时。就绪门控只覆盖“等挂起完成再发射”的 Level-1 对端（HP 进程），对两个对等的 Level-3 进程（例如 `xserver` 下的两个 LP 租户）不起作用；串行化脉冲也无法阻止上下文切换。缓解：DEF-40 的租户登记中，运行 Level-2/3 内核的进程另建 `.lv3` 标记；GPU 上存在另一个 Level-3 进程时不发脉冲，双方都退化为 Level-1 语义（`XSCHED_SM120_LV3_PEERS=allow` 放行）。支持的部署形态：每块 GPU 一个 Level-3 进程（其余进程 Level-1，经 GLB 就绪门控），或所有 Level-3 工作在同一个进程内（DEF-38 门控）。
- **第一梯队 4（驱动鲁棒性）**：引擎初始化失败（不认识的驱动布局）时原先会在 `PrepareLaunch` 的 `XASSERT` 处中止进程；现在路由到 Level-1 边界（`lv1_no_engine`），并以 `XSCHED_SM120_SIMULATE_UNSUPPORTED_DRIVER=1` 验证（`SYN-no-engine`）。
- **OBS-20（已查明，T-29）**：根因是测试中的流竞争，不是 XSched。`SYN-event-bridge-lv1` 的金标准 `gold = Y.clone()` 在传统默认流上异步执行，紧接着下一轮 `Y.zero_()` 在非阻塞的 LP 流上执行，二者没有顺序约束；GPU 有争用时清零抢在拷贝之前，金标准成了全零（打印出 `gold.abs().max() = 0.0`）。复现：同一 GPU 上并行一个原生 GEMM 负载，原生（不加载 XSched）18/25、带 shim 22/25 次“不确定”；在 clone 后加一次同步：0/20（未修复 20/20）。`S7-cumsum_64m` 两次“整数 cumsum 金标准不确定”同理：`golden_once()` 在 LP 流同步之后、于默认流上 clone 输出并返回，输出张量随即释放回 LP 流的缓存池，第二次金标准运行在 LP 流上复用这块内存（CUB 扫描临时存储），覆盖了尚未完成的拷贝；负载下旧版 9/10、修复后 0/10。修复：两处改为在 LP 流上 clone（或 clone 后同步），并加固 `test_real_llm_prod_p3.py` 中两处同类写法。第三次（`S7-linear_bwd_accum-lv3m5` 启动约 1 s 内 Xid 43 + 719）为单次事件，复跑 0/20，未能归因。
- **OBS-21（开放）**：交替 A/B（当前构建 vs f29aab3）各 30 次 `S8-prefill-lv3-slo`，共 9,635 个 LP 迭代 0 错。累计约 1/12,000，原因未定。
- **OBS-14 扩展（环境）**：RTX 5060 驱动桌面，高频抢占（多队列约 6,600 次 / 秒、flash 压测约 11,000 次 / 秒、`SYN-stress-g8`）时出现 Xid 109（`CTX SWITCH TIMEOUT`），桌面进程同时 Xid 31；2026-11-27 一次桌面冻结后整机重启。与 DEF-27 同一机理，但另一方是不受 XSched 管理的桌面。无显示的 5080 上，同样的测试只在另一用户的作业同时运行时出现过一次 Xid 109。
- **T-26（再次出现两次）**：`test_regpress255` 中 `out.zero_()` 在默认流上、内核在非阻塞流上，出现“整批错误”；跟踪显示该批次没有任何驱逐才定位为测试问题。测试改为在内核流上清零。
- **T-27**：首版合成测试 `regpress255` 用 `__launch_bounds__` 时 nvcc 忽略 `-maxrregcount`（寄存器数不变）；以宏去掉 launch bounds。
- **环境（共享 5080）**：另一用户以数秒间隔连续提交短作业，守护脚本只在每个条目开始前检查，曾有作业在测试中途启动（同时出现我方 Xid 109）。`p3_guarded_run.sh` 改为：条目开始前要求 GPU 连续空闲 `QUIET_S`（60 s）；运行中每秒轮询，出现非本测试进程树的计算进程时立即终止本测试（`ABORTED_FOREIGN_GPU_JOB`）；`GUARD_IGNORE` 排除本地桌面进程。**T-28**：该监视曾三次在测试打印 PASS 之后“停止”测试——本测试进程退出时 `nvidia-smi` 仍短暂列出它（`[No data]`，`/proc` 中已不存在、无法追溯父进程），被误判为外来作业；已离开 `/proc` 的进程现在不计入。

### 10.14 Phase 3.5 第三轮：残留问题收尾（2026-11-27 晚）

- **OBS-20 已查明**：见 §10.13（T-29，测试中默认流 clone 与非阻塞流复用的竞争）。
- **T-30（测试方法，严重）**：`test_multiqueue_lv3_p35.py` 与第一版 `test_lv3_peers_glb_p35.py` 的负载 `t.addmm_(A, t, beta=.5, alpha=.5)`（`A ~ randn/n`）每步把 `t` 减半，约 150 步后全为 0，之后的“结果一致”只是比较全零；且 GEMM 的输出与输入 `mat2` 是同一块内存，早期步骤本身就不确定。因此 DEF-38 的**卡死**证据有效，但此前的**正确性**证据是空的。改为无别名、有界的混沌递推（`u.addmm_(A, t)`、`u.tanh_()`、交换 t/u，A 放大到 1.5/√n），12 个检查点逐一比较并要求均值 > 0.05。重新验证：4 队列门控开启 6/6、两个进程各 2 队列 2/2，均为有效检查点、无误。
- **DEF-40 检测空窗**：改为**每次发脉冲（含恢复脉冲）前同步查询 NVML 进程列表**（5080 每个列表约 6 µs，5060 约 50 µs；已登记租户的分类缓存 200 ms），空窗由最长约 100 ms 缩短为检查与 BAR0 写入之间的几微秒；`XSCHED_SM120_FOREIGN_SYNC_CHECK=0` 退回轮询状态。S8 TTFT P50 7.48–7.91 ms（之前 7.39–7.84 ms），外来进程受害者测试 2/2 不受影响。
- **DEF-41 修复（不再只是禁用脉冲）**：(1) `xserver` HPF 策略的 Level-3 对等轮换：同一设备最高优先级上只有一个进程的 Level-3 队列处于恢复状态，每 `XSCHED_HPF_LV3_QUANTUM_US`（20 ms）在有就绪队列的进程间轮换，其余进程的 Level-3 队列（含空闲的）保持挂起（`XSCHED_HPF_LV3_EXCLUSIVE=0` 关闭）；(2) 每个 Level-3 进程在租户登记的 `.lv3` 文件中发布一个字节（'1' 可能有 Level-3 工作在 GPU 上，'0' 全部挂起并排空）；(3) 对端为 '0' 时允许发脉冲。`test_lv3_peers_glb_p35.py`（两个 Level-3 LP 进程 + 一个 Level-1 HP 进程，`xserver`）：轮换开启 3/3、关闭 1/1 通过，每个 LP 发出 505–988 次脉冲，LP 检查点全部一致，HP P99 1.47–1.60 ms，Xid 0（修复前同类配置 14/14 次 Xid 109）。
- **DEF-42（已修复）**：GLB 客户端在进程退出期间仍执行 `xserver` 发来的操作（轮换使服务端定期发送）：Resume → Reactivate → `cuStreamSynchronize` 在驱动已拆除后返回 `CUDA_ERROR_DEINITIALIZED`，`CUDA_ASSERT` 在工作线程中调用 `exit()`，而主线程已在 `exit()` 中等待该线程（死锁，gdb 确认 `__exit_lock`）；5080 上表现为进程退出时 SIGSEGV / SIGABRT。修复：调度器工作线程启动时注册 atexit 处理函数，进程开始退出后丢弃服务端操作；`CUDA_ASSERT` 遇到 `CUDA_ERROR_DEINITIALIZED` 只记一次日志、不再退出；同时修正 `CUDA_ASSERT` 在 `cuGetErrorString` 失败时打印未初始化指针（日志中的乱码）。
- **OBS-21（仍未复现，已排除多种机理）**：(a) 恢复脉冲路径：测试开关强制恢复脉冲 / 在驱逐与排空进行中额外注入脉冲（`XSCHED_SM120_TEST_EXTRA_PULSES`），flash、GEMM、255 寄存器、多队列、S8 共约 3,700 次额外脉冲，结果全部一致；(b) 驱动内部的 memset / 设备内拷贝内核在脉冲下：1 GiB memset + 拷贝、约 13,000 次脉冲，逐字节一致；(c) S8 本身无 T-29 类竞争。OBS-21 那次运行 TTFT P99 50 ms（平常 8–11 ms）与“有外来上下文同时运行”相符：当时守护脚本只在条目开始前检查，另一用户的短作业可能在运行中启动；受控实验中（同卡运行一个原生 GEMM 进程、放行脉冲）出现 Xid 109 与一次 4.3 s 的停顿、TTFT P99 约 25 ms，但 LP 结果的错误未能复现（约 350 次 LP 迭代）。现在 DEF-40 在有外来上下文时不发脉冲、守护脚本在外来作业出现 1 s 内停止测试，这一最可能的机理已被阻止；此后未再出现。
- **OBS-23（新，已加防护）**：网格极大的无循环内核（1 GiB 逐元素 / 归约，约 50 万个 CTA）在极高挂起频率下没有前进：每次挂起要让剩余 CTA 全部经过 Guardian 排空，每次恢复要重新发射整个网格让已完成的 CTA 跳过；挂起间隔短于这一循环时内核永远完成不了（256 MiB、约 0.4 ms 一次挂起：120 s 无进展；1 GiB、2 ms：慢约 10 倍）。防护：同一内核连续 `XSCHED_SM120_STARVATION_LIMIT`（8）次在挂起时仍在执行，则下一次挂起不发脉冲、在 Level-1 边界等它完成（路由统计 `starvation_lv1`）。结果：256 MiB / 0.4 ms 5.1 s 完成，1 GiB / 0.4 ms 19.9 s，结果逐字节一致；常规压力测试不受影响。
- **第三轮最终回归**：RTX 5080 全量 **288/288**（Level-3 脉冲有效；含 `SYN-lv3-peers-glb`、`SYN-driver-kernels`、改正后的 `SYN-multiq-lv3`、`SYN-event-bridge-hazard` 实际演示出危害），运行期间无 Xid。RTX 5060 全量 287/288：桌面 GPU 上 DEF-40 不发脉冲，`SYN-event-bridge-hazard` 无法演示危害，现改为“未发脉冲则 SKIP”（复跑 SKIP）。
- **环境**：5080 在回归结束 31 分钟后被 `unattended-upgrades` 重启（升级 `nvidia-container-toolkit`，驱动未变 580.95.05），`/tmp` 下的构建与安装目录被清除，已重建；重启后冒烟 3/3 通过、Xid 0。与本项目无关。
- **MPS：跨进程冲突的结构性方案（已验证）**：两个 Level-3 进程作为同一 MPS 服务器的客户端（共用一个硬件上下文、无时间片切换），放行对等进程且不轮换：8/8 次运行、每进程约 3,800 次脉冲、全部一致、0 Xid（无 MPS 时 14/14 Xid 109）；`xserver` 下关闭轮换 2/2 通过。默认设置下（DEF-40/41 识别 MPS：同一 MPS 服务器的客户端之间不适用对等规则，MPS 服务器进程本身不算外来）：两客户端各约 3,400 次脉冲 3/3 通过、`xserver` 无轮换 2/2 通过、原生 MPS 客户端不受影响。**非 XSched 的 MPS 客户端仍会被脉冲打到**（放行时 Xid 13，并使整个 MPS 服务器退出），DEF-40 检查仍必需。回归 `SYN-mps-lv3`（测试自带前台 MPS 守护进程与私有目录）：5080 2/2、5060 1/1 通过。

