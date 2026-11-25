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
| **DEF-17**（P1，性能） | CTA 数多且单 CTA 极短的访存算子 Lv3 稳态开销最高 +101% | Entry Guardian 每条指令串行等待 + 热点控制头的冗余重读 | **改善**：默认去冗余重读（5080 `rope_apply` +101% → +56.5%、`embedding` +23% → +1.9%；5060 `transpose_copy` +28.5% → +18.1%）；更激进的异步版本为实验开关（§9.3） | 剩余部分**中–难** |
| **DEF-22**（P2，潜在） | 设备函数调用检测漏掉 `CALL.ABS` | 只匹配 `0x943/0x944`，`CALL.ABS.NOINC` 编码为 `0x343` | **已修复**（§9.2） | — |
| **DEF-23**（P2，显存） | 含设备调用的内核每线程固定预留 4 KiB 栈保存区（LayerNorm arena 231 MiB） | 预留值固定为 `max(static, 4096)` | **已修复**：按“静态局部 + `CU_LIMIT_STACK_SIZE`”精确预留（§9.2） | — |
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
| OBS-1..12 | — | 观察，非缺陷 | 记录 | — | — |

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

### 2.7 DEF-20b（P3，残留）：255 寄存器内核的 R254

见 §2.1“剩余风险”。可选修复：对 255 寄存器内核在 Cave 2/5 单独以 32 位存取 `R254`（需确认 R254 是否被 Trap 机制保留）。

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
| T-7 | 第一版驱逐现场诊断从控制头读取布局 | RMSNorm 的“PC”实为浮点数据（布局被排队内核覆盖，见 OBS-9） | 改为按活跃内核记录重算布局并使用抢占时保存的 arena 地址 |
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
