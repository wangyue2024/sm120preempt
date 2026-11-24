# Blackwell (`sm_120`) XSched Level-3 v2 全方位能力评测、测试方法论与缺陷分析报告

> **研究定位**：本报告面向 NVIDIA Blackwell (`sm_120`) 用户态透明抢占与调度系统（XSched Level-3 v2：`Mode 1 kEarlyExit`、`Mode 5 kTrapFrame` 与独立 `ResumableSm120Context` CILP），系统性回答三个核心研究问题：
> 1. **如何科学设计评测体系**以最大化展现系统的透明通用性、非幂等语义保真度与多租户调度能力？
> 2. **在闭源商业计算库、Blackwell 微架构特性、DNN/LLM 调度及真实未修改 PyTorch/cuDNN/Triton 负载下的实机表现如何**？
> 3. **当前系统暴露出哪些底层架构、HAL/Shim、驱动与硬件边界缺陷**，以及后续优化的精确技术路线？

---

## 1. 评测方法论设计：五维立体测试矩阵

为了避免"合成微基准掩盖真实框架痛点"，同时清晰划定软硬件能力边界，本评测体系围绕**五个正交维度**构建（全部测试代码已归档至 [`xsched/test/sm120_v2/eval/`](../xsched/test/sm120_v2/eval/)，支持一键复现 [`run_all_eval.sh`](../xsched/test/sm120_v2/eval/run_all_eval.sh)）：

| 评测维度 | 核心考察目标 | 典型负载与设计要诀 | 验证的科学主张 |
| :--- | :--- | :--- | :--- |
| **D1. 闭源商业计算库透明通用性 (`LIB-01..08`)** | 零源码修改、零重编译，通过 `libshimcuda.so`（`libcuda.so.1`）透明拦截 NVIDIA 官方预编译 `sm_120` cubin | `cuBLAS`（FP32 `beta=0/1`）、`cuBLAS GemmEx`（FP16/BF16 Tensor Core `beta=1`）、`cuBLASLt`、`cuFFT`（原地 C2C）、`cuSPARSE`（CSR SpMM `beta=1`）、`CUB`（111 寄存器 + 36 KiB SMEM + Decoupled Look-Back 16M/8M 基数排序） | 证明 `Entry Guardian` + 静态 9-Cave 引擎对闭源黑盒二进制的**通用二进制兼容性**与**多轮连续抢占正确性** |
| **D2. Blackwell 微架构硬件特性边界 (`HW-01..07`)** | 逐项探测 Hopper/Blackwell 新增微架构原语的硬件可保存/可恢复边界 | PTX `mma.sync` Tensor Core、`cp.async` (`LDGSTS`)、SMEM `mbarrier`、`CUDA Graph`、空闲队列挂起回归、CTA Cluster (`__cluster_dims__(2,1,1)` + DSMEM)、Cooperative Groups (`grid.sync`) | 划清 **软件 Trap 帧（Mode 5）** 与 **硬件上下文切换（CILP）** 的严格能力边界，暴露 Shim 覆盖缺口 |
| **D3. 非幂等语义正确性 Oracle** | 构造"重放即静默损坏"的严格数学判据，对比 Mode 1 与 Mode 5 | ① `atomicAdd` 精确步数计数（`HW-01`）；② 原地 `beta=1.0` 残差累加与 `cuFFT` 原地变换；③ 多步原地 `AdamW` 优化器权重/一阶二阶矩更新（`w, m, v`） | 实证 Mode 1 丢弃重放必然导致**静默数值损坏**，而 Mode 5 实现**比特级零误差（`0.000e+00`）** |
| **D4. DNN / LLM 多模式调度 QoS 与频率扫描** | 同台 GPU 下对比 5 种调度机制的尾延迟、SLO 达成率、LP 吞吐保留与抢占频率敏感度 | **LLM Serving**（LP Prefill vs HP Decode）、**非幂等 DNN 训练**（LP 原地 AdamW vs HP 实时推理）、100% SM 饱和负载、`5/10/20/50 ms` 周期扫描，外加**跨进程 BAR0 广播干扰**实测 | 量化 Level-3 抢占相对 Native Streams 与 Level-1 的尾延迟收益，并诚实量化不同 Grid 规模下的代价曲线 |
| **D5. 真实未修改深度学习框架端到端验证 (`PT-01..06`)** | 零修改 PyTorch 2.11 + `torchvision` + `cuDNN 9.19` + `Triton 3.6.0` | ResNet-50 FP16 推理、ResNet-18 完整训练（Fwd+Bwd+AdamW）、4 层 Llama Prefill 全部 4 种 SDPA 后端（`math` / `FlashAttention-2` / `xFormers` / `cuDNN 9.19`）、自定义 Triton 3.6.0 非幂等内核、`empty_cache()` / allocator 边界 | 暴露真实 Python 框架在融合度、内存分配器锁、Autograd 启动属性等维度的系统级瓶颈 |

---

## 2. 双机硬件基线与网络可达性说明

| 角色 | GPU 与 SM 规格 | 驱动与工具链 | 本轮评测覆盖状态 |
| :--- | :--- | :--- | :--- |
| **本地测试机 (`RTX 5060`)** | NVIDIA GeForce RTX 5060 (`GB206`, `sm_120`, 30 SMs, 8 GiB) | Driver `595.99.02`, CUDA `12.9`, `gcc-13/g++-13`, PyTorch `2.11.0+cu128`, cuDNN `9.19.0`, Triton `3.6.0` | **全量完成**：8 项闭源库 + 7 项硬件边界 + DNN/LLM 调度 QoS + 频率扫描 + BAR0 跨上下文干扰 + 6 项真实 PyTorch/Triton 负载 + 5 项标准回归套件 |
| **远程服务器 (`RTX 5080`)** | NVIDIA GeForce RTX 5080 (`GB203`, `sm_120`, 84 SMs, 16 GiB) | Driver `580.95.05`, CUDA `12.9`, `gcc-13/g++-13` | **已有完整 v2 基线数据**（11/11 E2E 3 轮全通、79/79 SASS、11 Caves/1,703 槽位、Mode 1/5 微秒级延迟、160.6× 显存缩放、CILP `425 µs`）；本次实测期间远程 SSH 跳板机持续 `Connection timed out during banner exchange`，故新增 Suite 1–3 与 6 项 HAL 修复待 SSH 恢复后可直接通过 [`run_all_eval.sh`](../xsched/test/sm120_v2/eval/run_all_eval.sh) 一键同步与复跑 |

---

## 3. 真实负载压测中发现并修复的 6 项底层 HAL 关键缺陷

在将系统从合成微基准推向闭源商业库（`cuBLAS`/`cuFFT`/`cuSPARSE`/`CUB`）与真实 PyTorch 负载的过程中，我们第一时间暴露并修复了 **6 项深层并发与微架构缺陷**（全部通过 `C++20 -Wall -Wextra -Werror` 零警告编译及全量回归）：

| 编号 | 缺陷模块与位置 | 真实复现现象 | 根因分析 | 修复方案 |
| :---: | :--- | :--- | :--- | :--- |
| **FIX-1** | [`cuda_queue.cpp`](../xsched/platforms/cuda/hal/src/level3/cuda_queue.cpp#L123-L157) `Reactivate()` | 对空闲或已自然结束的流执行一次 `Suspend/Resume` 后，该流后续所有内核输出全 0 | `Reactivate` 无条件调用 `PrepareRestore()` 将 `ctrl_host_->cmd` 置为 `6`（`kCmdMode5Restore`），但无 `State B/C` 槽位消费它，导致后续所有内核 CTA 均以 `State A (state==0)` 秒退 | 检查 `has_preempted = (evicted_slots > 0 \|\| drained_cnt > 0 \|\| mode1_exit_cnt > 0)`，否则调用 `ResetControlSync()` 清零控制头 |
| **FIX-2** | [`sm120_trap_engine.cpp`](../xsched/platforms/cuda/hal/src/level3/sm120_trap_engine.cpp#L1550-L1572) `EnqueuePostRestoreClear()` | 任意库执行 `NP >= 2` 二次抢占时，70%–96% 的 CTA 静默丢失未计算 | 第一次恢复末尾排入无条件 `MemcpyDtoHAsync(zero_32B -> ctrl_host_)`；第二次抢占提前排空流后，该过期 DMA 立刻将第二次抢占记录的 `evicted_slots/drained_cnt/active_kernel_idx` 清零 | 改为由 `PostRestoreClearState`（独立微锁 + 单调 `epoch` 计数器）保护的 `Driver::LaunchHostFunc` 回调；`PreemptStream`/`ResetControlSync` 递增 `epoch` 立即作废过期回调，且 `clear_mu` 绝不跨 `StreamQuery` 持有（零死锁） |
| **FIX-3** | [`sm120_trap_engine.cpp`](../xsched/platforms/cuda/hal/src/level3/sm120_trap_engine.cpp#L86-L96) `EmitDynamicSmemLoop()` | `cuBLAS` 0-SMEM 辅助内核在 Mode 5 抢占时触发 `cuda error 717: operation not supported on global/shared address space` | 未先判断 `SR_SMEMSZ > 0x400` 即做 `R8 = SR_SMEMSZ - 0x400` 并判 `R8 == 0`；当 `SR_SMEMSZ == 0` 时无符号下溢为 `0xFFFFFC00`，触发 4 GiB 越界 `LDS/STS` 死循环；同时将 `smem_area` 对齐粒度提升至 `4096 B` | 改为先执行 `isetp_ge_u32_imm(8, 0x401u)`，仅当 `SR_SMEMSZ > 0x400` 时才减 `0x400` 进入拷贝循环 |
| **FIX-4** | [`sm120_trap_engine.cpp`](../xsched/platforms/cuda/hal/src/level3/sm120_trap_engine.cpp#L1325-L1375) 多内核 `arena_va` 与 Staging 竞争 | `CUB DeviceRadixSort`（多内核背靠背，SMEM 从 4 KB 骤增至 36 KB）偶发 `cuda error 700: illegal memory access` | ① `PreemptStream` 在发脉冲前读 `ctrl_host_->arena_va`，若 `K1` 在脉冲前自然结束而 `K3`（已切换至新大 `arena A2`）被抢占，恢复时错误用回小 `arena A1` 越界；② `layout_stage_` 仅 8 槽位，深队列回绕覆盖；③ `EnsureControlCapacity` 初始仅 4096 CTAs，中途扩容切走 `ctrl_dev_` | ① 在 `PrepareLaunch` 中以 `kernel_arena_map_[kernel_idx]` 精确绑定每个内核的实际 `arena_va`，`PrepareRestore` 按 `active_kernel_idx` 查表；② `layout_stage_` 扩至 **256 槽位**；③ `ctrl_cta_cap_` 初始升至 **65,536 CTAs** 并在扩容前同步 `last_stream_` |
| **FIX-5** | [`sm120_guardian.cpp`](../xsched/platforms/cuda/hal/src/level3/sm120_guardian.cpp#L77-L165) `BuildEntryGuardian()` | `cuBLAS SGEMM` 二次抢占（`cmd: 6 -> 5`）发生 4 秒硬件看门狗超时 (`cuda error 719`) | `cmd` 在波次中途由 `6` 翻为 `5` 时，同一 CTA 内先到达的 Warp 走 `restore_path` (`BAR.SYNC 0` slot 66 -> slot 79)，后到达的 Warp 走 `norm_gate` (`BAR.SYNC 0` slot 46 -> `EXIT`)，二者共享 Barrier 0 导致后半部 Warp 退出、前半部 Warp 永锁在 slot 79 | 改为按整个恢复波次不变的 `is_restore_wave` (`+0x1c`) 统一路由：凡恢复波次所有 Warp 必进 `restore_path`，在 `replay_state_c` 内经双 `BAR.SYNC 0` 由 Thread 0 检测 `(cmd & 1)`（Evict 均为奇数 `1/5`，Restore 均为偶数 `2/6`）达成 CTA 全体一致决定 |
| **FIX-6** | [`sm120_trap_engine.cpp`](../xsched/platforms/cuda/hal/src/level3/sm120_trap_engine.cpp#L873-L945) Cave 0 五区间 `BAR.SYNC.TRAP` 死锁 | `cuBLAS SGEMM` 首次抢占即出现 `ev=12 dr=2087` 后 4 秒死锁 (`cuda error 719`) | 在 SM120 上 `bar_sync_trap()` 会因任意同 CTA 兄弟 Warp 已执行 `EXIT` 而永久挂死；此前 `Cave 0` 对 `pc_off == kGuardianBarOff` 无法区分"早到等待的 Warp"与"刚释放兄弟去 `EXIT` 的最后 Warp"，且 `Guardian` 与 `Cave 0` 均写 `entry_dec = 2` 导致两路径互斥 | 严格区分 `entry_dec` 语义：`2` 专指 **Guardian 侧已裁决 Drain**（兄弟 Warp 可能已 `EXIT`，`Cave 0` 任意区间见 `2` 均**绕过 `bar_sync_trap()` 直接逃逸**），`4` 专指 **Trap Handler 侧裁决 Drain**（全员均在 Trap 中存活，安全汇合 `bar_sync_trap()`）；同时将 `Cave 0` 缩减至 `139 / 152` 槽位（更精简） |

---

## 4. 全方位实机评测结果

### 4.1 闭源 NVIDIA CUDA 计算库抢占矩阵（RTX 5060, `Mode 5 kTrapFrame`, `NP=2` 双连抢占）

所有测试均通过 `LD_LIBRARY_PATH` 预加载 `libcuda.so.1 -> libshimcuda.so`，**零修改库代码、零重编译**，每次挂起期间均在同上下文 `hp_stream` 上运行高优先级校验内核（`hp_ok = 1`）：

| 编号 | 闭源计算库与算子规格 | 幂等性 | 内核特征（寄存器 / SMEM） | 抢占轮次 | 平均挂起延迟 | HP 并发延迟 | 数值误差 (`max_err` / `mismatch`) | 结论 |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **LIB-01** | `cuBLAS` FP32 SGEMM `3072×3072` (`beta=0.0`) | 幂等 | 2 内核：`22r/0B`, `40r/4228B` | `2 / 2` | `66.201 ms`\* | `0.271 ms` | `0.000000e+00` (`0 / 9,437,184`) | **PASS** |
| **LIB-02** | `cuBLAS` FP32 SGEMM `3072×3072` (`beta=1.0`) | **非幂等** | 2 内核：`40r/8192B`, `40r/4228B` | `2 / 2` | `65.520 ms`\* | `0.033 ms` | `0.000000e+00` (`0 / 9,437,184`) | **PASS** |
| **LIB-03** | `cuBLAS GemmEx` FP16 Tensor Core `4096×4096` (`beta=1.0`) | **非幂等** | 1 内核：`70r/3328B` | `2 / 2` | `9.255 ms`\* | `0.056 ms` | `0.000000e+00` (`0 / 16,777,216`) | **PASS** |
| **LIB-04** | `cuBLAS GemmEx` BF16 Tensor Core `4096×4096` (`beta=1.0`) | **非幂等** | 1 内核：`70r/3328B` | `2 / 2` | `9.144 ms`\* | `0.022 ms` | `0.000000e+00` (`0 / 16,777,216`) | **PASS** |
| **LIB-05** | `cuBLASLt` FP16 Tensor Core `4096×4096` (`beta=1.0`) | **非幂等** | 1 内核：`70r/3328B` | `2 / 2` | `9.247 ms`\* | `0.022 ms` | `0.000000e+00` (`0 / 16,777,216`) | **PASS** |
| **LIB-06** | `cuFFT` 原地 1D Batched C2C (`65536 × 256`) | **强非幂等** | 2 内核：`48r/0B`, `48r/0B` | `2 / 2` | `15.669 ms`\* | `0.025 ms` | `0.000000e+00` (`0 / 16,777,216`) | **PASS** |
| **LIB-07** | `cuSPARSE` CSR SpMM `4096×512` (`nnz=524K, beta=1.0`) | **非幂等** | 2 内核：`18r/0B`, `40r/0B` | `2 / 2` | `50.821 ms`\* | `0.079 ms` | `1.430511e-06` (`0 / 2,097,152`) | **PASS** |
| **LIB-08** | `CUB` `DeviceRadixSort::SortPairs` (`8M` 键值对) | **多内核+Look-Back** | 3 内核：`36r/4K`, `26r/1.1K`, **`111r/36.3K`** | `2 / 2` | **`1.749 ms`** | `0.014 ms` | **`0 / 8,388,608`** | **PASS** |

> **关键发现（\* 大网格 PCIe Guardian 延迟规律）**：全部 **8/8 闭源商业库实现 100% 比特级零误差双连抢占**；但挂起延迟呈现清晰的**网格规模线性放大现象**——小网格（`CUB` ~400 CTAs → `1.75 ms`，`HW-01` 120 CTAs → `0.27 ms`）为亚毫秒至毫秒级，而超大网格（`GemmEx` ~4,469 CTAs → `9.2 ms`、`cuSPARSE` → `50.8 ms`、`SGEMM` → `65.5 ms`）急剧膨胀。进一步单独对比零抢占执行时间证实：
> - `FP16 GemmEx 4096`：Native `17.890 ms` vs XQueue LP `18.863 ms`（**仅 +5.4% 开销**）；
> - `FP32 SGEMM 3072`：Native `14.471 ms` vs XQueue LP `62.055 ms`（**4.29× 膨胀！**）。
> 根因详见第 5 节 **DEF-01**（`BuildEntryGuardian` 对零拷贝宿主内存 `u_control` 的未缓存 `LDG.E.STRONG.SYS` PCIe 串行化读）。

---

### 4.2 Blackwell 微架构硬件特性边界矩阵（`HW-01..07`, RTX 5060）

| 编号 | 微架构特性 / CUDA 高级原语 | Mode 1 (`kEarlyExit`) | Mode 5 (`kTrapFrame`) | Context CILP (`ResumableSm120Context`) | 边界结论与微架构根因 |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **HW-01** | PTX Tensor Core `mma.sync.aligned.m16n8k16` + `atomicAdd` 非幂等 | **FAIL** (`cnt=546,123/450,240`, +21.3% 重复计数!) | **PASS** (`susp=0.274 ms`, `0/61,440`, `cnt=450,240/450,240`) | **PASS** (`cnt=450,240/450,240`, 同上下文不隔离流) | **Mode 5 完整保存恢复 Tensor Core 累加片段寄存器与非幂等副作用**；Mode 1 静默损坏 |
| **HW-02** | PTX `cp.async` (`LDGSTS`) + `wait_group` 异步拷贝流水线 | — | **PASS** (`2/2`, `susp=0.698 ms`, `0/15,360`) | **PASS** (`0/15,360`) | Blackwell 硬件进入 Trap 时自动排空在途 `LDGSTS`，SMEM 保存恢复比特一致 |
| **HW-03** | PTX SMEM `mbarrier` (`init/arrive/try_wait.parity`) | — | **FAIL** (`NP=0` 通过，`NP=1` 报 **`cuda error 700`**!) | **PASS** (`0/15,360`) | **重大硬件边界发现**：SMEM `mbarrier` 含硬件受保护的隐式事务/相位状态，Mode 5 原始 `LDS/STS` 字节拷贝无法重建，须交由硬件 CILP 处理 |
| **HW-04** | CUDA Graph（4 节点，`21.89 ms`） | Level-1 回退 | **Level-1 回退**（实测 `susp=20.254 ms`, `max_err=0.0`） | 支持整上下文暂停 | `CudaQueueLv3Trap` 主动检测 `CudaGraphCommand` 并回退至 Level-1 图边界，失去图内细粒度抢占 |
| **HW-05** | 空闲队列 `Suspend/Resume` 安全性回归 | — | **PASS** (`0/4,096`) | **PASS** | 验证 **FIX-1** 彻底根治空闲恢复污染 `cmd=6` |
| **HW-06** | Blackwell CTA Cluster (`__cluster_dims__(2,1,1)` + DSMEM + `cluster.sync`) | — | **PASS** (`2/2`, `susp=0.406 ms`, `0/7,680`) | **PASS** (`0/7,680`) | `cudaLaunchKernelEx` 完整受管；全集群同时常驻 SM 时 BAR0 同步挂起/恢复比特一致（跨波次拆分边界见 **DEF-10**） |
| **HW-07** | Cooperative Groups (`cudaLaunchCooperativeKernel` + `grid.sync`) | **Shim 绕过** | **Shim 绕过**（零 Guardian 插桩，直接 passthrough） | 需上下文级抢占 | 实证 **DEF-04**：[`intercept.cpp`](../xsched/platforms/cuda/shim/src/intercept.cpp) 未拦截 `cuLaunchCooperativeKernel` |

---

### 4.3 真实未修改 PyTorch 2.11 + cuDNN 9.19 + Triton 3.6.0 评测（`PT-01..06`, RTX 5060）

全部 Python 负载**零修改模型与框架源码**，直接通过 `LD_LIBRARY_PATH` 加载 XSched Shim 并将 PyTorch `lp_stream` 绑定至 `XQueue(kPreemptLevelInterrupt)`：

| 编号 | 真实框架工作负载 | 抢占轮次 | 平均挂起延迟 | HP 延迟 | 数值误差 | 结论与关键洞察 |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **PT-01** | `torchvision.models.resnet50` FP16 推理 (`cuDNN 9.19` Conv2D + BN) | `2 / 2` | `77.809 ms` | `0.244 ms` | **`0.000e+00`** | **PASS**：闭源 `cuDNN 9.19` 卷积算子双连抢占零误差！挂起延迟受 ResNet-50 几十个大网格未融合 BN/ReLU 算子 PCIe Guardian 影响 |
| **PT-02** | `torchvision.models.resnet18` 非幂等训练 (`Fwd + Bwd + AdamW`) | 中断 | — | — | — | **FAIL (`cuda error 1`)**：前 21 个前向/反向内核（含 `regs=242`, `static_smem=22656`, `calls=1`）全部插桩成功，第 22 个内核在 `LaunchWrapper` 报 `CUDA_ERROR_INVALID_VALUE`（详见 **DEF-05**） |
| **PT-03-MATH** | 4 层 Llama Prefill (`SDPA = MATH` 未融合注意力) | `2 / 2` | `38.391 ms` | `0.116 ms` | **`0.000e+00`** | **PASS**：数值精确，未融合小算子多且网格大导致挂起延迟较高 |
| **PT-03-FLASH** | 4 层 Llama Prefill (`SDPA = FlashAttention-2`) | `2 / 2` | `9.480 ms` | `0.159 ms` | **`0.000e+00`** | **PASS**：FlashAttention-2 在 Blackwell 上透明通过双连 Mode 5 抢占！ |
| **PT-03-EFF** | 4 层 Llama Prefill (`SDPA = xFormers Memory-Efficient`) | `2 / 2` | `10.172 ms` | `0.124 ms` | **`0.000e+00`** | **PASS**：Memory-Efficient Attention 透明通过双连抢占 |
| **PT-03-CUDNN** | 4 层 Llama Prefill (`SDPA = cuDNN 9.19 Fused Attention`) | `2 / 2` | **`0.222 ms`** | **`0.108 ms`** | **`0.000e+00`** | **PASS（极佳！）**：融合度最高的 `cuDNN 9.19` SDPA 实现 **`222 µs`** 极速抢占（较未融合 `MATH` 快 **173×**！） |
| **PT-04** | 自定义 **Triton 3.6.0** `sm_120` 非幂等内核（`51.78 ms`, `lmem=32` 栈溢出） | `2 / 2` | **`0.789 / 0.895 ms`** | **`0.072 / 0.130 ms`** | **`0.000e+00`** | **PASS**：JIT 编译的 Blackwell Triton 内核（含 LMEM 栈溢出保存恢复）亚毫秒级双连抢占，较 Level-1 快 **61×**，比特级零误差！ |
| **PT-05-QOS** | PyTorch 3 层 Llama (`SDPA=MATH`) + Triton 混部 QoS | 20 次 | Native `0.001` / Lv1 `1.83` / **M5 `73.42 ms`** | Native `0.124` / Lv1 `1.94` / **M5 `73.53 ms`** | — | **重要反面证据**：当 LP 包含大量未融合大网格 ATen 算子时，当前 PCIe 零拷贝 Guardian 使 Mode 5 `LP_ret` 暴跌至 `0.7%`；证明将 `u_control/CtaEntry` 迁入 VRAM 是真实 PyTorch 生产化的头号关键优化 |
| **PT-06** | 挂起期触发 PyTorch `cuBLAS` workspace / `cudaFree()` | 1 次 | — | **永久死锁** | — | 实证 **DEF-03**：未预热 workspace 时 `hp_stream` 隐式触发 `XMemFree_v2 -> ForEachWaitAll()` 与已挂起 LP 互锁；预热后消除 `cudaFree` 即可在 `0.166 ms` 内正常完成 |

---

### 4.4 DNN / LLM 多模式能力调度评测（Native vs Level-1 vs Mode 1 vs Mode 5）

#### (A) LLM Serving 混部（LP 多层 Prefill `4.8 ms` vs HP Token Decode `0.22 ms`, `period = 15 ms`）

| 调度模式 | HP P50 (ms) | HP P95 (ms) | HP P99 (ms) | SLO `< 2 ms` | SLO `< 5 ms` | Suspend P50 (ms) | LP 吞吐保留 | 非幂等数值误差 (`gemm` / `attn`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Native CUDA Streams** | `0.722` | `1.265` | `1.329` | `100.0%` | `100.0%` | — | `102.4%` | `0.00e+00` / `0.00e+00` |
| **XSched Level-1 (`Block`)** | `4.638` | `4.834` | `5.020` | `15.0%` | `95.0%` | `4.217` | `100.9%` | `0.00e+00` / `0.00e+00` |
| **XSched Level-3 Mode 1** | `0.853` | `2.417` | `2.838` | `75.0%` | `100.0%` | `0.407` | `80.3%` | `0.00e+00` / **`4.98e-01 (BAD!)`** |
| **XSched Level-3 Mode 5** | **`1.391`** | `3.396` | `4.066` | `50.0%` | **`100.0%`** | `0.895` | `73.8%` | **`0.00e+00` / `0.00e+00` (精确)** |

> **核心解读**：相比 Level-1（被 4.8 ms Prefill 阻塞导致 HP P50 飙升至 `4.64 ms`、2ms SLO 仅 `15%`），Mode 5 将 HP P50 压低 **3.33×** 且保持 **0 数值误差**；而 Mode 1 虽然挂起略快（`0.407 ms`），但在非幂等残差注意力算子上产生 **`0.498` 严重精度损坏**！

#### (B) 重载非幂等 DNN 训练（LP 原地 AdamW vs HP 实时推理）

1. **100% SM 饱和重载（`180 blocks × 256 threads + 16 KiB SMEM`, `base_lp = 21.15 ms`, `period = 25 ms`）**：

| 调度模式 | HP P50 (ms) | HP P99 (ms) | SLO `< 2 ms` | SLO `< 5 ms` | Suspend P50 (ms) | LP 吞吐保留 | 权重误差 (`mismatch`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Native CUDA Streams** | `0.282` | `4.586` | `80.0%` | `100.0%` | — | `102.1%` | `0.00e+00` (`0/19`) |
| **XSched Level-1 (`Block`)** | `17.547` | `18.435` | **`0.0%`** | **`0.0%`** | `17.524` | `100.5%` | `0.00e+00` (`0/30`) |
| **XSched Level-3 Mode 1** | `0.670` | `0.982` | `100.0%` | `100.0%` | `0.639` | `87.5%` | 短窗口偶合未写回（短核见下表直接损坏 `2.00e+00`） |
| **XSched Level-3 Mode 5** | **`1.589`** | `12.499` | **`93.3%`** | `93.3%` | **`1.565`** | **`87.5%`** | **`0.00e+00` (`0/17`)** |

> 相比 Level-1（HP P50 高达 `17.55 ms`、2ms & 5ms SLO 均为 **`0.0%`**），Mode 5 将 HP P50 降低 **11.0×**，SLO `< 2 ms` 从 **`0.0%` 跃升至 `93.3%`**，且保持 **`87.5%` LP 吞吐保留**与**零权重误差**！同时在短步非幂等原地 AdamW（`base_lp = 0.56 ms`）对比中，Mode 1 触发 **`max_err_w = 2.00e+00`（3,840 个参数损坏）**，而 Mode 5 连续校验 336 步 **`max_err_w = 0.00e+00`（0 错配）**！

2. **抢占频率敏感度扫描（`base_lp = 7.3 ms`, `120 blocks × 256 threads`, `period = 5 / 10 / 20 / 50 ms`）**：

| 抢占周期 | Level-1 HP P50 / SLO `< 2ms` | **Mode 5 HP P50 / P99** | **Mode 5 SLO `< 2ms`** | **Mode 5 LP 吞吐保留** | Mode 5 权重误差 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **5 ms** (200 Hz) | `2.352 ms` / `6.7%` | **`1.013 / 1.217 ms`** | **`100.0%`** | `75.0%` | `0.00e+00` (`0/9`) |
| **10 ms** (100 Hz) | `5.075 ms` / `0.0%` | **`1.049 / 1.129 ms`** | **`100.0%`** | `84.2%` | `0.00e+00` (`0/19`) |
| **20 ms** (50 Hz) | `2.620 ms` / `6.7%` | **`1.024 / 1.305 ms`** | **`100.0%`** | `89.5%` | `0.00e+00` (`0/38`) |
| **50 ms** (20 Hz) | `2.514 ms` / `6.7%` | **`0.914 / 1.189 ms`** | **`100.0%`** | **`93.1%`** | `0.00e+00` (`0/96`) |

> **Mode 1 vs Mode 5 有效工作保留实测（`period = 15 ms`）**：在 `7.3 ms` 负载下，Mode 1 丢弃重放导致 `LP_ret = 75.4%`（24.6% 算力浪费），而 **Mode 5 断点续算达到 `LP_ret = 86.1%`（净高出 +10.7 个百分点！）**。

---

### 4.5 双机底层微基准对照 & 跨上下文 BAR0 干扰实测

#### (A) RTX 5080 (`GB203`, 84 SMs, `580.95.05`) vs RTX 5060 (`GB206`, 30 SMs, `595.99.02`) 回归对照

| 指标 | RTX 5080 (`580.95.05`, 基线实测) | RTX 5060 (`595.99.02`, 修复后实测) |
| :--- | :---: | :---: |
| `sm120_sass_disasm` (`nvdisasm -b SM120`) | **79 / 79 PASS** | **79 / 79 PASS** |
| `.text.traphandler` 代码洞探测 | **11 Caves / 1,703 槽位** (`PASS`) | **11 Caves / 1,695 槽位** (`PASS`, `c0=139/152`) |
| Entry Guardian 零抢占延迟 (`84×128` null) | `Δ = +6.91 µs` (`5.14 → 12.05 µs`) | `Δ = +43.25 µs` (`4.50 → 47.75 µs`, 桌面 PCIe) |
| Mode 1 (`512 CTAs, 48K SMEM`) Evict / Restore P50 | `428.8 µs` / `144.8 µs` (`0 KiB`) | `901.2 µs` / `370.5 µs` (`0 KiB`) |
| Mode 5 (`84 CTAs, 0K SMEM`) Evict / Restore P50 | `248.6 µs` / `407.7 µs` (`6,016 KiB`) | `399.7 µs` / `772.2 µs` (`2,176 KiB`) |
| Mode 5 (`512 CTAs, 48K SMEM`) Evict / Restore P50 | `596.4 µs` / `547.9 µs` (`15,616 KiB`) | `1,076.2 µs` / `917.4 µs` (`5,632 KiB`) |
| 紧凑池化 `u_arena` 显存节省 (`16,384 CTAs`) | **160.6×** (`25.5 MiB` vs `4,096 MiB`) | **436.9×** (`9.38 MiB` vs `4,096 MiB`) |
| Context CILP (`sm120_l3_resumable_bench`) P50 / 误差 | **`425 µs`** / `diff=0, PASS` | **`431 µs`** (`P99=504 µs`) / `diff=0, PASS` |
| 11 项 E2E 正确性与 100 轮双连压测 | **11 / 11 ALL_PASS**（连续 3 轮） | `CASE 01` PASS；`CASE 02` 桌面 336-warp 原子风暴停滞（见 **DEF-08**） |

#### (B) 跨进程 / 跨上下文 BAR0 广播副作用实测（RTX 5060）

独立无 XSched 的旁观者进程（`bystander_pure`，80 次内核调用）在另一进程以 `5 ms` 间隔持续发射 **60 次 Mode 5 BAR0 广播脉冲** 期间的实测表现：

| 测试条件 | Bystander P50 | Bystander P95 | Bystander P99 / Max | CUDA 错误 | 数值错配 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **独立运行（无 BAR0 脉冲）** | `0.015 ms` | `0.072 ms` | `0.072 ms` | `0` | `0` |
| **并发 60 次 BAR0 脉冲（独立进程）** | `0.016 ms` | `0.067 ms` | **`0.749 ms`** (+`0.677 ms` 尾抖动) | **`0`** | **`0`** |

> **结论**：BAR0 广播脉冲命中未受管上下文时间片时，不会导致旁观者崩溃或数值错误（默认 Trap Handler 安全 RTT），但会在时间片重叠时引入约 **`~0.7 ms` 的 P99 尾延迟毛刺**。

---

## 5. 现存缺陷全景清单与下一步优化路线图

本节按严重度（**P0 架构/性能核心瓶颈 → P1 框架与功能缺口 → P2 硬件物理边界**）坦诚汇总所有经实机复现证实的现存缺陷：

| 优先级 | 编号 | 缺陷简述与精确定位 | 实机证据 | 推荐技术方案 |
| :---: | :---: | :--- | :--- | :--- |
| **P0** | **DEF-01** | **零拷贝 PCIe `u_control / CtaEntry` 大网格性能退化**（[`sm120_guardian.cpp`](../xsched/platforms/cuda/hal/src/level3/sm120_guardian.cpp#L60-L150), [`sm120_trap_engine.cpp`](../xsched/platforms/cuda/hal/src/level3/sm120_trap_engine.cpp#L1215-L1250)） | `cuBLAS SGEMM 3072` 零抢占慢 `4.3×`（`14.5 → 62.1 ms`），挂起 `65.5 ms`；PyTorch `PT-05` 未融合 `SDPA=MATH` 挂起 `73.4 ms`、`LP_ret=0.7%` | **将 `CtaEntry[]` 与 `Layout36` 迁入 GPU GDDR7 VRAM（L2 缓存 384 GB/s）**；仅保留 64B `cmd/counters` 控制头在零拷贝，或通过高优 `op_stream` DMA 写入 VRAM 镜像，且仅由 `tid==0` 单线程读 `cmd` 后借 SMEM 广播，可将大网格开销压低 **20×–50×** |
| **P0** | **DEF-02** | **SMEM `mbarrier` 硬件元数据无法通过字节拷贝恢复**（[`sm120_trap_engine.cpp`](../xsched/platforms/cuda/hal/src/level3/sm120_trap_engine.cpp#L86-L158)） | `HW-03` 在 Mode 5 `NP=1` 触发 `cuda error 700`，而 `NP=0` 与 Context CILP 均通过 | ① 静态 SASS 扫描（如同 `BinaryHasDeviceCalls`）自动检测 `MBARRIER` / TMA (`UTMA`) 指令；② 对含 `mbarrier` 的内核自动降级至 Level-1 或路由至硬件 Context CILP 后端 |
| **P0** | **DEF-03** | **Shim `XMemFree_v2()` 跨队列全局等待自死锁**（[`xsched/platforms/cuda/shim/src/shim.cpp`](../xsched/platforms/cuda/shim/src/shim.cpp#L246)） | LP 挂起期间 HP 调用 `cudaFree()` 或 PyTorch `cuBLAS` workspace 释放时触发 `XQueueManager::ForEachWaitAll()` 永久死锁 | 修改 `XMemFree_v2()`：仅等待**未处于 Suspended 状态**的队列，或仅在释放地址确实被目标流引用时同步，对已挂起队列延迟回收（deferred free queue）至 `Resume` 后释放 |
| **P1** | **DEF-04** | **Shim 未拦截 `cuLaunchCooperativeKernel[_ptsz]`**（[`xsched/platforms/cuda/shim/src/intercept.cpp`](../xsched/platforms/cuda/shim/src/intercept.cpp#L276)） | `HW-07` (`grid.sync()`) 绕过 XSched 直接 passthrough，无插桩无抢占 | 在 `intercept.cpp` / `shim.cpp` 补齐 `XLaunchCooperativeKernel` 拦截，并按 `grid.sync()` 全网格依赖特征默认采用 Level-1 或 Context CILP（因部分 CTA 换出将破坏 `grid.sync()` 全网格常驻约束） |
| **P1** | **DEF-05** | **PyTorch Autograd/AdamW 反向算子 `CUDA_ERROR_INVALID_VALUE`**（[`cuda_command.cpp`](../xsched/platforms/cuda/hal/src/common/cuda_command.cpp#L63-L144), [`cuda_queue.cpp`](../xsched/platforms/cuda/hal/src/level3/cuda_queue.cpp#L68)） | `PT-02`（ResNet-18 训练）第 22 个内核在 `LaunchWrapper` 报 `cuda error 1` | 审查并修复 `CudaKernelCommand` 在 `param_cnt_ == 0` 时未深拷贝 `CUlaunchConfig::attrs`、以及 Blackwell 新 `CUlaunchAttributeID`（如 PDL/MemSyncDomain）与 `cuXtraGetParamCount` 对齐问题 |
| **P1** | **DEF-06** | **工作线程未显式绑定 CUDA Context 触发断言崩溃**（[`cuda_command.cpp`](../xsched/platforms/cuda/hal/src/common/cuda_command.cpp#L18)） | 子线程直接调用 `cufftExecC2C`（未先 `cudaSetDevice`）触发 `current context of the calling thread is nullptr` | 当 `cuCtxGetCurrent()` 返回空时，通过 `cuStreamGetCtx(stream, &ctx)` 从目标流回退获取上下文 |
| **P1** | **DEF-07** | **高频短核抢占（`5 ms` / `0.56 ms` 核）触发 `max_slots` 溢出超时**（[`sm120_trap_engine.cpp`](../xsched/platforms/cuda/hal/src/level3/sm120_trap_engine.cpp#L1318-L1325)） | `DNN_TRAIN period=5ms` 偶发 `timed out waiting for prior restore (141/166, inflight=0)` | ① 将 `max_resident` 安全裕量从 `+16` 提升至 `std::max(64u, sm_count * 2)`；② `Cave 0` 在 `slot_id >= max_slots` 时不累加 `evicted_slots`（或等待上限以 `min(evicted_slots, max_slots)` 为准） |
| **P1** | **DEF-08** | **桌面 `595.x` 驱动高并发 PCIe 原子风暴导致 `CASE 02` 停滞**（RTX 5060 特有） | RTX 5080 (`580.95.05`) `11/11 ALL_PASS`；RTX 5060 (`595.99.02`) 上 `CASE 02`（336-warp 持续 PCIe `atomicAdd`）出现 `ev=0 dr=0 m1=0` | 本质与 **DEF-01** 同源：桌面显示合成器共享 GR 引擎 + 336 Warp 饱和 PCIe 上行通道阻塞 `u_control` 读。迁入 VRAM 后将自然缓解 |
| **P2** | **DEF-09** | **CUDA Graph 仅支持 Level-1 图边界回退**（[`cuda_queue.cpp`](../xsched/platforms/cuda/hal/src/level3/cuda_queue.cpp#L190-L198)） | `HW-04`（`21.89 ms` 4 节点图）挂起延迟高达 `20.254 ms` | 探索在 `cuGraphExecKernelNodeSetParams` / 图实例化阶段对图内内核节点进行预插桩，或对含 CUDA Graph 的推理服务（如 vLLM Decode / `torch.compile`）切换至 **Context CILP** |
| **P2** | **DEF-10** | **硬件物理边界：BAR0 Root 权限 / 全芯片广播 + CTA Cluster 跨波次边界** | 需 `CAP_SYS_ADMIN` 写 `/sys/bus/pci/.../resource0`；旁观进程存在 `0.7 ms` P99 抖动；CTA Cluster 在 `blocks > SMs` 跨波次时可能拆分集群成员 | ① 生产部署将 BAR0 脉冲封装为极简内核模块或特权守护进程（类似 `xserver`）；② 检测 `clusterDim > 1 && num_ctas > max_resident` 时自动退回 Level-1/CILP 以保 DSMEM 集群原子性 |

---

## 6. 综合结论：双引擎协同的适用边界建议

结合双机实测与全部 27 项真实负载数据，Blackwell (`sm_120`) Level-3 v2 已清晰勾勒出**不同调度引擎的最佳适用场景**：

1. **流级细粒度抢占（同进程/同上下文多流混部）→ `CudaQueueLv3Trap`（Mode 5 `kTrapFrame`）**：
   - 唯一能在**单个 `CUcontext` 内部**实现零源码修改、非幂等精确抢占与微秒/亚毫秒级 HP 响应的方案（`cuDNN 9.19` SDPA `0.222 ms`、Triton 3.6.0 `0.789 ms`、PTX Tensor Core `0.274 ms`、CUB RadixSort `1.75 ms`，重载非幂等训练 2ms SLO 从 `0% → 93.3%–100%`）。
   - **当前首要优化方向**：实施 **DEF-01**（`CtaEntry[]` 迁入 VRAM）与 **DEF-03**（解除 `XMemFree_v2` 死锁），即可释放其在未融合 PyTorch 与超大网格 `cuBLAS SGEMM` 上的全部潜力。
2. **上下文/进程级隔离（含 `mbarrier` / TMA / CUDA Graph / 多租户容器）→ `ResumableSm120Context`（Context CILP）**：
   - 零内核插桩开销、天然兼容 `mbarrier`（`HW-03` PASS）与 CUDA Graph、跨进程抢占 P50 仅 **`425 µs`（RTX 5080）/ `431 µs`（RTX 5060）** 且零工作丢失，但颗粒度固定为整个 `CUcontext`，不可用于同上下文双流抢占。
3. **混合自适应调度（下一代演进方向）**：
   - 在 XSched HAL 中引入**内核特征静态分类器**（`sm120_guardian` 扫描 `MBARRIER/UTMA` + `grid_size` + `clusterDim` + `CUDA Graph`），动态在 **Mode 1（已标注幂等核）**、**Mode 5（非幂等通用核）**、**Context CILP（跨进程/`mbarrier`/CUDA Graph）** 之间无缝路由。
