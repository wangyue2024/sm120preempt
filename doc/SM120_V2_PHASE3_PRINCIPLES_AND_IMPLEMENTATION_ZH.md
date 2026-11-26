# SM120 Level-3 抢占系统：原理与实现详解（截至 Phase 3.4，2026-11-26）

> 适用代码：分支 `phase3-def20-production-testing`（`7d1da87` 及之后）。
> 本文把系统中每一个机制讲到“为什么这样设计、具体怎么实现、有哪些参数、实测代价是多少”。v2.2.1 中未改变的底层细节（代码洞定位、79 条手写指令逐条编码、各 Cave 的逐条指令流）以 [`SM120_V2_COMPLETE_TECHNICAL_MANUAL_ZH.md`](SM120_V2_COMPLETE_TECHNICAL_MANUAL_ZH.md) 为准，本文给出它们的原理摘要并指明出处；Phase 3 中修正或新增的部分在本文完整展开。
> 配套文档：开发与探索过程见 [`SM120_V2_PHASE3_EXPLORATION_REVIEW_ZH.md`](SM120_V2_PHASE3_EXPLORATION_REVIEW_ZH.md)；缺陷台账见 [`SM120_V2_PHASE3_DEFECT_LEDGER_ZH.md`](SM120_V2_PHASE3_DEFECT_LEDGER_ZH.md)。

---

## 目录

0. 术语
1. 系统全景：XSched 分层与三个抢占级别
2. XQueue 运行时：命令如何被提交、发射、同步、挂起与恢复
3. SM120 Level-3 Trap 引擎：驱逐与恢复的完整原理
4. Entry Guardian：每个 CTA 入口处的 80 条指令
5. 自适应路由：哪些内核走 Level-3，哪些走 Level-1
6. 显存：u_arena、LMEM 保存区与空闲回收
7. 发射路径开销：直接提交、调用方同步与透传发射
8. 非 root 抢占：`xsched-bar0d` 脉冲代理
9. 跨进程调度：全局调度器、就绪门控与相关修复
10. 调度策略与理论上限
11. 环境变量完整参考
12. 构建、部署与测试操作

---

## 0. 术语

| 术语 | 含义 |
|---|---|
| HP / LP | 高优先级（在线、延迟敏感）/ 低优先级（离线、吞吐型）工作 |
| XQueue | XSched 对一条硬件队列（CUDA 流）的可抢占封装；应用的 CUDA 调用经 shim 拦截后变成 XQueue 命令 |
| HwQueue / HAL | 平台相关的硬件队列实现；SM120 的 Level-3 实现为 `CudaQueueLv3Trap` |
| Level-1 / 2 / 3 | 抢占级别：1 = 停止发射新命令（已在 GPU 上的命令跑完）；2 = 可作废未开始的命令；3 = 中断正在执行的内核 |
| Mode 1 / Mode 5 | SM120 Level-3 的两种中断方式：Mode 1 早退 + 重放（仅幂等内核）；Mode 5 保存完整线程状态再原位恢复（通用） |
| CTA | 线程块（cooperative thread array） |
| State A / B / C | 被抢占内核中 CTA 的三种状态：A = 已完成；B = 执行中被驱逐，现场保存在 arena；C = 尚未进入用户代码就被排空 |
| Guardian | 插桩时放在每个内核入口的 80 条 SASS 指令，每个 CTA 启动时先执行它 |
| Cave | `.text.traphandler` 中可写的空闲代码区，Trap 处理逻辑写在 9 个 Cave 里 |
| 控制头 / u_ctrl | 每个流一块设备内存：`0x50` 字节控制头 + 每 CTA 16 字节表项（+ 其前方 32 KiB 门控镜像） |
| BAR0 脉冲 | 向 GPU 寄存器 `0x424684` 写入 `STOP_TRIGGER`，让所有 SM 进入 Trap 处理程序 |
| 占空比（duty） | HP 请求处理时间占总时间的比例 |

---

## 1. 系统全景

### 1.1 调用路径

```
应用 (PyTorch / cuBLAS / cuDNN / Triton ...)
  │  CUDA Driver API 调用（cuLaunchKernel / cuMemcpyAsync / cuStreamSynchronize ...）
  ▼
shim (libcuda.so.1 软链接到 XSched 的拦截库)
  │  该流是否绑定了 XQueue？否 → 直接透传给真实驱动（PERF-2 优化了这条路径）
  ▼
XQueue (preempt/)        命令缓冲区 + 发射线程 + 命令日志；Suspend / Resume
  ▼
HwQueue (platforms/cuda/hal/)  Level-1：CudaQueueLv1；Level-3：CudaQueueLv3Trap
  │  Level-3 在发射内核前做插桩 / 写控制参数；挂起时触发 BAR0 脉冲
  ▼
GPU（RTX 5060 GB206 30 SM / RTX 5080 GB203 84 SM，sm_120）
```

调度决策有三种来源（`XSCHED_SCHEDULER`）：`APP`（应用自己调用 `XQueueSuspend/Resume`，本项目所有单进程测试使用）、`LCL`（进程内调度器）、`GLB`（跨进程：各进程把事件发给 `xserver`，由它统一决策，见 §9）。

### 1.2 三个抢占级别的本质区别

| 级别 | 挂起时做什么 | 挂起延迟上界 | 正确性要求 |
|---|---|---|---|
| Level-1（Block） | 发射线程暂停，不再发新命令；已在 GPU 上的命令（最多 `threshold`=16 条）照常执行完 | 已发射命令的执行时间（毫秒级到秒级） | 无 |
| Level-2（Deactivate） | 另外作废尚未开始执行的命令 | 同上，但积压更少 | 被作废命令须可重发 |
| Level-3（Interrupt） | 另外**中断正在执行的内核**：SM120 上通过 BAR0 脉冲让所有 CTA 进 Trap，保存现场后退出 | 一次脉冲 + 保存现场：实测挂起 P50 约 0.02–0.3 ms | 恢复后结果必须与未被中断时逐位一致 |

SM120 的 Level-3 是本项目的核心：**不改内核源码、不假设幂等**，在用户内核任意一条指令处打断并恢复。

---

## 2. XQueue 运行时

### 2.1 数据结构

- **命令缓冲区（`CommandBuffer`）**：提交线程放入命令，发射线程取出。维护队列状态 `Idle / Ready`：从 Idle 进入第一条硬件命令时变为 Ready 并发出 `XQueueReadyEvent`；缓冲区排空且最后一条命令是已完成的 `WaitAll` 时变为 Idle 并发出 `XQueueIdleEvent`。**约定**：Idle 时“最后入队的命令”必然是一个已完成的 `XQueueWaitAllCommand`，`EnqueueXQueueWaitAllCommand()` 据此在 Idle 时直接返回它而不再等待（DEF-25 的教训正来自破坏了这条约定，见探索回顾）。
- **发射线程（`LaunchWorker`）**：取出命令并调用 `HwQueue::Launch()`。维护**命令日志**（已发射、未确认完成的命令，最多 `threshold`=16 条，`XSCHED_DEFAULT_COMMAND_THRESHOLD`），每 `batch_size`=8 条使一条可同步，日志满时同步最早的可同步命令腾出空位。
- **挂起状态**：`Pause()` 只在互斥锁下把状态设为 Paused，不等待发射线程；发射线程在发射前检查状态（OBS-15：日志满、等待空位之后不再检查，见 §9.6）。

### 2.2 挂起与恢复的调用序列

```
XQueueSuspend(flags)                         XQueueResume(flags)
  launch_worker.Pause()                        HwQueue.Restore(log)      (Lv3: no-op)
  Level≥2: HwQueue.Deactivate()                HwQueue.Reactivate(log)   (Lv3: 恢复被中断的内核)
  Level≥3: HwQueue.Interrupt()   ← BAR0 脉冲   launch_worker.Resume()
  flags & SyncHwQueue: HwQueue.Synchronize()
```

Level-1 的恢复只是唤醒发射线程；Level-3 的 `Reactivate()` 要重新发射被中断的内核并准备恢复波次（§3.5），这一步需要 CUDA 上下文（DEF-26，§9.3）。

### 2.3 Phase 3.3 的发射路径优化（PERF-1）

| 机制 | 原理 | 开关 / 参数 |
|---|---|---|
| 直接提交 | 提交线程在“前面没有排队命令、发射线程未持有有序命令”时直接在本线程调用驱动发射，省掉交给发射线程的一次唤醒。条件：发射线程在休眠，**或**上一次调用（两次 `WaitAll` 之间）只有 1 条命令（单命令调用没有可流水的批量） | `XSCHED_XQUEUE_DIRECT`（默认 1） |
| 调用方 `WaitAll` | 所有已提交命令都已发射时，同步在调用线程上直接对硬件流做，不再入队 `WaitAll` 等发射线程处理；若同步期间发生了挂起（`pause_count_` 变化）则判失败并退回原路径 | `XSCHED_XQUEUE_CALLER_SYNC`（默认 1） |
| 有界自旋 | 发射线程在缓冲区空后先自旋一小段再休眠，背靠背提交免去 futex 唤醒 | `XSCHED_SPIN_US`（默认 30 µs，上限 100000） |
| 布局缓存 | Level-3 每次发射需把内核布局（槽位大小、帧偏移等）写入控制头；按 `(函数, 网格, 块, SMEM)` 缓存已写入的布局，相同时跳过拷贝 | `XSCHED_SM120_LAYOUT_CACHE`（默认 1） |

调用方同步成功后，Ready→Idle 由发射线程在其后台同步完成后补做，实测在同步返回后约 21 µs（最大 41 µs）内（`test_xqueue_idle_p34.py`，回归项 `SYN-xq-idle`）。

实测（C++ 微基准 `test_percall_p3`，单内核调用 + 同步，相对 native 的增量）：Level-1 +1.1–1.2 µs、Level-3 +2.2–2.4 µs（两卡；优化前 8–17 µs）。

---

## 3. SM120 Level-3 Trap 引擎

### 3.1 中断的触发：BAR0 驱逐脉冲

GPU 的 PGRAPH 有一个调试用的广播寄存器 `0x00424684`，写入 bit 31（`STOP_TRIGGER`）会使**所有 SM 上正在运行的线程束**进入各自上下文的 Trap 处理程序。脉冲序列（`FireBar0Pulse()`，`sm120_bar0_proxy.h`）：

1. 写 `0x80001c01`（置位 `STOP_TRIGGER`）；
2. 读一次 `NV_PTIMER_TIME_0`（`0x9400`）——一个**非 PRI** 寄存器，用来把前一个写操作刷出去，保证脉冲窗口约 1 µs；
3. 写 `0x00001c01`（撤销）。

硬件规则 D10：`STOP_TRIGGER` 有效期间**绝不能读 PRI 广播寄存器**（否则可能挂死），所以刷新用 PTIMER 而不是读回原寄存器。

脉冲是**整块 GPU 的广播**，这是两个跨进程现象的根源：DEF-27（别的上下文正要切换进来时，本上下文处于 Trap 中 → 切换超时，§9.4）和 OBS-14（同时运行的桌面图形上下文偶发 MMU 故障，§9.7）。

访问 BAR0 需要映射 `/sys/bus/pci/<bdf>/resource0`，默认只有 root 可以；非 root 通过 `xsched-bar0d` 代理（§8）。

### 3.2 Trap 处理程序挂钩与 9 个代码洞

CUDA 驱动为每个上下文在 ToolsMemory 中放置 `.text.traphandler`。引擎在其中找到 9 段可写的空闲区（Cave），写入自己的处理逻辑，并把原处理程序入口处的一条指令改为 `CALL.ABS` 跳入 Cave 0（定位、指纹校验和挂钩的细节见手册第 2 章）。各 Cave 职责：

| Cave | 职责 |
|---|---|
| 0 | 入口过滤（区分本引擎的驱逐 / 恢复与其他 Trap）、五区间边界判断、Mode 1 / Mode 5 驱逐主控 |
| 1 | 重入防护与分发、**0 脉冲恢复主控**（由 Guardian 的 `BPT.TRAP 0x1` 自陷触发）、干净退出桩 |
| 2 / 5 | 通用寄存器级联保存 / 恢复（`SR_REGALLOC` + `BRX` 跳表） |
| 3 | 命名屏障 0–15 状态（`B2R` / `R2B`） |
| 4 / 6 | `UR0..UR63`、`UPR`、收敛屏障 `B0..B15` |
| 7 | 线程局部内存（LMEM，含 `__noinline__` 调用栈） |
| 8 | 动态共享内存（`SR_SMEMSZ`）与 112 B 硬件帧 |

Cave 0 的五区间边界会用到 Guardian 的运行时锚点（`GuardianRestorePathOff()`、`GuardianReplayCIdxOff()`），因此 Guardian 布局变化（`BAR.RED`）时锚点随之移动（§4.3）。

### 3.3 Mode 5 驱逐：保存什么、怎么保存

一个 CTA 在执行用户代码时收到 Trap，Mode 5 把它的**完整架构状态**写进 arena 的一个槽位，然后退出：

| 状态 | 保存方式 |
|---|---|
| `R0..R13`、PC 等 | 112 B 硬件 Trap 帧 |
| `R14..R(N−1)`（N = 内核寄存器数） | Cave 2 用 64 位寄存器对逐对存储；起点由 `S2R SR_REGALLOC` 算出，用 `BRX` 跳表跳到对应位置 |
| 统一寄存器 `UR0..UR63`、`UPR` | Cave 4 / 6 |
| 收敛屏障 `B0..B15` | Cave 4 / 6（`BMOV.32`） |
| 命名屏障 0–15 | Cave 3 |
| 共享内存 | Cave 8：长度 `SR_SMEMSZ − 0x400`（`SR_SMEMSZ` 含 1 KiB 系统保留区，用户 SMEM 从 `0x400` 开始） |
| 局部内存（栈） | Cave 7：容量见 §6.2 |
| 线程束存活掩码 | 写入 CTA 表项 `warp_mask`（恢复时只恢复驱逐时存活的线程束） |

**DEF-20（P0，Phase 3 修复）：寄存器级联的上界。** 原实现以 `SR_REGALLOC − 3` 为最高寄存器，认为 `SR_REGALLOC` 额外包含硬件保留的 2 个寄存器。实测（二进制改写探针，把 `S2R SR_TID.X` 改成 `S2R SR_REGALLOC`）表明 `SR_REGALLOC` **严格等于** `CU_FUNC_ATTRIBUTE_NUM_REGS`。结果是每个内核最高的 1–3 个寄存器从未被保存/恢复——它们恰好在 Trap 时刻存活时，恢复后数据错误或越界访问。这是 dtype 转换核崩溃、CUB 排序 / 扫描 / topk 崩溃、Triton RMSNorm 静默算错（DEF-14/15/16）的共同根因。修复：

1. 上界改为 `SR_REGALLOC − 1`；
2. 插桩时把寄存器数**向上取偶**（`(max(raw,16)+1) & ~1`，`raw < 255` 时），因为级联按寄存器对进行，奇数寄存器数时最后一对会越界。插桩通过 `cuXtraSetLocalRegsPerThread` 改变分配，探针确认 `SR_REGALLOC` 随之变化（25→26、45→46）。

残留边界 DEF-20b：用满 255 个寄存器的内核无法取偶，`R254` 不在保存范围（生产内核中未见）。

**槽位分配**：驱逐的 CTA 用 `ATOMG.ADD` 在控制头 `evicted_slots` 上取得紧凑槽号，写入表项 `slot_id`；arena 只需按“同时驻留的 CTA 上限”而不是网格大小分配（§6.1）。

### 3.4 CTA 状态与 Mode 1

驱逐时每个 CTA 落入三种状态之一（表项 `state`）：

- **A（0）**：已经自然完成——恢复时直接退出。
- **B（5）**：执行中被驱逐，现场在 arena——恢复时自陷还原现场后从被打断处继续。
- **C（2）**：在 Guardian 中尚未进入用户代码就看到了抢占命令，排空退出——恢复时从头执行。

Mode 1（`XSCHED_SM120_FORCE_MODE1=1` 或命令带“幂等”属性）不保存现场：执行中的 CTA 也直接退出并标记为“重放”（`state=1`），恢复时从头重跑。它只适用于幂等内核，Phase 3 的生产测试全部使用 Mode 5。

### 3.5 恢复：0 脉冲自陷还原

恢复不需要再发 BAR0 脉冲：

1. **`PrepareRestore()`**（主机）：在高优先级拷贝流上把 `restored_slots=0`、`is_restore_wave=1`、`cmd=6`（Mode 5 恢复）写入控制头，并同步写门控镜像（§4.4）。写入在重新发射内核之前完成（流序）。
2. **重新发射被中断的内核**（同一网格），即“恢复波次”。每个 CTA 进入 Guardian：看到 `is_restore_wave=1`，走 `restore_path`：
   - 不是被中断的那个内核（`my_idx != active_kernel_idx`）→ 退出（它只是排在后面的内核，之后会正常重发）；
   - 读自己表项的 `state`：A → 退出；B → 执行 `BPT.TRAP 0x1` **自陷**，Cave 1 识别为恢复请求，从 arena 还原全部现场后返回到被打断的 PC；C → `replay_state_c`：清零状态后从头执行原内核。
3. **恢复后条件清零（DEF-12 / DEF-21）**：恢复波次之后流上排一个小内核，**只有当 `cmd` 仍为 6（或 Mode 1 的 2）时**才把控制头 `0x00..0x1f` 与 `is_restore_wave` 清零；如果恢复途中又发生了一次抢占（`cmd` 已被改成 5），就保留 `is_restore_wave` 与 `active_kernel_idx`，让下一轮恢复仍然正确路由。该内核现在用 256 个线程同时清零所有门控镜像行（§4.4）。
4. 下一次抢占前，`PreemptStream()` 若发现上一轮是 Mode 5 恢复，会先等 `restored_slots` 达到驱逐数且 `trap_inflight==0`（所有线程束都离开了 Cave 1），上限 2 s；发脉冲之前再确认一次 `trap_inflight` 归零（上限 500 ms），避免脉冲打在仍处于恢复处理中的线程束上。

**恢复途中再次抢占（mid-restore re-preemption）**：Guardian 在 `replay_state_c` 里用 `cmd & 1` 区分“正常恢复（cmd=2/6）”与“恢复中又被抢占（cmd=1/5）”，后者保持状态、写 `entry_dec=2` 后退出，保证同一 CTA 不会被执行两次。约 3,000 次压测抢占验证（DEF-21 关闭）。

### 3.6 控制平面：数据布局

控制头（`Sm120ControlHeader`，`0x50` 字节，设备内存，主机通过拷贝流写入）：

| 偏移 | 字段 | 写入者 / 含义 |
|---|---|---|
| `+0x00` | `cmd` | 主机：0 正常、1 Mode 1 驱逐、2 Mode 1 重放、5 Mode 5 驱逐、6 Mode 5 恢复 |
| `+0x04` | `evicted_slots` | GPU 原子加：已分配的驱逐槽位数 |
| `+0x08` | `active_kernel_idx` | Guardian：被中断内核的 64 位序号（第一个看到抢占的 CTA 写入） |
| `+0x10` | `restored_slots` | GPU：恢复完成的槽位数 |
| `+0x14` | `drained_cnt` | GPU：State C 退出计数 |
| `+0x18` | `mode1_exit_cnt` | GPU：Mode 1 退出计数 |
| `+0x1c` | `is_restore_wave` | 主机置 1；条件清零内核清 0 |
| `+0x20` | `arena_va` | 本次发射使用的 arena 地址 |
| `+0x28..+0x44` | 槽位跨度、槽位上限、线程帧 / SMEM / LMEM 区偏移、LMEM 容量、CTA 表容量 | 主机：内核布局（`PrepareLaunch`，有布局缓存） |
| `+0x48` | `overflow_err` | GPU：槽位不足时置位 |
| `+0x4c` | `trap_inflight` | GPU：处于 Cave 1 中的线程束数 |

CTA 表项（每 CTA 16 字节，从 `u_ctrl + 0x50 + 16 × cta_id` 开始）：`state`、`entry_dec`、`slot_id`、`warp_mask`。

内核如何找到控制头：发射前通过 `cuXtraSetDebuggerParams` 写入每个函数的常量窗口 `c[0][0x170..0x187]`：`u_control_va`（`0x170`）、内核描述符地址（`0x178`）、`kernel_idx`（`0x180`）。`u_control_va == 0` 表示该次发射不受管（例如高优先级流的透传发射），Guardian 直接跳过（§4.2）。PERF-2 之后只对确实写过非零参数的函数做清零（§7.2）。

**写入顺序保证**：所有控制头 / 镜像写入都在 BAR0 脉冲之前完成（拷贝流同步后再触发脉冲），所以一个在脉冲前读到 `cmd==0` 的 CTA 必然已驻留，要么已完成、要么会被脉冲打断——这一论证让 Guardian 可以去掉 slot 28 的重读（§4.3）。

---

## 4. Entry Guardian

### 4.1 为什么需要它

脉冲只能打断**已驻留**的 CTA。一个网格往往有很多波 CTA，抢占命令发布之后才开始的 CTA 必须自己发现“现在处于抢占状态”并排空，恢复波次中的 CTA 必须找到自己的状态并分派。Guardian 就是在插桩时插到每个内核入口的这段逻辑（`0x500` 字节、80 条指令），原内核代码搬到 `ep_inst + 0x500`。

### 4.2 当前默认布局（门控镜像 + `BAR.RED`）

| 槽位 | 指令 | 作用 |
|---|---|---|
| 0–3 | `LDC c[0][0x170]`、`LDC c[0][0x174]`、`LOP3`、`@!P0 BRA` | **快速旁路**：`u_control_va==0`（未受管发射）直接跳到原内核 |
| 4–11 | `S2R SR_CTAID.{X,Y,Z}`、`LDC gridDim`、`IMAD`×3 | 计算线性 `cta_id` 与表项地址 `R6:R7 = u_ctrl + 16·cta_id` |
| 12–14 | `S2R SR_VIRTUALSMID`（`0x43`）、`IMAD.WIDE`、`LD.E.64 [R8 − 0x8000]` | **门控镜像**：一次读出本 SM 的 `{cmd, is_restore_wave}` |
| 15–16 | `ISETP`、`@P0 BRA restore_path` | 恢复波次 → `restore_path` |
| 17–18 | `ISETP`、`@!P0 BRA norm_gate` | `cmd==0` → 直接去 `norm_gate` |
| 19–24 | 读 `active_kernel_idx`、比较 `c[0][0x180]`、`@P0 EXIT` | 已有别的内核被中断时，排在后面的内核 CTA 退出 |
| 25–38 | `norm_gate`：`S2R SR_TID.*`、`P0 = (tid==0 && cmd!=0)`，`@P0` 写表项 `state/entry_dec`（2 = State C，Mode 1 时为 1）、写 `active_kernel_idx`、`drained_cnt++` | 抢占已发布：由线程 0 代表 CTA 登记“排空” |
| 39（`0x270`） | `BAR.RED.OR 0, P0` | 全 CTA 屏障，同时把“是否有线程登记了排空”做 OR 归约 |
| 40–44 | `B2R.RESULT P1`、`@P1 LD entry_dec`、`ISETP`、`@P0 EXIT`、`BRA orig` | 只有归约为真时才读 `entry_dec`；排空则退出，否则跳到原内核 |
| 45–59（`0x2d0` 起） | `restore_path`：内核隔离、读 `state`、A 退出 / B `BPT.TRAP 0x1` / C 转 `replay_state_c` | §3.5 |
| 60–79（`0x3c0` 起） | `replay_state_c`：屏障、`cmd & 1` 判定正常恢复或恢复中再抢占、`0x4c0` 处最终屏障 | §3.5 |

锚点（Cave 0 用于区间判断）：`BAR` `0x270`、`restore_path` `0x2d0`、`replay_state_c` `0x3c0`、replay 屏障 `0x4c0`；原版（`XSCHED_SM120_GUARDIAN_V1=1` 或关闭 `BAR.RED`）为 `0x2c0` / `0x3b0`。

### 4.3 Guardian 的四步优化（DEF-17）及原理

起点：Phase 3 测得短 CTA 访存算子 Lv3 稳态开销最高 +101%（5080 `rope_apply`）。Guardian 每个 CTA 执行一次，CTA 越短、越多，占比越大。

| 步骤 | 原理 | 效果 |
|---|---|---|
| ① 去掉 slot 28 的 `cmd` 重读 | 原版在 `norm_gate` 再读一次 `cmd`。由 §3.6 的写入顺序保证：读到 `cmd==0` 的 CTA 已驻留，必被脉冲打断或已完成，重读多余 | 5080 `embedding` +22.8% → +1.9%、`gelu` +18.3% → +0.3% |
| ② 短 CTA 路由 | 见 §5：无回边、无调用、2–1024 波的内核改走 Level-1 | `transpose_copy` / `rope_apply` 降到 < 3% |
| ③ `BAR.RED.OR` | 原版每个 CTA 在 `BAR.SYNC` 后都要读一次 `entry_dec`（一次全局访存，串行在关键路径上）。改为 `BAR.RED.OR 0, P0` 在屏障的同时把“线程 0 是否登记排空”归约出来，只有可能排空时才读 | 5060 `sort` +13% → +7% |
| ④ 门控镜像 | 见 §4.4：消除数千 CTA 同读一条控制头缓存行的 L2 热点 | 5080 `sort` +14.8% → +3.3%、`rope`（纯 Mode 5）+54% → +7% |

最终（Suite 7，29 个算子）：Lv3 批量开销中位数 +0.3–0.5%，最高 +4.0–5.7%（`topk` / `sort`，随轮次）；关闭短 CTA 路由的纯 Mode 5 最高 +7–8%。

**未采用：实验 Guardian v2（EXP-1）**：把 `LDC/S2R/LD` 异步发出（计分板 SB2–SB5）以隐藏延迟，短 CTA 开销再降 20–35%，但在“双 Lv3 队列 + HP 流”（`LLM-08`）下 5080 出现 11/61 次错误（串行版本 0/78）。推测是 Trap 送达时 Guardian 发出的访存尚未完成；未查明前只作为实验开关（`XSCHED_SM120_GUARDIAN_V2=1`）。

新确认的 SASS 编码（用于 ③④，均经 `nvdisasm` 反汇编核对）：`BAR.RED.OR 0x0, P0` = `{0x7b1d, 0x000fec0000014800}`；`B2R.RESULT RZ, Pd` = `{0xff731c, 0x000e640000004000 | (pd&7)<<17}`；`ISETP.NE.AND` 带谓词源 `(ps<<23)|0x00705270`；`S2R SR_VIRTUALSMID` 编号 `0x43`；无描述符 `LD.E` 的偏移是 32 位有符号数（`0xffff8000` 反汇编为 `-0x8000`）；`LD.E.CONSTANT`（`0x109900`）与弱序 `LD.E`（`0x101900`）的通用地址形式在运行时报非法指令 715，不能用于 Guardian。修改固定延迟指令的控制字时只能追加等待位，不能缩短 stall（T-12）。

### 4.4 门控镜像（gate mirror）

**问题**：所有 CTA 启动时都读控制头的同一条缓存行（`cmd` 与 `is_restore_wave`）。5080 有 84 个 SM，短 CTA 内核每秒启动数百万 CTA，这条行成为 L2 热点；测量探针（把两次读取换成常数 0，会使抢占失效）把 5080 纯 Mode 5 的 `rope_apply` 从 +54% 降到 +5%，证明热点是主要成本。

**设计**：控制分配在头部之前增加 256 行 × 128 字节（每行独占一条缓存行），每行存 `{cmd, is_restore_wave}`。Guardian 用 `SR_VIRTUALSMID` 选行，一次 64 位读取：`LD.E.64 [u_ctrl − 0x8000 + 128 × smid]`。

**一致性规则**（违反它会出错，见 T-21）：镜像必须**严格跟随控制头的写入者**——
- `PreemptStream()`：只写各行 `cmd`（它读到的 `is_restore_wave` 可能已被清零内核改掉，是过期值）；
- `PrepareRestore()`：写 `{cmd, is_restore_wave=1}`；
- 恢复后条件清零内核：256 个线程先都读 `cmd`，屏障后各自清一行，线程 0 清控制头（保持 DEF-12 语义）；
- `ResetControlSync()` 与重新分配：全部清零。

调试：`XSCHED_SM120_GATE_MIRROR_CHECK=1` 在每次主机读取控制头时比对全部镜像行；`XSCHED_SM120_GATE_MIRROR=0` 关闭镜像（回到读控制头）。

---

## 5. 自适应路由

内核提交到 Level-3 队列时（`CudaQueueLv3Trap::OnHwCommandSubmit`），按以下顺序判断，命中则标记为不可中断（`killable=false`），挂起时退回 Level-1 边界：

| 顺序 | 条件 | 原因 | 统计名 |
|---|---|---|---|
| 1 | `XSCHED_SCHEDULER=GLB` 且关闭了就绪门控（`XSCHED_GLB_READY_GATE=0`）且未设 `XSCHED_SM120_GLB_LV3=1` | DEF-27：无门控时跨进程 Lv3 会触发 Xid 109（§9.4） | `lv1_glb_sched` |
| 2 | 协作启动（`grid.sync`） | 需要整个网格同时驻留，无法部分驱逐 | `lv1_cooperative` |
| 3 | 线程块 Cluster（多波，或单波且未设 `XSCHED_SM120_CLUSTER_MODE5=1`） | DEF-19：Cluster 屏障相位、DSMEM 映射不在 Trap 帧中，5080 上 1–2 个 CTA 偶发错 | `lv1_cluster` |
| 4 | 插桩失败，或含硬件 mbarrier（`SYNCS`）指令 | mbarrier 状态在共享内存中有硬件语义，无法保存 / 重建 | `lv1_uninstrumentable` / `lv1_mbarrier` |
| 5 | 无回边、无间接分支、无调用（`loop_free`），且波数在 `[2, 1024]` | 每条指令每个线程至多执行一次，CTA 很短；Level-1 边界的抢占延迟就是该内核本身的短时长，却省掉了每个 CTA 的 Guardian 成本（DEF-17） | `lv1_short_cta` |
| 6 | CUDA Graph | 图在驱动内部发射，无法逐内核插桩（DEF-09） | `lv1_graph` |

`loop_free` 的判定（`BinaryIsLoopFree()`）：扫描内核 SASS，任一 `BRA` 的偏移为负（回边）、任一 `BRX`（间接分支，`0x949`）、任一调用（`CALL.REL(.NOINC)` `0x943/0x944`、`CALL.ABS.NOINC` `0x343`、间接 `CALL.REL.NOINC Rx` `0x344`）或 `RET`（`0x950`）都判为非 loop-free。结尾填充用的 `BRA self` 忽略。`BRA` 偏移字段的编码通过 3,653 条分支拟合确认。

代价：被路由的内核在**自己结束时**才让出 GPU（5060 `transpose_copy` HP P50 1.09 → 2.37 ms）。需要严格亚毫秒抢占的部署可用 `XSCHED_SM120_SHORT_CTA_LV1=0` 关闭（门控镜像之后纯 Mode 5 也满足 < 10% 目标）。

调用检测同时决定 LMEM 保存区：含调用的内核要保存调用栈（§6.2）。DEF-22（`CALL.ABS` 漏检）与 DEF-24（间接调用 / `RET` 漏检）是这里的两个修复。

---

## 6. 显存

### 6.1 u_arena 槽位池

每个驱逐的 CTA 需要一个槽位：线程帧区（每线程：寄存器 + 硬件帧 + LMEM）+ SMEM 区。槽位数按“GPU 上同时驻留的 CTA 上限”而非网格大小分配，用 `evicted_slots` 原子计数取紧凑槽号。16,384 CTA 网格下比按网格分配节省 140–437 倍（手册 6.3 节有逐字节公式）。

### 6.2 LMEM 保存容量（DEF-23）

原实现对含调用的内核按每线程固定 4 KiB 预留栈保存区（LayerNorm 在 5060 上需要 231 MiB）。改为“静态局部内存 + `CU_LIMIT_STACK_SIZE`（运行时实际栈上限）”精确预留。`XSCHED_SM120_LMEM_CAP_BYTES` 可覆盖。

### 6.3 arena 空闲回收（MEM-1）

arena 随更大的内核增长，旧 arena 被替换后原来永不释放。`CudaQueueLv3Trap::Synchronize()` 之后调用 `TrimArenas()`：同步期间没有新发射、也没有待恢复的驱逐状态时，释放被替换的旧 arena；若当前 arena 在 64 次同步的窗口内一直大于需求的 2 倍且大于 64 MiB（`XSCHED_SM120_ARENA_TRIM_MIN_MB`），也释放当前 arena（下次发射重新按需分配）。`XSCHED_SM120_ARENA_TRIM=0` 关闭。回归项 `SYN-stress-trim` 以 1 MiB 阈值在抢占压力下验证。

---

## 7. 透传发射与参数窗口（PERF-2）

### 7.1 问题

加载 XSched 后，HP 流（不受 XQueue 管理）的每次发射也经过 shim。`perf` 采样显示真实 decode 中 `cuXtraKernelGetFunction`（`CUkernel → CUfunction` 转换）占全部样本的 8.4%：它在每个内核命令构造时执行，包括透传发射；另外 SM120 透传每次都调用一次 `cuXtraSetDebuggerParams` 清零参数窗口。单内核微基准看不出来，因为驱动对同一句柄的转换有缓存，而 decode 每步发射数百个不同内核（T-20）。

### 7.2 修复

- 命令在构造时不再解析 `CUfunction` 与参数个数，`FuncHandle()` 在真正需要时（Level-3 插桩 / 写参数）才解析；
- 记录哪些函数曾被写入非零参数窗口（`g_params_set`），透传发射只对这些函数清零（检查与发射在 `LaunchMutex` 内完成）；`XSCHED_SM120_ALWAYS_CLEAR_PARAMS=1` 恢复每次清零；
- `DirectLaunch` 的上下文 / 架构查询按线程缓存。

效果：5080 上 HP decode TPOT 相对无 XSched 由 1.23× 降到 1.01–1.06×；5060 仅加载 shim 的 decode 开销 +16% → +1.3–3.8%。

---

## 8. 非 root 抢占：`xsched-bar0d`

完整设计与安全模型见 [`SM120_V2_PHASE3_4_DEF10_BAR0_PROXY_ZH.md`](SM120_V2_PHASE3_4_DEF10_BAR0_PROXY_ZH.md)。要点：

- **范围确认**：非 root 进程中 Trap 处理程序打补丁、Guardian 插桩、控制内存全部正常，只有脉冲需要 root。
- **代理**：root 启动，只 `mmap` 两页 BAR0（`0x9000` 与 `0x424000`），建 `root:<group> 0660` 的 `SOCK_SEQPACKET` socket（`/run/xsched/bar0-<bdf>.sock`），然后 `setgroups(0)`、`setgid/setuid(nobody)`、`no_new_privs`。只接受 `PULSE` 与 `HELLO`；每连接令牌桶（5000 次/s、突发 64）；单线程事件循环串行化所有进程的脉冲。
- **客户端**（`Sm120Arena`）：能直接映射就直接映射；否则连代理，`HELLO` 校验 PCI BDF，断线后最多每 200 ms 重连一次。
- **代价**：一次 socket 往返 4.5–17 µs；同会话 A/B 挂起延迟 +2–15 µs。

---

## 9. 跨进程调度

### 9.1 架构

```
进程 A（HP）                      xserver（LocalScheduler + 策略 HPF）            进程 B（LP）
 XQueue Ready/Idle 事件 ──SysV msgq(key 0x1)──▶ 更新状态 → policy.Sched()
                                                  → 每个进程一个 SchedOperation
 SchedExecutor ◀──msgq(key = 进程 pid)──────────  {running 列表, suspended 列表}  ──▶ SchedExecutor
   对 running 调 Resume、对 suspended 调 Suspend                                     执行后回报 OperationComplete
```

HPF（最高优先级优先）：每个设备上找出“有就绪命令”的最高优先级，挂起所有更低优先级的队列，恢复其余队列。优先级用 `XHintPriority()` 设置，数值越大越高。事件、操作都经 SysV 消息队列传递：服务端接收队列键 `0x1`，每个客户端接收队列键 = 进程 pid。

### 9.2 OBS-7：单进程测试的 GIL 干扰

Suite 8 中 HP 与 LP 是同一 Python 进程的两个线程，按请求挂起时 HP TPOT 仍为 1.02–1.07×。多进程测试（`bench_mp_serving_p34.py`）把 HP / LP 分到两个进程后，同样的策略 TPOT 为 1.00×，证实残余来自 GIL 争用。

### 9.3 DEF-26：调度器线程没有 CUDA 上下文

全局 / 本地调度器在自己的执行线程上调用 `Suspend/Resume`，该线程从未设置 CUDA 上下文。Level-3 的 `Reactivate()` 要重新发射内核、分配 / 写控制内存，因而失败（`sm120 PrepareLaunch failed`）。修复：`CudaQueueLv3Trap::{Deactivate, Interrupt, Reactivate}` 在调用线程没有本队列上下文时临时压入（`CtxScope`）。

### 9.4 DEF-27：Trap 与时间片上下文切换的竞争，及就绪门控

**现象**：跨进程透明调度（`glb-lv3`）每次运行出现一次 Xid 109 `CTX SWITCH TIMEOUT`，驱动随之恢复，两个进程同时报 `cuda error 719`，HP 有请求被阻塞约 4.3 s；5080 上未出现超时但 TPOT P99 为 3.1–3.5×（19–22 ms 卡顿）。

**机制**：两个进程是两个上下文，GPU 按时间片在它们之间切换。HPF 下 HP 一变为 Ready 就继续发射，同时服务端异步挂起 LP。若 GPU 对 LP 上下文的切换请求恰好在 LP 线程束处于 Trap 处理程序中时到达，该上下文无法被切出，切换超时（约每 300 次挂起一次）。应用层“HP 等 LP 驱逐确认后再发射”的 `req-lv3` 从不触发；单进程内 HP / LP 共用一个上下文，也不存在。

**根本修复：就绪门控（ready gate，GLB 下默认开启）**

```
HP 客户端                         xserver                                   LP 客户端
 队列 Idle→Ready，Ready 事件编号 n ─▶ 处理事件（该进程 ready_seen = n）
 该命令不走直接提交，发射线程在         HPF：挂起 LP
 Dequeue() 中等待 ack ≥ n           ── 第 1 阶段：发送 LP 的操作 ───────────────▶ Suspend（Lv3：脉冲 + 等流排空）
                                    等待 LP 回报 OperationComplete（≤100 ms） ◀── OperationComplete
                                 ◀─ 第 2 阶段：发送 HP 的操作（ready_ack = n）
 执行操作后 ReadyGate::Ack(n) → 发射
```

- 客户端（`preempt/.../sched/ready_gate.{h,cpp}`）：每个 `XQueueReadyEvent` 都编号（编号与服务端计数一一对应）；门控开启时，使队列变 Ready 的那条命令不走直接提交，由发射线程在 `Dequeue()` 中等待确认（超时 `XSCHED_GLB_READY_GATE_TIMEOUT_US`，默认 10 ms，超时则照常发射并告警）。执行器在执行完每个 `SchedOperation` 后调用 `ReadyGate::Ack(op->ReadyAck())`。
- 协议：`SchedOperation` 新增 `ready_ack`（服务端发出该操作时已处理的、该进程的 Ready 事件数，来自 `ProcessStatus::ready_seen`）。
- 服务端（`LocalScheduler::SetSuspendFirst()`，仅 `xserver` 启用）：每轮先发送挂起操作；**其中挂起 Level-3 队列的进程**等它们回报完成，再发送其余进程的操作。Level-1/2 的挂起不等待（没有 Trap，等待只会推迟 HP）。本轮的 `to_suspend / to_resume` 集合在发送后清空。
- 默认：`XSCHED_SCHEDULER=GLB` 时客户端与 `xserver` 都默认开启，`XSCHED_GLB_READY_GATE=0` 关闭（两端须一致）。开启时 Level-3 在 GLB 下不再回退到 Level-1（§5 规则 1）。

**效果**：5060 上门控 Lv3 运行 0 次 Xid 109（无门控 4/4 次运行出现）；5080 TPOT P99 3.1–3.5× → 1.10–1.29×（10 次）；HP 门控等待均值 160–580 µs（即 LP 驱逐时间），0 次超时。对 Level-1 无可测影响（8×8 交替 A/B）。`XSCHED_GLB_READY_GATE_TRACE=1` 在进程退出时打印门控统计。

### 9.5 DEF-28：`xserver` 的 IPC 清理

`xserver` 原先只在 SIGINT / SIGQUIT 时删除消息队列；被 SIGTERM 结束时队列残留。root 运行留下的键 `0x1` 队列，普通用户启动的 `xserver` 无法删除（权限），客户端继续向这个无人接收的队列发事件，队列满后发送阻塞在发射路径里，进程挂起。修复：`xserver` 处理 SIGTERM；测试脚本用 SIGINT 结束服务，并且**只要键 `0x1` 的队列属于其他用户就拒绝启动**（包括 root——root 启动的服务端会删除另一用户正在使用的服务端队列，T-24）。

### 9.6 OBS-15：暂停的 Level-1 发射线程在等待日志空位后仍发射一条命令

`LaunchHwCommand()` 只在命令日志未满时检查暂停；日志已满时先同步最早命令腾空位，之后不再检查就发射。所以挂起期间可能漏出一条命令（Level-3 下该内核被 Guardian 立即排空，不影响正确性）。补做检查的改动经交替 A/B 无收益（`request` 策略无差别；`glb-lv1` 略差），已回退，保持上游行为。

### 9.7 OBS-14：显示 GPU 上的图形上下文

5060 同时驱动桌面。Lv3 运行期间四次出现桌面图形进程（`ghostty`、`gnome-shell`、`Xwayland`）的 Xid 31（`MMU Fault: ENGINE GRAPHICS GPC2 GPCCLIENT_GCC faulted @ 0x0`），进程均未崩溃。早先的计算型旁观进程测试（60 次脉冲）没有出错，说明受影响的是**图形**上下文。Level-3 现在在 `CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT != 0`（驱动显示器的 GPU）时启动告警一次；生产部署建议使用无显示的 GPU。

---

## 10. 调度策略与理论上限

### 10.1 单进程策略（`bench_real_serving_p3.py --policy`）

| 策略 | 何时挂起 LP | 特点（原负载，两卡） |
|---|---|---|
| `request` | 整个 HP 请求期间 | TPOT P50 1.00×；LP 吞吐 Lv1 69–78%、Lv3 56–68% |
| `prefill` | 只在 HP 的 prefill 期间 | LP 最高（77–95%），TPOT 1.3–1.7×，不推荐 |
| `token` | 每个 GPU 步前后 | 与 `request` 同样的 TPOT，LP 不高于 `request`（5060 单独测量：Lv1 67%、Lv3 52%；每次挂起 / 恢复都有成本），不推荐 |
| `slo` | prefill 挂起；decode 共置，首次超过 TPOT SLO 后挂起到请求结束 | P50 正常，P99 1.5–1.9× |

### 10.2 上限：`LP ≤ 1 − HP 占空比`

挂起式时间共享中，HP 请求期间 LP 完全不运行，所以 LP 吞吐（相对独占）不会超过 `1 − 占空比`。原负载（HP 平均间隔 500 ms，占空比 30–38%）下上限 62–74%，Lv1 已接近上限，Lv3 因驱逐 / 恢复成本略低。HP 间隔 1.5 s / 3 s 时 LP 79–95%。

### 10.3 空间共享尝试

用 CUDA green context（`cuDevSmResourceSplitByCount` + `cuGreenCtxCreate`）把 LP 限制在一部分 SM 上，HP 在主上下文的高优先级流上运行，不挂起 LP（`--lp-sms`；green context 流可直接被 PyTorch `ExternalStream` 使用）。原负载下 5060（LP 24/30 SM）LP 79–80%、TPOT P50 不变但 P99 1.44×；5080（LP 64–72/84 SM）LP 86–87% 但 P50 +27–43%。推测原因：green context 只划分 SM，不划分显存带宽与 L2。结论：原负载下“LP > 80% 且 TPOT 不劣化”对时间共享与空间共享都不可达。

### 10.4 跨进程

`req-*`（应用层按请求挂起，等确认）：TPOT 1.00×，LP 与单进程相当；`glb-*`（透明）：LP 71–78%（在 decode 步之间的 CPU 间隙里恢复 LP，可超过按请求挂起的上限），TPOT P50 0.95–1.04×，P99 在运行间波动大（Lv1 1.2–2.4×）。

---

## 11. 环境变量完整参考

### 11.1 Level-3 引擎与 Guardian

| 变量 | 默认 | 作用 |
|---|---|---|
| `XSCHED_SM120_SHORT_CTA_LV1` | 1 | 短 CTA 路由；`_MIN_WAVES` / `_MAX_WAVES` 默认 2 / 1024 |
| `XSCHED_SM120_GUARDIAN_BARRED` | 1 | `BAR.RED` norm_gate；0 = `BAR.SYNC` + 无条件读 `entry_dec` |
| `XSCHED_SM120_GATE_MIRROR` | 1 | 门控镜像；`_CHECK=1` 每次读头时比对镜像（调试） |
| `XSCHED_SM120_GUARDIAN_V1` | 0 | 1 = 原版 Guardian（含 slot 28 重读） |
| `XSCHED_SM120_GUARDIAN_V2` | 0 | 1 = 实验异步版本（EXP-1，不安全）；`_SERIAL_HDR` / `_SERIAL_SREG` 二分用 |
| `XSCHED_SM120_CLUSTER_MODE5` | 0 | 1 = 单波 Cluster 内核走实验性 Mode 5 |
| `XSCHED_SM120_DETECT_CALL_ABS` | 1 | 0 = 不把 `CALL.ABS` 视为调用 |
| `XSCHED_SM120_LMEM_CAP_BYTES` | 自动 | 覆盖每线程 LMEM 保存容量 |
| `XSCHED_SM120_LAYOUT_CACHE` | 1 | 布局缓存 |
| `XSCHED_SM120_ARENA_TRIM` / `_MIN_MB` | 1 / 64 | arena 空闲回收 |
| `XSCHED_SM120_ALWAYS_CLEAR_PARAMS` | 0 | 1 = 每次透传发射都清零参数窗口（PERF-2 之前的行为） |
| `XSCHED_SM120_FORCE_MODE1` | 0 | 1 = 全部用 Mode 1（仅幂等负载） |
| `XSCHED_SM120_L3_BACKEND` / `_L3_MODE` | `trap` | Level-3 后端：`trap`（本文所述 Trap 引擎）、`resumable` / `tsg`（Context CILP，上下文级）、`GDB`（调试器中断实验后端） |
| `XSCHED_SM120_TRACE_LAUNCH` | 0 | 打印每次发射、每次恢复的诊断信息 |
| `XSCHED_SM120_DUMP_DIR` | — | 插桩二进制转储目录（调试） |

### 11.2 BAR0 与代理

| 变量 | 默认 | 作用 |
|---|---|---|
| `XSCHED_SM120_BAR0_RESOURCE` / `BAR0_PATH` | — | 指定直接映射的 resource0 路径 |
| `XSCHED_SM120_BAR0_PROXY` | `/run/xsched/bar0-<bdf>.sock` | 代理 socket；`0` 关闭 |
| `XSCHED_SM120_BAR0_FORCE_PROXY` | 0 | 1 = 即使是 root 也走代理 |
| `XSCHED_SM120_BAR0_PROXY_TIMEOUT_MS` | 200 | 每次脉冲的回复超时 |

### 11.3 XQueue 与调度

| 变量 | 默认 | 作用 |
|---|---|---|
| `XSCHED_XQUEUE_DIRECT` | 1 | 直接提交 |
| `XSCHED_XQUEUE_CALLER_SYNC` | 1 | 调用方 `WaitAll` |
| `XSCHED_SPIN_US` | 30 | 发射线程 / 调用方自旋预算（µs） |
| `XSCHED_SCHEDULER` | APP | `APP` / `LCL` / `GLB` |
| `XSCHED_POLICY` | — | 本地调度器策略（如 `HPF`） |
| `XSCHED_GLB_READY_GATE` | GLB 下 1 | 就绪门控（客户端与 `xserver` 须一致） |
| `XSCHED_GLB_READY_GATE_TIMEOUT_US` | 10000 | 门控等待上限 |
| `XSCHED_GLB_READY_GATE_TRACE` | 0 | 退出时打印门控统计 |
| `XSCHED_SM120_GLB_LV3` | 0 | 门控关闭时仍在 GLB 下用 Level-3 |
| `XSCHED_SCHEDULER_SUSPEND_{SYNC_HWQ,WAIT_ALL,WAIT_IDLE}` | OFF | 调度器挂起队列时附加的标志 |
| `XSCHED_AUTO_XQUEUE*` | — | 自动为流创建 XQueue 及其级别、优先级、阈值（上游功能） |

---

## 12. 构建、部署与测试操作

### 12.1 构建

```bash
cmake -S xsched -B build -DCMAKE_BUILD_TYPE=Release -DPLATFORM_CUDA=ON -DSHIM_SOFTLINK=ON \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_CUDA_COMPILER=<cuda-12.9>/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=g++-13 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j && cmake --install build --prefix <install>
bash xsched/test/sm120_v2/eval/build_regress_bins.sh <env_file>     # 回归用 C++/CUDA 测试程序
```

运行时把 `<install>/lib`（含 `libcuda.so.1` 软链接到 shim）放在 `LD_LIBRARY_PATH` 最前面；native 对照用不含它的 `LD_LIBRARY_PATH`。环境文件模板：`xsched/test/sm120_v2/eval/regress_env.example.sh`。

### 12.2 回归

```bash
python3 regress_p3.py plan --env <env_file> --suites full --result-dir <dir> > plan.txt
sudo -S bash p3_guarded_run.sh plan.txt run.log        # 每个条目前检查 GPU 上无他人进程
python3 regress_p3.py summarize run.log
```

`full` = 合成（`SYN`，含 `SYN-xq-idle`、`SYN-stress-trim`）、Suite 1（库 + 硬件特性，含 `LIB-08`×30）、4、5（DNN）、6（LLM）、7（29 算子 × 5 模式 + 纯 Mode 5）、8（真实服务）、9（多进程）、验收组 `ACC`；另加低负载组 `LL`。机制套件关闭短 CTA 路由以保持 Mode 5 覆盖。共享主机上：只有 GPU 与 IPC（键 `0x1` 队列）都空闲才开始，证据中的他人进程信息须脱敏。

### 12.3 代理部署

```bash
sudo install -m 0755 xsched-bar0d /usr/local/bin/ && sudo groupadd -r xsched && sudo usermod -aG xsched <user>
sudo cp xsched-bar0d@.service /etc/systemd/system/ && sudo systemctl enable --now xsched-bar0d@0000:01:00.0
```

### 12.4 跨进程运行

```bash
XSCHED_SCHEDULER=GLB xserver HPF 50000 &                  # 就绪门控默认开启
XSCHED_SCHEDULER=GLB python hp.py   # HP：XQueue + XHintPriority(xq, 1)
XSCHED_SCHEDULER=GLB python lp.py   # LP：XQueue(level 3) + XHintPriority(xq, 0)
kill -INT %1                        # 或 SIGTERM（DEF-28 之后也会清理 IPC）
```

参考实现与测量：`xsched/test/sm120_v2/eval/bench_mp_serving_p34.py`（6 种模式，`--lp-env` 只给 LP 进程设置变量）。
