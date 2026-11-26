# Phase 3.4-1：DEF-10 非 root Level-3 抢占（BAR0 脉冲代理 `xsched-bar0d`）

> 日期：2026-11-26 · 分支 `phase3-def20-production-testing` · 硬件：RTX 5060（本地）、RTX 5080（共享服务器）
> 结论：**DEF-10 已修复**。普通用户进程通过一个只能发出抢占脉冲的特权代理获得 Level-3 抢占；代理映射两页 BAR0 后降权为 `nobody`。非 root 验证 5060 37/37、5080 50/50（含 `LLM-08` 24/24、双进程并发、守护进程中途重启），全部比特一致；抢占延迟比 root 直接映射多 2–15 µs。

---

## 1. 问题

Level-3 中唯一需要 root 的操作是“驱逐脉冲”：向 BAR0 的 Trap 寄存器（`0x424684`）写 `STOP_TRIGGER`，读一次 `NV_PTIMER_TIME_0`（`0x9400`）刷新，再撤销。此前进程必须以 root 映射整个 `/sys/bus/pci/.../resource0`（16 MiB）。

非 root 实测（RTX 5060，修复前）：Trap 处理程序打补丁、Guardian 插桩、控制内存全部正常，只有脉冲失败；XSched 退化为“等当前内核跑完”，结果仍正确，但挂起延迟从约 0.1 ms 变为 9–16 ms（`test_partial_exit_p3`）。所以 DEF-10 的范围就是这一个寄存器操作。

## 2. 设计

```
 普通用户进程（XSched Lv3）                     xsched-bar0d（root 启动 → nobody）
 Sm120Arena::FireEvictPulse()                   单线程 epoll 循环
   ├─ root：直接映射 BAR0（原路径）             ├─ 只映射 BAR0 两页：0x9000、0x424000
   └─ 否则：SOCK_SEQPACKET  ── PULSE ──▶        ├─ 令牌桶限流（每连接）
             /run/xsched/bar0-<bdf>.sock ◀── OK ─┴─ FireBar0Pulse()（与直接路径同一实现）
             (root:<group> 0660)
```

| 方面 | 做法 |
|---|---|
| 授权 | Socket 文件 `root:<group> 0660`，只有该组成员能连接（与 `docker.sock` 同一模式）；`SO_PEERCRED` 记录 pid / uid |
| 能力范围 | 只有 `PULSE` 与 `HELLO` 两个操作；客户端不能指定寄存器地址或值 |
| 守护进程自身 | 映射两页、创建 socket 后 `setgroups(0)`、`setgid/setuid(nobody)`、`PR_SET_NO_NEW_PRIVS`、不可 dump；实测 `CapEff=0`、`NoNewPrivs=1`，`/proc/<pid>/maps` 中只有两页 `resource0` |
| 防滥用 | 每连接令牌桶（默认 5000 次/秒、突发 64），超限回复 `kRateLimited` 并记录日志 |
| 多进程互斥 | 单线程事件循环串行处理，不同进程的脉冲序列不会交错 |
| 防接错 GPU | 客户端连接时发 `HELLO`，代理返回其服务的 PCI BDF；与本上下文的设备不一致则拒绝使用 |
| 故障处理 | 请求超时（默认 200 ms）或连接断开 → 关闭连接，本次脉冲按失败处理（退化为等待内核结束），下一次脉冲重连（最多每 200 ms 尝试一次）；失败告警每秒最多一条 |
| 选择顺序 | 能直接映射 BAR0（root）就直接映射；否则使用代理；两者都不可用时保持原有的退化行为 |

协议定义与脉冲实现放在共享头文件 `hal/include/xsched/cuda/hal/level3/sm120_bar0_proxy.h`，直接路径与代理调用同一个 `FireBar0Pulse()`，保证脉冲序列（含 D10 硬件规则：`STOP_TRIGGER` 有效期间不读 PRI 广播寄存器）完全一致。

新增 / 修改的文件：

| 文件 | 内容 |
|---|---|
| `platforms/cuda/bar0d/xsched_bar0d.cpp` | 守护进程（约 300 行，无 CUDA 依赖） |
| `platforms/cuda/bar0d/xsched-bar0d@.service`、`README.md` | systemd 模板单元（实例名 = BDF，未安装到任何主机）与安全模型说明 |
| `hal/include/.../sm120_bar0_proxy.h` | 协议、寄存器常量、`FireBar0Pulse()` |
| `hal/src/level3/sm120_arena.{h,cpp}` | `EnsureBar0Mapped()` = 直接映射 → 代理；`ConnectProxyLocked()`、`ProxyRoundTrip()`；锁顺序 `mu_ → proxy_mu_` |
| `hal/src/level3/sm120_trap_engine.cpp` | 脉冲失败告警改为每秒最多一条并带累计次数 |
| `test/sm120_v2/eval/test_bar0_proxy_p34.py` | 协议检查（只用 `HELLO` 与畸形请求，不触碰硬件） |

环境变量：`XSCHED_SM120_BAR0_PROXY`（socket 路径，默认 `/run/xsched/bar0-<bdf>.sock`；`0` 关闭）、`XSCHED_SM120_BAR0_FORCE_PROXY=1`（即使是 root 也走代理，用于测试）、`XSCHED_SM120_BAR0_PROXY_TIMEOUT_MS`（默认 200）。

## 3. 验证

所有非 root 测试以普通用户身份运行（计划中每条命令先断言 `id -u ≠ 0`），短 CTA 路由关闭（保证走 Mode 5）。5080 上守护进程只在测试窗口内运行，socket 组为测试用户自己的组，结束后停止并删除 `/run/xsched`；每个条目前仍检查 GPU 上无其他用户进程（本轮 0 次让路）。

### 3.1 功能与压力

| 测试 | RTX 5060 | RTX 5080 |
|---|---:|---:|
| 协议检查（`HELLO`、4 种畸形请求、畸形请求后连接存活、64 个并发连接） | 通过 | 通过 |
| `test_partial_exit_p3 all 5 3` | 4/4 | 4/4 |
| 双进程同时抢占（`partial_exit` + `convbar C1 ×10`） | 通过 | 通过 |
| 多波次压测 `GRID_MULT=8 CHAIN=6` ×3、`GAP_US=300`、12 内核链 | 5/5 | 5/5 |
| `LIB-01/04/06/08`、`HW-01/03/06` | 7/7 | 7/7 |
| `LLM-08`（双 Lv3 队列 + HP，真实 Qwen2.5-0.5B） | 12/12 | **24/24** |
| `DNN-06` | 2/2 | 2/2 |
| 生产算子 Lv3（8 个，含 `sort` / `topk` / cuBLAS / cuDNN / SDPA） | 8/8 比特一致 | 8/8 比特一致 |
| **合计** | **37/37** | **50/50** |

（5060 计划第一次运行中有 2 个条目因计划文件里的算子名写错而在启动时退出，改正后通过；不是被测系统的问题。）

### 3.2 故障注入（RTX 5060）

| 场景 | 结果 |
|---|---|
| 无守护进程 | 测试通过；挂起退化为等待内核结束（9–16 ms） |
| 用户不在授权组（`--group root`） | `connect` 返回 `EACCES`，告警一次，测试通过（退化） |
| 限流（`--rate 2 --burst 2`） | 2 次脉冲成功，14 次被拒（`status -2`），测试通过（退化），守护进程记录限流 |
| 代理服务另一块 GPU（伪造 `HELLO` 返回 `0000:99:00.0`） | 客户端拒绝使用该代理，测试通过（退化） |
| socket 路径超过 107 字节 | 告警一次，不使用代理 |
| 压测进行中杀掉守护进程，3 s 后重启（同一进程内） | 杀前 1,980 次脉冲；中断期间 1,089 次脉冲失败并退化；重启后自动重连再发 4,131 次；12/12 通过，无非法指令 |

### 3.3 开销

| 指标 | RTX 5060 主机 | RTX 5080 主机 |
|---|---:|---:|
| 代理往返（`HELLO`，连续请求）P50 / P99 | 12.0 / 15.6 µs | 4.5 / 4.9 µs |
| 往返（请求间隔 50 ms，守护进程处于睡眠）P50 | 16.9 µs | — |

同一会话交替 A/B（RTX 5060，root 进程，直接映射 vs `XSCHED_SM120_BAR0_FORCE_PROXY=1`，每种 3 次，取中位数）：

| 算子 | 挂起 P50 直接 → 代理 | HP 端到端 P50 直接 → 代理 |
|---|---:|---:|
| `gemm_bf16_4k` | 124 → 136 µs（+12） | 253 → 262 µs（+9） |
| `sdpa_prefill_causal` | 112 → 115 µs（+2） | 220 → 224 µs（+4） |
| `conv3x3_nhwc` | 153 → 168 µs（+15） | 327 → 278 µs（−49） |
| `softmax` / `layernorm` | 239 → 214 / 224 → 174 µs | 376 → 348 / 369 → 307 µs |

`gemm` 与 `sdpa` 的三次运行都很集中，代理带来 +2–12 µs，与一次 socket 往返相符；其余算子的运行间波动（约 ±50 µs）大于代理开销。跨运行比较（非 root 代理 vs 此前 root 回归）的中位数差为挂起 +22 / +27 µs、HP +12 / +17 µs（5060 / 5080），同样处在运行间波动范围内。

### 3.4 回归

root 路径的脉冲代码同样被改写（共享 `FireBar0Pulse()`、直接 / 代理选择），因此在两卡上重跑全量回归（以 root 运行，走直接映射）：结果见 §5。

## 4. 局限

- **组成员身份等同于调度特权**：脉冲是整块 GPU 的广播。此前的跨进程实测（另一进程发 60 次脉冲期间，不使用 XSched 的旁观进程持续发射内核）未见错误或变慢（[`SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md`](SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md)）；但在同时驱动桌面的 RTX 5060 上，有两次多波次压测开始时桌面图形进程出现 Xid 31 MMU 故障（进程未崩溃，OBS-8，见 [`SM120_V2_PHASE3_4_MULTIPROCESS_ZH.md`](SM120_V2_PHASE3_4_MULTIPROCESS_ZH.md) §2.4），建议 Lv3 使用无显示的 GPU。此外组成员仍可在限流允许的频率内让 GPU 反复进入 Trap。共享 GPU 上应只授予作业由同一调度器管理的用户。
- 每次脉冲多一次 socket 往返（4.5–17 µs）。字符设备内核模块可以把它降到一次系统调用，但需要在共享服务器上加载自定义内核模块，风险与维护成本都更高，暂不采用。
- 代理不验证“调用者确实有被 XSched 插桩的内核在运行”；限流只限制频率。
- 每块 GPU 一个守护进程实例（`--bdf`）；多 GPU 主机用 systemd 模板单元。

## 5. 全量回归（最终构建）

两卡全量回归（以 root 运行，走直接映射路径；`regress_p3.py --suites full` + 低负载组）：**RTX 5060 265/265、RTX 5080 265/265**。Suite 7 Lv3 两卡 29/29 比特一致（默认与纯 Mode 5 两种配置），Lv3 批量开销中位数 +0.5% / +0.3%、最高 +5.7% / +4.9%，与上一轮（门控镜像）一致；5080 上 0 次让路。证据：[`evidence/phase3_4_def10_20261126/`](evidence/phase3_4_def10_20261126/)（非 root 两卡日志、故障注入、A/B、回归日志与汇总）。
