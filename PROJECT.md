# PROJECT.md — Blackwell (`sm_120`) XSched Level-3 v2 工程说明

## 1. 项目定位与研究范围

- **项目目标**：基于 XSched 框架开展面向 **NVIDIA Blackwell (`sm_120`)** 架构的用户端透明 GPU 空间抢占与调度学术研究，重点攻克**不依赖内核幂等性假设、零内核源码修改**的硬件 Trap 级细粒度抢占与原位状态恢复（Mode 5 `kTrapFrame`），并与幂等早退（Mode 1 `kEarlyExit`）及上下文级 CILP（`ResumableSm120Context`）形成完整 Level-3 抢占体系。
- **编译与代码标准**：`C++20`，`-fPIC -Wall -Wextra -Werror` 全目标零警告构建；CUDA 设备端测试目标采用 `CUDA 12.9` + `sm_120` 编译。

---

## 2. 硬件与软件基线环境

| 角色 | GPU 型号与架构 | SM / SMEM 规格 | 驱动与 CUDA 工具链 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **主验证与构建机 (`5080`)** | NVIDIA GeForce RTX 5080 (`GB203`, `sm_120`) | 84 SMs / `100 KiB` SMEM/SM | Driver `580.95.05`, CUDA `12.9` (`/home/wangyue/.local/cuda-12.9`), `g++-13` | 无头计算服务器，主构建与 11/11 E2E/Bench 基准验证平台 |
| **本地交叉验证机 (`5060`)** | NVIDIA GeForce RTX 5060 (`GB206`, `sm_120`) | 30 SMs / `100 KiB` SMEM/SM | Driver `595.99.02`, CUDA `12.9`, `g++-13`, PyTorch `2.11+cu128`, Ubuntu 26.04 | 桌面独显环境，`v2.2.0` 全方位生产加固（Suite 1–3, 15/15 库特性 + 11/11 PyTorch）实机验证平台 |

---

## 3. 精简仓库目录结构

```text
.
├── README.md                                  # 项目总览与构建/回归快速入口
├── PROJECT.md                                 # 工程定位、模块清单与运维安全规范
├── CHANGELOG.md                               # 版本变更与修复日志
├── doc/
│   ├── README.md                              # 文档索引
│   ├── SM120_V2_ARCHITECTURE_ZH.md            # v2 静态 9-Cave 引擎与状态机架构详解
│   ├── SM120_V2_VERIFICATION_AND_BENCHMARK_ZH.md # 实机验证、基准数据与微架构发现
│   ├── SM120_V2_COMPREHENSIVE_EVALUATION_ZH.md # 闭源库/DNN/LLM/PyTorch 全方位评测与缺陷剖析
│   ├── SM120_V2_STATUS_PROBLEMS_AND_ROADMAP_ZH.md # 阶段状态汇报、10 项问题台账与 Phase 1-3 未来规划
│   └── SM120_V2_RTX5060_VS_RTX5080_DIFF_ZH.md # RTX 5060 vs RTX 5080 跨芯片/跨驱动差异对比
└── xsched/
    ├── CMakeLists.txt                         # XSched 顶层构建配置 (C++20)
    ├── 3rdparty/cuxtra/                       # 底层 CUDA ToolsMemory 与 EntryPoint 操纵库
    ├── docs/                                  # XSched OSDI'25 论文与原始说明文档
    ├── include/, preempt/, protocol/, sched/, service/, utils/, tools/
    ├── platforms/cuda/
    │   ├── CMakeLists.txt                     # CUDA HAL + 5 项 SM120 v2 测试/基准目标
    │   └── hal/
    │       ├── include/xsched/cuda/hal/level3/
    │       │   ├── sm120_sass.h               # 79 条 SM120 128-bit SASS 编码器
    │       │   ├── sm120_arena.h              # .text.traphandler 探测与 BAR0 0x00424684 触发
    │       │   ├── sm120_guardian.h           # 首次发射永久 0x500 B Entry Guardian
    │       │   ├── sm120_trap_engine.h        # Context 级静态 9-Cave 引擎 + Per-Stream 上下文
    │       │   ├── sm120_rm_handles.h         # 底层 RM ioctl 句柄捕获与管理
    │       │   ├── resumable_sm120.h          # Context 级 CILP 独立可恢复抢占模块
    │       │   └── cuda_queue.h               # CudaQueueLv3Trap / CudaQueueLv3Resumable
    │       └── src/level3/
    │           ├── sm120_arena.cpp
    │           ├── sm120_guardian.cpp
    │           ├── sm120_trap_engine.cpp
    │           ├── sm120_rm_handles.cpp
    │           ├── resumable_sm120.cpp
    │           ├── cuda_queue.cpp
    │           └── interrupt.cpp
    └── test/
        ├── sm120_v2/
        │   ├── test_sass_disasm.cpp           # sm120_sass_disasm: 79 条指令 nvdisasm 校验
        │   ├── cave_probe.cu                  # sm120_cave_probe: 代码洞与 9-Cave 初始化探测
        │   ├── sm120_v2_e2e_suite.cu          # sm120_v2_e2e_suite: 11 项端到端正确性与压测套件
        │   └── sm120_v2_bench.cu              # sm120_v2_bench: Guardian/Evict/Restore/显存基准
        └── sm120_l3_resumable/
            └── sm120_l3_resumable_bench.cu    # sm120_l3_resumable_bench: CILP 独立基准
```

---

## 4. 关键工程与硬件安全规范

1. **共享 GPU 安全红线**：`0x00424684` (`NV_PGRAPH_PRI_GPCS_TPCS_SM_SFE_DBGR_CONTROL0`) 为全芯片广播硬件中断寄存器。在共享服务器运行前**必须**执行 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`，严禁在他人 GPU 计算任务运行时触发 BAR0 脉冲。
2. **严禁盲扫 BAR0 与置位期读 `0x424684`（D9 / D10 硬件红线）**：
   - 仅允许访问已证实安全的 `0x00424684`（Trap 广播）与 `0x00009400`（`NV_PTIMER_TIME_0` 非 PRI 刷新读）。
   - 在 `0x00424684` 置位 `0x80001c01` 期间严禁读取 `0x00424684` 或其他 PGRAPH PRI 寄存器，须使用非 PRI 的 `NV_PTIMER_TIME_0` 刷新首个 Posted Write 后再写 `0x00001c01`。
3. **非入侵与零造假准则**：所有基准与端到端测试均动态链接 `libhalcuda.so` 与 `libpreempt.so`，在未修改的真实 GPU 内核上触发真实硬件 Trap 与 `BPT.TRAP 0x1` 状态恢复；所有延迟与正确性指标均来自真实设备执行结果。
4. **反回退架构守则**：严禁回退至 v1 阶段的 Mode 2/3/4、`cmd=7`、运行时 `func_pc` 热补丁或 2,000 次盲目 BAR0 轮询；仅在极少数极端并发/共享上下文进度停滞窗口（`>= 15 ms`）触发严格有界（`<= 8` 次）且受 `exit_stub_va` 重入保护的恢复确认脉冲。
