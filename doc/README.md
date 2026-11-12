# Blackwell SM120 细粒度硬件 Trap 空间抢占与恢复架构 — 核心技术问答专题 (`doc/`)

本目录专门收录针对 **Blackwell (`SM120` / `GB206`) 硬件级微秒抢占与恢复系统** 关键原理、底层接口、代码洞微架构与性能边界的深度 Q&A 专题文档，每个文件完整阐述一个核心专题：

---

## 专题目录

1. **[`QA01_TRAP_TRIGGER_AND_RM_MECHANISM_ZH.md`](QA01_TRAP_TRIGGER_AND_RM_MECHANISM_ZH.md)**
   - **专题一：原论文 Trap 触发机制失效根因、RM 接口可用性与 Blackwell SM120 新型硬件触发机制全解**
   - 涵盖内容：为什么 `cuXtraTriggerTrap` (`0x00419e84`) 在 Blackwell 上返回 `0xbadf1100` 总线中毒？`0x20800122` / `0x83de0317` / `0x20801210` 等 RM 函数可用性与 56 字节 ABI 修复；新触发地址 `BAR0 + 0x00424684` 与双写脉冲 (`0x0004007f -> 0x0004807f`) 是如何通过 Open-RM `595.99.02` 逆向找到的？进程内钉住 `NV83DE` 配合 BAR0 MMIO 实现 `1.89 µs` 触发的完整原理与测试对比。

2. **[`QA02_CODE_CAVES_AND_HOST_KERNEL_COMMUNICATION_ZH.md`](QA02_CODE_CAVES_AND_HOST_KERNEL_COMMUNICATION_ZH.md)**
   - **专题二：可编辑代码洞（Code Caves）定位、主机-设备双向通信、Target ID 过滤与 Kernel 元数据获取全解**
   - 涵盖内容：为什么 `cuXtraInstrMemBlockAlloc` 会导致 Context Cancel (`Error 719`)？三级代码洞（`ext_code` @ `tools_pc + 0x0980`、`ext2_code` + `entry_tramp` @ `handler_va + 0x0860`、`ext3_code` @ `handler_va + 0x2520`）的精确虚拟地址与容量分布；如何通过 `control_va` 全局显存双向交换命令与完成计数；如何利用 `c[0x0][0x170..0x18f]` 绑定 Per-Stream Target ID 并在代碼洞内读取 `blockDim`、`gridDim`、`num_regs` 与 `SR_CTAID`/`SR_TID`。

3. **[`QA03_STATE_SAVING_LAYOUT_AND_CAPABILITY_BOUNDARIES_ZH.md`](QA03_STATE_SAVING_LAYOUT_AND_CAPABILITY_BOUNDARIES_ZH.md)**
   - **专题三：全状态保存机制、显存地址布局、可保存状态全集与不可保存边界深度剖析**
   - 涵盖内容：运行中 CUDA Kernel 的 8 大类架构状态盘点；代码洞内分步保存 `112B` 硬件 Trap 栈帧（`local[0xffff90..0xffffff]`）、`64 KiB` Shared Memory（跳过 `0x400` 驱动保留头部）、`UR4..UR15`、`B2R.WARP` 0 号硬件屏障、以及 `R14..R252` 全架构寄存器（遵循 `SR_REGALLOC - 2` 铁律）的完整机制；`u_arena` 显存池与每线程 `1,152 B` (`0x480`) 栈帧逐字节偏移表；已支持与暂不支持状态（如用户态 LMEM Stack Spill、在途 TMA/`cp.async`）的清晰界定。

4. **[`QA04_RESTORE_STATE_MACHINE_CALL_FLOW_AND_LATENCY_ZH.md`](QA04_RESTORE_STATE_MACHINE_CALL_FLOW_AND_LATENCY_ZH.md)**
   - **专题四：抢占与恢复状态机更新逻辑、端到端调用方式与分步耗时/效率定量拆解**
   - 涵盖内容：四层核心状态机（全局命令字 `0/5/7/6`、每线程幂等 `status` `0/5/6`、多波次三态 `(w, ack)` `State A/B/C` 条件入口分流、多 Warp `W -> 2*W` 协同 SMEM 屏障与 `BAR.SYNC.DEFER_BLOCKING 0x0` 重建）；`TrapSm120Context` 5 大 C++ API 签名与标准调用代码范式；抢占与恢复每一步耗时的微秒级定量拆解与全模式效率对比表。

---

## 关联核心文档

- [《Blackwell SM120 通用零源码修改细粒度空间抢占与硬件恢复架构白皮书》](../mmio/spill/UNIVERSAL_TRAPFRAME_ARCHITECTURE_ZH.md)
- [《Blackwell SM120 Level 3 空间配额抢占（Mode 4）完整实现原理与 Mode 5 探索详解》](../mmio/spill/MODE4_IMPLEMENTATION_ZH.md)
- [《XSched Blackwell (SM120) Level 3 细粒度抢占研究与工程实现全景总结报告》](../BLACKWELL_SM120_LEVEL3_SUMMARY_ZH.md)
