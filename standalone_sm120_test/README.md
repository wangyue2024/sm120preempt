# Blackwell SM120 (RTX 5060 / RTX 5090 & Other GPUs) Level-3 Hardware Preemption & Universal Trap-Frame Standalone Suite

本目录是一份**100% 自包含、零外部路径依赖、可直接拷贝至任意机器（如 RTX 5090 或其他架构 GPU 服务器）编译并运行**的完整硬件抢占与现场恢复集成测试包。

它集成了：
1. **五种上下文内（Intra-Context）细粒度硬件 Trap 抢占与恢复模式（Mode 1 ~ Mode 5）**：
   - **Mode 1**：Stream 级快速排空退出（Early-Exit）与 `blockIdx` 执行掩码记录
   - **Mode 2**：无源码修改、非破坏性原地暂停与恢复（Normal-Mode Yielding Trampoline）
   - **Mode 3**：单 Kernel 内部逐 CTA 选择性混合调度（指定 `cta_id` 原地暂停或退出腾退）
   - **Mode 4**：2.45 µs 硬件 Trap 异步信令 + 零稳态开销（`O1 = 0%`）安全点空间腾退与重入恢复（覆盖高寄存器、64 KB 共享内存、高并发多块、2D Tiled FP32 SGEMM）
   - **Mode 5（Universal Mode 5 通用零源码修改任意 PC 硬件抢占与 `RTT` 原子恢复）**：
     - **全量架构寄存器覆盖（`1 ~ 255 regs/thread`）**：通过三级串联代码洞（`Cave 1 ext_code` + `Cave 2 ext2_code` + `Cave 3 ext3_code`，共 316 条指令）保存并恢复 112B Hardware Trap Frame（`PC`、`R0..R15`、`P0..P6`、`CS2R`）+ `R16..R252` 全量线程寄存器 + `UR4..UR43` 统一寄存器 + 最高 `64 KB/CTA` Shared Memory。
     - **多波次 Grid 三态精确恢复（Multi-Wave 3-State Restore）**：精确区分已自然完成的先序波次（`State A = 4`）、被 Trap 抢占腾退的在途波次（`State B = 5 -> 6`）与被排空跳过的后续波次（`State C = 7`），确保非幂等 Kernel 零重复累加、零数据损坏。
2. **跨上下文（Cross-Context）硬件级 CILP 指令级抢占与恢复引擎（`sm120_l3_resumable_bench`）**：
   - 基于 RM 控制调用 `0x20801210`（CILP/CTA Preemption Mode）+ `0x83de0317/0x83de0318` + `0xa06c0105`。
3. **跨显卡/跨架构硬件探测与代码洞兼容验证工具（`cave_probe`）**：
   - **无需 root 权限**即可自动探测目标显卡（如 RTX 5090 `170 SMs`）的硬件参数、256 MiB CUDA Context Internal Arena 布局、`.text.traphandler` 基址以及全部 4 个硬件 Trap 代码洞（`Cave 0 ~ Cave 3`）的容量与兼容性。

---

## 1. 快速开始（拷贝到 RTX 5090 或新机器后如何运行）

### 方式 A：一键全自动编译并运行全部 7 阶段集成测试
```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```
> `run_all_tests.sh` 会自动：
> 1. 提权至 `sudo`（仅 BAR0 MMIO `0x00424684` 全芯片 Trap 脉冲广播需要 root 权限）；
> 2. 扫描 `/sys/bus/pci/devices/` 自动识别当前机器上 NVIDIA GPU 的 PCIe Bus ID 与 `resource0` 物理基址（无需手动修改 PCI 地址）；
> 3. 调用 `Makefile` 自动检测当前 GPU 的 Compute Capability（如 `sm_120`）并编译全部 12 个二进制工具；
> 4. 依次执行全部 7 个阶段的硬件探测、延迟基准、100% 满载压力测试与 Mode 1~5 / Cross-Context 全量验证。

### 方式 B：分步独立运行（推荐在新显卡上先跑无 root 探测）

1. **编译全部测试程序（无需 root）**：
   ```bash
   make clean all
   ```
   *(若测试其他架构显卡，可通过 `make GPU_ARCH=sm_90 all` 指定架构)*

2. **第一步（无需 root）：探测新显卡架构与 4 级代码洞兼容性**：
   ```bash
   ./cave_probe
   ```
   - 若要将新显卡运行时的 `.text.traphandler` 与 `tools` 机器码导出为二进制文件以便用 `sdis.py` 反汇编分析：
     ```bash
     ./cave_probe --dump
     python3 sdis.py live_handler.bin 0 0x900
     ```

3. **第二步（无需 root）：运行跨上下文硬件 CILP 抢占测试**：
   ```bash
   ./sm120_l3_resumable_bench
   ```

4. **第三步（需要 sudo）：运行 Mode 4 + Universal Mode 5 核心基准测试套件**：
   ```bash
   sudo ./checkpoint_bench all 5
   ```
   也可以单独运行指定子项：
   - `sudo ./checkpoint_bench trapframe 5`：测试 Mode 5 基础版 + **Universal Mode 5 单波次（`1w`）、双波次三态恢复（`2w`）、255 寄存器极限负载（`hireg`）**
   - `sudo ./checkpoint_bench sgemm 5`：测试真实 2D Tiled FP32 SGEMM（64 KB SMEM/CTA）空间抢占与 `O1` 零稳态开销
   - `sudo ./checkpoint_bench registers 5` / `shared 5` / `warps 5`：测试 Mode 4 三类硬件资源瓶颈下的空间腾退与恢复

5. **第四步（需要 sudo）：运行全芯片满载竞争与 50 轮暂停/恢复压力测试**：
   ```bash
   sudo ./preempt_rigorous_bench
   sudo ./hal_fine_grained_test
   sudo ./stream_preempt_test
   sudo ./trap_latency_test 6
   ```

---

## 2. 针对 RTX 5090 (`170 SMs`) 及其他显卡的自适应设计说明

本独立测试包已完成以下多显卡/多 SM 规模自适应改造：
- **PCIe BAR0 自动发现**：`bar0.h` 与 `libhalcuda.so` (`EnsureBar0Mapped`) 自动遍历 `/sys/bus/pci/devices/*/vendor` (`0x10de`) 与 `class` (`0x03*`)，自动适配任意 PCIe 插槽并在 `resource0` 不可用时自动读取 `resource` 物理基址回退至 `/dev/mem`。
- **动态 SM 数量感知 (`FG.sm_count`)**：
  - `TrapSm120Context::InstallFineGrainedTrapHandler` 动态查询 `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`（RTX 5060 为 `30 SMs`，RTX 5090 为 `170 SMs`）。
  - `RequestStreamTrapFrameEvict` 在多波次抢占时动态等待 `FG.sm_count` 个在途驻留 CTA 完成全量现场转储（`ack=5`）后，再切换 `cmd=7` 排空后续未执行波次。
  - `checkpoint_bench` 与 `preempt_rigorous_bench` 的 Grid 规模均根据 `gpu.multiProcessorCount` 动态缩放（`1 * SM`、`2 * SM`、`8 * SM`）。
- **64 MiB 线程帧全局 Arena 分区 (`0x4000000`)**：
  - 每个线程的寄存器/Trap Frame 槽位固定为 `1,152 B (0x480)`，Shared Memory 转储区基址设为 `dump_base + (cta_id + 1024) * 65536`，即前 `64 MiB` 专用于存放线程帧，可容纳最多 **58,254 个并发线程（455 个 128 线程 CTA）**，完整覆盖 **RTX 5090 (170 SMs) 的双波次（340 CTAs = 43,520 线程）** 全量现场转储而不发生任何地址重叠。

---

## 3. 目录文件清单

| 文件 / 目录 | 说明 |
| :--- | :--- |
| [`run_all_tests.sh`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/run_all_tests.sh) | 一键编译并运行全部 7 阶段集成验证脚本 |
| [`Makefile`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/Makefile) | 自动检测 CUDA 路径与 GPU 架构、`DT_RPATH ($ORIGIN/lib)` 零外部依赖构建脚本 |
| [`cave_probe.cu`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/cave_probe.cu) | **免 root** GPU 硬件参数、Internal Arena 与 4 级代码洞容量/兼容性探测工具 |
| [`checkpoint_bench.cu`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/checkpoint_bench.cu) | **Mode 4 空间腾退 + Universal Mode 5（1~255 寄存器 + 多波次三态恢复 + 64KB SMEM）+ 2D Tiled SGEMM** 核心基准套件 |
| [`sm120_l3_resumable_bench.cu`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/sm120_l3_resumable_bench.cu) | **免 root** 跨上下文硬件 CILP (`0x20801210` + `0x83de0317`) 抢占与恢复基准测试 |
| [`hal_fine_grained_test.cu`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/hal_fine_grained_test.cu) | XSched `libhalcuda.so` `TrapSm120Context` C++ API 集成测试（Mode 1/2/3） |
| [`preempt_rigorous_bench.cu`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/preempt_rigorous_bench.cu) | 全芯片多 SM 满载竞争、100% 线程槽饱和（Mode 3 混合腾退+暂停）与 50 轮压力测试 |
| [`stream_preempt_test.cu`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/stream_preempt_test.cu) | 独立三粒度细粒度硬件 Trap 抢占测试（含中文详注版 `stream_preempt_test_commented.cu`） |
| [`trap_latency_test.cu`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/trap_latency_test.cu) | BAR0 `0x00424684` 全芯片广播硬件 Trap 脉冲延迟测试 |
| [`src_hal_level3/`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/src_hal_level3) | `libhalcuda.so` 核心实现源码备份（`trap_sm120.cpp`、`resumable_sm120.cpp`） |
| [`include/`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/include) & [`lib/`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/lib) | 完整头文件与预编译便携动态库/静态库（`libhalcuda.so`、`libpreempt.so`、`libshimcuda.so`、`libcuxtra_linux_x86_64.a`） |
| [`docs/`](file:///run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode/standalone_sm120_test/docs) | 完整技术报告、架构白皮书与 4 篇深度 Q&A 专题文档（`QA01 ~ QA04`） |
