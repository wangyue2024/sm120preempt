# Blackwell SM120 (`GB206` / RTX 5060) 细粒度微秒级硬件 Trap 与非幂等原位可恢复抢占代码归档库 (`mmio/`)

本目录（`/mnt/data/1file/Desktop/code/code/xsched/level3opencode/mmio/`）完整封存了针对 NVIDIA Blackwell 架构（GeForce RTX 5060, `GB206`, Compute Capability `12.0` / `sm_120`, Driver `595.99.02`）研发的 **微秒级全芯片广播硬件 Trap 触发器 (`0x00424684`)** 与 **单 `CUcontext` 内三粒度（Per-Stream Early-Exit / Per-Stream 非幂等原位暂停恢复 / Sub-Stream Per-CTA 同一 Kernel 内分块暂停恢复）统一硬件 Trap 引擎** 的全部核心头文件、微架构探针、基准测试与一键验证脚本。

Mode 4 的 CTA 安全点换出、资源准入对照、HAL 接口和实机进度记录见 [CTA 状态换出实验](spill/README.md)。

---

## 1. 目录文件清单与功能说明

| 文件名 | 类型 | 作用与核心内容 |
| :--- | :--- | :--- |
| `bar0.h` | 核心头文件 | 通过 `/sys/bus/pci/devices/0000:01:00.0/resource0` 映射 RTX 5060 的 64 MB BAR0 MMIO 空间，提供零开销的 `rd32(off)` 与 `wr32(off, val)` 内联原语。 |
| `sass120.h` | 核心头文件 | 轻量级 Blackwell `SM120` 128-bit (16-Byte) SASS 机器码手写汇编器（支持 `MOV`, `S2R`, `CS2R`, `IMAD.WIDE.U32`, `ST.E[.64]`, `IADD`, `RET.ABS R12 0x20`, `EXIT`, `NOP` 及调用 `nvdisasm -b SM120` 在线反汇编校验）。 |
| `livepatch.h` | 核心头文件 | 封装对当前 `CUcontext` 活跃 `.text.traphandler`（位于 `arena_base + 0x15f0f00`）的指纹校验、将自定义指令安装至 `tools_pc + 0x0a00` Code Cave、在 `+0x0390` 打入 `CALL.ABS.NOINC` 补丁以及测试后无损还原的完整生命周期管理。 |
| `bar0tool.c` | 命令行工具 | 通用 BAR0 寄存器读写工具：`sudo ./bar0tool r <off> [count]` 或 `sudo ./bar0tool w <off> <val>`。 |
| `spin.cu` | 辅助负载 | 持续运行指定秒数的 GPU 旋转负载 Kernel，通过主机映射内存实时输出心跳计数。 |
| `smstop.cu` | 微架构探针 | 单 SM 硬件 `STOP_TRIGGER` (`Bit 31`) / `RUN_TRIGGER` (`Bit 30`) 状态机与 `DBGR_STATUS0` (`0x504700`)、`WARP_ESR`、`GLOBAL_ESR` 探针。 |
| `trapprobe.cu` | 微架构探针 | 初期 `.text.traphandler + 0x0390` 拦截探针，对比单 SM Unicast (`0x504704`) 与旧架构 `0x419e84` 的硬件响应与 `R2` 原因码。 |
| `trap_latency_test.cu` | 全芯片广播探针 | 对比 7 种触发模式（`trig_mode 0..6`：单 SM、单 TPC、15 个活跃 TPC 逐一写入、6 个 GPC 广播寄存器写入、以及 **单次写入全芯片广播寄存器 `0x00424684`**），实测证明 `trig_mode = 6` (`0x00424684`) 仅需 **`1.89 µs`** 即可 100% 触发全部 **30/30 个 SM**，且清除 `R2` 的 Bit 3 (`0x8`) 后 Kernel 零错误自然完成。 |
| `stream_preempt_test.cu` | **核心三模态基准测试** | 在单个 `CUcontext` 内完整实现并验证 58 条指令 ITS-Safe Normal-Mode `NANOSLEEP 0x100` Trampoline 统一 Trap 引擎的全部三种细粒度抢占模态：<br>1. **TEST 1 (Mode 2)**：Per-Stream 非 Abort、非幂等原位暂停与原位恢复（On-GPU 暂停延迟 `1.02 µs`，同 Context 高优先级 Stream 全速运行，原位恢复延迟 `5.02 µs`，1,920 线程非幂等递推与全局 `atomicAdd` 零误差）；<br>2. **TEST 2 (Mode 3)**：Sub-Stream Per-CTA 同一 Kernel 内选择性分块暂停与原位恢复（冻结 CTAs `0..7` 为 `delta = 0`，同时同一 Kernel 的 CTAs `8..14` 继续全速执行，恢复后校验零误差）；<br>3. **TEST 3 (Mode 1)**：Per-Stream 微秒级选择性 Early-Exit + XSched Per-CTA `restore_flag_of_block[cta]` 与 `preempt_idx` 记录（MMIO `1.95 µs`，Host `cuStreamSynchronize` `11.97 µs`，`15/15` CTA 标志准确记录）。 |
| `preempt_rigorous_bench.cu` | **全芯片 30 SM 满载竞争与开销严苛评测套件** | 包含四大定量评测模块：<br>1. **BENCH 1**：全芯片 30 SM 满载算力竞争与重分配（30,720 线程非幂等低优流 vs 15,360 线程 8.05 GFLOP 高优突发流，实现 **`29.32x` 加速**，恢复空载 GPU **`96.3%` 算力**，On-GPU 算力让出仅 **`0.90 µs`**）；<br>2. **BENCH 2**：100% SM 硬件线程槽位饱和（`46,080 / 46,080` 线程）下的 Mode 3 Hybrid Evict+Pause（**`73.59x` 加速**，前 30 CTA 提前退出腾槽位 + 后 30 CTA 原地暂停并无损恢复）与 Mode 1 240-CTA 多波次驱逐；<br>3. **BENCH 3**：2D Tiled SGEMM（`1024×1024`）三项开销拆解（`O1` 静态注入开销 `~0.0%`，`O2` 广播脉冲穿透税 `16.40 µs/trap`，`O3` 30 SM 共驻暂停 CTA 蹦床税仅 `+0.86%`）；<br>4. **BENCH 4**：全芯片 30 SM 满载下连续 50 轮 Pause/Resume 循环的延迟分位数统计（零回退常数实测：MMIO `p50=2.46 µs`，On-GPU Yield `min=0.90 µs, p50=1.41 µs`，Resume `min=11.33 µs, p50=25.89 µs`，`atomic=61440/61440, checksum_errs=0, timing_errs=0`）。 |
| `hal_fine_grained_test.cu` | **XSched HAL 集成测试** | 直接链接 `libhalcuda.so`，调用 `xsched::cuda::TrapSm120Context` 新增的细粒度 C++ 接口（`InstallFineGrainedTrapHandler`, `BindKernelPreemptBuffer`, `PauseStreamInPlace`, `ResumeStreamInPlace`, `ConfigureAndPauseCtasInPlace`, `EarlyExitStream`, `RestoreOriginalTrapHandler`）完成端到端验证。 |
| `Makefile` | 构建脚本 | 一键编译本目录下全部 8 个工具与基准测试程序（`make all`）。 |
| `run_all_tests.sh` | 一键验证脚本 | 自动编译并依次运行全芯片广播探针、独立三模态基准测试、XSched HAL 集成测试以及全芯片 30 SM 严苛竞争评测套件（`sudo ./run_all_tests.sh`）。 |

---

## 2. 一键构建与运行方式

```bash
cd /mnt/data/1file/Desktop/code/code/xsched/level3opencode/mmio
chmod +x run_all_tests.sh
sudo ./run_all_tests.sh
```

> **硬件操作安全提示**：
> 1. 严禁对 `0x00420000 .. 0x0042ffff` 广播区间执行盲目 `rd32` 扫描（其中 `0x004246a8` 为只写图形复位寄存器，读取会触发 `Xid 13`）。
> 2. 严禁在 `0x00424684` 的 `STOP_TRIGGER` (`0x80001c01`) 置位期间读取 `0x00424684` 或出厂屏蔽的 TPC（`tpc0, tpc6, tpc10, tpc12`），始终使用连续两次 Posted 写入：`wr32(0x424684, 0x80001c01); wr32(0x424684, 0x00001c01);`。
