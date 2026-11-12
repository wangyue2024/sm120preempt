# Blackwell SM120 (RTX 5060 / 5090) 细粒度硬件抢占最小独立测试套件

本目录是针对 NVIDIA Blackwell SM120 架构（GB20x 家族，包括 RTX 5060、RTX 5070、RTX 5080、RTX 5090）的**自包含、零外部依赖最小测试包**。所有必需的头文件、静态库（`libcuxtra.a`）、动态库（`libhalcuda.so`、`libpreempt.so`、`libshimcuda.so`）与编译规则均已完整收拢在本目录内。

---

## 一、 目录依赖与拓扑结构

```
standalone_sm120_test/
├── Makefile                          # 便携 Makefile (自动寻找 CUDA 12.9 / GCC 13，RPATH 指向 $ORIGIN/lib)
├── run_all_tests.sh                  # 一键全自动测试脚本 (含 PCIe BAR0 自动探测、环境变量注入)
├── README.md                         # 本说明文档
│
├── bar0.h                            # PCIe BAR0 MMIO 映射原语 (支持自动探测 NVIDIA GPU BDF 与 BAR0_PATH 覆盖)
├── sass120.h                         # SM120 128-bit SASS 机器指令编码器
├── livepatch.h                       # Utility Arena 热补丁注入辅助宏
├── arena.h                           # 驱动 Utility Arena 与 Slot 20 Code Cave 探测头文件
│
├── include/                          # 本地独立头文件目录
│   ├── arena.h
│   ├── cuxtra/
│   │   └── cuxtra.h                  # cuXtra 底层驱动逆向接口头文件
│   └── xsched/cuda/hal/
│       ├── level3/trap_sm120.h       # XSched Level 3 C++ HAL 接口头文件
│       └── common/cuda.h             # HAL 基础 CUDA 类型定义
│
├── lib/                              # 本地独立二进制库依赖
│   ├── libcuxtra_linux_x86_64.a      # cuXtra 驱动逆向静态库 (用于探测导出表与安全内存拷贝)
│   ├── libhalcuda.so                 # XSched Level 3 C++ HAL 核心动态库
│   ├── libpreempt.so                 # 抢占调度器基础运行时动态库
│   ├── libshimcuda.so                # CUDA Driver API Shim 库
│   ├── libcuda.so -> libshimcuda.so  # 软链接
│   └── libcuda.so.1 -> libshimcuda.so# 软链接
│
└── 核心测试源码/
    ├── trap_latency_test.cu          # [测试 1] 全芯片 MMIO 0x00424684 广播脉冲打断时延与覆盖率评测
    ├── stream_preempt_test.cu        # [测试 2] 单 Context 细粒度三模态基准测试 (原版)
    ├── stream_preempt_test_commented.cu # [测试 3] 单 Context 细粒度三模态逐行原理解析版 (894 行超详尽注释)
    ├── hal_fine_grained_test.cu      # [测试 4] XSched C++ HAL API 集成三模态测试
    ├── preempt_rigorous_bench.cu     # [测试 5] 全芯片 30-SM 饱和竞争与四大严苛基准评测
    ├── trapprobe.cu                  # 交互式 Trap 探测工具
    ├── smstop.cu                     # 单 SM 停止触发器
    ├── spin.cu                       # 长时间自旋占载工具
    └── bar0tool.c                    # BAR0 物理寄存器快速读写命令行工具
```

---

## 二、 依赖环境要求 (在 RTX 5090 / 同架构机器上运行的前提)

若要将本文件夹拷贝至另一台机器（例如搭载 **GeForce RTX 5090** 的工作站/服务器）并成功运行，目标机器必须满足以下前置依赖：

| 依赖项 | 最低版本要求 | 为什么需要该版本？ |
| :--- | :--- | :--- |
| **GPU 硬件** | Blackwell 架构 (`sm_120`) | 包含 RTX 5060 (GB206), RTX 5070 (GB205), RTX 5080 (GB203), **RTX 5090 (GB202)** 等。 |
| **NVIDIA 驱动** | **R570+** (推荐 `575.x`, `580.x` 或更高) | 旧版驱动（如 550.x、535.x）不包含 Blackwell GSP 固件，也不开放 Blackwell ABI。驱动必须启用 Open Kernel Module。 |
| **CUDA Toolkit** | **CUDA 12.8** 或更高 (12.8 / 12.9 / 13.0+) | 只有 CUDA 12.8+ 的 `nvcc` 支持编译 `-arch=sm_120` 架构代码。低版本 nvcc 会报 `Unsupported gpu architecture`。 |
| **主机编译器** | **GCC 11 ~ GCC 13 (`g++-13`)** | CUDA 12.8/12.9 的 host_config.h 支持 GCC 11~14，系统若默认使用 GCC 15 会被拦截，推荐 `g++-13`。 |
| **系统权限** | **`root` (`sudo`)** | 访问 `/sys/bus/pci/devices/<BDF>/resource0` 映射物理 PCIe BAR0 属于内核受控特权操作，必须使用 `sudo` 执行。 |
| **宿主环境要求** | **物理裸机 或 特权容器 (`--privileged`)** | 若在 Docker / K8s / 云端容器运行，必须赋予 `--privileged` 并且 `/sys` 允许读写（禁止以只读 `ro` 挂载）。 |
| **内核安全策略** | **Secure Boot 宽松或禁用** | 若内核开启 Strict Lockdown (`kernel.lockdown=confidentiality`)，内核可能会禁止用户态通过 sysfs 映射 BAR0。 |

---

## 三、 我能在 RTX 5090 机器上直接运行吗？有哪些异同？

### 1. 为什么能够直接运行？（底层一致性）
1. **统一的 SASS sm_120 指令集**：RTX 5090 (GB202) 与 RTX 5060 (GB206) 的着色器内核完全一致，均为 Blackwell 128-bit 机器字，寄存器映射、栈帧返回地址 `[0xffffa8]`、`RET.ABS` 机制完全相同。
2. **统一的 GSP Client 驱动运行时**：Slot 20 Utility Arena 是 NVIDIA R595 驱动在主机/GSP 内存中构建的调试陷阱代理，其内部在 `tools_pc + 0xa00` 至 `+0x1290` 天然存在相同的 2192 字节全 NOP Code Cave。
3. **全芯片广播寄存器通用性**：`0x00424684` 属于 Blackwell GPCS（GPC Broadcast Aperture）广播端口。无论芯片拥有 3 个 GPC（5060）还是 11~12 个 GPC（5090），往该广播寄存器写入脉冲均会全芯片分发。

### 2. RTX 5090 与 RTX 5060 的物理拓扑差异及应对机制
1. **PCIe Bus ID (BDF 路径) 可能不同**：
   - 5060 机器上显卡位于 `0000:01:00.0`。
   - 5090 若插在其它插槽、多卡服务器或外接坞，可能位于 `0000:02:00.0`、`0000:81:00.0` 等。
   - **本测试包的自动解法**：`bar0.h` 与 `run_all_tests.sh` 已内置 **PCIe BDF 自动探测引擎**，在启动时自动扫描 `/sys/bus/pci/devices/` 并锁定 Vendor ID `0x10de` 显卡。您也可以随时通过环境变量强制指定：
     ```bash
     export BAR0_PATH="/sys/bus/pci/devices/0000:02:00.0/resource0"
     ```
2. **SM 数量差异（30 SMs vs ~170 SMs）**：
   - RTX 5060 启用了 30 个 SM；RTX 5090 启用了约 170~192 个 SM。
   - `stream_preempt_test` 发射 15 个 Block；`trap_latency_test` 发射 30 个 Block；它们在 5090 上均能全量运行并被 100% 成功捕获。
   - `preempt_rigorous_bench` 的 BENCH 2 在 5060 上发射 46,080 线程恰好 100% 占满 30 个 SM（$30 \times 1536$）；在 5090 上，该测试依然能通过并跑出微秒级挂起恢复指标，但若要让 5090 全卡 170 个 SM 全量饱和，可将 Block 数量适度调大至约 340 个 Block。

---

## 四、 编译与运行方法

### 1. 一键全量编译与验证（推荐）
```bash
chmod +x run_all_tests.sh
sudo ./run_all_tests.sh
```
该脚本将依次：
1. 自动探测显卡 PCIe BDF 并导出 BAR0 映射路径；
2. 调用 `make all` 编译所有测试程序；
3. 执行 `trap_latency_test 6` 测量全芯片硬件广播时延；
4. 执行 `stream_preempt_test` 验证三模态抢占与非幂等原子累加安全；
5. 执行 `hal_fine_grained_test` 验证 XSched C++ HAL 库集成；
6. 执行 `preempt_rigorous_bench` 运行 4 项全芯片竞争与压力基准；
7. 执行 `stream_preempt_test_commented` 运行详尽注释版基准测试。

### 2. 单独编译指定测试
```bash
# 编译全部
make all

# 单独编译 3 模态测试
make stream_preempt_test

# 单独编译详尽注释版
make stream_preempt_test_commented

# 单独执行（需要 root 权限访问 MMIO）
sudo ./stream_preempt_test
```
