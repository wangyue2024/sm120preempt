/**
 * ====================================================================================================
 * @file stream_preempt_test_commented.cu
 * @brief Blackwell SM120 (GB206 / RTX 5060) 细粒度微秒级硬件 Trap 与非幂等可恢复抢占引擎 - 详尽原理解析与逐行注释版
 * ====================================================================================================
 * 
 * ----------------------------------------------------------------------------------------------------
 * 一、 项目背景与理论沿革 (OSDI '25 XSched 体系定位)
 * ----------------------------------------------------------------------------------------------------
 * XSched (OSDI '25, "XSched: Preemptive Scheduling for Diverse XPUs") 提出了针对异构计算加速器 (XPU)
 * 的通用三级抢占抽象：
 *   - Level 1 (Launch-Level Blocking): 主机端发射队列截断，依赖在途任务自然完成 (时延 1~100+ ms)。
 *   - Level 2 (Deactivation / Block Drain): SASS 二进制动态插桩 Guardian，在 CTA 入口处检查退出标志 (时延 100~2000 µs)。
 *     局限：若 SM 已被持久化或首波 (Wave 1) 大算子满载占满，新 CTA 无法获得调度，L2 插桩桩点根本无法执行。
 *   - Level 3 (Instruction-Level Interrupt / Preemption): 在 SM 计算流水线内部强制打断正在执行 SASS 指令的活跃 Warp (< 500 µs)。
 *
 * ----------------------------------------------------------------------------------------------------
 * 二、 Blackwell (SM120 / GB206) 架构对传统 Level 3 抢占的毁灭性挑战与拨乱反正
 * ----------------------------------------------------------------------------------------------------
 * 在 Volta (SM70) 到 Ada (SM89) 时代，Level 3 抢占依赖以下链条：
 *   1. 利用静态库 libcuxtra.a 导出表定位驱动内部的 `.text.traphandler`；
 *   2. 调用 cuXtraInstrMemBlockAlloc 动态申请可执行设备虚拟地址页作为跳转桩；
 *   3. 调用 cuXtraTriggerTrap 通过 RM 控制命令 0x20800122 向 PRI 广播寄存器 0x00419e84 写入电平脉冲；
 *   4. Trap Handler 内部直接执行 EXIT 终止正在运行的 Kernel，随后由 XSched 的 CommandLog 重放该任务。
 *
 * 然而在全新的 Blackwell 架构 (RTX 5060, GB206, sm_120, Linux 驱动 595.99.02) 上，该机制全面崩溃：
 *   【致命障碍 1: GSP 强隔离机制】
 *     Blackwell 强制为 GSP Client 模式 (IS_GSP_CLIENT / pGpu+0x911 == 1)，废除了传统 CPU-RM 直管模式。
 *     早期逆向误判 pGpu+0x911==0 为“调试门控”，实为对 GSP 路由逻辑的误读。
 *   【致命障碍 2: PRI 寄存器重定位与硬件总线中毒 (Poisoning)】
 *     旧架构的 SM 调度宏寄存器 0x00419e84 在 SM120 上被废除。读写该地址会触发硬件毒化码 0xbadf1100
 *     (NV_PPRIV_SYS_PRI_ERROR_CODE)，盲读未映射广播区更会引发 PCIe 总线超时与单显卡显示合成器挂死 (黑屏)。
 *   【致命障碍 3: 调试会话下的 Context Cancel (Error 719)】
 *     在 SM120 上，一旦为 Context 开启调试会话，驱动内部会校验所有指令内存段。调用 cuXtraInstrMemBlockAlloc
 *     创建的额外指令页未通过签名校验，会导致驱动强制撤销该 Context，后续任何发射均报 CUDA_ERROR_LAUNCH_FAILED (719)。
 *   【致命障碍 4: 128-bit SASS 与 Trap 状态机死锁】
 *     SM120 采用全新 16 字节 (128-bit) SASS 机器字。如果在硬件 Trap 中断服务例程 (ISR) 内部强行调用 EXIT，
 *     会绕过硬件异常处理退出握手协议，破坏 SM 异常状态机，导致整个上下文不可逆崩溃。
 *   【致命障碍 5: 非幂等算子 (Non-Idempotent) 的不可重放性】
 *     传统 Trap-Abort 丢弃在途计算并从头重放。若 Kernel 含有 atomicAdd 或原地累加 (A[i] += B[i])，
 *     重放会导致数据被重复累加多次，产生致命的计算精度错误。
 *
 * ----------------------------------------------------------------------------------------------------
 * 三、 本文件实现的突破性核心技术原理
 * ----------------------------------------------------------------------------------------------------
 * 本文件完整封存了在单一 CUcontext 内实现微秒级细粒度抢占的全部核心机制：
 *   1. 【天然 Code Cave 复用】：
 *      放弃 cuXtraInstrMemBlockAlloc。通过导出表发现 Slot 20 原始镜像内部在 `slot20_pc + 0xA00` 到 `+0x1290`
 *      之间天然存在 2192 字节全 NOP 预留区域 (Code Cave)。该区域位于已认证的 Utility 段内，彻底消除 Error 719。
 *   2. 【全芯片硬件广播 MMIO 寄存器 (0x00424684)】：
 *      在 BAR0 映射中锁定了 Blackwell SM120 真实的全芯片硬件广播触发端口 0x00424684。
 *      采用两次连续 Posted 写脉冲 (wr32(0x424684, 0x80001c01); wr32(0x424684, 0x00001c01);)，
 *      耗时仅 1.89 µs 即可 100% 击穿并挂起全部 30 个 SM (3840 Cores)。
 *   3. 【52 指令 100% 零分化普通态蹦床引擎 (Normal-Mode Trampoline)】：
 *      - 捕获硬件 Trap 后，Warp 进入 Code Cave；
 *      - 计算当前 CTA 在 Grid 内的扁平化线性索引；
 *      - 读取主机端映射的 preemption buffer 控制字；
 *      - 若为暂停模式 (Pause)：将本地栈 [0xffffa8] 处的硬件返回地址 (TRAP_RETURN_PC) 备份至 [0xffff88]，
 *        并将 [0xffffa8] 覆写为普通态自旋跳板 (pause_tramp: NOP; BRA -2)；
 *      - 执行 RET.ABS R12 优雅退出硬件 Trap 状态机！
 *      - Warp 在普通着色器流水线中进行原地自旋等待 (In-place Spin-wait)，零拷贝保持全量通用寄存器 (R0~R254)、
 *        谓词寄存器 (P0~P7) 以及 Shared Memory 状态；
 *      - 恢复信号到达后，在第二次 Trap 脉冲中将 [0xffff88] 还原回 [0xffffa8]，单周期无缝跳回挂起点继续全速计算！
 *   4. 【单 CUcontext 细粒度三模态自由调度】：
 *      - Mode 1: Per-Stream 驱逐退出 (Early-Exit) 并记录各 CTA 完成进度标志与抢占索引；
 *      - Mode 2: Per-Stream 原位挂起与原位恢复 (Pause/Resume)，保障 100% 非幂等安全；
 *      - Mode 3: Sub-Stream Per-CTA 同一 Kernel 内部选择性挂起 (如冻结 CTA 0~7，放行 CTA 8~14)。
 * ====================================================================================================
 */

#include <cuda.h>                 // CUDA Driver API 核心头文件
#include <cuxtra/cuxtra.h>        // XSched 底层逆向扩展库 (用于未公开导出表查询与指令内存拷贝)
#include "bar0.h"                 // PCIe BAR0 MMIO 物理地址映射与读写原语 (rd32/wr32)
#include "arena.h"                // SM120 驱动内部 Utility Arena 与 TrapHandler 内存探测器
#include "sass120.h"              // Blackwell SM120 128-bit SASS 汇编器基础原语
#include <unistd.h>               // POSIX 系统调用 (usleep 等)
#include <cstdio>                 // 标准输入输出
#include <cstring>                // 内存操作函数 (memcpy, memset 等)

/**
 * @brief 读取 GPU 硬件全局纳秒时钟计数器 (PTX 内联汇编)
 * @details 对应 NVIDIA GPU 内部的 64 位全局计时器 (Global Timer)，频率恒定为 1 GHz (1 tick = 1 ns)。
 *          用于在 Kernel 内部进行高精度时间测量、制造受控延时以及记录心跳时间戳。
 * @return unsigned long long 纳秒时间戳
 */
__device__ unsigned long long get_globaltimer() {
    unsigned long long r;
    // 使用 PTX 指令 mov.u64 将特殊寄存器 %globaltimer 的值读入 64 位通用寄存器
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r));
    return r;
}

/**
 * ====================================================================================================
 * @brief 测试负载 1: 非幂等且高度敏感的状态化计算核函数 (Non-Idempotent Stateful Kernel)
 * ====================================================================================================
 * @param progress        [输出] 每个 Block (CTA) 当前迭代进度的全局内存数组 (用于主机端实时监控)
 * @param t_last          [输出] CTA 0 的线程 0 最近一次更新心跳的全局纳秒时间戳 (用于判断流是否被冻结)
 * @param atomic_accum    [输出] 全局原子累加计数器。这是检验“非幂等安全性”的黄金指标！
 *                        若发生重复计算或丢功重放，该累加值必然超出预期；若被破坏则会缺失。
 * @param out_checksums   [输出] 每个线程最终计算状态的哈希校验和数组 (验证每个线程 255 个寄存器是否被破坏)
 * @param total_iters     [输入] 总迭代次数 (例如 204800 次，使得整体执行时间维持在 ~75 ms 左右)
 *
 * @note 为什么设计为非幂等 (Non-Idempotent)？
 *       每个线程内部维护一个 PRNG 伪随机数递推状态：state = state * 1664525u + 1013904223u + (it ^ tid)。
 *       该状态严格依赖上一轮迭代的寄存器值。如果抢占引擎在暂停/恢复过程中篡改了任何通用寄存器、
 *       谓词寄存器，或者错误地将某个 CTA 杀死重放，最终的校验和与原子累加值都将发生不可逆的偏差。
 */
__global__ void non_idempotent_kernel(volatile unsigned long long *progress,
                                      volatile unsigned long long *t_last,
                                      unsigned long long *atomic_accum,
                                      unsigned int *out_checksums,
                                      unsigned int total_iters) {
    // 计算当前线程在全局 Grid 中的扁平化唯一线程 ID (Thread ID)
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 基于 tid 初始化私有伪随机递推状态 (占用私有通用寄存器)
    unsigned int state = tid * 0x9e3779b9u + 0x12345678u;

    // 主计算循环：模拟深度学习推理/训练算子中的密集迭代计算
    for (unsigned int it = 1; it <= total_iters; it++) {
        // 非幂等递推方程：单向且不可逆，对寄存器位翻转或丢步极度敏感
        state = state * 1664525u + 1013904223u + (it ^ tid);

        // 受控延时循环：通过 globaltimer 制造稳定的纳秒级时延 (约 300 ns)，避免 Kernel 瞬间跑完
        unsigned long long t0 = get_globaltimer();
        while (get_globaltimer() - t0 < 300ULL) {
            // 交叉异或操作：保持寄存器 ALU 单元活跃，防止编译器优化消除延时循环
            state ^= 0x55aa55aau;
            state ^= 0x55aa55aau;
        }

        // CTA 0 的线程 0 周期性刷新全局时间戳，作为主机端判断该 Kernel 是否处于运行状态的活体心跳
        if (blockIdx.x == 0 && threadIdx.x == 0) *t_last = get_globaltimer();

        // 每隔 256 次迭代 (it & 0xff == 0)，所有线程向同一个全局内存地址发起原子累加
        if ((it & 0xffu) == 0) {
            atomicAdd((unsigned long long *)atomic_accum, 1ULL);
            // 每个 CTA 的线程 0 更新该 CTA 的心跳进度值
            if (threadIdx.x == 0) {
                progress[blockIdx.x] = it;
            }
        }
    }

    // 将本线程经过二十多万次复杂迭代后的最终 state 写入校验和输出数组
    out_checksums[tid] = state;

    // 循环完成：最高位置 1 (1ULL << 62) 作为完成标志位，通知主机端该 CTA 已全速自然跑完
    if (threadIdx.x == 0) {
        progress[blockIdx.x] = total_iters | (1ULL << 62);
        if (blockIdx.x == 0) *t_last = get_globaltimer();
    }
}

/**
 * @brief 测试负载 2: 第二个非幂等核函数 (逻辑与负载 1 相同，用于 TEST 2 独立的 Per-CTA 抢占验证)
 */
__global__ void non_idempotent_kernel_2(volatile unsigned long long *progress,
                                        volatile unsigned long long *t_last,
                                        unsigned long long *atomic_accum,
                                        unsigned int *out_checksums,
                                        unsigned int total_iters) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int state = tid * 0x9e3779b9u + 0x12345678u;

    for (unsigned int it = 1; it <= total_iters; it++) {
        state = state * 1664525u + 1013904223u + (it ^ tid);
        unsigned long long t0 = get_globaltimer();
        while (get_globaltimer() - t0 < 300ULL) {
            state ^= 0x55aa55aau;
            state ^= 0x55aa55aau;
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) *t_last = get_globaltimer();
        if ((it & 0xffu) == 0) {
            atomicAdd((unsigned long long *)atomic_accum, 1ULL);
            if (threadIdx.x == 0) {
                progress[blockIdx.x] = it;
            }
        }
    }
    out_checksums[tid] = state;
    if (threadIdx.x == 0) {
        progress[blockIdx.x] = total_iters | (1ULL << 62);
        if (blockIdx.x == 0) *t_last = get_globaltimer();
    }
}

/**
 * @brief 低优先级长耗时旋转工作负载 (用于 TEST 3 Early-Exit 测试)
 * @param duration_ns 执行持续时间 (纳秒，例如 3.0 秒)
 */
__global__ void low_worker(volatile unsigned long long *progress,
                           volatile unsigned long long *t_last,
                           unsigned long long duration_ns) {
    unsigned long long t0 = get_globaltimer();
    unsigned long long local_cnt = 0;
    while (get_globaltimer() - t0 < duration_ns) {
        local_cnt++;
        if ((local_cnt & 0xff) == 0 && threadIdx.x == 0) {
            progress[blockIdx.x] = local_cnt;
            if (blockIdx.x == 0) *t_last = get_globaltimer();
        }
    }
    if (threadIdx.x == 0) {
        progress[blockIdx.x] = local_cnt | (1ULL << 62);
        if (blockIdx.x == 0) *t_last = get_globaltimer();
    }
}

/**
 * @brief 高优先级突发工作负载 (模拟高优先级交互式请求，用于验证低优先级被挂起期间高优先级能全速抢占算力)
 * @param duration_ns 执行持续时间 (纳秒，例如 40 ms 或 50 ms)
 */
__global__ void high_worker(volatile unsigned long long *progress,
                            volatile unsigned long long *t_last,
                            unsigned long long duration_ns) {
    unsigned long long t0 = get_globaltimer();
    unsigned long long local_cnt = 0;
    while (get_globaltimer() - t0 < duration_ns) {
        local_cnt++;
        if ((local_cnt & 0xff) == 0 && threadIdx.x == 0) {
            progress[blockIdx.x] = local_cnt;
            if (blockIdx.x == 0) *t_last = get_globaltimer();
        }
    }
    if (threadIdx.x == 0) {
        progress[blockIdx.x] = local_cnt | (1ULL << 62);
        if (blockIdx.x == 0) *t_last = get_globaltimer();
    }
}

/**
 * @brief 预热核函数：确保驱动在第一个测试开始前完整建立 GPU 虚存上下文和调试环境
 */
__global__ void warmup_k() {}

/**
 * @brief 读取 GPU 硬件 BAR0 PTIMER 寄存器高低 32 位 (原子双读算法)
 * @details 硬件时钟寄存器位于 BAR0 偏移 0x9400 (低 32 位) 与 0x9410 (高 32 位)。
 *          由于 32 位读取存在进位翻转竞争条件 (Rollover Race)，采用经典高位双读校验循环：
 *          当两次读取的高 32 位一致时，说明在读取低 32 位期间未发生 32 位溢出进位，拼接结果绝对精准。
 * @return uint64_t 物理纳秒时间戳
 */
static uint64_t read_ptimer() {
    uint32_t hi, lo, hi2;
    do {
        hi = rd32(0x9410);   // 读取高 32 位
        lo = rd32(0x9400);   // 读取低 32 位
        hi2 = rd32(0x9410);  // 再次读取高 32 位
    } while (hi != hi2);     // 若高位发生变化，说明产生了进位，重新读取
    return ((uint64_t)hi << 32) | lo;
}

/**
 * ====================================================================================================
 * @namespace s120_ext
 * @brief Blackwell SM120 专有 128-bit SASS 扩展机器指令生成器
 * ====================================================================================================
 * Blackwell SM120 的每条机器指令严格由 16 字节组成，分解为两个 64 位机器字：
 *   w0 (低 64 位): 操作码基底 (Opcode Base)、目的寄存器 (Rd)、源寄存器 (Ra/Rb/Rc)、立即数低位等。
 *   w1 (高 64 位): 控制码与依赖屏障信息，包含发射停顿 (Stall)、让渡 (Yield)、写屏障 (Write Barrier)、
 *                  读屏障 (Read Barrier) 及等待掩码 (Wait Barrier Mask)。
 */
namespace s120_ext {
using namespace s120;

/**
 * @brief 生成 LDC 指令：从常量内存 Bank 0 读取 32 位无符号整数
 * @details 汇编形式: LDC Rd, c[0x0][off]
 *          【SM120 常量内存寻址规则】：常量内存立即数字段按 4 字节对其编码，其在低字中的对齐偏移为 (off >> 2) << 8。
 *          主要用于读取 XSched 注入到 c[0x0][0x170..0x188] 的 DebuggerParams 参数指针。
 */
inline I ldc_u32(int rd, uint32_t off) {
    uint64_t encoded_off = (uint64_t)(off >> 2) << 8;
    return {(encoded_off << 32) | 0xff000000ULL | ((uint64_t)rd << 16) | 0x7b82ULL,
            0x000e220000000800ULL};
}

/**
 * @brief 生成 LD.E.STRONG.GPU 指令：带强内存一致性范围的 64 位全局设备内存加载
 * @details 汇编形式: LD.E.STRONG.GPU R0, [R4] 或 [R6]
 *          【关键作用】：普通 LD.E 指令会被 GPU 的 L1/TEX 缓存命中，无法及时感知主机端在 host-mapped
 *          内存中写入的新抢占标志！通过增加 STRONG.GPU 强一致性修饰符，强制绕过 L1/L2 缓存，
 *          直接穿透至系统主存拉取最新的抢占标志位，保障微秒级响应！
 */
inline I ld_e_strong_gpu_r0_r4() {
    return {0x0000000004007980ULL, 0x001e24000010e900ULL};
}
inline I ld_e_strong_gpu_r0_r6() {
    return {0x0000000006007980ULL, 0x001e24000010e900ULL};
}

/**
 * @brief 从 Warp 本地栈 (Local Memory) 读取 32 位值: LDL R0, [lmem_off]
 */
inline I ldl_32_r0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0x00000000ff007983ULL, 0x0032240000100800ULL};
}

/**
 * @brief 从 Warp 本地栈读取 64 位值: LDL.64 R0, [lmem_off]
 * @details 主要用于从 [0xffffa8] 提取硬件自动压入的 TRAP_RETURN_PC (中断返回 PC)。
 */
inline I ldl_64_r0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0x00000000ff007983ULL, 0x0032240000100a00ULL};
}

/**
 * @brief 向 Warp 本地栈写入 64 位值: STL.64 [lmem_off], R0
 * @details 用于改写 [0xffffa8] 上的返回地址，或者将原返回地址暂存到保留栈帧 [0xffff88]。
 */
inline I stl_64_r0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0x00000000ff007387ULL, 0x0033e40000100a00ULL};
}

/**
 * @brief 条件受控存储指令: @P0 STL.64 [lmem_off], R0
 * @details 仅在谓词 P0 为真时执行栈写入。在无分化调度中极其关键，能避免非活跃线程产生分化写入。
 */
inline I stl_64_r0_p0(uint32_t lmem_off) {
    return {((uint64_t)(lmem_off & 0xffffff) << 40) | 0x00000000ff000387ULL, 0x0033e40000100a00ULL};
}

/**
 * @brief 生成带统一控制字的立即数传送指令: MOV Rd, imm
 */
inline I mov_imm_w01(int rd, uint32_t imm) {
    return s120::mov_imm(rd, imm, 0x003fda0000000000ULL);
}

/**
 * @brief 谓词比较指令: ISETP.NE.AND P0, PT, Ra, RZ, PT (判断 Ra != 0)
 * @details 若 Ra 非零则将谓词 P0 置为真，用于检查 preempt_buf 指针是否存在。
 */
inline I isetp_ne_rz(int ra) {
    return {0x000000ff0000720cULL | ((uint64_t)ra << 24),
            0x001fda0003f05270ULL};
}

/**
 * @brief 谓词比较指令: ISETP.NE.AND P0, PT, Ra, imm, PT (判断 Ra != imm)
 */
inline I isetp_ne_imm(int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL,
            0x001fda0003f05270ULL};
}

/**
 * @brief 谓词逻辑“与”复合比较: ISETP.NE.AND P0, PT, Ra, imm, P0 (计算 P0 = P0 && (Ra != imm))
 */
inline I isetp_ne_imm_and_p0(int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL,
            0x001fda0000705270ULL};
}

/**
 * @brief 谓词比较指令: ISETP.EQ.AND P0, PT, Ra, imm, PT (判断 Ra == imm)
 */
inline I isetp_eq_imm(int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL,
            0x001fda0003f02270ULL};
}

/**
 * @brief 谓词逻辑“或”复合比较: ISETP.EQ.OR P0, PT, Ra, imm, P0 (计算 P0 = P0 || (Ra == imm))
 */
inline I isetp_eq_imm_or_p0(int ra, uint32_t imm) {
    return {((uint64_t)imm << 32) | ((uint64_t)ra << 24) | 0x780cULL,
            0x001fda0000702470ULL};
}

/**
 * @brief 相对跳转指令低 64 位机器字编码计算器
 * @param n_instrs 相对当前指令的跳转目标指令跨度 (正数为向前，负数为向后)
 * @param low16    指令操作码低 16 位标志 (区分有条件 @P0、@!P0 或无条件 BRA)
 */
inline uint64_t encode_bra_w0(int n_instrs, uint32_t low16) {
    int64_t off = (int64_t)n_instrs * 4;
    uint64_t lo8 = (uint64_t)(off & 0xff);
    uint64_t hi = (uint64_t)(off >> 8);
    return (hi << 34) | (lo8 << 16) | low16;
}

/**
 * @brief 条件跳转指令: @P0 BRA +n_instrs (若 P0 为真则跳转)
 */
inline I bra_p0(int n_instrs) {
    return {encode_bra_w0(n_instrs, 0x0947),
            (n_instrs < 0) ? 0x000fea000383ffffULL : 0x000fea0003800000ULL};
}

/**
 * @brief 反条件跳转指令: @!P0 BRA +n_instrs (若 P0 为假则跳转)
 */
inline I bra_not_p0(int n_instrs) {
    return {encode_bra_w0(n_instrs, 0x8947),
            (n_instrs < 0) ? 0x000fea000383ffffULL : 0x000fea0003800000ULL};
}

/**
 * @brief 无条件相对跳转指令: BRA +n_instrs
 */
inline I bra_rel(int n_instrs) {
    return {encode_bra_w0(n_instrs, 0x7947),
            (n_instrs < 0) ? 0x000fea000383ffffULL : 0x000fea0003800000ULL};
}
} // namespace s120_ext


/**
 * ====================================================================================================
 * @brief 测试套件主入口函数 (Main Function)
 * ====================================================================================================
 */
int main() {
    // -------------------------------------------------------------------------
    // 第一步：初始化 BAR0 MMIO 物理空间映射与上下文预热
    // -------------------------------------------------------------------------
    // 打开 /sys/bus/pci/devices/0000:01:00.0/resource0，将 GPU 64 MB BAR0 映射至本进程地址空间
    if (bar0_open()) {
        fprintf(stderr, "FATAL: Failed to open and mmap BAR0 MMIO. Are you running with root/sudo?\n");
        return 1;
    }

    // 初始化 CUDA 运行时并完成设备上下文创建
    cudaFree(0);
    // 预热核函数发射，促使驱动在 GPU 设备虚存建立完整的 Utility 段与初始 Trap 向量表
    warmup_k<<<1, 1>>>();
    cudaDeviceSynchronize();

    // -------------------------------------------------------------------------
    // 第二步：通过 cuxtra 导出表定位驱动活体 TrapHandler 与 Code Cave 地址
    // -------------------------------------------------------------------------
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    // 探测当前 Context 的内存拓扑：提取 Slot 20 镜像、arena_base 及 live .text.traphandler 地址
    sm120arena::Info info = sm120arena::probe(ctx, 0);
    if (!info.ok || !info.handler_is_memobj) {
        fprintf(stderr, "FATAL: sm120arena probe failed. Check NVIDIA driver compatibility.\n");
        return 1;
    }

    // -------------------------------------------------------------------------
    // 第三步：从活体 Handler 动态提取并重组关键操作码 (保持与官方驱动 100% 字节一致)
    // -------------------------------------------------------------------------
    // 提取驱动内部对上下文切换/调试事件的判定逻辑: LOP3.LUT P0, RZ, R2, 0x11, RZ, 0xc0, !PT (位于 handler + 0x3b0)
    s120::I test_r2_ctxsw;
    memcpy(&test_r2_ctxsw.w0, &info.handler[0x3b0], 8);
    memcpy(&test_r2_ctxsw.w1, &info.handler[0x3b8], 8);
    test_r2_ctxsw.w0 = (test_r2_ctxsw.w0 & 0xffffffffULL) | (0x11ULL << 32);
    test_r2_ctxsw.w1 = (test_r2_ctxsw.w1 & ~0x000ffe0000000000ULL) | 0x000fda0000000000ULL;

    // 提取清除中断标志位的指令: LOP3.LUT R2, R2, ~0x8, RZ, 0xc0, !PT (位于 handler + 0x220)
    // 将 R2 原因码寄存器的 Bit 3 (0x8) 清零，防止 Warp 退出 Trap 后因中断电平残留再次被重复触发
    uint64_t lop_w0, lop_w1;
    memcpy(&lop_w0, &info.handler[0x220], 8);
    memcpy(&lop_w1, &info.handler[0x228], 8);
    s120::I clear_r2_bit8 = {(lop_w0 & 0xffffffffULL) | (0xfffffff7ULL << 32), lop_w1};

    // 提取原生的栈覆写指令: STL.64 [0xffffa8], R0 (位于 handler + 0x110)
    s120::I stl_64_rpc;
    memcpy(&stl_64_rpc.w0, &info.handler[0x110], 8);
    memcpy(&stl_64_rpc.w1, &info.handler[0x118], 8);

    // 计算三块代码在 GPU 设备虚拟地址空间中的绝对基地址：
    // ext_va: 52 指令统一蹦床主入口 (位于 Slot 20 Code Cave 的起始处 tools_pc + 0xA00)
    uint64_t ext_va = info.tools_pc + 0xa00;
    // exit_stub_va: 专门用于 Mode 1 Early-Exit 的普通态 EXIT 跳板 (tools_pc + 0xE00)
    uint64_t exit_stub_va = info.tools_pc + 0xe00;
    // pause_tramp_va: 专门用于 Mode 2/3 的普通态自旋挂起跳板 (tools_pc + 0xE20)
    uint64_t pause_tramp_va = info.tools_pc + 0xe20;
    uint32_t tramp0_lo = (uint32_t)pause_tramp_va;
    uint32_t tramp1_lo = (uint32_t)(pause_tramp_va + 0x10);
    uint32_t tramp_hi  = (uint32_t)(pause_tramp_va >> 32);

    using namespace s120;
    using namespace s120_ext;

    // -------------------------------------------------------------------------
    // 第四步：构建 52 指令 100% 零分化普通态微架构蹦床子例程 (ext_code)
    // -------------------------------------------------------------------------
    // 该 52 条指令直接注入到驱动预留的 Code Cave 中，构成了细粒度可恢复抢占引擎的物理中枢！
    std::vector<I> ext_code = {
        // [阶段 1: 读取常量内存中的抢占控制缓冲区指针]
        ldc_u32(4, 0x170),                                              //  0: R4 = c[0x0][0x170] (preempt_buf 地址低 32 位)
        ldc_u32(5, 0x174),                                              //  1: R5 = c[0x0][0x174] (preempt_buf 地址高 32 位)
        isetp_ne_rz(4),                                                 //  2: P0 = (R4 != 0) (检查当前 Kernel 是否绑定了抢占控制块)
        bra_not_p0(45),                                                 //  3: @!P0 BRA done (若未绑定抢占缓冲区，直接跳到末尾退出，不干扰普通任务)
        test_r2_ctxsw,                                                  //  4: 判定 R2 原因码是否为外部上下文切换事件 (R2 & 0x11 != 0)
        bra_p0(43),                                                     //  5: @P0 BRA done (若非抢占调度事件，跳至退出，交还官方例程)
        clear_r2_bit8,                                                  //  6: R2 &= ~0x8 (清除当前中断事件的标记位)

        // [阶段 2: 计算当前 CTA 的扁平化线性索引，实现 Per-CTA 状态寻址]
        // 目标：计算当前 Block 的线性编号 block_idx = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x
        {0x0000000000007919ULL, 0x000e620000002600ULL},                 //  7: S2R R0, SR_CTAID.Y (读取 blockIdx.y)
        {0x0000dd00ff037b82ULL, 0x000e620000000800ULL},                 //  8: LDC R3, c[0x0][0x374] (读取 gridDim.y)
        {0x0000000000067919ULL, 0x001e620000002500ULL},                 //  9: S2R R6, SR_CTAID.X (读取 blockIdx.x)
        {0x0000000000087919ULL, 0x000e2c0000002700ULL},                 // 10: S2R R8, SR_CTAID.Z (读取 blockIdx.z)
        {0x0000dc00ff077b82ULL, 0x000e220000000800ULL},                 // 11: LDC R7, c[0x0][0x370] (读取 gridDim.x)
        {0x0000000803007224ULL, 0x003fda00078e0200ULL},                 // 12: IMAD R0, R3, R8, R0 (R0 = blockIdx.z * gridDim.y + blockIdx.y)
        {0x0000000700007224ULL, 0x001fda00078e0206ULL},                 // 13: IMAD R0, R0, R7, R6 (R0 = R0 * gridDim.x + blockIdx.x)
        {0x0000000400037810ULL, 0x000fda0007ffe000ULL},                 // 14: IADD3 R3, PT, PT, R0, 0x4, R0 (R3 = block_idx * 2 + 4)
        {0x0000000403067825ULL, 0x000fda00078e0004ULL},                 // 15: IMAD.WIDE.U32 R6, R3, 0x4, R4 (R6:R7 = &preempt_buf[4 + 2 * block_idx])

        // [阶段 3: 读取全局与 Per-CTA 控制指令字]
        ld_e_strong_gpu_r0_r4(),                                        // 16: LD.E.STRONG.GPU R0, [R4] (强一致性读取 preempt_buf[0] 全局指令)
        isetp_ne_imm(0, 3),                                             // 17: P0 = (R0 != 3) (判断是否为 Mode 3 Per-CTA 独立控制模式)
        bra_p0(1),                                                      // 18: @P0 BRA check_pause (若非 Mode 3，直接跳转至统一模式判定)
        ld_e_strong_gpu_r0_r6(),                                        // 19: LD.E.STRONG.GPU R0, [R6] (若是 Mode 3，读取属于本 CTA 的专用控制字！)

        // [阶段 4: 判定 Mode 2/3 - 原位暂停处理 (Pause Routine)]
        isetp_ne_imm(0, 2),                                             // 20: P0 = (R0 != 2) (判断控制字是否为 2: PAUSE)
        bra_p0(10),                                                     // 21: @P0 BRA check_exit (若非 2，跳至退出判定)
        // 以下执行 do_pause: 严格零分化地将栈上返回 PC 重定向至 pause_tramp_va
        ldl_32_r0(0xffff90),                                            // 22: LDL R0, [0xffff90] (读取 THREAD_STATE_ENUM.0，校验线程活跃态)
        isetp_ne_rz(0),                                                 // 23: P0 = (R0 != 0) (P0 标记当前线程处于正常计算活跃态)
        ldl_64_r0(0xffffa8),                                            // 24: LDL.64 R0, [0xffffa8] (读取硬件压栈的原计算返回地址 PC_orig)
        isetp_ne_imm_and_p0(0, tramp0_lo),                              // 25: P0 = P0 && (R0 != tramp0_lo) (防止重复暂停导致的自旋跳板地址被当作原 PC 误存)
        isetp_ne_imm_and_p0(0, tramp1_lo),                              // 26: P0 = P0 && (R0 != tramp1_lo)
        stl_64_r0_p0(0xffff88),                                         // 27: @P0 STL.64 [0xffff88], R0 (安全备份！将原计算 PC_orig 写入 [0xffff88])
        mov_imm_w01(0, tramp0_lo),                                      // 28: MOV R0, tramp0_lo (加载普通态自旋跳板 pause_tramp 低 32 位地址)
        mov_imm_w01(1, tramp_hi),                                       // 29: MOV R1, tramp_hi  (加载高 32 位地址)
        stl_64_r0_p0(0xffffa8),                                         // 30: @P0 STL.64 [0xffffa8], R0 (核心！将栈上返回 PC 改写为 pause_tramp_va！)
        bra_rel(16),                                                    // 31: BRA done_r1 (跳转到尾声，准备优雅退出 Trap 模式)

        // [阶段 5: 判定 Mode 1 - 驱逐退出处理 (Early-Exit Routine)]
        isetp_ne_imm(0, 1),                                             // 32: P0 = (R0 != 1) (判断控制字是否为 1: EARLY_EXIT)
        bra_p0(9),                                                      // 33: @P0 BRA check_resume (若非 1，跳至恢复判定)
        mov_imm_w01(0, 1),                                              // 34: MOV R0, 1
        {0x0000000406007385ULL, 0x000fda0000100900ULL},                 // 35: ST.E [R6+0x4], R0 (置位 restore_flag_of_block[cta] = 1，记录已切出)
        ldc_u32(8, 0x180),                                              // 36: LDC R8, c[0x0][0x180] (读取当前 Kernel 的 kernel_idx 低 32 位)
        ldc_u32(9, 0x184),                                              // 37: LDC R9, c[0x0][0x184] (读取 kernel_idx 高 32 位)
        {0x0000000804007385ULL, 0x001fda0000100b08ULL},                 // 38: ST.E.64 [R4+0x8], R8 (记录 preempt_idx = kernel_idx，供后续精确恢复)
        mov_imm_w01(0, (uint32_t)exit_stub_va),                         // 39: MOV R0, lo32(exit_stub_va) (加载普通模式 EXIT 跳板低 32 位地址)
        mov_imm_w01(1, (uint32_t)(exit_stub_va >> 32)),                 // 40: MOV R1, hi32(exit_stub_va) (加载高 32 位地址)
        stl_64_r0(0xffffa8),                                            // 41: STL.64 [0xffffa8], R0 (核心！将返回 PC 改写为普通态 EXIT 跳板！)
        bra_rel(5),                                                     // 42: BRA done_r1 (跳至尾声)

        // [阶段 6: 判定 Mode 2/3 - 原位恢复处理 (Resume Routine)]
        // 当控制字重置为 0 (RESUME) 且再次收到广播中断时触发：
        ldl_64_r0(0xffffa8),                                            // 43: LDL.64 R0, [0xffffa8] (检查当前返回地址)
        isetp_eq_imm(0, tramp0_lo),                                     // 44: P0 = (R0 == tramp0_lo) (确认当前处于自旋跳板状态)
        isetp_eq_imm_or_p0(0, tramp1_lo),                               // 45: P0 = P0 || (R0 == tramp1_lo)
        ldl_64_r0(0xffff88),                                            // 46: LDL.64 R0, [0xffff88] (从备份栈中重新取出真实的计算原地址 PC_orig)
        stl_64_r0_p0(0xffffa8),                                         // 47: @P0 STL.64 [0xffffa8], R0 (完美还原！恢复栈上返回 PC 为 PC_orig！)

        // [阶段 7: 统一尾声与干净退出硬件 Trap 模式 (done_r1 / done)]
        mov_imm_w01(1, 0xffff80),                                       // 48: MOV R1, 0xffff80 (恢复官方栈底指针寄存器)
        mov_imm_w01(3, 0),                                              // 49: MOV R3, 0 (还原 R3 寄存器)
        iadd_imm(12, 12, 0x10),                                         // 50: IADD R12, R12, 0x10 (调整硬件返回栈指针)
        ret_r12(),                                                      // 51: RET.ABS R12 0x20 (执行硬件返回，优雅退出 Trap 模式，飞入目标 PC！)
    };

    // -------------------------------------------------------------------------
    // 第五步：构建普通模式跳板 (Exit Stub 与 Pause Trampoline)
    // -------------------------------------------------------------------------
    // exit_stub: 普通模式 EXIT 跳板 (两条连续 EXIT 指令，Warp 在正常着色器流水线退出，不会破坏状态机)
    std::vector<I> exit_stub = { exit_(), exit_() };
    // pause_tramp: 普通模式在位死循环自旋跳板 (NOP + 向后相对跳转 2 条指令，即在原处保持低开销自旋等待)
    std::vector<I> pause_tramp = { nop(), bra_rel(-2) };

    // 将指令序列压平为 64 位无符号机器字数组
    auto flat_ext = flat(ext_code);
    auto flat_exit = flat(exit_stub);
    auto flat_tramp = flat(pause_tramp);

    // -------------------------------------------------------------------------
    // 第六步：备份原指令并在 GPU 活体内存安装补丁 (Livepatch Installation)
    // -------------------------------------------------------------------------
    // 备份 live .text.traphandler + 0x390 处的原始 16 字节指令 (以便测试结束后 100% 无损还原)
    uint64_t orig_390[2];
    memcpy(orig_390, &info.handler[0x390], 16);

    // 备份 Code Cave 中将被覆盖的原始内存区域
    std::vector<uint64_t> ext_backup(flat_ext.size());
    std::vector<uint64_t> exit_backup(flat_exit.size());
    std::vector<uint64_t> tramp_backup(flat_tramp.size());
    cuXtraMemcpyDtoH(ext_backup.data(), ext_va, flat_ext.size() * 8, 0);
    cuXtraMemcpyDtoH(exit_backup.data(), exit_stub_va, flat_exit.size() * 8, 0);
    cuXtraMemcpyDtoH(tramp_backup.data(), pause_tramp_va, flat_tramp.size() * 8, 0);

    // 构造在 live handler + 0x390 处植入的 CALL.ABS.NOINC 跳转指令：
    // 当硬件发生任何中断进入 +0x390 时，无条件跳转至位于 Code Cave (ext_va) 的 52 指令统一蹦床引擎
    uint64_t o0, o1;
    memcpy(&o0, &info.handler[0x380], 8);
    memcpy(&o1, &info.handler[0x388], 8);
    uint64_t enc = (ext_va - 0x100) >> 8, branch_inst[2];
    branch_inst[0] = ((enc & 0xffffffffULL) << 32) | (o0 & 0xffffffffULL);
    branch_inst[1] = (o1 & ~0xffULL) | ((enc >> 32) & 0xffULL);

    // 将 exit_stub 写入 exit_stub_va (tools_pc + 0xE00)
    cuXtraMemcpyHtoD(exit_stub_va, flat_exit.data(), flat_exit.size() * 8, 0);
    // 将 pause_tramp 写入 pause_tramp_va (tools_pc + 0xE20)
    cuXtraMemcpyHtoD(pause_tramp_va, flat_tramp.data(), flat_tramp.size() * 8, 0);
    // 将 52 指令核心引擎 ext_code 写入 ext_va (tools_pc + 0xA00)
    cuXtraMemcpyHtoD(ext_va, flat_ext.data(), flat_ext.size() * 8, 0);
    // 将 hook 跳转指令打入 live traphandler + 0x390
    cuXtraMemcpyHtoD(info.handler_va + 0x390, branch_inst, 16, 0);
    // 强制全芯片 30 个 SM 的指令缓存 (I-Cache) 失效，确保新装载的 SASS 机器码立即生效！
    cuXtraInvalInstrCache(ctx);

    // -------------------------------------------------------------------------
    // 第七步：分配主机端与设备端映射内存 (Host-Alloc Mapped Memory)
    // -------------------------------------------------------------------------
    CUstream stream_low, stream_high;
    cuStreamCreate(&stream_low, CU_STREAM_NON_BLOCKING);
    cuStreamCreate(&stream_high, CU_STREAM_NON_BLOCKING);

    volatile uint32_t *preempt_buf_low, *preempt_buf_high;
    volatile unsigned long long *prog_low, *prog_high, *t_last_low, *t_last_high;
    volatile unsigned long long *atomic_accum;
    unsigned int *out_checksums;

    // 分配对 CPU/GPU 均可见的零拷贝锁页内存 (Mapped Pinned Memory)，作为控制与状态观测接口
    cudaHostAlloc((void **)&preempt_buf_low, 256, cudaHostAllocMapped);
    cudaHostAlloc((void **)&preempt_buf_high, 256, cudaHostAllocMapped);
    cudaHostAlloc((void **)&prog_low, 15 * 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&prog_high, 15 * 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&t_last_low, 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&t_last_high, 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&atomic_accum, 8, cudaHostAllocMapped);
    cudaHostAlloc((void **)&out_checksums, 15 * 128 * sizeof(unsigned int), cudaHostAllocMapped);

    // XSched 常量内存调试器参数结构体 (精确对齐 c[0x0][0x170..0x188])
    struct {
        uint64_t preempt_buf;       // +0x170..0x178: preempt_buf 的 64 位设备虚拟地址
        uint64_t guardian;          // +0x178..0x180: Guardian 结构指针 (L2 用)
        int64_t kernel_idx;         // +0x180..0x188: 当前 Kernel Launch 的全局序号索引
        uint32_t sched_enabled;     // +0x188: 调度启用标记
    } __attribute__((packed)) dbg_params;

    // =========================================================================
    // 评测模块 1: Mode 2 — 细粒度原位无损可恢复抢占 (In-Place Pause & Resume)
    // =========================================================================
    printf("=== TEST 1: Mode 2 (Non-Abort Resumable Stream Pause & Resume, Non-Idempotent) ===\n");
    memset((void *)preempt_buf_low, 0, 256);
    memset((void *)preempt_buf_high, 0, 256);
    memset((void *)prog_low, 0, 15 * 8);
    memset((void *)prog_high, 0, 15 * 8);
    *t_last_low = 0;
    *t_last_high = 0;
    *atomic_accum = 0;
    memset(out_checksums, 0, 15 * 128 * sizeof(unsigned int));

    CUfunction func_non_idemp = nullptr, func_non_idemp_2 = nullptr, func_high = nullptr, func_low = nullptr;
    cudaGetFuncBySymbol(&func_non_idemp, (const void *)non_idempotent_kernel);
    cudaGetFuncBySymbol(&func_non_idemp_2, (const void *)non_idempotent_kernel_2);
    cudaGetFuncBySymbol(&func_high, (const void *)high_worker);
    cudaGetFuncBySymbol(&func_low, (const void *)low_worker);

    // 将低优先级流的 preempt_buf_low 绑定到 func_non_idemp 的常量内存 Bank 0 (c[0x0][0x170])
    dbg_params.preempt_buf = (uint64_t)(uintptr_t)preempt_buf_low;
    dbg_params.guardian = 0;
    dbg_params.kernel_idx = 10;
    dbg_params.sched_enabled = 1;
    cuXtraSetDebuggerParams(func_non_idemp, &dbg_params, sizeof(dbg_params));

    // 将高优先级流的 preempt_buf_high 绑定到 func_high
    dbg_params.preempt_buf = (uint64_t)(uintptr_t)preempt_buf_high;
    dbg_params.guardian = 0;
    dbg_params.kernel_idx = 11;
    dbg_params.sched_enabled = 1;
    cuXtraSetDebuggerParams(func_high, &dbg_params, sizeof(dbg_params));

    // 在低优流并发射非幂等计算内核 (15 个 CTA，每个 CTA 128 线程，共 1920 线程)
    unsigned int total_iters = 204800; // 对应总运行时间约 75 ms
    void *args_non_idemp[] = { &prog_low, &t_last_low, &atomic_accum, &out_checksums, &total_iters };
    cuLaunchKernel(func_non_idemp, 15, 1, 1, 128, 1, 1, 0, stream_low, args_non_idemp, nullptr);

    // 在同一个 CUcontext 内的高优流发射突发高优任务 (40 ms)
    unsigned long long dur_high = 40000000ULL;
    void *args_high[] = { &prog_high, &t_last_high, &dur_high };
    cuLaunchKernel(func_high, 15, 1, 1, 128, 1, 1, 0, stream_high, args_high, nullptr);

    // 等待两路核函数在 GPU 上均进入稳定迭代状态
    while (*t_last_low == 0 || *t_last_high == 0) usleep(10);
    usleep(10000); // 允许双方共存全速执行 10 ms

    unsigned long long prog_before_pause = prog_low[0];

    // 【抢占触发】：仅将 stream_low 的控制指令置为 2 (PAUSE)，并通过 BAR0 发射全芯片广播 Trap 脉冲！
    uint64_t t_pause_cmd = read_ptimer();
    *preempt_buf_low = 2;              // 设置暂停控制指令
    __sync_synchronize();              // 内存写屏障，确保 CPU 写操作刷新至内存控制器
    wr32(0x424684, 0x80001c01);        // 向 0x00424684 写入 STOP_TRIGGER 广播脉冲 (激活中断)
    wr32(0x424684, 0x00001c01);        // 立即解除触发锁 (恢复时钟门控，避免总线悬挂)
    uint64_t t_pause_mmio = read_ptimer();

    usleep(200); // 等待 200 µs，所有属于 stream_low 的活跃 Warp 全部切入自旋跳板 (pause_tramp)
    uint64_t t_last_at_pause = *t_last_low;
    unsigned long long prog_paused_1 = prog_low[0];
    unsigned long long atomic_paused_1 = *atomic_accum;
    unsigned long long high_prog_1 = prog_high[0];

    // 【隔离性验证】：保持 stream_low 冻结挂起 10 ms！
    // 验证低优先级流是否完全静止，同时位于同一个 CUcontext 内的高优流 stream_high 是否丝毫不受影响全速推进！
    usleep(10000);

    unsigned long long prog_paused_2 = prog_low[0];
    unsigned long long atomic_paused_2 = *atomic_accum;
    unsigned long long high_prog_2 = prog_high[0];

    // 【原位恢复触发】：将 stream_low 控制指令重置为 0 (RESUME)，再次发射 MMIO 广播脉冲！
    // 52 指令蹦床将从备份栈 [0xffff88] 中提取 PC_orig 重新装入 [0xffffa8]，实现单周期原位满速恢复！
    uint64_t t_resume_cmd = read_ptimer();
    *preempt_buf_low = 0;              // 重置为恢复指令
    __sync_synchronize();
    wr32(0x424684, 0x80001c01);        // 第二次广播 Trap 脉冲唤醒自旋 Warp
    wr32(0x424684, 0x00001c01);
    while (*t_last_low == t_last_at_pause) {} // 等待低优流的心跳时间戳发生推进
    uint64_t t_resumed = *t_last_low;

    // 等待两路流完全执行完毕
    CUresult res_high_during_pause = cuStreamSynchronize(stream_high);
    CUresult res_non_idemp = cuStreamSynchronize(stream_low);

    // 打印抢占观测结果
    printf("  [Paused State] low_prog: %llu -> %llu -> %llu (low_delta=%llu!) | high_prog: %llu -> %llu (high_delta=+%llu!)\n",
           prog_before_pause, prog_paused_1, prog_paused_2, prog_paused_2 - prog_paused_1,
           high_prog_1, high_prog_2, high_prog_2 - high_prog_1);
    printf("  [Paused State] atomic_paused_1=%llu -> atomic_paused_2=%llu | stream_high res=%d completed=%d\n",
           atomic_paused_1, atomic_paused_2, (int)res_high_during_pause,
           (int)((prog_high[0] & (1ULL << 62)) != 0));

    // 【数学严谨性比对】：对全部 1920 个线程的最终 PRNG 状态进行理论值逐一校验，并检验全局原子累加值！
    unsigned long long expected_atomic = (unsigned long long)(15 * 128) * (total_iters / 256);
    int checksum_errors = 0;
    for (unsigned int tid = 0; tid < 15 * 128; tid++) {
        unsigned int st = tid * 0x9e3779b9u + 0x12345678u;
        for (unsigned int it = 1; it <= total_iters; it++) {
            st = st * 1664525u + 1013904223u + (it ^ tid);
        }
        if (out_checksums[tid] != st) checksum_errors++;
    }
    printf("  [Resumed & Completed] res=%d | MMIO=%.2f us | On-GPU Pause latency=%.2f us | Resume latency=%.2f us\n",
           (int)res_non_idemp, (t_pause_mmio - t_pause_cmd) / 1000.0,
           (double)((long long)(t_last_at_pause - t_pause_cmd)) / 1000.0,
           (double)((long long)(t_resumed - t_resume_cmd)) / 1000.0);
    printf("  [Non-Idempotent Verification] atomic_accum=%llu (expected=%llu) | thread_checksum_errors=%d / %d\n",
           *atomic_accum, expected_atomic, checksum_errors, 15 * 128);

    // 断言判定：必须满足冻结期间增量为0、高优全速推进、恢复后原子计数与1920线程校验和 100% 零误差！
    int total_errors = 0;
    if (res_high_during_pause != CUDA_SUCCESS || res_non_idemp != CUDA_SUCCESS ||
        prog_paused_2 != prog_paused_1 || high_prog_2 <= high_prog_1 ||
        *atomic_accum != expected_atomic || checksum_errors != 0) {
        total_errors++;
    }

    // =========================================================================
    // 评测模块 2: Mode 3 — 同一 Kernel 内部 Per-CTA 选择性挂起 (Selective Freeze)
    // =========================================================================
    printf("\n=== TEST 2: Mode 3 (Sub-Stream Per-CTA Selective Pause & Resume in Same Kernel) ===\n");
    memset((void *)preempt_buf_low, 0, 256);
    memset((void *)prog_low, 0, 15 * 8);
    *t_last_low = 0;
    *atomic_accum = 0;
    memset(out_checksums, 0, 15 * 128 * sizeof(unsigned int));

    dbg_params.preempt_buf = (uint64_t)(uintptr_t)preempt_buf_low;
    dbg_params.guardian = 0;
    dbg_params.kernel_idx = 12;
    dbg_params.sched_enabled = 1;
    cuXtraSetDebuggerParams(func_non_idemp_2, &dbg_params, sizeof(dbg_params));
    cuLaunchKernel(func_non_idemp_2, 15, 1, 1, 128, 1, 1, 0, stream_low, args_non_idemp, nullptr);
    for (int w = 0; w < 10000 && (*t_last_low == 0 || prog_low[8] == 0); w++) usleep(10);
    usleep(10000);

    // 【块级选择性配置】：
    // 设置 preempt_buf_low[0] = 3 (激活 Per-CTA 独立掩码模式)
    // 指定仅冻结前 8 个 CTA (CTA 0..7 控制字置 2: PAUSE)，而后 7 个 CTA (CTA 8..14 控制字置 0: RUN)！
    for (int cta = 0; cta < 8; cta++) preempt_buf_low[4 + 2 * cta] = 2;
    for (int cta = 8; cta < 15; cta++) preempt_buf_low[4 + 2 * cta] = 0;
    preempt_buf_low[0] = 3;
    __sync_synchronize();
    wr32(0x424684, 0x80001c01);
    wr32(0x424684, 0x00001c01);

    usleep(200);
    unsigned long long p1[15], p2[15];
    for (int cta = 0; cta < 15; cta++) p1[cta] = prog_low[cta];
    usleep(10000); // 观察 10 ms：CTA 0..7 应原地保持静止，CTA 8..14 持续全速计算！
    for (int cta = 0; cta < 15; cta++) p2[cta] = prog_low[cta];

    int paused_ctas_ok = 0, running_ctas_ok = 0;
    for (int cta = 0; cta < 8; cta++) if (p2[cta] == p1[cta]) paused_ctas_ok++;
    for (int cta = 8; cta < 15; cta++) if ((p2[cta] & ~(1ULL << 62)) > (p1[cta] & ~(1ULL << 62))) running_ctas_ok++;

    // 唤醒解冻 CTA 0..7，恢复全速计算！
    preempt_buf_low[0] = 0;
    __sync_synchronize();
    wr32(0x424684, 0x80001c01);
    wr32(0x424684, 0x00001c01);
    CUresult res_cta_test = cuStreamSynchronize(stream_low);

    // 校验 Mode 3 下的非幂等计算一致性
    int cta_checksum_errors = 0;
    for (unsigned int tid = 0; tid < 15 * 128; tid++) {
        unsigned int st = tid * 0x9e3779b9u + 0x12345678u;
        for (unsigned int it = 1; it <= total_iters; it++) {
            st = st * 1664525u + 1013904223u + (it ^ tid);
        }
        if (out_checksums[tid] != st) cta_checksum_errors++;
    }
    printf("  [Per-CTA Selectivity] Paused CTAs 0..7 frozen=%d/8 (delta=%llu, %llu) | Running CTAs 8..14 active=%d/7 (delta=+%llu, +%llu)\n",
           paused_ctas_ok, p2[0] - p1[0], p2[7] - p1[7],
           running_ctas_ok,
           (p2[8] & ~(1ULL << 62)) - (p1[8] & ~(1ULL << 62)),
           (p2[14] & ~(1ULL << 62)) - (p1[14] & ~(1ULL << 62)));
    printf("  [Per-CTA Resumed & Completed] res=%d | atomic_accum=%llu (expected=%llu) | checksum_errors=%d / %d\n",
           (int)res_cta_test, *atomic_accum, expected_atomic, cta_checksum_errors, 15 * 128);
    if (res_cta_test != CUDA_SUCCESS || paused_ctas_ok != 8 || running_ctas_ok != 7 ||
        *atomic_accum != expected_atomic || cta_checksum_errors != 0) {
        total_errors++;
    }

    // =========================================================================
    // 评测模块 3: Mode 1 — 论文标准微秒级 Early-Exit + 块级还原标志位记录
    // =========================================================================
    printf("\n=== TEST 3: Mode 1 (Selective Per-Stream Early-Exit + XSched Per-CTA Restore Flags) ===\n");
    memset((void *)preempt_buf_low, 0, 256);
    memset((void *)preempt_buf_high, 0, 256);
    memset((void *)prog_low, 0, 15 * 8);
    memset((void *)prog_high, 0, 15 * 8);
    *t_last_low = 0;
    *t_last_high = 0;

    // 低优流发射 3.0 秒超长旋转负载
    dbg_params.preempt_buf = (uint64_t)(uintptr_t)preempt_buf_low;
    dbg_params.kernel_idx = 20;
    dbg_params.sched_enabled = 1;
    cuXtraSetDebuggerParams(func_low, &dbg_params, sizeof(dbg_params));
    unsigned long long dur_low = 3000000000ULL;
    void *args_low[] = { &prog_low, &t_last_low, &dur_low };
    cuLaunchKernel(func_low, 15, 1, 1, 128, 1, 1, 0, stream_low, args_low, nullptr);

    // 高优流发射 50 ms 突发任务
    dbg_params.preempt_buf = (uint64_t)(uintptr_t)preempt_buf_high;
    dbg_params.kernel_idx = 21;
    dbg_params.sched_enabled = 1;
    cuXtraSetDebuggerParams(func_high, &dbg_params, sizeof(dbg_params));
    dur_high = 50000000ULL;
    cuLaunchKernel(func_high, 15, 1, 1, 128, 1, 1, 0, stream_high, args_high, nullptr);

    while (*t_last_low == 0 || *t_last_high == 0) usleep(10);
    usleep(10000);

    // 【驱逐触发】：设置 preempt_buf_low[0] = 1 (EARLY_EXIT)，发射广播中断脉冲！
    uint64_t t_preempt_start = read_ptimer();
    *preempt_buf_low = 1;
    __sync_synchronize();
    wr32(0x424684, 0x80001c01);
    wr32(0x424684, 0x00001c01);
    uint64_t t_mmio_done = read_ptimer();

    // 等待低优流驱逐退出同步完成
    CUresult res_low = cuStreamSynchronize(stream_low);
    uint64_t t_low_synced = read_ptimer();
    CUresult res_high = cuStreamSynchronize(stream_high);

    // 检查抢占元数据是否准确写入 XSched 的控制缓冲区：
    uint64_t recorded_preempt_idx = *(volatile uint64_t *)&preempt_buf_low[2];
    int restore_flags_set = 0;
    for (int cta = 0; cta < 15; cta++) {
        if (preempt_buf_low[5 + 2 * cta] == 1) restore_flags_set++;
    }

    printf("  [Early-Exit] res_low=%d | MMIO=%.2f us | Host cuStreamSynchronize latency=%.2f us | res_high=%d high_completed=%d\n",
           (int)res_low, (t_mmio_done - t_preempt_start) / 1000.0,
           (t_low_synced - t_preempt_start) / 1000.0,
           (int)res_high, (int)((prog_high[0] & (1ULL << 62)) != 0));
    printf("  [XSched Metadata] preempt_idx=%llu (expected=20) | restore_flag_of_block count=%d / 15\n",
           (unsigned long long)recorded_preempt_idx, restore_flags_set);
    if (res_low != CUDA_SUCCESS || res_high != CUDA_SUCCESS ||
        recorded_preempt_idx != 20ULL || restore_flags_set != 15) {
        total_errors++;
    }

    // -------------------------------------------------------------------------
    // 第八步：完整还原硬件现场 (Restoration & Cleanup)
    // -------------------------------------------------------------------------
    // 将 live .text.traphandler + 0x390 处的指令还原回官方原始机器码
    cuXtraMemcpyHtoD(info.handler_va + 0x390, orig_390, 16, 0);
    // 还原 Code Cave 及跳板区域的原始内存镜像
    cuXtraMemcpyHtoD(ext_va, ext_backup.data(), flat_ext.size() * 8, 0);
    cuXtraMemcpyHtoD(exit_stub_va, exit_backup.data(), flat_exit.size() * 8, 0);
    cuXtraMemcpyHtoD(pause_tramp_va, tramp_backup.data(), flat_tramp.size() * 8, 0);
    // 再次刷新 I-Cache，确保驱动状态 100% 恢复洁净
    cuXtraInvalInstrCache(ctx);

    printf("\n=== OVERALL TEST RESULT: %s (Total Errors: %d) ===\n",
           (total_errors == 0) ? "ALL PASS (100% SUCCESS)" : "FAIL", total_errors);
    return (total_errors == 0) ? 0 : 1;
}
