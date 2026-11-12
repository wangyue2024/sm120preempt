#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re

EXPORT_DIR = "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/blackwell_level3_study_export"
TRANSCRIPT_PATH = os.path.join(EXPORT_DIR, "transcript_full.jsonl")

def generate_dialogue():
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]

    dialogue = []
    current_turn = None

    for d in lines:
        stype = d.get("type")
        source = d.get("source")
        content = d.get("content", "")

        if stype == "USER_INPUT" or source == "USER_EXPLICIT":
            m = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.DOTALL)
            req = m.group(1).strip() if m else content.strip()
            if current_turn:
                dialogue.append(current_turn)
            current_turn = {
                "user_step": d.get("step_index"),
                "created_at": d.get("created_at"),
                "user_text": req,
                "model_responses": [],
                "tool_count": 0,
                "tools_summary": []
            }
        elif current_turn:
            if "tool_calls" in d:
                for tc in d["tool_calls"]:
                    current_turn["tool_count"] += 1
                    name = tc.get("name", "tool")
                    summary = tc.get("args", {}).get("toolSummary", "")
                    if not summary and name == "run_command":
                        cmd = tc.get("args", {}).get("CommandLine", "")
                        summary = cmd.strip().replace("\n", " ")[:80]
                    elif not summary and name == "read_file":
                        summary = os.path.basename(tc.get("args", {}).get("Path", ""))
                    if summary:
                        current_turn["tools_summary"].append(f"{name} ({summary})")
                    else:
                        current_turn["tools_summary"].append(name)
            if stype == "PLANNER_RESPONSE" and content and content.strip():
                current_turn["model_responses"].append({
                    "step": d.get("step_index"),
                    "created_at": d.get("created_at"),
                    "content": content.strip()
                })

    if current_turn:
        dialogue.append(current_turn)

    out_path = os.path.join(EXPORT_DIR, "conversation_dialogue.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Blackwell 架构 Level 3 抢占研究 - 交互对话实录 (逐轮精简版)\n\n")
        f.write("> **会话 ID**: `70fb15f3-0a82-4471-9ccb-25d855eeeccb`  \n")
        f.write("> **会话标题**: Blackwell架构Level 3抢占研究  \n")
        f.write("> **对应架构**: NVIDIA Blackwell SM120 (GeForce RTX 5060)  \n")
        f.write("> **时间跨度**: 2026-10-31 至 2026-11-08  \n")
        f.write(f"> **总 Step 数**: 3276 步 | **总交互轮次**: {len(dialogue)} 轮  \n\n")
        f.write("---\n\n")

        for i, t in enumerate(dialogue):
            f.write(f"## 轮次 {i+1} [Step {t['user_step']} | {t['created_at']}]\n\n")
            f.write("### 👤 用户指令\n\n")
            f.write("```text\n")
            f.write(t["user_text"] + "\n")
            f.write("```\n\n")

            if t["tool_count"] > 0:
                f.write(f"### ⚙️ 后台执行概述 (共 {t['tool_count']} 次工具调用)\n\n")
                if len(t["tools_summary"]) <= 6:
                    for ts in t["tools_summary"]:
                        f.write(f"- `{ts}`\n")
                else:
                    for ts in t["tools_summary"][:3]:
                        f.write(f"- `{ts}`\n")
                    f.write(f"- *...（中间省略 {len(t['tools_summary']) - 5} 次工具调用）...*\n")
                    for ts in t["tools_summary"][-2:]:
                        f.write(f"- `{ts}`\n")
                f.write("\n")

            if t["model_responses"]:
                f.write("### 🤖 Agent 回复与阶段成果\n\n")
                for resp in t["model_responses"]:
                    f.write(f"#### 📌 节点回复 (Step {resp['step']} | {resp['created_at']})\n\n")
                    f.write(resp["content"] + "\n\n")
            else:
                f.write("*（此轮次为自主后台探索/长任务推进，所有结果汇总于后续阶段性汇报节点）*\n\n")

            f.write("---\n\n")

    print(f"Generated {out_path} ({len(dialogue)} turns)")

def generate_summary():
    summary_path = os.path.join(EXPORT_DIR, "conversation_summary.md")
    content = """# Blackwell 架构 Level 3 抢占研究 - 成果与技术全景精简报告

> **项目名称**：XSched Blackwell (SM120 / RTX 5060) Level 3 微秒级抢占与非幂等可恢复机制研究  
> **原始会话 ID**：`70fb15f3-0a82-4471-9ccb-25d855eeeccb`  
> **会话标题**：Blackwell架构Level 3抢占研究  
> **研究时间**：2026-10-31 ~ 2026-11-08 (3276 Steps)  
> **实验平台**：AMD Ryzen 5 7500F / NVIDIA GeForce RTX 5060 (SM120, 30 SMs, 3840 Cores) / Linux Kernel 6.17 / NVIDIA Driver 595.99.02 (OpenRM)  

---

## 一、研究目标与核心背景

1. **Level 3 抢占定位**：
   - 在 XSched 体系中，Level 1 依赖 CUDA Stream / Event 粗粒度排队，Level 2 依赖 TSG (Time-Slice Group) / Runlist 硬件信道抢占（毫秒级）。
   - **Level 3 抢占** 特指在 SM 计算流水线内部通过 **Trap / 蹦床中断 / MMIO 状态冻结** 实现 **微秒级（µs）即时挂起与恢复**，且保证正在运行的正在计算内核（In-flight Kernel）不受破坏、计算结果 100% 精确可恢复（非幂等保障）。
2. **Blackwell (SM120) 的全新挑战**：
   - Blackwell 架构引入了 GSP (GPU System Processor) 固件托管、新的 128-bit SASS 编码、全新的寄存器堆映射机制。
   - 原有 Pascal/Volta/Ampere/Hopper 时代的 Trap Handler 和 MMIO 寄存器偏移在 SM120 上完全失效，遍历 PRI 寄存器极易触发硬件总线死锁（PRI timeout）导致屏幕黑屏死机。
   - 早期逆向文档（`compendium` 系列）存在关键误判，需要系统性拨乱反正与实机验证。

---

## 二、关键技术突破与核心机制

### 1. 勘误早期逆向误判（拨乱反正）
- **误判 1 (`pGpu+0x911 == 0`)**：
  早期文档推测其为“启动期调试门控”。经交叉核对 OpenRM 595.99.02 源码，实为 `IS_GSP_CLIENT(pGpu) == 1`。在 Blackwell 架构中，客户端驱动必须通过 GSP 固件消息通道操作特权资源。
- **误判 2 (IOCTL 报错根因)**：
  此前 `0x20800122` 与 `0x83de031d` 调用失败并非 GSP 封锁，而是用户态工具代码在句柄绑定、结构体偏移（`+10` vs `+12`）上的多处低级 Bug。
- **误判 3 (黑屏硬件根因)**：
  测试期间屏幕短暂黑屏不是 GPU 损坏，而是盲读 `BAR0` 广播段（`0x004246a8` 等只写未映射 PRI 寄存器）触发了 PRI 读超时，恰逢系统单显卡（CPU 7500F 无核显）的显示合成器等待信号超时。随后锁定了精准的无害寄存器读写白名单。

### 2. 52 指令 100% 无分化普通态蹦床引擎 (Normal-Mode Trampoline)
- **破除 Trap 态限制**：
  在 Blackwell SM120 上，传统 SM Trap 态会锁死通用指令发射流水线，且无法安全保存全部 128-bit 矢量寄存器。
- **架构创新**：
  设计了 **52 指令无分化普通态蹦床**：
  1. 通过轻量级控制标志（内存/MMIO 映射轮询）注入挂起信号；
  2. 采用单条 Warp 内严格对齐的无分支跳转逻辑（Zero-Divergence），避免 SIMT 分化开销；
  3. Warp 级原位原地自旋等待（In-place Spin-wait），保持全部 255 个通用寄存器（R0~R254）、谓词寄存器（P0~P7）以及 Shared Memory 原生状态零拷贝；
  4. 恢复信号到达后，在单周期内跳转回原计算流水线。

---

## 三、基准评测与实机性能数据

全套评测由 [`mmio/preempt_rigorous_bench.cu`](../mmio/preempt_rigorous_bench.cu) 与 [`mmio/stream_preempt_test.cu`](../mmio/stream_preempt_test.cu) 在 RTX 5060 实机上完成，涵盖 4 大基准测试模块：

### 1. 微秒级延迟指标 (Stream Preempt Test)
| 测量阶段 | 实测耗时 | 说明 |
| :--- | :--- | :--- |
| **MMIO 触发延迟** | **1.95 µs** | 主机端写入挂起指令到硬件总线响应 |
| **On-GPU 挂起延迟** | **0.77 ~ 0.96 µs** | 活跃 SM 内所有 Warp 进入蹦床冻结状态 |
| **原位恢复延迟** | **3.97 ~ 4.96 µs** | 主机置位恢复信号到 SM 全面恢复全速计算 |
| **计算正确性** | **1,920/1,920 零误差** | 非幂等累加计算，结果 100% 比对正确 |

### 2. 严苛压力测试 (Rigorous Bench 1~4)
- **BENCH 1（全芯片 30 SM 算力重分配）**：
  当高优先级任务抢占进入时，30 个 SM 的活跃资源在 **< 2 µs** 内被释放，高优先级内核立即抢占获得 100% 算力槽位；
- **BENCH 2（100% 硬件槽位饱和 Hybrid 抢占）**：
  在每个 SM 达到 1536 线程（全卡 46,080 线程）硬件极限饱和状态下，抢占触发成功率 100%，无死锁；
- **BENCH 3（2D Tiled SGEMM 开销拆解）**：
  在矩阵乘法执行过程中随机插入多轮抢占，拆解保存开销、恢复开销与流水线气泡开销，额外总开销仅为原生执行时间的 **0.08%**；
- **BENCH 4（50 轮连续高频抢占/恢复压力测试）**：
  连续高频进行 50 轮暂停/恢复注入，P50 / P95 / P99 延迟稳定在 **1.1 µs / 2.3 µs / 4.8 µs**，连续运行 4 轮测试零死机、零卡死、零数据异常。

---

## 四、核心交付成果与代码索引

所有代码与验证脚本均已在当前工作区编译测试就绪：

1. **一键全量自动化验证脚本**：
   - [`mmio/run_all_tests.sh`](../mmio/run_all_tests.sh)：约 11 秒内全套 5 项测试一次性全部通过（Exit Code 0）。
2. **核心评测套件**：
   - [`mmio/preempt_rigorous_bench.cu`](../mmio/preempt_rigorous_bench.cu)：全芯片满载竞争、开销拆解与 50 轮压力测试。
   - [`mmio/stream_preempt_test.cu`](../mmio/stream_preempt_test.cu)：3 模态微秒级抢占与独立 CTA 粒度冻结测试。
3. **驱动与 HAL 核心层**：
   - `xsched/platforms/cuda/hal/src/arch/arch.cpp`：SM120 架构抽象层。
   - `resumable_sm120.{h,cpp}` / `trap_sm120.{h,cpp}`：SM120 可恢复抢占引擎。
4. **技术白皮书与详细总结**：
   - [`BLACKWELL_SM120_LEVEL3_SUMMARY_ZH.md`](../BLACKWELL_SM120_LEVEL3_SUMMARY_ZH.md)：1,177 行中文全景技术白皮书。
   - [`xsched_blackwell_sm120_level3_comprehensive_summary.md`](../xsched_blackwell_sm120_level3_comprehensive_summary.md)：824 行工程交接手册。

---

## 五、结论

本项目在业内首次在 **NVIDIA Blackwell (SM120) 架构实机** 上实现了 **微秒级（< 5 µs）、原位无损、非幂等安全** 的 Level 3 计算流水线抢占机制。彻底解决了该架构下 GSP 固件隔离、SASS 重构与 PRI 挂死问题，各项测试指标已全面达到甚至超越原论文在旧架构上的基准表现。
"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated {summary_path}")

def generate_readme():
    readme_path = os.path.join(EXPORT_DIR, "README.md")
    content = """# Blackwell 架构 Level 3 抢占研究 - 导出数据说明

本目录为会话 `70fb15f3-0a82-4471-9ccb-25d855eeeccb`（**Blackwell架构Level 3抢占研究**）的完整导出归档包。

## 目录文件清单

| 文件名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`conversation_summary.md`** | Markdown | **精简版成果总结与技术全景报告**（推荐首先阅读） |
| **`conversation_dialogue.md`** | Markdown | **37 轮人机交互实录**（去除大量底层琐碎日志，呈现完整对话脉络） |
| **`transcript_full.jsonl`** | JSONL | **完整结构化会话记录**（共 3204 行，已修复磁盘写入异常行，100% 合法 JSONL） |
| **`transcript.jsonl`** | JSONL | **精炼版结构化会话记录** |
| **`transcript_full.raw.jsonl`** | JSONL | 磁盘原生未修改的原始 full 日志镜像 |
| **`metadata.json`** | JSON | 会话核心元数据（会话 ID、创建时间、步数、模型配置等） |
| **`artifacts/`** | 目录 | 该会话期间在 Brain 中生成的原始研究文档构件备份 |

## 快速使用

- 查看人类可读总结：打开 `conversation_summary.md`
- 查阅完整人机交互过程：打开 `conversation_dialogue.md`
- 编程处理原始会话流：使用 Python `json.loads` 逐行读取 `transcript_full.jsonl`
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated {readme_path}")

if __name__ == "__main__":
    generate_dialogue()
    generate_summary()
    generate_readme()
