# Blackwell 架构 Level 3 抢占研究 - 导出数据说明

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
