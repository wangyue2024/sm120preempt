Teamwork设定：对于主要的探索和思考，使用Argon ， 其余低难度subtask使用 gemini 3.8 Flash。阅读当前目录下xsched项目的论文文件（尤其是关注level3抢占模型，尤其是原论文实现方法），开展针对Blackwell架构（本机RTX 5060）的level3抢占（特指trap方法而非tsg）适配与探索研究，解决trap无法触发的问题，复现论文级别level3的抢占，或者探索类似的、（数百）微秒级别的trap抢占，或者探索并开发出更加优质的抢占方式（甚至最好是不abort进程，可恢复的抢占）。

Working directory: /run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/level3opencode
Integrity mode: development

## Reference Material
- 当前目录下的 `xsched` 项目论文文件（重点关注 Level 3 抢占模型及原论文实现方法）
- 当前仓库其他 git 分支中已研发测试完成的 Level 2 实现
- 象征性参考之前的研究成果：`/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/Level3forLinux/xsched/test/sm120_l3_trap_v2/compendium`（需要批判地判断和验证，里面的结论和方法并不一定合适）

## Requirements

### R1. 原论文研读与既有研究批判性验证
阅读当前目录下 `xsched` 项目的论文文件，重点关注 Level 3 抢占模型尤其是原论文实现方法；参考其他 git 分支中已研发测试完成的 Level 2 实现；象征性参考之前的研究成果 `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/Level3forLinux/xsched/test/sm120_l3_trap_v2/compendium`，对其中的结论和方法进行批判性的判断和验证，分析并定位之前研究中 trap 无法触发的问题。允许并且相当鼓励参考其他论文和访问网上其他研究和资料来获得启发和借鉴。

### R2. Blackwell架构（RTX 5060）Level 3抢占探索与开发
针对 Blackwell 架构（本机 RTX 5060）开展 Level 3 抢占（特指 trap 方法而非 tsg）的适配与探索：复现论文级别 Level 3 的抢占，或者探索类似的、（数百）微秒级别的 trap 抢占，或者探索并开发出更加优质的抢占方式（甚至最好是不 abort 进程，可恢复的抢占）。

### R3. 系统环境权限与安全约束
这个 Linux 系统密码是 `123`，可以执行任何操作，本机的 RTX 5060 可以随意使用和调试，可以重新编译开源驱动和装载、修改 GPU 参数等等，在不明显破坏 GPU 的情况下可以深入展开研究。目前本机装载了双系统，最好不要破坏系统文件，且不要明显破坏 GPU。

## Verification Resources
- 当前目录下的 `xsched` 项目论文文件与代码库（含其他 git 分支中的 Level 2 实现与测试）
- `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/Level3forLinux/xsched/test/sm120_l3_trap_v2/compendium`
- 本机 RTX 5060 (Blackwell) GPU 实机环境

## Acceptance Criteria

### 论文与既有研究验证
- [ ] 完成对当前目录下 `xsched` 论文中 Level 3 抢占模型及原论文实现方法的分析，并对 `/run/media/wy/40D41982D4197AFC/1file/Desktop/code/code/xsched/Level3forLinux/xsched/test/sm120_l3_trap_v2/compendium` 中的结论与方法以及 Blackwell 架构下 trap 无法触发的原因给出基于实测验证的分析结论。

### Blackwell (RTX 5060) 抢占实机验证
- [ ] 在本机 RTX 5060 (Blackwell) 上通过可运行的实机测试程序与实测数据验证：成功复现论文级别的 Level 3 trap 抢占（非 tsg），或实现（数百）微秒级别的 trap 抢占，或实现更加优质的抢占方式（如不 abort 进程、可恢复的抢占）；实测输出需包含抢占成功触发的客观证据与实测抢占延迟数据。

### 系统与硬件状态
- [ ] 研究与测试结束后，双系统引导及系统文件保持完好，本机 RTX 5060 GPU 与驱动处于可正常工作状态。
