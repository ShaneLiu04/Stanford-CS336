# Stanford CS336 · Language Modeling from Scratch

面向自学者的非官方学习仓库，围绕 Stanford CS336 系统记录「从零构建语言模型」的
中文讲义笔记、五份作业实现、可复现实验与复盘。所有个人实现均注明对应版本，并附带
可追溯的原始指标、自动生成图表与 SHA-256 manifest；官方作业快照的来源、分支与 commit
固定记录在 [`UPSTREAM.md`](UPSTREAM.md)，Spring 2025 与 Spring 2026 的差异见
[`resources/2025-vs-2026.md`](resources/2025-vs-2026.md)。

> 本项目与 Stanford University、课程教师及助教团队无隶属关系。官方材料版权与许可归原作者所有。

---

## 项目概览

这门课的核心主张是：**语言模型不是黑盒，而是可以逐行实现的系统**。本仓库沿这条主线，
把每一讲的「概念—公式—shape/复杂度—代码—实验」串联起来，最终产出五份可直接复现的作业：

- **A1 Basics**：字节级 BPE、Transformer、RoPE、AdamW、训练与生成闭环；
- **A2 Systems**：Profiling、Triton FlashAttention、DDP/FSDP 与显存分析；
- **A3 Scaling**：IsoFLOP scaling laws、实验设计与外推；
- **A4 Data**：Common Crawl 提取、过滤、去重与数据配比；
- **A5 Alignment**：SFT、DPO、GRPO 及其变体的推理训练。

每一份作业都附带**中文 LaTeX 报告**、**自动生成的图片**、**原始指标**、**summary** 与
**SHA-256 manifest**，保证数值可追溯、图表可重建。

---

## 学习路线

| 阶段 | 主题 | 主要产出 |
| --- | --- | --- |
| 1 | Basics | BPE、Transformer、AdamW、训练与生成 |
| 2 | Systems | Profiling、Triton FlashAttention、分布式训练 |
| 3 | Scaling | Scaling laws、实验设计与外推 |
| 4 | Data | Common Crawl、过滤、去重与数据配比 |
| 5 | Alignment | SFT、DPO、GRPO 与推理训练 |

---

## 作业成果概览

### A1 · Basics — 从字节级 BPE 到 Transformer 训练闭环

75 个可复现训练 run、27 组图表，覆盖 RTX 4080 SUPER / RTX 6000D 两代硬件。
在 TinyStories 上，三个 batch-32 种子达到最优验证 loss **1.371 ± 0.002**，batch 256 进一步到 **1.325**。

![OpenWebText 训练曲线](assignments/assignment1-basics/report/figures/owt_training.png)

架构消融显示 NoPE 退化到 **1.439**、post-norm 到 **1.385**，而 SiLU FFN 与 SwiGLU 在该规模接近；
移除 RMSNorm 即使跑满 327.68M token 仍不稳定（最优 loss **7.512**）。tied embeddings 把 32K 词表模型
从 45.22M 参数降到 28.84M，同时把 OWT 最优 loss 从 4.116 压到 **4.097**。

![生成质量面板](assignments/assignment1-basics/report/figures/generation_quality.png)

### A2 · Systems — Triton FlashAttention 与分布式训练

228 条性能记录、256 条数值误差记录、12 条 checkpointing 记录。自写 Triton kernel 在
`N=8192, d=64` 的 BF16 forward 上跑出 **0.307 ms**（对比 SDPA 3.122 ms，10.17×），
峰值显存 12.16 MiB 对比 853.13 MiB（70.2×）。

![Fused Triton 端到端对比](assignments/assignment2-systems/report/results/figures/fused_triton_end_to_end.png)

在 `N=32768, d=64` 的 fused forward+backward 上，Triton 达到 10.06 ms（8.80×），峰值显存
48.5 MiB 对比 SDPA 16,444 MiB（**339× 缩减**）。activation checkpointing 把 large 模型的峰值显存
从 56.48 GiB 降到 11.53 GiB。

![Transformer 扩展性](assignments/assignment2-systems/report/results/figures/transformer_scaling.png)

### A3 · Scaling — IsoFLOP 拟合与联合 scaling law

官方 IsoFLOP 分析（72 条记录、9 个计算档位）+ 22-run RTX 6000D TinyStories proxy，
完成幂律拟合、bootstrap 不确定性与外推敏感性分析。

![联合 scaling law 曲面](assignments/assignment3-scaling/report/results/figures/joint_law_surface.png)

![Proxy 最优参数分配](assignments/assignment3-scaling/report/results/figures/proxy_optimal_scaling.png)

### A4 · Data — Common Crawl 过滤与去重

21/21 公开测试通过。离线报告基于 400 篇受控文档、1000 条官方 sample WET records 与
12 组过滤消融，覆盖 HTML 提取、语言识别、PII 掩码、NSFW/toxicity、Gopher 质量与 MinHash 去重。

![真实 WET 过滤漏斗](assignments/assignment4-data/report/results/figures/wet_filter_funnel.png)

![质量阈值 Pareto](assignments/assignment4-data/report/results/figures/quality_threshold_pareto.png)

### A5 · Alignment — GRPO 系列推理训练

主作业 + supplement 共 26/26 tests 通过。实现 GRPO / Dr.GRPO / MaxRL / RFT、off-policy
GRPO / GSPO、SFT packing 与 DPO，并用 RTX 6000D proxy 覆盖 7 种 objectives × 4 seeds × 160 steps
（4480 条指标）。

![奖励曲线](assignments/assignment5-alignment/report/results/figures/reward_curves.png)

![Pass@k](assignments/assignment5-alignment/report/results/figures/pass_at_k.png)

---

## 实验硬件环境与限制

### 实际使用环境

| 角色 | 配置 |
| --- | --- |
| 本地开发 | Windows · Python 3.12/3.13（CPU） |
| 主力 GPU | AutoDL 云实例 · NVIDIA RTX 4080 SUPER（约 32 GB） |
| 大显存 GPU | AutoDL 云实例 · NVIDIA RTX 6000D（约 85 GB） |
| 软件栈 | PyTorch 2.8.0+cu128 · CUDA 12.8 · Triton 3.4.0 |
| 课程官方环境 | 2×B200 / 8×B200 / SUNET_ID / Modal —— **未使用** |

### 限制与取舍

1. **无 B200 与多卡 NCCL。** A2 的多卡 DDP/FSDP overlap 与 2×B200 leaderboard 无法实测，
   报告明确标注「未实测」，不会用 CPU/Gloo 或理论值冒充多卡 GPU 结果。
2. **无课程凭据。** 没有 Stanford A3 API key、SUNET_ID 或 Modal 权限，因此 A3 不调用官方训练 API，
   A4 不下载 2500-WET、不跑 8×B200 训练，A5 不使用官方 OLMo-2/B200。
3. **单卡是唯一 GPU。** 跨硬件绝对吞吐不可直接比较；所有算法结论均来自同机 controlled comparisons。
4. **Windows 缺少 `resource` 模块。** A1 有两个 Linux-only 内存测试被跳过（其余 23 core + 23 tokenizer 测试通过）。
5. **资源受限替代实验。** A3 用 22-run RTX 6000D proxy 替代官方计算矩阵；A4 用 400 受控文档 +
   1000 WET 样本；A5 用 vectorized objective proxy（4480 指标）。这些 proxy 均明确标注，不冒充 leaderboard 成绩。

一句话：**受限环境下尽量回答「方法学是否正确」，而非「绝对分数有多高」**，并在每一处明确标注不可比因素。

---

## 仓库结构

```text
assignments/
  assignment1-basics/   # A1 实现 + 报告 + 75 runs + 27 图
  assignment2-systems/  # A2 实现 + 报告 + 228 性能记录
  assignment3-scaling/  # A3 实现 + 报告 + 官方 IsoFLOP + 22-run proxy
  assignment4-data/     # A4 实现 + 报告 + 21/21 tests + 过滤/去重
  assignment5-alignment/# A5 实现 + 报告 + 26/26 tests + 4480 proxy 指标
notes/                  # Spring 2026 全部 17 讲原创中文笔记
experiments/            # 官方题目索引、论文/工具导读、社区资料与实验卡片
resources/              # 官方与第三方精选资料索引
scripts/                # 上游同步与仓库检查脚本
templates/              # 笔记、作业和实验记录模板
UPSTREAM.md             # 官方快照的来源、分支和 commit
```

---

## 快速开始

1. 阅读 [`resources/official.md`](resources/official.md) 并选择课程版本。
2. 用 [`resources/2025-vs-2026.md`](resources/2025-vs-2026.md) 确认版本差异，避免混用接口。
3. 复制 [`templates/lecture-note.md`](templates/lecture-note.md) 开始写讲义笔记。
4. 在对应作业目录中实现代码，并按 [`templates/assignment-writeup.md`](templates/assignment-writeup.md) 记录设计与结果。
5. 数据集、模型权重和密钥只保存在本地；下载方式见各官方作业 README。

Spring 2026 全部 17 讲原创中文自学笔记见 [`notes/README.md`](notes/README.md)，并提供
[`术语表`](notes/GLOSSARY.md) 与 [`复习路线`](notes/READING-ROADMAP.md)。

系统化的官方题目索引、论文/工具导读、社区资料评估和可复现实验卡片见
[`experiments/README.md`](experiments/README.md)。该资料库采用 official-first 与 spoiler 分级，
并在 [`experiments/SOURCES.yaml`](experiments/SOURCES.yaml) 记录来源和许可。

---

## 进度

| 作业 | 状态 | 关键成果 |
| --- | --- | --- |
| A1 · Basics | 完成 | 75 runs、27 图、BPE/Transformer/训练实验 |
| A2 · Systems | 完成 | fused FlashAttention、DDP/FSDP、228 条性能记录 |
| A3 · Scaling | 完成 | 官方 IsoFLOPs + 22-run RTX proxy |
| A4 · Data | 完成 | 21/21 tests、真实 WET 过滤/去重分析 |
| A5 · Alignment | 完成 | GRPO/DPO/SFT 接口与 4480-row proxy |

详细进度见 [`PROGRESS.md`](PROGRESS.md)。

---

## 报告入口

| 作业 | 报告 |
| --- | --- |
| A1 · Basics | [`report/main.pdf`](assignments/assignment1-basics/report/main.pdf) |
| A2 · Systems | [`report/main.pdf`](assignments/assignment2-systems/report/main.pdf) |
| A3 · Scaling | [`report/main.pdf`](assignments/assignment3-scaling/report/main.pdf) |
| A4 · Data | [`report/main.pdf`](assignments/assignment4-data/report/main.pdf) |
| A5 · Alignment | [`report/main.pdf`](assignments/assignment5-alignment/report/main.pdf) |

---

## 引用与学术诚信

- 官方模板按其原始许可证保留版权说明，来源固定记录在 [`UPSTREAM.md`](UPSTREAM.md)。
- 第三方笔记和实现只在 `resources/` 中链接，不复制其内容。
- 先独立实现和记录失败过程，再查看参考答案；引用任何思路时在 writeup 中注明来源。
- 不将本仓库内容作为在读学生的课程提交。
- 作者：[ShaneLiu04](https://github.com/ShaneLiu04)。

---

## License

本仓库原创内容采用 [MIT License](LICENSE)。导入的官方材料继续适用各自目录中的原始许可证。
