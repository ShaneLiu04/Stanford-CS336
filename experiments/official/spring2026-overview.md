# CS336 Spring 2026 官方作业总览

> 本页是本仓库的中文导读，不是 Stanford 官方题面，也不是答案。实际要求以各目录中的 Spring 2026 handout、原始测试和课程通知为准。请先阅读课程的 AI policy；题面明确禁止用 AI 实现课程作业。本仓库内容仅用于非课程提交的个人研究与复现。

## 快速地图

| 作业 | 核心问题 | 官方主要交付 | 主要资源门槛 | 本仓库导读 / 报告 |
| --- | --- | --- | --- | --- |
| A1 Basics | 从字节级 BPE 到 Transformer 训练闭环 | `writeup.pdf`、`code.zip`，另有 leaderboard | OWT tokenizer 最多 100 GB RAM；实验预算以 B200 小时计 | [导读](a1-basics.md) · [报告源码](../../assignments/spring2026/assignment1-basics/report/main.tex) · [PDF](../../assignments/spring2026/assignment1-basics/report/main.pdf) |
| A2 Systems | profiling、FlashAttention、DDP、状态/参数分片 | `writeup.pdf`、`code.zip`，另有双 B200 leaderboard | Linux/CUDA/Triton；部分实验 2–6 GPU；榜单 2×B200 | [导读](a2-systems.md) · [报告源码](../../assignments/spring2026/assignment2-systems/report/main.tex) · [PDF](../../assignments/spring2026/assignment2-systems/report/writeup.pdf) |
| A3 Scaling | 用小规模试验外推 48 B200-hour compute-optimal 配置 | `writeup.pdf`、`code.zip`、API `/final_submission` | Stanford 网络/API；拟合预算硬上限 12 B200-hours | [导读](a3-scaling.md) · [报告源码](../../assignments/spring2026/assignment3-scaling/report/main.tex) · [PDF](../../assignments/spring2026/assignment3-scaling/report/writeup.pdf) |
| A4 Data | 从 Common Crawl 构造可训练、高质量、去重语料 | `writeup.pdf`、`code.zip` | 2,500 WET；最终训练 8×B200、约 8.6B tokens | [导读](a4-data.md) · [报告源码](../../assignments/spring2026/assignment4-data/report/main.tex) · [PDF](../../assignments/spring2026/assignment4-data/report/writeup.pdf) |
| A5 Alignment | 在 GSM8K 上实现并比较 GRPO 系列策略梯度 | `writeup.pdf`、`code.zip` | OLMo-2-1B；训练 GPU + vLLM GPU；大量 B200-hours | [导读](a5-alignment.md) · [报告 PDF](../../assignments/spring2026/assignment5-alignment/report/writeup.pdf) |

## 贯穿五份作业的工作流

1. **先锁版本。** 查看根目录 [`UPSTREAM.md`](../../UPSTREAM.md) 的固定 commit，再读对应 PDF；不要依赖上游默认分支当前内容。
2. **先过接口测试。** 官方测试通过 `tests/adapters.py` 调用实现；adapter 应保持为薄胶水，不应承载核心逻辑。
3. **再做小规模正确性。** CPU/Gloo、短序列、小数据和单卡只能回答正确性或趋势问题，不能冒充 NCCL、多卡 B200 或官方 leaderboard 测量。
4. **最后扩实验矩阵。** 每个 run 固化配置、随机种子、commit、硬件/软件版本、原始指标及失败/OOM 状态；图表应由原始记录生成。
5. **提交前用官方脚本。** 脚本名称在版本间有 `make_submission.sh` 与 `test_and_make_submission.sh` 的差异，必须以该快照 README/handout 为准。

## 资源受限学习路径

- **CPU / Windows：** A1 的 tokenizer 与模块单测、A3 的离线 IsoFLOP 拟合、A4 的小样本过滤和去重、A5 的纯张量 loss 单测。
- **单张 NVIDIA GPU：** A1 缩小模型/数据；A2 单卡 profiling 与 attention；A3 自建 proxy matrix；A4 小模型/小数据消融；A5 缩小模型或只跑 objective proxy。
- **多卡但非 B200：** 可验证通信、分片和相对趋势；报告设备拓扑与绝对数值，不向 B200 外推“官方成绩”。
- **没有课程凭据：** 不调用 A3 API，不声称 A4 共享数据/8×B200 结果，不声称 A5 官方 OLMo-2/B200 accuracy。用本地代理实验回答方法学问题，并明确不可比因素。

## 版本与授权风险

- [`UPSTREAM.md`](../../UPSTREAM.md) 固定了 A1–A4 的 Spring 2026 commit；上游后续修订不会自动进入本仓库。
- A5 的固定 commit 仅索引、未作为官方源码快照导入：同步时发现该版本没有许可证，公开可读不等于允许再发布。本地 A5 目录是独立自学实现与研究材料，不能据此推断官方源码授权。
- PDF 的版本号分别应在复现实验时记录（本快照 A3 `26.0.5`、A4 `26.0.1`、A5 `26.0.0`；其余以 PDF 首页为准）。
- 本仓库报告明确标注非课程提交；其数据、接口扩展和代理实验可能超出官方最低要求，不能反向当作题面。

## 官方题面入口

- [A1 handout](../../assignments/spring2026/assignment1-basics/cs336_assignment1_basics.pdf)
- [A2 handout](../../assignments/spring2026/assignment2-systems/cs336_assignment2_systems.pdf)
- [A3 handout](../../assignments/spring2026/assignment3-scaling/cs336_assignment3_scaling.pdf)
- [A4 handout](../../assignments/spring2026/assignment4-data/cs336_assignment4_data.pdf)
- [A5 handout](../../assignments/spring2026/assignment5-alignment/cs336_spring2026_assignment5_alignment.pdf)
- [A5 可选 safety/RLHF supplement](../../assignments/spring2026/assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)
