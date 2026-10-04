# CS336 Spring 2026 十七讲自学笔记

原创中文笔记，保留英文术语；不逐页翻译，也不替代官方 lecture materials。
重点连接“概念—公式—shape/复杂度—代码—实验—作业”。

## 课程索引

| # | 日期 | 官方主题 | 笔记 | 作业/实践 |
|---:|---|---|---|---|
| 1 | 03-30 | Overview, Tokenization | [L1](lecture-01-overview-tokenization.md) | A1 BPE |
| 2 | 04-01 | PyTorch, Resource Accounting | [L2](lecture-02-pytorch-accounting.md) | A1 accounting |
| 3 | 04-06 | Architectures, Hyperparameters | [L3](lecture-03-architectures-hyperparameters.md) | A1 model/training |
| 4 | 04-08 | Attention Alternatives, MoE | [L4](lecture-04-attention-moe.md) | architecture extensions |
| 5 | 04-13 | GPUs, TPUs | [L5](lecture-05-gpus-tpus.md) | A2 profiling |
| 6 | 04-15 | Kernels, Triton | [L6](lecture-06-kernels-triton.md) | A2 FlashAttention |
| 7 | 04-20 | Parallelism I | [L7](lecture-07-parallelism-percy.md) | DDP/collectives |
| 8 | 04-22 | Parallelism II | [L8](lecture-08-parallelism-tatsu.md) | TP/PP/FSDP |
| 9 | 04-27 | Scaling Laws I | [L9](lecture-09-scaling-laws-i.md) | A3 IsoFLOPs |
| 10 | 04-29 | Inference | [L10](lecture-10-inference.md) | serving/KV cache |
| 11 | 05-04 | Scaling Laws II | [L11](lecture-11-scaling-laws-ii.md) | fitting/extrapolation |
| 12 | 05-06 | Evaluation | [L12](lecture-12-evaluation.md) | metrics/error analysis |
| 13 | 05-11 | Data: Sources, Datasets | [L13](lecture-13-data-sources.md) | A4 provenance |
| 14 | 05-13 | Data: Filtering, Dedup, Mixing | [L14](lecture-14-data-filtering-dedup.md) | A4 pipeline |
| 15 | 05-18 | Mid/Post-training: SFT, RLHF | [L15](lecture-15-sft-rlhf.md) | A5 SFT/DPO |
| 16 | 05-20 | Post-training: RLVR | [L16](lecture-16-rlvr.md) | A5 GRPO |
| 17 | 05-27 | Multimodal Alignment | [L17](lecture-17-multimodal-alignment.md) | multimodal extension |

官方课程主页：https://cs336.stanford.edu/
官方 lecture sources：https://github.com/stanford-cs336/lectures

## 辅助导航
- [术语与符号表](GLOSSARY.md)
- [按目标/时间预算的阅读路线](READING-ROADMAP.md)
- [论文、工具与社区资料库](../experiments/README.md)
- [A1–A5 完整实现与报告](../assignments/spring2026/)

## 笔记使用法
1. 首读每讲的“目标→核心概念→小结”。
2. 二读公式并手算 shape、FLOPs、bytes、通信量。
3. 按“实现映射”运行测试或实验，不只阅读代码。
4. 用“易错点”做反例检查，再完成练习。
5. 回到对应作业报告核对真实指标与资源边界。
6. 面试冲刺：逐题口述“大厂面试真题与答题框架”，用“系统设计题”练白板架构、
   “代码实现题”限时手写；“行业现状与最新进展（2024–2026）”补工业界视野，
   “必背数字（工业界参照）”供面试前 5 分钟速记。

## 每讲深度版结构（面试增强）

除原有的“摘要→目标→核心概念→关键机制→实现映射→易错点→练习→作业关联→讨论”
主链外，每讲在“面试要点速记”后新增四个模块：

| 模块 | 作用 |
|---|---|
| 行业现状与最新进展（2024–2026） | 本讲概念对照工业界实践与关键数字 |
| 大厂面试真题与答题框架 | 6–7 道：考点 + 可口述框架 + 加分项 + 踩坑 |
| 系统设计题 | 需求澄清→规模估算→架构→trade-off→评测→追问 |
| 代码实现题 | 可动手的 Python 骨架 + 考察点 + 验收标准 |

同时每讲扩充“高频问题”“必背数字（工业界参照）”“延伸阅读（含访问日期）”与“易错点”。

截图、长引用和第三方代码必须先确认许可；本目录只写原创摘要和短引用。
