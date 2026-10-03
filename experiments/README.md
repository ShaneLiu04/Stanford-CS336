# CS336 Experiments 学习资料库

本目录以 Spring 2026 五份官方作业为骨架，把题目要求、核心论文、工具文档、
社区材料和本仓库已完成实验组织成可检索的自学知识库。

> Official-first。先阅读本地 handout 并独立实现，再打开标为 spoiler 的社区材料。
> 本目录只提供原创摘要、引用和明确许可的小型副本，不是官方答案集。

## 四种入口

### 按作业
- [课程与五阶段总览](official/spring2026-overview.md)
- [A1 Basics](official/a1-basics.md)
- [A2 Systems](official/a2-systems.md)
- [A3 Scaling](official/a3-scaling.md)
- [A4 Data](official/a4-data.md)
- [A5 Alignment](official/a5-alignment.md)

### 按主题
- [Tokenization、Transformer 与训练基础](topics/tokenization-and-basics.md)
- [GPU Systems、FlashAttention 与分布式](topics/systems.md)
- [Scaling Laws 与外推](topics/scaling-laws.md)
- [Data Curation、过滤与去重](topics/data-curation.md)
- [SFT、DPO、GRPO 与 Alignment](topics/alignment.md)

### 按社区资料
- [社区资料使用规则](community/README.md)
- [A1](community/a1.md) · [A2](community/a2.md) · [A3](community/a3.md) ·
  [A4](community/a4.md) · [A5](community/a5.md)

### 按可复现实验
- [BPE、训练与架构消融](catalog/a1-training.md)
- [Fused FlashAttention 与 Systems](catalog/a2-systems.md)
- [IsoFLOP 与 Proxy Scaling](catalog/a3-scaling.md)
- [WET Filtering 与 Deduplication](catalog/a4-data.md)
- [GRPO Objective Mechanics](catalog/a5-alignment.md)

## 资源预算路线
- **CPU only**：官方题目索引、BPE 小样本、IsoFLOP 拟合、过滤/去重 fixtures。
- **单张 GPU**：A1 小模型训练、A2 kernel benchmark、A3/A5 proxy、A4 小规模 downstream。
- **多 GPU / hosted**：FSDP/NCCL、A3 B200 API、A4 Modal 8×B200、A5 2×B200。
  校外无法访问时必须明确标注 proxy，不伪造 leaderboard。

## Provenance
- [`SOURCES.yaml`](SOURCES.yaml)：URL、作者、版本、主题、license 与可信度。
- [`LICENSES.md`](LICENSES.md)：镜像和引用政策。
- [`source-report.md`](source-report.md)：自动校验结果。
- 官方快照 commit 见根目录 [`UPSTREAM.md`](../UPSTREAM.md)。

大型数据、模型权重、完整日志和 profiler trace 不入库；只保存配置、命令、
汇总指标、小型图表与校验和。
