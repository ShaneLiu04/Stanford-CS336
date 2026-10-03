# A1 Basics：从 tokenizer 到语言模型训练

> 中文导读；不替代[官方 handout](../../assignments/spring2026/assignment1-basics/cs336_assignment1_basics.pdf)，不包含题目答案。分值和预算按本地 Spring 2026 PDF 记录。

## 目标

独立搭出最小但完整的语言模型栈：理解 Unicode；训练和流式运行 byte-level BPE；实现 pre-norm Transformer（RMSNorm、RoPE、causal MHA、SwiGLU）；实现 loss、AdamW、schedule、梯度裁剪、批处理和 checkpoint；最后在 TinyStories / OpenWebText 上进行可复现实验与生成。

## Problem、交付物与分值

### Tokenizer

| Problem | 分值 | 核心 deliverable |
| --- | ---: | --- |
| `unicode1` / `unicode2` | 1 / 3 | Unicode code point、UTF-8 编解码分析及短答 |
| `train_bpe` | 15 | byte-level BPE 训练函数，通过 `run_train_bpe` |
| `train_bpe_tinystories` | 2 | TinyStories tokenizer 的时间、内存、词表观察 |
| `train_bpe_expts_owt` | 2 | OWT tokenizer；官方约束为无 GPU、≤12 小时、≤100 GB RAM |
| `tokenizer` | 15 | `Tokenizer` 的 encode/decode、special token、迭代/流式接口 |
| `tokenizer_experiments` | 4 | 语料内/跨语料压缩率、吞吐等比较 |

### Transformer 与训练基础设施

| Problem | 分值 | 核心 deliverable |
| --- | ---: | --- |
| `linear`, `embedding`, `rmsnorm` | 1 / 1 / 1 | 基础模块及 adapter |
| `positionwise_feedforward`, `rope` | 2 / 2 | SwiGLU 与 Rotary Position Embedding |
| `softmax`, `scaled_dot_product_attention` | 1 / 5 | 数值稳定 softmax、可 mask 的 SDPA |
| `multihead_self_attention` | 5 | batched causal MHA，并支持 RoPE 路径 |
| `transformer_block`, `transformer_lm` | 3 / 3 | pre-norm block 与完整 LM |
| `transformer_accounting` | 5 | 参数、FLOPs 与加载内存核算 |
| `cross_entropy`, `learning_rate_tuning` | 1 / 1 | 稳定交叉熵与简单 LR 行为实验 |
| `adamw`, `adamw_accounting` | 2 / 2 | AdamW 及训练内存/计算估算 |
| `learning_rate_schedule`, `gradient_clipping` | 1 / 1 | warmup-cosine 与全局 L2 裁剪 |
| `data_loading`, `checkpointing` | 2 / 1 | 随机 next-token batch、状态保存/恢复 |
| `training_together`, `decoding`, `experiment_log` | 4 / 3 / 3 | 训练脚本、temperature/top-p 采样、实验日志 |

### 实验

| Problem | 官方预算 / 分值 | 核心 deliverable |
| --- | --- | --- |
| `learning_rate` | 2 B200-hours / 3 | LR 曲线；TinyStories validation loss ≤1.45 |
| `batch_size_experiment` | 1 B200-hour / 1 | 从 batch 1 扫到显存上限，讨论吞吐/收敛 |
| `generate` | 1 | 至少 256 tokens（或提前 EOS）的样本与评论 |
| `layer_norm_ablation` | 0.5 B200-hours / 1 | 去 RMSNorm 对照 |
| `pre_norm_ablation` | 0.5 B200-hours / 1 | post-norm 对照 |
| `no_pos_emb` | 0.5 B200-hours / 1 | RoPE vs NoPE |
| `swiglu_ablation` | 0.5 B200-hours / 1 | SwiGLU vs SiLU |
| `main_experiment` | 2 B200-hours / 2 | OWT 学习曲线与生成 |
| `leaderboard` | 10 B200-hours / 6 | 在 0.75 B200-hour 训练窗口内优化验证 loss；另走榜单 PR |

## 关键测试接口

`tests/adapters.py` 是官方契约层，主要接口包括：

- tokenizer：`run_train_bpe(...) -> (vocab, merges)`、`get_tokenizer(vocab, merges, special_tokens)`；
- 模型：`run_linear`、`run_embedding`、`run_swiglu`、`run_rope`、`run_scaled_dot_product_attention`、两种 MHA、`run_transformer_block`、`run_transformer_lm`；
- 训练：`run_cross_entropy`、`get_adamw_cls`、`run_get_lr_cosine_schedule`、`run_gradient_clipping`、`run_get_batch`；
- 状态：`run_save_checkpoint`、`run_load_checkpoint`。

形状、dtype、state-dict key 和 mask 语义以 adapter docstring 与原始 tests 为准。不要把核心实现塞进 adapter，也不要修改 tests 来“通过”测试。

## 推荐实验矩阵

1. **Tokenizer：** `{TinyStories, OWT}` × `{10K, 32K vocab}`；记录训练 wall time、peak RSS、bytes/token、encode/decode throughput，并做跨语料压缩比较。
2. **正确性：** 每个模块先用 tiny tensor 与官方 tests；再测试整块/整模的 shape、causal leakage、checkpoint round-trip。
3. **主训练：** 固定 tokenizer、模型和 token budget，扫 learning rate 与 physical/global batch；至少 3 seeds 用于最终基线。
4. **受控消融：** 每次只改 RMSNorm、norm placement、position encoding、FFN activation 中一个因素；保持初始化、数据顺序、token budget 一致。
5. **系统扩展（非官方必需）：** precision、`torch.compile`、gradient accumulation、weight tying；单独标记为扩展，不与官方 leaderboard 混写。

每个 run 至少保存 config、seed、git commit、环境、逐步 train/val loss、tokens/s、peak VRAM、checkpoint 和失败状态。

## 硬件与时间

- tokenizer 训练以 CPU/RAM 为主；OWT 官方上限是 12 小时、100 GB RAM。
- 题面给出的 TinyStories 推荐配置在 1×B200 上约 20–30 分钟；各实验预算见上表。
- leaderboard 提交运行最多 45 分钟（B200），训练目标窗口为 0.75 B200-hour。
- 非 B200 的吞吐、显存和 wall time 不能直接与官方预算或排名比较；应报告实际 GPU、dtype、PyTorch/CUDA 版本。

## 提交物

- 运行本快照题面指定的 `make_submission.sh` 生成包；
- Gradescope：`writeup.pdf` 与 `code.zip`；
- 大数据、tokenized corpus 和 checkpoints 应从压缩包排除；
- leaderboard 是独立仓库 PR，不等于 Gradescope 提交；
- 报告中逐题给出文字回答、曲线/生成样本及复现命令。

## 资源受限替代

- CPU/集显：完整做 tokenizer、模块测试和短 smoke training；对 OWT 使用确定性 sample。
- 8–16 GB GPU：减小 `d_model`、层数、context、physical batch，用 accumulation 保持 global batch；用固定 token budget 比较趋势。
- 无法跑官方 leaderboard：报告本地 proxy 的模型、数据、预算和 loss，明确写“非 B200、非官方排名”。
- Windows 缺少 `resource` 模块时，两项 RSS 测试应在 Linux/WSL 复核，不能把 skip 当作通过。

## 版本风险与本仓库报告

- 固定上游 commit：`a158843b20107949f1a8d7df1b05cd33b9166712`，见 [`UPSTREAM.md`](../../UPSTREAM.md)；默认分支可能已变化。
- 本地实现加入了消融、compile、weight tying 等扩展，不能据此扩大官方要求。
- 本仓库实验报告：[源码](../../assignments/spring2026/assignment1-basics/report/main.tex) · [PDF](../../assignments/spring2026/assignment1-basics/report/main.pdf) · [资产说明](../../assignments/spring2026/assignment1-basics/report/README.md) · [验证记录](../../assignments/spring2026/assignment1-basics/VERIFICATION.md)。
