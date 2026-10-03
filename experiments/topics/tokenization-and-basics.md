# Tokenization and Basics：从字节到可训练语言模型

本主题回答两个相连的问题：怎样把开放词表文本稳定地映射为整数序列，以及怎样把这些序列变成一个可训练、可评估的 Transformer language model。建议先把完整训练链路跑通，再讨论架构细节；否则 tokenizer、初始化、优化器和采样中的错误很容易被混在一起。

## 推荐阅读顺序

### 1. 先建立端到端地图

- **官方 lecture**：[Lecture 1 — Overview, tokenization](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py) 与 [Lecture 2 — PyTorch、FLOPs、memory、arithmetic intensity](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)。
- **解决的问题**：明确训练数据如何经过 normalization、tokenization、batching、forward、loss、backward 和 optimizer update；同时学会按 tensor shape 与资源成本检查实现。
- **关键结论**：语言模型训练目标是 next-token negative log-likelihood
  \[
  \mathcal L=-\frac1T\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}),
  \qquad \mathrm{PPL}=e^{\mathcal L}.
  \]
  不同 tokenizer 的 token loss/PPL 不能直接横比，因为一个 token 覆盖的字节数不同；跨 tokenizer 比较宜换算为 nats/byte 或 bits/byte。
- **对应作业**：[Assignment 1 — Basics](https://github.com/stanford-cs336/assignment1-basics) 的 tokenizer、resource accounting、training loop 与 generation；本仓库快照位于 `assignments/spring2026/assignment1-basics/`。

### 2. 实现 byte-level BPE，而不是只调用现成库

- **论文**：[Sennrich et al., 2016, Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) 介绍 BPE 进入 subword modeling 的动机；[SentencePiece](https://aclanthology.org/D18-2012/) 展示从 raw text 直接训练语言无关 subword model 的工程方案。
- **工具文档**：[OpenAI tiktoken](https://github.com/openai/tiktoken) 可用于结果与速度基准；它是参考工具，不应替代作业中的独立实现。
- **解决的问题**：word vocabulary 有 out-of-vocabulary，纯 character/byte sequence 又过长。byte-level BPE 从 256 个字节符号开始，反复合并训练语料中高频相邻 pair，在开放词表能力与序列长度之间折中。
- **关键结论**：
  - 训练时每轮选取频数最高 pair 并更新 merge ranks；编码时按已学习 rank 应用合法 merge，而不是重新按输入频率训练。
  - byte-level 方案对任意 UTF-8 输入可逆，但 token boundary 不等于字符、词或语义边界。
  - `bytes/token` 越小通常代表压缩越强，但 vocabulary embedding、领域迁移和 downstream 效果也要一起考虑。
- **对应作业**：A1 的 BPE training、special token、streaming encode/decode 与 tokenizer benchmark。先用极小语料手算 merge，再测 round-trip、chunk boundary 和 special-token precedence。
- **局限**：BPE 的贪心频率目标不等价于最优语言模型目标；Unicode normalization、脏字节和多语言分布会改变统计。比较 tokenizer 时必须固定语料与 normalization。

### 3. 理解现代 decoder-only Transformer

- **官方 lecture**：[Lecture 3 — Architectures, hyperparameters](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf) 与 [Lecture 4 — Attention alternatives and mixture of experts](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)。
- **论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[RoFormer / RoPE](https://arxiv.org/abs/2104.09864)、[RMSNorm](https://arxiv.org/abs/1910.07467) 和 [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。
- **解决的问题**：用可并行训练的 causal self-attention 建模长程依赖，并通过 normalization、position encoding 和 gated MLP 改善优化。
- **关键公式/结论**：
  \[
  \operatorname{Attention}(Q,K,V)=
  \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M_{\text{causal}}\right)V.
  \]
  RoPE 对 query/key 成对维度施加位置相关旋转，使内积自然携带相对位置信息；RMSNorm 仅按 root mean square 缩放；SwiGLU 以门控乘法替代普通激活。pre-norm 通常比 post-norm 更利于深层网络的梯度传播，但不是无条件最优。
- **对应作业**：A1 的 linear/embedding、RoPE、causal multi-head attention、RMSNorm、SwiGLU 与完整 Transformer LM；用 shape test、gradient test 和小模型 overfit test 分层验证。
- **局限**：标准 attention 的 score matrix 是 \(O(T^2)\)；RoPE 外推、head dimension、normalization placement 与初始化相互耦合，不能仅凭单次小规模 ablation 推广到大模型。

### 4. 补齐优化与训练稳定性

- **论文**：[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)、[Decoupled Weight Decay Regularization / AdamW](https://arxiv.org/abs/1711.05101)。
- **工具文档**：[PyTorch automatic mixed precision](https://pytorch.org/docs/stable/amp.html)、[gradient clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)。
- **解决的问题**：在有限 compute 下稳定降低 validation loss，并让 checkpoint/resume 与实验记录可复现。
- **关键公式/结论**：Adam 用一、二阶指数滑动平均及 bias correction 缩放更新；AdamW 将 weight decay 与自适应梯度更新解耦。常用 cosine schedule 为
  \[
  \eta_t=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})
  \left(1+\cos(\pi\,p_t)\right),
  \]
  前置 linear warmup 可降低训练初期不稳定。global norm clipping 限制异常 step，但频繁触发通常意味着更深层问题。
- **对应作业**：A1 的 AdamW、warmup/cosine schedule、gradient clipping、checkpoint 和 hyperparameter sweep。
- **局限**：FP16 需要 loss scaling；BF16 的 exponent range 更安全但精度仍有限。只保存 model weights 不足以精确续训，还要保存 optimizer、scheduler、step、RNG state 与数据位置。

### 5. 最后做评估与消融

- 固定 validation split、token budget 和随机种子，分别改变一个因素：RoPE、norm placement、FFN、weight tying、batch size。
- 同时报告 validation loss、tokens/s、peak memory、参数量和至少三个 seeds；不要以训练 loss 代替泛化。
- generation 只能作定性 sanity check。温度采样
  \[
  p_i(\tau)\propto \exp(z_i/\tau)
  \]
  与 top-\(p\) 会改变输出分布，但“看起来更好”不是可重复指标。

## 最小实践闭环

1. 在几十行文本上训练 BPE，验证 encode/decode 与 special token。
2. 让小 Transformer 过拟合一个 batch，排除 mask、shift 和 optimizer 错误。
3. 在固定 token budget 下训练 baseline，记录 config、commit、seed 与环境。
4. 做单变量 ablation，并用 nats/byte 处理跨 tokenizer 比较。
5. 再进入 Systems 主题优化速度；未先确认数学正确性时，性能数字没有解释力。

## 阅读边界

这些材料给出的是设计原理与可验证实现路径，不存在适用于所有语料和规模的“最佳” tokenizer 或超参数。A1 的 TinyStories/OpenWebText 结果主要用于教学；模型规模、数据质量和硬件变化都会改变架构与优化结论。
