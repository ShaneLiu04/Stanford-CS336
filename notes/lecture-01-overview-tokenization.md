---
title: "Lecture 01 — Overview & Tokenization"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-03-30"
lecturer: "Percy Liang"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py"
  - "../assignments/spring2026/assignment1-basics/"
---

# Lecture 01 — Overview 与 Tokenization：从数据契约到离散符号系统

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
- 适用对象：自学者、数据/基础设施工程师、Tokenizer 与多语言 NLP 研究者

## 摘要

Tokenization 决定语言模型接收的离散符号、序列长度、词表参数、训练计算量以及 loss 的计量单位，
因而不是独立于模型的预处理步骤。本文以 byte-level Byte Pair Encoding（BPE）为主线，
从 Unicode/UTF-8 数据契约、pre-tokenization、special-token protocol、greedy merge
算法与复杂度出发，系统讨论 tokenizer 的可逆性、确定性、流式处理和工程优化。进一步结合
compression、nats/byte、数字与代码 tokenization、跨域迁移与 multilingual fairness，
提出可复现 benchmark 与 controlled experiment 设计，并总结 reference implementation、
property-based testing、artifact versioning 和线上兼容性要求。与仅描述算法流程的讲义不同，
本文重点回答四个问题：tokenizer 如何改变 downstream compute；何时工程优化会破坏语义契约；
数字与多语言场景为何是 frequency objective 的系统性盲区；token-free 架构如何改变
tokenizer 的必要性边界。

**关键词：** Language Modeling；Tokenization；Byte Pair Encoding；Unicode；
Pre-tokenization；Compression；Reproducibility；Multilingual Fairness

## 本文贡献

1. 给出 tokenizer 作为 **model/data systems contract** 的统一视角；
2. 从朴素 \(O(MN)\) 训练到倒排索引、heap、linked-list encoding，整理可验证优化路径；
3. 建立 compression、runtime、memory、nats/byte 与 downstream loss 的多维评价协议；
4. 将数字 tokenization、multilingual fairness 与 token-free/multiscale 架构纳入
   设计空间分析，明确 BPE 的系统性盲区与替代路线的适用边界；
5. 提供面向工程实践的测试金字塔、artifact schema、故障排查和研究型 ablation 模板；
6. 将理论讨论映射到本仓库 Spring 2026 A1 实现、测试与真实 GPU/CPU 实验。

> 本讲不是“先学一个分词工具”，而是建立语言模型训练的第一份数据契约：任意文本必须稳定、可逆地变成有限词表上的整数序列。Tokenizer 会同时改变序列长度、模型输入输出层大小、训练 FLOPs 和 loss 的解释方式。

## 学习目标

完成本讲后，应能：

1. 区分 Unicode code point、UTF-8 bytes、显示字形和 token；
2. 手算一次 byte-level BPE merge，并解释训练与编码为何是两个不同过程；
3. 正确处理 pre-tokenization、special token、merge rank 与流式编码；
4. 用 `bytes/token`、throughput、内存和 `nats/byte` 比较 tokenizer；
5. 讨论数字/代码 tokenization 与 token-free 架构对 tokenizer 设计空间的挑战；
6. 把 tokenizer 的每个概念定位到本仓库 A1 代码、测试和实验记录。

## 先修知识

- Python `str` / `bytes`、字典、正则表达式；
- 基本概率：条件概率、negative log-likelihood；
- 基本复杂度分析；无需预先掌握 Transformer。

## 核心概念与公式推导路线

本讲沿“Unicode/UTF-8 → byte vocabulary → pre-tokenization → BPE train/encode → LM metric”展开。
关键推导包括 pair frequency \(C(p)\)、最大 merge 数 \(V-256-S\)、next-token NLL，以及
从 token loss 到 nats/byte 的单位换算；每个公式都在后文同时给出 shape 或资源含义。

## 1. 从课程全景理解 tokenization

语言模型训练链路可压缩为：

\[
\text{raw text}\xrightarrow{\text{encode}}x_{1:T}
\xrightarrow{f_\theta}\text{logits}
\xrightarrow{\text{cross-entropy}}\mathcal L
\xrightarrow{\text{backprop}}\theta'.
\]

自回归目标为

\[
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}),\qquad
\mathcal L_{\text{tok}}=-\frac1T\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
\]

这一目标可视为对离散符号序列的 cross-entropy coding：在真实分布未知时，模型平均 NLL
刻画了使用模型分布编码数据的期望码长，与 Shannon 信息论建立直接联系
[[1]](#ref-1)。但 token 本身是人为构造单位，因此 `loss/token` 不是跨 tokenizer
不变量。

这里的 \(x_t\) 由 tokenizer 决定。相同文本若被切成不同数量的 tokens，`loss/token` 和 perplexity
\(\exp(\mathcal L_{\text{tok}})\) 就不在同一标尺上。Tokenizer 也改变：

- 序列长度 \(T\)：影响 attention 的 \(T^2\) 成本；
- 词表大小 \(V\)：影响 embedding / LM head 的 \(O(Vd)\) 参数与 \(O(BTdV)\) 计算；
- 数据 token budget：影响“训练过多少数据”的含义；
- 领域适配：Web tokenizer 未必压缩儿童故事最好。

因此 tokenizer 不是无关紧要的前处理，而是模型架构和实验定义的一部分。

### 1.1 用“系统接口”而不是“文本切词”理解 tokenizer

一个可训练、可部署的 tokenizer 至少定义五件事：

1. **输入规范**：接收 Unicode `str`、bytes，还是已经 normalization 的文本；
2. **边界规范**：pre-token、special token、document boundary 如何处理；
3. **模型规范**：vocabulary 与 ordered merge rules；
4. **序列化规范**：token ID 是否稳定，旧 checkpoints 能否继续使用；
5. **错误规范**：非法 UTF-8、未知 ID、超长文档如何失败或恢复。

只保存 `vocab.json` 而不保存 merge order、normalization、regex 和 special-token policy，
无法完整复现实验。Tokenizer 一旦变化，embedding/LM head 的每一行语义都会变化；
即使 vocabulary size 相同，也不能直接加载旧模型权重。

### 1.2 Tokenizer 设计空间

| 方法 | 基本单位 | 训练思想 | 优点 | 常见代价 |
|---|---|---|---|---|
| byte-level BPE | bytes | 频繁 pair merge | 开放词表、工程成熟 | greedy merge、领域噪声 |
| WordPiece | 字符/子词 | 最大化似然增益近似 | BERT 生态成熟 | `[UNK]`/实现差异 |
| Unigram LM | 候选子词 | 删除低贡献 token | 有概率模型、可采样 | 训练更复杂 |
| SentencePiece | raw text framework | BPE/Unigram + whitespace symbol | 不依赖外部分词 | normalization 需固定 |
| pure bytes | bytes | 无训练 | 完全可逆、词表极小 | 序列长 |
| word tokenizer | words | 词表计数 | 易解释 | OOV、超大词表 |

选择方法时，先写清 objective：追求训练吞吐、跨语言公平、压缩率、形态学边界，
还是 checkpoint compatibility。不存在脱离数据和模型的“最佳 tokenizer”。

### 1.3 相关工作脉络

BPE 原本是通用数据压缩算法，在 neural machine translation 中被系统化为 open-vocabulary
subword method [[2]](#ref-2)。GPT-2 将 byte-level base vocabulary 与 regex
pre-tokenization 结合，避免未知字符并形成现代 decoder-only LM 常见方案
[[3]](#ref-3)。SentencePiece 把 normalization、whitespace 与 BPE/Unigram LM 统一在
raw-text framework 内 [[4]](#ref-4)；Unigram LM 还支持 subword regularization，
通过采样多个 segmentation 改善鲁棒性 [[5]](#ref-5)。

后续研究表明，greedy BPE 不是给定 merge budget 下的全局最优压缩方案，其训练可从
combinatorial optimization 角度分析 [[6]](#ref-6)；BPE-dropout 则随机丢弃 merge，
在不改变 vocabulary 的情况下引入 segmentation noise [[7]](#ref-7)。
Multilingual 研究发现 tokenizer fertility 与 vocabulary allocation 会造成语言间显著差异，
专用 tokenizer 在多个语言/任务上改善 monolingual performance [[8]](#ref-8)。
这类差异不仅是 accuracy 问题，也会直接改变推理 token 数、延迟和成本。

另一条路线完全减少对 subword vocabulary 的依赖：ByT5 直接建模 UTF-8 bytes
[[9]](#ref-9)，CANINE 在 character level 编码后下采样 [[10]](#ref-10)，
Charformer 学习可微的 subword-like representations [[11]](#ref-11)。
它们避免固定词表的 OOV/fairness 问题，但通常以更长序列和更高 encoder compute 为代价。
近年 multiscale 与 dynamic-patching 架构显著改善了这一权衡：MEGABYTE 把 byte
序列切成 local patches 由小模型处理，再用 global Transformer 聚合 patch
representations，使 byte-level 建模扩展到百万字节序列 [[12]](#ref-12)；
Byte Latent Transformer 放弃静态 tokenizer，按 next-byte entropy 动态构造
patch，在匹配推理 FLOPs 的条件下与 BPE-based 模型竞争，并在 noise/robustness
上占优 [[13]](#ref-13)。这表明"byte 序列过长"的代价可以被架构吸收，
tokenizer 的必要性边界正在被重新划定——但 BPE 仍是当前训练-推理生态的事实
标准（checkpoint、serving、工具链均以固定词表为契约）。

数字 tokenization 则暴露了 frequency objective 的语义盲区：常见整数被 merge
成不规则片段，算术错误中相当部分可直接归因于 digit tokenization 的分组方式
而非模型能力 [[14]](#ref-14)（详见 §2.4）。

## 2. Unicode、UTF-8 与 byte-level 建模

### 2.1 四个容易混淆的层次

| 层次 | 示例“牛” | 关键性质 |
|---|---|---|
| 字形 grapheme | 屏幕上的“牛” | 由字体与渲染决定 |
| Unicode code point | `U+725B` | 抽象字符编号 |
| UTF-8 bytes | `E7 89 9B` | 3 个 bytes，变长编码 |
| tokenizer token | 1–3 个或更多 IDs | 取决于训练出的 merges |

Python `len("牛") == 1`，但 `len("牛".encode("utf-8")) == 3`。不能逐 byte 调用 UTF-8
`decode`：continuation byte 单独不是合法字符。Byte-level tokenizer 的正确边界是：

1. 整段文本先编码为 UTF-8 bytes；
2. 在 byte 序列上做 BPE；
3. decode 时先拼回完整 bytes，再整体解码。

### 2.2 为什么从 256 个 bytes 起步

Word vocabulary 会遇到 out-of-vocabulary；Unicode code point vocabulary 很大且组合复杂；
纯 byte vocabulary 永远能表示任意输入，却会拉长序列。Byte-level BPE 从
\(\{0,\ldots,255\}\) 开始，学习常见 byte 片段，在“开放词表”和“较短序列”之间折中。

若 UTF-8 输入合法，且没有人为丢弃信息，则理想性质是

\[
\operatorname{decode}(\operatorname{encode}(s))=s.
\]

它只保证字节可逆，不保证 token boundary 对齐字符、词或语义。一个中文字符可能跨多个 token，
一个 token 也可能覆盖空格加英文词。

### 2.3 Unicode normalization 是数据决策

视觉上相同的字符可能有不同 code-point 序列。例如 `é` 可表示为单个 U+00E9，也可表示为
`e` + combining acute accent。NFC/NFKC normalization 能减少表面变体，但 NFKC 还可能改变
数学符号、全角字符等语义。工程上必须回答：

- normalization 在训练语料前做，还是 tokenizer 内部做；
- encode/decode 的“可逆”是恢复 normalization 后文本，还是原始 bytes；
- 用户输入和训练数据是否走完全相同路径；
- 是否需要保留原文用于审计或安全取证。

研究报告应把 normalization 作为 controlled variable，而不是隐含默认值。对多语言数据，
至少统计 script/category 分布，并测试 emoji、ZWJ、combining marks、RTL text 和 malformed bytes。

### 2.4 数字、代码与结构化文本

BPE 的 merge 由 corpus frequency 驱动，而数字的分布性质（长尾、进位结构、对齐方式）
与自然语言词完全不同：`1234` 可能被切成 `12`+`34`、`123`+`4` 或逐位，同一个数值
在不同上下文中片段化方式不同，模型因此难以学习稳定的位值/进位结构。Singh 与
Strouse 对 frontier LLM 的系统实验表明，digit 的分组方式（逐位 vs 固定宽度分组）
显著改变加减乘法准确率，部分模型的算术失误可通过更换 number tokenization 直接
修复 [[14]](#ref-14)。工程含义：

- 训练数据中数学/财务/表格占比高时，应考虑强制逐位（或固定 3 位分组）的
  pre-tokenization 规则，而不是接受无约束 merge；
- 代码场景中，缩进、camelCase/snake_case、运算符与换行符的切分质量同样影响
  downstream：应把 code slice 纳入 tokenizer 评估，而不是只看自然语言压缩率；
- 与 `bytes/token` 一样，这些规则属于 tokenizer contract，一旦上线便不可静默
  更换（embedding 语义随之改变）。

## 3. Byte Pair Encoding（BPE）

### 3.1 训练算法

设 pre-token multiset 为 \(\mathcal W\)，某个 byte-symbol 序列 \(w\) 的语料频数为 \(c(w)\)。
相邻 pair \(p=(a,b)\) 的计数是

\[
C(p)=\sum_{w\in\mathcal W}c(w)\,
\#\{i:(w_i,w_{i+1})=(a,b)\}.
\]

每轮：

1. 选择 \(p^\*=\arg\max_p C(p)\)；
2. 新建 symbol \(ab\)；
3. 在每个 pre-token 内把不重叠的 \(a,b\) 替换为 \(ab\)；
4. 记录 merge 顺序，直至词表达到目标大小或已无 pair。

若初始有 256 个 byte tokens、\(S\) 个不重复 special tokens、目标词表 \(V\)，最多执行

\[
M=V-256-S
\]

次 merges。A1 的 TinyStories 10K 词表和一个 special token 对应 \(9743\) 次 merges。
需要强调：greedy 选择当前最大频 pair 是 heuristic，而不是对最终 compression ratio 的
全局最优保证；形式化结果与近似性质见 Zouhar 等人的分析 [[6]](#ref-6)。

### 3.2 一个手算例子

语料中 `low` 出现 5 次、`lower` 2 次。初始按 bytes 表示：

```text
(l,o,w): 5
(l,o,w,e,r): 2
```

pair 计数为 \(C(l,o)=7,\ C(o,w)=7,\ C(w,e)=2,\ C(e,r)=2\)。
若 tie-break 规定字典序更大的 pair 优先，则必须固定并测试该规则；否则并行或哈希遍历顺序会使
vocabulary 不可复现。合并顺序不是集合：编码时 rank 较早的 merge 优先。

### 3.3 训练与编码不可混为一谈

- **训练**：根据 corpus frequency 学出 ordered merges；
- **编码**：在一个输入 pre-token 中，反复应用当前可用且 rank 最小的 merge；
- **错误做法**：编码新句子时重新选择“该句最高频 pair”，这等于重新训练，无法保持词表稳定。

本仓库 `Tokenizer._encode_bytes` 每轮扫描相邻 pair，选择最小 `merge_ranks`，再一次合并该
pair 的所有不重叠出现。这是清晰的参考实现，但对超长 pre-token 仍有重复扫描成本。

### 3.4 训练性能：不要每轮重扫全语料

朴素实现每次 merge 都重新扫描所有 pre-tokens，若有 \(M\) 次 merges、总 symbol 数 \(N\)，
上界接近 \(O(MN)\)。更实用的数据结构是：

- `word_counts`: pre-token tuple → corpus frequency；
- `pair_counts`: pair → weighted count；
- `pair_to_words`: pair → 含该 pair 的 pre-token IDs；
- max-heap：按 `(-count, tie_break_key)` 取候选；
- lazy invalidation：heap 中旧计数弹出时与当前 `pair_counts` 对比。

合并 pair \(p\) 后，只更新 `pair_to_words[p]` 指向的 pre-tokens。对每个被修改序列：

1. 减去旧相邻 pairs 的 weighted counts；
2. 进行 non-overlapping merge；
3. 加上新相邻 pairs；
4. 更新倒排索引和 heap。

这里最容易出错的是**重叠 pair**。序列 `a a a` 中 pair `(a,a)` 出现两次，但一次 merge
只能得到 `aa a`，不能同时合并两个重叠位置。计数定义、替换语义和局部增量必须一致。

### 3.5 编码性能：从重复扫描到局部优先队列

参考实现反复扫描长度 \(L\) 的 pre-token，最坏可达 \(O(L^2)\)。生产 tokenizer 常把
symbols 放在 linked list 中，并为相邻 pair 建 priority queue：

```text
node: token_id, prev, next, alive
candidate: merge_rank, left_node, right_node, generation
```

每次弹出 rank 最小且仍相邻的 pair，合并后只把新邻居推入 queue，复杂度可接近
\(O(L\log L)\)。`generation` 或 alive/version check 用于丢弃 stale candidates。

不过优化前应先 profile：自然语言 pre-token 通常很短，复杂数据结构的常数可能大于简单扫描。
正确流程是记录 pre-token length histogram，再决定是否值得优化 tail latency。

### 3.6 Determinism 是可复现实验的一部分

并行 pre-tokenization、unordered hash map、heap tie 和文件遍历顺序都会改变同频 pair 的选择。
可复现训练至少固定：

- 文件列表与 document 顺序；
- regex/library 版本；
- special tokens 的顺序和 IDs；
- pair tie-break；
- worker 数变化时的 reduce 顺序；
- 输出 vocab/merges 的 canonical serialization 与 SHA-256。

Tokenizer artifact 应像模型 checkpoint 一样被 versioned，而不是由运行时“顺便重训”。

## 4. Pre-tokenization 与 special tokens

### 4.1 Pre-tokenization 的作用

若直接让 BPE 跨整份文件合并，高频片段可能跨越任意词和文档边界。GPT-2 风格 regex 先把文本分成
大致的文字、数字、标点和空白片段，使 merge 只在片段内发生。它不是最终 tokenizer，只是约束
BPE 的搜索空间。

仓库中的 `GPT2_PATTERN` 保留 leading space，因此常见 token 会是 `" the"` 而非 `"the"`。
这改善英文空格建模，却意味着 token boundary 不能直接当 word boundary。

### 4.2 Special token 必须是硬边界

`<|endoftext|>` 一类 token 有协议语义：

- 永远作为一个 token；
- BPE 不能跨过其边界；
- 多个 special tokens 重叠时应 longest match；
- 训练时是否从普通统计中排除，必须服从题面契约。

本仓库先按长度降序构造 escaped regex，再切分普通文本与 special token；流式并行预分块仅在单一
special delimiter 条件下启用。任意 byte offset 切块会切断 UTF-8 字符或 special token，是错误的。

### 4.3 Streaming 的语义边界

“能分块读取”不等于“任意切块后分别 encode 再拼接，与整串 encode 相同”。若一个 pre-token
跨 chunk，独立编码会错过跨 chunk merge。可选契约有：

1. 输入 iterable 的每个元素就是独立 document/line，边界有语义；
2. 维护 carry buffer，直到 regex 确认 pre-token 已结束；
3. 只在 special delimiter 或已知 record boundary 切块；
4. 接受不等价，但明确用于近似/吞吐优先场景。

测试应同时覆盖：chunk 正好落在 ASCII word 中、UTF-8 多字节中、special token 中和
连续空白中。内存测试必须测 peak RSS，而不只测 Python object 数量。

## 5. 评价 tokenizer

### 5.1 压缩率与跨 tokenizer loss

令 corpus 的 UTF-8 总长度为 \(N_{\text{byte}}\)，编码后 token 数为 \(N_{\text{tok}}\)：

\[
\text{bytes/token}=\frac{N_{\text{byte}}}{N_{\text{tok}}}.
\]

数值越大表示平均每个 token 覆盖更多 bytes、序列更短。它不是压缩文件的完整码长：token ID
本身需要 \(\lceil\log_2V\rceil\) bits，词表也有存储成本。

跨 tokenizer 比较语言模型时，可把 token loss 换为

\[
\text{nats/byte}
=\frac{\text{total NLL}}{N_{\text{byte}}}
\approx\frac{\mathcal L_{\text{tok}}}{\text{bytes/token}},
\qquad
\text{bits/byte}=\frac{\text{nats/byte}}{\ln 2}.
\]

本仓库报告中 OWT 10K 与 32K tokenizer 的 token loss 分别为 3.081 与 4.116；仅看 token loss
会误判 10K 更好。按各自压缩率归一化后约为 0.971 与 0.942 nats/byte，32K 实际更优。

### 5.2 还应报告什么

- train wall-clock、peak RSS；
- encode/decode bytes/s 与 tokens/s；
- 同域、跨域 `bytes/token`；
- round-trip、special-token precedence、chunk-boundary 正确性；
- 最长/异常 tokens，检查 mojibake 或重复噪声；
- 固定抽样规则与 document-level uncertainty。

报告的 OWT 32K BPE 实测约 2.97 小时、46.45 GiB peak RSS，说明瓶颈是 pair-count state，
不是最终 encode。更大的 vocabulary 也会扩大 \(Vd\) embedding/head，因此不能只追求压缩率。

### 5.3 高质量 benchmark 协议

Tokenizer benchmark 很容易被 I/O、cache 和 Python overhead 污染。建议把阶段拆开：

| 阶段 | 计时边界 | 主要指标 |
|---|---|---|
| 读取 | disk → bytes | MB/s、cold/warm cache |
| pre-tokenize | bytes → pre-tokens | bytes/s、worker scaling |
| BPE train | pre-tokens → vocab/merges | wall-clock、peak RSS |
| encode | text → IDs | bytes/s、tokens/s、p50/p95 doc latency |
| decode | IDs → text | tokens/s、round-trip failures |

最少报告 1 次 warm-up、5 次独立 timing、均值/标准差；大语料还需报告硬件、worker 数、
文件系统、regex/tokenizer 版本。对多进程实验，画 speedup 与 parallel efficiency：

\[
S_p=\frac{T_1}{T_p},\qquad E_p=\frac{S_p}{p}.
\]

若增加 workers 后吞吐不再增长，区分 CPU saturation、memory bandwidth、serialization、
process startup 和 storage bottleneck。不要只展示最快的一次运行。

### 5.4 Research-style tokenizer 比较

一个有解释力的研究问题应先写 hypothesis，再固定变量。例如：

> 在相同 vocabulary size 下，domain-matched tokenizer 会提高同域 bytes/token，
> 但跨域优势可能消失；其 downstream 收益取决于 sequence shortening 是否超过
> embedding/head 参数成本。

Controlled matrix：

| 训练 corpus | 词表大小 | eval corpus | tokenizer metric | downstream metric |
|---|---:|---|---|---|
| TinyStories / OWT | 10K / 32K | TinyStories / OWT | bytes/token、tok/s | nats/byte、step time |

要求模型 architecture、training tokens（最好按 bytes 同时报告）、optimizer、seed、compute
保持一致。只训练一个模型就归因于 tokenizer 是不充分的；至少多个 seeds，或把结论限制为
descriptive evidence。

Document-level bootstrap 比 token-level bootstrap 更合理，因为同一文档内 tokens 高度相关。
对文档指标 \(z_i\)，重复有放回抽样 documents，得到均值置信区间。若两 tokenizer interval
高度重叠，应报告“不足以区分”，而不是只比较小数点后三位。

Multilingual 场景还应报告 **fertility**（每词/每 normalization unit 的平均 tokens）和
per-language token cost。Rust 等人在控制 pretraining data 后发现，语言专用 tokenizer
在多数单语任务上优于共享 multilingual tokenizer，说明 vocabulary allocation 与下游性能
相关 [[8]](#ref-8)。Bostrom 与 Durrett 则指出 BPE segmentation 往往偏离形态边界，
Unigram LM 在若干任务上表现更好 [[15]](#ref-15)。这些工作提醒我们：
global bytes/token 最优并不保证每种语言公平，也不保证 downstream accuracy 最优。

Tokenizer 自身的训练也值得作为研究对象：Dagan 等人发现 tokenizer 训练语料在约
数十 GB 量级后收益进入平台期，而 domain adaptation 阶段对已有 tokenizer 继续训练
（小规模扩展词表并微调 embedding）能以极低成本改善领域下游损失
[[16]](#ref-16)。这给出两条工程准则：其一，tokenizer 训练 corpus 的规模与
组成应作为超参数显式报告，而不是默认"越多越好"；其二，领域适配不必从零重训
tokenizer，但任何词表扩展都必须与 embedding resize、special-token 审计和旧
checkpoint 迁移策略一起设计。

## 6. Shape 与复杂度

### 6.1 数据形状

```text
文本 str
  -> UTF-8 bytes: [N_byte]
  -> token IDs: [N_tok]
  -> sampled input x: [B, T]
  -> shifted target y: [B, T]
  -> embedding: [B, T, d]
  -> logits: [B, T, V]
```

当 \(V\le 65536\) 时，落盘 token IDs 可用 `uint16`；送入 embedding 前要转成 PyTorch
`int64`。A1 的 10K/32K 词表都满足这一条件。

### 6.2 成本边界

- 朴素 BPE 训练若每轮扫描全部 symbol：约 \(O(MN_{\text{sym}})\)；
- 本仓库用 pair counts、heap 与 `pair_to_words` 倒排索引局部更新，实际成本取决于受影响 pre-tokens；
- 当前单个 pre-token 编码最坏会反复扫描，粗略可到 \(O(L^2)\)；
- streaming encode 的额外空间应接近 \(O(L_{\max\ chunk})\)，而非 \(O(N_{\text{file}})\)；
- downstream full attention 随 token 长度为 \(O(T^2d)\)，因此少量 token-length 改善可能显著节省长上下文成本。

## 7. 面向工程的开发与测试方法

### 7.1 先写 contract，再写优化

推荐开发顺序：

1. 定义 encode/decode/special-token/invalid-input contract；
2. 用最小 reference implementation 通过 property tests；
3. 建立 correctness corpus 和 benchmark corpus；
4. profile 后只优化热点；
5. 优化实现与 reference 做 differential testing；
6. 序列化 artifact，记录版本与 checksum；
7. 在下游训练前冻结 tokenizer。

Reference 版本应短小、明显正确，即使慢；optimized 版本可以使用 multiprocessing、heap、
倒排索引或 Rust/C++，但必须逐样本与 reference 对齐。

### 7.2 测试金字塔

**Unit tests**

- 单个 pair merge、重叠 merge、tie-break；
- special token longest match；
- empty input、NUL、invalid ID；
- encode/decode Unicode edge cases。

**Property-based tests**

- 对随机合法 Unicode：`decode(encode(s)) == s`；
- 所有 encode IDs 都在 vocabulary range；
- 添加 unrelated special token 不应改变普通文本编码；
- serialization round-trip 后 IDs 不变。

**Differential tests**

- 小 vocab 与可信 reference 比较；
- streaming 与 whole-document 在合法边界上比较；
- 单进程与多进程训练比较 vocab/merge SHA。

**Performance regression**

- 固定 5 MiB/100 MiB corpus；
- 记录 wall-clock、peak RSS、throughput；
- CI 只设宽松 regression threshold，避免把共享 runner 抖动当 bug。

### 7.3 Artifact 与上线检查

模型发布时 tokenizer 目录至少包含：

```text
tokenizer/
  vocab.json
  merges.json
  special_tokens.json
  tokenizer_config.json
  provenance.json      # corpus/version/code SHA
  checksums.sha256
```

上线前验证：

- serving 与 training 使用同一 artifact；
- BOS/EOS/PAD IDs 与 model config 一致；
- prompt template 不会重复插入 special token；
- truncation 是按 tokens 而不是字符；
- logging 不泄漏原始敏感文本；
- tokenizer 升级有 migration/version gate，而不是静默替换。

## 8. 实现映射（本仓库）

| 概念 | 路径 / 符号 | 验证重点 |
|---|---|---|
| GPT-2 pre-tokenization | `assignments/spring2026/assignment1-basics/cs336_basics/tokenizer.py`：`GPT2_PATTERN` | leading space、Unicode 类别 |
| BPE training | 同文件：`train_bpe` | tie-break、special boundary、局部计数更新 |
| 流式/并行切块 | 同文件：`_chunk_boundaries`、`_count_pretokens` | 不切断 delimiter 和 UTF-8 |
| BPE encoding | 同文件：`Tokenizer._encode_bytes`、`encode` | merge rank，不按新频率 |
| decode / iterable | 同文件：`decode`、`encode_iterable` | bytes 拼接、惰性输出 |
| 接口契约 | `assignments/spring2026/assignment1-basics/tests/adapters.py`：`run_train_bpe`、`get_tokenizer` | shape/type/返回值 |
| 正确性测试 | `.../tests/test_train_bpe.py`、`test_tokenizer.py` | parity、special token、round-trip |
| 实验脚本 | `.../scripts/tokenizer_experiments.py` | 压缩率、吞吐、RSS |
| 实验结论 | `.../report/main.tex` 第 2 节；`report/problem_walkthrough.tex` | 区分实测、推导与缺失值 |

## 9. 易错点与反例

1. **把字符当 byte。** 中文、emoji 和 combining marks 会立刻暴露错误。
2. **逐 byte decode。** 合法多字节 UTF-8 会被误报为非法。
3. **让 merge 跨 pre-token/special token。** 训练词表与协议边界都会被污染。
4. **忽略 tie-break。** 频数相同会导致不同运行生成不同 merges。
5. **编码时按频率合并。** 正确依据是训练得到的 merge rank。
6. **任意文本块独立 encode。** 普通 pre-token 可能跨 chunk；`encode_iterable` 的语义是逐输入字符串，
   不自动保证任意分块与整串完全等价。
7. **用 `errors="replace"` 掩盖 round-trip bug。** 它适合防御非法 token ID 序列，不能替代合法输入测试。
8. **直接比较不同 tokenizer 的 PPL。** token 单位不同，优先报告 nats/byte。
9. **认为大词表必然更好。** 它缩短序列，也增加 embedding/head 参数并可能记住领域噪声。
10. **只测一个短样本。** Python 调用开销会扭曲 throughput，文档抽样噪声也会扭曲压缩率。
11. **把 tokenizer 训练语料当"越多越好"。** 收益到数十 GB 量级即进入平台期（Dagan 等
    2024），配比与清洗比堆量更关键。
12. **上线前不做 per-language 成本审计。** 相同语义内容跨语言 token 数差异可达数倍，
    直接转化为上下文容量、延迟与 API 计费的系统性差异。

## 10. 实践 Checklist

- [ ] 用 ASCII、中文、emoji、combining mark、NUL 做 encode/decode round-trip。
- [ ] 在 2–3 个 pre-tokens 上手算 pair counts 与前两次 merge。
- [ ] 验证相同频数 pair 的确定性 tie-break。
- [ ] 测试 overlapping special tokens 与连续 special tokens。
- [ ] 比较整串和合理边界下的分块编码。
- [ ] 固定 corpus sample，报告 bytes/token、bytes/s、tokens/s、peak RSS。
- [ ] 做 train-domain × eval-domain 的 \(2\times2\) 压缩率实验。
- [ ] 检查最长 tokens 与不可读 bytes，识别数据噪声。
- [ ] 跨 tokenizer 模型比较换算 nats/byte。
- [ ] 保存 vocabulary、ordered merges、normalization/pre-tokenization 规则和版本。

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| round-trip 乱码 | decode 是否先拼 bytes | 逐 token UTF-8 decode |
| vocab 每次不同 | 同频 pair 与 reduce order | 未固定 tie-break |
| special token 被拆 | split 顺序与 regex escaping | 未 longest match |
| 多进程结果不同 | chunk boundary、Counter merge | 切断 pre-token / 非确定 reduce |
| encode 极慢 | pre-token length histogram | 超长噪声串导致 \(O(L^2)\) |
| 算术/数值任务准确率异常 | digit merge 方式 | 检查数字切分规则与分组契约 [[14]](#ref-14) |
| 内存持续增长 | pair_to_words 清理、heap stale entries | 倒排索引未删除旧引用 |
| 下游 loss 异常 | IDs/BOS/EOS/PAD 与 checkpoint | tokenizer artifact 不匹配 |
| 吞吐很高但模型更慢 | \(V\)、\(T\)、LM head FLOPs | 只优化 tokenizer 本身 |

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| 训练 loss 突升或输出乱码 | tokenizer 版本与 checkpoint 配套性 | vocab/merges 未随权重一起版本化 |
| 算术/数字任务异常差 | 数字 tokenization 策略 | BPE 频率目标把数字切成任意碎片 |
| 多语言 bytes-per-token 偏高 | pre-tokenization regex 与语料配比 | regex 按英文空白设计、merge 域偏斜 |
| 生成出现 `<|endoftext|>` 类泄漏 | special token 过滤与 escape | 语料含模板字符串未消毒 |
| 流式编码在块边界出错 | pre-tokenization 回退窗口 | BPE merge 跨 chunk 边界被截断 |
| FLOPs/序列长度预算对不上 | bytes-per-token 换算与 EOT 计数 | tokenizer 变更改变序列长度与 loss 单位 |
| 与文献 loss 不可比 | loss 计量单位（token vs byte） | 统一换算 bits-per-byte 再比较 |
| vocab 查表返回 UNK | byte-level 回退是否实现 | 未实现 byte fallback 的 BPE 词表 |

## 11. 作业关联

- A1 `unicode1/2`：区分 code point、bytes 与非法 UTF-8；
- `train_bpe`：确定性 pair counting、merge 与性能；
- `train_bpe_tinystories` / `train_bpe_expts_owt`：资源和词表观察；
- `tokenizer`：encode/decode、special tokens、iterable；
- `tokenizer_experiments`：同域/跨域压缩与吞吐；
- `data_loading`：token array 到右移一位的 \([B,T]\) 训练对；
- `main_experiment`：跨 tokenizer loss 必须按 bytes/token 解释。

自学时应以本地 Spring 2026 handout 与原始 tests 为契约；仓库报告是可复现实验材料，不是社区答案模板。

## 12. 讨论：效度威胁与研究边界

### 13.1 Construct validity

- `bytes/token` 衡量序列压缩，不等价于语义质量、形态合理性或模型 accuracy；
- `tokens/s` 同时受 token 数和实现速度影响，不能单独代表 bytes/s；
- token-level perplexity 的单位随 tokenizer 变化，跨 tokenizer 应转为 nats/byte，
  但 nats/byte 仍不等价于 downstream task quality。

### 13.2 Internal validity

- 不同 tokenizer 若训练语料、normalization、special tokens、模型参数量或 byte budget
  不一致，无法把结果归因于 segmentation；
- vocabulary 增大会同时改变序列长度和 embedding/head 参数，必须二者同时 accounting；
- 单 seed training 差异可能来自初始化与 data order，而非 tokenizer。

### 13.3 External validity

- TinyStories 上的最佳 vocabulary 未必迁移到 Web、code、数学或 multilingual data；
- 200-document sample 无法代表完整 corpus long tail；
- Python reference implementation 的吞吐不能代表 Rust/C++ production tokenizer；
- 离线批处理结果不能直接推断 online serving 的 p99 latency。

### 13.4 Fairness、隐私与安全

高资源语言主导的 frequency objective 可能让低资源语言产生更高 fertility，从而获得更短
effective context、更多训练/推理成本和更高服务费用 [[8]](#ref-8)。Petrov 等人进一步
量化了这种不公平：相同语义内容在不同语言间的 token 数差异可达数倍，直接转化为上下文
容量、延迟与 API 计费的系统性差异 [[17]](#ref-17)。Tokenizer 还可能把
PII、URL、API key 或 offensive strings 记成单 token，增加 memorization 与 probing 风险。
因此 artifact audit 应包含 per-language cost、敏感 pattern tokenization、异常 control
characters 与 denial-of-service 长输入，而不仅是英文 compression。

## 面试要点速记

**高频问题与答题要点**

1. **Q：为什么现代 LLM 普遍采用 byte-level BPE？** 要点：任意字节串可编码、无
   OOV、可逆且确定；词表大小与压缩率可调。代价是低资源语言 bytes-per-token 偏高。
2. **Q：词表大小如何权衡？** 要点：大词表序列短（attention 与步数省），但
   embedding/softmax 参数多、低频 token 表示差；小词表反之。最优词表随模型规模
   变化，需 sweep 验证。
3. **Q：BPE 训练复杂度？** 要点：朴素每次 merge 全量统计为 O(L·M)；增量更新
   （相邻对计数+惰性堆）近似线性。
4. **Q：tokenizer 如何影响公平性与可比性？** 要点：不同语言 bytes-per-token
   不同→同样内容的训练/推理成本不均；跨 tokenizer 比较模型必须换算
   bits-per-byte，直接比 perplexity 无效。
5. **Q：tokenizer 变更后如何迁移旧模型？** 要点：token 语义随词表整体漂移，
   即使 \(V\) 相同也不能直接加载旧权重；低成本路线=已有 tokenizer 续训+
   词表小规模扩展+embedding resize（Dagan 等 2024）；上线必须 version gate。
6. **Q：模型算术差，怎么排查是不是 tokenizer 的锅？** 要点：先打印数字切分
   样例；digit 分组（逐位 vs 固定宽度）显著改变 frontier LLM 加减乘准确率，
   部分失误可仅更换 number tokenization 修复（arXiv:2402.14903）；修复后
   冻结规则并加算术 slice 回归。
7. **Q：tokenizer 训练语料要多少？** 要点：数十 GB 量级后收益进入平台期
   （Dagan 等 2024）；规模与配比应作为超参数显式报告，而不是默认越多越好。
8. **Q：token-free（如 BLT）会淘汰 tokenizer 工程吗？** 要点：BLT 按
   next-byte entropy 动态构造 patch，匹配推理 FLOPs 下与 BPE 模型竞争且
   noise/robustness 占优；但 checkpoint/serving/工具链仍以固定词表为契约，
   短期是职责重划而非消失。

**必背数字**

- GPT-2 词表 50,257；GPT-4 级 ~100k；special tokens 须与训练协议严格一致。
- pre-tokenization regex 决定 merge 的作用域，是 multilingual 行为的第一决定因素。

**工业界参照**

- 数据预算量级：GPT-2（2019）约 100B tokens、训练成本约 4 万美元；FineWeb 15T
  tokens（96 个 Common Crawl 快照）；Llama-3 405B 用 15T tokens；Qwen 预训练
  去重过滤后 2.2T tokens。Common Crawl 自 2007 年起爬取，到 2024 年已索引约
  27 亿网页。
- 词表量级：Qwen-1.8B 用 15 万+ token 词表，在 cl100k_base 基础上扩展中文/
  多语言，对中英代码高效编码；embedding 与 lm_head 不共享权重
  （tie_word_embeddings=False，以空间换性能）。
- tokenizer 训练语料：数十 GB 量级后收益进入平台期（Dagan 等 2024）；
  领域适配=续训+词表小规模扩展+embedding resize，成本低。
- 数字 tokenization：digit 分组方式（逐位 vs 固定宽度）显著改变 frontier LLM
  加减乘准确率，部分算术失误可通过更换 number tokenization 直接修复
  （arXiv:2402.14903）。
- 多语言公平性：相同语义内容跨语言 token 数差异可达数倍，直接转化为上下文
  容量、延迟与 API 计费的系统性差异；fertility 是核心指标（arXiv:2305.15425）。
- 动态 patch：BLT 按 next-byte entropy 构造 patch，匹配推理 FLOPs 下与
  BPE 模型竞争，noise/robustness 占优（arXiv:2412.09871）。

## 行业现状与最新进展（2024–2026）

### 主流词表规模与 token 预算量级

| 模型 / 数据集 | 词表规模 | 预训练 tokens | 备注 |
|---|---:|---:|---|
| GPT-2（2019） | 50,257 | ~100B | 训练成本约 4 万美元；byte-level BPE + regex 范式起点 |
| GPT-4 级（cl100k_base） | ~100k | — | tiktoken 事实标准；词表较 GPT-2 约翻倍 |
| Llama-3 405B | 约 128k（据报道） | 15T | 公开权重模型的数据预算标杆 |
| Qwen-1.8B | 15 万+ | 2.2T（去重过滤后） | 在 cl100k_base 上扩展中文/多语言，对中英代码高效编码 |
| FineWeb | —（数据集） | 15T | 96 个 Common Crawl 快照；CC 自 2007 年起，2024 年已索引约 27 亿网页 |

本讲概念 ↔ 工业界实践对照：

| 本讲概念 | 工业界实践/数字（2024–2026） |
|---|---|
| 词表大小 trade-off | 50,257（GPT-2）→ ~100k（cl100k_base）→ 15 万+（Qwen）：中文/多语言+代码是扩表主因 |
| `bytes/token` 与语料配比 | Qwen 扩表后对中英代码高效编码；FineWeb 用 96 个 CC 快照构建 15T tokens |
| tokenizer 训练语料规模 | 数十 GB 量级后收益平台期（Dagan 等 2024）：配比与清洗 > 堆量 |
| 数字 tokenization 契约 | digit 分组方式可修复部分 frontier LLM 算术失误（Singh & Strouse 2024） |
| 训练-推理一致性契约 | Qwen：embedding/lm_head 不共享权重；RoPE 逆频率矩阵用 FP32 而非 BF16 计算 |

### 数据流水线：FineWeb 与 FineWeb-edu 的工程经验

- FineWeb：trafilatura 抽取质量高于默认 WET；流水线为 URL 过滤→语言过滤→
  MinHash 去重→质量过滤；去重存在收益递减临界点，过度去重反而性能恶化。
- FineWeb-edu：用 llama-3-70b-instruct 对 50 万样本按教育质量 0–5 打分，
  过滤 <3 分；规模更小但性能超过 FineWeb 及其他公开数据集——说明
  tokenizer/数据质量评估可以用小模型标注完成。
- 与本讲的映射：A1 的 `train_bpe` 语料采样同样应"配比显式化"；报告
  tokenizer 训练语料的规模与组成，而不是默认越多越好。

### 数字、多语言与 token-free：三条工程修正路线

- 数字（arXiv:2402.14903）：逐位 vs 固定宽度分组显著改变加减乘准确率，
  部分模型的算术失误可通过更换 number tokenization 直接修复 → 数字切分
  规则应在训练前作为契约冻结。
- 多语言（arXiv:2305.15425）：相同语义内容跨语言 token 数差异可达数倍 →
  fertility 与 per-language cost 应进 CI 与发布审计。
- token-free（arXiv:2412.09871，本讲已引 [[13]](#ref-13)）：BLT 按
  next-byte entropy 动态构造 patch，匹配推理 FLOPs 下与 BPE 模型竞争、
  noise/robustness 占优 → "byte 序列长"的代价可被架构吸收；但
  checkpoint/serving/工具链仍以固定词表为契约，BPE 短期不会退场。

**对本讲学习者的启示**：本讲的抽象契约（可逆、确定、版本化、nats/byte
计量）在 2024–2026 的工业实践中全部被放大为真金白银的问题——词表扩到
15 万+ 换多语言效率，数字切分规则换算术准确率，fertility 换计费公平，
artifact 版本化换线上安全。学 BPE 不是学一个 1994 年的压缩算法，而是学
现代 LLM 数据系统的第一层接口；每条"工程修正"路线都对应本讲的一个章节
（§2.4 数字、§5.4 多语言、§7.3 artifact）。

## 大厂面试真题与答题框架

以下均为高频面试题（公开面经风格），不指向任何具体公司。

**题目 1：现代 LLM 词表为何从 5 万涨到 10 万–15 万+？如何为你的模型选词表大小？**
- 考点：词表大小的系统 trade-off；多语言/代码 compression 与参数成本。
- 答题框架：1) 收益：序列更短（attention \(T^2\) 与训练步数省）、多语言/代码
  fertility 改善（Qwen 15 万+ 词表对中英代码高效编码）；2) 代价：embedding/lm_head
  \(O(Vd)\) 参数与 \(O(BTdV)\) 计算、低频 token 表示差；3) 演进证据：GPT-2
  50,257 → cl100k_base ~100k → Qwen 15 万+，扩表主因是中文/多语言；4) 方法：
  按模型规模与语料配比 sweep，并与 embedding resize、softmax 成本一起核算。
- 加分项：Qwen 以不共享 embedding/lm_head（以空间换性能）承接大词表；
  Llama-3 405B 15T tokens 预算背景下序列长度的成本权重。
- 踩坑：只报压缩率不报参数/计算账；忽略低频 token 训练不足；忽略词表变更
  对旧 checkpoint 的破坏。

**题目 2：两个模型 tokenizer 不同，token loss 3.081 vs 4.116，谁更好？**
- 考点：loss/token 不是跨 tokenizer 不变量；nats/byte 换算。
- 答题框架：1) token 单位不同 → PPL/loss 不可直接比；2) 换算
  nats/byte ≈ loss/token ÷ bytes/token，bits/byte 再除以 \(\ln 2\)；3) 实例：
  本仓库 OWT 10K（3.081）→ 约 0.971 nats/byte，32K（4.116）→ 约 0.942，
  结论反转、32K 更优；4) 结论配 document-level 置信区间。
- 加分项：指出 bytes/token 本身有抽样口径问题（文档抽样、special token 计数）；
  同时报告参数与 FLOPs 账。
- 踩坑：直接比小数点后三位；用 token-level bootstrap。

**题目 3：BPE 训练朴素实现 O(MN)，如何优化？**
- 考点：增量 pair-count、倒排索引、heap + lazy invalidation、重叠 pair 语义。
- 答题框架：1) 朴素做法每轮全扫语料重新计数；2) 建 word_counts / pair_counts /
  pair_to_words 三张表；3) 每次 merge 只更新受影响 pre-tokens：先减旧相邻
  pair 的加权计数→non-overlapping 合并→加新 pair→更新倒排索引与 heap；
  4) max-heap 按 `(-count, tie_break)` 弹出，弹出时与 pair_counts 校验
  （lazy invalidation）；5) 固定 tie-break 与遍历顺序保证确定性。
- 加分项：重叠 pair（`a a a` 只能合并一个位置）的计数/替换一致性；本仓库
  实测 OWT 32K 约 2.97 小时、46.45 GiB peak RSS，说明瓶颈在 pair-count state
  而非 encode。
- 踩坑：忘校验 stale heap 项导致错误 merge；tie-break 不固定导致 vocab 不可
  复现；并行 reduce 顺序未固定。

**题目 4：业务要加新领域/新语言，tokenizer 如何迁移、旧模型怎么办？**
- 考点：domain adaptation、artifact versioning、embedding resize。
- 答题框架：1) 不必从零重训：tokenizer 训练语料数十 GB 后收益平台期
  （Dagan 等 2024）；2) 低成本路线：已有 tokenizer 续训+词表小规模扩展+
  embedding resize（Qwen 在 cl100k_base 上扩中文是先例）；3) 旧 checkpoint：
  新 token embedding 初始化+续训，审计 special token/BOS/EOS/PAD 与 ID
  稳定性；4) version gate：新旧 artifact 双跑 diff，通过后切换，可回滚。
- 加分项：artifact 带 provenance 与 SHA-256；指出即使 \(V\) 相同也不能直接
  加载旧权重（每行语义变了）。
- 踩坑：静默替换线上 artifact；只扩词表不 resize embedding；忘 merge rank
  是有序契约。

**题目 5：模型算术很差，如何判断是不是 tokenizer 的问题？**
- 考点：digit tokenization；frequency objective 的语义盲区；controlled ablation。
- 答题框架：1) 先取证：打印加减乘样例的数字切片（逐位 vs `12`+`34` 类任意
  片段）；2) 引证：Singh & Strouse（arXiv:2402.14903）表明 digit 分组方式
  显著改变 frontier LLM 加减乘准确率，部分失误可仅换 number tokenization
  修复；3) 实验：固定模型与数据，只改数字 pre-tokenization 规则（逐位/固定
  宽度）做对照；4) 工程：数字切分作为 tokenizer contract 冻结，配算术 slice
  回归测试。
- 加分项：机制解释（位值/进位结构难以从碎片化 token 学习）；代码场景
  （缩进、camelCase/snake_case）同理纳入 slice 评估。
- 踩坑：先怀疑模型能力忽略 tokenizer；改数字规则后不做其他任务回归与
  bytes/token 复查。

**题目 6：多语言产品中同样内容不同语言 token 数差数倍，怎么办？**
- 考点：fertility；多语言公平性的量化与修复。
- 答题框架：1) 量化：Petrov 等（arXiv:2305.15425）——相同语义内容跨语言
  token 数差异可达数倍，直接转化为上下文容量、延迟与 API 计费的系统性差异，
  fertility（每词平均 token 数）为核心指标；2) 审计：per-language
  fertility/bytes/token 进发布 gate；3) 修复：扩词表/语言专用续训（Qwen
  15 万+ 先例；Rust 等显示专用 tokenizer 单语更优）；4) 产品侧：按 bytes
  或语义单位折算计费的补偿方案。
- 加分项：低资源语言 effective context 更短的质量/安全影响；sensitive
  pattern 的 per-language tokenization 审计。
- 踩坑：只用英文 benchmark 验收；把公平性当纯伦理问题而非可量化工程指标。

**题目 7：BLT 等 token-free 路线会取代 BPE tokenizer 吗？**
- 考点：token-free 的必要性边界；工程生态惯性。
- 答题框架：1) BLT：按 next-byte entropy 动态构造 patch，匹配推理 FLOPs
  下与 BPE 模型竞争，noise/robustness 占优；2) 含义："byte 序列过长"的
  代价可由架构（multiscale/dynamic patching）吸收；3) 反面：checkpoint、
  serving、评测与工具链均以固定词表为契约，BPE 是事实标准；4) 定位：
  tokenizer 的职责边界被重划而非消失，tokenizer 工程能力迁移到 patch
  策略与数据契约。
- 加分项：ByT5/CANINE/Charformer/MEGABYTE 演进脉络；动态 patch 对
  next-byte entropy 估计质量的依赖。
- 踩坑：把"匹配 FLOPs 下竞争"夸大为"全面超越"；忽略 serving 复杂度与
  生态迁移成本。

## 系统设计题

**设计题 1：为"中文+代码+数学"领域模型设计 tokenizer 训练与评测方案**
- 需求澄清：语料配比（中/英/代码/数学各占比）、目标词表量级、是否兼容已有
  生态（如从 cl100k_base 续扩）、是否存在需迁移的旧 checkpoint、数字与
  标识符切分契约、serving 延迟与内存预算。
- 规模估算：tokenizer 训练语料数十 GB 量级即进入收益平台期（Dagan 等 2024）；
  词表参照 Qwen 15 万+（在 cl100k_base 上扩展中文/多语言）；预训练数据量级
  参照 Qwen 2.2T（去重过滤后）与 Llama-3 405B 的 15T。
- 架构：分层采样（按目标配比，清洗参照 FineWeb 式 URL 过滤→语言过滤→
  MinHash 去重→质量过滤）→ pre-tokenization regex（中文、数字、缩进/驼峰
  规则）→ byte-level BPE 训练（增量 pair-count+固定 tie-break）→ 词表审计
  （数字分组、代码 token、异常长 token）→ special token 协议 → artifact
  序列化（vocab/merges/regex/config/provenance/SHA-256）→ 评测台。
- trade-off：

| 决策 | 选项 A | 选项 B | 权衡 |
|---|---|---|---|
| 起点 | 从零训练 | 在 cl100k_base 上续训+扩词表 | A 自由度高、成本高；B 快且生态兼容（Qwen 先例） |
| 数字切分 | 逐位 | 固定宽度分组 | 逐位对算术最稳（可修复部分失误）；固定宽度序列更短 |
| 词表大小 | ~100k | 15 万+ | 序列长度/压缩率 vs embedding 参数与低频 token 质量 |
| 中文策略 | 强制字级边界 | 自由 merge | 字级可控、可逆、跨语言公平；自由 merge 压缩率更高 |

- 评测方案：分域 bytes/token（中文/代码/数学）；下游 nats/byte 对照（同模型
  预算）；中英平行文本 fertility；算术 slice（两种数字规则 ablation）；
  train/encode 吞吐与 peak RSS；round-trip、special-token、chunk 边界
  property tests。
- 追问预案：词表再加 5 万的代价（embedding 参数、低频 token、softmax）；
  旧模型迁移（embedding resize+续训）；数字规则变更后既有评测如何对齐；
  上线后发现问题如何回滚（version gate+双跑 diff）。

**设计题 2：tokenizer artifact 版本化与线上热更新**
- 需求澄清：更新频率与触发方（数据侧/产品侧）、是否允许既有 token ID 变更、
  训练与 serving 是否强制同 artifact、回滚时效要求、并行服务的模型-版本
  组合数。
- 规模估算：词表 ~100k–15 万+，embedding 行数随版本变化；相同语义内容跨语言
  token 数差异可达数倍 → 计费与限长逻辑对 tokenizer 版本高度敏感，版本切换
  必须可审计。
- 架构：artifact 注册中心（版本号、SHA-256、provenance：语料快照/代码/
  regex/config）→ 训练侧启动时 pin 版本 → serving 侧双加载、灰度流量切换 →
  计费与日志按 artifact 版本打标 → 一键回滚。数据流：新版本发布前跑固定
  prompt 集的新旧编码 diff（token 长度分布、差异率）→ 指标守门（fertility、
  bytes/token 回归阈值）→ 灰度 → 全量。
- trade-off：

| 决策 | 选项 A | 选项 B | 权衡 |
|---|---|---|---|
| 更新方式 | 静默热更新 | 版本化+灰度+可回滚 | A 快但 embedding 语义漂移、乱码风险；B 安全但双倍加载与流程成本 |
| ID 兼容 | 允许重排 ID | 只追加新 token | 追加可保旧 checkpoint 可用（配合 embedding resize）；重排需全量迁移 |
| 审计时机 | 发布前 gate | 上线后周期审计 | gate 防问题上线；周期才能覆盖语料漂移 |

- 评测方案：固定 prompt 集新旧 diff 报告；round-trip 与边界回归；per-language
  fertility 稳定性；线上 p95 延迟、cache 命中与 token 长度分布监控。
- 追问预案：新增 special token 与用户文本冲突（escape+longest match）；多模型
  共享同一新 artifact 的兼容矩阵；版本切换瞬间在途请求的语义一致性（按请求
  pin 版本）。

**设计题 3：多语言 API 的 tokenization 公平性审计系统**
- 需求澄清：覆盖语言清单与优先级、审计触发（每次 tokenizer 发布 vs 周期性）、
  指标口径（fertility / bytes/token / per-language 计费偏差）、阻断阈值还是
  仅告警。
- 规模估算：以平行语料（同义内容多语言版本）为基准；Petrov 等
  （arXiv:2305.15425）量化相同语义内容跨语言 token 数差异可达数倍，审计
  系统要能按语言给出相对英文基线的倍数分布。
- 架构：平行测试集 → 统一 tokenize → per-language 统计（fertility、
  bytes/token、相对基线比值）→ 阈值告警/阻断 → 报告归档进 artifact 审计
  记录（与 checksum、provenance 同级）。
- trade-off：

| 决策 | 选项 A | 选项 B | 权衡 |
|---|---|---|---|
| 修复手段 | 扩词表+续训（治本） | 计费/限长侧补偿（治标） | 扩词表成本高但改善所有下游（Qwen 先例）；补偿快但留性能债 |
| 指标 | fertility（每词 token 数） | bytes/token | fertility 直接反映成本；bytes/token 跨 tokenizer 可比 |
| 审计时机 | 发布 gate | 周期审计 | gate 防问题上线；周期覆盖语料漂移 |

- 评测方案：语言×指标 heatmap 与版本间回归对比；换算为"同义内容的上下文
  容量/延迟/费用差"；低资源语言样本不足时用平行新闻/宗教语料补充并标注
  不确定度。
- 追问预案：扩词表对其他语言的词表预算挤出；审计发现数倍差异时对 SLA 与
  定价的连锁调整；如何向业务方量化"公平性债务"。

## 代码实现题

**代码题 1：增量更新的 BPE pair-count（含重叠 pair 处理）**
- 题目：实现 `merge_word`：对单个 pre-token 应用一次 merge，增量维护
  `word_counts` / `pair_counts` / `pair_to_words`，并正确处理重叠 pair
  （`a a a` 中 `(a,a)` 只能合并一个位置）。
- 考察点：non-overlapping 从左到右合并语义；"先减旧、后加新"的增量一致性；
  倒排索引维护；与全量重算 reference 的 differential testing。
- Python 骨架：

```python
from collections import Counter, defaultdict

def merge_word(word, pair, new_sym, word_counts, pair_counts, pair_to_words):
    """word: 当前 symbol tuple；应用一次 merge 并增量更新三张统计表。"""
    cw = word_counts[word]
    for i in range(len(word) - 1):
        pair_counts[(word[i], word[i + 1])] -= cw
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(new_sym)
            i += 2                    # 重叠出现不会被同时合并
        else:
            out.append(word[i])
            i += 1
    out = tuple(out)
    word_counts[word] -= cw
    word_counts[out] += cw
    for i in range(len(out) - 1):
        p = (out[i], out[i + 1])
        pair_counts[p] += cw
        pair_to_words[p].add(out)
    return out
```

- 验收标准：与"每次 merge 后全量重扫"的 reference 在随机小语料上逐步对齐；
  `("a","a","a")` 合并 `(a,a)` 得 `("aa","a")` 而非 `("aa","aa")`；
  `pair_counts` 全程无负数、无 stale 膨胀；固定 tie-break 下多轮结果可复现。

**代码题 2：bytes/token 与 nats/byte 换算的评测脚本**
- 题目：给定 tokenizer 与一批（文本, 该文档 token NLL 总和），输出分域的
  bytes/token、nats/byte、bits/byte。
- 考察点：跨 tokenizer 可比计量；`nats/byte ≈ (loss/token) ÷ (bytes/token)`
  的换算；分域聚合与均值口径。
- Python 骨架：

```python
import math
from collections import defaultdict

def evaluate(tok, docs, doc_nll, group_of=None):
    key = group_of or (lambda t: "all")
    agg = defaultdict(lambda: [0, 0, 0.0])        # bytes, tokens, total_nll
    for text, nll in zip(docs, doc_nll):
        a = agg[key(text)]
        ids = tok.encode(text)
        a[0] += len(text.encode("utf-8"))
        a[1] += len(ids)
        a[2] += nll
    report = {}
    for g, (nb, nt, nll) in agg.items():
        npb = nll / nb
        report[g] = {
            "bytes/token": nb / nt,
            "nats/byte": npb,
            "bits/byte": npb / math.log(2),
        }
    return report
```

- 验收标准：对 OWT 10K（token loss 3.081）与 32K（4.116）样例，换算后分别约
  0.971 与 0.942 nats/byte，能得出"32K 更优"的反直觉结论；同一 tokenizer 下
  `loss/token × bytes/token ≈ nats/byte` 守恒；空文档、纯 special token 输入
  不崩溃。

**代码题 3：数字切分规则对比器（逐位 vs 固定宽度）**
- 题目：实现两种数字 pre-tokenizer（逐位、3 位固定宽度），对算术题数据统计
  token 数并抽样打印切分结果，供下游准确率 ablation 使用。
- 考察点：pre-tokenization regex 作为可冻结契约；digit 分组对序列长度与
  位值可学习性的影响；ablation 的对照组设计。
- Python 骨架：

```python
import re

def split_digits(text, mode):
    def repl(m):
        s = m.group()
        if mode == "per-digit":
            return " ".join(s)
        return " ".join(re.findall(r"\d{1,3}", s))  # 从高位起固定宽度
    return re.sub(r"\d+", repl, text)
```

- 验收标准：`split_digits("1234", "per-digit")` 得 `"1 2 3 4"`；fixed3 得
  `"123 4"`；两种模式在同一题集上的 token 数统计可对比；小数点、负号、
  千分位逗号的处理作为显式契约记录并测试。

## 13. 结论与本讲小结

Tokenizer 定义了语言模型观察世界的离散单位，也定义了数据量、上下文长度和 loss 的标尺。
Byte-level BPE 用 256-byte 完备基底保证开放词表，再用有序 merges 换取较短序列。
正确实现的关键不只是合并循环，还包括 UTF-8 边界、pre-tokenization、special-token 协议、
确定性与流式处理。评价 tokenizer 必须同时看压缩、资源、领域迁移和 downstream 成本。
频率驱动的 merge 在数字、代码与低资源语言上存在系统性盲区，需要专门的 slice 评估与
契约化规则；而 multiscale/dynamic-patching 架构正在把"序列过长"的代价转移给模型侧，
token-free 并非倒退，而是重新分配 tokenizer 与 architecture 的职责边界。

## 参考文献

<a id="ref-1"></a>[1] C. E. Shannon. “A Mathematical Theory of Communication.”
*Bell System Technical Journal*, 1948. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x

<a id="ref-2"></a>[2] R. Sennrich, B. Haddow, A. Birch. “Neural Machine Translation
of Rare Words with Subword Units.” *ACL*, 2016.
https://aclanthology.org/P16-1162/

<a id="ref-3"></a>[3] A. Radford et al. “Language Models are Unsupervised Multitask
Learners.” OpenAI Technical Report, 2019.
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

<a id="ref-4"></a>[4] T. Kudo, J. Richardson. “SentencePiece: A Simple and Language
Independent Subword Tokenizer and Detokenizer for Neural Text Processing.”
*EMNLP System Demonstrations*, 2018. https://aclanthology.org/D18-2012/

<a id="ref-5"></a>[5] T. Kudo. “Subword Regularization: Improving Neural Network
Translation Models with Multiple Subword Candidates.” *ACL*, 2018.
https://aclanthology.org/P18-1007/

<a id="ref-6"></a>[6] V. Zouhar et al. “A Formal Perspective on Byte-Pair Encoding.”
*Findings of ACL*, 2023. https://doi.org/10.18653/v1/2023.findings-acl.38

<a id="ref-7"></a>[7] I. Provilkov, D. Emelianenko, E. Voita. “BPE-Dropout:
Simple and Effective Subword Regularization.” *ACL*, 2020.
https://aclanthology.org/2020.acl-main.170/

<a id="ref-8"></a>[8] P. Rust, J. Pfeiffer, I. Vulić, S. Ruder, I. Gurevych.
“How Good is Your Tokenizer? On the Monolingual Performance of Multilingual
Language Models.” *ACL-IJCNLP*, 2021. https://doi.org/10.18653/v1/2021.acl-long.243

<a id="ref-9"></a>[9] L. Xue et al. “ByT5: Towards a Token-Free Future with
Pre-trained Byte-to-Byte Models.” *TACL*, 2022.
https://aclanthology.org/2022.tacl-1.17/

<a id="ref-10"></a>[10] J. H. Clark et al. “CANINE: Pre-training an Efficient
Tokenization-Free Encoder for Language Representation.” *TACL*, 2022.
https://aclanthology.org/2022.tacl-1.5/

<a id="ref-11"></a>[11] Y. Tay et al. “Charformer: Fast Character Transformers via
Gradient-based Subword Tokenization.” *ICLR*, 2022.
https://openreview.net/forum?id=JtBRnrlOEFN

<a id="ref-12"></a>[12] L. Yu, D. Simig, C. Flaherty, A. Aghajanyan,
L. Zettlemoyer, et al. “MEGABYTE: Predicting Million-byte Sequences with
Multiscale Transformers.” *NeurIPS*, 2023. [arXiv](https://arxiv.org/abs/2305.07185)

<a id="ref-13"></a>[13] A. Pagnoni, R. Pasunuru, P. Rodriguez, et al.
“Byte Latent Transformer: Patches Scale Better Than Tokens.” arXiv:2412.09871,
2024. [link](https://arxiv.org/abs/2412.09871)

<a id="ref-14"></a>[14] A. K. Singh and D. J. Strouse. “Tokenization Counts:
the Impact of Tokenization on Arithmetic in Frontier LLMs.” arXiv:2402.14903,
2024. [link](https://arxiv.org/abs/2402.14903)

<a id="ref-15"></a>[15] K. Bostrom, G. Durrett. “Byte Pair Encoding is Suboptimal for
Language Model Pretraining.” *Findings of EMNLP*, 2020.
https://aclanthology.org/2020.findings-emnlp.414/

<a id="ref-16"></a>[16] G. Dagan, G. Synnaeve, and B. Rozière. “Getting the
Most Out of Your Tokenizer for Pre-Training and Domain Adaptation.”
arXiv:2402.01035, 2024. [link](https://arxiv.org/abs/2402.01035)

<a id="ref-17"></a>[17] A. Petrov, E. La Malfa, P. H. S. Torr, and A. Bibi.
“Language Model Tokenizers Introduce Unfairness Between Languages.”
arXiv:2305.15425, 2023. [link](https://arxiv.org/abs/2305.15425)

## 延伸阅读与复现材料

- Stanford CS336, [Spring 2026 Lecture 1](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py)
- Sennrich, Haddow, Birch, [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)
- Kudo, Richardson, [SentencePiece](https://aclanthology.org/D18-2012/)
- OpenAI, [tiktoken](https://github.com/openai/tiktoken)
- Unicode Consortium, [The Unicode Standard](https://www.unicode.org/standard/standard.html)
- [Tokenization 主题导航](../experiments/topics/tokenization-and-basics.md)
- [A1 官方资料与作业](../experiments/official/a1-basics.md)
- [Tokenization Counts: the Impact of Tokenization on Arithmetic in Frontier LLMs（arXiv:2402.14903）](https://arxiv.org/abs/2402.14903)（访问日期 2026-10-04）
- [Language Model Tokenizers Introduce Unfairness Between Languages（arXiv:2305.15425）](https://arxiv.org/abs/2305.15425)（访问日期 2026-10-04）
- [Getting the Most Out of Your Tokenizer for Pre-Training and Domain Adaptation（arXiv:2402.01035）](https://arxiv.org/abs/2402.01035)（访问日期 2026-10-04）
- [Byte Latent Transformer: Patches Scale Better Than Tokens（arXiv:2412.09871）](https://arxiv.org/abs/2412.09871)（访问日期 2026-10-04）
