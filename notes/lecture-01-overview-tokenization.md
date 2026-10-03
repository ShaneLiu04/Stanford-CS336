---
title: "Lecture 01 — Overview & Tokenization"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-03-30"
lecturer: "Percy Liang"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py"
  - "../assignments/assignment1-basics/"
---

# Lecture 01 — Overview 与 Tokenization：从数据契约到离散符号系统

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述，非课程提交
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
| GPT-2 pre-tokenization | `assignments/assignment1-basics/cs336_basics/tokenizer.py`：`GPT2_PATTERN` | leading space、Unicode 类别 |
| BPE training | 同文件：`train_bpe` | tie-break、special boundary、局部计数更新 |
| 流式/并行切块 | 同文件：`_chunk_boundaries`、`_count_pretokens` | 不切断 delimiter 和 UTF-8 |
| BPE encoding | 同文件：`Tokenizer._encode_bytes`、`encode` | merge rank，不按新频率 |
| decode / iterable | 同文件：`decode`、`encode_iterable` | bytes 拼接、惰性输出 |
| 接口契约 | `assignments/assignment1-basics/tests/adapters.py`：`run_train_bpe`、`get_tokenizer` | shape/type/返回值 |
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
| 算术/数字任务异常差 | digit merge 方式 / 数字 tokenization 策略 | BPE 频率目标把数字切成任意碎片 [[14]](#ref-14) |
| 内存持续增长 | pair_to_words 清理、heap stale entries | 倒排索引未删除旧引用 |
| 训练 loss 突升或下游 loss 异常 | tokenizer 版本与 checkpoint 配套性 | vocab/merges 未随权重一起版本化 |
| 吞吐很高但模型更慢 | \(V\)、\(T\)、LM head FLOPs | 只优化 tokenizer 本身 |
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

### 12.1 Construct validity

- `bytes/token` 衡量序列压缩，不等价于语义质量、形态合理性或模型 accuracy；
- `tokens/s` 同时受 token 数和实现速度影响，不能单独代表 bytes/s；
- token-level perplexity 的单位随 tokenizer 变化，跨 tokenizer 应转为 nats/byte，
  但 nats/byte 仍不等价于 downstream task quality。

### 12.2 Internal validity

- 不同 tokenizer 若训练语料、normalization、special tokens、模型参数量或 byte budget
  不一致，无法把结果归因于 segmentation；
- vocabulary 增大会同时改变序列长度和 embedding/head 参数，必须二者同时 accounting；
- 单 seed training 差异可能来自初始化与 data order，而非 tokenizer。

### 12.3 External validity

- TinyStories 上的最佳 vocabulary 未必迁移到 Web、code、数学或 multilingual data；
- 200-document sample 无法代表完整 corpus long tail；
- Python reference implementation 的吞吐不能代表 Rust/C++ production tokenizer；
- 离线批处理结果不能直接推断 online serving 的 p99 latency。

### 12.4 Fairness、隐私与安全

高资源语言主导的 frequency objective 可能让低资源语言产生更高 fertility，从而获得更短
effective context、更多训练/推理成本和更高服务费用 [[8]](#ref-8)。Petrov 等人进一步
量化了这种不公平：相同语义内容在不同语言间的 token 数差异可达数倍，直接转化为上下文
容量、延迟与 API 计费的系统性差异 [[17]](#ref-17)。Tokenizer 还可能把
PII、URL、API key 或 offensive strings 记成单 token，增加 memorization 与 probing 风险。
因此 artifact audit 应包含 per-language cost、敏感 pattern tokenization、异常 control
characters 与 denial-of-service 长输入，而不仅是英文 compression。

## 13. 面试备考（Interview Prep）

> Tokenization 是 LLM 面试的高频开篇题：概念门槛低、却能一路追问到深度。面试官通常从
> 「BPE 是什么」切入，追到「为什么 loss 跨 tokenizer 不可比」「训练复杂度怎么优化」
> 「数字与多语言有什么系统性坑」。下面按「一页速览 → 高频题 → 手撕 → 追问」四层组织，
> 每道题用统一框架回答：**定义 → 为什么/原理 → 公式/复杂度 → 工程落地 → 边界/反例**。

### 13.1 一页速览卡（面试前 1 分钟）

**核心主张**：tokenizer 是「模型—数据」之间的系统契约，决定序列长度 \(T\)、词表参数
\(Vd\)、训练 FLOPs 与 loss 的计量单位；它不是无关紧要的预处理，而应像 checkpoint 一样被版本化。

**必背数字与公式**

- GPT-2 词表 **50,257**（256 bytes + 1 EOT + 50,000 merges）；GPT-4 级约 100k。
- 最大 merge 数 \(M = V - 256 - S\)（\(S\) 为 special tokens）。
- `bytes/token = N_byte / N_tok`；`nats/byte ≈ L_tok / (bytes/token)`；`bits/byte = nats/byte / ln2`。
- 朴素 BPE 训练 \(O(MN)\)，增量更新近似线性；单个 pre-token 编码最坏 \(O(L^2)\)。

**三句话答高频**

1. byte-level BPE：256 bytes 起底 → 任意输入可编码、无 OOV；有序 merge 换较短序列。
2. 训练 ≠ 编码：训练学「有序 merge rank」，编码按 rank 贪心合并，绝不按新文本频率重选。
3. 跨 tokenizer 不可比：token 是人为单位，比较模型必须换算 `nats/byte` / `bits/byte`。

### 13.2 高频面试题与答题框架

**Q1：解释 BPE 的训练过程与编码过程，为什么是两回事？**

- **定义**：训练在语料上反复合并最高频相邻 pair，产出「有序 merge 表」；编码对新文本按这张表的顺序贪心合并。
- **为什么**：训练基于 corpus 统计，编码基于已冻结的 rank。若编码时按当前句子频率重选 pair，等于每次重训，词表不稳定、checkpoint 无法复用。
- **公式**：pair 计数 \(C(p)=\sum_w c(w)\,\#\{i:(w_i,w_{i+1})=p\}\)，每轮取 \(\arg\max C(p)\)，最多 \(V-256-S\) 次。
- **工程**：`pair_counts` + `pair_to_words` 倒排索引 + max-heap，只局部更新受影响 pre-token，避免每轮重扫全语料。
- **边界**：greedy merge 是启发式，非压缩率全局最优（Zouhar et al. 2023）；overlap pair 一次 merge 只合并不重叠出现。

**Q2：为什么现代 LLM 普遍用 byte-level BPE？解决什么、代价是什么？**

- **解决**：word 词表的 OOV、code-point 词表的稀疏；任意字节串可编码、可逆且确定。
- **为什么**：256 bytes 是完备基底，再学常见 byte 片段，在「开放词表」与「短序列」之间折中。
- **代价**：低资源语言 fertility 高、bytes/token 偏高（Petrov et al. 2023）；数字被切成不规则片段伤算术（Singh & Strouse 2024）。

**Q3：BPE、WordPiece、Unigram（SentencePiece）三者区别？**

- **BPE**：合并最高频 pair，无概率模型，deterministic，工程最成熟。
- **WordPiece**：每次选「最大化似然增益」的 pair（约等于频数 / 两子词频数积），BERT 生态，需处理 `##` 前缀。
- **Unigram LM**：先过大量候选再按损失删低贡献 token，有概率模型、支持 subword regularization（采样多种切分）。
- **SentencePiece** 是「raw-text framework」：统一 normalization 与 whitespace 符号，BPE 与 Unigram 都可运行其上，不依赖外部分词。

**Q4：词表大小如何权衡？**

- **大词表**：序列短 → attention \(O(T^2)\) 与训练步数省；但 embedding/head \(O(Vd)\) 参数多、低频 token 表示差、可能记领域噪声。
- **小词表**：参数省，但序列长、算术/长尾表示差。
- **结论**：最优词表随模型规模与数据规模变化，需 sweep；Dagan et al. 2024 指出 tokenizer 训练语料约数十 GB 后进入平台期，领域适配可增量扩词表 + embedding resize，不必从零重训。

**Q5：为什么直接比较两个模型的 perplexity 可能误导？**

- perplexity = \(\exp(L_{\text{tok}})\)，而 \(L_{\text{tok}}\) 的单位是 token——token 是人为构造单位。
- 相同文本被切成不同 token 数，「loss 的分母」不同，标尺不同。
- **正确做法**：换算 `nats/byte = L_tok / (bytes/token)`。本仓库实例：OWT 10K 与 32K tokenizer 的 token loss 分别为 3.081 与 4.116，只看 token loss 会误判 10K 更好；归一化后为 0.971 vs 0.942 nats/byte，32K 实际更优。

**Q6：pre-tokenization 的作用？为什么 special token 必须是硬边界？**

- **pre-tokenization**：先用 regex 把文本切成大致文字/数字/标点片段，约束 BPE 只在片段内 merge，防止跨词/跨文档合并。
- GPT-2 的 `GPT2_PATTERN` 保留 leading space，所以 token 常是 `" the"` 而非 `"the"`——token boundary ≠ word boundary。
- **special token**（如 `<|endoftext|>`）有协议语义：必须整体作为一个 token、BPE 不得跨边界、多个重叠时 longest match。

**Q7：数字 tokenization 为什么是 BPE 的系统性盲区？**

- BPE 由 corpus frequency 驱动，数字是长尾 + 进位结构，`1234` 可能被切成 `12`+`34` 或逐位，模型难学位值/进位。
- Singh & Strouse 2024：逐位 vs 固定宽度分组显著改变加减乘准确率，部分算术失误可由更换 number tokenization 直接修复。
- **工程**：数学/财务语料应强制逐位或固定分组 pre-tokenization，而不是接受无约束 merge。

**Q8：tokenizer 如何造成多语言不公平？**

- 高资源语言主导 frequency objective，低资源语言 fertility 更高 → 相同语义内容的 token 数可达数倍差异。
- **后果**：更短 effective context、更高训练/推理成本与 API 计费（Petrov et al. 2023）。
- **工程**：报告 per-language token cost / fertility，而不只看英文压缩率。

**Q9：GPT-2 的 pre-tokenization regex 为什么保留 leading space？token boundary 为什么不等于 word boundary？**

- **为什么**：`GPT2_PATTERN` 把空格并入前面的 word（如 `" the"` 而非 `"the"`），使空格处理在 BPE 之外统一、避免词尾空格歧义，也提高英文压缩率。
- **边界含义**：token boundary 由 regex 决定，不是词边界；一个 token 可能覆盖「空格 + 英文词」，一个中文字符可能跨多个 token。
- **追问点**：pre-tokenization 是 multilingual 行为的第一决定因素——英文优化的 regex 对无空格语言（中日韩）和代码并不理想。

**Q10：SentencePiece 与 tiktoken 的定位差异？为什么现代 decoder 常直接处理 raw bytes？**

- **SentencePiece**：raw-text framework，统一 normalization + whitespace 符号（`▁`），不依赖外部分词，可跑 BPE/Unigram；训练和推理走同一 normalize 路径。
- **tiktoken**：OpenAI 的 byte-level BPE（GPT-2/GPT-4 风格），直接以 UTF-8 bytes 为基底 + regex pre-tokenization，可逆、无 OOV，但无概率模型/采样。
- **为什么 raw bytes**：byte 基底保证任意输入可编码、可逆、确定，配合正则约束即可获得「开放词表 + 短序列」，避免 code-point 词表的稀疏。

**Q11：领域适配时如何调整 tokenizer？为什么通常不从头重训？**

- **为什么**：tokenizer 一旦冻结，embedding/LM head 的每一行语义就固定了；换词表等于换「观察世界的单位」，旧 checkpoint 无法复用。
- **做法**（Dagan et al. 2024）：对已有 tokenizer 继续训练、小规模扩展领域词表 + embedding resize + 少量 continued training，以极低成本改善领域 loss。
- **前提**：任何词表扩展都要与 embedding resize、special-token 审计、旧 checkpoint 迁移策略一起设计，不能静默替换。

**Q12：tokenizer 会引入哪些安全与隐私风险？**

- **memorization/probing**：PII、URL、API key 或 offensive 串被记成单 token，更易被模型记忆和攻击者 probing。
- **不公平**：低资源语言 fertility 高 → 上下文容量、延迟与 API 计费的系统性差异（Petrov et al. 2023）。
- **DoS/审计**：异常 control characters、超长输入、恶意 special-token 注入需在 artifact audit 中显式覆盖，而不只看英文压缩率。

### 13.3 手撕代码要点（BPE）

面试手撕 BPE 时，先把「训练」和「编码」分开写（面试官最在意你会不会混淆两者），
给出可运行的 naive 版本，再主动补一句优化思路。下面是完整、正确的参考实现：

```python
from collections import Counter

# ---------- 训练：反复合并最高频 pair ----------
def train_bpe(pretokens, vocab_size=1000):
    """pretokens: list[list[int]]，每个元素是一段 byte 序列（已 pre-tokenize）"""
    merges = []                                      # 有序 merge 表 [(a,b), ...]
    vocab = {i: bytes([i]) for i in range(256)}      # 初始 256 个 byte tokens
    tie_break = lambda kv: (kv[1], kv[0])            # 同频时取字典序更大的 pair

    while len(vocab) < vocab_size:
        pairs = Counter()
        for w in pretokens:
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), _ = max(pairs.items(), key=tie_break)  # 最高频 pair
        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append((a, b))

        # non-overlapping 合并：重叠 pair 一次只合并一个
        pretokens = [merge_once(w, a, b, new_id) for w in pretokens]
    return merges, vocab

def merge_once(w, a, b, new_id):
    out, i = [], 0
    while i < len(w):
        if i + 1 < len(w) and w[i] == a and w[i+1] == b:
            out.append(new_id); i += 2              # 跳过重叠位置
        else:
            out.append(w[i]); i += 1
    return out

# ---------- 编码：按 merge rank 贪心合并，绝不重新统计频率 ----------
def encode(word, merges):
    """word: list[int] byte 序列；merges 的顺序就是 rank"""
    ranks = {pair: rank for rank, pair in enumerate(merges)}
    word = list(word)
    while True:
        best_rank, best_i = float("inf"), -1
        for i in range(len(word) - 1):
            r = ranks.get((word[i], word[i+1]), float("inf"))
            if r < best_rank:
                best_rank, best_i = r, i
        if best_i == -1:
            return word
        word[best_i:best_i+2] = [256 + best_rank]    # 新 id = 256 + rank
```

**优化思路（面试官追问「训练怎么加速」时）**：朴素训练每轮全量扫一遍是 \(O(MN)\)；用
`pair_counts` + `pair_to_words` 倒排索引 + max-heap 做增量更新，只在受影响的 pre-token
上重算相邻 pair，接近线性；编码用 linked list + priority queue 可从 \(O(L^2)\) 降到
\(O(L\log L)\)。

**三个必踩坑**

1. **overlap**：`a a a` 中 `(a,a)` 出现两次，一次 merge 只能得 `aa a`，不能同时合并两个重叠位。
2. **tie-break**：同频 pair 的选择顺序必须固定（如字典序最大优先），否则 vocab 不可复现。
3. **determinism**：文件顺序、regex 版本、special token 顺序、多进程 reduce 顺序都要固定。

### 13.4 高频追问与陷阱

| 追问 | 正确方向 |
| --- | --- |
| greedy BPE 是最优压缩吗？ | 不是，是启发式；已有 combinatorial 分析证明非全局最优 |
| 编码时能按新文本频率选 pair 吗？ | 不能，等于重训，破坏词表稳定与 checkpoint 兼容 |
| 任意分块独立 encode 再拼接等于整串吗？ | 不等价，pre-token 可能跨 chunk，需 carry buffer 或按 record 边界切 |
| 大词表一定更好吗？ | 否，缩短序列但增大 embedding/head 参数，还可能记领域噪声 |
| bytes/token 能代表质量吗？ | 不能，只代表序列压缩，不等价于语义/形态/下游 accuracy |
| token-free 会取代 BPE 吗？ | 未必，BPE 仍是 checkpoint/serving 的事实标准；MEGABYTE/BLT 把长序列代价转给架构 |

### 13.5 模拟追问链（还原面试官的层层深入）

面试官通常从「BPE 是什么」一路追到工程细节。下面是一段典型追问链，注意每一层都在往
「复杂度 / 边界 / 反例」递进，答满前四层是优，主动补边界是加分：

> **面试官**：介绍一下 BPE 分词。
> **你**：byte-level BPE 从 256 bytes 起底，反复合并最高频相邻 pair，得到「有序 merge 表」；训练学 merge 表，编码按表的顺序贪心合并，两者是不同过程。
>
> **面试官**：训练时每轮怎么选要合并的 pair？
> **你**：统计所有相邻 pair 的频数 \(C(p)=\sum_w c(w)\,\#\{(w_i,w_{i+1})=p\}\)，取 \(\arg\max\)；最多 \(V-256-S\) 次。
>
> **面试官**：复杂度多少？能优化吗？
> **你**：朴素每轮全量扫描是 \(O(MN)\)；用 pair 计数 + 倒排索引 + 堆做增量更新，只在受影响的 pre-token 上重算，接近线性。
>
> **面试官**：序列 `aaa` 里 pair `aa` 出现了两次，一次 merge 能都合并吗？
> **你**：不能，合并是 non-overlapping 的，一次只能得 `aa a`；计数定义和替换语义必须一致。
>
> **面试官**：两个词表大小相同的模型，为什么 perplexity 不能直接比？
> **你**：perplexity 是 token-level 指数 loss，而 token 是人为构造单位；相同文本被切成的 token 数不同，「分母」不同。要换算成 `nats/byte = L_tok/(bytes/token)` 再比。
>
> **面试官**：那是不是词表越大越好？
> **你**：不是。大词表缩短序列、省 attention 和步数，但增大 embedding/head 的 \(Vd\) 参数、低频 token 表示差、还可能记领域噪声，最优值要 sweep。
>
> **面试官**：数字 tokenization 有什么坑？
> **你**：BPE 由频率驱动，数字是长尾 + 进位结构，`1234` 可能被切成 `12`+`34` 或逐位，模型难学位值/进位；数学/财务语料应强制逐位或固定分组 pre-tokenization。

**分层自测**：能答出「训练/编码区别 + 复杂度」为**初级**；能补「增量更新数据结构 +
nats/byte 换算」为**中级**；能主动讲「数字盲区、多语言 fertility、token-free 边界」为**高级**。

## 14. 结论与本讲小结

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
