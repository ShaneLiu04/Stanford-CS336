# Data Curation：从 Common Crawl 到可训练语料

Data curation 不是把若干 filters 串起来，而是优化一个多目标系统：在质量、覆盖、多样性、安全、隐私、去重和 compute cost 之间取舍，并保留足够 provenance 让结果可审计。推荐按 extraction → measurement → filtering → deduplication → mixing → evaluation 的顺序学习。

## 推荐阅读顺序

### 1. 认识数据来源与 extraction

- **官方 lecture**：[Lecture 13 — Data sources and datasets](https://github.com/stanford-cs336/lectures/blob/main/lecture_13.py)。
- **工具文档**：[Common Crawl data](https://commoncrawl.org/get-started)、[WARC specification](https://iipc.github.io/warc-specifications/) 与 [warcio](https://warcio.readthedocs.io/)。
- **解决的问题**：网页 crawl 包含 HTTP metadata、HTML、导航模板、脚本、编码异常和重复 boilerplate；训练需要稳定提取正文，并为每个 document 保存 source/provenance。
- **关键结论**：WARC/WET 是容器和提取层级，不等于高质量文档。pipeline 应明确 document boundary、字符编码、失败处理与内容 hash；抽样检查 extraction error 往往比立即调 filter threshold 更重要。
- **对应作业**：[Assignment 4 — Data](https://github.com/stanford-cs336/assignment4-data) 的 WET streaming、HTML text extraction 与 Common Crawl processing；本仓库快照位于 `assignments/spring2026/assignment4-data/`。
- **局限**：WET 已丢失部分 HTML structure，便于处理但限制正文恢复；不同语言、网页类型与时期需要不同 extraction 假设。

### 2. 先测量语言和基本质量，再决定阈值

- **论文**：[CCNet](https://arxiv.org/abs/1911.00359) 展示 language identification、deduplication 和 language-model perplexity filtering；[Gopher](https://arxiv.org/abs/2112.11446) 附录给出大规模文本 quality heuristics 的代表性设计。
- **工具文档**：[fastText language identification](https://fasttext.cc/docs/en/language-identification.html)。
- **解决的问题**：过滤错误语言、极短/异常文本、模板列表、符号噪声和低信息页面，同时尽量不伤害专业、小语种或非标准写作。
- **关键结论**：
  - language ID 应保留置信度，threshold 是 precision-recall 选择，不是事实标签。
  - length、平均词长、symbol ratio、stop-word fraction、重复 n-gram 等 heuristic 应在 document 级组合，并记录每条 rejection reason。
  - funnel 中每级通过率与交集比单个总保留率更有诊断价值；filter order 还会影响成本。
- **对应作业**：A4 的 `is_english`、Gopher quality rules 与 filter ablation。
- **局限**：fastText 与 heuristic 都继承训练数据偏差；代码、数学、方言、混合语言和 OCR 文本常被误杀。阈值不能只在合成样本上调。

### 3. 学习 model-based quality filtering

- **论文**：[RefinedWeb](https://arxiv.org/abs/2306.01116) 讨论大规模 web filtering 与 deduplication；[FineWeb](https://arxiv.org/abs/2406.17557) 通过可控 ablation 和 downstream evaluation 研究过滤 recipe。
- **解决的问题**：手写规则容易覆盖表面噪声，却难判断教育价值、连贯性和领域质量。classifier 可从高质量/低质量示例学习更柔性的 decision boundary。
- **关键结论**：quality score 不是绝对真值。训练 classifier 时要审计 positive/negative source，分离 train/evaluation，并画 threshold 下 retained tokens 与 downstream proxy 的 Pareto curve。先做轻量规则可减少昂贵 model inference。
- **对应作业**：A4 的 quality classifier、score threshold sweep，以及用固定 LM recipe 对 filtered corpus 做比较。
- **局限**：以 Wikipedia、books 等为正例会把“像参考语料”误当质量，造成风格 homogenization 和领域排斥；classifier drift 与 adversarial SEO 也会降低泛化。

### 4. 分开处理安全、隐私与许可

- **工具文档**：[Microsoft Presidio](https://microsoft.github.io/presidio/) 提供 PII detection/anonymization 的工程参考；[Common Crawl terms](https://commoncrawl.org/terms-of-use) 说明原始数据访问条件。
- **解决的问题**：识别 email、phone、IP 等 PII，以及 NSFW/toxicity 风险；决定删除、redact、隔离还是降低采样权重。
- **关键结论**：PII/safety detector 要分别报告 precision 与 recall；redaction 需避免破坏文档边界或引入可逆线索。数据可公开抓取不等于可任意训练或再分发，license、robots、jurisdiction 与用途需要独立审查。
- **对应作业**：A4 的 PII masking、NSFW/toxicity filtering 与 rejection statistics。
- **局限**：正则表达式适合结构化 PII，但漏掉上下文身份信息；toxicity classifier 在引用、少数群体术语与多语言上可能误判。课程实现不是完整法律或合规方案。

### 5. 掌握 exact 与 near-duplicate removal

- **官方 lecture**：[Lecture 14 — Filtering, deduplication, mixing, synthetic data](https://github.com/stanford-cs336/lectures/blob/main/lecture_14.py)。
- **论文**：[On the Dangers of Stochastic Parrots](https://dl.acm.org/doi/10.1145/3442188.3445922) 提供数据规模、治理与社会风险背景；[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) 直接研究重复数据对语言模型的影响。
- **工具文档**：[datasketch MinHash](https://ekzhu.com/datasketch/minhash.html)。
- **解决的问题**：模板页、镜像站和复制文章会浪费 token budget、放大来源权重，并增加 benchmark contamination 与 memorization 风险。
- **关键公式/结论**：
  - exact line/document hash 能廉价去除完全相同内容，但无法处理轻微编辑。
  - 将文档表示为 shingles 集合，Jaccard similarity 为
    \[
    J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
    \]
    MinHash 满足单个 hash 下 \(\Pr[h(A)=h(B)]=J(A,B)\)，多个 permutations 估计相似度；LSH 以 bands/rows 在召回率与候选数之间折中。
  - 去重单位与保留策略很重要：document-level、paragraph-level、line-level 会产生不同数据分布；cluster 中保留哪一份也会引入 source bias。
- **对应作业**：A4 的 exact-line dedup、MinHash signatures、LSH candidate generation 与 cluster removal。
- **局限**：短文档的 Jaccard 方差大；常见短语可能导致 false positive；跨语言翻译、语义改写和局部拼接不容易被 lexical MinHash 发现。

### 6. 用训练结果验证 data recipe

- 固定 tokenizer、model、training tokens、optimizer 与 evaluation，比较不同 filtered datasets；否则无法把效果归因于数据。
- 除 validation loss 外，报告 retained documents/tokens、语言/域分布、重复率、PII/safety audit 和 contamination checks。
- mixing weights 决定各 source 被看到的期望次数；低资源 domain 可重采样，但过度重复会增加 memorization。
- **对应作业**：A4 最终用统一 training code 在 curated corpus 上训练并比较。
- **局限**：小 proxy model 对数据质量的排序未必在大模型上保持；单一 benchmark 也可能奖励 contamination 或狭窄风格。

## 建议实验闭环

1. 对随机文档做人工标注小集，先测 extraction、语言与每类 filter 的错误。
2. 输出逐级 funnel 和 rejection-reason overlap，不只看最终保留率。
3. 在 threshold sweep 上同时画 retained tokens、人工 precision/recall 与训练 proxy。
4. 对 dedup cluster 抽样检查 false positive/negative，并固定 deterministic survivor rule。
5. 保存 manifest、代码版本、原始对象标识和聚合统计；不要把含 PII/受限内容的原文提交到仓库。

## 阅读边界

不存在与语境无关的“高质量文本”。大规模清洗容易把主流、规范、英文写作风格固化成质量定义。技术指标必须与 provenance、representativeness、privacy、license 和受影响群体审查同时进行；课程 leaderboard 不能代替数据治理。
