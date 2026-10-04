---
title: "Lecture 13 — Data Sources & Datasets"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-05-11"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_13.py"
  - "../assignments/spring2026/assignment4-data/"
---

# Lecture 13 — 数据来源与数据集：先定义数据，再谈规模

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
- 适用对象：自学者、数据工程师与 LLM 研究者

## 摘要

数据是语言模型的燃料，但“数据集”从来不是一个中性的文件集合：训练分布由来源发现、抓取
时机、抽取器、过滤器和混合权重共同构成的采样过程决定。本文从采样过程视角出发，系统梳理
网页、书籍、代码、论文、对话与合成数据等来源的收益与系统性偏差；深入 Common Crawl 的
WARC/WAT/WET 三层对象与 snapshot 生态，说明为什么 `WET record ≠ 网页正文 ≠ 训练文档`。
在治理层面，本文讨论 license、provenance、PII 与删除请求的可追踪数据契约；在统计层面，
推导 token 预算下的期望遍历次数 \(E_s\)、数据受限场景的边际收益递减与记忆风险；在工程
层面，给出 manifest schema、逐级 funnel 统计、A4 实现映射与诊断速查表。目标是让读者把
“多大规模的数据”升级为“什么分布、什么许可、什么暴露次数的数据”，并以可复现实验而非
肉眼直觉验证数据价值。

**关键词：** Data Sources；Common Crawl；WARC/WET；Provenance；Data Governance；
Data-Constrained Scaling；Memorization；Synthetic Data；Model Collapse

## 本文贡献

1. 把数据集形式化为采样过程 \(p_{\text{train}}(x)=\sum_s w_s\,p_s(x\mid\cdot)\)，并区分混合权重的三种口径与期望暴露 \(E_s\)；
2. 系统比较七类来源的收益与“隐藏价格”，覆盖 license、毒性、PII 与合成数据的 model collapse 风险；
3. 给出 WARC/WAT/WET 的取舍分析与可删除、可审计、可重放的 manifest 契约；
4. 推导 \(E_s=Tw_s/N_s\) 与 usable-token funnel，并连接 data-constrained scaling 与 memorization 文献的结论边界；
5. 提供从登记来源到下游验证的完整工作流、诊断速查表与论文级数据报告规范。

## 学习目标

1. 区分网页、书籍、代码、学术论文、对话和合成数据的优势与系统性偏差；
2. 理解 crawl、snapshot、record、document、token 五个层级，避免把 Common Crawl 当成“干净文本集”；
3. 能从 WARC/WET 建立带 provenance、许可和审计字段的数据清单；
4. 用覆盖、质量、重复、安全、隐私和成本共同评价数据源，而不是只报 token 数；
5. 用 \(E_s\)、data-constrained scaling 曲线与 memorization 证据判断“多少数据够用、多少遍越界”。

## 先修知识

- Lecture 01：tokenizer 与 bytes-per-token（token 预算的计量单位）。
- Lecture 09：数据预算与 scaling 的耦合（数据也是被优化的资源）。
- Lecture 12：contamination 与评估效度（数据决策的下游后果）。

## 相关工作与数据生态

早期大规模语料确立了“清洗网页 + 多源拼接”的范式。C4（Colossal Clean Crawled Corpus）
通过启发式规则从 Common Crawl 得到约 156B token 的语料 [[1]](#ref-1)，Dodge 等对其做的
文档化审计揭示了 blocklist、机器翻译文本与社会偏见等隐藏结构，成为“数据集必须被审计”的
标志性工作 [[2]](#ref-2)；The Pile 则把网页、书籍、代码、论文、对话等 22 个来源拼接为
800GB 的多源语料 [[3]](#ref-3)。近年开源 curation 走向“透明管线 + 可复现工具”：
Dolma 发布 3T token 语料与完整工具链，使 ablation 可以被社区复用 [[4]](#ref-4)；
RefinedWeb 证明仅凭严格过滤的网页数据即可在等预算下超越 curated 混合 [[5]](#ref-5)；
FineWeb 进一步用小模型驱动的 ablation 在 15T token 规模上系统比较抽取与过滤决策，
并给出可下载的中间产物 [[6]](#ref-6)。

数据治理方面，Data Provenance Initiative 对数千个常用数据集的 license 与 attribution
做了大规模审计，发现大量许可标注缺失或与原始条款不符 [[7]](#ref-7)。统计方面，
Muennighoff 等的 data-constrained scaling 研究表明：固定数据下重复训练的收益按幂律衰减，
约 4 个 epoch 后新增 epoch 的边际收益趋近于零，且重复数据可与唯一数据分别建模
[[8]](#ref-8)；Carlini 等系统量化了 memorization 随模型规模、样本重复次数的增长，
并展示大模型会在少量重复样本上逐字复述 [[9]](#ref-9)。合成数据方面，Phi 系列展示
高质量“教科书式”数据可以放大训练效率 [[10]](#ref-10)，而 Shumailov 等的理论与实验
分析警告：在递归生成的数据上反复训练会导致分布尾部丢失与 model collapse
[[11]](#ref-11)。

这些工作共同说明：来源选择不是“越多越好”，而是分布、许可、暴露次数与下游验证的联合决策。

## 1. 数据集是采样过程，不是文件集合

训练分布由“哪些来源被发现、何时抓取、怎样抽取、如何过滤、怎样混合”共同决定。可写成

\[
p_{\text{train}}(x)=\sum_{s=1}^{S}w_s\,p_s(x\mid
\text{crawl},\text{extract},\text{filter}),
\qquad \sum_s w_s=1.
\]

\(s\) 是来源，\(w_s\) 是混合权重。权重按文档数、字节数或 token 数计算会产生不同分布；
真正影响优化的是训练时被采到的 token 概率。

这个视角的直接推论是：**同一个 URL 集合在不同管线下的 \(p_s\) 完全不同**。C4、
RefinedWeb 与 FineWeb 都源自 Common Crawl，但抽取器、语言过滤与质量阈值差异使其
最终分布大相径庭 [[1]](#ref-1)[[5]](#ref-5)[[6]](#ref-6)。比较“网页数据”的研究结论时，
必须先比较其管线，否则是在比较不同的采样过程。

### 1.1 抓取层的时间偏差

Common Crawl 是周期性 snapshot 的集合，不是“当前网页”。任何 snapshot 都存在：

- **时间快照偏差**：训练数据永远滞后于部署分布；新闻、价格、API 文档等时效内容尤甚；
- **发现偏差**：爬虫沿链接游走，孤立页面、需要 JS 渲染或登录的内容系统性缺失；
- **robots 与封锁**：遵守 robots.txt 意味着部分高价值站点整体缺席；
- **snapshot 间不稳定**：同一 URL 在不同 snapshot 中可能内容改变、消失或被重定向。

因此 manifest 必须记录 snapshot/crawl ID；“来自 Common Crawl”不是充分描述。

## 2. 常见来源及其“隐藏价格”

- **开放网页**：覆盖广、更新快；代价是模板、SEO、镜像、来源集中、许可不清和 PII。
  RefinedWeb 与 FineWeb 证明其经过严格过滤后足以支撑强基线 [[5]](#ref-5)[[6]](#ref-6)。
- **书籍**：长程连贯；代价是版权、年代与出版选择偏差——书籍语料天然偏向能被出版的声音。
- **Wikipedia/百科**：结构和事实密度高；风格单一，用作“高质量正例”会把百科腔误当普适
  质量。Dodge 等对 C4 的审计显示，类似的高质量锚点会连带引入 blocklist 与社会偏见
  [[2]](#ref-2)。
- **代码**：形式约束强、可执行验证；许可证、密钥和自动生成代码需单独处理。
- **论文**：技术密度高；公式抽取、PDF OCR 和订阅许可复杂。
- **论坛/对话**：贴近真实问答；毒性、身份信息、上下文缺失和用户同意问题更突出。
- **人工标注/合成数据**：目标明确；昂贵或继承 teacher 偏差。Phi 系列显示精心设计的
  “教科书式”合成数据能显著提高 token 效率 [[10]](#ref-10)；但 Shumailov 等的分析
  表明，若后续模型在含大量生成内容的语料上递归训练，分布尾部会逐步丢失（model
  collapse）[[11]](#ref-11)。合成数据的正确用法是**补充**稀缺分布，而非无节制稀释
  真实数据。

一个务实的评估框架：对每个来源分别问覆盖率（多少目标任务域）、密度（单位 token 的
信息量）、风险（法律/隐私/毒性）、成本（获取与过滤）、可验证性（能否证明它带来了收益）。

## 3. Common Crawl 的三层对象

- **WARC**：保存请求/响应、header、URL 和原始 payload，最适合重新抽取和审计。
- **WAT**：解析后的 metadata、links 等结构信息。
- **WET**：预抽取纯文本，吞吐友好，但 DOM、链接上下文和部分 provenance 已丢失。

因此 `WET record ≠ 网页正文 ≠ 训练文档`。同一页面可能被抓取多次；一个 record 也可能
只剩导航、错误页或乱码。FineWeb 的核心经验之一是：抽取器的选择（trafilatura 等）对
最终质量的影响与后续过滤同量级，且必须用下游信号而非肉眼评估 [[6]](#ref-6)。

### 3.1 规模与结构统计

做任何来源决策前先建立基线统计：语言分布（fastText/CLD3 只作估计）、域名集中度
（头部域名常贡献不成比例的 token）、MIME 分布、每 record 字符数直方图、时间戳分布。
Dolma 与 FineWeb 的做法值得效仿：把每级过滤的保留率、拒绝原因与样本抽查同时发布，
使他人可以复核"漏斗"而非只看结果 [[4]](#ref-4)[[6]](#ref-6)。

## 4. Provenance 与数据治理：数据结构的一部分

最小 manifest 建议包含：

```text
document_id, source_uri, crawl_id, warc_path, record_offset,
retrieved_at, mime_type, extractor_version, language_score,
content_hash, license_evidence, filter_decisions, parent_document_id
```

不要把含 PII 的原文或访问凭据写入 manifest。删除请求应能由 `document_id → shard →
tokenized artifact` 追踪，而不是只在原始文本层“删除”。

### 4.1 许可与用途：四个不同的问题

“公开可访问”不等于“允许训练”，也不等于“允许再分发”，更不等于“允许商用”。license
audit 工作显示，主流数据集的许可标注错误率足以影响合规判断 [[7]](#ref-7)。实践上：

- 对每个来源保留**许可证据**（条款 URL、抓取时间、存档快照），而不是结论；
- 区分训练用途、再分发、衍生数据集三种授权需求；
- 记录 opt-out（robots.txt、noai/noimageai meta tag、DMCA、行业协议）的遵守情况；
- 对无法确定许可的来源，单独标记并在报告敏感性分析。

### 4.2 PII、删除与“被遗忘权”

PII 处理的最小原则：最小化采集、假名化统计、审查不复制原文。删除请求的工程挑战在于
传播：删除一个文档必须影响所有下游 shard、tokenized artifact、索引，以及“是否重训/
继续训练”的策略决定。这要求 document_id 在所有衍生层保持可追踪——这正是 manifest
作为一等数据结构存在的原因。

## 5. Token 预算、重复暴露与记忆

### 5.1 期望遍历次数

若训练总预算为 \(T\) tokens，来源 \(s\) 含 \(N_s\) 个 token，采样权重为 \(w_s\)，则该来源
token 的平均期望遍历次数约为

\[
E_s=\frac{T w_s}{N_s}.
\]

小数据源即使权重不大也可能被反复看到，增加记忆风险。只报告 \(w_s\) 而不报告 \(E_s\)
不足以判断 mixing。

### 5.2 数据受限时的边际收益

Muennighoff 等的 data-constrained scaling law 将“唯一 token 数 \(D_u\)”与“重复次数
\(R\)”解耦：损失可近似写成 \(L(D_u, R)\) 的联合幂律，重复带来的收益随 \(R\) 衰减，
约 4 个 epoch 后趋近饱和，相当于把有效数据量约束在 \(4D_u\) 附近 [[8]](#ref-8)。
两个工程推论：

- 报告“训练了 \(T\) tokens”必须同时报告唯一 token 数与最大 \(E_s\)；
- “再加一个 epoch”是决策，不是默认——其收益可以由 scaling curve 预估。

### 5.3 记忆与提取风险

Carlini 等显示，逐字记忆的概率随模型规模与样本重复次数显著增长：低重复（1 次）样本
几乎不被逐字复述，而高重复样本的提取率可达可观比例 [[9]](#ref-9)。结合 \(E_s\)：
含 PII 或隐私敏感内容的来源即使权重很小，只要 \(E_s\) 高，就构成实质性风险。缓解手段
包括 dedup（Lecture 14）、PII scrubbing 与 canary 插入监测——训练语料中植入已知
canary 序列，训练后测试模型的提取率，把“是否记忆”变成可测量指标。

## 6. 有效数据量：funnel 而非乘积

原始字节数不能衡量训练价值。一个实用分解是

\[
N_{\text{usable}}
=N_{\text{raw}}\cdot r_{\text{extract}}\cdot r_{\text{lang}}
\cdot r_{\text{quality}}\cdot r_{\text{safety}}\cdot r_{\text{dedup}}.
\]

这些通过率并不独立，所以生产管线应记录逐级 funnel 和 rejection reason 的交集，不能把
各比例机械相乘作精确估计。开源管线的公开 funnel（Dolma、FineWeb）显示，从原始 crawl
到最终语料的总通过率常在个位数百分比量级 [[4]](#ref-4)[[6]](#ref-6)——"raw PB"与
"usable tokens"之间有一个数量级以上的差距。

## 7. 实现映射（本仓库）

| 概念 | 本仓库位置 | 可验证的契约 |
| --- | --- | --- |
| A4 总体任务与数据边界 | `experiments/official/a4-data.md` | WARC/WET、2500 WET、GPT-2 EOT 与固定训练预算 |
| HTML/WET 输入 | `assignments/spring2026/assignment4-data/cs336_data/wet_files.py` | 流式读取 record，不把全 shard 载入内存 |
| 字节到文本 | `cs336_data/extract_lang.py` | 解码、正文抽取、语言 label 与 score |
| 文档级统计 | `cs336_data/pipeline.py::PipelineStats` | 输入/输出文档和字符数、各拒绝原因 |
| 文档边界 | `pipeline.py::tokenize_documents` | 每篇文档编码后追加 GPT-2 EOT，写 `uint16` |
| 报告证据 | `assignment4-data/report/main.tex` | 1000 条 WET 是探索样本，不能外推全量 |

这里的 `filter_documents` 接收内存中的 `list[str]`，适合作业验证，不是 web-scale 架构。
生产实现应流式处理、分片写出，并把 provenance 与文本分离存储。

## 8. 实用工作流

1. **登记来源**：所有者、获取方式、时间范围、许可证据和已知排除项。
2. **随机抽样**：先人工看原始 payload、抽取文本与 metadata 的对应关系。
3. **定义文档边界**：重定向页、分页文章、评论区和附件如何处理。
4. **建立基线统计**：语言、域名、长度、时间、MIME、hash 和抽取失败率。
5. **再做过滤与去重**：保留 score、阈值、模型版本和每条 decision。
6. **冻结 snapshot**：保存对象列表和 checksum；“同一 URL”不保证内容不变。
7. **下游验证**：固定模型与 token 预算比较，而非凭肉眼宣布数据更好。

其中第 7 步是 FineWeb 的方法论核心：用小模型在候选语料上做受控对比，把“哪个版本更好”
交给 validation loss 与下游任务，而不是人工打分 [[6]](#ref-6)。

## 9. 易错点、伦理与安全

- 把“公开可访问”等同于“允许训练或再分发”；访问条款、版权、隐私和用途限制是不同问题。
- 只保存清洗后文本，导致无法解释误删、来源占比或删除请求。
- 用 URL 去重：URL 会变化，同 URL 内容也会变化；应同时保留内容 hash。
- 将 fastText 语言标签当真值；短文本、代码混写和小语种容易错分。
- 抽样只看 retained data；必须分层检查 discarded data，才能发现群体性误杀——C4 的
  blocklist 争议正是文档化审计才暴露的 [[2]](#ref-2)。
- 将危险内容直接复制进报告。人工审查应最小化暴露，只保留必要摘要和聚合统计。
- 训练/验证按文档随机切分后再去重，会让镜像跨 split 泄漏；应先建立近重复簇，再按簇切分。
- 高 \(E_s\) 的小来源与低 \(E_s\) 的大来源“权重相同”，但记忆风险完全不同 [[9]](#ref-9)。
- 把合成数据当作免费午餐，忽视递归训练下的 collapse 风险 [[11]](#ref-11)。
- 把质量分类器分数当客观真值：FineWeb-edu 式“教育质量 0–5 分”继承打分模型的
  偏好，换领域必须重校准，否则等于把某一个模型的偏好固化为训练分布。
- 让数学/公式内容走通用网页抽取管道：现有抽取器会丢方程、扭曲符号、拍平代码，
  Nemotron 数学管线为此从 98 个 CC 快照重抓原始 HTML 并用 lynx 保留布局。

## 10. Checklist

- [ ] 数据卡记录来源、版本、时间、许可与责任人。
- [ ] 原始对象、抽取器和过滤配置都有不可变版本标识。
- [ ] manifest 可从训练 shard 追溯到来源，但不泄露凭据和 PII。
- [ ] 报告文档数、字符数、token 数、域分布和期望遍历次数 \(E_s\)。
- [ ] retained/discarded 都做按来源、语言、长度分层抽样。
- [ ] 去重和 contamination 检查发生在 split 之前。
- [ ] 删除/屏蔽请求能传播到衍生数据和后续 checkpoint 策略。
- [ ] 明确区分本地样本实测、proxy 结果和未运行的官方规模实验。

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| 语言分布与预期不符 | 抽样 + 语言 score 分布 | fastText 阈值过松、编码错误 |
| 有效 token 远小于预期 | 逐级 funnel | 抽取失败、重复 template 主导 |
| 下游 loss 不升反降 | mixing 权重与 \(E_s\) | 小源过采样、污染 split |
| 同 URL 内容漂移 | crawl_id 与 hash | snapshot 混用、缓存陈旧 |
| 某域名 token 占比畸高 | 域名集中度统计 | SEO 农场、镜像站 |
| 抽取文本混入导航/乱码 | 抽取器版本与样本 | WET 质量差、需换 WARC 重抽 |
| 删除请求无法落实 | manifest 追踪链 | tokenized 层无 document_id |
| canary 被逐字复述 | 记忆测试 | dedup 缺失、\(E_s\) 过高 |

## 11. 讨论：效度威胁与结论边界

### Construct validity

- “token 数”不是“信息量”；\(N_{\text{usable}}\) 的每级比率都依赖阈值与工具版本；
- 语言/质量 score 是模型输出，不是真值；其系统性偏差会被下游放大；
- 人工抽查样本量小，只能发现粗错误，不能证明分布正确。

### Internal validity

- 来源比较若不固定抽取器、tokenizer 与训练配方，差异无法归因于来源本身；
- funnel 比率之间的相关性使“乘积估计”失真；
- 本仓库 A4 只在 1000–2500 条 WET 上验证，任何全量结论都是外推。

### External validity

- 单一 snapshot 的语言/域名分布不外推到其他时期；
- 小模型的语料偏好（FineWeb 式 ablation）未必与大模型一致 [[6]](#ref-6)；
- 合成数据的结论强烈依赖生成器与领域，不构成普适配方。

论文式表述应限定 snapshot、抽取器版本、过滤配置与验证协议，并公开被拒绝样本的聚合统计，
而不是只报告“我们的语料更好”。

## 面试要点速记

**高频问题与答题要点**

1. **Q：WARC/WAT/WET 分别是什么？** 要点：原始 HTTP 响应（WARC）/ 元数据
   （WAT）/ 提取文本（WET）；只存 WET 会丢 URL 与 provenance，治理与审计断层。
2. **Q：\(E_s\)（期望暴露）衡量什么？** 要点：来源 s 的样本被模型看见的
   期望次数×数据权重；\(E_s\) 越高记忆风险越大（canary 实验可校准阈值）。
3. **Q：数据受限时的结论？** 要点：重复 ~4 epochs 内近似等效新数据；数据
   受限时 compute-optimal 模型应更小。
4. **Q：选 crawl snapshot 的偏差来源？** 要点：抓取时间、语言/域覆盖、
   robots 屏蔽与封禁；部署域与快照时间的错配是系统性偏差，不是噪声。
5. **Q：FineWeb 为什么不用默认 WET 而用 trafilatura 重抽？** 要点：
   WET 混入导航/模板噪声；trafilatura 对正文判定更准，小模型 ablation
   下游验证更优——抽取器选择与后续过滤同量级重要。
6. **Q：FineWeb-edu 规模显著更小，为什么训练出的模型反而更强？** 要点：
   llama-3-70b-instruct 对 50 万样本按教育质量 0–5 打分、滤掉 <3 分；
   质量 > 规模的标志性案例；但“教育质量”口径不等于普适质量。
7. **Q：合成数据什么时候安全、什么时候危险？** 要点：补足稀缺分布
   （Anthropic 用合成数据补情景、Cosmopedia 重建教科书分布）一般安全；
   多代替换真实数据有 model collapse 风险——真实+合成混合、比例受控
   是主流做法。
8. **Q：model collapse 争论的最新结论？** 要点：牛津/剑桥 Nature 论文
   显示崩溃在各 AI 架构（含微调 LLM）普遍存在，“接触少量原始数据可防
   退化（按 PPL 衡量）”的观点被挑战；持续获取多样人类数据成关键
   （“先行者优势”）。

**必背数字**

- 4 epochs 上限；\(E_s\) 与记忆率的单调关系；fine-tune“特殊数据”的收益
  边际递减且随规模缩水。

**工业界参照**

- Common Crawl：2007 年起持续爬取，2024 年已索引约 27 亿网页；原始数据
  须经 URL 黑名单 → 文本抽取 → 语言过滤 → 去重 → PII 移除后才可用。
- FineWeb（2024）：96 个 CC 快照 → 15T tokens；trafilatura 抽取质量高于
  默认 WET（尽管数据集显著更小）。
- FineWeb-edu：50 万样本按教育质量 0–5 打分、滤掉 <3 分；规模显著更小却
  优于 FineWeb 与其他公开数据集。
- Nemotron-CC：预训练 20T tokens 三阶段（0–60% 多样性优先 / 60–90%
  提高高质量占比 / 90–100% Wikipedia 级）；英文另用 8 个新增 CC 快照，
  多语言 3 快照覆盖 15 种语言。
- 合成数据锚点：Cosmopedia 用 Mixtral-8x7B-Instruct 生成 30M+ 文件、
  25B tokens 教科书式数据；Nemotron STEM 实践 88.6k 题 × 3 类 prompt ×
  4 模型扩增后模糊去重。
- 数据量演进：GPT-2（2019）约 100B tokens、约 4 万美元训练成本；Qwen
  去重过滤后 2.2T tokens；Llama-3 405B 用 15T tokens。

## 行业现状与最新进展（2024–2026）

### 开源预训练数据集谱系与规模

| 数据集 / 语料 | 年代 | 规模 | 关键做法 |
| --- | --- | --- | --- |
| GPT-2（WebText） | 2019 | 约 100B tokens | 外链过滤 + 去重；训练成本约 4 万美元 |
| C4 | 2020 | 约 156B tokens | 启发式清洗的 CC 子集 [[1]](#ref-1) |
| The Pile | 2020 | 800GB / 22 源 | 多源拼接 [[3]](#ref-3) |
| Dolma | 2024 | 3T tokens | 透明工具链、可复现 ablation [[4]](#ref-4) |
| FineWeb | 2024 | 15T tokens | 96 个 CC 快照、trafilatura 重抽 [[6]](#ref-6) |
| FineWeb-edu | 2024 | 显著小于 FineWeb | 教育质量分类器过滤（<3 分剔除） |
| Qwen 预训练 | 2024 | 去重过滤后 2.2T tokens | 全网文本/百科/书籍/代码/数学/垂类 |
| Llama-3（405B） | 2024 | 15T tokens | 多源混合；大规模开源先例 |
| Nemotron-CC | 2025–2026 口径 | 20T tokens | 三阶段配比 + 8 个新增英文 CC 快照 |

解读：token 预算约十五年增长两个数量级（100B → 15–20T），但同期质量口径
同步收紧——FineWeb 证明抽取器与过滤决策同量级重要，FineWeb-edu 证明“更小
但更准”可以胜出；从 raw crawl 到 usable tokens 的总通过率常在个位数百分比
量级 [[4]](#ref-4)[[6]](#ref-6)，"raw PB"与"usable tokens"之间差一个数量级以上。

### 垂类专用管线：数学与代码

通用网页管线对公式与代码有系统性偏差。数学侧（Nemotron 实践）：现有抽取
管道（OpenWebMath、MegaMath 等）会丢方程、扭曲符号、拍平代码，因此从
98 个 CC 快照（2014–2024）重新抓原始 HTML，用 lynx 文本浏览器保留布局，
由 Phi-4 归一化为 LaTeX，FineMath 分类器保留高质量样本，再经 MinHash-LSH
去重与 LLM Decontaminator 去污染。代码侧：全部来自 GitHub 原始源码，走
类 BigCode 许可证检测管线（检测并删除无许可文件），哈希精确去重 +
MinHash-LSH 模糊去重，并用 OpenCoder 式启发式过滤。

| 环节 | 通用网页管线 | 数学专用管线 |
| --- | --- | --- |
| 抓取 | WET 预抽取文本 | 从 98 个 CC 快照重抓原始 HTML |
| 抽取 | trafilatura 等正文抽取 | lynx 文本浏览器保留布局 |
| 归一化 | 编码 / 空白清理 | Phi-4 归一化为 LaTeX |
| 过滤 | 质量分类器 | FineMath 分类器 |
| 去重 | 精确 hash + MinHash-LSH | MinHash-LSH |
| 去污染 | n-gram / 重叠检查 | LLM Decontaminator |

### 合成数据的工业采纳

- **Cosmopedia（HF）**：用 Mixtral-8x7B-Instruct 生成 30M+ 文件、
  250 亿 tokens 的合成教科书/博客/故事，重建 Phi-1.5 训练分布。
- **Phi 家族**：以合成数据为主要来源，“教科书式”数据放大 token 效率 [[10]](#ref-10)。
- **NVIDIA Nemotron-4-340B 家族**：专为生成合成数据设计、许可宽松。
- **Magpie**：从对齐 LLM 直接提取高质量指令数据，微调模型有时可媲美
  Llama-3-8B-Instruct。
- **Anthropic**：训练 Claude 3 时用合成数据补足训练数据可能缺失的情景。
- **STEM 合成实践（Nemotron）**：88.6k 题（GSM8K/MATH/AOPS/Stemez/
  OpenStax）× 3 类 prompt（Similar/Harder/Varied）× 4 模型
  （Qwen3-30B-A3B、Qwen3-235B-A22B thinking、DeepSeek-R1、DeepSeek-V3）
  扩增后模糊去重。

### model collapse 争论与“先行者优势”

- 牛津/剑桥 Nature 论文：崩溃现象在各 AI 架构（含微调 LLM）普遍存在，
  “预训练或定期接触少量原始数据可防退化（按 PPL 衡量）”的观点被挑战。
- 但真实世界通常真实+合成累积混合而非完全替代，比例不过高时一般可避免
  性能下降。
- 持续获取多样人类数据成为关键——形成“先行者优势”：谁握有新鲜、多样的
  人类数据，谁就避免递归退化。

**对本讲学习者的启示**：工业界 2024–2026 的主线与本讲的采样过程视角完全
一致——FineWeb/Nemotron-CC 的每一步都是对 \(p_s(x\mid\text{crawl},
\text{extract},\text{filter})\) 的显式设计；质量分层（FineWeb-edu、三阶段
配比）说明“token 数”正让位于“有效 token 数”；合成数据从尝鲜走向标配
（Cosmopedia/Nemotron-4-340B/Magpie），但 collapse 争论提醒：合成是分布的
补充而非替代，\(E_s\) 与配比约束仍是第一性工具。面试中能同时给出“管线
细节 + 数字锚点 + 风险边界”的候选者明显占优。

## 大厂面试真题与答题框架

以下为高频面试题（公开面经风格），不指向任何特定公司的真题。

**题目 1：FineWeb 流水线每一步的动机是什么？**
- 考点：URL 黑名单 → 文本抽取 → 语言过滤 → 去重 → PII 移除的全链路理解。
- 答题框架：(1) 原始 crawl 被模板/SEO/镜像主导，URL 黑名单先挡源头；
  (2) 抽取器决定正文 vs 导航（trafilatura 优于默认 WET，需下游 ablation
  验证）；(3) 语言过滤控分布（fastText 只作估计）；(4) 去重控 \(E_s\) 与
  记忆风险；(5) PII 移除是合规底线，逐级记录 funnel 与拒绝原因。
- 加分项：报出 96 个 CC 快照、15T tokens；抽取器选择与过滤同量级重要。
- 踩坑：只背步骤名说不出“为什么 WET 不够”；把语言分数当真值。

**题目 2：合成数据的正确用法与风险？**
- 考点：补缺 vs 稀释、model collapse、去重与质控成本。
- 答题框架：(1) 定位——补足稀缺分布（Anthropic 补情景、Cosmopedia 重建
  教科书分布），不是无节制稀释真实数据；(2) 生成器——专用模型
  （Nemotron-4-340B）或从对齐 LLM 直接提取（Magpie）；(3) 质控——扩增后
  模糊去重 + 污染检查；(4) 风险——递归训练 collapse（Nature 论文挑战
  “少量原始数据可防退化”）；(5) 配比——真实+合成混合、比例受控。
- 加分项：提“先行者优势”——持续获取多样人类数据成关键。
- 踩坑：宣称“合成数据免费”；忽略 teacher 偏差继承。

**题目 3：如何为领域模型选数据源组合？**
- 考点：来源发现、覆盖/密度/风险/成本/可验证性五维评估。
- 答题框架：(1) 明确目标任务分布，列候选来源（通用网页 + 领域语料 +
  代码/数学 + 合成）；(2) 逐源过五维评估；(3) 算 \(E_s\)——领域小语料
  权重稍大就可能高暴露；(4) 固定模型与 token 预算做受控 ablation 验证
  配比；(5) 冻结 snapshot 并写 manifest。
- 加分项：引用 Nemotron-CC 三阶段（0–60% 多样性 / 60–90% 高质量 /
  90–100% Wikipedia 级）作为课程式配比先例。
- 踩坑：只按 token 数配比不看 \(E_s\)；把领域爬虫当干净数据。

**题目 4：数据法务/许可怎么管？**
- 考点：license 证据链、opt-out、删除传播。
- 答题框架：(1) “公开可访问 ≠ 允许训练/再分发/商用”四问分开；
  (2) 每来源保留许可证据（条款 URL、抓取时间、存档快照）而非只存结论；
  (3) 记录 opt-out 遵守（robots、noai meta tag、DMCA）；(4) 删除请求沿
  document_id → shard → tokenized artifact 传播；(5) 许可不明的来源单独
  标记 + 敏感性分析。
- 加分项：提 Data Provenance Initiative——大量许可标注缺失或与原始条款
  不符 [[7]](#ref-7)。
- 踩坑：只存清洗后文本导致无法响应删除；许可结论存了、证据没存。

**题目 5：FineWeb-edu 比 FineWeb 小，为什么模型反而更强？**
- 考点：质量 > 规模、数据受限 scaling。
- 答题框架：(1) 做法——llama-3-70b-instruct 对 50 万样本按教育质量 0–5
  打分、滤掉 <3 分；(2) 机理——数据受限时有效数据量约受 \(4D_u\) 约束，
  质量过滤提升单位 token 信息密度；(3) 边界——“教育质量”口径不等于
  普适质量，换领域需重校准打分器。
- 加分项：连接 Muennighoff：约 4 epochs 后重复收益趋零 [[8]](#ref-8)。
- 踩坑：泛化成“数据越小越好”；忽略打分器偏好被固化的问题。

**题目 6：数学语料为什么不能直接用通用网页管线？**
- 考点：领域抽取偏差、专用管线设计。
- 答题框架：(1) 问题——现有抽取管道（OpenWebMath/MegaMath 等）会丢
  方程、扭曲符号、拍平代码；(2) 方案——收集数学 URL（InfiMM-WebMath
  等）→ 98 个 CC 快照（2014–2024）重抓原始 HTML → lynx 保留布局 →
  Phi-4 归一化 LaTeX → FineMath 分类器 → MinHash-LSH → LLM
  Decontaminator；(3) 通用教训——抽取器对领域内容的偏差是系统性的，
  必须按领域评估。
- 加分项：指出这解释了“同一 URL 集合在不同管线下分布完全不同”。
- 踩坑：以为换质量阈值就能救回公式；忽略 benchmark 去污染。

**题目 7：\(E_s\) 高的来源意味着什么？如何处置？**
- 考点：期望暴露、记忆、mixing 决策。
- 答题框架：(1) 定义 \(E_s=Tw_s/N_s\)，小源即使权重小也可能高暴露；
  (2) 后果——记忆与提取风险随重复次数增长 [[9]](#ref-9)，canary 可测；
  (3) 处置——降权重、去重或补同类数据摊薄；(4) 报告规范——同时报
  \(w_s\) 与 \(E_s\)。
- 加分项：连接 ~4 epochs 饱和与“再加一个 epoch 是决策不是默认”。
- 踩坑：只报权重不报 \(E_s\)；把平均 \(E_s\) 当上限（分布有长尾）。

## 系统设计题

**设计题 1：为 3T token 预算的领域模型设计数据源组合与配比框架**

- 需求澄清：目标领域与任务？语言范围（单语/多语）？许可与合规边界？是否
  允许合成数据？训练 compute 预算与数据获取预算各是多少？
- 规模估算：参照 Qwen 去重过滤后 2.2T tokens 的多源构成（全网文本/百科/
  书籍/代码/数学/垂类）；3T 预算可按 Nemotron-CC 三阶段思路分段（0–60%
  多样性优先、60–90% 提高高质量占比、90–100% 领域 Wikipedia 级）；funnel
  口径——raw crawl → usable 的总通过率常在个位数百分比，倒推原始抓取量
  需预留 1–2 个数量级冗余。
- 架构：(1) 来源登记层（manifest：owner/许可证据/snapshot ID）；(2) 抽取
  与语言过滤层（trafilatura 式重抽 + fastText 估计）；(3) 质量分层
  （分类器打分 + retained/discarded 分层抽查）；(4) 去重层（精确 hash +
  MinHash-LSH）；(5) 配比与采样层（按 \(E_s\) 约束解混合权重）；(6) 验证层
  （固定小模型 + token 预算做受控对比）。
- trade-off：

| 决策 | 选项 A | 选项 B | 取舍 |
| --- | --- | --- | --- |
| 抽取 | 默认 WET | WARC 重抽 | A 便宜但丢正文与 provenance；B 可审计可重放 |
| 质量 | 高阈值一刀切 | 分层保留低分 | A 密度高但覆盖窄；B 覆盖广需分层管理 |
| 合成 | 不用 | 领域补缺 | A 无 collapse 风险；B 补缺口但需去重与配比 |
| 去重 | 仅精确 hash | + MinHash-LSH | A 快但镜像漏网；B 慢但控 \(E_s\) 长尾 |

- 评测方案：固定模型规模与 token 预算，比较候选配比的 validation loss 与
  领域 benchmark；canary 插入测记忆；对 retained/discarded 做分层抽查。
- 追问预案：领域语料只有 50B tokens？——用 \(E_s\) 算暴露，阶段后置 +
  比例受控的领域合成补缺；benchmark 被污染？——LLM Decontaminator 式
  去污染 + 换 held-out 集。

**设计题 2：公司级“数据来源审计与许可合规”管线**

- 需求澄清：覆盖哪些来源（外购/爬取/开源/合成）？opt-out 政策口径？
  删除请求 SLA？审计报告受众（法务/工程/外部）？
- 规模估算：Data Provenance Initiative 对数千个常用数据集做审计即发现
  大量许可标注缺失 [[7]](#ref-7)——企业级来源常在百到千量级；每来源需
  保存条款 URL、抓取时间与存档快照三件套。
- 架构：来源注册表（许可证据快照）→ 爬取执行器（robots/noai 遵守记录）
  → 入库门禁（无许可证据即拒收、进待审区）→ document_id 贯通层
  （raw → shard → tokenized）→ 删除传播引擎 → 审计报告生成（按来源/
  许可/风险聚合）。
- trade-off：

| 决策 | 选项 A | 选项 B | 取舍 |
| --- | --- | --- | --- |
| 无许可来源 | 直接弃用 | 标记待审后用 | A 合规稳但覆盖损失；B 需敏感性分析兜底 |
| 删除粒度 | 文档级屏蔽 | 派生层全量重算 | A 快但 token 层可能残留；B 彻底但贵 |
| 证据存储 | 条款文本摘录 | 存档快照 | A 轻量但可争辩；B 可举证但存储重 |

- 评测方案：抽样审计通过率；删除请求端到端演练（提出 → 各层可追踪 →
  复查无残留）；许可证据覆盖率报告。
- 追问预案：训练已开始才收到删除请求？——按 checkpoint 策略决定继续/
  重训，document_id 保证可定位；许可条款在训练后变更？——保留“训练时点
  证据”快照，法务评估是否重训。

**设计题 3：合成数据生产与质控平台**

- 需求澄清：用途（预训练补缺 vs 指令微调）？生成器来源（自研/专用模型/
  对齐 LLM 提取）？允许的合成占比上限？
- 规模估算：Cosmopedia 用 Mixtral-8x7B-Instruct 生成 30M+ 文件、25B
  tokens；Nemotron STEM 实践 88.6k 题 × 3 类 prompt × 4 模型——生成侧
  吞吐是放大器，质控（去重/污染检查）必须同规模扩展。
- 架构：种子设计层（题目/prompt 模板）→ 生成调度层（多模型 × 多 prompt
  变体）→ 质控层（模糊去重 + 质量分类器 + 污染检查）→ 配比层（合成
  占比硬上限 + \(E_s\) 联合约束）→ 监控层（跨版本分布漂移检测，防递归
  退化）。
- trade-off：

| 决策 | 选项 A | 选项 B | 取舍 |
| --- | --- | --- | --- |
| 生成器 | 通用 instruct 模型 | 专用合成模型（Nemotron-4-340B 式） | A 便宜；B 许可宽松、分布可控 |
| 指令数据 | 人工标注 | 对齐 LLM 直接提取（Magpie 式） | A 贵但多样；B 快但继承 teacher 偏差 |
| 占比控制 | 软配比 | 硬上限 | A 灵活；B 防 collapse 但可能留缺口 |

- 评测方案：固定基座对比“+合成 vs 纯真实”微调（Magpie 报告中微调模型
  有时可媲美 Llama-3-8B-Instruct 可作上界参照）；PPL 跨代监控退化；
  下游任务回归。
- 追问预案：多代累积后退化怎么发现？——PPL 趋势 + 尾部分布检测；Nature
  论文口径下“少量原始数据可防退化”不可依赖，应控真实数据比例；合成数据
  混入 benchmark 题？——污染检查前置于入库。

## 代码实现题

**代码题 1：Common Crawl 快照的 URL→语言→质量粗过滤流水线骨架**

- 题目：给定 WARC/WET record 流，实现流式粗过滤：URL 黑名单 → 文本抽取
  → 语言过滤 → 质量打分，并输出逐级 funnel 统计。
- 考察点：流式处理（内存与 record 数无关）、funnel 统计、每条 decision
  可追溯。

```python
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class FunnelStats:
    rejects: Counter = field(default_factory=Counter)
    retained: int = 0

    def reject(self, stage: str) -> None:
        self.rejects[stage] += 1

    def report(self, n_input: int) -> dict:
        return {
            "n_input": n_input,
            "rejects": dict(self.rejects),
            "retained": self.retained,
            "retention_rate": self.retained / n_input if n_input else 0.0,
        }


def url_blacklisted(url: str, blacklist: set) -> bool:
    host = url.split("://", 1)[-1].split("/", 1)[0]
    return any(host == d or host.endswith("." + d) for d in blacklist)


def pipeline(records, blacklist, lang_id, min_lang_score,
             quality_scorer, min_quality):
    """records 逐条产出 (url, raw_text, extract_fn)，全程流式。"""
    stats = FunnelStats()
    kept = []
    n_input = 0
    for url, raw, extract in records:
        n_input += 1
        if url_blacklisted(url, blacklist):
            stats.reject("url_blacklist")
            continue
        text = extract(raw)  # trafilatura 式正文抽取
        if len(text) < 200:
            stats.reject("too_short")
            continue
        lang, score = lang_id(text)  # fastText 只作估计，不当真值
        if score < min_lang_score:
            stats.reject("lang_filter")
            continue
        if quality_scorer(text) < min_quality:
            stats.reject("quality_filter")
            continue
        stats.retained += 1
        kept.append(text)
    return kept, stats.report(n_input)
```

- 验收标准：内存占用与 record 总数无关；报告含每级拒绝数与总保留率；
  同一输入重跑结果逐字节一致（无隐藏随机性）。

**代码题 2：合成数据配比与模糊去重统计脚本**

- 题目：给定多来源 token 数与目标配比，输出各来源采样量与 \(E_s\)，并对
  文档集合做 MinHash-LSH 近重复率统计；合成来源占比超硬上限时自动缩裁。
- 考察点：配比口径（token vs 文档）、\(E_s=Tw_s/N_s\) 落地、LSH 分桶。

```python
import hashlib
from dataclasses import dataclass


def minhash_signature(shingles, n_perms: int = 128):
    sig = []
    for i in range(n_perms):
        salt = int(hashlib.sha1(str(i).encode()).hexdigest(), 16)
        sig.append(min(
            salt ^ int.from_bytes(
                hashlib.sha1(s.encode()).digest()[:8], "big")
            for s in shingles))
    return sig


def near_duplicate_rate(docs, n_perms: int = 128, n_bands: int = 16,
                        jaccard_threshold: float = 0.8) -> float:
    sigs = [minhash_signature(set(d.lower().split()), n_perms) for d in docs]
    rows = n_perms // n_bands
    buckets = {}
    for doc_id, sig in enumerate(sigs):
        for b in range(n_bands):
            key = hash(tuple(sig[b * rows:(b + 1) * rows]))
            buckets.setdefault(key, []).append(doc_id)
    dup, seen = 0, set()
    for members in buckets.values():
        for i in members:
            for j in members:
                if j <= i or j in seen:
                    continue
                agree = sum(x == y for x, y in zip(sigs[i], sigs[j]))
                if agree / n_perms >= jaccard_threshold:
                    seen.add(j)
                    dup += 1
    return dup / len(docs) if docs else 0.0


@dataclass
class MixPlan:
    total_tokens: int        # T
    synthetic_cap: float     # 合成占比硬上限，如 0.3


def plan_mix(token_counts, weights, plan: MixPlan):
    """按 token 配比采样并报告 E_s = T * w_s / N_s；合成超上限先缩裁。"""
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    out = {}
    for s, w in weights.items():
        sampled = min(int(plan.total_tokens * w), token_counts[s])
        out[s] = {
            "sampled_tokens": sampled,
            "E_s": plan.total_tokens * w / max(token_counts[s], 1),
        }
    synth = sum(v["sampled_tokens"] for s, v in out.items()
                if s.startswith("synthetic"))
    cap = int(plan.total_tokens * plan.synthetic_cap)
    if synth > cap and synth:
        scale = cap / synth
        for s, v in out.items():
            if s.startswith("synthetic"):
                v["sampled_tokens"] = int(v["sampled_tokens"] * scale)
    return out
```

- 验收标准：权重和校验；每个来源都输出 \(E_s\)；合成占比不超上限；
  MinHash 结果确定性（sha1 无随机性）；近重复率随重复文档比例单调。

**代码题 3：\(E_s\) 计算与高暴露来源风险报告器**

- 题目：给定来源 token 数、混合权重与训练预算，输出各来源 \(E_s\) 排名，
  对超过阈值（默认 4 epochs）的来源给出处置建议。
- 考察点：\(E_s\) 公式落地、阈值化建议、报告规范（权重与暴露同时报）。

```python
from dataclasses import dataclass


@dataclass
class Source:
    name: str
    tokens: int      # N_s
    weight: float    # w_s


def exposure_report(sources, total_tokens: int,
                    warn_epochs: float = 4.0):
    """约 4 epochs 后重复收益趋零 [[8]](#ref-8)；高暴露同时推高记忆风险。"""
    rows = []
    for s in sources:
        e_s = total_tokens * s.weight / max(s.tokens, 1)
        rows.append({
            "source": s.name,
            "N_s": s.tokens,
            "w_s": s.weight,
            "E_s": round(e_s, 2),
            "action": ("dedup_or_downweight_or_add_data"
                       if e_s > warn_epochs else "ok"),
        })
    return sorted(rows, key=lambda r: -r["E_s"])
```

- 验收标准：纯函数无副作用；报告按 \(E_s\) 降序；超阈值来源给出可执行
  处置建议；\(N_s=0\) 时不除零。

## 12. 小结

数据规模只有在采样过程、来源边界和有效 token 暴露次数明确时才有意义。WARC 提供可重
抽取能力，WET 提供处理便利，但二者都不是现成训练集；开源语料生态（C4 → The Pile →
Dolma/RefinedWeb/FineWeb）的演进方向是透明管线与可复现 ablation，而非单纯堆量
[[1]](#ref-1)[[3]](#ref-3)[[4]](#ref-4)[[5]](#ref-5)[[6]](#ref-6)。Lecture 13 的交付
不是某个神奇数据源，而是一套可追溯、可删除、可比较的数据契约；Lecture 14 才在这个契约
上讨论过滤、去重和重加权。

## 参考文献

<a id="ref-1"></a>[1] C. Raffel, N. Shazeer, A. Roberts, et al. “Exploring the
Limits of Transfer Learning with a Unified Text-to-Text Transformer.”
*JMLR*, 2020. [link](https://arxiv.org/abs/1910.10683)

<a id="ref-2"></a>[2] J. Dodge, M. Sap, A. Marasović, et al. “Documenting
Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus.”
*EACL*, 2021. [link](https://arxiv.org/abs/2104.08758)

<a id="ref-3"></a>[3] L. Gao, S. Biderman, S. Black, et al. “The Pile: An
800GB Dataset of Diverse Text for Language Modeling.” arXiv:2101.00027,
2020. [link](https://arxiv.org/abs/2101.00027)

<a id="ref-4"></a>[4] L. Soldaini, R. Kinney, A. Bhagia, et al. “Dolma: An
Open Corpus of Three Trillion Tokens for Language Model Pretraining
Research.” *ACL*, 2024. [link](https://arxiv.org/abs/2402.00159)

<a id="ref-5"></a>[5] G. Penedo, Q. Malartic, D. Hesslow, et al. “The
RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web
Data, and Web Data Only.” *NeurIPS*, 2023. [link](https://arxiv.org/abs/2306.01116)

<a id="ref-6"></a>[6] G. Penedo, H. Kydlíček, L. Ben allal, et al. “The
FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.”
arXiv:2406.17557, 2024. [link](https://arxiv.org/abs/2406.17557)

<a id="ref-7"></a>[7] S. Longpre, R. Mahari, A. Chen, et al. “The Data
Provenance Initiative: A Large Scale Audit of Dataset Licensing &
Attribution in AI.” *NeurIPS Datasets and Benchmarks*, 2023.
[link](https://arxiv.org/abs/2310.16787)

<a id="ref-8"></a>[8] N. Muennighoff, A. Rush, B. Barak, et al. “Scaling
Data-Constrained Language Models.” *NeurIPS*, 2023.
[link](https://arxiv.org/abs/2305.16264)

<a id="ref-9"></a>[9] N. Carlini, D. Ippolito, M. Jagielski, et al.
“Quantifying Memorization Across Neural Language Models.” *ICLR*, 2023.
[link](https://arxiv.org/abs/2202.07646)

<a id="ref-10"></a>[10] S. Gunasekar, Y. Zhang, J. Aneja, et al.
“Textbooks Are All You Need.” arXiv:2306.11644, 2023.
[link](https://arxiv.org/abs/2306.11644)

<a id="ref-11"></a>[11] I. Shumailov, Z. Shumaylov, Y. Zhao, et al. “The
Curse of Recursion: Training on Generated Data Makes Models Forget.”
arXiv:2305.17493, 2023. [link](https://arxiv.org/abs/2305.17493)

## 延伸阅读与复现材料

- Stanford CS336, [Lecture 13 — Data Sources and Datasets](https://github.com/stanford-cs336/lectures/blob/main/lecture_13.py)
- [Common Crawl — Get Started](https://commoncrawl.org/get-started)
- [IIPC WARC specifications](https://iipc.github.io/warc-specifications/)
- [Data Curation 主题导航](../experiments/topics/data-curation.md)
- [A4 Data 官方题面](../assignments/spring2026/assignment4-data/cs336_assignment4_data.pdf)
- [Scaling Data-Constrained Language Models（Muennighoff et al., arXiv:2305.16264）](https://arxiv.org/abs/2305.16264)（访问日期 2026-10-04）
- [The Llama 3 Herd of Models（arXiv:2407.21783）](https://arxiv.org/abs/2407.21783)（访问日期 2026-10-04）
- [DeepSeek-V3 Technical Report（arXiv:2412.19437）](https://arxiv.org/abs/2412.19437)（访问日期 2026-10-04）
- [Stanford CS336 课程主页](https://cs336.stanford.edu)（访问日期 2026-10-04）
