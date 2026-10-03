---
title: "Lecture 13 — Data Sources & Datasets"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-05-11"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_13.py"
  - "../assignments/assignment4-data/"
---

# Lecture 13 — 数据来源与数据集：先定义数据，再谈规模

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述，非课程提交
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
| HTML/WET 输入 | `assignments/assignment4-data/cs336_data/wet_files.py` | 流式读取 record，不把全 shard 载入内存 |
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

### 11.1 Construct validity

- “token 数”不是“信息量”；\(N_{\text{usable}}\) 的每级比率都依赖阈值与工具版本；
- 语言/质量 score 是模型输出，不是真值；其系统性偏差会被下游放大；
- 人工抽查样本量小，只能发现粗错误，不能证明分布正确。

### 11.2 Internal validity

- 来源比较若不固定抽取器、tokenizer 与训练配方，差异无法归因于来源本身；
- funnel 比率之间的相关性使“乘积估计”失真；
- 本仓库 A4 只在 1000–2500 条 WET 上验证，任何全量结论都是外推。

### 11.3 External validity

- 单一 snapshot 的语言/域名分布不外推到其他时期；
- 小模型的语料偏好（FineWeb 式 ablation）未必与大模型一致 [[6]](#ref-6)；
- 合成数据的结论强烈依赖生成器与领域，不构成普适配方。

论文式表述应限定 snapshot、抽取器版本、过滤配置与验证协议，并公开被拒绝样本的聚合统计，
而不是只报告“我们的语料更好”。

## 12. 面试备考（Interview Prep）

> 数据来源是 LLM 面试的数据工程高频题：面试官常从「WARC/WAT/WET 是什么」切入，追到
> 「\(E_s\) 期望暴露」「数据受限 4 epochs」「model collapse」「provenance 怎么追踪」。
> 核心是把「数据集」理解成**采样过程**（\(p_{\text{train}}=\sum w_s p_s\)），把「多大规模」
> 升级为「什么分布、什么许可、什么暴露次数」。下面按「一页速览 → 高频题 → 手撕 → 追问」四层组织。

### 12.1 一页速览卡（面试前 1 分钟）

**核心主张**：数据集不是中性文件集合，而是由「来源发现、抓取时机、抽取器、过滤器、混合权重」
共同决定的采样过程；同一 URL 集合在不同管线下分布完全不同。

**必背数字与公式**

- 训练分布 \(p_{\text{train}}(x)=\sum_s w_s\,p_s(x\mid \text{crawl,extract,filter})\)。
- 期望遍历次数 \(E_s=\frac{T\,w_s}{N_s}\)（比 \(w_s\) 更直接关联记忆风险）。
- 数据重复约 **4 epochs** 后边际收益趋近于零（Muennighoff）。
- 有效 token funnel \(N_{\text{usable}}=N_{\text{raw}}\cdot r_{\text{extract}}r_{\text{lang}}r_{\text{quality}}r_{\text{safety}}r_{\text{dedup}}\)，总通过率常在个位数百分比。
- WARC（原始响应）/ WAT（元数据）/ WET（纯文本）。

**三句话答高频**

1. 数据集是采样过程：C4/RefinedWeb/FineWeb 都源自 Common Crawl，但管线不同、分布大相径庭。
2. \(E_s\) 衡量来源 token 被看到的期望次数，小来源即便权重低、\(E_s\) 高也构成记忆风险。
3. 数据重复 4 epochs 后饱和；合成数据递归训练会 model collapse，只能补充稀缺分布。

### 12.2 高频面试题与答题框架

**Q1：WARC / WAT / WET 分别是什么？为什么 WET record ≠ 网页正文？**

- **WARC**：保存请求/响应、header、URL、原始 payload，可重新抽取与审计；**WAT**：解析后的 metadata/links；**WET**：预抽取纯文本，吞吐友好但丢 DOM、链接上下文与部分 provenance。
- 同一页面可能被抓取多次，一个 record 可能只剩导航/错误页/乱码；所以 `WET record ≠ 网页正文 ≠ 训练文档`。
- 只存 WET 会丢治理与审计能力，生产应保留 WARC 或等价 provenance。

**Q2：\(E_s\)（期望暴露）衡量什么？为什么比 \(w_s\) 更重要？**

- \(E_s=Tw_s/N_s\)：来源 \(s\) 的 token 平均被看到的次数，混合权重 \(w_s\)、总预算 \(T\)、来源大小 \(N_s\) 共同决定。
- 小来源即便权重不大，\(E_s\) 高就反复暴露 → memorization 风险高。只报 \(w_s\) 不足以判断 mixing。
- Carlini 等：逐字记忆概率随重复次数显著增长，低重复样本几乎不被复述，高重复样本提取率可观。

**Q3：数据受限时的结论？（4 epochs）**

- Muennighoff 等：把「唯一 token \(D_u\)」与「重复次数 \(R\)」解耦，重复收益按幂律衰减，约 4 epochs 后趋近饱和。
- 等价地有效数据量约束在约 \(4D_u\)；「再加一个 epoch」是决策不是默认，收益可由 scaling curve 预估。

**Q4：数据混合权重怎么定？三种口径？**

- 权重按**文档数 / 字节数 / token 数**计算会得到不同分布；真正影响优化的是训练时被采到的 token 概率。
- 报告 mixing 必须同时给 \(w_s\)（权重）与 \(E_s\)（暴露），否则「来源占比」无法解释。

**Q5：七类来源的「隐藏价格」是什么？**

- 开放网页：覆盖广但模板/SEO/镜像/许可/PII；书籍：长程连贯但版权+出版选择偏差；百科：结构密度高但风格单一；代码：可执行验证但许可证/密钥；论文：技术密度高但 OCR/订阅复杂；论坛对话：贴近真实但毒性/身份/上下文缺失；合成：目标明确但继承 teacher 偏差。

**Q6：model collapse 是什么？合成数据怎么用？**

- 在递归生成的数据上反复训练，分布尾部逐步丢失（Shumailov 等），导致「失忆」与多样性坍缩。
- 正确用法是**补充**稀缺分布（如 Phi 的教科书式合成数据），而非无节制稀释真实数据。

**Q7：provenance 与删除请求怎么追踪？**

- manifest 至少含 `document_id, source_uri, crawl_id, warc_path, record_offset, retrieved_at, extractor_version, content_hash, license_evidence, filter_decisions`。
- 删除请求要能由 `document_id → shard → tokenized artifact` 传播，而不是只在原始文本层「删除」；这就是 manifest 作为一等数据结构存在的原因。

**Q8：有效数据量 funnel？为什么不能机械相乘？**

- \(N_{\text{usable}}=N_{\text{raw}}\cdot r_{\text{extract}}r_{\text{lang}}r_{\text{quality}}r_{\text{safety}}r_{\text{dedup}}\)。
- 各通过率不独立，应记录逐级 funnel 与 rejection reason 交集；从 raw crawl 到最终语料总通过率常在个位数百分比（Dolma/FineWeb）。

**Q9：memorization 风险怎么度量？**

- 逐字记忆概率随模型规模与样本重复次数增长；canary 序列（已知内容）植入训练语料后测提取率，把「是否记忆」变成可测量指标。
- 缓解：dedup、PII scrubbing、canary 监测；含 PII 的小来源只要 \(E_s\) 高就构成实质性风险。

**Q10：为什么「公开可访问」≠「允许训练」？**

- 「公开可访问」「允许训练」「允许再分发」「允许商用」是四个不同问题；Data Provenance Initiative 发现大量数据集许可标注缺失或与原始条款不符。
- 应保留许可证据（条款 URL、抓取时间、存档快照）而非结论，并记录 robots.txt/noai/DMCA opt-out 遵守情况。

### 12.3 手撕要点（\(E_s\) 与 funnel）

面试让「算期望暴露」或「估有效 token」时，按公式写：

```text
E_s = T * w_s / N_s     （来源 s 的平均期望遍历次数）
  例：T=1T tokens, w_s=0.01, N_s=10B
  E_s = 1e12 * 0.01 / 1e10 = 1 次

usable funnel = raw × r_extract × r_lang × r_quality × r_safety × r_dedup
  （各比率不独立，需记录逐级 funnel 与 rejection 交集）
```

**三个必踩坑**

1. **别把 token 数当信息量**：\(N_{\text{usable}}\) 每级比率依赖阈值与工具版本。
2. **别只报 \(w_s\)**：小来源 \(E_s\) 高才是记忆风险的直接来源。
3. **别在 split 后再去重**：镜像会跨 split 泄漏；应先建近重复簇，再按簇切分。

### 12.4 高频追问与陷阱

| 追问 | 正确方向 |
| --- | --- |
| 用 URL 去重可以吗？ | 否，URL 会变、同 URL 内容也会变，应保留内容 hash |
| fastText 语言标签是真值吗？ | 否，短文本/代码混写/小语种易错分 |
| 抽样只看 retained 够吗？ | 否，必须分层看 discarded 才能发现群体性误杀 |
| 合成数据是免费午餐吗？ | 否，递归训练有 model collapse 风险 |
| 公开数据能随便训练吗？ | 否，访问/版权/隐私/用途是不同问题 |

## 13. 小结

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
- [A4 Data 官方题面](../assignments/assignment4-data/cs336_assignment4_data.pdf)
