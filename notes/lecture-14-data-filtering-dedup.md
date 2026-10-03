---
title: "Lecture 14 — Data Filtering, Deduplication & Mixing"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-05-13"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_14.py"
  - "../assignments/assignment4-data/"
---

# Lecture 14 — 过滤、去重与重加权：把规则变成可测系统

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述，非课程提交
- 适用对象：自学者、数据工程师与 LLM 研究者

## 摘要

原始语料到训练分布之间的每一步——过滤、去重与重加权——都是一个可测量、可重放、可证伪的
决策系统，而不是一组“清洗脚本”。本文系统梳理四类核心方法：以 fastText 与 Gopher
heuristics 为代表的廉价过滤、以 exact hash 去重与 MinHash/LSH 为代表的集合相似度去重、
以 SemDeDup 为代表的语义去重，以及以 DSIR/DoReMi 为代表的分布重加权。对每种方法给出
数学定义、复杂度、超参数语义与失效模式：LSH 的 \(1-(1-s^r)^b\) 候选率、MinHash 的
无偏估计方差、DSIR 的密度比爆炸与 ESS 诊断、DoReMi 的 group-DRO 域权重学习。工程上，
本文给出“便宜规则 → 模型过滤 → 全局去重 → 混合”的分层管线、阈值选择的多目标协议
（人工精度 × token 产量 × 域覆盖 × 下游 loss），以及每级决策必须留存的审计字段；
研究上，讨论 contamination、classifier bias、代表性损失与效度威胁。目标是让读者能把
“数据清洗”升级为可复核的数据决策系统。

**关键词：** Filtering；fastText；Gopher Rules；MinHash；LSH；Deduplication；
DSIR；DoReMi；Data Mixing；Contamination

## 本文贡献

1. 统一四类方法的数学语义：阈值过滤、集合相似度去重、语义去重与密度比重采样；
2. 推导 MinHash 无偏性、LSH 候选率曲线与 DSIR 权重的数值陷阱（ratio 爆炸、ESS 崩塌）；
3. 给出分层管线的复杂度分析与审计字段规范（score/decision/reason/version 可重放）；
4. 将阈值选择形式化为多目标优化，避免“越严越好”与 classifier accuracy 误导；
5. 提供误杀审计、contamination 防控、分组误差报告与论文级 recipe 报告规范。

## 学习目标

1. 深入理解 fastText、Gopher heuristics、MinHash、LSH 和 DSIR 分别解决什么问题；
2. 能设计“便宜规则 → 模型过滤 → 全局去重 → 混合”的可扩展管线；
3. 会用人工误差、保留 token、下游 loss 和分布覆盖共同选择阈值；
4. 识别 contamination、memorization、代表性损失和 classifier bias；
5. 能为一次数据 recipe 变更写出可证伪的 ablation 协议。

## 先修知识

- Lecture 13：数据来源、provenance 与 token 预算（过滤的输入契约）。
- Lecture 09：有效数据量与 scaling（过滤改变 \(D\) 的“有效质量”）。
- 概率基础：Jaccard 相似度、无偏估计、重要性采样权重。

## 相关工作与技术谱系

过滤线沿“规则 → 分类器 → 受控评估”演进。C4 用一组启发式规则定义了第一代 web 清洗
基线 [[1]](#ref-1)；fastText 以词/字符 n-gram 线性分类提供廉价的语言与质量打分
[[2]](#ref-2)；Gopher 系统化了结构化 heuristics 并分析其对训练分布的影响
[[3]](#ref-3)。RefinedWeb 表明严格的规则过滤 + 去重即可让纯网页数据超越 curated
混合 [[4]](#ref-4)；FineWeb 进一步用小模型 ablation 把“哪个过滤器更好”变成受控实验
问题 [[5]](#ref-5)。

去重线建立在集合相似度理论上：Broder 的 MinHash 用最小哈希无偏估计 Jaccard
[[6]](#ref-6)，Indyk-Motwani 的 locality-sensitive hashing 把近邻搜索降到次二次
[[7]](#ref-7)。Lee 等证明去重显著降低 perplexity 并减少 memorization [[8]](#ref-8)；
SemDeDup 进一步在 embedding 空间删除语义重复，以更少数据等质量训练 [[9]](#ref-9)。
数据受限场景的研究还表明，唯一 token 数主导 scaling 收益，这为去重提供了规模论据
[[10]](#ref-10)。

选择与混合线：DSIR 用 hashed n-gram 密度比做 importance resampling [[11]](#ref-11)；
DoReMi 用 group DRO 在小 proxy 模型上学习域权重，再用该权重训练大模型，等预算下
加速收敛 [[12]](#ref-12)。这两类方法改变的是采样分布，而不是删除决策——与过滤、
去重构成正交的数据控制轴。

## 1. Filtering 不是越严格越好

设过滤器以阈值 \(\tau\) 保留 \(q(x)\ge\tau\) 的文档。提高 \(\tau\) 往往提升人工精度，
却减少领域和语言覆盖。应寻找

\[
\max_\tau\; U(\tau)
=\text{downstream quality}
-\lambda C_{\text{tokens}}
-\gamma\,\text{distribution shift},
\]

而不是最大化 classifier accuracy。每个 stage 应输出 score、decision 和 reason，才能
重放不同阈值。

这解释了 FineWeb 的方法论选择：与其在规则上争论，不如固定训练配方，对候选语料做
小模型受控对比，让 validation loss 与下游任务裁决 [[5]](#ref-5)。

## 2. fastText：廉价的语言与质量分类

fastText 用词/字符 n-gram 的向量和表示文本，再做线性分类 [[2]](#ref-2)：

\[
h(x)=\frac{1}{|G(x)|}\sum_{g\in G(x)}z_g,\qquad
p(y\mid x)=\operatorname{softmax}(Wh(x)).
\]

字符 n-gram 让它对拼写变化和未登录词稳健，推理快，适合 web-scale 初筛。实务中：

- 语言识别保留 `top_label + confidence`，不要立刻丢掉完整分数。
- 对短文、代码、转写文本和 code-switching 单独校准。
- 质量分类的正负样本定义了“质量”：以 Wikipedia 为正、随机网页为负，学到的可能是
  来源风格——RefinedWeb 明确拒绝了 classifier-based curation，部分原因即在此
  [[4]](#ref-4)。
- score 常未校准，0.9 不必然意味着 90% 正确；用人工 holdout 画 precision-recall/yield 曲线。

本仓库 `quality.py::classify_quality` 可注入 fastText-compatible model；无模型时的固定
权重只是可解释 proxy，不应冒充生产分类器。

## 3. Gopher rules：高吞吐、可解释的第一道门

Gopher 风格规则用文档统计抓取明显异常 [[3]](#ref-3)，例如：

- 总词数过少/过多；
- 平均词长异常；
- 省略号结尾行比例过高；
- 含字母词比例过低；
- 实际系统还常检查 stop words、symbol、重复 n-gram、模板行等。

仓库 A4 子集明确为：词数 \(50\)–\(100000\)，平均词长 \(3\)–\(10\)，省略号行比例不超过
\(0.30\)，含字母词比例至少 \(0.80\)。这些是课程接口，不是跨语言通用常数。

规则优势是便宜、可解释；弱点是边界跳变和文化/领域偏差。诗歌、表格、数学、代码、聊天和
小语种可能“质量高但形态不合规则”。正确做法是按文档类型统计误杀，而非只看全局通过率；
C4 的 blocklist 争议（Lecture 13）已经说明规则会携带隐性价值判断。

## 4. Exact dedup：先明确删除语义

完全去重可以在 document、paragraph 或 line 级做：

\[
k(x)=H(\operatorname{normalize}(x)).
\]

“每簇保留一份”与“删除所有重复出现的行”含义不同。本仓库 exact-line 实现两遍计数，并
删除**所有全局频次大于 1 的行**。它能去模板，但也会把合法免责声明或常见短句从每篇文档
删掉；生产系统通常还需最小长度、频率上限或 boilerplate 专用策略。

去重的价值有两条证据线：Lee 等显示去重后训练的模型 perplexity 更低、且逐字记忆更少
[[8]](#ref-8)；data-constrained scaling 则表明收益主要由**唯一** token 驱动——重复
token 的边际价值随 epoch 数快速衰减 [[10]](#ref-10)。二者合起来意味着：与其在重复
数据上花计算，不如把同等预算投向更多唯一数据（或用去重把“虚高”的语料量变成真实覆盖）。

## 5. MinHash：用碰撞估计 Jaccard

把规范化文档变成 word \(n\)-gram shingles 集合 \(A\)。相似度为

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
\]

对随机排列/哈希 \(h_k\) 取集合最小值 [[6]](#ref-6)：

\[
m_k(A)=\min_{a\in A}h_k(a),\qquad
\Pr[m_k(A)=m_k(B)]=J(A,B).
\]

用 \(K\) 个 hash 的碰撞率估计 Jaccard：

\[
\widehat J(A,B)=\frac1K\sum_{k=1}^{K}
\mathbf 1[m_k(A)=m_k(B)],
\quad
\operatorname{Var}(\widehat J)=\frac{J(1-J)}{K}.
\]

增大 \(K\) 降低估计方差，但增加 CPU、内存和 I/O。规范化、shingle 长度与短文处理比
“换一个 hash 库”更影响语义。

## 6. LSH：MinHash 不是候选检索本身

将 \(K=br\) 个 signature 分成 \(b\) 个 bands，每 band \(r\) 行；至少一个 band 完全相同
即成为候选 [[7]](#ref-7)。相似度为 \(s\) 时，成为候选的概率为

\[
P_{\text{cand}}(s)=1-(1-s^r)^b.
\]

- 增大 \(b\)：召回提高，也产生更多候选。
- 增大 \(r\)：条件更严格，候选更少。
- 近似拐点约为 \(s^\star\approx(1/b)^{1/r}\)，但最终阈值仍应通过数据校准。

LSH 只负责缩小 pair 数；A4 实现随后计算真实 Jaccard，再用 union-find 合并簇，并按规范
路径顺序确定 survivor。需警惕传递闭包：\(A\sim B\)、\(B\sim C\) 不保证 \(A\sim C\)，
但 union-find 会把三者放入同簇。

### 6.1 语义去重：SemDeDup

MinHash 只能捕捉表面重叠。SemDeDup 先用预训练 encoder 把文档嵌入向量空间，对每个
cluster 内的样本做基于 cosine 相似度的修剪，删除语义冗余 [[9]](#ref-9)。它在
web-scale 数据上以约减半的数据达到等质量，其代价是 embedding 计算、聚类成本，以及
“语义相近但事实不同”被误删的风险——对事实密集语料应设置更高的保留阈值并审计簇内容。

## 7. DSIR：不删除，也能把 raw 分布拉向目标

Data Selection via Importance Resampling 训练两个轻量密度模型：目标分布
\(p_{\text{target}}\) 与原始池 \(p_{\text{raw}}\) [[11]](#ref-11)。每个样本权重

\[
w(x)=\frac{p_{\text{target}}(x)}{p_{\text{raw}}(x)},\qquad
\tilde w_i=\frac{w(x_i)}{\sum_j w(x_j)}.
\]

再按 \(\tilde w_i\) 无放回采样。实践常用 hashed n-gram bag-of-words 和朴素 Bayes/多项式
模型，在 log 空间计算：

\[
\log w(x)=\sum_g c_g(x)
\left[\log p_{\text{target}}(g)-\log p_{\text{raw}}(g)\right].
\]

关键陷阱：

- 目标集太小或被 benchmark 污染，会把污染放大到训练集。
- \(p_{\text{raw}}\) 很小时 ratio 爆炸，需要 smoothing、log-weight clipping。
- DSIR 匹配的是所选特征分布，不保证事实性、安全或语义质量。
- 重采样后的有效样本量可用
  \[
  \mathrm{ESS}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}
  \]
  诊断；ESS 很低表示少数文档支配数据。

DSIR 与 classifier filtering 的差异是：前者估计密度比并保留概率性多样性，后者按决策边界
截断。两者都依赖 reference data，均需审计 reference 的来源。

## 8. Data Mixing 与 DoReMi：学习域权重

混合权重 \(w_s\)（Lecture 13）通常由经验设定。DoReMi 把它变成学习问题：先训练一个小
proxy 模型，在其上以 group DRO 优化各域的参考分布权重——给当前模型表现最差的域加大
权重——再把学到的域配比用于正式大模型训练 [[12]](#ref-12)。等计算预算下，DoReMi
学到的混合比经验配比收敛更快、下游更优。工程要点：

- proxy 模型必须足够小，否则“学权重”的开销超过收益；
- 域定义（domain granularity）本身就是超参数：过粗掩盖结构，过细放大噪声；
- 权重学习与后续训练之间不应更换 tokenizer/数据版本，否则权重失效；
- 报告 learned weights 与初始配比的差异，并做固定预算对比而非只报最终 loss。

## 9. 推荐管线与复杂度

1. 解码/正文抽取，记录失败。
2. cheap structural rules 与语言识别。
3. PII 掩码、安全和学习式质量 score。
4. exact hash 去重。
5. MinHash/LSH 全局近重复去重（可选：SemDeDup 语义去重）。
6. DSIR 或 DoReMi 域权重做分布匹配。
7. tokenizer + EOT，固定训练预算做 ablation。

对 \(N\) 文档、每篇平均 \(M\) shingles、\(K\) hashes，朴素 all-pairs 是 \(O(N^2)\)；
MinHash 构造约 \(O(NMK)\)，LSH 目标是让精确比较只发生在小候选集。仍要记录最大
bucket、候选 pair 数和 peak RSS，避免热门模板形成超大 bucket。

## 10. 关键公式速查

- Jaccard：\(J(A,B)=|A\cap B|/|A\cup B|\)。
- MinHash 碰撞：\(\Pr[m(A)=m(B)]=J(A,B)\)。
- LSH 候选率：\(1-(1-s^r)^b\)。
- DSIR 权重：\(w(x)=p_{\text{target}}(x)/p_{\text{raw}}(x)\)。
- 重加权有效样本量：\(\mathrm{ESS}=(\sum_iw_i)^2/\sum_iw_i^2\)。

## 11. 实现映射（本仓库）

| 方法 | 本仓库位置 | 关键行为/测试 |
| --- | --- | --- |
| Gopher | `cs336_data/quality.py::gopher_quality_filter` | 四类规则，Unicode word regex |
| fastText 接口 | `quality.py::classify_quality` | model 注入、label 归一化、返回 confidence |
| 过滤漏斗 | `cs336_data/pipeline.py::filter_documents` | PII→Gopher→quality→safety，并累计拒绝数 |
| exact-line | `cs336_data/dedup.py::exact_line_deduplication` | 两遍 hash，重复行全部删除 |
| normalization/shingles | `dedup.py::_normalize`, `_word_ngrams` | NFD、小写、去重音/标点、word n-grams |
| MinHash/LSH | `dedup.py::minhash_deduplication` | band 候选→真 Jaccard→确定性 union-find |
| A4 结果边界 | `assignment4-data/report/main.tex` | 21 tests 与 1000-WET 样本；未声称官方终训 |
| DSIR/DoReMi | 当前 A4 未实现 | 应另加 reference split、weight/ESS 审计，不能假装已有 |

## 12. 易错点、伦理与安全

- 在 train/validation 切分后才去重，造成跨 split contamination。
- Python 内置 `hash()` 跨进程/版本不稳定；使用固定算法和 seed。
- LSH 候选直接视为重复，没有真 Jaccard 复核。
- MinHash 空集合全部同 signature，使空文档聚成一簇；应先过滤空文档。
- survivor 永远选“最早抓到”，可能系统性偏向特定域；策略必须可解释并审计来源占比。
- quality/safety classifier 把少数群体术语当低质或有害；保留分组误差与申诉/回滚路径。
- 只优化 validation loss，可能奖励 benchmark 泄漏或风格窄化。
- 去重不能解决语义改写、翻译重复和拼接污染；也不证明模型不会记忆 [[8]](#ref-8)。
- DoReMi/DSIR 的目标域若含 benchmark，等于把泄漏写进采样分布 [[11]](#ref-11)[[12]](#ref-12)。

## 13. Checklist

- [ ] 每个过滤器都有 score、版本、阈值、reason 和 retained/discarded 抽样。
- [ ] 阈值 sweep 同时报人工 precision/recall、token yield、域覆盖和下游 proxy。
- [ ] exact 去重明确 document/line 单位及“保留一份还是全删”。
- [ ] MinHash 固定 normalization、shingle、hash 数、bands、seed。
- [ ] 报告候选数、真 Jaccard 拒绝率、簇大小和 survivor 来源。
- [ ] split 在近重复簇形成后完成。
- [ ] DSIR 的 target/raw 数据互斥，权重有 smoothing/clipping，并报告 ESS。
- [ ] DoReMi 的 proxy 模型规模、域定义与权重学习成本单独报告。
- [ ] 安全与 PII 审计不被“质量分高”替代。

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| 过滤后某语言几乎消失 | 分组通过率 | 阈值对短文/小语种过严 |
| LSH 找不到明显重复 | band/r 参数 | \(r\) 过大、拐点高于实际相似度 |
| 候选 pair 爆炸 | bucket 大小分布 | 模板簇形成超大 bucket |
| 去重后 loss 反升 | survivor 与簇审计 | 误删合法高频结构、过度删除 |
| DSIR 权重集中少数文档 | ESS 与权重直方图 | ratio 爆炸、缺 clipping |
| DoReMi 权重震荡 | proxy loss 曲线 | 域过细、学习率过高 |
| validation 异常偏好 | split 前去重顺序 | 跨 split 近重复泄漏 |
| 峰值内存超限 | LSH bucket/streaming | 一次性载入全量签名 |

## 14. 讨论：效度威胁与结论边界

### 14.1 Construct validity

- “质量”由正负样本定义，classifier accuracy 不是数据质量的度量；
- Jaccard 阈值是表面相似度，与“信息冗余”不一一对应；
- ESS 衡量权重集中度，不衡量分布匹配的正确性。

### 14.2 Internal validity

- 阈值、去重、混合同时变更时，无法归因单一因素；
- 下游 proxy 模型过小，其偏好未必代表目标规模模型 [[5]](#ref-5)；
- 人工标注样本量小且标注者偏差存在，precision 数字有置信区间。

### 14.3 External validity

- 在一个语料/语言上校准的阈值与 \(b,r\) 不外推到其他分布；
- 规则与分类器的误杀模式依赖时代（web 风格演化）；
- 本仓库 A4 在 1000-WET 样本上验证，全量行为是外推。

论文式表述应报告完整管线版本、每级通过率、阈值敏感性分析与失败样本聚合统计，
而不是只给“我们的 recipe 更好”的最终 loss。

## 15. 面试备考（Interview Prep）

> 过滤/去重/重加权是数据工程面试的高频题：面试官常从「MinHash 为什么能估 Jaccard」切入，
> 追到「LSH 候选率」「DSIR vs DoReMi」「去重前先定义什么」「过滤为何不是越严越好」。
> 核心是把「数据清洗」理解成**可测量、可重放、可证伪的决策系统**，而非清洗脚本。
> 下面按「一页速览 → 高频题 → 手撕 → 追问」四层组织。

### 15.1 一页速览卡（面试前 1 分钟）

**核心主张**：原始语料到训练分布的每一步（过滤/去重/重加权）都是决策系统，每个 stage 应输出
score、decision、reason 以便重放；「越严越好」和 classifier accuracy 都是误导。

**必背数字与公式**

- MinHash 碰撞 \(\Pr[m(A)=m(B)]=J(A,B)\)，估计方差 \(\operatorname{Var}(\hat J)=J(1-J)/K\)。
- LSH 候选率 \(P_{\text{cand}}(s)=1-(1-s^r)^b\)，近似拐点 \(s^\star\approx(1/b)^{1/r}\)。
- DSIR 权重 \(w(x)=p_{\text{target}}/p_{\text{raw}}\)，有效样本量 \(\mathrm{ESS}=(\sum w)^2/\sum w^2\)。
- 典型 Jaccard 阈值约 0.8；去重常带来数倍数据压缩。

**三句话答高频**

1. MinHash 用随机排列取最小哈希，碰撞概率等于 Jaccard；LSH 分 band 控制候选规模。
2. 去重前先定义「重复」语义（URL/hash/MinHash 阈值）与删除单位（文档/段落/行）。
3. 过滤越严不一定越好，要平衡人工精度 × token 产量 × 域覆盖 × 下游 loss。

### 15.2 高频面试题与答题框架

**Q1：MinHash 为什么能估计 Jaccard？LSH 的候选率？**

- 把文档变 word n-gram shingles 集合；对随机排列/哈希取集合最小值 \(m_k(A)=\min_{a\in A}h_k(a)\)，则 \(\Pr[m_k(A)=m_k(B)]=J(A,B)\)。
- \(K\) 个 hash 的碰撞率无偏估计 Jaccard，方差 \(J(1-J)/K\)，增大 \(K\) 降方差但增成本。
- LSH 把 \(K=br\) 个 signature 分 \(b\) 个 band、每 band \(r\) 行，至少一个 band 全同即为候选：\(P_{\text{cand}}(s)=1-(1-s^r)^b\)；增大 \(b\) 提召回、增大 \(r\) 更严格。

**Q2：去重前必须先定义什么？**

- 「重复」的语义（URL / 内容 hash / MinHash 阈值）与删除单位（document / paragraph / line）。
- 语义不同结论完全不同：exact hash 只删完全一致；MinHash 删近重复；SemDeDup 删语义冗余。
- 还要定义「保留一份 vs 删除所有重复行」——本仓库 exact-line 删所有全局频次 >1 的行，会连带删合法免责声明。

**Q3：为什么过滤 + 去重同时改善 loss 与下游？**

- 低质与重复内容浪费 token 预算、推高记忆与泛化风险；去重后唯一 token 驱动收益（data-constrained scaling）。
- Lee 等证明去重显著降 perplexity 并减少 memorization；Gopher rules + 精确/模糊去重是 RefinedWeb/FineWeb 的标准组合。

**Q4：DSIR 与 DoReMi 的定位差异？**

- **DSIR**：估计 raw→target 密度比 \(w=p_{\text{target}}/p_{\text{raw}}\) 做 importance resampling（**不删除**，保留概率多样性）。
- **DoReMi**：在小 proxy 模型上用 group DRO 学域混合权重（给最差域加大权重），再用该权重训练大模型。
- 两者都改变采样分布，与「删除式」过滤/去重正交；都依赖 reference data 且需审计其来源。

**Q5：fastText 与 Gopher rules 的取舍？**

- **Gopher rules**：文档统计启发式（词数、平均词长、省略号行、字母比例），便宜、可解释，但边界跳变 + 文化/领域偏差。
- **fastText**：词/字符 n-gram 线性分类，推理快适合 web-scale 初筛；但「质量」由正负样本定义（Wikipedia 为正、随机网页为负学的可能是来源风格）。
- 实务：先便宜规则做结构初筛，再用学习式分类器，最后小模型 ablation 裁决（FineWeb 方法论）。

**Q6：exact dedup、MinHash、SemDeDup 的区别？**

- **exact**：hash 完全一致才删，只去完全重复；**MinHash**：近似 Jaccard 去近重复；**SemDeDup**：embedding 空间按 cosine 去语义冗余，约减半数据等质量。
- 代价递进：exact 最便宜、SemDeDup 需 encoder+聚类；语义去重有「语义相近但事实不同」被误删的风险。

**Q7：DSIR 的 ESS 诊断什么？ratio 爆炸怎么处理？**

- \(\mathrm{ESS}=(\sum w)^2/\sum w^2\) 衡量有效样本量；ESS 很低表示少数文档支配数据。
- \(p_{\text{raw}}\) 很小时 ratio 爆炸 → 需 smoothing、log-weight clipping；目标集被 benchmark 污染会把污染放大到训练集。

**Q8：DoReMi 的 group DRO 怎么做？**

- 先训小 proxy 模型，在其上以 group DRO 优化域权重——给当前模型表现最差的域加大权重（minimax 保住最差域）。
- 学到的域配比用于正式大模型训练，等预算下收敛更快。
- 前提：proxy 足够小（否则学权重开销超收益）、域定义合适、tokenizer/数据版本与后续训练一致。

**Q9：过滤为什么不是越严越好？**

- 提高阈值提升人工精度，却减少领域/语言覆盖；目标是 \(\max_\tau U(\tau)=\text{quality}-\lambda C_{\text{tokens}}-\gamma\,\text{distribution shift}\)。
- 每 stage 应输出 score/decision/reason，才能重放不同阈值；只优化 validation loss 可能奖励 benchmark 泄漏或风格窄化。

**Q10：LSH 的 \(b\) 和 \(r\) 怎么调？**

- \(b\) 大：召回高、候选多；\(r\) 大：更严格、候选少；拐点约 \(s^\star\approx(1/b)^{1/r}\)。
- 最终阈值仍要通过数据校准；LSH 只缩小 pair 数，之后还要算真 Jaccard 复核（候选 ≠ 重复）。

### 15.3 手撕要点（MinHash / LSH / DSIR）

面试让「推导 MinHash 无偏性」或「算 LSH 候选率」时，按公式写：

```text
MinHash: m_k(A) = min_{a∈A} h_k(a),  Pr[m_k(A)=m_k(B)] = J(A,B)
  Ĵ = (1/K) Σ 1[m_k(A)=m_k(B)],  Var(Ĵ) = J(1-J)/K

LSH (K=br, b bands × r rows):
  P_cand(s) = 1 - (1-s^r)^b,  拐点 s* ≈ (1/b)^{1/r}

DSIR: w(x) = p_target(x)/p_raw(x)
  ESS = (Σw)² / Σw²   （低 ESS = 少数文档支配数据）
```

**三个必踩坑**

1. **LSH 候选 ≠ 重复**：还要算真 Jaccard 复核，用 union-find 合并簇。
2. **Python `hash()` 不稳定**：跨进程/版本不同，必须用固定算法 + seed。
3. **split 前去重**：train/validation 切分后再去重会跨 split 泄漏，应先建近重复簇再切分。

### 15.4 高频追问与陷阱

| 追问 | 正确方向 |
| --- | --- |
| 去重能解决所有污染吗？ | 否，语义改写、翻译重复、拼接污染仍在，且不证明模型不记忆 |
| classifier accuracy 是数据质量吗？ | 否，只是正负样本定义的「风格」，精度有 CI |
| survivor 选「最早抓到」对吗？ | 可能系统偏向特定域，策略需可解释并审计来源占比 |
| 目标域含 benchmark 会怎样？ | DSIR/DoReMi 会把泄漏写进采样分布 |
| MinHash 空集合会怎样？ | 全部同 signature 聚成一簇，应先过滤空文档 |

## 16. 小结

fastText 和 Gopher 适合廉价初筛，MinHash 估计集合相似度，LSH 控制候选规模，SemDeDup
把去重推进到语义层，DSIR/DoReMi 则通过密度比与域权重改变采样分布。它们不是同一种
“质量算法”，也不能互相替代。可靠的数据 recipe 必须能解释每个阈值造成的产量、覆盖和
风险变化，并最终用固定训练设置验证——这正是 Lecture 13 数据契约在本讲的具体化。

## 参考文献

<a id="ref-1"></a>[1] C. Raffel, N. Shazeer, A. Roberts, et al. “Exploring the
Limits of Transfer Learning with a Unified Text-to-Text Transformer.”
*JMLR*, 2020. [link](https://arxiv.org/abs/1910.10683)

<a id="ref-2"></a>[2] A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov.
“Bag of Tricks for Efficient Text Classification.” *EACL*, 2017.
[link](https://arxiv.org/abs/1607.01759)

<a id="ref-3"></a>[3] J. W. Rae, S. Borgeaud, T. Cai, et al. “Scaling
Language Models: Methods, Analysis & Insights from Training Gopher.”
arXiv:2112.11446, 2021. [link](https://arxiv.org/abs/2112.11446)

<a id="ref-4"></a>[4] G. Penedo, Q. Malartic, D. Hesslow, et al. “The
RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web
Data, and Web Data Only.” *NeurIPS*, 2023.
[link](https://arxiv.org/abs/2306.01116)

<a id="ref-5"></a>[5] G. Penedo, H. Kydlíček, L. Ben allal, et al. “The
FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.”
arXiv:2406.17557, 2024. [link](https://arxiv.org/abs/2406.17557)

<a id="ref-6"></a>[6] A. Z. Broder. “On the Resemblance and Containment of
Documents.” *Compression and Complexity of Sequences*, 1997.
https://doi.org/10.1109/SEQEN.1997.666900

<a id="ref-7"></a>[7] P. Indyk, R. Motwani. “Approximate Nearest Neighbors:
Towards Removing the Curse of Dimensionality.” *STOC*, 1998.
https://doi.org/10.1145/276698.276876

<a id="ref-8"></a>[8] K. Lee, D. Ippolito, A. Nystrom, et al.
“Deduplicating Training Data Makes Language Models Better.” *ACL*, 2022.
[link](https://arxiv.org/abs/2107.06499)

<a id="ref-9"></a>[9] A. Abbas, K. Tirumala, D. Simig, et al. “SemDeDup:
Data-efficient Learning at Web-scale through Semantic Deduplication.”
arXiv:2303.09540, 2023. [link](https://arxiv.org/abs/2303.09540)

<a id="ref-10"></a>[10] N. Muennighoff, A. Rush, B. Barak, et al. “Scaling
Data-Constrained Language Models.” *NeurIPS*, 2023.
[link](https://arxiv.org/abs/2305.16264)

<a id="ref-11"></a>[11] S. M. Xie, S. Santurkar, T. Ma, and P. Liang.
“Data Selection for Language Models via Importance Resampling.” *ICML*,
2023. [link](https://arxiv.org/abs/2302.03169)

<a id="ref-12"></a>[12] S. M. Xie, H. Pham, X. Dong, et al. “DoReMi:
Optimizing Data Mixtures Speeds Up Language Model Pretraining.” *NeurIPS*,
2023. [link](https://arxiv.org/abs/2305.10429)

## 延伸阅读与复现材料

- Stanford CS336, [Lecture 14 — Filtering, Deduplication and Mixing](https://github.com/stanford-cs336/lectures/blob/main/lecture_14.py)
- [Data Curation 主题导航](../experiments/topics/data-curation.md)
- [A4 Data 官方题面](../assignments/assignment4-data/cs336_assignment4_data.pdf)
