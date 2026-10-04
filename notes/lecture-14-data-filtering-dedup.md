---
title: "Lecture 14 — Data Filtering, Deduplication & Mixing"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-05-13"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_14.py"
  - "../assignments/spring2026/assignment4-data/"
---

# Lecture 14 — 过滤、去重与重加权：把规则变成可测系统

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
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
- 把“去重越多越好”当默认：FineWeb 实测去重收益先升后降，过度去重导致性能
  恶化；阈值与去重范围（per-snapshot vs 全局）变更必须走小模型受控 ablation。
- 去污染只做精确匹配：改写/翻译型污染会逃过哈希；应叠加 n-gram/语义级检测
  （如 LLM Decontaminator）并辅以第三方评测交叉验证。

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

### Construct validity

- “质量”由正负样本定义，classifier accuracy 不是数据质量的度量；
- Jaccard 阈值是表面相似度，与“信息冗余”不一一对应；
- ESS 衡量权重集中度，不衡量分布匹配的正确性。

### Internal validity

- 阈值、去重、混合同时变更时，无法归因单一因素；
- 下游 proxy 模型过小，其偏好未必代表目标规模模型 [[5]](#ref-5)；
- 人工标注样本量小且标注者偏差存在，precision 数字有置信区间。

### External validity

- 在一个语料/语言上校准的阈值与 \(b,r\) 不外推到其他分布；
- 规则与分类器的误杀模式依赖时代（web 风格演化）；
- 本仓库 A4 在 1000-WET 样本上验证，全量行为是外推。

论文式表述应报告完整管线版本、每级通过率、阈值敏感性分析与失败样本聚合统计，
而不是只给“我们的 recipe 更好”的最终 loss。

## 面试要点速记

**高频问题与答题要点**

1. **Q：MinHash 为什么能估计 Jaccard？** 要点：k-shingle 集合经随机置换取
   最小哈希，两签名相等的概率 = Jaccard；b 带 × r 行的 LSH 用
   \(1-(1-s^r)^b\) 控 recall/precision 曲线。
2. **Q：去重前必须先定义什么？** 要点：“重复”的语义（URL/hash/MinHash 阈值）
   与删除单位（文档/段落/序列）；语义不同结论完全不同。
3. **Q：为什么过滤+去重同时改善 loss 与下游？** 要点：低质与重复内容浪费
   token 预算、推高记忆与泛化风险；Gopher rules/模型分类器 + 精确/模糊去重
   是 RefinedWeb/FineWeb 的标准组合。
4. **Q：DSIR 与 DoReMi 的定位差异？** 要点：DSIR 学 raw→target 分布的重要性
   权重做重采样（不删除）；DoReMi 用 group-DRO 学域混合权重，minimax 保住
   最差域。
5. **Q：为什么基于 PPL 的质量过滤通常保留中间段？** 要点：PPL 过低 =
   模型已学会（信息量小、边际收益低）、PPL 过高 = 多为异常点（乱码/噪声），
   两端都有害；用分位数切中间段，并以下游 ablation 验证切分位置。
6. **Q：FineWeb-edu 规模更小为何反而更强？** 要点：llama-3-70b-instruct
   给约 50 万样本按 0–5 打教育质量分、滤 <3；质量过滤 > 数量；用固定训练
   配方的受控小模型 ablation 做裁决。
7. **Q：评测集污染（contamination）怎么防？** 要点：评测集混入训练数据会让
   benchmark 虚高（“刷榜”）；对策 = 污染检测工具（如 LLM Decontaminator）
   + 消融实验 + 第三方评测；exact 匹配抓不住改写/翻译型污染。
8. **Q：语义去重的具体做法？与 MinHash 的差异？** 要点：embedding 聚类成
   N 簇 → 簇内余弦相似度高于阈值视为语义重复 → 仅保留离簇中心最近的一条；
   能捕捉改写/翻译级冗余，但成本更高且有“语义近但事实不同”的误删风险。

**必背数字**

- LSH 概率式 \(1-(1-s^r)^b\)；典型 Jaccard 阈值 0.8 级；去重常带来
  数倍数据压缩与下游增益并存。
- 工业界参照：FineWeb 15T tokens（96 快照）、Llama-3 15T、GPT-2 约 100B、
  Qwen 2.2T（去重过滤后）——主流公开语料的量级锚点。
- 工业界参照：FineWeb-edu 用 llama-3-70b-instruct 给约 50 万样本按 0–5 打
  教育质量分、滤 <3，规模显著更小仍优于 FineWeb（质量 > 数量）。
- 工业界参照：DoReMi 式先导（2.6B 小模型，中文/英文/代码/电信四域）——
  电信域 PPL 2.8→1.76、中文 3.33→2.68、英文 6.02→5.73、代码 3.66→3.31，
  通用能力不劣化。
- 工业界参照：Nemotron 数学管线——98 个 CC 快照重抓 + FineMath 分类器 +
  MinHash-LSH 去重 + LLM Decontaminator 去污染。
- 工业界参照：FineWeb 实测去重收益先升后降，过度去重导致性能恶化——
  去重阈值与范围没有“越大越好”。

## 行业现状与最新进展（2024–2026）

### 去重收益的边际递减与临界点

FineWeb 的系统实测给出了一个反直觉结论：去重对性能的提升存在临界点，越过
之后边际收益递减，**过度去重最终导致性能恶化**。因此 FineWeb 的 MinHash
去重按每次导入（per-snapshot dump）独立进行，而非跨全部 96 个快照的激进
全局去重；过滤器侧还对比了 C4 派生的选择性过滤器与自定义过滤器，用受控
小模型 ablation 裁决。工业界通行的去重层级如下：

| 层级 | 手段 | 解决的问题 | 量级成本 |
| --- | --- | --- | --- |
| 1 | URL 黑名单过滤 | 已知低质/违规来源 | 极低 |
| 2 | 语言过滤（fastText 级） | 非目标语言 | 低 |
| 3 | 哈希精确去重 | 完全重复 | 低，\(O(N)\) |
| 4 | MinHash-LSH 模糊去重 | 近重复/模板/镜像 | 中，\(O(NMK)\) |
| 5 | 语义去重（进阶） | 改写/翻译级冗余 | 高（embedding + 聚类） |

工程细节上，去重粒度本身是超参数：如以产品族而非单个产品为粒度，覆盖范围
更大、去重更彻底——粒度选择直接决定“删除语义”。

### 质量过滤的“分类器时代”

FineWeb-edu 用 llama-3-70b-instruct 给约 50 万样本按 0–5 打“教育质量”分，
过滤掉低于 3 分的文档；其规模显著小于 FineWeb，却优于 FineWeb 与其他公开
数据集——“质量 > 数量”的直接证据。同一路线还有 FineMath 分类器（数学语料
筛选）与 Nemotron 的 LLM Decontaminator（评测去污染，见下）。这一范式与
Gopher 规则互补：规则管结构与形态异常，分类器管“内容对学习者是否有价值”，
二者都需人工抽检与分组误差报告。

### 语义去重与配比优化

语义去重实践（某大模型团队）：把文档 embedding 聚类成 N 个簇，簇内计算
余弦相似度，高于阈值视为语义重复，仅保留与簇中心最近的一条，删除其余。
相比 MinHash 只捕捉表面重叠，这一层能删掉改写/同义级冗余。

配比优化实践（DoReMi 复现，某大模型团队）：基于 2.6B 小模型先导实验，按
多 domain PPL 迭代调整中文/英文/代码/电信四域权重，结果如下（量级参照）：

| 域 | 先导 PPL（前→后） | 说明 |
| --- | --- | --- |
| 中文 | 3.33 → 2.68 | 通用域同步改善 |
| 英文 | 6.02 → 5.73 | 通用域同步改善 |
| 代码 | 3.66 → 3.31 | 通用域同步改善 |
| 电信 | 2.8 → 1.76 | 目标域大幅改善，通用能力不劣化 |

其数据构成示例：ICT 领域数据中文 9B / 英文 10B tokens；通用数据中文 10B /
英文 15B / 代码 15B。核心思想与 DoReMi 一致：小模型学权重、大模型吃配比，
先导成本远低于直接在大模型上试错。

### 去污染与评测公正

Nemotron 数学管线给出了领域语料的完整参照链路：收集数学 URL → 从 98 个
CC 快照重抓 HTML → lynx 保留页面布局 → Phi-4 归一化 LaTeX → FineMath
分类器过滤 → MinHash-LSH 模糊去重 → LLM Decontaminator 去污染。评测集
混入训练数据会造成 benchmark 虚高（“刷榜”），使模型对比结论失效；对策 =
污染检测工具 + 消融实验 + 第三方评测，且去污染应放在管线末端以覆盖全部
上游环节引入的污染。

**对本讲学习者的启示**：2024 年后的行业共识是“质量与唯一性优先于原始
规模”。FineWeb/FineWeb-edu 证明受控小模型 ablation 是裁决一切数据决策的
最高法院；去重与过滤都存在“过犹不及”的临界点；语义去重、PPL 驱动配比与
去污染已从可选项变成标配。学习本讲时应把每个阈值都视为待验假设，而不是
工程默认值，并习惯性地问“这个决策被什么实验支撑”。

## 大厂面试真题与答题框架

以下为高频面试题（公开面经风格），覆盖本讲核心考点。

**题目 1：讲一下 MinHash-LSH 的原理，b/r 参数怎么选？**
- 考点：MinHash 无偏性、LSH 候选率公式、参数与阈值的定量关系。
- 答题框架：1) 定义 Jaccard \(J=|A\cap B|/|A\cup B|\)；2) 最小哈希碰撞概率
  = Jaccard；3) \(K\) 个哈希估计，方差 \(J(1-J)/K\)；4) 分 \(b\) 带 ×
  \(r\) 行，候选率 \(1-(1-s^r)^b\)；5) 拐点 \(s^\star\approx(1/b)^{1/r}\)，
  按目标相似度校准 b/r，LSH 之后必须用真 Jaccard 复核再 union-find 合并。
- 加分项：FineWeb 实测去重收益先升后降，故按 per-snapshot 独立去重；
  工程上监控最大 bucket、候选 pair 数与 peak RSS。
- 踩坑：把 LSH 候选直接判重不做复核；空文档 signature 全同聚成一簇；
  survivor 永远取“最早抓取”造成来源偏差。

**题目 2：精确、模糊、语义去重分别解决什么？如何分层？**
- 考点：三级去重的语义差异与工业层级。
- 答题框架：1) 哈希精确去重删完全重复，\(O(N)\) 最便宜；2) MinHash-LSH
  删近重复/模板；3) 语义去重（聚类 + 簇内余弦阈值，保留离簇中心最近样本）
  删改写级冗余；4) 顺序从便宜到贵，先删量大的；5) 前置 URL 黑名单与语言
  过滤减少后级输入量。
- 加分项：粒度选择影响彻底程度（按产品族而非产品为粒度，去重更彻底）；
  每级记录 score/decision/reason/version 以便重放。
- 踩坑：一上来就语义去重浪费算力；跨层阈值不统一导致审计困难。

**题目 3：为什么基于 PPL 的过滤通常保留中间段？**
- 考点：PPL 作为质量信号的双向失效模式。
- 答题框架：1) PPL 过低 = 模型已学会，信息量小、边际收益低；2) PPL 过高 =
  多为乱码/异常点；3) 两端都有害，保留中间段；4) 用分位数（如 20%–90%）
  切分并做下游 ablation 验证。
- 加分项：PPL 依赖参照模型选择，跨域需重新校准；与分类器过滤互补而非替代。
- 踩坑：把 PPL 单调当“越低质量越高”；用同一模型既打分又验证造成循环论证。

**题目 4：描述 DoReMi 的思路。预算有限时怎么落地？**
- 考点：group-DRO 域权重学习与 proxy 范式。
- 答题框架：1) 训练小 proxy 模型；2) group DRO 给当前最差域加权，学到域
  权重；3) 用该权重训练正式大模型，等预算下收敛更快；4) 落地参照：2.6B
  小模型先导 + 中/英/代码/电信四域 PPL 监控，电信 2.8→1.76、通用不劣化。
- 加分项：权重学习与正式训练之间不能更换 tokenizer/数据版本；应报告
  learned weights 与初始配比的差异。
- 踩坑：proxy 太大导致“学权重”成本反噬；域定义过细导致权重震荡。

**题目 5：去重与数据多样性的张力怎么权衡？**
- 考点：过度去重的危害与临界点意识。
- 答题框架：1) 重复 token 边际价值随 epoch 衰减，去重有正收益；2) 但
  FineWeb 实测收益先升后降，过度去重性能恶化；3) 阈值与去重范围变更必须
  走小模型受控 ablation；4) 按文档类型/语言分组审计误杀与覆盖变化。
- 加分项：引用 data-constrained scaling（唯一 token 主导收益）与 FineWeb
  实测两条证据线。
- 踩坑：把“去重越多越好”当默认；只报最终 loss 不报多样性/覆盖指标。

**题目 6：评测集污染（contamination）为什么危险？怎么防？**
- 考点：去污染意识与工程防线。
- 答题框架：1) 评测集混入训练数据 → benchmark 虚高（“刷榜”），对比结论
  失效；2) 防线一：管线末端加去污染（Nemotron 数学管线在 MinHash-LSH 后
  接 LLM Decontaminator）；3) 防线二：消融实验隔离污染影响；4) 防线三：
  第三方评测交叉验证。
- 加分项：指出 exact 匹配抓不住改写/翻译型污染，需 n-gram/语义级检测。
- 踩坑：只在训练前做一次去污染，之后新增数据不复查。

**题目 7：FineWeb-edu 规模更小却更强，为什么？对数据策略有何启示？**
- 考点：质量 > 数量；LLM-as-judge 打分的过滤器设计。
- 答题框架：1) llama-3-70b-instruct 给约 50 万样本按 0–5 打教育质量分，
  滤 <3；2) 规模显著更小仍优于 FineWeb 与其他公开数据集；3) 启示：token
  预算应投向高质量子集，“质量”的定义即打分 prompt 的定义，需人工抽检；
  4) 用固定训练配方的受控 ablation 验证。
- 加分项：讨论 LLM 打分的偏差与标注成本（50 万样本量级）；与 RefinedWeb
  拒绝分类器路线的对照。
- 踩坑：把分类器分数当校准概率；忽视打分模型自身对“教育价值”的偏差。

## 系统设计题

**设计题 1：为 15T tokens 级 web 语料设计“过滤 + 去重 + 配比”分布式管线（含 PII 与去污染）**

- 需求澄清：目标 token 预算（FineWeb 量级 15T / 96 快照；参照 Llama-3
  15T、Qwen 2.2T 去重过滤后）；语言范围；PII 合规等级；评测集清单与
  去污染要求；可用算力、工期与吞吐 SLA。
- 规模估算：15T tokens 对应数十亿文档量级；MinHash 构造 \(O(NMK)\)
  （\(K\approx128\)），签名存储约 128 B/文档 × N；LSH 按 band shuffle，
  需估算最大 bucket 与候选 pair 数；按 per-snapshot 去重可复用 FineWeb
  的成本结构。
- 架构：Spark/Ray 分 stage：URL 黑名单 → fastText 语言过滤 → Gopher
  结构规则 → PII 掩码 → 质量分类器打分（保留 score）→ 哈希精确去重 →
  MinHash-LSH 模糊去重（per-snapshot）→（进阶）聚类 + 余弦语义去重 →
  DoReMi 式 2.6B proxy 学域配比 → 去污染（评测集 n-gram/语义检测）→
  tokenizer + 固定预算 ablation。每 stage 落 score/decision/reason/version
  审计字段，支持阈值重放。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 权衡 |
| --- | --- | --- | --- |
| 去重范围 | per-snapshot（FineWeb） | 全局跨快照 | A 保留多样性、规避过度去重；B 更彻底但实测可能性能恶化 |
| 质量过滤 | 规则（Gopher） | LLM 打分分类器 | A 便宜可解释；B 贴近目标但贵且有偏差 |
| 配比来源 | 人工经验 | proxy 模型学习（DoReMi 式） | A 上手快；B 等预算更优但有先导成本 |
| 去污染 | exact 匹配 | 语义级检测 | A 快；B 能抓改写型污染 |

- 评测方案：小模型受控 ablation（固定训练配方比 validation loss 与下游）；
  人工 precision 分语言/文档类型抽样；去重后多样性/域覆盖报表；去污染
  前后 benchmark 对比与评测集命中率检查（应趋近 0）。
- 追问预案：热门模板形成超大 bucket → 限制桶容量 + 二次复核 + 拆 band；
  小语种被规则误杀 → 分组通过率监控 + 单独阈值；增量更新 → 新快照对增量
  + 采样旧集做去重；PII 误掩码 → 掩码版原文双写留审计。

**设计题 2：为领域增训（以电信/ICT 为例）设计“通用能力不劣化”的数据策略**

- 需求澄清：领域目标（电信域 PPL 显著下降）；硬约束（通用 benchmark 不
  劣化）；先导实验预算；中/英/代码三域通用数据基线配比。
- 规模估算：参照某大模型实践——ICT 领域数据中文 9B / 英文 10B tokens；
  通用数据中文 10B / 英文 15B / 代码 15B；先导模型 2.6B，多轮短训迭代
  权重。
- 架构：领域语料管线（URL 收集 → CC 快照重抓 → lynx 保留布局 → 文本
  归一化 → 领域分类器（FineMath 式）→ MinHash-LSH 去重 → 去污染）→
  2.6B proxy 上按四域 PPL 迭代 domain weight（DoReMi 式 group-DRO）→
  目标配比下训练正式模型 → 领域/通用双轨评测。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 权衡 |
| --- | --- | --- | --- |
| 配比来源 | 固定人工配比 | PPL 驱动迭代 | A 简单；B 实测电信 2.8→1.76 且通用不劣化 |
| 领域数据粒度 | 按产品 | 按产品族 | 族级范围更大、去重更彻底 |
| proxy 规模 | 1B 级 | 2.6B 级 | 更小更快，但偏好可能不外推到大模型 |
| 评测方式 | 自建 benchmark | 自建 + 第三方 | 前者可能污染虚高，后者更可信但有泄露顾虑 |

- 评测方案：四域 PPL 追踪（量级参照：中文 3.33→2.68、英文 6.02→5.73、
  代码 3.66→3.31、电信 2.8→1.76）；通用 benchmark 回归（不劣化判据 +
  置信区间）；领域下游任务集；第三方评测防刷榜。
- 追问预案：proxy 与大模型偏好不一致 → 大模型上小样本复验配比；tokenizer
  变更 → 已学权重作废重跑；域定义过粗/过细 → 做粒度 sweep 并观察权重
  震荡。

**设计题 3：设计数学领域语料管线（Nemotron/FineMath 式），含去重与去污染**

- 需求澄清：目标规模与质量等级；LaTeX/公式保真要求；评测集去污染等级。
- 规模估算：URL 种子池 → 98 个 CC 快照重抓 HTML（Nemotron 实践量级）；
  存储按快照膨胀系数估算；FineMath 分类器打分与 MinHash-LSH 计算量随
  过滤后留存率下降。
- 架构：收集数学 URL → 98 个 CC 快照重抓 HTML → lynx 保留页面布局 →
  Phi-4 归一化 LaTeX → FineMath 分类器过滤 → MinHash-LSH 模糊去重 →
  LLM Decontaminator 去污染 → 固定预算小模型 ablation 定稿。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 权衡 |
| --- | --- | --- | --- |
| 正文抽取 | lynx 保布局 | 纯文本抽取 | A 保留公式/结构；B 便宜但损失数学内容 |
| 去污染时机 | 去重前 | 管线末端（去重后） | 末端做（Nemotron 顺序）可覆盖全部上游污染 |
| 分类器 | 规则关键词 | LLM 打分（FineMath 式） | A 便宜；B 区分度更高但推理成本大 |

- 评测方案：数学 benchmark 前后对比 + 通用能力回归；污染检测报告（评测集
  命中率应为 ~0）；小模型 ablation 比较管线变体（如 lynx vs 纯文本抽取）。
- 追问预案：分类器把竞赛题当高质样本（潜在污染源）→ 评测集黑名单前置；
  LaTeX 归一化不一致 → 抽样人工核对渲染结果。

## 代码实现题

**代码题 1：MinHash 签名 + LSH 分桶去重（骨架）**

- 题目：实现 minhash 签名、LSH 分桶、候选复核与 union-find 合并；输入
  文档 dict，输出保留的 survivor 集合。
- 考察点：规范化与 shingling；\(K/b/r\) 与阈值的定量关系；LSH 候选必须用
  真 Jaccard 复核；确定性合并顺序。

```python
import re
import hashlib
from collections import defaultdict

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def shingles(text: str, n: int = 5) -> set:
    words = normalize(text).split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

def minhash_sig(sh: set, k: int = 128, seed: int = 0) -> tuple:
    sig = []
    for i in range(k):
        h = hashlib.blake2b(digest_size=8, person=str(seed + i).encode())
        sig.append(min(int.from_bytes(h(s.encode()).digest(), "little") for s in sh))
    return tuple(sig)

def lsh_keys(sig: tuple, b: int = 16):
    r = len(sig) // b
    for band in range(b):
        yield (band, sig[band * r:(band + 1) * r])

def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)

def minhash_dedup(docs: dict, threshold: float = 0.8,
                  k: int = 128, b: int = 16) -> set:
    sh_by_id, buckets = {}, defaultdict(list)
    for doc_id, text in docs.items():
        sh = shingles(text)
        if not sh:                       # 空文档不进签名，避免聚成一簇
            continue
        sh_by_id[doc_id] = sh
        for key in lsh_keys(minhash_sig(sh, k), b):
            buckets[key].append(doc_id)

    parent = {d: d for d in sh_by_id}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    checked = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, c = members[i], members[j]
                if (a, c) in checked:
                    continue
                checked.add((a, c))
                # LSH 只给候选；必须用真 Jaccard 复核再合并
                if jaccard(sh_by_id[a], sh_by_id[c]) >= threshold:
                    parent[find(a)] = find(c)

    keep = {}
    for doc_id in sh_by_id:              # 每簇保留一个确定性 survivor
        root = find(doc_id)
        if root not in keep or doc_id < keep[root]:
            keep[root] = doc_id
    return set(keep.values())
```

- 验收标准：对注入完全重复/近重复的合成集，召回 ≥ 95%、误删 ≤ 1%；
  空文档不被误并；同输入多次运行结果一致；\(b=16, r=8\) 时拐点
  \((1/16)^{1/8}\approx0.76\)，与阈值 0.8 匹配；报告候选 pair 数与最大
  bucket 大小。

**代码题 2：基于小模型 PPL 的质量过滤 + DoReMi 式域配比搜索脚本**

- 题目：给定 proxy 模型在候选文档上的 loss 与各域验证 loss，实现 1) PPL
  中间段过滤；2) group-DRO 风格的域权重迭代更新。
- 考察点：PPL = exp(mean NLL)；分位数过滤的双向失效理解；最差域加权、
  clip、归一化；先导-放大两阶段流程与可复现性。

```python
import numpy as np

def ppl_filter(losses: np.ndarray, q_low: float = 0.2,
               q_high: float = 0.9) -> np.ndarray:
    """PPL 过低 = 模型已学会（信息量小）；过高 = 异常点 → 保留中间段。"""
    ppl = np.exp(losses)
    lo, hi = np.quantile(ppl, [q_low, q_high])
    return (ppl >= lo) & (ppl <= hi)

def doremi_update(weights: dict, domain_loss: dict,
                  lr: float = 0.2, cap: float = 0.5,
                  floor: float = 0.02) -> dict:
    """简化 group-DRO：最差域（loss 最大）加权、最优域减权，clip 后归一化。"""
    worst = max(domain_loss, key=domain_loss.get)
    best = min(domain_loss, key=domain_loss.get)
    w = dict(weights)
    w[worst] *= 1 + lr
    if best != worst:
        w[best] *= 1 - lr
    w = {d: min(max(v, floor), cap) for d, v in w.items()}
    s = sum(w.values())
    return {d: v / s for d, v in w.items()}

# 先导流程（参照 2.6B proxy 四域实践：中文/英文/代码/电信）
# weights = {"zh": 0.25, "en": 0.25, "code": 0.25, "telecom": 0.25}
# for step in range(n_steps):
#     proxy.train_one_round(weights)            # 2.6B 小模型短训
#     domain_loss = proxy.eval_domain_nll()     # 各域验证平均 NLL
#     weights = doremi_update(weights, domain_loss)
#     print({d: float(np.exp(l)) for d, l in domain_loss.items()})
#     # 监控量级参照：电信 PPL 2.8→1.76、中文 3.33→2.68，通用域不劣化
```

- 验收标准：`ppl_filter` 在合成双峰 loss 分布上恰好删除两端分位样本；
  `doremi_update` 输出权重和为 1 且全部落在 \([floor, cap]\)；对“电信域
  loss 最高”的输入，若干轮后 telecom 权重单调上升并触顶 cap；固定种子
  全程可复现。

## 15. 小结

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
- [A4 Data 官方题面](../assignments/spring2026/assignment4-data/cs336_assignment4_data.pdf)
- [Scaling Data-Constrained Language Models（Muennighoff et al., 唯一 token 主导收益的原始证据）](https://arxiv.org/abs/2305.16264)（访问日期 2026-10-04）
- [The Llama 3 Herd of Models（15T tokens 级数据管线的工业参照）](https://arxiv.org/abs/2407.21783)（访问日期 2026-10-04）
- [DeepSeek-V3 Technical Report（预训练数据管线与配比的工程参照）](https://arxiv.org/abs/2412.19437)（访问日期 2026-10-04）
- [Stanford CS336 课程主页](https://cs336.stanford.edu)（访问日期 2026-10-04）
