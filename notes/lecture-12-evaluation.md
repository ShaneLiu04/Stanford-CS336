---
title: "Lecture 12 — Evaluation"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-05-06"
lecturer: "Percy Liang"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_12.py"
  - "https://crfm.stanford.edu/helm/"
  - "https://arxiv.org/abs/2403.04132"
---

# Lecture 12 — Evaluation：从抽象能力到可信测量

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述，非课程提交
- 适用对象：自学者、评估平台工程师、LLM Evaluation 研究者

## 摘要

评估不是在静态题集上算一个准确率，而是把抽象能力目标转化为可审计测量的全过程。
本文沿"目标构念 → 任务/用户分布 → 推理协议 → grader → 统计量 → 决策"的测量链，
系统推导 perplexity/bits-per-byte、multiple-choice 三种 adaptation、pass@k 无偏估计、
pairwise win rate 与 Bradley–Terry/Elo 排名等核心指标；随后给出配对与聚类不确定性
（Wilson 区间、McNemar 检验、cluster bootstrap）、多重比较与选择偏差的规范流程。
针对 benchmark contamination，建立从数据侧 n-gram/MinHash 匹配到模型侧
exchangeability、选项扰动、time-travel 探针的分层检测与缓解框架。对 LLM-as-a-judge，
归纳 single-answer/pairwise/CoT 概率加权/长度控制回归四类协议，审计 position、
verbosity、self-preference 等系统性偏差并给出人类校准方法。最后讨论三层效度
（reliability、construct validity、ecological validity）、benchmark 饱和与退化、
agent 评估的成本控制，以及失败分析、报告规范与伦理边界。核心论点：一个分数是
\((\text{model},\text{data},\text{protocol},\text{grader})\) 四元组的函数；
不存在唯一真实 benchmark，只有是否适配具体决策、是否具备足够 reliability 与
validity 的评估组合。

**关键词：** Language Model Evaluation；Benchmark；Perplexity；pass@k；
Bradley–Terry；LLM-as-a-Judge；Data Contamination；Uncertainty
Quantification；Construct Validity；Agentic Evaluation

## 本文贡献

1. 把评估形式化为六环测量链，并区分方法/模型/系统三类评估对象；
2. 统一推导 PPL/bpb、MC 协议、pass@k、BT/Elo 的定义、shape 与协议敏感性；
3. 给出配对/聚类/分层不确定性、McNemar 检验与多重比较的可执行流程；
4. 建立 contamination 的分类学、五类检测手段与缓解策略及其证据边界；
5. 归纳 LLM-judge 协议族与偏差诊断，并给出人类盲测校准协议；
6. 将效度三层、饱和/退化、成本控制与失败分析整合为 evaluation portfolio
   与报告规范。

## 学习目标

- 将"模型好"拆解为目标构念、任务分布、推理协议、grader 与统计量；
- 为 accuracy、win rate、perplexity 报告适当的不确定性；
- 理解 pass@k 估计量的无偏性推导与数值实现要点；
- 比较 BT 与 Elo 两种排名模型的信息利用与失败模式；
- 识别 benchmark contamination、重复调参和饱和带来的虚假进步；
- 设计并审计 human/LLM judge，处理 position、length、self-preference 偏差；
- 为 agent 系统设计成本受控的评估并正确归因分数来源。

## 先修知识

- Lecture 09/11：以 loss 为核心指标的能力测量及其外推边界。
- Lecture 04：架构差异（MQA/GQA、MoE）如何影响评测协议设计。
- 统计基础：置信区间、假设检验、多重比较校正。

## 相关工作与问题演进

静态题集范式从 GLUE/SuperGLUE 发展到覆盖 57 个学科的 MMLU [[1]](#ref-1)、
汇聚众包困难任务的 BIG-bench [[2]](#ref-2) 和研究生级、搜索引擎难以解答的
GPQA [[3]](#ref-3)。这一范式提供了标准化、廉价的 measurement，但也带来
饱和、格式 exploit 和污染问题。HELM [[4]](#ref-4) 系统化了 holistic
evaluation：把 scenario、adaptation（生成/选择概率/排序）与 metric 解耦，
并同时报告 accuracy、calibration、robustness、fairness、bias、toxicity 和
efficiency 七类 desiderata，是"评估组合"思想的代表实现。

人类偏好评估从众包 A/B 测试演化为持续运行的 Chatbot Arena [[5]](#ref-5)，
其用 Bradley–Terry 变体替代朴素 Elo 排名；相关分析指出 Elo 对对局顺序与
噪声敏感 [[6]](#ref-6)。自动化评分方面，MT-Bench 提出 LLM-as-a-judge 的
single-answer 与 pairwise 协议 [[7]](#ref-7)，G-Eval 用 CoT 评分并按 token
概率加权 [[8]](#ref-8)，Length-Controlled AlpacaEval 用回归显式控制长度
混淆 [[9]](#ref-9)。随后工作揭示了 position bias [[10]](#ref-10) 与
self-preference [[11]](#ref-11) 等系统性偏差。

评估有效性研究沿三条线展开：contamination 的检测与证明
[[12]](#ref-12)[[13]](#ref-13)[[14]](#ref-14)；benchmark 老化与过拟合的
实证——ImageNet/CIFAR 复测下降 [[15]](#ref-15)、GSM8K 与同分布新题 GSM1k
的双位数差距 [[16]](#ref-16)、按月轮换题目的 LiveBench [[17]](#ref-17)；
以及 agent 评估，如以单元测试为 grader 的 SWE-bench [[18]](#ref-18) 和
主张成本受控比较的 AI Agents That Matter [[19]](#ref-19)。统计实践方面，
Miller 系统整理了为 eval 加 error bar 的配对与聚类方法 [[20]](#ref-20)。

## 1. Evaluation 是测量设计

评估链条是：

\[
\text{目标构念}
\rightarrow\text{任务/用户分布}
\rightarrow\text{样本}
\rightarrow\text{推理协议}
\rightarrow\text{grader}
\rightarrow\text{统计量}
\rightarrow\text{决策}.
\]

任何一环改变，分数含义都会改变。"模型 A 80 分"在缺少 dataset version、
prompt、shots、sampling、工具、预算和 grader 时不是可复现实验。一个分数应
理解为四元组的函数：

\[
\text{score}=f(\text{model},\text{data},\text{protocol},\text{grader}),
\]

四者任一变化都是在测量另一个量。

先问清评估对象：

- **方法**：固定数据/算力/协议，比较算法；
- **模型**：允许各自训练 recipe，但推理协议固定；
- **系统/agent**：模型、scaffold、tools、retrieval、重试和预算整体比较。

agent benchmark 的成功不能自动归因于 base model；同理，比较两个模型时应固定
协议，而不是让每个团队自带 prompt 与 decoding。

### 1.1 协议要素清单

HELM 把"怎么考"显式化为 adaptation [[4]](#ref-4)：同一 scenario 可用
生成式（自由文本）、loglikelihood 式（对每个选项打分）或排序式作答。报告
评估时至少固定并公开：

- dataset 名称、版本、split 与采样方式；
- prompt template、system prompt 与 few-shot examples；
- decoding（temperature、top-p、max tokens、stop 序列）；
- 工具与环境（检索、代码执行、重试上限、时间/token 预算）；
- grader 类型（规则/参考答案/judge 模型及版本）与解析失败的处理。

## 2. 指标、shape 与关键推导

### 2.1 Perplexity 与 bits-per-byte

对 token 序列 \(x_{1:T}\)，平均 negative log-likelihood 为

\[
\bar\ell=-\frac1T\sum_{t=1}^{T}\log p(x_t\mid x_{<t}),
\qquad
\operatorname{PPL}=e^{\bar\ell}.
\]

实现上输入 logits shape 为 \([B,T,V]\)、labels 为 \([B,T]\)：先 `gather`
每个 label 的 log-prob 得到 \([B,T]\)，再用 mask 聚合。PPL 平滑、适合
scaling 分析（A3 的 validation loss 即此指标），但强依赖 tokenizer——
不同 tokenizer 的 token 数不同，per-token PPL 不可直接比较。跨 tokenizer
应换算为每字节比特数：

\[
\operatorname{bpb}
=\frac{\bar\ell}{\ln 2}\cdot\frac{T}{B_{\text{bytes}}},
\]

其中 \(B_{\text{bytes}}\) 为文本总字节数。对 prompt-response 任务应只在
response mask 上计算 conditional PPL，避免 prompt 长度支配指标；报告时
必须注明是否含 prompt、mask 规则与 tokenizer revision。

### 2.2 Accuracy 与 multiple-choice 协议

\[
\hat p=\frac1n\sum_{i=1}^{n}\mathbf{1}[\hat y_i=y_i].
\]

multiple-choice 有三种常用 adaptation，测的不是同一个量：

1. **生成式**：模型生成选项字母/文本，再解析匹配；
2. **choice log-prob**：对每个选项计算条件 log-likelihood，取 argmax
   （或按长度归一化的 PMI 变体）；
3. **排序式**：比较选项间相对似然。

不同协议可能改变模型排名，且模型对选项顺序本身高度敏感
[[21]](#ref-21)——这种敏感性既是混淆，也是污染探针（§4.2）。协议必须在
报告前预注册，不能事后挑对自己有利的作答方式。

### 2.3 pass@k 的无偏估计

对代码等可验证任务，设单次采样通过率为 \(p\)，定义

\[
\operatorname{pass@k}=1-(1-p)^k.
\]

实践中对每题采样 \(n\) 次（\(n\ge k\)），其中 \(c\) 次通过。Codex 给出的
无偏估计为 [[22]](#ref-22)：

\[
\boxed{\;
\widehat{\operatorname{pass@k}}
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
\;}
\]

**无偏性推导**：从 \(n\) 个样本中不放回均匀抽取 \(k\) 个，全部未命中的
概率是超几何比值 \(\binom{n-c}{k}/\binom{n}{k}\)；对
\(c\sim\mathrm{Binomial}(n,p)\) 取期望，由 Vandermonde 恒等式可得

\[
\mathbb{E}_c\!\left[\frac{\binom{n-c}{k}}{\binom{n}{k}}\right]=(1-p)^k,
\]

故估计量期望恰好等于真实 \(\operatorname{pass@k}\)。朴素替代
\(1-\binom{n-k}{c}/\binom{n}{c}\) 之类则不必无偏。实现上应在 log 空间
计算组合数避免溢出，并对全部题目平均。pass@k 随 \(k\) 递增，刻画"多样
性/覆盖"；pass@1 刻画单次质量——两者可以背离，应同时报告多个 \(k\)。

### 2.4 Pairwise win rate、Bradley–Terry 与 Elo

对 A/B 响应记录 win/tie/loss，简单 win rate 可把 tie 记 0.5。多模型
不平衡对阵时，Bradley–Terry（BT）把每个模型参数化为潜在分数
\(s_i\)：

\[
P(A\succ B)=\sigma(s_A-s_B),
\]

并对全部对局做 MLE。BT 能复用传递性信息：A 胜 B、B 胜 C 的证据会部分
传递给 A–C 比较，从而比只看直接对阵更稳健。

Elo 是另一种参数化：预期得分
\(E_A=1/\big(1+10^{(R_B-R_A)/400}\big)\)，每局后按
\(R_A\leftarrow R_A+K(S_A-E_A)\) 在线更新。两者差异值得强调：

| 维度 | Bradley–Terry | Elo |
|---|---|---|
| 估计方式 | 离线 MLE，可重拟合 | 在线序贯更新 |
| 顺序敏感性 | 无（对局集合内等价） | 依赖对局顺序 |
| 不确定性 | 可给 Fisher/bootstrap CI | 需额外处理 |
| 平局/tie | 可加 tie margin 参数 | 需特殊记分 |

Chatbot Arena 用带 tie margin 与风格控制项的 BT 变体，并报告 bootstrap
置信区间；其分析同时指出朴素 Elo 对出场顺序、参赛集合变化和噪声更敏感
[[5]](#ref-5)[[6]](#ref-6)。应按 prompt 或 user cluster 聚类，不能把
同一 prompt 的重复判断当独立样本。

### 2.5 Calibration

正确率之外的置信度维度：模型给出的概率应与经验频率一致。期望校准误差

\[
\operatorname{ECE}
=\sum_{b=1}^{B}\frac{n_b}{n}\,
\big|\operatorname{acc}(b)-\operatorname{conf}(b)\big|
\]

将预测按置信度分桶，比较每桶平均置信度与实际正确率（reliability
diagram 的数值摘要）。校准对部署决策（阈值截断、selective prediction、
abstain）至关重要，HELM 将其列为与 accuracy 并列的第一类指标
[[4]](#ref-4)。

## 3. 不确定性：分数不是事实常数

### 3.1 区间估计

二元指标的标准误近似为

\[
\operatorname{SE}(\hat p)
=\sqrt{\frac{\hat p(1-\hat p)}{n}}.
\]

小样本、\(\hat p\) 接近 0/1 或需要严格覆盖时，用 Wilson 区间：

\[
\frac{\hat p+\frac{z^2}{2n}
\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}
{1+\frac{z^2}{n}}.
\]

分层数据（多领域 benchmark、重复采样）更适合 cluster bootstrap：点估计
\(+\) 95% 区间 \(+\) 样本量是报告的最小集合 [[20]](#ref-20)。

### 3.2 配对比较

两个模型在同一批题上的比较是配对实验。二元结果的 McNemar 检验只使用
答案不同的题目：令 \(b\) 为 A 对 B 错、\(c\) 为 A 错 B 对，

\[
\chi^2=\frac{(|b-c|-1)^2}{b+c}\;\sim\;\chi^2_1,
\]

\(b+c\) 较小时改用精确二项检验。对连续或复杂指标，paired bootstrap 对
每题差值 \(\Delta_i\) 重采样：

\[
\bar\Delta^{(r)}=\frac1n\sum_{i}\Delta_{i_r},
\qquad r=1,\dots,R,
\]

取分位数区间。配对设计消除了题目难度的共同方差，通常远比两个独立区间
更有功效——"A 的 CI 与 B 的 CI 相交"不等于"无法区分 A 与 B"。

### 3.3 应按什么单位 bootstrap

- 每题独立：重采样 examples；
- 同一题多 seeds/samples：重采样 prompt cluster，再在 cluster 内处理重复；
- 多领域 benchmark：stratified bootstrap，保留领域权重；
- pairwise judge：按 prompt/user cluster，而不是按每个 vote；
- agent task：task 是单位，trajectory 重试不是新任务。

错误选择重采样单位是区间过窄的最常见原因，效果等同于假装相关样本独立
（对照 Lecture 11 的 tier-cluster bootstrap，问题结构相同）。

### 3.4 多重比较与选择偏差

如果尝试 50 个 prompt/template 后只报告最好一个，普通 CI 不包含这一步
选择。应：

- 预注册主指标与 protocol；
- 将开发集用于调 prompt，最终集只评一次；
- 报告尝试次数与 sensitivity；
- 多 benchmark 时区分 primary 与 exploratory。

leaderboard 反复提交并按 test 分数选 checkpoint，是同一问题的平台化版本：
每次提交都是一次对 held-out 集的查询，隐性地把测试集拉进了选择闭环。

## 4. Contamination：测试集进入了训练闭环

### 4.1 分类学

Contamination 不只等于 exact text overlap：

- 原题/答案出现在 pretraining 或 post-training；
- paraphrase、翻译、题解、论坛讨论泄漏；
- benchmark 被用来选择数据、prompt、checkpoint 或模型；
- grader 的参考答案泄漏到被评模型；
- leaderboard 反复提交造成隐性 test-set overfitting。

### 4.2 检测

1. **数据侧匹配**：exact hash、normalized n-gram（GPT-3 使用 13-gram
   级别的重叠检测 [[1]](#ref-1)）、MinHash/LSH、embedding 检索。可给出
   覆盖率报告，但召回受 paraphrase/翻译限制。
2. **模型侧推断**：
   - verbatim completion：让模型补全题干，异常高的逐字复现是记忆迹象，
     与训练数据提取攻击同源 [[23]](#ref-23)；
   - exchangeability 检验：构造字典序等"可预测排列"，若模型对基准顺序
     的似然显著高于随机打乱，可给出 p 值上界的污染证据 [[12]](#ref-12)；
   - 选项顺序扰动：交换 MC 选项后排名翻转率异常 [[21]](#ref-21)；
   - time-travel 探针：要求模型补全"数据集发布之后才存在"的私有变体
     信息 [[13]](#ref-13)。
   模型侧信号只能提供迹象，不能证明训练数据来源。
3. **对照构造**：按同分布重新出题（GSM1k [[16]](#ref-16)）、按时间边界
   切分（LiveBench [[17]](#ref-17)），比较新旧题分数差。

### 4.3 缓解

- **fresh evals**：按时间或构造原则出新题；但网页复制会破坏时间边界，
  公开后仍会进入后续训练闭环；
- **private/canary evals**：控制访问、轮换题库；注意内部泄漏与外部效度
  损失；
- **去污染训练**：用上述匹配器在训练管线中过滤（对照 Lecture 14）；
- **报告 overlap**：按 benchmark/category 给覆盖率，并分别报告去污染
  分数与全量分数。

"没有发现 overlap"不等于"没有 contamination"；任何检测器都有召回率边界。

### 4.4 实证证据

复测显示经典视觉 benchmark 在多年调参后新复测集上普遍下降数个百分点
[[15]](#ref-15)；GSM8K 与按同分布新造的 GSM1k 之间，部分模型出现最高
达两位数百分比的差距，且差距与训练数据中 GSM8K 式内容暴露度相关
[[16]](#ref-16)。Sainz et al. 进一步显示，仅通过发布渠道（arXiv 摘要等）
即可检测到 NLP 基准的泄漏 [[14]](#ref-14)。饱和与污染常常同时发生：
分数逼近上限时，既可能是能力溢出，也可能是记忆与格式 exploit。

## 5. LLM-as-a-Judge

开放式回答没有唯一 reference，pairwise judge 往往比绝对打分稳定。

### 5.1 协议族

| 协议 | 输入 | 输出 | 代表 |
|---|---|---|---|
| single-answer grading | 题目+回答+rubric | 分数/短评 | MT-Bench [[7]](#ref-7) |
| pairwise comparison | 题目+两个匿名回答 | A/B/tie | MT-Bench、Arena [[5]](#ref-5) |
| CoT 概率加权 | 题目+回答+评分 CoT | 分数 token 概率加权 | G-Eval [[8]](#ref-8) |
| 回归控制 | 题目+回答+基线回答 | 控制混淆后的胜率 | AlpacaEval-LC [[9]](#ref-9) |

G-Eval 用"先生成评分理由再打分"的 CoT 提升与人类相关性，并以分数 token
的概率分布取加权平均，得到连续分数 [[8]](#ref-8)。AlpacaEval-LC 对 judge
偏好显式回归长度项，报告"若长度相等时的胜率"，是处理混淆的范式转移：
从"事后报告偏差"到"估计量内消除偏差" [[9]](#ref-9)。

### 5.2 可审计 judge protocol

一个可审计的 judge protocol 应包含：

- 明确任务定义与 rubric/checklist；
- 匿名化模型身份；
- 随机化 A/B 顺序，并做 swap consistency；
- 独立判断正确性、相关性、风格、安全等维度；
- 允许 tie / abstain；
- 固定 judge model/version、temperature 和 prompt；
- 保存原始 verdict、理由和 parse failure；
- 用盲测 human subset 校准。

### 5.3 偏差与诊断

- **position bias**：偏好先出现/后出现的答案；交换顺序后 verdict 翻转
  是直接诊断，MT-Bench 系统测量了该翻转率 [[7]](#ref-7)，并显示即使
  GPT-4 judge 在指令遵循任务上也存在可观翻转 [[10]](#ref-10)；
- **length/verbosity bias**：更长被误判更好；长度可解释 judge 偏好的
  很大份额 [[9]](#ref-9)；
- **style bias**：标题、礼貌、引用格式掩盖事实错误；
- **self-preference**：judge 系统性地偏爱与自身同家族/同措辞的输出；
  这与"自我识别"能力相关且可区分 [[11]](#ref-11)；
- **reference leakage**：judge 看到不该见的信息；
- **sycophancy**：迎合用户立场而非判断真实性；
- **non-determinism**：同一输入多次 verdict 不一致。

诊断工具箱：双向评估取平均、长度/风格匹配对照、无参考/有参考对照、多
judge ensemble、人工 adjudication。Arena 排名只是外部效度证据之一，不
证明每个领域可靠；judge 与人类的总体高一致也可以与领域级偏差共存。

### 5.4 与人类校准

MT-Bench 报告 GPT-4 judge 与人类偏好一致率约 85%，与人类之间的一致率
相当 [[7]](#ref-7)。但"一致率高"不等于"无偏"：校准应分领域、分维度进行，
并对 judge 与人类分歧的样本做错误类型分析——分歧往往集中在模糊地带与
高风险地带，恰恰是部署中最需要谨慎的区域。

## 6. 从 benchmark 到现实效度

### 6.1 三层效度

评估质量至少分三层：

1. **reliability**：重复测量是否稳定（seed、协议微扰、judge 重复）；
2. **construct validity**：指标是否真的测目标能力，而非格式、记忆或
   grader 偏好；
3. **ecological validity**：任务、用户、工具、成本和失败代价是否接近
   部署。

MMLU/GPQA 控制性强、易评分，但与真实工作有距离；Arena prompt 更真实，
却用户分布不受控；SWE-bench 以测试判定 agent 行为，但 test coverage 与
环境可复现性会成为 grader 瓶颈 [[18]](#ref-18)。没有单一 benchmark 能
替代目标场景。

### 6.2 饱和、退化与 gaming

- **饱和**：头部模型在经典题集上挤在顶部，区分度下降；此时应补充更难
  slice 或改用连续指标（对照 Lecture 11：离散指标的"涌现/平台"常是
  测量阈值效应 [[24]](#ref-24)）。
- **退化**：社区反复对同一 test set 调参，等效于集体 overfitting；
  复测集显示显著下降 [[15]](#ref-15)。
- **gaming**：为榜单指标直接优化（如针对性数据、模板化 prompt），分数
  上涨不反映构念提升 [[16]](#ref-16)。

对策是周期性轮换（LiveBench [[17]](#ref-17)）、私有题库和把 benchmark
选择从开发闭环中隔离。

### 6.3 Agent 评估与成本控制

agent 评估的统计单位是 task，grader 通常是测试或规则。SWE-bench 用真实
GitHub issue 与 fail-to-pass/pass-to-pass 单元测试，使 grader 客观化，
但也把"分数"限制在测试覆盖的能力域内 [[18]](#ref-18)。Kapoor et al.
指出 agent 榜单普遍忽略成本，prompt hack 即可无代价抬分；他们主张在
accuracy–cost Pareto 前沿上做成本受控比较 [[19]](#ref-19)。agent 分数的
归因尤其困难：模型、scaffold、工具、预算、重试都贡献分数；对 base model
的任何结论都需要固定其余变量。

### 6.4 Evaluation portfolio

建议建立评估组合（HELM 的七 desiderata 是一个起点 [[4]](#ref-4)）：

- capability：知识、推理、代码、长上下文；
- interaction：指令遵循、对话、多轮；
- robustness：扰动、分布移位、拒答边界；
- safety：具体风险 taxonomy 与 adversarial prompts；
- efficiency：延迟、成本、tokens、工具调用；
- domain/private：真实业务任务和专家 rubric。

## 7. 成本与复杂度

| 评估 | 模型调用 | grader 成本 | 统计单位 |
|---|---:|---:|---|
| perplexity | \(O(nT)\) teacher-forced tokens | 无/规则 | document 或 token cluster |
| exact-match QA | \(O(nT_{\text{gen}})\) | \(O(n)\) 解析 | question |
| pass@k | \(O(nk\,T_{\text{gen}})\) | \(O(nk)\) 测试执行 | question |
| pairwise judge | 生成 \(2n\) responses | \(O(n)\) judge calls | prompt |
| \(m\) 模型全对阵 | 约 \(O(nm)\) 生成 | 最坏 \(O(nm^2)\) | prompt/pair |
| agent eval | 多轮、工具与重试 | tests/judge | task |
| bootstrap | 已有结果上 \(R\) 次 | \(O(Rn)\) | 保留 cluster |

缓存模型输出可避免重复生成，但 cache key 必须包含 model/version、
prompt、decoding 和 tool environment。否则"复现"可能读到错误协议的旧
结果——这是评估系统中最隐蔽的正确性 bug 之一。

## 8. 实现映射

| 模块 | 建议输入/输出 | 关键测试 |
|---|---|---|
| dataset loader | immutable `example_id`, prompt, reference, category | hash/version 固定 |
| runner | request → raw response + usage + seed | retry 不重复计样本 |
| parser | raw response → structured answer/parse error | adversarial formatting |
| grader | answer/reference → score + metadata | human gold subset |
| judge runner | 双向 A/B + swap | order-swap 一致率 |
| aggregator | per-example records → metric + CI | paired/cluster bootstrap |
| contamination scan | train fingerprints × eval examples | known-positive recall |
| report | config + per-category + failures | 可追溯到 example |

生态参考：HELM [[4]](#ref-4) 与
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
展示了 scenario/adaptation/metric 解耦的工程实现；两者的共同教训是协议
配置必须与结果一起版本化。

课程中的 A3 validation loss 是平滑的 scaling target，但不能推出开放式
生成、calibration、安全或 inference 成本。Evaluation 层应与 scaling fit
分开，避免用 benchmark 反复选择 scaling model 后仍称其为 held-out。

## 9. 失败分析比总分更有行动价值

保留 per-example record，至少包含：

- example/category/version；
- 完整 prompt template 与 few-shot examples；
- raw output、parsed output、parse error；
- model/version/seed/decoding/tools；
- grader/version/rubric/verdict；
- latency/tokens/cost；
- contamination flag 与人工备注。

按领域、难度、长度、语言、工具、风险类别和错误类型切片。总分上涨可能
只来自简单大类；小而高风险切片退化会被 macro/micro average 掩盖。错误
类型分析（解析失败 vs 知识缺失 vs 推理断裂 vs 拒答不当）直接决定下一步
是改 prompt、改 parser 还是补数据。

## 10. 常见误区

- **测试集越大越真实：** 样本量降低方差，不能修复构念偏差。
- **PPL 可跨 tokenizer 直接比：** token 单位不同，应换 bpb。
- **MC 的 choice log-prob 是"真实能力"：** 协议只是测量的一部分，
  换协议排名会变 [[21]](#ref-21)。
- **pass@k 高等于单次质量好：** 覆盖广度与单次正确率可背离。
- **Elo/BT 分数是绝对能力刻度：** 只是相对偏好；换参赛集合与锚点会
  整体平移 [[6]](#ref-6)。
- **judge 给理由就可信：** fluent rationale 不是判定正确性的证明。
- **高 human correlation 代表无偏：** 总体相关可共存于领域偏差。
- **fresh benchmark 不会污染：** 公开后会迅速进入后续训练和调参闭环。
- **benchmark 饱和等于问题解决：** 可能是污染、格式 exploit 或难度不足。
- **agent 分数是 LM 分数：** scaffold、工具、预算和重试同样贡献。
- **只报平均值：** 隐藏 uncertainty、类别权重与严重失败。

## 11. 讨论：效度威胁、伦理与决策边界

### 11.1 Construct validity

分数测的是"模型+协议+grader"的复合体。contamination、格式 exploit、
judge 偏好都会让指标与构念脱钩；calibration、robustness 等辅助指标
（HELM [[4]](#ref-4)）能提供构念三角验证。

### 11.2 Internal validity

协议微扰（shot 数、template、decoding）、checkpoint cherry-picking、
parse failure 的静默丢弃都是研究者自由度；预注册与开发/最终集分离是
标准防线。judge 的 non-determinism 未报告时，复现实验可能无法区分
真实差异与噪声。

### 11.3 External validity

benchmark 分数到部署表现的迁移依赖用户分布、工具环境与失败代价的
相似性。对高风险决策（医疗、法律、安全关键），应要求领域内私有评估
与专家 rubric，而非外推通用榜单。

### 11.4 伦理与激励

leaderboard 会扭曲研究激励：为 0.5 分差异投入巨额算力、针对性清洗
训练数据、隐藏失败切片。评估报告应公开不确定性、负结果与失败案例
（model card 实践 [[25]](#ref-25)），并把 benchmark 数据的版权与隐私
边界纳入审计。评估自身的计算成本（尤其多 judge、agent 重试）也应被
核算——评估基础设施同样是需要 carbon/compute accounting 的系统。

## Checklist

- [ ] 写出目标构念、决策用途和评估对象（方法/模型/系统）。
- [ ] 固定 dataset/version、prompt、shots、sampling、tools 和预算。
- [ ] 预注册 MC 作答协议，并做选项顺序稳定性检查。
- [ ] 保存 per-example 原始输出与 grader 记录。
- [ ] 使用适合层级结构的 paired/cluster CI。
- [ ] 报告样本量、parse failures、重试和 abstentions。
- [ ] 做 contamination 扫描并声明检测方法的召回边界。
- [ ] judge 做 order swap、长度/风格偏差和 human calibration。
- [ ] 对代码任务同时报告多个 \(k\) 的 pass@k。
- [ ] agent 评估报告 accuracy–cost Pareto 与失败模式。
- [ ] 同时报告总体、切片、成本与严重失败。
- [ ] 开发集调协议，最终集限制访问和重复提交。

### 诊断速查表

| 现象 | 最可能问题 | 后续动作 |
|---|---|---|
| MC 换选项顺序排名翻转 | 协议敏感/污染迹象 | 报告 order-shuffle 稳定性 [[21]](#ref-21) |
| judge 交换 A/B 后 verdict 翻转 | position bias | 双向评估取平均，报告翻转率 |
| bootstrap 区间异常窄 | 重采样单位错误 | 按 prompt/task cluster 重采样 |
| 新旧模型在饱和题集并列 | 区分度不足 | 加难 slice、连续指标 [[24]](#ref-24) |
| 同分布新题分数骤降 | contamination | 报告新旧题差、审计数据暴露 [[16]](#ref-16) |
| pass@k 上升而 pass@1 不变 | 覆盖 vs 单次质量 | 多 \(k\) 联合报告 |
| Arena 排名与内部 eval 矛盾 | 人群/任务分布差 | 分域比较，勿强行调和 |
| agent 分数涨但成本翻倍 | 无成本控制 | accuracy–cost Pareto [[19]](#ref-19) |
| "复现"结果与记录不符 | cache key 不完整 | key 加入协议全字段 |

## 12. 面试备考（Interview Prep）

> LLM 评估是面试的「统计+方法论」高频题：面试官常从「pass@k 无偏估计」切入，追到
> 「MC 三种 adaptation 为什么排名会变」「LLM judge 三大偏差」「contamination 怎么检测」
> 「为什么分数必须带区间」。核心是把评估当**测量设计**，牢记「一个分数是
> (model, data, protocol, grader) 四元组的函数」。下面按「一页速览 → 高频题 → 手撕 → 追问」四层组织。

### 12.1 一页速览卡（面试前 1 分钟）

**核心主张**：评估是把抽象能力目标转成可审计测量的工程+统计过程；一个分数只在四元组
\((\text{model},\text{data},\text{protocol},\text{grader})\) 固定时才有定义，先问「怎么测」再谈「多少分」。

**必背数字与公式**

- pass@k 无偏估计 \(\widehat{\text{pass@k}}=1-\binom{n-c}{k}/\binom{n}{k}\)。
- PPL \(=e^{\bar\ell}\)，跨 tokenizer 须换 \(\text{bpb}=\frac{\bar\ell}{\ln2}\cdot\frac{T}{B_{\text{bytes}}}\)。
- 二元标准误 \(\operatorname{SE}(\hat p)=\sqrt{\hat p(1-\hat p)/n}\)，小样本用 Wilson。
- 校准误差 \(\operatorname{ECE}=\sum_b\frac{n_b}{n}|\operatorname{acc}(b)-\operatorname{conf}(b)|\)。

**三句话答高频**

1. 一个分数是四元组函数——protocol/grader 一变，测的就是另一个量。
2. LLM judge 三大偏差：position（双向取平均）、verbosity（长度回归）、self-preference（隐藏身份+人工校准）。
3. contamination 分层检测：数据侧 n-gram/MinHash → 模型侧 exchangeability/选项扰动 → time-travel。

### 12.2 高频面试题与答题框架

**Q1：pass@k 的无偏估计是什么？为什么？**

- 单次通过率 \(p\) 时，\(\text{pass@k}=1-(1-p)^k\)；但实现是每题采样 \(n\) 次、\(c\) 次通过。
- Codex 无偏估计 \(\widehat{\text{pass@k}}=1-\binom{n-c}{k}/\binom{n}{k}\)：从 \(n\) 个样本不放回抽 \(k\) 个，全未命中概率是超几何比值，对 \(c\sim\mathrm{Binomial}(n,p)\) 取期望由 Vandermonde 恒等式得 \((1-p)^k\)。
- 朴素 `c/n` 在小 \(c\) 时高估；应在 log 空间算组合数避免溢出；pass@k 随 \(k\) 递增，与 pass@1（单次质量）可背离。

**Q2：multiple-choice 的三种 adaptation？为什么排名会变？**

- ① 生成式（生成选项字母/文本再解析）；② choice log-prob（每个选项条件似然取 argmax）；③ 排序式（比较选项相对似然）。
- 三种 protocol 测的不是同一量，且模型对选项顺序高度敏感——换 protocol 或顺序排名会变。
- 因此 protocol 必须预注册，不能事后挑对自己有利的作答方式。

**Q3：LLM-as-a-judge 的三大偏差与对策？**

- **position bias**：偏好先/后出现的答案 → 双向评估取平均、报告 swap 翻转率。
- **length/verbosity bias**：更长被误判更好 → 长度回归控制（AlpacaEval-LC）、长度匹配对照。
- **self-preference**：偏爱与自身同家族/同措辞的输出 → 隐藏模型身份、人工盲测校准。
- 另有 style bias、sycophancy、reference leakage、non-determinism，需多 judge ensemble + 人工 adjudication。

**Q4：contamination 怎么分层检测？**

- **数据侧**：exact hash、n-gram（GPT-3 用 13-gram）、MinHash/LSH、embedding 检索——给覆盖率，但召回受 paraphrase/翻译限制。
- **模型侧**：verbatim completion、exchangeability 检验（基准顺序似然显著高于随机打乱）、选项顺序扰动、time-travel 探针（补全「数据集发布后才有」的信息）。
- **对照构造**：同分布新题（GSM1k vs GSM8K）、按时间切分（LiveBench）。注意「没发现 overlap ≠ 没有污染」。

**Q5：为什么分数必须带区间？paired comparison 为什么更有功效？**

- seed/采样噪声下单点分数不可比；用 Wilson/bootstrap 区间 + 多重比较校正。
- paired 设计（两模型在同一批题上）消除题目难度共同方差：McNemar 检验只用答案不同的题目，paired bootstrap 对每题差值重采样。
- 「A 的 CI 与 B 的 CI 相交」≠「无法区分 A 与 B」；错误选重采样单位（如把同 prompt 的重复判断当独立）是区间过窄的最常见原因。

**Q6：perplexity 为什么不能跨 tokenizer 直接比？**

- \(\operatorname{PPL}=e^{\bar\ell}\)，单位是 token；不同 tokenizer 的 token 数不同，per-token PPL 标尺不同。
- 换 bpb（每字节比特数）才可比；对 prompt-response 任务只在 response mask 上算 conditional PPL，避免 prompt 长度支配指标。

**Q7：Bradley–Terry 与 Elo 的区别？**

- BT：离线 MLE，把模型参数化为潜在分数 \(s_i\)，\(P(A\succ B)=\sigma(s_A-s_B)\)；可复用传递性信息，对局集合内等价、可给 CI。
- Elo：在线序贯更新，依赖对局顺序，tie 需特殊记分，不确定性难处理。
- Chatbot Arena 用带 tie margin + 风格控制项的 BT 变体；朴素 Elo 对出场顺序/参赛集合/噪声更敏感。

**Q8：calibration / ECE 是什么？为什么重要？**

- ECE 把预测按置信度分桶，比较每桶平均置信度与实际正确率，是 reliability diagram 的数值摘要。
- 校准对部署决策（阈值截断、selective prediction、abstain）至关重要；HELM 把它列为与 accuracy 并列的一级指标。

**Q9：agent 评估为什么难？**

- 统计单位是 task，grader 是测试/规则（SWE-bench 用 fail-to-pass 单元测试），把分数限制在测试覆盖域内。
- 模型、scaffold、工具、预算、重试都贡献分数，归因难；Kapoor 等指出榜单普遍忽略成本，prompt hack 可无代价抬分。
- 应在 accuracy–cost Pareto 前沿上做成本受控比较，固定其余变量才能谈 base model 贡献。

**Q10：为什么「一个分数」不完整？**

- 分数是四元组函数：protocol、grader、数据、采样任一变化都在测另一个量。
- 总分上涨可能只来自简单大类，小而高风险切片退化会被 macro/micro average 掩盖；失败分析（解析失败 vs 知识缺失 vs 推理断裂）比总分更有行动价值。

### 12.3 手撕要点（pass@k 与区间）

面试让「推导 pass@k 无偏性」或「算区间」时，按公式写：

```text
pass@k = 1 - (1-p)^k，无偏估计 = 1 - C(n-c,k)/C(n,k)
  无偏性：E_c[C(n-c,k)/C(n,k)] = (1-p)^k   （超几何 + Vandermonde）

二元指标区间：SE(p̂) = sqrt(p̂(1-p̂)/n)，小样本/边界用 Wilson
配对比较：McNemar χ² = (|b-c|-1)²/(b+c)，b+c 小用精确二项
bootstrap 单位：每题 / prompt cluster / 领域 stratified / task，别把相关样本当独立
```

**三个必踩坑**

1. **pass@k 别用朴素 `c/n`**：小 \(c\) 时高估；用无偏公式并在 log 空间算组合数。
2. **PPL 别跨 tokenizer 比**：换 bpb，且只在 response mask 上算 conditional PPL。
3. **别把同 prompt 的重复判断当独立样本**：错选重采样单位导致区间过窄。

### 12.4 高频追问与陷阱

| 追问 | 正确方向 |
| --- | --- |
| 测试集越大越真实吗？ | 否，样本量降低方差，不能修复构念偏差 |
| judge 给理由就可信吗？ | 否，fluent rationale 不是判定正确性的证明 |
| 高 human correlation 代表无偏吗？ | 否，总体相关可共存于领域偏差 |
| fresh benchmark 不会污染吗？ | 否，公开后会迅速进入后续训练/调参闭环 |
| benchmark 饱和等于解决吗？ | 否，可能是污染、格式 exploit 或难度不足 |
| agent 分数是 LM 分数吗？ | 否，scaffold/工具/预算/重试同样贡献 |

## 13. 本讲小结

评估是把抽象目标转成可审计测量的工程与统计过程。一个分数只在
\((\text{model},\text{data},\text{protocol},\text{grader})\) 四元组固定
时才有定义：perplexity 依赖 tokenizer 与 mask，MC 分数依赖作答协议，
pass@k 依赖采样预算，judge 分数依赖 judge 身份与顺序，agent 分数依赖
系统与预算。可信结论要求同时陈述协议、配对/聚类不确定性、污染边界与
grader 偏差，并以失败分析与切片报告支撑行动。不存在唯一真实 benchmark，
只有 reliability、construct validity 与 ecological validity 证据强度
不同的评估组合；评估的价值最终由它支持的决策质量衡量。

## 参考文献

<a id="ref-1"></a>[1] D. Hendrycks et al. "Measuring Massive Multitask
Language Understanding." *ICLR*, 2021. [arXiv](https://arxiv.org/abs/2009.03300)

<a id="ref-2"></a>[2] A. Srivastava et al. "Beyond the Imitation Game:
Quantifying and Extrapolating the Capabilities of Language Models
(BIG-bench)." *TMLR*, 2023. [arXiv](https://arxiv.org/abs/2206.04615)

<a id="ref-3"></a>[3] D. Rein et al. "GPQA: A Graduate-Level
Google-Proof Q&A Benchmark." arXiv:2311.12022, 2023.

<a id="ref-4"></a>[4] P. Liang et al. "Holistic Evaluation of Language
Models." arXiv:2211.09110, 2022. [link](https://arxiv.org/abs/2211.09110)

<a id="ref-5"></a>[5] W. Chiang et al. "Chatbot Arena: An Open Platform
for Evaluating LLMs by Human Preference." *ICLR*, 2025.
[arXiv](https://arxiv.org/abs/2403.04132)

<a id="ref-6"></a>[6] A. Boubdir et al. "Elo Uncovered: Robustness
and Best Practices in Language Model Evaluation." arXiv:2311.17295,
2023.

<a id="ref-7"></a>[7] L. Zheng et al. "Judging LLM-as-a-Judge with
MT-Bench and Chatbot Arena." *NeurIPS*, 2024.
[arXiv](https://arxiv.org/abs/2306.05685)

<a id="ref-8"></a>[8] Y. Liu et al. "G-Eval: NLG Evaluation using GPT-4
with Better Human Alignment." *EMNLP*, 2023.
[arXiv](https://arxiv.org/abs/2303.16634)

<a id="ref-9"></a>[9] Y. Dubois et al. "Length-Controlled AlpacaEval:
A Simple Way to Debias Automatic Evaluators." *ICML*, 2024.
[arXiv](https://arxiv.org/abs/2404.04475)

<a id="ref-10"></a>[10] P. Wang et al. "Large Language Models are not
Fair Evaluators." *ACL*, 2024. [arXiv](https://arxiv.org/abs/2305.17926)

<a id="ref-11"></a>[11] A. Panickssery, S. R. Bowman, and S. Feng.
"LLM Evaluators Recognize and Favor Their Own Generations." *NeurIPS*,
2024. [arXiv](https://arxiv.org/abs/2404.13076)

<a id="ref-12"></a>[12] Y. Oren et al. "Proving Test Set Contamination
in Black-Box Language Models." *ICLR*, 2024.
[arXiv](https://arxiv.org/abs/2310.17623)

<a id="ref-13"></a>[13] S. Golchin and M. Surdeanu. "Time Travel in
LLMs: Tracing Data Contamination in Large Language Models." *ICLR*,
2024. [arXiv](https://arxiv.org/abs/2308.08493)

<a id="ref-14"></a>[14] O. Sainz et al. "NLP Evaluation in Trouble: On
the Need to Measure LLM Data Contamination for Each Benchmark." *EMNLP*,
2023. [arXiv](https://arxiv.org/abs/2310.18018)

<a id="ref-15"></a>[15] B. Recht et al. "Do ImageNet Classifiers
Generalize to ImageNet?" *ICML*, 2019.
[arXiv](https://arxiv.org/abs/1902.10811)

<a id="ref-16"></a>[16] H. Zhang et al. "A Careful Examination of LLM
Performance on Grade School Arithmetic." arXiv:2405.00332, 2024.

<a id="ref-17"></a>[17] C. White et al. "LiveBench: A Challenging,
Contamination-Free LLM Benchmark." arXiv:2406.19314, 2024.

<a id="ref-18"></a>[18] C. Jimenez et al. "SWE-bench: Can Language
Models Resolve Real-World GitHub Issues?" *ICLR*, 2024.
[arXiv](https://arxiv.org/abs/2310.06770)

<a id="ref-19"></a>[19] S. Kapoor et al. "AI Agents That Matter."
*NeurIPS*, 2024. [arXiv](https://arxiv.org/abs/2407.01502)

<a id="ref-20"></a>[20] J. Miller. "Adding Error Bars to Evals: A
Statistical Approach to Language Model Evaluations." arXiv:2411.00640,
2024.

<a id="ref-21"></a>[21] H. Zheng et al. "Large Language Models Are Not
Robust Multiple Choice Selectors." *ICLR*, 2024.
[arXiv](https://arxiv.org/abs/2309.03882)

<a id="ref-22"></a>[22] M. Chen et al. "Evaluating Large Language Models
Trained on Code." arXiv:2107.03374, 2021.

<a id="ref-23"></a>[23] N. Carlini et al. "Extracting Training Data
from Large Language Models." *USENIX Security*, 2021.
[arXiv](https://arxiv.org/abs/2012.07805)

<a id="ref-24"></a>[24] R. Schaeffer, B. Miranda, and S. Koyejo. "Are
Emergent Abilities of Large Language Models a Mirage?" *NeurIPS*, 2023.
[arXiv](https://arxiv.org/abs/2304.15004)

<a id="ref-25"></a>[25] M. Mitchell et al. "Model Cards for Model
Reporting." *ACM FAT\**, 2019. [arXiv](https://arxiv.org/abs/1810.03993)

## 延伸阅读

- Stanford CS336, [Lecture 12 — Evaluation](https://github.com/stanford-cs336/lectures/blob/main/lecture_12.py).
- [HELM 官方站点与排行榜](https://crfm.stanford.edu/helm/)。
- [A3 scaling 实验目录](../experiments/catalog/a3-scaling.md)：validation
  loss 作为评估指标的边界。
