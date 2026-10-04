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
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
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
- **两家报告的榜单分数默认可比：** 2023 年底 Gemini 报 MMLU 用 CoT@32，
  GPT-4 用 few-shot——协议错位即可制造“超越”。
- **代码基准的测试用例足够强：** HumanEval+ 把用例扩约 80× 即检测到
  更多未发现错误——弱测试系统性高估代码能力。

## 11. 讨论：效度威胁、伦理与决策边界

### Construct validity

分数测的是"模型+协议+grader"的复合体。contamination、格式 exploit、
judge 偏好都会让指标与构念脱钩；calibration、robustness 等辅助指标
（HELM [[4]](#ref-4)）能提供构念三角验证。

### Internal validity

协议微扰（shot 数、template、decoding）、checkpoint cherry-picking、
parse failure 的静默丢弃都是研究者自由度；预注册与开发/最终集分离是
标准防线。judge 的 non-determinism 未报告时，复现实验可能无法区分
真实差异与噪声。

### External validity

benchmark 分数到部署表现的迁移依赖用户分布、工具环境与失败代价的
相似性。对高风险决策（医疗、法律、安全关键），应要求领域内私有评估
与专家 rubric，而非外推通用榜单。

### 伦理与激励

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

## 面试要点速记

**高频问题与答题要点**

1. **Q：pass@k 的无偏估计？** 要点：\(1-\binom{n-c}{k}/\binom{n}{k}\)（不重复
   采样版本）；朴素 c/n 在小 c 时高估。
2. **Q：multiple-choice 的三种 adaptation？** 要点：完整序列 LM scoring、
   按选择长度归一化、逐字符补全打分；adapter 选择本身会改变模型排名。
3. **Q：LLM judge 的三大偏差与对策？** 要点：position（双向取平均）、
   verbosity（长度控制/回归校正）、self-preference（隐藏身份 + 人工校准）。
4. **Q：contamination 怎么分层检测？** 要点：数据侧 n-gram/MinHash 匹配 →
   模型侧 exchangeability/选项扰动 → time-travel（只用基准发布前数据训练
   对照组）。
5. **Q：为什么分数必须带区间？** 要点：seed 与采样噪声下，单点分数不可比；
   Wilson/bootstrap 区间 + 多重比较校正是报告底线。
6. **Q：两个模型的榜单分数能直接比吗？** 要点：分数是四元组函数——核对
   data 版本、protocol（CoT@32 vs few-shot，参照 Gemini/GPT-4 的 MMLU
   争议）、grader；必要时用第三方（HELM/OpenCompass）同协议结果交叉验证。
7. **Q：什么时候用人工评测/Arena？** 要点：开放式生成、ecological
   validity 优先、静态集饱和或污染时；人类双盲 → Elo/BT；成本高则
   LLM-judge 初筛 + 盲测子集校准，按领域切片报告。
8. **Q：如何为公司搭建评测体系？** 要点：L0 通用基准选型（MMLU 系、
   GSM8K/MATH、LiveCodeBench、MMMU）→ L1/L2 场景化自定义指标（精准率/
   召回率/编辑距离相似度）→ 私有题库防污染 + 与第三方对标。
9. **Q：HumanEval 高分但线上代码能力差，为什么？** 要点：grader 太弱
   （HumanEval+ 扩测约 80× 即暴露大量隐藏错误）、题目老旧易污染、
   函数级补全 ≠ 仓库级任务。

**必背数字**

- pass@k 公式；一个分数是 (model, data, protocol, grader) 四元组的函数——
  面试中先问“怎么测”再谈“多少分”。

**工业界参照**

- MMLU：57 学科多选、总计 15,908 题（test ≈14,079）；衍生 MMLU-Pro
  （约 12,000 题、10 选项、更难去噪）与 MMLU-Redux（5,700 题手工重标注，
  NAACL 2025）。
- GPQA Diamond：主集 448 题；领域专家（博士）正确率约 65%，跨学科非专家
  约 34%——“Google-proof”设计。
- HumanEval+：测试用例扩约 80× 后 pass rate 大幅下降——弱测试高估代码能力。
- SuperGPQA：25,957 题、285 个研究生级学科——frontier 综合极限评测量级。
- VSI-Bench：SOTA MLLM 空间推理最好仅 48.8%（Gemini-1.5 Pro）。
- 可比性案例：Gemini CoT@32 vs GPT-4 few-shot（MMLU）——协议不对齐，
  分数不可直接比。

## 行业现状与最新进展（2024–2026）

### benchmark 谱系与演化：扩量、加难、去噪、防污染

| 基准 | 规模与形式 | 定位与要点 |
|---|---|---|
| MMLU | 57 学科多选，总计 15,908 题（test ≈14,079） | 最常用通识基准；接近饱和、污染面大 |
| MMLU-Pro | 约 12,000 题，10 选项 | 更难、去噪，压缩猜测侥幸空间 |
| MMLU-Redux | 5,700 题手工重标注 | 纠错去污染的“干净版 MMLU”（NAACL 2025） |
| GPQA（Diamond） | 主集 448 题，博士级理工多选 | “Google-proof”：领域专家约 65%，跨学科非专家约 34% |
| SuperGPQA | 25,957 题、285 个研究生级学科 | frontier 综合极限评测 |
| HumanEval → HumanEval+ | 测试用例扩约 80× | 弱测试高估代码能力：扩测后 pass rate 大幅下降 |
| HumanEval → LiveCodeBench | 按时间窗收集新题 | 时间切分防污染，DeepSeek-V3 等采用 |

模型发布的“选测矩阵”（行业洞察整理）：MMLU 几乎必选（衍生
MMLU-Pro/Redux/Multilingual-MMLU）；数学 GSM8K/MATH 必选；代码
HumanEval → LiveCodeBench；多模态 MMMU 必选 + 各家百花齐放。

### 数据污染与评测可比性

- **可比性案例**：2023 年底 Gemini 在 MMLU 上得分高于 GPT-4，但技术报告
  显示其采用 CoT@32（思维链 + 32 次尝试取最好），而 GPT-4 报的是
  few-shot——协议不同引发公正性质疑；同一测试集在论文中的结果也常因
  prompt/metric 不同难以直接比较。
- **行业对策**：污染检测工具、消融实验、有公信力第三方——HELM、
  OpenCompass；信通院“方升”体系（2023-12 发布）以测试指标/方法/数据集/
  工具四要素、32 个二级维度标准化评测。
- Open LLM Leaderboard 聚合 ARC/HellaSwag/MMLU/TruthfulQA/WinoGrande/
  GSM8K 六项，是开源模型选型的公开参照；静态集同样面临饱和与污染。

### 人类偏好与自动化评测

- **LMSYS Chatbot Arena**：用户双盲对比 → Elo/BT 排名，成为人类偏好
  事实标准；但用户分布不受控，生态效度与内部效度互斥。
- **LLM-as-judge**：single-answer/pairwise 协议 + 裁判模型普及化；
  position/verbosity/self-preference 偏差审计成为发布标配环节。
- **空间智能案例 VSI-Bench（李飞飞/谢赛宁）**：SOTA MLLM 空间推理不足，
  最好的 Gemini-1.5 Pro 仅 48.8%——训练数据需补空间知识；说明新能力
  维度会持续催生新基准。

**对本讲学习者的启示**：本讲的四元组框架正是解读上述行业争议的工具——
Gemini 案例是 protocol 环错位，HumanEval+ 是 grader 环太弱，
LiveBench/LiveCodeBench 是 data 环的时间控制，Arena 是 data 分布真实但
不受控。面试中用测量链逐环归因，比罗列榜单更能体现功底。

## 大厂面试真题与答题框架

以下为高频面试题（公开面经风格），非任何公司真题。

**题目 1：如何检测训练数据对评测集的污染？**
- 考点：contamination 分类学、数据侧/模型侧检测、证据边界。
- 答题框架：
  1. 先分类：exact 泄漏 / paraphrase 与题解讨论 / 选择闭环（用 test 分数选 checkpoint、prompt）；
  2. 数据侧：normalized 13-gram 重叠（GPT-3 先例）、MinHash/LSH、embedding 检索，输出覆盖率报告；
  3. 模型侧：verbatim completion、exchangeability 检验（顺序似然差）、选项扰动、time-travel 探针；
  4. 对照构造：同分布新题（GSM1k 思路）或时间切分（LiveBench），比较新旧题分差；
  5. 声明召回边界：“未检出 ≠ 未污染”。
- 加分项：提到仅经发布渠道（arXiv 摘要等）即可检测到基准泄漏；建议同时报告去污染前后两套分数。
- 踩坑：只跑 exact-match 就下结论；把模型侧信号当成训练数据来源的“证明”。

**题目 2：两个模型 benchmark 分数不可比，怎么办？**
- 考点：分数是 (model, data, protocol, grader) 四元组的函数。
- 答题框架：
  1. 核对 data：同版本、同 split、同采样；
  2. 核对 protocol：shots、prompt、decoding、采样预算——举 2023 年底 Gemini CoT@32 vs GPT-4 few-shot 的 MMLU 案例说明协议错位制造虚高；
  3. 核对 grader：解析规则、judge 模型及版本；
  4. 无法重测时，用第三方（HELM/OpenCompass）同协议结果交叉验证；
  5. 给出带区间的配对结论（McNemar/paired bootstrap），而非只比点分。
- 加分项：主动指出同一测试集在不同论文中因 prompt/metric 不同也难以直接比较。
- 踩坑：直接拿两家技术报告的数字相减；忽视采样次数与预算差异。

**题目 3：什么情况下用人工评测/Arena，而不是静态 benchmark？**
- 考点：ecological validity、开放式生成、偏好分布。
- 答题框架：
  1. 判据：答案开放、评价维度多（风格/安全/有用性）、静态集饱和或污染严重；
  2. 人类双盲对比 → Elo/BT 排名，覆盖真实用户分布；
  3. 成本高 → LLM-judge 初筛 + 人工盲测子集校准（judge–human 一致率、swap 翻转率）；
  4. 按领域切片报告；Arena 总排名不保证每个领域可靠。
- 加分项：Arena 用带 tie margin 与风格控制的 BT 变体而非朴素 Elo；统计单位按 prompt cluster 聚类。
- 踩坑：把 Arena 排名当绝对能力刻度；把同一 prompt 的重复投票当独立样本。

**题目 4：为公司设计一个内部评测体系。**
- 考点：评测维度分层、基准选型、防污染、可持续治理。
- 答题框架：
  1. L0 通用能力：公开基准（MMLU 系、GSM8K/MATH、LiveCodeBench、MMMU）做开源模型选型；
  2. L1/L2 场景化：按业务定义能力（如代码检视、测试脚本生成），用精准率/召回率/编辑距离相似度等自定义指标做业务看护；
  3. 防污染：私有题库、时间切分、定期轮换、开发/最终集分离；
  4. 治理：预注册主指标、协议版本化、失败分析看板、与第三方（HELM/OpenCompass/信通院“方升”）对标。
- 加分项：agent 场景报 accuracy–cost Pareto；per-example 记录可追溯。
- 踩坑：只有 L0 榜单没有业务指标；用同一测试集反复选 checkpoint。

**题目 5：模型 HumanEval 分数很高，线上代码能力却差，怎么解释？**
- 考点：弱测试高估、grader 强度、分布偏移。
- 答题框架：
  1. grader 环：HumanEval 原测试弱——HumanEval+ 把用例扩约 80× 即检测到更多未发现错误、pass rate 大幅下降；
  2. data 环：题目老旧、污染概率高；
  3. 分布：函数级补全 ≠ 真实仓库级任务；
  4. 改进：LiveCodeBench 时间切分 + 仓库级任务（SWE-bench 类）+ 多 k 联合报告。
- 加分项：点出“分数是 grader 的函数”这一通用原理。
- 踩坑：不审计测量链，直接归因于“模型退步”。

**题目 6：LLM-as-judge 有哪些系统性偏差？如何审计？**
- 考点：position/verbosity/self-preference 偏差与人类校准。
- 答题框架：
  1. 列偏差：position、verbosity、style、self-preference、sycophancy、non-determinism；
  2. 审计：swap 双向取平均并报翻转率；长度/风格控制（Length-Controlled AlpacaEval 思路）；隐藏模型身份；
  3. 校准：盲测 human subset，分领域报告一致率（MT-Bench：GPT-4 judge 与人类约 85%）；
  4. 允许 tie/abstain，保存原始 verdict 与 parse failure。
- 加分项：从“事后报告偏差”到“估计量内消除偏差”的范式转变。
- 踩坑：总体一致率高就断言无偏；只用单一 judge 且不固定版本。

**题目 7：pass@1 不变但 pass@k 上升，说明什么？**
- 考点：覆盖 vs 单次质量；采样预算是协议的一部分。
- 答题框架：
  1. 定义：pass@k = 1-(1-p)^k，无偏估计用超几何比值；
  2. 解读：多样性/覆盖提升，单次正确率未变；
  3. 场景：多候选 + 重排/选择器管线受益；单轮对话场景未必；
  4. 报告：多 k 联合报告，注明 n 与 temperature。
- 加分项：实现细节——log 空间计算组合数防溢出。
- 踩坑：把 pass@k 提升宣传为“能力提升”而不说明采样预算。

## 系统设计题

**设计题 1：为公司模型迭代设计自动化评测平台（“超自动化基准测试平台”方向）**

- 需求澄清：评对象是 checkpoint 还是线上系统？触发时机（每次训练完成 / nightly / 发版前）？预算与延迟约束？结果消费者是谁（研究员/产品/管理层）？
- 规模估算：每次评测约 10–50 个 benchmark × 数千到数万题 × 每题 1–32 次采样；pairwise judge 再乘模型对数；按 token 计费估算单次全量成本，据此决定全量与抽样分层。
- 架构：
  1. 任务分发：benchmark registry（数据版本 + 协议配置 + grader 定义，全部 hash 固定）→ 生成评测任务 DAG；
  2. 分布式执行：runner 集群消费任务；cache key 包含 (model, data, protocol, grader) 全字段，防止协议漂移读到旧结果；
  3. 结果统计：per-example record 落库 → aggregator 计算 pass@k、Elo/BT、paired/cluster bootstrap；
  4. 裁判模型：judge 服务（固定版本、双向 A/B + swap、rubric 版本化）；
  5. 人工复核工作流：judge 与人类分歧样本、高风险切片进入标注队列，双盲 + 仲裁。
- trade-off：

| 决策 | 选项 A | 选项 B | 取舍 |
|---|---|---|---|
| 执行时机 | 每次提交全量 | 触发式 + nightly 增量 | 覆盖 vs 成本 |
| 缓存 key | 协议全字段 | 仅 prompt | 正确性 vs 命中率 |
| judge | 单强模型 | 多 judge ensemble | 成本 vs 偏差鲁棒 |
| 题库 | 全公开 | 公开 + 私有轮换 | 外部可比 vs 防污染 |

- 评测方案：平台自身的“元评测”——用已知差异的模型对做回归测试验证统计功效；污染扫描纳入流水线；报告必须带 CI、切片与失败分析。
- 追问预案：如何防内部题库泄漏（权限分级、轮换、canary）；新 benchmark 如何快速接入（registry schema + 适配器接口）；agent 任务如何报 accuracy–cost。

**设计题 2：设计防污染的代码评测系统（LiveCodeBench 时间切分思路）**

- 需求澄清：评测 base model 还是带工具的 agent？支持哪些语言？是否允许执行不可信代码？
- 规模估算：每月新增数百题；每题采样 n≈20 次执行 × 平均 10+ 测试用例；HumanEval+ 的教训是用例数量决定 grader 强度（扩约 80× 才暴露隐藏错误）。
- 架构：
  1. 题目采集：按时间窗收集发布日期晚于训练截止的题目，记录 provenance；
  2. 测试增强：自动生成 + 人工审核用例，覆盖边界条件（参照 HumanEval+ 思路）；
  3. 沙箱执行：资源限额、超时、网络隔离；输出 pass/fail + 失败日志；
  4. 统计：pass@1/pass@k（log 空间无偏估计）+ 按 time window 切片报告；
  5. 时间卫生：模型卡声明训练截止日期；旧窗口分数标注时效提示。
- trade-off：

| 决策 | 选项 A | 选项 B | 取舍 |
|---|---|---|---|
| 题源 | 竞赛题（质量高） | 仓库真实 issue（生态效度） | 构念 vs 现实 |
| 时间边界 | 严格按发布日切分 | 人工去污染 | 严格 vs 题量 |
| 执行 | 本地容器沙箱 | 远程隔离集群 | 成本 vs 安全 |

- 评测方案：用已知污染与干净模型对照验证窗口有效性；监控各窗口分数漂移作为污染预警。
- 追问预案：网页转载破坏时间边界怎么办（provenance 审计 + 窗口轮换）；私有部署如何与公开榜单保持可比（同协议开源 runner）。

**设计题 3：设计内部版 Chatbot Arena（人类偏好评测平台）**

- 需求澄清：用户是内部员工还是真实用户？双盲匿名是否可行？投票量级与激励？
- 规模估算：Elo/BT 收敛需要每模型数百到数千场对局；按模型对数 × 场次估算众包/内测成本。
- 架构：双盲 UI → vote 日志（prompt、两个匿名响应、verdict、元数据）→ BT 离线拟合（tie margin + 风格控制项）+ bootstrap CI → 分领域/分语言切片榜单。
- trade-off：

| 决策 | 选项 A | 选项 B | 取舍 |
|---|---|---|---|
| 排名算法 | 朴素 Elo 在线更新 | BT 离线 MLE | 简单 vs 顺序鲁棒 |
| 统计单位 | 每 vote 独立 | prompt cluster 聚类 | 便利 vs 统计正确 |
| 用户 | 真实用户 | 内部专家 | 生态效度 vs 构念效度 |

- 评测方案：与静态 benchmark 交叉验证排名合理性；监控新模型入场时的排名震荡（Elo 对出场顺序敏感）。
- 追问预案：低投票量领域如何出榜（设最低场次数 + CI 重叠即并列）；如何防刷票与对抗性投票。

## 代码实现题

**实现题 1：n-gram 污染检测器（评测集 vs 训练语料 13-gram 重叠）**

- 题目：给定评测集与训练语料，用 13-gram 重叠（GPT-3 先例）输出每道评测题的污染标记与语料级覆盖率。
- 考察点：文本规范化、n-gram 索引、内存效率、召回边界意识。
- Python 骨架：

```python
import re
from collections import defaultdict

N = 13

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def ngrams(text: str, n: int = N):
    toks = normalize(text).split()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}

def build_index(corpus_docs):
    idx = defaultdict(list)          # ngram -> doc ids
    for doc_id, doc in enumerate(corpus_docs):
        for g in ngrams(doc):
            idx[g].append(doc_id)
    return idx

def scan(eval_examples, corpus_index, threshold: int = 1):
    flagged = []
    for ex_id, ex in enumerate(eval_examples):
        grams = ngrams(ex)
        hits = sum(1 for g in grams if g in corpus_index)
        if hits >= threshold:
            flagged.append({"example_id": ex_id, "ngram_hits": hits,
                            "total_ngrams": len(grams)})
    return flagged
```

- 验收标准：注入的已知重叠样本召回 100%；对大小写/空白规范化鲁棒；结果可按 example_id 追溯；输出覆盖率报告而非仅布尔结论。

**实现题 2：pass@k 无偏估计**

- 题目：给定每题 n 次采样中的通过次数 c，实现无偏 pass@k 并避免溢出。
- 考察点：超几何比值、log 空间数值稳定性、无偏性来源。
- Python 骨架：

```python
import math
from functools import lru_cache

@lru_cache(maxsize=None)
def log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:          # 失败样本不足 k 个 → 任取 k 个必含通过
        return 1.0
    return 1.0 - math.exp(log_comb(n - c, k) - log_comb(n, k))

def mean_pass_at_k(results, k: int) -> float:
    return sum(pass_at_k(n, c, k) for n, c in results) / len(results)
```

- 验收标准：与暴力枚举 \(\binom{n}{k}\) 组合的精确值一致（小 n 对照）；c=0 时返回 0；n-c<k 时返回 1；大 n 不溢出。

**实现题 3：Arena Elo 更新模拟**

- 题目：模拟双盲对战流上的 Elo 在线更新，并演示其对局顺序敏感性。
- 考察点：期望胜率公式、K 因子、在线 vs 离线估计差异。
- Python 骨架：

```python
def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

def update_elo(ra, rb, score_a, k=32):
    ea = expected(ra, rb)
    return (ra + k * (score_a - ea),
            rb + k * ((1 - score_a) - (1 - ea)))

def simulate(matches, ratings=None, k=32):
    ratings = dict(ratings or {})
    for a, b, score_a in matches:   # score_a: 1 win / 0.5 tie / 0 loss
        ra, rb = ratings.get(a, 1200), ratings.get(b, 1200)
        ratings[a], ratings[b] = update_elo(ra, rb, score_a, k)
    return ratings
```

- 验收标准：交换对战顺序后最终分数不同（演示顺序敏感性）；score=0.5 时双方分值变化等量反向；K 越大单场影响越大。

## 本讲小结

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
- [Stanford CS336 课程主页](https://cs336.stanford.edu/)（访问日期 2026-10-04）。
- [CS336 lectures 仓库](https://github.com/stanford-cs336/lectures)（访问日期 2026-10-04）。
