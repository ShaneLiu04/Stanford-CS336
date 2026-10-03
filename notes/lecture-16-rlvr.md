---
title: "Lecture 16 — RL with Verifiable Rewards"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-05-20"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf"
  - "../assignments/spring2026/assignment5-alignment/cs336_spring2026_assignment5_alignment.pdf"
---

# Lecture 16 — RLVR：可验证奖励下的推理强化学习

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述，非课程提交
- 适用对象：自学者、后训练/推理强化学习工程师与研究者

## 摘要

RL with Verifiable Rewards（RLVR）把策略优化锚定在可程序化验证的信号上：数学答案比对、
代码单元测试、形式约束满足。本文以策略梯度为公共骨架，系统比较五种算法变体的隐含
重加权：GRPO 用组内均值-标准差归一化定义 advantage，Dr.GRPO 移除标准差与逐序列
长度两类归一化偏差，RFT 退化为对成功 rollout 的加权模仿，MaxRL 按组均值缩放以
聚焦边界任务，GSPO 把 importance ratio 从 token 级提升到 sequence 级。对每个变体
给出精确目标函数、数据契约（response mask、组 reshape、old log-probs 的行为策略
可追溯性）与失效信号（零梯度组、长度偏置、parser exploitation、entropy collapse）。
同时区分 outcome verifier 与 process verifier 的证据强度，并把“reward 上升”与
“可迁移推理能力”严格分离：在 DeepSeek-R1 类训练中，可验证 RL 被证明足以激励长链
推理，但结论始终限定在 verifier 覆盖的任务分布内。核心论点是：RLVR 的学习信号由采样温度、组内
基线、尺度归一化、长度归一化与 off-policy 校正粒度共同塑造，任何算法对比都必须固定
这些自由度并报告完整诊断指标。

**关键词：** RLVR；GRPO；Dr.GRPO；RFT；MaxRL；GSPO；Importance Ratio；Verifiable
Rewards；Reasoning

## 本文贡献

1. 在统一策略梯度框架下推导五种算法变体，显式给出每个归一化选择引入的 prompt/长度
   重加权；
2. 给出 response mask、组归一化、loss 归一化与 off-policy ratio 的精确实现契约及
   对应单测清单；
3. 区分 reward 改善、verifier exploitation 与真实推理能力三个层次的证据；
4. 整理 outcome/process verifier、难度课程与 safety verifier 的设计空间；
5. 提供从 verifier 测试到多 seed 对照实验的完整闭环与诊断速查表。

## 学习目标

1. 理解 outcome verifier、组采样、advantage 和 importance ratio 的关系。
2. 深入比较 GRPO、Dr.GRPO、RFT、MaxRL 与 GSPO 的隐含重加权。
3. 正确实现 response mask、组归一化、长度归一化和 on/off-policy loss。
4. 区分 reward 改善、verifier exploitation 与真实推理能力。

## 先修知识

- Lecture 15：SFT、reward modeling 与 KL 约束优化（RLVR 的方法学前置）。
- Lecture 12：评估与 judge 风险（reward 可信度的对应问题）。
- 强化学习基础：策略梯度、importance sampling、advantage 估计。

## 相关工作与技术谱系

RLVR 的直接前身是 PPO 的 clipped surrogate 目标，它提供了无需 critic 二阶信息的
稳定策略更新，是后续所有组内相对优化方法的公共组件 [[1]](#ref-1)。GRPO 由
DeepSeekMath 提出：放弃独立 value network，用同一 prompt 的组内相对奖励估计
advantage，显著降低了推理任务的 RL 系统复杂度 [[2]](#ref-2)。DeepSeek-R1 进一步
证明，在纯可验证奖励（rule-based，无神经 reward model）下，RL 足以激励模型自发
形成反思、验证等长链推理行为，而无需任何监督推理数据 [[3]](#ref-3)。

对 GRPO 偏差的系统批判来自 R1-Zero 类训练的分析工作：Liu 等识别出标准差归一化
会放大低方差组、逐序列长度归一化会引入长度偏置，并提出 Dr.GRPO 的移除方案
[[4]](#ref-4)。GSPO 则针对 token 级 importance ratio 在长推理序列上方差过大的
问题，把 ratio 定义为 response 内平均 log-ratio 的指数（几何均值），并证明其
sequence 级映射的收敛性质 [[5]](#ref-5)。

在奖励信号设计一侧，Lightman 等的 process supervision 研究表明：对推理的每一步
而非仅最终答案打分，可以在困难数学任务上显著提升 pass@1，说明 outcome reward 只是
推理质量的下界代理 [[6]](#ref-6)。Tulu 3 则把 RLVR 纳入 SFT → DPO → RLVR 的
开源后训练课程，验证了它作为独立训练阶段的可复现性 [[7]](#ref-7)。

## 1. RLVR 的适用边界

RL with Verifiable Rewards 用可程序化验证器给数学答案、代码测试或形式约束打分。对
prompt \(x\)，策略生成 \(y\)，奖励 \(r=V(x,y)\)。优势是 reward 比人工偏好客观且可
扩展；限制是 verifier 通常只看最终答案或可执行结果，无法保证推理忠实、安全或可读
[[3]](#ref-3)[[6]](#ref-6)。Tulu 3 的实践也表明 RLVR 应与 SFT/DPO 互补而非替代
[[7]](#ref-7)。

基本策略梯度：

\[
\nabla_\theta J(\theta)=
\mathbb E_{y\sim\pi_\theta}
\left[(r-b(x))\nabla_\theta\log\pi_\theta(y\mid x)\right].
\]

baseline 不改变 on-policy 期望梯度，但会改变方差；归一化和有限组估计还会引入
prompt reweighting。

## 2. GRPO：组内相对优势

同一 prompt 采样 \(G\) 个 responses，奖励 \(r_i\)。标准组优势 [[2]](#ref-2)：

\[
A_i^{\text{GRPO}}
=\frac{r_i-\bar r}{s_r+\epsilon},
\quad
\bar r=\frac1G\sum_i r_i.
\]

对旧策略 rollout，token-level clipped objective 可写为 [[1]](#ref-1)

\[
L=-\sum_{i,t}m_{i,t}
\min\left(\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\right),
\quad
\rho_{i,t}=e^{\log\pi_\theta-\log\pi_{\text{old}}}.
\]

整组全对或全错时 \(r_i-\bar r=0\)，该 prompt 无梯度。组大小和温度决定“组内出现
正负样本”的概率；binary reward 成功率为 \(p\) 时，全同概率为
\(p^G+(1-p)^G\)。这构成一种隐式的难度课程：RLVR 对当前策略恰好“可解但不可稳解”
的 prompt 投入最多学习信号。

## 3. Dr.GRPO：移除两类归一化偏差

课程/A5 中 Dr.GRPO 主要对应 [[4]](#ref-4)：

- advantage 不除组内标准差：\(A_i=r_i-\bar r\)；
- loss 用固定常数而非每条 response 长度归一化。

标准差归一化会放大奖励方差很小的组；对 binary reward，它隐式改变不同难度 prompt
的权重。sequence normalization

\[
L_{\text{seq}}=\frac1B\sum_i
\frac{\sum_t m_{i,t}\ell_{i,t}}{\sum_t m_{i,t}}
\]

让每条序列总权重近似相同，因而短序列的单 token 权重更大。constant normalization

\[
L_{\text{const}}=
\frac{\sum_{i,t}m_{i,t}\ell_{i,t}}{C}
\]

避免当前 response 长度进入分母，但 \(C\) 必须与 batch/预设最大长度契约一致。
这两类偏差共同解释了 R1-Zero 类训练中观察到的响应长度异常增长 [[4]](#ref-4)。

## 4. RFT：只模仿成功 rollout

Rejection Sampling Fine-Tuning / Reinforced Fine-Tuning 的简化形式是采样后只保留
verifier 通过的 response，再做 SFT，DeepSeekMath 的数学训练即包含这一阶段 [[2]](#ref-2)：

\[
L_{\text{RFT}}=-r_i\sum_t m_{i,t}\log\pi_\theta(y_{i,t}\mid x_i).
\]

它不使用组内负 advantage，直观稳定，但 early policy 成功率低时信号稀疏；还会把
“偶然猜中”当正例。应去重成功轨迹、检查推理有效性，并记录每 prompt 的成功样本数量。

## 5. MaxRL：按组平均奖励缩放

A5 的接口将 MaxRL 表示为 mean normalization：

\[
A_i^{\text{MaxRL}}=
\frac{r_i-\bar r}{\bar r+\epsilon}.
\]

当 \(\bar r\) 小时，少数成功样本被强烈放大，强调较难但并非完全无解的 prompt；当
\(\bar r=0\) 时分子也为零。优点是集中学习边界任务，风险是小 denominator、梯度尖峰
和 seed variance。必须监控 advantage 范围和 grad norm。

## 6. GSPO：sequence-level importance ratio

token ratio 的乘积在长序列上高方差，而逐 token clipping 又可能破坏整条 response
的一致性。GSPO 使用 response token 平均 log-ratio 的几何均值 [[5]](#ref-5)：

\[
\rho_i^{\text{GSPO}}
=\exp\left(
\frac{1}{|y_i|}\sum_t m_{i,t}
[\log\pi_\theta(y_{i,t})-\log\pi_{\text{old}}(y_{i,t})]
\right),
\]

再对 sequence ratio clipping，并把同一 ratio 用于该 response 的所有 token。它更贴近
sequence-level 更新稳定性，但仍是 surrogate；平均 log-ratio 会隐藏局部 token 的
极端变化。

## 7. On-policy 与 off-policy

- **On-policy**：生成后立即更新，ratio 约为 1，偏差较小、生成成本高。
- **Off-policy**：一个 rollout batch 复用多次，样本效率高，但 policy drift 增加
  importance-weight 方差 [[1]](#ref-1)。
- `old_log_probs` 必须来自真实 behavior policy；在训练途中重算或拿当前 policy 冒充，
  会使 ratio 校正失效。
- clipping 有意引入 bias 换稳定性；比较算法时同时报告 clip fraction、KL、ESS 和
  wall-clock。

## 8. Verifier 的证据层次

- **Outcome verifier** 只检查最终产物（答案等价、测试通过），训练信号密度低但
  鲁棒；**process verifier** 对每步推理打分，信号更密集但需要步级标注，Lightman 等
  证明其 pass@1 优势随任务难度增大 [[6]](#ref-6)。
- Verifier 本身是被攻击面：格式投机、答案字符串泄漏、sandbox 逃逸。设计上应保持
  held-out verifier 与训练 parser 的独立性。
- 数学/代码成功不证明 truthfulness、安全或现实世界可行动性；展示链式思维作为
  “真实性证据”不可靠 [[3]](#ref-3)。
- 课程化视角：Tulu 3 将 RLVR 排在 SFT/DPO 之后，作为“最后一公里”能力强化阶段，
  避免其与指令遵循、安全目标相互挤压 [[7]](#ref-7)。

## 关键公式速查

- GRPO：\(A_i=(r_i-\bar r)/(s_r+\epsilon)\)。
- Dr.GRPO：\(A_i=r_i-\bar r\)，并使用固定 loss normalization。
- RFT：\(L=-r_i\sum_tm_{i,t}\log\pi_\theta(y_{i,t})\)。
- MaxRL：\(A_i=(r_i-\bar r)/(\bar r+\epsilon)\)。
- GSPO：\(\rho_i=\exp(|y_i|^{-1}\sum_tm_{i,t}\log(\pi_\theta/\pi_{\text{old}}))\)。
- 零梯度组概率（binary reward）：\(p^G+(1-p)^G\)。

## 实现映射

| 概念 | 本仓库位置 | 形状/契约 |
| --- | --- | --- |
| prompt/response shift | `cs336_alignment/grpo.py::tokenize_prompt_and_output` | `[B,L]` inputs/labels/mask |
| token log-prob/entropy | `get_response_log_probs` | gather label log-prob，entropy `[B,L]` |
| verifier components | `compute_rollout_rewards` | total/format/answer 分开聚合 |
| GRPO/DrGRPO/RFT/MaxRL | `compute_group_normalized_rewards` | baseline `{mean,none}` × normalizer `{std,none,mean}` |
| token GRPO clip | `compute_policy_gradient_loss(..., method="grpo")` | \(\rho_{i,t}\) clipped |
| GSPO | 同函数 `method="gspo"` | mask 内平均 log-ratio，sequence clip |
| length normalization | `aggregate_loss_across_microbatch` | `{sequence,constant}` |
| 完整更新 | `grpo_train_step` | 全 batch padding、microbatch accumulation、grad clip |
| 报告 | `assignment5-alignment/report/main.tex` | 7 objectives×4 seeds 的 vectorized proxy；非 GSM8K 正式成绩 |

配置对应关系：

- 标准 GRPO：`baseline=mean, normalizer=std, loss=sequence`。
- Dr.GRPO：`baseline=mean, normalizer=none, loss=constant`。
- RFT：`baseline=none, normalizer=none`，reward 直接作权重。
- MaxRL：`baseline=mean, normalizer=mean`。
- GSPO：off-policy 时 `importance_reweighting_method=gspo`。

仓库在切 microbatch 前统一 padding，避免各 microbatch 长度不同导致 old log-prob
对齐变化；这是常见但隐蔽的正确性细节。

## 训练与诊断流程

1. 先为 verifier 写空输出、错误格式、等价答案、单位、浮点和注入攻击测试。
2. 固定 prompt、temperature/top-p、group size、最大长度与 parser。
3. 运行 prompting/RFT baseline，确认 base policy 存在非零成功率。
4. 50-step sanity：查看 rollouts、answer/format reward、entropy、grad norm 和长度。
5. 标准 GRPO 多 seed 后才做 Dr.GRPO/RFT/MaxRL 控制变量对照。
6. off-policy sweep 同时报 clip fraction、staleness、ratio 分布和每次 rollout 的
   更新次数。
7. held-out verifier 与人工抽样检查，确认不是 parser exploitation。

## 易错点、伦理与安全

- response mask 错一位，把最后 prompt token 或 padding 纳入 loss。
- `torch.std` 的有偏/无偏约定不同，导致小组结果明显变化。
- 整组同 reward 仍除极小 \(\epsilon\)，数值上放大本应为零的噪声。
- microbatch loss 又除 accumulation steps，或漏除，改变有效学习率。
- 只看 mean reward，不看 seed variance、entropy collapse 和长度漂移。
- verifier 接受“答案字符串出现在废话中”，模型学会格式投机。
- 代码 reward 运行不可信生成代码，造成文件、网络或资源攻击；必须沙箱、限时限资源。
- 数学/代码成功不证明 truthfulness、安全或现实世界可行动性。
- 展示链式思维作为“真实性证据”不可靠，也可能暴露敏感数据；评估应关注可核验产物。

## Checklist

- [ ] rollout policy 版本、old log-probs 和采样参数可追溯。
- [ ] group reshape 不跨 prompt；每 prompt 恰有 \(G\) 个样本。
- [ ] reward components、parse failures 与 total reward 分开记录。
- [ ] advantage 公式、std convention 和 zero-variance 组有单测。
- [ ] response mask、padding 和 shifted labels 有逐 token 测试。
- [ ] 同时报 reward、accuracy、KL、entropy、length、grad norm、clip fraction。
- [ ] 变体实验只改变一个 normalization/importance 因素。
- [ ] 至少多 seed，并保留失败 rollout 与 verifier adversarial audit。
- [ ] 代码执行 verifier 使用隔离沙箱。

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| loss 不降、reward 不动 | 零梯度组比例 | 组全对/全错、成功率极端、温度过低 |
| reward 升但 held-out 不升 | parser exploitation 审计 | 格式投机、verifier 泄漏 |
| 响应长度持续增长 | loss 归一化方式 | sequence normalization 的长度偏置 [[4]](#ref-4) |
| off-policy 训练发散 | ratio 分布与 clip fraction | old log-probs 来源错误、复用过久 |
| 梯度尖峰/NaN | advantage 范围 | MaxRL 小分母、零方差组除 \(\epsilon\) |
| 输出趋同、多样性下降 | entropy 轨迹 | entropy collapse、KL 约束缺失 |
| 长 response 训练不稳 | token vs sequence ratio | token 乘积方差过大，考虑 GSPO [[5]](#ref-5) |
| 各 seed 结论不一致 | 逐 seed reward 曲线 | 成功率接近边界、组信号稀疏 |

## 讨论：效度威胁与结论边界

### Construct validity

- “reward 上升”测的是对 \(V(x,y)\) 的优化程度，不是推理能力本身；
- benchmark 通过率与真实推理之间存在测试集效度问题；
- 响应长度、格式变化都可能被误读为推理深度。

### Internal validity

- 变体对比中同时改变 baseline、normalizer 与 loss 归一化时无法归因；
- old log-probs 与采样参数不可追溯时，off-policy 结论不可复核；
- 单 seed 的算法排序常被组信号方差淹没。

### External validity

- 数学/代码可验证任务上的结论不外推到开放式对话；
- 小模型+小 \(G\) 上的归一化选择，在大模型上可能因成功率分布不同而反转；
- “涌现推理”的叙事依赖于 verifier 分布，换一个 parser 结论可能消失 [[3]](#ref-3)。

论文式表述应报告：零梯度组比例、verifier 对抗审计结果、held-out 与训练 parser 的
独立性、多 seed 置信区间，以及明确声明结论仅在特定 verifier 与任务分布内成立。

## 面试要点速记

**高频问题与答题要点**

1. **Q：GRPO 为什么不需要 critic？** 要点：同一 prompt 采样 G 个 response，
   用组内相对奖励（均值-标准差归一化）作 advantage，以组基线替代 value network。
2. **Q：零信号组的概率？** 要点：binary reward 下全同概率
   \(p^G+(1-p)^G\)；隐式课程把信号集中在“可解但非稳解”的难度带。
3. **Q：Dr.GRPO 修正哪两处？** 要点：advantage 不除组内 std（避免放大低方差
   组）；loss 用常数而非逐序列长度归一（消除短序列单 token 权重放大与
   长度偏置）。
4. **Q：GSPO 的动机？** 要点：token 级 importance ratio 在长推理序列上方差
   过大；改为序列平均 log-ratio 的指数（几何均值）+ 序列级 clipping。
5. **Q：RFT 的风险？** 要点：只学成功 rollout → “偶然猜中”被当正例；早期
   成功率低时信号稀疏；需去重成功轨迹并审计推理有效性。

**必背数字**

- \(p^G+(1-p)^G\)；DeepSeek-R1 证明纯 rule-based reward 足以激励长链推理；
  RLVR 结论始终限定在 verifier 覆盖的分布内。

## 小结

RLVR 把可验证任务转成策略优化，但学习信号由采样、组内基线、尺度归一化和长度归一化
共同塑造。GRPO、Dr.GRPO、RFT、MaxRL 不是简单别名，它们对 prompt 难度和 response
长度施加不同权重；GSPO 则改变 off-policy 校正粒度。reward 曲线上升只有在 verifier、
行为分布和独立评估都可信时才能解释为能力提升。

## 参考文献

<a id="ref-1"></a>[1] J. Schulman, F. Wolski, P. Dhariwal, et al. “Proximal
Policy Optimization Algorithms.” arXiv:1707.06347, 2017.
[link](https://arxiv.org/abs/1707.06347)

<a id="ref-2"></a>[2] Z. Shao, P. Wang, Q. Zhu, et al. “DeepSeekMath: Pushing
the Limits of Mathematical Reasoning in Open Language Models.” arXiv:2402.03300,
2024. [link](https://arxiv.org/abs/2402.03300)

<a id="ref-3"></a>[3] DeepSeek-AI (D. Guo, D. Yang, H. Zhang, et al.).
“DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement
Learning.” arXiv:2501.12948, 2025. [link](https://arxiv.org/abs/2501.12948)

<a id="ref-4"></a>[4] Z. Liu, C. Chen, W. Li, et al. “Understanding
R1-Zero-Like Training: A Critical Perspective.” arXiv:2503.20783, 2025.
[link](https://arxiv.org/abs/2503.20783)

<a id="ref-5"></a>[5] C. Zheng, S. Liu, M. Li, et al. “Group Sequence Policy
Optimization.” arXiv:2507.18071, 2025. [link](https://arxiv.org/abs/2507.18071)

<a id="ref-6"></a>[6] H. Lightman, V. Kosaraju, Y. Burda, et al. “Let's
Verify Step by Step.” *ICLR*, 2024. [link](https://arxiv.org/abs/2305.20050)

<a id="ref-7"></a>[7] N. Lambert, J. Morrison, V. Pyatkin, et al. “Tulu 3:
Pushing Frontiers in Open Language Model Post-Training.” arXiv:2411.15124,
2024. [link](https://arxiv.org/abs/2411.15124)

## 延伸阅读与复现材料

- Stanford CS336, [Lecture 16 — RLVR](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf)
- [Alignment 主题导航](../experiments/topics/alignment.md)
- A5 主 handout：`../assignments/spring2026/assignment5-alignment/cs336_spring2026_assignment5_alignment.pdf`（及同目录 `grpo.py` 与 `report/`）
