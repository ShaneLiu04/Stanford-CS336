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
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
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
- 格式奖励被滥用：模型输出凑格式而非解题——奖励必须对齐真实目标，并用独立
  评测集监控（训练 reward 升而 held-out 不升即 hacking 信号）。
- off-policy 数据复用过久：staleness 增大导致 importance ratio 方差爆炸，
  须监控 staleness 分布并限制每个 rollout batch 的更新次数。

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
6. **Q：R1 为什么需要冷启动 SFT？R1-Zero 不是证明不需要吗？** 要点：
   R1-Zero 证明能力上不需要（AIME 2024 15.6%→86.7%，纯规则奖励、无冷启动）；
   但基座直接 RL 输出可读性差、语言混杂；R1 四阶段 = 冷启动 SFT → 推理 RL →
   拒绝采样（600k 推理 + 200k 通用）全场景 SFT → RLVR。
7. **Q：RLVR 里 reward hacking 长什么样？** 要点：格式奖励被滥用（凑格式
   而非解题）、弱测试用例让错误解得高分、长度奖励导致冗长；对策：奖励对齐
   真实目标、独立评测集、规则收紧。
8. **Q：异步 RL 的 staleness 是什么？** 要点：rollout 数据相对当前权重的
   陈旧度，是异步 RL 核心监控量；用 importance sampling π_θ/μ 校正
   off-policy 偏差；staleness 过大时 ratio 方差爆炸。
9. **Q：蒸馏 vs 同规模直接 RL？** 要点：R1 蒸馏 Qwen-32B 优于同规模直接
   RL——强教师蒸馏性价比高；600k+200k 蒸馏配方即第三阶段数据的复用。

**必背数字**

- \(p^G+(1-p)^G\)；DeepSeek-R1 证明纯 rule-based reward 足以激励长链推理；
  RLVR 结论始终限定在 verifier 覆盖的分布内。

**工业界参照**

- R1-Zero（DeepSeek，2025-01）：纯 RL、规则奖励（答案正确性+格式）、无冷启动；
  AIME 2024 准确率 15.6%→86.7%；奖励曲线与响应长度同步增长。
- R1 四阶段：冷启动 SFT → 大规模推理 RL → 拒绝采样 600k 推理 + 200k 通用
  样本全场景 SFT → RLVR（规则奖励、覆盖多任务）。
- 蒸馏对照：R1 蒸馏（Qwen-32B）> 同规模直接 RL——强教师蒸馏性价比高。
- GSPO 公开案例：CUA-Gym 训练后 OSWorld 62.2%→72.6%。
- RL rollout 的 prefix 重复量级 80–95%：同组 G 条共享 prompt 前缀，
  prefix caching 复用 KV 对训练系统收益巨大。
- verl（Volcano Engine）：hybrid controller + worker 异构资源池；
  vLLM/SGLang rollout；FSDP/Megatron 训练后端。

## 行业现状与最新进展（2024–2026）

### R1-Zero 纯 RL 里程碑与 R1 四阶段课程

- R1-Zero（DeepSeek，2025-01）：不经过任何冷启动 SFT，直接在基座上用
  规则奖励（答案正确性 + 格式）做大规模 RL；AIME 2024 准确率 15.6%→86.7%。
- 训练中自发涌现 aha moment（“Wait，让我重新评估”式自我反思、主动分配
  更多思考时间）；训练奖励曲线与响应长度同步增长。
- R1 完整版在 R1-Zero 之上补了课程，四阶段如下：

| 阶段 | 方法 | 数据/奖励 | 目的 |
| --- | --- | --- | --- |
| ① 冷启动 | 少量长 CoT SFT | 高质量长推理样本 | 防 RL 从基座启动的可读性差/语言混杂 |
| ② 推理 RL | 大规模 RL | 规则奖励（正确性+格式） | 激励长链推理、涌现反思行为 |
| ③ 全场景 SFT | 拒绝采样 + 通用数据 | 600k 推理 + 200k 通用 | 恢复通用能力、清洗推理语料 |
| ④ RLVR | 再次 RL | 规则奖励、覆盖多任务 | 多任务能力强化与对齐 |

- 蒸馏对照：R1 蒸馏到 Qwen-32B 优于同规模模型直接做 RL——强教师蒸馏
  性价比高；600k+200k 蒸馏配方即第三阶段数据的复用。

### GRPO 偏差修正族与序列级变体

| 算法 | 核心机制 | 修正的问题 | 适用场景 |
| --- | --- | --- | --- |
| GRPO | 组内均值归一 advantage，免 critic | 省去 value 网络（每 token 级省 O(模型参数量) 显存） | 数学/代码等可验证域 |
| Dr.GRPO | 去掉组内 σ 与逐序列长度归一 | 组权重偏差、长度偏差（防“长答案占优”） | 长 CoT 训练稳定性 |
| DAPO | dynamic sampling 丢弃全同奖励组 | 全对/全错组无梯度、组内不可比 | 提高有效样本利用率 |
| GSPO | 序列级重要性比（token 级 KL 替换） | token 级 ratio 长序列方差过大 | 多模态/GUI；CUA 案例 OSWorld 62.2%→72.6% |

### RL 基建与异步化

- verl（Volcano Engine 开源）：hybrid controller（单控制器编排）+ worker
  异构资源池；rollout 支持 vLLM/SGLang，训练后端支持 FSDP/Megatron，
  内置 PPO/GRPO 等主流算法。
- 异步 RL：rollout 与训练解耦提升吞吐，但 rollout 数据相对当前权重的
  陈旧度（staleness）成为核心监控量；用 importance sampling π_θ/μ 校正
  off-policy 偏差，staleness 过大时 ratio 方差爆炸。
- prefix 共享：GRPO 同组 G 条 rollout 共享 prompt 前缀，RL 场景 80–95%
  token 重复的量级，prefix caching 复用 KV 对训练系统收益巨大。
- Agentic RL 综述口径：环境噪声与 OOD（分布漂移）是轨迹崩坏主因；
  课程/难度过滤与回放缓冲是稳定手段。

### 对本讲学习者的启示

R1-Zero 证明“信号 > 数据”：只要 verifier 覆盖任务分布，规则奖励足以激励
长链推理；但 R1 最终仍回归四阶段课程，说明纯 RL 的可读性与通用性损失需要
SFT 修复。对本讲的意义：GRPO/Dr.GRPO/MaxRL 的归一化选择不是学术洁癖，
而是工业训练中长度偏置与难度加权的直接来源；异步化与 prefix 共享则说明
RLVR 的瓶颈一半在算法、一半在系统。学完本讲应能同时从“目标函数”与
“训练基建”两个视角解释任何一条 RLVR 训练曲线。

## 大厂面试真题与答题框架

高频面试题（公开面经风格）。

**题目 1：GRPO 为什么能省掉 critic？省在哪？**
- 考点：组内相对优势 vs value network；显存账。
- 答题框架：① 同一 prompt 采样 G 条 → ② 组内均值作 baseline，
  advantage=(r_i−r̄)（可再除 σ）→ ③ 组基线替代 value 网络的方差缩减
  作用 → ④ 显存：省一个与 policy 同规模的 value model，每 token 级省
  O(模型参数量) 显存。
- 加分项：指出全同奖励组无梯度（p^G+(1-p)^G），构成隐式难度课程；
  提 DAPO 用 dynamic sampling 丢弃全同组保持组内可比。
- 踩坑：说“GRPO 没有 baseline”——组均值就是 baseline；混淆 on-policy
  场景下是否需要 importance ratio。

**题目 2：规则奖励 vs PRM（process reward model），各自何时用？**
- 考点：outcome vs process verifier 的证据强度与成本。
- 答题框架：① 规则奖励：可验证域（数学答案比对、代码单测），零成本、
  信号客观 → ② PRM：步级信号密集，适合不可程序化验证的推理 →
  ③ PRM 本身是被攻击面，可能引入偏好偏差 → ④ 工业优先规则奖励，
  PRM 用于不可验证域或过程诊断。
- 加分项：R1-Zero 用纯规则奖励（正确性+格式）把 AIME 2024 从 15.6%
  拉到 86.7%，说明可验证域里 PRM 非必需。
- 踩坑：把 PRM 当“更高级”的默认选项；忽略 PRM 标注成本与被 hacking 风险。

**题目 3：reward hacking 在 RLVR 里长什么样？怎么防？**
- 考点：verifier 是被攻击面；格式投机。
- 答题框架：① 典型形态：格式奖励被滥用（输出凑格式而非解题）、测试用例
  弱导致错误解得高分、长度奖励导致冗长 → ② 诊断：训练 reward 升但
  held-out 独立评测不升 → ③ 对策：奖励对齐真实目标、独立评测集、规则
  收紧 → ④ 代码域必须沙箱 + 独立测试集。
- 加分项：引用 Dr.GRPO 的长度偏置分析——逐序列长度归一化本身就会让
  模型学会“写得长”。
- 踩坑：只报 mean reward 不做 held-out 对照；认为规则奖励天然免疫
  hacking。

**题目 4：R1 为什么在推理 RL 之后还要拒绝采样 + 全场景 SFT？**
- 考点：R1 四阶段课程设计；纯 RL 的副作用。
- 答题框架：① 纯 RL（R1-Zero 路线）可读性差、语言混杂、通用能力受损
  → ② 拒绝采样从 RL 后策略采出高质量推理样本（600k）+ 200k 通用样本
  → ③ 全场景 SFT 同时恢复推理与通用能力 → ④ 再做 RLVR（规则奖励、
  覆盖多任务）收尾。
- 加分项：蒸馏对照——R1 蒸馏 Qwen-32B 优于同规模直接 RL，说明第三阶段
  数据（600k+200k）本身就是强蒸馏配方。
- 踩坑：把四阶段记成流水账，答不出“每阶段修什么缺陷”。

**题目 5：冷启动 SFT 为什么必要？R1-Zero 不是证明不需要吗？**
- 考点：R1-Zero vs R1 的对照；可读性问题。
- 答题框架：① R1-Zero 证明能力上不需要冷启动（AIME 15.6%→86.7%）→
  ② 但基座直接 RL 输出可读性差、语言混杂 → ③ 少量长 CoT 冷启动固定
  输出风格与语言一致性 → ④ R1 用冷启动换工程稳定性，不换能力上限。
- 加分项：aha moment 在 R1-Zero 中自发涌现（自我反思、分配更多思考
  时间），说明推理行为不需要监督数据注入。
- 踩坑：把“纯 RL 可行”错答成“纯 RL 全场景最优”。

**题目 6：异步 RL 的 staleness 是什么？怎么校正？**
- 考点：off-policy 程度；importance sampling。
- 答题框架：① rollout 与训练解耦提升吞吐 → ② rollout 用的权重 μ 落后
  当前 θ，staleness = 数据陈旧度，是核心监控量 → ③ importance sampling
  π_θ/μ 校正 off-policy 偏差 → ④ staleness 过大时 ratio 方差爆炸，
  需 clip、丢弃或限制复用步数。
- 加分项：提 verl 的 hybrid controller 架构如何在异步流水线下保证数据
  契约（old log-probs 与采样参数可追溯）。
- 踩坑：以为异步只是工程问题、没有算法代价；拿当前 policy 冒充
  old log-probs。

**题目 7：RL rollout 里 prefix caching 为什么收益巨大？**
- 考点：GRPO 组采样结构；系统-算法协同。
- 答题框架：① GRPO 同组 G 条 rollout 共享 prompt 前缀 → ② RL 场景
  80–95% token 重复的量级 → ③ prefix caching 复用 KV，rollout 吞吐
  大幅提升 → ④ 组内共享使 GRPO 比逐条采样天然更适配 caching。
- 加分项：rollout 引擎（vLLM/SGLang）对 prefix caching 的支持差异；
  长思维链场景 response 占比升高时收益结构变化。
- 踩坑：只从“KV cache 复用”作答，答不出 GRPO 组结构才是收益来源。

## 系统设计题

**设计题 1：为代码能力设计一套 RLVR 训练系统**
- 需求澄清：目标能力（竞赛题/工程任务/API 使用）？基座规模与现有 SFT
  水平？单测从哪来（人工标注/自动生成+校验）？是否含 agentic 多轮
  工具调用？
- 规模估算：参照 R1 类训练——推理 RL 阶段需大规模 prompt 池；拒绝采样
  阶段 600k 量级推理样本（按任务域缩放）；代码执行需为每条 rollout
  提供隔离沙箱（限时限资源）。
- 架构：prompt 池（带难度标签与单测）→ rollout（verl 类基建：
  vLLM/SGLang rollout + FSDP/Megatron 训练后端）→ 沙箱执行单测 →
  规则奖励（pass/fail，可加编译错误部分反馈）→ GRPO 更新 → 定期拒绝
  采样回灌 SFT 数据池。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 取舍 |
| --- | --- | --- | --- |
| 奖励 | 纯单测 pass/fail | + 格式/风格奖励 | B 信号密但可被格式投机钻营 |
| 组大小 G | 小（4–8） | 大（16+） | 大组基线稳但 rollout 贵 |
| 执行 | 本地容器 | 远程沙箱池 | 远程安全但延迟高、吞吐受限 |
| 采样 | 同步 step | 异步流水 | 异步吞吐高但 staleness 需 IS 校正 |

- 评测方案：held-out 题库（与训练池独立、防泄漏）；独立 verifier
  （不用训练单测）；对比 prompting/RFT baseline；多 seed。
- 追问预案：reward hacking（测试用例弱让错解得高分）→ 收紧规则、
  独立评测集；沙箱逃逸 → 网络隔离、限时限资源；长度爆炸 → 检查
  长度归一化（Dr.GRPO 口径）。

**设计题 2：设计 GRPO 训练基建（rollout 引擎与组采样）**
- 需求澄清：模型规模？G 与 batch？on-policy 还是允许 1-step off-policy？
  是否长 CoT（response 万 token 级）？
- 规模估算：免 critic 每(token 级)省 O(模型参数量) 显存；同组 prefix
  重复 80–95% 量级 → prefix caching 收益巨大。
- 架构：hybrid controller（单控制器编排）+ worker 异构资源池（rollout
  用 vLLM/SGLang、训练用 FSDP/Megatron）；组采样调度：同 prompt 的 G 条
  进同一 rollout batch 以共享 prefix；weight 同步通道；old log-probs 与
  采样参数全链路可追溯。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 取舍 |
| --- | --- | --- | --- |
| rollout 引擎 | vLLM | SGLang | prefix caching/调度各有差异，需实测 |
| 采样 | 同步 step | 异步流水 | 异步吞吐高但需 staleness 监控 + IS 校正 |
| 全同组处理 | 直接丢弃（无梯度） | dynamic sampling 补采 | 补采保组内可比（DAPO 口径） |
| ratio 粒度 | token 级 | sequence 级（GSPO） | 长序列 token 级方差大 |

- 评测方案：监控零梯度组比例、clip fraction、staleness 分布、rollout/
  训练利用率；多 seed 对照归一化变体。
- 追问预案：staleness 过大 → 限制复用步数 + importance sampling π_θ/μ
  校正；prefix cache 失效 → 检查组调度是否打散同组样本。

**设计题 3：为 GUI/Agent（CUA）场景设计 RLVR 训练**
- 需求澄清：环境形态（截图 + 动作空间）？奖励来源（任务完成校验）？
  episode 长度与多轮结构？
- 规模估算：公开案例量级——序列级算法（GSPO 口径）训练 CUA 后 OSWorld
  62.2%→72.6%；多模态长序列 → 序列级 ratio 的必要性上升。
- 架构：环境（模拟器/GUI 沙箱）→ 多轮 rollout（截图输入、动作输出）→
  任务完成规则奖励 → 序列级算法（token 级 KL 替换为序列级重要性比）→
  课程/难度过滤与回放缓冲。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 取舍 |
| --- | --- | --- | --- |
| 算法 | GRPO（token 级） | GSPO（序列级） | 长多模态序列 token 级方差大 |
| 环境 | 真实网站 | 模拟器 | 模拟器可控但分布漂移（OOD） |
| 奖励 | 仅任务完成 | + 过程启发式 | 过程奖励信号密但引入 hacking 面 |

- 评测方案：held-out 环境（OSWorld 类基准）；轨迹人工抽检；监控环境
  噪声与 OOD 导致的轨迹崩坏率。
- 追问预案：OOD 轨迹崩坏 → 难度过滤 + 回放缓冲；环境噪声 → 多次采样
  取多数奖励校验。

## 代码实现题

**实现题 1：GRPO loss（组内标准化 advantage + token 级 clip）**

考察点：组 reshape、零方差组、response mask、clip。

```python
def grpo_loss(logprobs, old_logprobs, advantages, response_mask, eps=0.1):
    # logprobs/old_logprobs: [B, L] 当前/行为 policy 的 token log-prob
    # advantages: [B]（组内已标准化），response_mask: [B, L]
    adv = advantages.unsqueeze(1)                    # [B, 1] 广播到 token 级
    ratio = torch.exp(logprobs - old_logprobs)       # rho_{i,t}
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    per_token = -torch.min(unclipped, clipped)       # [B, L]
    # 只对 response token 求平均；全同奖励组 advantage=0，天然无梯度
    loss = (per_token * response_mask).sum() / response_mask.sum()
    return loss

def group_advantages(rewards, eps=1e-6):
    # rewards: [N, G] 每 prompt G 条
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)           # 注意有偏/无偏约定
    adv = (rewards - mean) / (std + eps)             # 零方差组优势≈0
    return adv.reshape(-1)
```

验收标准：① 零方差组 loss 恰为 0；② on-policy（ratio≡1）时退化为
REINFORCE 口径；③ mask 错位时单测必须失败；④ 组内均值/标准差与手算
对拍一致。

**实现题 2：规则奖励函数（答案匹配 + 格式奖励，含防作弊）**

考察点：verifier 是被攻击面；奖励组件分离；防格式投机。

```python
import re

def rule_reward(response, gold_answer, extract_fn):
    # 格式奖励：有 <answer>...</answer> 才给小额奖励
    m = re.search(r"<answer>(.*?)</answer>", response, re.S)
    if m is None:
        return {"format": 0.0, "answer": 0.0, "total": 0.0}
    fmt = 0.1
    # 答案奖励：只信任提取归一后的答案，绝不全文匹配（防答案串出现在废话中）
    pred = extract_fn(m.group(1).strip())
    gold = extract_fn(gold_answer)
    ans = 1.0 if pred is not None and pred == gold else 0.0
    # 防作弊：gold 答案串在答案块之外反复出现 → 判投机，整体归零
    outside = re.sub(r"<answer>.*?</answer>", "", response, flags=re.S)
    if gold_answer in outside:
        ans, fmt = 0.0, 0.0
    return {"format": fmt, "answer": ans, "total": fmt + ans}
```

验收标准：① 空输出/错误格式得 0；② 等价表述（分数/小数、单位）经
extract_fn 归一后得分一致；③ 答案串出现在正文废话中不得分；④ 组件
（format/answer/total）分开记录以便审计格式投机。

**实现题 3：拒绝采样过滤器（按正确性采样，每 prompt 上限 N 条）**

考察点：R1 第三阶段/蒸馏数据管线的核心组件；去重与多样性。

```python
import random

def rejection_sample(prompt, policy, verifier, k=64, keep=4, max_len=4096):
    # 采样 k 条、只保留 verifier 通过的、按答案去重后最多 keep 条
    samples = policy.generate(prompt, n=k, max_tokens=max_len)
    kept, seen = [], set()
    for resp in samples:
        ok, answer = verifier(prompt, resp)      # 规则奖励判定
        if not ok:
            continue
        if answer in seen:                       # 同答案去重（多样性）
            continue
        seen.add(answer)
        kept.append({"prompt": prompt, "response": resp, "answer": answer})
        if len(kept) >= keep:
            break
    return kept

def build_sft_pool(prompts, policy, verifier, keep_per_prompt=4):
    # R1 第三阶段口径：拼出 600k 量级推理样本（按任务域缩放）
    pool = []
    for p in prompts:
        pool.extend(rejection_sample(p, policy, verifier, keep=keep_per_prompt))
    return pool
```

验收标准：① 每 prompt 输出 ≤ keep 条且全部通过 verifier；② 同答案
去重生效；③ 零通过率的 prompt 单独记录，供难度过滤/课程使用；
④ 固定 seed 可复现。

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
- [DeepSeekMath（GRPO 原始论文）](https://arxiv.org/abs/2402.03300)（访问日期 2026-10-04）
- [DeepSeek-R1（含 R1-Zero 与四阶段课程）](https://arxiv.org/abs/2501.12948)（访问日期 2026-10-04）
- [verl（Volcano Engine 开源 RL 训练框架）](https://github.com/volcengine/verl)（访问日期 2026-10-04）
- [Stanford CS336 课程主页](https://cs336.stanford.edu)（访问日期 2026-10-04）
