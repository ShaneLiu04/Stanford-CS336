---
title: "Lecture 15 — SFT & RLHF"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-05-18"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf"
  - "../assignments/spring2026/assignment5-alignment/"
---

# Lecture 15 — SFT 与 RLHF：从模仿到偏好优化

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
- 适用对象：自学者、对齐/后训练工程师与研究者

## 摘要

后训练把一个会预测 token 的基座模型变成一个能遵循指令、拒答有害请求并被人类偏好的助手。
本文沿“模仿 → 偏好 → 策略优化”的主线系统梳理四个环节：以 response-only likelihood 为
目标的 SFT、以 Bradley–Terry 为基础的 pairwise reward modeling、以 KL 约束与 clipped
surrogate 为核心的 PPO-RLHF，以及把 KL-regularized 偏好优化闭式化成分类损失的 DPO。
对每种方法给出精确的目标函数、数据契约（prompt/response mask、chosen/rejected 共享
prompt、reference policy 冻结）、实现陷阱（长度偏置、模板泄漏、log-prob 求和口径）与
失败模式（reward hacking、coverage 不足、helpful/harmless 冲突）。工程上给出可复核的
实验闭环——固定 base/template/sampler/parser，capability 与 safety 分轴评估、多 seed
报告；研究上讨论偏好数据的构造效度、自动 judge 的系统性偏差与结论边界。核心论点是：
“对 reward 优化”不等于“对人的真实意图对齐”，任何对齐结论都必须限定到 rubric、分布
与评估协议。

**关键词：** SFT；Reward Modeling；Bradley–Terry；PPO；RLHF；DPO；Preference Data；
Reward Hacking；KL Constraint；Alignment

## 本文贡献

1. 统一四种方法的数据形态、目标函数与约束机制（模仿、偏好压缩、在线优化、离线偏好分类）；
2. 给出 SFT mask/DPO log-prob/RM split 的精确数据契约与常见实现错误清单；
3. 区分 KL 约束与 PPO clipping 两个不同层面的 drift 控制及其失效信号；
4. 把 reward hacking、长度偏置、judge 偏好与 helpful/harmless 冲突形式化为可检测的指标；
5. 提供从 SFT 到 PPO/DPO 的实验闭环、诊断速查表与论文级对齐评估规范。

## 学习目标

1. 区分 SFT、reward modeling、PPO-RLHF 和 DPO 的数据、目标与失败模式；
2. 能正确构造 response-only loss、pairwise preference loss 与 reference-policy 约束；
3. 会把自动指标、人工偏好和安全指标拆开报告；
4. 理解“对 reward 优化”不等于“对人的真实意图对齐”；
5. 能为一次对齐实验写出预注册的评估协议与失败判据。

## 先修知识

- Lecture 01–04：基座模型的 tokenization、架构与训练目标（后训练的起点）。
- Lecture 12：评估协议、judge 偏差与不确定性（对齐结论的可信度来源）。
- 概率基础： logistic 模型、KL 散度、策略梯度。

## 相关工作与技术谱系

RLHF 的方法论源自深度 RL 中从人类偏好学习的经典工作：Christiano 等证明少量成对偏好
即可训练符合人类判断的 reward model [[1]](#ref-1)。InstructGPT 把该范式引入语言模型，
确立了 pretrain → SFT → RM → PPO 的标准管线，并系统报告了 helpfulness 与
truthfulness 的权衡 [[2]](#ref-2)；Anthropic 的 HH 系列则在更大规模上研究了
helpful 与 harmless 两类偏好的显式冲突，以及以“rubric + 标注分歧”管理安全边界的
方式 [[3]](#ref-3)。

算法层面，PPO 的 clipped surrogate 为策略梯度提供了无需二阶信息的稳定更新
[[4]](#ref-4)，成为 RLHF 在线优化的默认选择；DPO 则证明在 KL-regularized 偏好
优化下，reward 可被最优策略闭式表示，从而把 RL 问题化简为对 chosen/rejected
log-odds 的分类损失 [[5]](#ref-5)。

数据与实践层面，LIMA 表明约 1000 条高质量、多样且风格一致的示范即可让基座模型产生
显著的指令遵循能力——对齐能力主要来自预训练，SFT 更像“激活接口”
[[6]](#ref-6)；LoRA 等参数高效微调则通过低秩增量把 SFT/偏好优化的显存成本降低一个
量级，成为多任务后训练的基础设施 [[7]](#ref-7)。

## 1. SFT：先让模型学会任务接口

对 prompt \(x\) 和示范 response \(y=(y_1,\ldots,y_T)\)，常用 response-only 目标：

\[
\mathcal L_{\text{SFT}}
=-\sum_{t=1}^{T}m_t\log\pi_\theta(y_t\mid x,y_{<t}),
\]

其中 \(m_t=1\) 仅覆盖 response token。SFT 的价值不只是知识注入，还包括对话模板、
格式、拒答边界和任务风格。LIMA 的证据表明：数据的质量、多样性与风格一致性远比数量
重要，且对齐能力的大部分已存在于预训练分布中 [[6]](#ref-6)。

关键工程点：

- prompt 与 response 分开 tokenize，再拼接，避免字符串边界与 token 边界混淆。
- shift 后 mask 的第一个 response 位置是 `prompt_length - 1`。
- packing 提高利用率，但必须明确 EOS、跨文档 attention 和 prompt-token loss。
- 数据质量通常比盲目增加 instruction 数量更重要；应去模板泄漏和冲突示范。
- 全参数微调并非唯一选择：LoRA 冻结基座、只训练低秩增量 \(\Delta W=BA\)，可在接近
  全参微调质量下大幅降低 optimizer state 与显存开销，并便于多任务/多适配器管理
  [[7]](#ref-7)。

## 2. Preference data：偏好不是绝对分数

对同一 prompt 收集 \((y_w,y_l)\)，表示 annotator 更偏好 winner。需记录 rubric、annotator
群体、随机化顺序和 disagreement。偏好可能被长度、礼貌、格式和自信语气等捷径影响；
InstructGPT 与 HH 的经验都表明标注指南（rubric）本身就是对齐目标的显式定义
[[2]](#ref-2)[[3]](#ref-3)。

Bradley–Terry reward model 假设

\[
P(y_w\succ y_l\mid x)
=\sigma\!\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right),
\]

对应损失

\[
\mathcal L_{\text{RM}}
=-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)).
\]

pairwise accuracy 高不代表 reward calibrated，更不保证分布外可靠。需要按任务、安全类别
和长度评估，并检查 reward margin。

## 3. PPO-RLHF：在 reward 与 reference 之间优化

经典流程是：pretrain → SFT → preference/RM → PPO [[2]](#ref-2)。策略常优化

\[
\max_\theta\;
\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
\left[r_\phi(x,y)
-\beta\log\frac{\pi_\theta(y\mid x)}
{\pi_{\text{ref}}(y\mid x)}\right].
\]

PPO clipped surrogate 为 [[4]](#ref-4)：

\[
L^{\text{clip}}(\theta)=
\mathbb E_t\left[
\min(\rho_tA_t,\,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)
\right],
\quad
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\text{old}}(a_t\mid s_t)}.
\]

KL 约束控制 policy drift，PPO clipping 控制单次更新；两者角色相关但不等价。还需
value/critic 估计 advantage，因此系统复杂、显存和稳定性成本高。

### 3.1 Reward hacking：优化目标不是目标

Reward hacking 指 policy 找到 RM 的系统性漏洞而非真正提升质量：变长、堆砌好词、
自信语气、模仿 RM 训练分布的风格。InstructGPT 观察到 PPO 之后的模型在 RM 分数上
远超人类示范，但人工评估并未同比例提升 [[2]](#ref-2)；HH 训练也显示无害性与有用性
存在真实的偏好冲突，需要 rubric 与多目标评估显式管理 [[3]](#ref-3)。检测手段：

- reward 与人工 win rate 的斜率随训练下降（overoptimization 信号）；
- 长度/format 指标与 reward 同步飙升；
- held-out RM 与训练 RM 的分数分歧；
- KL 快速逼近上限而 capability 下降。

## 4. DPO：把 KL-regularized 偏好优化化成分类

DPO 在特定 reward/preference 和最优策略关系下，直接优化 [[5]](#ref-5)：

\[
\mathcal L_{\text{DPO}}=
-\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right]\right).
\]

\(\beta\) 控制 reference-relative log-odds 的尺度。实现时 chosen/rejected 必须共享
prompt；只累计 response token log-prob，并让 reference 前向 `no_grad`。DPO 免去显式
RM 和 online rollout，但没有免去偏好假设、coverage 和数据偏差：它仍然是在“对偏好
数据 + KL 约束”优化，只是把推断 reward 的步骤闭合进了损失函数。

## 5. 四种方法如何选

| 方法 | 所需数据 | 优点 | 主要风险 |
| --- | --- | --- | --- |
| SFT | 高质量 demonstrations | 简单稳定，建立接口 | 只会模仿覆盖到的行为 |
| RM | pairwise preferences | 得到可复用 score | reward hacking、失准 |
| PPO-RLHF | RM + online rollouts | 可探索当前策略的新输出 | 系统复杂、方差和训练不稳定 |
| DPO | offline preference pairs | 简洁、无需 critic | offline coverage、reference/长度偏差 |

合理顺序通常是先建立 SFT 与固定评估，再根据是否有可靠在线 reward、生成预算和探索
需求选择 PPO 或 DPO。二者并非互斥：实践中常见 PPO 与 DPO 系方法在同一流水线的不同
阶段混用，比较时必须固定 base、数据与评估协议。

## 6. 关键公式速查

- SFT：\(-\sum_tm_t\log\pi_\theta(y_t\mid x,y_{<t})\)。
- RM：\(-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\)。
- PPO ratio：\(\rho_t=\pi_\theta(a_t\mid s_t)/\pi_{\text{old}}(a_t\mid s_t)\)。
- RLHF 正则目标：\(\mathbb E[r_\phi-\beta\log(\pi_\theta/\pi_{\text{ref}})]\)。
- DPO：对 policy 与 reference 的 chosen/rejected log-odds 差做 logistic loss。
- LoRA：\(W=W_0+BA,\ B\in\mathbb R^{d\times r},\ A\in\mathbb R^{r\times d'},\ r\ll d\)。

## 7. 实现映射（本仓库）

| 概念 | 本仓库位置 | 关键行为 |
| --- | --- | --- |
| Packed SFT | `cs336_alignment/supplement.py::PackedSFTDataset` | Alpaca 模板、EOS、连续 token stream 切块 |
| SFT shift | `PackedSFTDataset.__getitem__` | 取 `seq_length+1`，构造 inputs/labels |
| 指标 parser | `supplement.py::parse_mmlu_response`, `parse_gsm8k_response` | 格式失败与答案失败应分开 |
| DPO log-prob | `supplement.py::_sequence_log_probability` | token log-prob 求和，reference no-grad |
| DPO loss | `compute_per_instance_dpo_loss` | policy preference margin 减 reference margin |
| RL token mask | `cs336_alignment/grpo.py::tokenize_prompt_and_output` | response mask 与 shifted labels 对齐 |
| 实验边界 | `assignment5-alignment/report/main.tex` | 26 tests 和 vectorized proxy，不是正式 1B/B200 结果 |

注意：当前 `PackedSFTDataset` 把完整模板（含 prompt）都放入 labels，属于 full-sequence
LM loss；Lecture 中常见的 response-only SFT 需要额外 mask。当前 DPO helper 对格式化
后的完整 chosen/rejected 序列求和；因 prompt 相同理论上差分会抵消，但不同 response 会
改变上下文长度和浮点路径，生产实现仍应显式 response mask。

## 8. 实验闭环

1. 固定 base/reference checkpoint、chat template、sampling config、parser 和 held-out split。
2. SFT 前后同时测 capability、instruction following、拒答、安全和 calibration。
3. RM 报 pairwise accuracy、margin、分组误差和 adversarial preference。
4. PPO 报 reward、KL、entropy、clip fraction、value loss、长度和人工 win rate。
5. DPO 报 chosen/rejected margin、reference KL proxy、长度与 held-out preference accuracy。
6. 所有方法至少多 seed；不得只展示 reward 最高 checkpoint。

## 9. 易错点、伦理与安全

- 对 prompt token 也计 SFT loss，却声称是 response-only。
- chosen/rejected 使用不同 prompt template，DPO 学到模板差异。
- sequence log-prob 直接求和造成长度偏好；不能未经审计改成平均，因为那也改变目标。
- RM train/test 来自同一 prompt 或同一生成簇，评估泄漏。
- 自动 judge 偏爱更长、更自信或与自身同族模型的输出。
- KL 系数过小导致 reward hacking，过大则几乎不学习。
- 将“helpful”和“harmless”压成单分数，掩盖对某群体的性能退化 [[3]](#ref-3)。
- 标注者接触创伤性内容；需限量、支持退出并减少原文传播。
- 对齐数据可能含个人对话和敏感请求，许可与同意应独立审计。
- 把 DPO 的“无需 RL”误读为“无需偏好假设”——它继承了偏好数据的一切偏差 [[5]](#ref-5)。
- 把“蒸馏小模型跑分高”误读为“RL 没用”：R1-Zero 证明纯 RL 可行（AIME
  15.6%→86.7%），蒸馏优势的前提是教师显著强于学生、学生自身探索能力不足。
- 把合成数据当免费午餐：三角色合成与两步安全过滤若缺少规则化校验层，错误
  轨迹与有害样本会直接进入 SFT 目标，“合成”不等于“已质检”。

## 10. Checklist

- [ ] chat template、EOS、padding side 和 loss mask 有单测。
- [ ] SFT packing 不跨越不应共享 attention 的安全边界。
- [ ] preference pair 共享 prompt，顺序随机化并记录 disagreement。
- [ ] RM 有独立 prompt-level split 与 calibration/分组分析。
- [ ] reference policy 固定，log-prob tokenizer/template 完全一致。
- [ ] PPO 同时记录 reward、KL、entropy、clip/value/length。
- [ ] DPO 检查长度、格式和 source shortcuts。
- [ ] capability 与 safety 分轴评估，并保留人工盲测。

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| SFT loss 降但指令不跟随 | mask 与 template 审计 | prompt token 计损、模板不一致 |
| RM accuracy 高但 PPO 后变差 | reward-human 斜率 | reward hacking、RM OOD |
| PPO reward 上升 KL 爆炸 | \(\beta\) 与 clip fraction | 约束过弱、学习率过高 |
| DPO margin 升但输出变差 | 长度/format 统计 | 长度偏置、rejected 过拟合 |
| 策略输出退化为套话 | entropy 与 KL 轨迹 | KL 过小、模式坍缩 |
| chosen/rejected 差恒为 0 | 共享 prompt 检查 | 同一 response、tokenize 错误 |
| judge 与人工评分背离 | judge 偏差测试 | 长度/自信/同族偏好 |
| 拒答率异常升高 | safety 分轴评估 | 无害偏好过度泛化 |

## 11. 讨论：效度威胁与结论边界

### Construct validity

- preference ≠ 真实意图：标注者的比较受长度、语气、格式捷径影响；
- RM score 与人工 win rate 是不同构造，前者是代理；
- “对齐”没有单一标量：helpfulness、harmlessness、诚实性可能彼此冲突。

### Internal validity

- SFT/RM/PPO 阶段同时变化时无法归因；
- judge 模型与被评模型同族会造成系统性偏好；
- 单 seed 的 win rate 置信区间常宽于报告的差异。

### External validity

- 特定 rubric 与人群下收集的偏好不外推到其他文化或领域；
- 小模型上的 \(\beta\)/学习率结论不迁移到大模型；
- benchmark 的 instruction 分布不代表真实用户分布。

论文式表述应报告标注协议、judge 版本与偏差测试、评估的置信区间与失败样本，而不是
只给“我们的方法 win rate 更高”。

## 面试要点速记

**高频问题与答题要点**

1. **Q：SFT 的 loss mask 怎么做？** 要点：只对 response token 计损
   （response-only）；packed 场景须处理跨样本 attention 与边界 token。
2. **Q：BT 模型与 RM loss？** 要点：\(P(y_w\succ y_l)=\sigma(r_w-r_l)\)，
   loss = \(-\log\sigma(r_w-r_l)\)；pairwise accuracy 高不等于 reward
   calibrated。
3. **Q：PPO 中 KL 与 clip 的分工？** 要点：KL 约束限制对 reference 的整体
   分布 drift；PPO clip 限制单步 token 级更新幅度；二者不可互相替代。
4. **Q：DPO 一句话与常见坑？** 要点：KL-regularized 最优策略闭式代回，把 RL
   化为 chosen/rejected log-odds 差的 logistic loss；坑：长度偏置、模板
   泄漏、reference 冻结与 tokenizer 口径不一致。
5. **Q：reward hacking 怎么检测？** 要点：RM 分与人工 win rate 的斜率分离、
   长度/format 指标同步飙升、held-out RM 与训练 RM 分歧。
6. **Q：蒸馏 vs 直接 RL，小模型选哪条路？** 要点：DeepSeek-R1 口径——
   Qwen-32B 蒸馏版（800k 精选样本 SFT）性能超越同规模直接 RL 版；教师足够
   强时蒸馏性价比高；直接 RL 的上限依赖基座自身探索能力。
7. **Q：DPO 为什么能免 RM、免在线采样？** 要点：KL-regularized 偏好优化下
   reward 可由最优策略闭式表示，隐式 reward = 两策略 logratio，偏好对直接
   变成分类损失；但偏好数据偏差与 offline coverage 限制原样保留。
8. **Q：工业界 SFT 数据从哪来？** 要点：真实用户 prompts（LMSYS/HelpSteer/
   WildChat 类）+ 强教师生成响应（Qwen3-235B、DeepSeek-R1-0528）；安全与
   工具调用垂类走专门合成管线 + 规则化校验过滤。
9. **Q：reward hacking 的对策清单？** 要点：KL 参考约束、独立 held-out 评测、
   reward-人工斜率监控、早停与 RM 迭代重训；训练框架需同时记录 reward/KL/
   长度/entropy 曲线（如 verl 的多 worker 指标归集）。

**必背数字**

- InstructGPT 管线：pretrain → SFT → RM → PPO；KL 系数 \(\beta\) 过小
  → hacking、过大 → 不学习；LIMA：~1k 高质量样本即可激活对齐能力。
- 工业界参照：Nemotron Nano 2 posttrain 共 90B tokens，其中 SFT 80B；
  顺序 SFT → GRPO（强化指令遵循与对话）→ DPO（增强工具使用）→ RLHF
  （再强化指令遵循）；推理 SFT 响应来自 DeepSeek-R1-0528。
- 工业界参照：DeepSeek-R1 蒸馏 = 800k 精选样本（600k 推理 + 200k 通用）
  SFT；Qwen-32B 蒸馏版性能超越同规模直接 RL 训练版。
- 工业界参照：R1-Zero 纯 RL：AIME 15.6% → 86.7%（无冷启动 SFT、规则奖励）。
- 工业界参照：工具调用三角色合成——Qwen3-235B-A22B 分饰 User-Agent /
  Assistant-Agent / API-Server-Agent，规则化 tool-call 校验层只留成功轨迹。
- 工业界参照：安全数据两步法 = 提示生成 + guard 模型过滤（Nemotron Content
  Safety V2 / RedTeam2K / gretel-v1，响应由 DeepSeek-R1-0528 生成）。

## 行业现状与最新进展（2024–2026）

### SFT 数据的工业配方：Nemotron Nano 2 口径

- Posttrain 共 90B tokens，其中 SFT 占 80B——SFT 仍是后训练 token 预算的
  绝对大头，远未被 RL 取代。
- 阶段顺序是多算法串联而非单选：SFT → GRPO（强化指令遵循与对话）→ DPO
  （增强工具使用）→ RLHF（再强化指令遵循）。
- 会话数据：LMSYS / HelpSteer2 / HelpSteer3 / WildChat-1M（55 万子集）的
  真实 prompts，响应由 Qwen3-235B 生成；推理 SFT 响应来自 DeepSeek-R1-0528。
- 安全数据：Nemotron Content Safety V2 / HarmfulTasks / RedTeam2K /
  gretel-v1；两步法 = 提示生成 + guard 模型过滤，把红队流程自动化。
- 工具调用：Qwen3-235B-A22B 分饰三角色——User-Agent（审视工具、提出查询、
  判定任务成败）/ Assistant-Agent（调用工具、解读返回）/ API-Server-Agent
  （校验参数、返回成功/错误）；外接规则化 tool-call 校验层，仅保留成功轨迹。

### 蒸馏 vs 直接 RL：DeepSeek-R1 的对照

- R1 四阶段：①冷启动 SFT（少量长 CoT）→ ②推理 RL（规则奖励）→ ③拒绝
  采样 + 全场景 SFT（600k 推理 + 200k 通用）→ ④RLVR（规则奖励 + 多任务）。
- R1-Zero 纯 RL：无冷启动、纯规则奖励，AIME 从 15.6% 提升到 86.7%。
- 关键实证：Qwen-32B 蒸馏版性能超越同规模直接 RL 训练版——教师足够强时
  蒸馏性价比高；蒸馏配方即 800k 精选样本直接 SFT，无需复杂 RL 栈。

| 维度 | 蒸馏 SFT（R1 口径） | 直接 RL（R1-Zero 口径） |
| --- | --- | --- |
| 数据 | 800k 教师生成精选样本（600k 推理 + 200k 通用） | 无标注，规则奖励驱动自采样 |
| 算法 | 监督 SFT | 纯 RL + 规则奖励（无冷启动） |
| 成本 | 教师推理 + 数据筛选，训练简单 | 大量 rollout + 训练稳定性成本 |
| 上限 | 受教师能力封顶 | 基座足够强时可自我超越（AIME 15.6%→86.7%） |
| 适用场景 | 小模型快速获得推理能力 | 大基座突破现有分布 |

### RLHF 栈演进：PPO → DPO 与开源基建

- DPO（2023）直接用偏好对优化策略：隐式 reward = 两策略 logratio，免显式
  RM 与在线采样，成为开源社区主流首选之一。
- PPO 系（含 GRPO 等规则奖励变体）仍是在线探索与推理突破的主力；KL 参考
  约束与 reward hacking 监控（reward-人工斜率、长度/format 飙升、held-out
  RM 分歧）是生产标配。
- 开源基建：verl（Volcano Engine RL 训练框架）采用 hybrid controller——
  单控制器 + 多 worker 架构，支持多种 RL 算法与 vLLM/SGLang rollout，
  降低自建 RLHF 栈的门槛。

**对本讲学习者的启示：** InstructGPT 的三步范式（SFT → RM → PPO，2022）
仍是理解一切变体的坐标系，但 2024–2026 的工业实践已变成"多算法串联 + 数据
合成 + 规则化过滤"的组合拳：SFT 靠强教师蒸馏建立接口（80B tokens 量级），
DPO 与 RLHF 各取所长分阶段强化，安全与工具能力靠专门的数据合成管线而非
通用偏好数据。学习本讲时应把公式（loss mask、Bradley–Terry、KL 约束、DPO
闭式）当作审阅这些工业配方的"质检工具"——每看到一个新配方都追问：数据从
哪来、mask 对不对、reward 会不会被 hack、结论是否限定到评估协议。

## 大厂面试真题与答题框架

以下为高频面试题（公开面经风格），非任何公司原题。

**题目 1：SFT 的 loss mask 为什么要 mask 掉 prompt？不 mask 会怎样？**
- 考点：response-only likelihood、数据契约、模板泄漏。
- 答题框架：① SFT 的目标是学"给定 prompt 生成 response"的条件分布，
  prompt 是条件不是目标；② 不 mask 等于同时学 prompt 分布，把模型容量花在
  模仿用户输入上，且不同数据源的 prompt 分布差异会干扰响应风格；③ 工程上
  prompt 与 response 分开 tokenize 再拼接，shift 后 mask 从
  `prompt_length - 1` 起覆盖；④ packed 场景还要处理跨样本 attention 与 EOS。
- 加分项：指出本仓库 `PackedSFTDataset` 是 full-sequence loss 的真实反例；
  引 LIMA——质量与一致性比数量重要。
- 踩坑：说"不 mask 也一样"；mask 起点 off-by-one 是最高频实现 bug。

**题目 2：PPO 与 DPO 怎么取舍？**
- 考点：在线 vs 离线偏好优化、RM、KL 约束、系统复杂度。
- 答题框架：① 数据：PPO 需 RM + online rollout，DPO 只需 offline 偏好对；
  ② DPO 把 KL-regularized 最优策略闭式代回，隐式 reward = 两策略 logratio，
  免 RM 免采样；③ PPO 可探索当前策略的新输出，DPO 受 offline coverage
  限制；④ 成本：PPO 需 critic/value 与 rollout 引擎，DPO 只需双模型前向；
  ⑤ 实践不互斥——Nemotron Nano 2 顺序跑 GRPO → DPO → RLHF。
- 加分项：指出 DPO 没有免去偏好假设，继承偏好数据的一切偏差；引 R1 说明
  规则奖励在线 RL 仍是推理突破主力。
- 踩坑：把 DPO 说成"没有 KL"——有，只是闭式进了损失；把二者当单选题。

**题目 3：reward hacking 是什么？如何检测与缓解？**
- 考点：proxy reward 与真实意图的分离、诊断指标、KL 约束。
- 答题框架：① 定义：policy 利用 RM 的系统性漏洞（变长、堆砌好词、自信
  语气、模仿 RM 训练分布风格）而非真正提升质量；② 检测：RM 分与人工
  win rate 的斜率分离、长度/format 与 reward 同步飙升、held-out RM 与
  训练 RM 分歧、KL 逼近上限而 capability 下降；③ 缓解：KL 参考约束、
  独立评测、早停、RM 迭代重训。
- 加分项：引 InstructGPT——PPO 后模型 RM 分远超人类示范，但人工评估并未
  同比例提升。
- 踩坑：只拿 reward 曲线当"对齐提升"的证据。

**题目 4：为什么蒸馏小模型可能胜过直接 RL？**
- 考点：蒸馏 vs RL 的上限与成本、教师质量前提。
- 答题框架：① 实证：DeepSeek-R1 口径下 Qwen-32B 蒸馏版（600k 推理 +
  200k 通用样本 SFT）超越同规模直接 RL 训练版；② 原因：小基座自身探索
  能力弱，RL 难以自发涌现长 CoT，蒸馏直接继承教师推理轨迹；③ 前提：教师
  显著强于学生；④ 大基座相反——R1-Zero 纯 RL 从 AIME 15.6% 到 86.7%。
- 加分项：给出决策规则——基座小/预算紧 → 蒸馏；基座强 + 有规则奖励 → RL。
- 踩坑：把结论过度外推成"RL 没用"。

**题目 5：写出 BT 模型下的 RM loss；pairwise accuracy 高就够了吗？**
- 考点：Bradley–Terry、calibration、分布外可靠性。
- 答题框架：① \(P(y_w\succ y_l)=\sigma(r_w-r_l)\)，loss =
  \(-\log\sigma(r_w-r_l)\)；② accuracy 只看排序不看尺度，margin 才反映
  置信度；③ 需按任务、安全类别、长度分组评估；④ train/test 必须
  prompt-level split 防泄漏。
- 加分项：提 overoptimization——RM 是 proxy，PPO 优化它必然偏离真实意图。
- 踩坑：把 RM accuracy 当对齐质量本身。

**题目 6：工业界 SFT 数据从哪来？纯人工标注吗？**
- 考点：数据合成管线、教师蒸馏、质量控制。
- 答题框架：① 真实用户 prompts（LMSYS/HelpSteer/WildChat 类公开集或自家
  日志脱敏）；② 强教师生成响应（Qwen3-235B、DeepSeek-R1-0528 等）；③
  垂类专门管线：安全数据两步法（提示生成 + guard 模型过滤）、工具调用
  三角色模拟 + 规则化校验；④ 质量闸门：只留成功轨迹、去模板泄漏与冲突示范。
- 加分项：给出量级数字——Nemotron Nano 2 posttrain 90B tokens 中 SFT 80B；
  会话数据来自 WildChat-1M 的 55 万子集。
- 踩坑：以为"合成 = 低质量"——决定质量的是校验层而非数据来源。

**题目 7：DPO 实现里有哪些必查的坑？**
- 考点：reference 冻结、log-prob 口径、数据契约。
- 答题框架：① chosen/rejected 必须共享同一 prompt 与 template；② 只累计
  response token 的 log-prob；③ reference 前向必须 no_grad 且 tokenizer/
  template 与 policy 完全一致；④ 监控长度偏置与 rejected 过拟合；⑤ 报
  held-out preference accuracy 而非只看训练 margin。
- 加分项：指出直接求和造成长度偏好、但未经审计改成平均同样改变目标。
- 踩坑：reference 未冻结（被优化器更新）时 loss 会"异常好"——这是 bug。

## 系统设计题

**设计题 1：为公司助手模型设计 SFT → 偏好优化 → 安全对齐的全流程数据与训练管线**

- 需求澄清：模型规模与算力预算？内网部署还是 API？垂类（代码/客服/办公）？
  安全等级与拒答边界？评测基线与上线标准是什么？
- 规模估算：参照 Nemotron Nano 2——posttrain 90B tokens（SFT 80B）量级；
  自家场景可按 1–10B SFT tokens 起步；偏好对 10 万–100 万条量级；推理蒸馏
  参照 R1 配方 800k 精选样本（600k 推理 + 200k 通用）。
- 架构：① 数据层——用户真实 prompts（脱敏）+ 公开集（LMSYS/HelpSteer/
  WildChat 类）+ 垂类合成，教师模型生成响应，安全数据走两步法（提示生成 +
  guard 模型过滤）；② SFT 层——response-only mask + packing，capability/
  safety/工具调用分桶按配比混合；③ 偏好层——先 DPO（离线、便宜）建立偏好
  接口，再视预算上 GRPO/PPO（KL 约束 + reward hacking 监控面板）；④ 评测层
  ——capability 与 safety 分轴、多 seed、人工盲测 + 独立 judge 偏差测试。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 取舍 |
| --- | --- | --- | --- |
| 偏好算法 | DPO（离线偏好对） | PPO/GRPO（在线 + RM/规则奖励） | 成本与稳定性 vs 探索上限 |
| 响应来源 | 强教师蒸馏 | 人工标注 | 规模与成本 vs 质量上限 |
| 安全数据 | 两步合成（生成 + guard 过滤） | 纯红队人工 | 覆盖面与规模 vs 真实性 |
| 训练顺序 | SFT→DPO→RLHF 分阶段 | 单阶段端到端 | 稳定可归因 vs 迭代速度 |

- 评测方案：held-out prompt-level split；reward 与人工 win rate 斜率监控；
  长度/format 旁路指标；安全分轴（拒答率、过拒率）单独报告，不合并单分数。
- 追问预案：出现 reward hacking 怎么办（加大 KL、早停、RM 迭代重训）；
  helpful/harmless 冲突怎么管（rubric 显式加权、分轴报告）。

**设计题 2：设计工具调用能力的 SFT 数据合成平台（三角色模拟）**

- 需求澄清：工具数量与 schema 复杂度？单轮还是多轮？要不要覆盖失败恢复
  （工具报错后的重试轨迹）？目标模型规模与后训练预算？
- 规模估算：参照 Nemotron Nano 2 三角色方案——Qwen3-235B-A22B 分饰
  User-Agent（审视工具、提出查询、判定任务成败）/ Assistant-Agent（调用
  工具、解读返回）/ API-Server-Agent（校验参数、返回成功/错误）；成功轨迹
  按 10 万–100 万条量级规划，随工具 schema 数量线性扩展。
- 架构：① 角色引擎——同一强模型多角色 prompting，角色间状态机管理轮次；
  ② API 模拟层——mock server 按 schema 校验参数并返回成功/错误；③ 校验层
  ——规则化 tool-call 校验（JSON schema 合法、参数类型正确、调用序正确、
  User-Agent 判定任务成功），仅保留成功轨迹；④ 混合层——与通用对话数据
  按比例混合，防止工具调用格式过拟合。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 取舍 |
| --- | --- | --- | --- |
| API 层 | 真实调用 | mock server | 真实性 vs 可复现与成本 |
| 轨迹筛选 | 只留成功轨迹 | 保留失败 + 恢复轨迹 | 数据干净 vs 学会纠错 |
| 生成器 | 单模型分饰三角色 | 多异构模型分工 | 风格一致性 vs 多样性 |

- 评测方案：held-out 工具集（训练未见 schema）；成功率、参数准确率、多轮
  任务完成率分层报告；与真实用户日志分布做覆盖度对比。
- 追问预案：合成轨迹分布偏窄（注入 User-Agent 主动性、随机化工具组合）；
  模型模仿 mock 错误格式（错误信息也过 schema 校验）。

**设计题 3：设计 reward hacking 的在线监控与熔断系统**

- 需求澄清：RL 算法（PPO/GRPO）？训练集群规模？可接受的误熔断率？有没有
  独立评测预算（人工盲测/独立 RM）？
- 规模估算：指标按 step 粒度采集；人工抽评按每 N step 抽百条量级；held-out
  RM 维护一个独立 checkpoint，训练 RM 更新时同步评估分歧。
- 架构：① 指标采集——reward、KL、entropy、clip fraction、长度、format
  命中率；② 独立评测通道——held-out RM + 周期性人工盲测，计算 reward-人工
  斜率；③ 熔断规则——斜率跌破阈值 / KL 逼近上限 / 长度飙升超带 → 自动降
  学习率或停训；④ 可视化——单面板收口（参照 verl hybrid controller：
  单控制器 + 多 worker 的指标归集思路）。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 取舍 |
| --- | --- | --- | --- |
| 熔断动作 | 自动停训/降学习率 | 只告警人工决策 | 安全性 vs 打断训练 |
| 独立评测 | 人工盲测 | held-out RM | 可信度 vs 延迟与成本 |
| KL 约束 | 固定 \(\beta\) | 分阶段退火 | 稳定 vs 学习效率 |

- 评测方案：回放历史 hacking 案例验证熔断召回；对正常训练做误报率压测。
- 追问预案：KL 调大后学习停滞（\(\beta\) 分阶段退火）；无人工预算（多
  judge 交叉 + 偏差测试，明示结论局限）。

## 代码实现题

**代码题 1：带 loss mask + EOS 的 SFT collator**

- 题目：实现一个 collator，输入 batch 的 prompt/response 文本对，输出
  `input_ids`、`labels`（prompt 与 padding 置 -100，response 计损，末尾补
  EOS）、`attention_mask`。
- 考察点：prompt/response 分开 tokenize 再拼接；mask 起点 off-by-one；
  padding side 一致性；EOS 注入。

```python
import torch

IGNORE_INDEX = -100

def sft_collate(batch, tokenizer, max_len=2048):
    # batch: list of {"prompt": str, "response": str}
    input_ids_list, labels_list = [], []
    for ex in batch:
        # 1) prompt 与 response 分开 tokenize，避免 token 边界混淆
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        resp_ids = tokenizer(ex["response"], add_special_tokens=False)["input_ids"]
        resp_ids = resp_ids + [tokenizer.eos_token_id]  # 2) 补 EOS
        seq = prompt_ids + resp_ids
        # 3) mask 掉 prompt：只有 response 区间计损
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(resp_ids)
        seq, labels = seq[:max_len], labels[:max_len]
        input_ids_list.append(seq)
        labels_list.append(labels)

    maxlen = max(len(s) for s in input_ids_list)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), maxlen), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, (s, l) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, : len(s)] = torch.tensor(s)  # right padding
        labels[i, : len(l)] = torch.tensor(l)
        attn[i, : len(s)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
```

- 验收标准：① labels 中 prompt 区间与 pad 区间均为 -100；② 每条样本
  response 最后一个非 ignore token 是 EOS；③ 交叉熵只对非 ignore 位置求
  均值；④ shift 后（labels[:, 1:] 对 inputs[:, :-1]）mask 对齐正确；⑤
  单测覆盖空 response、超长截断、batch 内长度不齐三种边界。

**代码题 2：DPO 损失实现（含隐式 reward 与准确率）**

- 题目：给定 policy 与 frozen reference 的 chosen/rejected 序列级 log-prob，
  实现 DPO loss、隐式 reward 与 preference accuracy。
- 考察点：logratio 口径；reference no_grad；数值稳定；指标与 loss 分离。

```python
import torch
import torch.nn.functional as F

def dpo_loss(pol_chosen_lp, pol_rejected_lp,
             ref_chosen_lp, ref_rejected_lp, beta=0.1):
    """
    *_lp: (B,) 序列级 log-prob，只对 response token 求和。
    reference 分支必须由 no_grad 前向得到。
    """
    # 隐式 reward = policy 与 reference 的 logratio
    chosen_rw = pol_chosen_lp - ref_chosen_lp
    rejected_rw = pol_rejected_lp - ref_rejected_lp
    logits = beta * (chosen_rw - rejected_rw)
    loss = -F.logsigmoid(logits).mean()      # DPO 主损失
    acc = (logits > 0).float().mean()        # preference accuracy
    return loss, acc, chosen_rw.detach(), rejected_rw.detach()
```

- 验收标准：① reference 张量 requires_grad=False（或已 detach），否则
  断言报错；② chosen/rejected 完全相同时 loss = log 2、acc = 0.5（数据 bug
  探针）；③ 梯度只流向 policy 分支，reference 分支梯度为 None；④ 输出
  的隐式 reward 可直接用于监控长度偏置（对 reward 与长度做相关分析）。

**代码题 3：SFT 数据混合与打包（packing）脚本**

- 题目：给定多个数据源（通用对话/推理/安全/工具调用）及目标配比，输出
  packed 的连续 token stream，保证 EOS 分隔并记录每个样本的源归属。
- 考察点：分层采样保持配比；EOS 边界；可复现随机性；per-source 统计。

```python
import random

def pack_datasets(sources, tokenizer, seq_len=4096, seed=0):
    # sources: {"通用": [texts], "推理": [texts], "安全": [texts], "工具": [texts]}
    rng = random.Random(seed)
    pool = []
    for name, texts in sources.items():  # 1) 分源展开，保持配比
        for t in texts:
            ids = tokenizer(t, add_special_tokens=False)["input_ids"]
            # 2) 每个样本末尾补 EOS 作为分隔
            pool.append((name, ids + [tokenizer.eos_token_id]))
    rng.shuffle(pool)

    packs, cur, cur_srcs = [], [], []
    for name, ids in pool:
        if cur and len(cur) + len(ids) > seq_len:  # 3) 装不下则封包
            packs.append({"input_ids": cur, "sources": cur_srcs})
            cur, cur_srcs = [], []
        cur.extend(ids)
        cur_srcs.append(name)
    if cur:
        packs.append({"input_ids": cur, "sources": cur_srcs})
    return packs
```

- 验收标准：① 每包长度 ≤ seq_len；② 相邻样本间必有 EOS；③ 各源占比与
  输入配比偏差 < 2%（抽样校验）；④ 同 seed 输出逐字节一致；⑤ 输出
  per-source token 统计供后续 loss 加权；⑥ 明确跨样本 attention 的取舍——
  本实现允许共享，安全敏感场景需 attention mask 隔离（本仓库
  `PackedSFTDataset` 同样面临该边界问题）。

## 12. 小结

SFT 建立模型的任务接口，RM 把成对偏好压缩成代理奖励，PPO 在线优化该奖励，DPO 则直接
做 reference-relative 偏好分类。它们的核心区别是数据来自哪里、是否在线探索、如何限制
policy drift。任何“对齐提升”都必须限定到 rubric、分布和评估协议，不能由一个 reward
数字替代。

## 参考文献

<a id="ref-1"></a>[1] P. Christiano, J. Leike, T. Brown, et al. “Deep
Reinforcement Learning from Human Preferences.” *NeurIPS*, 2017.
[link](https://arxiv.org/abs/1706.03741)

<a id="ref-2"></a>[2] L. Ouyang, J. Wu, X. Jiang, et al. “Training Language
Models to Follow Instructions with Human Feedback.” *NeurIPS*, 2022.
[link](https://arxiv.org/abs/2203.02155)

<a id="ref-3"></a>[3] Y. Bai, A. Jones, K. Ndousse, et al. “Training a
Helpful and Harmless Assistant with Reinforcement Learning from Human
Feedback.” arXiv:2204.05862, 2022. [link](https://arxiv.org/abs/2204.05862)

<a id="ref-4"></a>[4] J. Schulman, F. Wolski, P. Dhariwal, et al.
“Proximal Policy Optimization Algorithms.” arXiv:1707.06347, 2017.
[link](https://arxiv.org/abs/1707.06347)

<a id="ref-5"></a>[5] R. Rafailov, A. Sharma, E. Mitchell, et al. “Direct
Preference Optimization: Your Language Model is Secretly a Reward Model.”
*NeurIPS*, 2023. [link](https://arxiv.org/abs/2305.18290)

<a id="ref-6"></a>[6] C. Zhou, P. Liu, P. Xu, et al. “LIMA: Less Is More
for Alignment.” *NeurIPS*, 2023. [link](https://arxiv.org/abs/2305.11206)

<a id="ref-7"></a>[7] E. J. Hu, Y. Shen, P. Wallis, et al. “LoRA: Low-Rank
Adaptation of Large Language Models.” *ICLR*, 2022.
[link](https://arxiv.org/abs/2106.09685)

## 延伸阅读与复现材料

- Stanford CS336, [Lecture 15 — Mid/post-training](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf)
- [Alignment 主题导航](../experiments/topics/alignment.md)
- [A5 Supplement — Safety & RLHF](../assignments/spring2026/assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)（访问日期 2026-10-04）
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)（访问日期 2026-10-04）
- [verl — Volcano Engine RL 训练框架](https://github.com/volcengine/verl)（访问日期 2026-10-04）
- [Stanford CS336 课程主页](https://cs336.stanford.edu)（访问日期 2026-10-04）
