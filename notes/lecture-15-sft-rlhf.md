---
title: "Lecture 15 — SFT & RLHF"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-05-18"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf"
  - "../assignments/assignment5-alignment/"
---

# Lecture 15 — SFT 与 RLHF：从模仿到偏好优化

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述，非课程提交
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

### 11.1 Construct validity

- preference ≠ 真实意图：标注者的比较受长度、语气、格式捷径影响；
- RM score 与人工 win rate 是不同构造，前者是代理；
- “对齐”没有单一标量：helpfulness、harmlessness、诚实性可能彼此冲突。

### 11.2 Internal validity

- SFT/RM/PPO 阶段同时变化时无法归因；
- judge 模型与被评模型同族会造成系统性偏好；
- 单 seed 的 win rate 置信区间常宽于报告的差异。

### 11.3 External validity

- 特定 rubric 与人群下收集的偏好不外推到其他文化或领域；
- 小模型上的 \(\beta\)/学习率结论不迁移到大模型；
- benchmark 的 instruction 分布不代表真实用户分布。

论文式表述应报告标注协议、judge 版本与偏差测试、评估的置信区间与失败样本，而不是
只给“我们的方法 win rate 更高”。

## 12. 面试备考（Interview Prep）

> SFT/RLHF 是 LLM 面试的「对齐」高频题：面试官常从「SFT 的 loss mask」「DPO 和 RLHF 区别」
> 切入，追到「PPO 里 KL 与 clip 的分工」「reward hacking 怎么检测」「DPO 为什么不用 reward model」。
> 核心是沿「模仿 → 偏好 → 策略优化」理解每种方法的数据、目标与失败模式，并牢记
> 「对 reward 优化 ≠ 对真实意图对齐」。下面按「一页速览 → 高频题 → 手撕 → 追问」四层组织。

### 12.1 一页速览卡（面试前 1 分钟）

**核心主张**：后训练把「预测 token 的基座」变成「遵循指令、被偏好的助手」，主线是
模仿(SFT) → 偏好(RM) → 策略优化(PPO/DPO)；任何「对齐提升」都要限定到 rubric、分布与评估协议。

**必背数字与公式**

- SFT：\(\mathcal L_{\text{SFT}}=-\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t})\)（\(m_t\) 只盖 response token）。
- RM：\(-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\)。
- PPO ratio \(\rho_t=\pi_\theta/\pi_{\text{old}}\)，clip 到 \(1\pm\epsilon\)。
- RLHF 目标 \(\mathbb E[r_\phi-\beta\log(\pi_\theta/\pi_{\text{ref}})]\)。
- DPO：对 chosen/rejected 的 reference-relative log-odds 差做 logistic loss。
- LoRA \(W=W_0+BA\)；LIMA 约 **1k** 高质量样本即可激活对齐。

**三句话答高频**

1. SFT 建立任务接口，RM 压缩偏好成代理奖励，PPO 在线优化，DPO 直接做偏好分类。
2. KL 约束控制对 reference 的整体 drift，PPO clip 控制单步 token 级更新，两者不等价。
3. reward hacking 检测：RM 分与人工 win rate 斜率分离、长度/format 同步飙升、held-out RM 分歧。

### 12.2 高频面试题与答题框架

**Q1：SFT 的 loss mask 怎么做？**

- 只对 response token 计损（response-only）：\(m_t=1\) 仅覆盖 response 位置，prompt 不计。
- prompt 与 response 分开 tokenize 再拼接；shift 后 mask 的第一个 response 位置是 `prompt_length - 1`。
- packing 提高利用率但须明确 EOS、跨文档 attention 与 prompt-token loss；数据质量 > 数量（LIMA）。

**Q2：Bradley–Terry 模型与 RM loss？**

- 假设 \(P(y_w\succ y_l)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\)，loss \(=-\log\sigma(r_w-r_l)\)。
- pairwise accuracy 高 ≠ reward calibrated，也不保证 OOD 可靠；需按任务/安全类别/长度评估并查 reward margin。

**Q3：PPO 里 KL 约束与 clip 的分工？**

- **KL 约束**（\(\beta\log(\pi_\theta/\pi_{\text{ref}})\)）限制对 reference 的整体分布 drift，防 reward hacking。
- **PPO clip**（\(\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)\)）限制单次更新的 token 级幅度，是优化稳定性手段。
- 两者角色相关但不等价：\(\beta\) 过小 hacking、过大不学习；clip 控制单步、KL 控制整体。

**Q4：DPO 为什么不需要 reward model？常见坑？**

- 在 KL-regularized 偏好优化下，最优策略可闭式表示 reward，代回后把 RL 化为 chosen/rejected 的 log-odds 差 logistic loss，免去显式 RM 与 online rollout。
- **常见坑**：长度偏置（log-prob 求和）、模板泄漏（chosen/rejected 用不同模板）、reference 冻结 + tokenizer 口径不一致。
- **注意**：DPO 免去「RL」，但没免去偏好假设、coverage 与数据偏差。

**Q5：reward hacking 怎么检测？**

- RM 分数远超人类示范、但人工 win rate 未同步提升 → reward-human 斜率下降（overoptimization）。
- 长度/format 指标与 reward 同步飙升；held-out RM 与训练 RM 分数分歧；KL 快速逼近上限而 capability 下降。

**Q6：RLHF 的完整管线？**

- `pretrain → SFT → preference/RM → PPO`（InstructGPT）。
- 数据契约：preference pair 共享 prompt、randomize 顺序、记录 rubric 与 disagreement；RM 有独立 prompt-level split。
- 报告：reward、KL、entropy、clip fraction、value loss、长度与人工 win rate，多 seed。

**Q7：SFT / RM / PPO / DPO 怎么选？**

- SFT：高质量 demonstrations，简单稳定但只模仿覆盖到的行为；RM：pairwise preferences，得可复用 score 但 hacking/失准；PPO：RM+online rollout，可探索新输出但系统复杂、方差高；DPO：offline pairs，简洁无 critic 但受 offline coverage/reference/长度偏差。
- 先建 SFT + 固定评估，再按「是否有可靠在线 reward、生成预算、探索需求」选 PPO 或 DPO。

**Q8：LoRA 是什么？**

- 冻结基座、只训练低秩增量 \(W=W_0+BA\)，\(B\in\mathbb R^{d\times r},A\in\mathbb R^{r\times d'},r\ll d\)。
- 大幅降低 optimizer state 与显存，接近全参微调质量，便于多任务/多适配器管理。

**Q9：LIMA 的发现？**

- 约 1000 条高质量、多样、风格一致的示范即可让基座产生显著指令遵循能力。
- 对齐能力大部分已存在于预训练分布，SFT 更像「激活接口」而非「注入知识」——数据质量 > 数量。

**Q10：为什么「对 reward 优化」≠「对人的真实意图对齐」？**

- RM 是代理：偏好受长度、语气、格式捷径影响；RM 分数与人工 win rate 是不同构造。
- helpfulness/harmlessness/诚实性可能冲突，不能压成单分数；任何结论都要限定 rubric、分布与评估协议。

### 12.3 手撕要点（SFT mask 与 DPO loss）

面试让「写 SFT loss mask」或「推导 DPO」时，按公式写：

```text
SFT（response-only）: L = -Σ_t m_t log π_θ(y_t | x, y_{<t})
  m_t = 1 仅 response token；shift 后第一个 response 位置 = prompt_len - 1

RM: L = -log σ(r_w - r_l)

DPO: L = -log σ( β [ log(π_θ(y_w|x)/π_ref(y_w|x))
                       - log(π_θ(y_l|x)/π_ref(y_l|x)) ] )
  只累计 response token log-prob；reference 前向 no_grad
```

**三个必踩坑**

1. **SFT 别对 prompt token 计损**：否则声称 response-only 实际是 full-sequence。
2. **chosen/rejected 必须共享 prompt**：用不同模板会学到模板差异而非偏好。
3. **DPO 的 reference 必须冻结 + tokenizer 一致**：log-prob 口径不一致会污染 margin。

### 12.4 高频追问与陷阱

| 追问 | 正确方向 |
| --- | --- |
| DPO 是「无需 RL」吗？ | 是，但仍有偏好假设、coverage 与数据偏差 |
| KL 过大/过小会怎样？ | 过小 reward hacking、过大几乎不学习 |
| RM 分高就对齐了吗？ | 否，RM 是代理，可能被 hacking |
| 把 helpful/harmless 压成单分数可以吗？ | 否，会掩盖对某群体的性能退化 |
| 自动 judge 可信吗？ | 有长度/自信/同族偏好，需偏差测试 + 人工盲测 |

## 13. 小结

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
- [A5 Supplement — Safety & RLHF](../assignments/assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)
