# Alignment：从 Supervised Finetuning 到 Reasoning RL

本主题把 alignment 理解为改变模型行为分布的一组训练与评估方法，而不是单一 objective。建议先建立 SFT 和 evaluation baseline，再读 preference optimization，最后进入可验证奖励的 RLVR/GRPO；reward 上升必须与能力、稳定性和安全评估一起解释。

## 推荐阅读顺序

### 1. 从 SFT 与评估契约开始

- **官方 lecture**：[Lecture 15 — Mid/post-training (SFT/RLHF)](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf)。
- **论文**：[InstructGPT](https://arxiv.org/abs/2203.02155) 给出 demonstration SFT、preference reward model 与 PPO 的经典三阶段流程。
- **工具文档**：[Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) 与 [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。
- **解决的问题**：pretrained next-token predictor 并未学会遵循 instruction、输出格式或任务边界。SFT 用高质量 prompt-response demonstrations 直接拟合期望行为。
- **关键公式/结论**：
  \[
  \mathcal L_{\mathrm{SFT}}
  =-\sum_{t\in\text{response}}\log\pi_\theta(y_t\mid x,y_{<t}).
  \]
  通常只对 response tokens 计 loss；packing 可减少 padding，但 attention mask、document boundary 与 position handling 必须正确。训练前应固定 prompt template、generation parameters、answer parser 和 metric，否则评估漂移会被误认为模型提升。
- **对应作业**：[Assignment 5 — Alignment](https://github.com/stanford-cs336/assignment5-alignment) 的 SFT packing、GSM8K/MMLU formatting、answer extraction 与 metrics；本仓库快照位于 `assignments/spring2026/assignment5-alignment/`。
- **局限**：SFT 只能模仿数据覆盖的行为；少量高质量数据与大量噪声数据的权衡依赖任务。exact-match parser 可能漏判等价答案，也可能被格式投机。

### 2. 理解 RLHF 中 reward 与 policy update

- **论文**：[Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) 与 [InstructGPT](https://arxiv.org/abs/2203.02155)。
- **工具文档**：[TRL PPOTrainer](https://huggingface.co/docs/trl/ppo_trainer)。
- **解决的问题**：demonstrations 昂贵且不能表达所有偏好；pairwise preferences 可训练 reward model，再优化 policy。
- **关键公式/结论**：PPO 常用 clipped surrogate
  \[
  \mathbb E_t\!\left[
  \min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)
  \right],
  \quad
  r_t=\frac{\pi_\theta(a_t\mid s_t)}
  {\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
  \]
  对 reference policy 的 KL penalty 用于抑制 policy 过快偏移。reward model accuracy 不等于 reward calibration，也不保证其分布外可用。
- **对应作业**：A5 supplement 中 safety alignment/RLHF 概念；主作业的 ratio、clipping 和 off-policy 分析沿用这一基础。
- **局限**：reward hacking、preference annotator bias、KL coefficient 敏感性与训练不稳定都可能让标量 reward 与真实目标分离。

### 3. 再读 DPO：把偏好学习写成分类目标

- **论文**：[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)。
- **工具文档**：[TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)。
- **解决的问题**：传统 RLHF 需要单独 reward model 与 online RL。DPO 在特定 Bradley–Terry/regularized policy assumptions 下，直接从 chosen/rejected response 优化 policy。
- **关键公式/结论**：
  \[
  \mathcal L_{\mathrm{DPO}}
  =-\log\sigma\!\left(\beta\left[
  \log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
  -
  \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
  \right]\right).
  \]
  \(\beta\) 控制偏离 reference 的强度。chosen/rejected 必须共享 prompt，sequence log-prob 的 mask、长度处理与 reference cache 要一致。
- **对应作业**：A5 supplement 的 DPO loss 与 preference metrics。
- **局限**：DPO 不是“无需 reward assumptions”；它依赖 preference 数据质量、reference policy 和离线 coverage。长度、格式等 spurious preference 可被模型放大。

### 4. 进入 RLVR 与 GRPO

- **官方 lecture**：[Lecture 16 — Post-training / RLVR](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf)。
- **论文**：[DeepSeekMath](https://arxiv.org/abs/2402.03300) 引入用于数学推理的 Group Relative Policy Optimization（GRPO）；[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 展示大规模 reasoning RL 的代表性结果。
- **工具文档**：[TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) 与 [vLLM](https://docs.vllm.ai/)。
- **解决的问题**：数学、代码等任务可用 verifier 给出 outcome reward，省去 learned critic；对同一 prompt 采样一组 completions，用组内相对 reward 构造 advantage。
- **关键公式/结论**：经典 group normalization 可写为
  \[
  A_i=\frac{r_i-\bar r}{s_r+\varepsilon}.
  \]
  policy objective 对 token probability ratio 做 PPO-style clipping，并可加入 reference KL。若整组 reward 相同，则相对 advantage 缺少学习信号；组大小、sampling temperature 与 pass rate 共同影响方差和探索。
- **对应作业**：A5 的 GRPO、Dr.GRPO、MaxRL、RFT、reward parsing 和多 seed proxy experiments。
- **局限**：binary verifier 只评价最终答案，模型可能 exploit parser、猜中答案或产生不可读 reasoning。组内标准差归一化会给 reward variance 很小的 prompt 过大尺度；Dr.GRPO 等变体尝试减少这类 normalization/length bias，但仍需实证检查。

### 5. 理解 on-policy、off-policy 与 sequence-level ratio

- **论文**：[Schulman et al., 2017, Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) 提供 clipped policy update 基础。
- **解决的问题**：在线生成成本高，复用旧 trajectories 能提高样本效率，但 behavior policy 与当前 policy 的分布偏移会增加 bias/variance。
- **关键结论**：
  - importance ratio 来自 \(\pi_\theta/\mu\)，其中 \(\mu\) 是真实 behavior policy；把 stale log-prob 当 current policy 会破坏校正。
  - token-level clipping 控制局部更新，sequence-level aggregation 更贴近整个 response 的分布变化，但长序列上的 ratio 易出现数值和方差问题。
  - clip fraction、KL、entropy、reward、response length 和 effective sample size 应联合监控。
- **对应作业**：A5 的 off-policy GRPO/GSPO、ratio/clipping tests 与 objective comparison。
- **局限**：过度 clipping 让梯度近乎消失，过少 clipping 又可能产生 destructive update；离线数据覆盖不到的新策略区域无法靠 importance sampling 补回。

### 6. 把 capability 与 safety 分开评估

- **官方 lecture**：[Lecture 17 — Alignment and multimodality](https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py)。
- **论文**：[Constitutional AI](https://arxiv.org/abs/2212.08073) 展示 rule-guided critique/revision 与 AI feedback 的一种路径。
- **解决的问题**：任务 accuracy、helpfulness、harmlessness、truthfulness 与拒答行为不是同一轴。单一 aggregate reward 会隐藏退化。
- **关键结论**：保留 held-out prompts 和独立 graders；报告 confidence interval、不同类别、长度和 seed；人工评估应随机化、盲测并记录 rubric。对开放式输出，LLM-as-a-judge 需要 position/order bias 与 self-preference 检查。
- **对应作业**：A5 supplement 的 safety alignment；主作业的 GSM8K/MMLU 只能覆盖 reasoning/capability 的狭窄切面。
- **局限**：benchmark contamination、grader leakage 和多次试验后的选择偏差会夸大改进。课程 proxy objective 的成功不能证明部署安全。

## 建议实验闭环

1. 固定 base/reference checkpoint、prompt template、sampling config、parser 和 held-out split。
2. 先跑 SFT/RFT baseline，再做 GRPO；每项至少多个 seeds，并保存逐样本 reward/length/KL。
3. 为 verifier 写 adversarial tests：错误格式、伪造答案、等价表达、超长输出和空输出。
4. 画 reward 与 exact accuracy、KL、entropy、length、clip fraction 的联合曲线；不要只展示 best checkpoint。
5. 对 preference/safety 方法记录数据来源、annotator/judge、rubric 和 disagreement。

## 阅读边界

Alignment objective 只是目标的代理。RLHF、DPO 或 RLVR 都不能从形式上保证 truthfulness、robustness 或 harmlessness；它们会继承数据、reward、verifier 和评估器的盲点。任何“对齐提升”都应限定到明确任务、分布、模型、预算和评估协议。
