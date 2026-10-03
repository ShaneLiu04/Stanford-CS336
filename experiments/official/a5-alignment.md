# A5 Alignment：Reasoning RL 与 GRPO 变体

> 中文导读；不替代[主 handout](../../assignments/spring2026/assignment5-alignment/cs336_spring2026_assignment5_alignment.pdf)或[可选 supplement](../../assignments/spring2026/assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)，不提供推导答案、实现代码或实验结论。

## 目标

用 OLMo-2-0425-1B 在 GSM8K 上建立 prompting baseline，并从策略梯度推导/实现 on-policy GRPO；随后控制变量比较 sequence/advantage normalization、RFT、Dr. GRPO、MaxRL，以及 off-policy 的无校正、token-level clipped GRPO、sequence-level GSPO，最后提出一个只改变单一因素的新 estimator。

## Problem、deliverable 与分值

### Baseline 与标准 GRPO

| Problem | 分值 | 核心交付 |
| --- | ---: | --- |
| `prompting_baselines` | 5 | question-only、zero-shot/few-shot R1-zero 的 GSM8K 指标、解析错误和样例分析 |
| `baseline_calcs` | 5 | 二元 policy gradient 在不同 baseline 下的 variance 推导 |
| `tokenize_prompt_and_output` | 1 | prompt/output 分开 tokenize 后直连，返回 shifted input/labels/response mask |
| `get_response_log_probs` | 1 | per-token conditional log-prob，可选 entropy |
| `compute_rollout_rewards` | 1 | rollout reward 与 component metadata |
| `compute_group_normalized_rewards_grpo` | 1 | group mean baseline + std normalization |
| `compute_policy_gradient_loss_on_policy` | 1 | per-token on-policy loss |
| `aggregate_loss_across_microbatch_sequence` | 0.5 | 先 sequence 内、再 batch 间平均 |
| `grpo_train_step_standard_on_policy` | 5 | accumulation、clip、optimizer step 与日志 |
| `grpo_experiments_standard_on_policy` | 10 | 4 seeds 完整曲线/rollouts；平均 validation accuracy ≥25% |
| `grpo_learning_rate` | 3 | 至少一个更低/更高 LR 的 sweep |
| `grpo_prompt_ablation` | 3 | question-only、zero-shot、three-shot prompt 对照 |

### On-policy 变体

| Problem | 分值 | 核心交付 |
| --- | ---: | --- |
| `think_about_length_normalization` | 1 | per-sequence 与 constant normalization 比较 |
| Dr.GRPO group / aggregate 两项 | 0.5 / 0.5 | 支持无 advantage normalization 与 constant loss normalization |
| `think_about_rft` | 2 | RFT 与 Dr.GRPO expectation/variance 比较 |
| `derive_difficulty_reweightings` | 6 | Dr.GRPO、GRPO、MaxRL 隐含 prompt reweighting 推导 |
| `think_about_advantage_normalization` | 2 | std/mean/none 的利弊 |
| `compute_group_normalized_rewards_maxrl` | 0.5 | mean normalization |
| `grpo_train_step_variants_on_policy` | 2.5 | 全 on-policy 组合并跳过 zero-advantage sequences |
| `grpo_experiments_variants_on_policy` | 10 | GRPO-constant、Dr.GRPO、RFT、MaxRL，4 seeds |

### Off-policy 与自选方法

| Problem | 分值 | 核心交付 |
| --- | ---: | --- |
| `derive_surrogate_objectives` | 2 | pairwise importance reweighting 的 surrogate objective |
| `compute_policy_gradient_loss_off_policy` | 1 | token-level no-clip 与 PPO/GRPO clip |
| `think_about_importance_reweighting` | 2 | none/token-level/GSPO 的 bias-variance 比较 |
| `compute_policy_gradient_loss_off_policy_gspo` | 1 | geometric-mean sequence weight + clipping |
| `grpo_train_step_off_policy` | 2.5 | old log-probs、cliprange 与完整参数组合 |
| `grpo_experiments_off_policy` | 10 | 32× off-policy 的四种算法，4 seeds，含 clip fraction |
| `try_your_own` | 10 | 只改变一个因素的新 estimator，理论动机、多 seed 对照 |

主作业可确认总分 90。题面列出的显式实验预算为标准 GRPO 2、LR 4、prompt 4、on-policy variants 8、off-policy variants 8 B200-hours；自选 estimator 另需预算。

## 关键接口

主作业 `tests/adapters.py`：

- `run_tokenize_prompt_and_output`：返回同形状 `input_ids`、`labels`、`response_mask`；
- `run_get_response_log_probs`；
- `run_compute_rollout_rewards`；
- `run_compute_group_normalized_rewards`：`baseline ∈ {mean, none}`，normalizer `{std, none, mean}`；
- `run_compute_policy_gradient_loss`：importance method `{none, noclip, grpo, gspo}`；
- `run_aggregate_loss_across_microbatch`：`{sequence, constant}`；
- `run_grpo_train_step`：组合上述选项，正确处理 gradient accumulation。

运行时另依赖 `VLLMServer.start/generate_completions/init_weight_sync/sync_policy_weights` 和 grader 的 `r1_zero_reward_fn` / `question_only_reward_fn`。response mask 与 shifted labels 对齐、old log-probs 的采样 policy 版本、mask 后 normalization denominator 是最常见的静默错误。

## 官方基线配置与实验矩阵

题面建议标准 run：6,400 train、1,024 validation、200 rollout steps、LR `1e-5`、rollout/train batch 256 responses、group size 8、32 accumulation steps、temperature/top-p 1、max 512 tokens、grad norm 1；AdamW betas `(0.9, 0.95)`、无 weight decay。每 10 rollout batches 验证，最终至少 4 seeds。

推荐按阶段冻结变量：

1. **Prompting：** 3 prompts × 固定 generation config；分开记录 answer/format reward、response length、parser false negative。
2. **Sanity：** 约 50 steps 看 validation reward、entropy、gradient norm 和 rollouts 是否合理，再投入多 seed。
3. **LR：** 标准 GRPO 先调 LR，其他算法沿用时只能作“在 baseline-tuned LR 下”的比较。
4. **On-policy：** 标准 GRPO、constant-GRPO、Dr.GRPO、RFT、MaxRL；统一 seed、prompt、sample budget。
5. **Off-policy：** rollout 256、train batch 8、accumulation 1；`none/noclip/grpo(ε=.2)/gspo(ε=3e-4)`。
6. **自选 estimator：** 相对一个已测 baseline 仅改一项，预注册主要指标和预期。

所有图应同时呈现 4 条 seed 轨迹或 mean + uncertainty，不能只画最佳 seed。保存 rollout 文本、reward components、length、entropy、gradient norm、clip fraction、old/current log-ratio 和环境信息。

## 硬件

- 模型为 OLMo-2-0425-1B；标准架构把 Hugging Face policy/optimizer 放 GPU 0，vLLM + KV cache 放 GPU 1，并通过 NCCL 同步权重。
- BF16 与 FlashAttention-2 用于节省训练显存；vLLM 默认 `gpu_memory_utilization=0.9`。
- 显式实验预算合计很高，且 RL seed variance 大；单次成功不能替代 4-seed 要求。
- 主作业的 B200-hour 标题是资源规划值，非 B200 上的 wall time 不可直接换算。

## 可选 safety / RLHF supplement

Supplement 完全可选，转向 Llama 3.1 8B 的通用对话对齐：MMLU/GSM8K/AlpacaEval/SimpleSafetyTests zero-shot、UltraChat SFT packing、pairwise preference 与 DPO；自动 judge 使用 Llama 3.3 70B Instruct。对应 adapter 包括 packed SFT dataset、batch iterator、MMLU/GSM8K parser 与 per-instance DPO loss。应与主作业 90 分 deliverables 分开记录。

## 提交物

- `writeup.pdf`：推导、设计选择、全指标曲线、seed variance、定性 rollouts、失败/不稳定性；
- `code.zip`：通过 `test_and_make_submission.sh` 生成；
- 代码需从头实现核心 RL 组件，可用 HF 加载模型、vLLM 推理，但不能以 Trainer 等高层训练器代替作业核心；
- 不把模型权重、HF/Modal token、SUNET_ID 或大型 checkpoint 打包。

## 资源受限替代

- CPU：完成纯张量 normalization/loss、parser 和 tiny causal LM 单测。
- 单 GPU：policy 与生成分时运行、缩小 batch/group/max tokens；可验证算法但吞吐和训练动力学不等同官方双卡配置。
- 无 OLMo-2/B200：用 tiny model 或合成 log-probs 做 objective/gradient equivalence；本仓库的 vectorized proxy 不是 GSM8K accuracy 实验。
- 预算有限时优先 correctness、50-step sanity 和少量 seeds；不要用一个 seed 冒充完整统计结论。
- 自动 grader 有系统性误差，需人工检查分类 2/3 的样本；但不可在评估中手工改 reward 来提高成绩。

## 版本与授权风险

- 主 handout 与 supplement 均为 `26.0.0`。
- [`UPSTREAM.md`](../../UPSTREAM.md) 仅记录固定 commit `c2734a26308710949fe13226960a1e8cece94b7e`，没有导入官方源码：同步时该版本未发现许可证。不要复制、再发布或把本地实现描述成官方快照。
- 本地目录是非课程提交的独立实现；adapter/测试/报告可能有自学扩展，需对照 PDF。
- HF model access、FlashAttention、vLLM/NCCL 和 Modal shared volume 都可能随版本或权限变化，应固定 revision 与环境。
- 本仓库报告：[PDF](../../assignments/spring2026/assignment5-alignment/report/writeup.pdf) · [资产说明](../../assignments/spring2026/assignment5-alignment/report/README.md) · [验证记录](../../assignments/spring2026/assignment5-alignment/VERIFICATION.md)。
