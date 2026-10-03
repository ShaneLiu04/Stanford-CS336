# A5：GRPO Objective Mechanics

- **实现**：tokenization/mask、log-prob/entropy、reward/advantage、GRPO/Dr.GRPO/
  MaxRL/RFT、off-policy GRPO/GSPO、SFT packing、metrics parsing、DPO。
- **测试**：26/26；AutoDL 同步归档。
- **Proxy**：7 objectives × 4 seeds × 160 steps，4480 metric rows。
- **报告**：[`writeup.pdf`](../../assignments/spring2026/assignment5-alignment/report/writeup.pdf)

## 观察维度
Answer/format reward、entropy、grad norm、response length、pass@k、clip fraction、
policy probability 与 seed variance。

## 学习重点
- Sequence normalization 会产生 length bias；constant normalization 保留 token 权重。
- Std/mean/no normalization 对 group difficulty 的权重不同。
- Token-level GRPO clipping 与 sequence-level GSPO clipping 约束不同。
- Reward 上升若伴随 entropy collapse 或 format shortcut，不能直接等同 reasoning 改善。

## 局限
无 SUNET_ID/Modal；向量化 bandit proxy 只验证 objective mechanics，不代表
OLMo-2/GSM8K/B200 accuracy。
