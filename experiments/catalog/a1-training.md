# A1：Tokenizer、训练与架构消融

- **目的**：验证从 byte-level BPE 到 Transformer training/generation 的完整链路。
- **原始记录**：[`assignment1-basics/report/results/raw`](../../assignments/spring2026/assignment1-basics/report/results/raw/)
- **报告**：[`main.pdf`](../../assignments/spring2026/assignment1-basics/report/main.pdf)
- **规模**：75 个 runs、27 组图；RTX 4080 SUPER + RTX 6000D。

## 代表实验
1. BPE：TinyStories/OWT 训练时间、RSS、compression、encoding throughput。
2. Controlled training：LR coarse/fine sweep、batch 1–768、三 seed stability。
3. Architecture：RMSNorm、pre/post-norm、RoPE/NoPE、SwiGLU/SiLU、weight tying。
4. Systems：`torch.compile`×precision、gradient accumulation 与 physical batch。
5. Generation：固定 prompts/temperature 的 repetition 与长度。

## 关键结论
- OWT BPE 的主要成本是 pair-count state（2.97 h、46.45 GiB RSS），不是 final encoding。
- Batch 256 后 RTX 6000D throughput 饱和；更大 batch 不自动改善 loss。
- BF16+compile 相比 eager BF16 约 2×；无 GradScaler 的 FP16 数值不稳定。
- RMSNorm 是稳定训练必要条件；合理初始化后的 tied embeddings 提升参数效率。

## 复现
```bash
cd assignments/spring2026/assignment1-basics
python scripts/build_report_assets.py --raw report/results/raw \
  --figures report/figures --summary report/results/summary.json
```
