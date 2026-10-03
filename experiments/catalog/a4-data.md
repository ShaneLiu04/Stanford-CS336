# A4：WET Filtering 与 Deduplication

- **实现**：HTML extraction、lid、PII、NSFW/toxic、Gopher、quality、exact line、
  MinHash+LSH、GPT-2 tokenization。
- **验证**：本地与 AutoDL 21/21 tests。
- **真实样本**：官方 example WET 中 1000 records。
- **报告**：[`writeup.pdf`](../../assignments/spring2026/assignment4-data/report/writeup.pdf)

## 实验设计
受控 corpus 隔离规则边界；真实 WET 用于观察 language/quality/safety/domain 分布。
12 组 Gopher×safety×quality-threshold 消融同时记录 keep rate、rejection reason、
characters、runtime。

## 结果
真实 WET：English 325、Chinese 239；733 通过 Gopher，88 Wiki-like；
NSFW/toxic 为 12/24。该 shard 不是全库随机样本，比例不可外推。

## 局限
无 SUNET_ID/Modal，未运行 2500 WET 与 8×B200；没有用不足语料伪造 430M downstream。
