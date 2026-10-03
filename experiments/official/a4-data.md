# A4 Data：从 Common Crawl 到语言模型训练集

> 中文导读；不替代[官方 handout](../../assignments/spring2026/assignment4-data/cs336_assignment4_data.pdf)，不包含分类阈值或 leaderboard 配方答案。

## 目标

构建可扩展、可审计的数据管线：HTML 抽取、语言识别、PII 掩码、有害内容识别、规则/学习式质量过滤、exact-line 与 MinHash+LSH 去重；最终从 2,500 个英文 WET 文件生成 GPT-2-tokenized 数据，在固定模型和训练过程下最小化 Paloma C4 100 domains validation perplexity。

## Problem、deliverable 与分值

| Problem | 分值 | 核心交付 |
| --- | ---: | --- |
| `look_at_cc` | 4 | 观察 WARC/WET、标注 25 个文档并讨论数据价值/风险 |
| `extract_text` | 3 | bytes→Unicode→Resiliparse 文本抽取；与 WET 对照 |
| `language_identification` | 6 | top language + confidence；20 个样本人工核验与阈值讨论 |
| `mask_pii` | 3 | email、US phone、IPv4 掩码，返回文本与替换数；误报/漏报分析 |
| `harmful_content` | 6 | NSFW/toxic label + confidence；20 样本核验与阈值讨论 |
| `gopher_quality_filters` | 3 | 实现题面指定 Gopher 规则并分析偏差 |
| `quality_classifier` | 15 | 训练质量分类器，返回 high/low quality 与 confidence |
| `exact_deduplication` | 3 | 两遍 hash-based exact-line dedup，保持输入文件名 |
| `minhash_deduplication` | 8 | normalization、MinHash、LSH、真 Jaccard 与 cluster retention |
| `filter_data` | 6 | 并行处理 2,500 WET；逐级保留/丢弃比例和总运行时间 |
| `inspect_filtered_data` | 4 | 5 个保留 + 5 个删除/修改样本，说明迭代 |
| `tokenize_data` | 2 | GPT-2 tokenizer + 每文档 EOS，`uint16.tofile`，报告 token 数 |
| `train_model` | 8 | 固定约 430M 模型训练；最佳 validation loss、曲线和方法说明 |

上述可确认总分为 71 分。Gradescope 提交 `writeup.pdf` 与 `code.zip`。

## 测试接口

`tests/adapters.py` 要求：

- `run_extract_text_from_html_bytes(bytes) -> str | None`；
- `run_identify_language(text) -> (label, score)`；
- `run_mask_emails` / `run_mask_phone_numbers` / `run_mask_ips -> (new_text, count)`；
- `run_classify_nsfw` / `run_classify_toxic_speech` / `run_classify_quality -> (label, score)`；
- `run_gopher_quality_filter(text) -> bool`；
- `run_exact_line_deduplication(input_files, output_directory)`；
- `run_minhash_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)`。

官方 tests 只是基本正确性/契约检查，不证明 classifier 在真实网页上校准良好。label 字符串、路径布局、确定性和 Unicode normalization 必须与 tests 一致。

## 数据与管线接口

- 输入格式包括 WARC/WET；建议 FastWARC 迭代 records、Resiliparse 抽取文本、fastText 做 language/safety/quality。
- 题面提供 `lid.176.bin`、Dolma/Jigsaw safety classifiers、Wikipedia external URLs 与英文 WET 共享数据。
- 最终 WET 起点已由 fastText 英文概率 ≥70% 预筛；你仍需设计后续过滤。
- validation 是 GPT-2-tokenized Paloma C4 100 domains。可用其设计 filter/classifier，但严禁把 validation 文本复制进训练集。
- 最终序列化必须为 GPT-2 token IDs、文档间追加 `<|endoftext|>`、`np.uint16` 原始二进制。

## 推荐实验矩阵

1. **Extractor：** 编码 `{UTF-8, legacy, malformed}` × 页面类型；与 Common Crawl WET 做长度、boilerplate 和人工质量对照。
2. **分类器校准：** language/NSFW/toxic/quality 各自做 threshold sweep；报告 retain rate、人工 precision/recall proxy 与典型错误。
3. **去重：** exact line on/off × MinHash `{num_hashes, bands, ngram, Jaccard threshold}`；测候选量、cluster size、保留率、CPU/RAM/时间。
4. **过滤消融：** base、逐步加 language/PII/safety/Gopher/quality/dedup；每一步记录输入、保留、修改、拒绝及 reason。
5. **数据训练：** 在小规模固定训练预算上比较 pipeline variants；选出少量候选后才进行昂贵主训练。
6. **最终审计：** retained/discarded 分层抽样，按 domain、长度、语言、质量分数和 rejection reason 检查偏差。

过滤顺序本身是变量：便宜/高召回规则通常前置，昂贵 classifier 和全局去重后置；但应通过实测验证，不凭直觉宣称最优。

## 硬件与规模

- 数据源为 2,500 个 raw WET；题面要求并行处理并估算扩展至整个 Common Crawl 的时间。
- 最终模型约 430M 参数，固定 8×B200、每卡 batch 128、16,384 steps、context 512，约采样 8.6B tokens；参考运行约 2 小时。
- 大规模 filtering 更依赖 CPU、RAM、磁盘吞吐与临时空间；MinHash/全局计数需特别记录 peak RSS 和中间数据大小。
- 修改模型架构或训练过程会破坏“只比较数据”的实验控制，官方最终训练不允许这样做。

## 提交物

- typeset `writeup.pdf`：所有短答、人工审查样本、阈值依据、filter funnel、运行时间、数据迭代、token 数、最终 learning curve/best loss；
- `code.zip`：模块、adapter、并行过滤、去重、tokenization 脚本及依赖；
- 使用 `test_and_make_submission.sh`，并确认解压后可再次运行同一脚本；
- 不提交共享模型、WET、classifier 权重或生成的大型 `.bin`，而是给下载/生成说明与校验值。

## 资源受限替代

- 使用官方 offline-only 数据或确定性 WET sample 完成全管线，再把“抽样结果”与 2,500-WET 正式结果分开。
- 用受控 synthetic documents 测每个 filter 的边界，用小真实样本做人工校准；不要仅用 synthetic 证明真实质量。
- 无 8×B200 时，固定一个更小模型和 token budget 比较数据消融；不把 loss 写成官方 leaderboard 成绩。
- 下载受限时可减少 `EnglishWetFiles.n_files`；报告样本选择和 coverage，避免外推全量吞吐。
- 有害/成人网页可能造成心理风险；抽样审查可最小化暴露、只记录必要片段，并避免在报告复制敏感 PII。

## 版本风险与本仓库报告

- 本地 handout 版本 `26.0.1`；固定 commit `0555bea66369872d912652debf10b115ca0688c8`，见 [`UPSTREAM.md`](../../UPSTREAM.md)。
- Common Crawl 路径、远端 URL、classifier artifact 与 Modal volume 可能失效；保留 checksum 和本地缓存来源。
- fastText/Resiliparse 的版本、文本 normalization、hash seed 会改变边界结果；须固定版本和随机种子。
- 本仓库只做 1,000 条 sample WET 和离线 proxy，不声称 2,500-WET/8×B200 成绩。
- 报告：[源码](../../assignments/spring2026/assignment4-data/report/main.tex) · [PDF](../../assignments/spring2026/assignment4-data/report/writeup.pdf) · [验证记录](../../assignments/spring2026/assignment4-data/VERIFICATION.md)。
