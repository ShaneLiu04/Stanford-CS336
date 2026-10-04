---
title: "Lecture 17 — Multimodal Models and Alignment"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-05-27"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py"
---

# Lecture 17 — 多模态对齐：让感知 token 接入语言模型

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
- 适用对象：自学者、多模态模型工程师与研究者

## 摘要

多模态语言模型的第一性问题，是把高带宽感知输入压缩成有限 token 并在计算预算内接入
语言模型。本文沿“token 化 → 连接结构 → 对齐阶段 → 评估证据”四层展开：图像经
ViT patch 化后 token 数随分辨率二次增长 [[1]](#ref-1)，对比学习建立全局语义对齐
[[2]](#ref-2)；连接结构从最小化的 linear projector，到 query 压缩的 Q-Former，再到
gated cross-attention 与交错文档预训练，分别代表成本-灵活性谱系上的不同取舍
[[3]](#ref-3)[[4]](#ref-4)[[5]](#ref-5)[[6]](#ref-6)；离散 VQ token 则为统一
自回归生成提供了另一条路线 [[7]](#ref-7)[[8]](#ref-8)。对齐被区分为表示对齐、
指令对齐、偏好与安全对齐三个阶段，各自有不同的数据契约与失败模式。评估上的核心
论点是：开放式“回答合理”无法区分真实感知与语言先验，必须用 counterfactual 图像
对、遮挡 ablation 与分轴报告证明模型确实读取了输入模态。本文同时给出
placeholder/mask/position ids 的形状审计清单、幻觉与 prompt injection 的测试设计，
以及多模态结论的效度边界。

**关键词：** Multimodal LLM；ViT；CLIP；Flamingo；Q-Former；Visual Instruction
Tuning；VQ-VAE；Grounding；Hallucination

## 本文贡献

1. 统一比较 projector、cross-attention 与离散统一 token 三种连接结构的计算复杂度、
   训练成本与部署代价；
2. 把多模态对齐分解为表示/指令/偏好-安全三阶段，并给出各阶段的数据契约；
3. 形式化“语言先验 vs 视觉证据”的诊断协议：counterfactual pairs、遮挡与 image
   ablation；
4. 整理感知-推理-grounding-幻觉-鲁棒-安全-效率的分层评估框架；
5. 给出 placeholder/mask/位置编码的形状审计清单与训练单测设计。

## 学习目标

- 理解图像/音频如何被 token 化，以及 token 数如何决定 attention 成本。
- 比较 projector、跨注意力和统一 token 序列三种连接方式。
- 区分表示对齐、指令对齐、安全对齐与人类偏好对齐。
- 设计覆盖感知、推理、grounding、幻觉、鲁棒性和安全性的评估。

## 先修知识

- Lecture 01：tokenization 与序列长度预算（视觉 token 化的同构问题）。
- Lecture 15：对齐三阶段与偏好数据（多模态对齐的直接方法学前置）。
- Lecture 16：verifiable rewards 与安全评估（多模态评估的对照范式）。

## 相关工作与技术谱系

视觉 token 化的基础是 ViT：将图像切分为不重叠 patch、线性映射后按序列建模，
确立了“图像即 token 序列”的范式 [[1]](#ref-1)。CLIP 用 4 亿 image-caption 对做
对比学习，证明自然语言监督足以产生可迁移的视觉表示，其 InfoNCE 目标也成为多模态
表示对齐的标准构件 [[2]](#ref-2)。

在连接结构上，Flamingo 以 gated cross-attention 把视觉信息按需注入冻结的 LLM，
支持交错多图的小样本学习 [[3]](#ref-3)；BLIP-2 用 Q-Former 以少量 learned queries
从冻结视觉 encoder 中提取与语言最相关的表示，把可变视觉 token 压缩为固定长度
[[4]](#ref-4)；LLaVA 则证明一个简单 linear projector 加 GPT-4 生成的指令数据即可
实现有效的 visual instruction tuning [[5]](#ref-5)；OBELICS 进一步用网络规模的
交错 image-text 文档做预训练，为多图对话提供了数据基础 [[6]](#ref-6)。

离散路线方面，VQ-VAE 引入 codebook 量化把连续观测转为离散隐变量 [[7]](#ref-7)；
VQGAN 将其与感知损失结合，实现高分辨率图像的 token 化，使单一自回归 Transformer
可统一建模多模态生成 [[8]](#ref-8)。

## 1. Multimodal tokenization

### 图像 patch tokens

对 \(H\times W\) 图像、patch 大小 \(P\times P\)，不重叠 patch 数为

\[
N_{\text{img}}=\frac HP\frac WP.
\]

每个 patch 展平并线性映射为 \(d_v\) 维，再经 ViT 得到视觉 tokens
\(Z_v\in\mathbb R^{N_{\text{img}}\times d_v}\) [[1]](#ref-1)。提高分辨率会二次增加
token 数；若直接做全序列 self-attention，成本近似
\(O((N_{\text{text}}+N_{\text{img}})^2)\)。

动态分辨率常把大图切成 tiles，加 thumbnail 或二维位置编码。必须保留 tile 顺序和
几何，否则 OCR、图表和空间关系会退化。

### 离散视觉/音频 tokens

VQ 方法由 encoder 得到连续表示，再选择 codebook 最近项 [[7]](#ref-7)：

\[
k^\star=\arg\min_k\|z_e(x)-e_k\|_2^2.
\]

VQGAN 在此基础上加入感知损失与对抗训练，使离散 codebook 能在高分辨率下保真
[[8]](#ref-8)。离散 tokens 可与语言 token 统一自回归建模，也便于生成；代价是量化
误差和大 codebook 训练。音频还可用连续帧、codec tokens 或 spectrogram patches；
长音频需要下采样/分块，否则序列过长。

### Tokenizer 的根本取舍

- token 少：便宜，但小字、对象计数和细粒度定位损失。
- token 多：保留细节，但显存、延迟和长上下文竞争加剧。
- 固定分辨率：批处理简单，但扭曲长宽比。
- 自适应分辨率：更准确，但 batching、位置编码和上限控制复杂。

## 2. Projector：最小连接器

当视觉 encoder 输出维度 \(d_v\)，LLM hidden size 为 \(d_\ell\)，projector 做

\[
H_v=f_{\text{proj}}(Z_v)\in
\mathbb R^{N_v\times d_\ell}.
\]

\(f\) 可以是线性层、MLP、卷积/downsampler 或 query resampler。然后把视觉 embeddings
放入 `<image>` 占位区域，与 text embeddings 拼接。

- **线性/MLP projector**：参数少、训练快，适合冻结 encoder/LLM 的第一阶段表示
  对齐；LLaVA 证明这一极简结构配合高质量指令数据已能取得强效果 [[5]](#ref-5)。
- **resampler/Q-Former**：用固定数量 learned queries 压缩可变视觉 tokens，控制
  LLM 上下文成本 [[4]](#ref-4)。
- 只训练 projector 能建立粗粒度语义桥梁，但复杂 grounding 往往需要解冻部分
  encoder/LLM 或更强连接器。

形状审计比层数更重要：batch、视觉 token 数、hidden size、attention mask 和
position ids 必须一致；`<image>` token 数与实际替换 embeddings 数不一致会静默错位。

## 3. Cross-attention：语言按需读取感知记忆

令语言 hidden states 为 \(H_t\)，视觉表示为 \(H_v\)：

\[
Q=H_tW_Q,\quad K=H_vW_K,\quad V=H_vW_V,
\]

\[
\operatorname{CrossAttn}(H_t,H_v)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]

相比把所有视觉 tokens 前置拼入：

- cross-attention 可在若干 LLM 层插入，让 text queries 按需读取视觉 memory，
  Flamingo 以此支持交错图文序列的少样本学习 [[3]](#ref-3)；
- 成本约 \(O(N_tN_v)\)，且视觉 memory 可复用；
- 但改动语言模型结构更多，预训练和部署复杂。

gated cross-attention 常以接近零的 gate 初始化，减少接入新模态时破坏原语言能力
[[3]](#ref-3)。门控过小也可能造成“模型忽略图像”，需做 image ablation 检查。

## 4. 三阶段 alignment

### 表示对齐

用 image-caption 对让 projector/encoder 输出落入 LLM 可使用的表示空间。可用
next-token caption loss，或对比学习（CLIP 范式）[[2]](#ref-2)：

\[
\mathcal L_{\text{InfoNCE}}
=-\frac1B\sum_i
\log\frac{\exp(s(v_i,t_i)/\tau)}
{\sum_j\exp(s(v_i,t_j)/\tau)}.
\]

### 多模态 instruction tuning

用图像问答、OCR、grounding、图表、文档和多轮数据训练 response-only loss。数据
混合必须控制 caption、OCR 和开放问答比例，避免模型只学会描述图像。LLaVA 表明
GPT-4 生成的指令跟随数据可以低成本撬动这一阶段 [[5]](#ref-5)；OBELICS 类交错
语料则为多图、多轮对话提供分布基础 [[6]](#ref-6)。

### 偏好与安全对齐

对同一 multimodal prompt 收集 chosen/rejected，可做 RM/DPO/PPO；但偏好标注必须让
annotator 真正看到所有模态。纯文本 judge 无法可靠判断图像事实。还需覆盖敏感图像、
身份推断、医疗建议、版权、深伪和 prompt injection。

## 5. 多模态幻觉与 grounding

模型可能：

- 描述不存在对象；
- 对存在对象给错属性、数量或位置；
- 依靠问题先验，不看输入模态；
- OCR 后遵循图中的恶意指令；
- 面对模糊输入仍过度自信。

仅用开放式“回答看起来合理”无法区分。应加入 counterfactual pairs：替换图像但保留
问题、遮挡关键区域、交换对象属性，验证输出随证据变化。

## 实现映射

本仓库 A4/A5 没有多模态 encoder/projector/cross-attention 实现，不能把文本模块宣称
为多模态实现。可复用的是训练与审计契约：

| 多模态概念 | 可复用的仓库模式 | 需要新增的测试 |
| --- | --- | --- |
| image-text 序列边界 | `pipeline.py::tokenize_documents` 的 EOT 思路 | `<image>` placeholder 数与视觉 embeddings 一致 |
| response-only mask | `grpo.py::tokenize_prompt_and_output` | 视觉/prompt token 不参与 response loss |
| SFT packing | `supplement.py::PackedSFTDataset` | 不把图像引用跨样本错配；跨文档 mask |
| DPO | `supplement.py::compute_per_instance_dpo_loss` | chosen/rejected 共用同一图像与 prompt |
| verifier 分量 | `grpo.py::compute_rollout_rewards` | OCR、grounding、answer、safety 分量 |
| provenance | A4 `PipelineStats` 与数据清单原则 | 图像 URL/许可、变换、hash、caption 来源 |
| 实验纪律 | A4/A5 报告 | 固定视觉预处理、分辨率和 judge 版本 |

建议的最小模块接口：

```text
vision_encoder(pixel_values[B,C,H,W]) -> visual_tokens[B,Nv,Dv]
projector(visual_tokens) -> multimodal_tokens[B,Nm,Dllm]
merge(input_ids, multimodal_tokens, image_positions)
  -> inputs_embeds[B,Nt+Nm,Dllm], attention_mask, labels
```

训练单测应检查不同分辨率、零图像、多图、截断、padding side 和 mixed precision。

## 评估框架

不要用一个综合 benchmark 代表全部能力。至少分层：

1. **感知**：对象、属性、计数、OCR、音频识别。
2. **关系与 grounding**：空间关系、bounding box、指代表达。
3. **知识与推理**：图表、文档、科学图、跨模态多步问题。
4. **生成质量**：relevance、事实一致性、可读性。
5. **幻觉**：不存在对象、否定问题、counterfactual image pairs。
6. **鲁棒性**：分辨率、压缩、裁剪、遮挡、噪声、语言变化。
7. **安全**：图像文字注入、敏感属性推断、隐私、医疗/地理定位、深伪。
8. **效率**：视觉 tokens、首 token 延迟、峰值显存和吞吐。

开放式 judge 应随机化候选顺序、隐藏模型身份、做人工校准并报告置信区间。OCR exact
match 需说明大小写、Unicode 和版面 normalization；grounding 同时报 IoU threshold
和平均 IoU。

## 关键公式速查

- 图像 patch 数：\(N_{\text{img}}=(H/P)(W/P)\)。
- Projector：\(H_v=f_{\text{proj}}(Z_v)\in\mathbb R^{N_v\times d_\ell}\)。
- Cross-attention：\(\operatorname{softmax}(QK^\top/\sqrt{d_k})V\)。
- VQ：\(k^\star=\arg\min_k\|z_e(x)-e_k\|_2^2\)。
- 表示对齐：InfoNCE 使匹配 image-text 相似度高于 batch 内负例。

## 易错点、伦理与安全

- 训练时 resize/crop 与评估不同，位置/文字被破坏。
- projector 输出 token 数变化后忘记同步 attention mask 与 position ids。
- 图像和文本 augmentation 独立 shuffle，形成错误配对。
- 多图对话中 `<image_1>/<image_2>` 引用错位。
- 用 text-only judge 评价视觉事实，得到虚假高分。
- benchmark 图片出现在预训练 crawl 中，构成 contamination。
- 人脸、车牌、屏幕、地理位置含 PII；公开 URL 不等于同意做身份识别。
- caption 会固化性别、种族、职业等刻板印象；模型还可能从背景推断敏感属性。
- 图中可嵌入 prompt injection；视觉 OCR 内容应视为不可信数据，而不是高优先级指令。
- 生成/编辑模型需考虑版权、水印、冒充与深伪滥用；能力提升不等于部署许可。
- 用单一 benchmark 平均分代表多模态能力；应分感知/推理/grounding/幻觉/安全
  分轴报告，MMMU 之外补 DocVQA/MathVista/EgoSchema 等专项。
- 选择题评测不做选项位置偏差控制；应采用循环评估（正确答案轮换到每个位置、
  全部答对才计对），开放式 judge 同步随机化候选顺序。

## Checklist

- [ ] 固定并记录 decoder、resize/crop、归一化和视觉 encoder revision。
- [ ] 视觉 token 数、projector 维度、placeholder、mask 和 position ids 有形状单测。
- [ ] image-caption 对有 hash/provenance/license，拆分按近重复簇完成。
- [ ] SFT loss 明确是否屏蔽视觉和 prompt tokens。
- [ ] 多图、长图、空图、损坏文件和极端宽高比有测试。
- [ ] 做 text-only/image-shuffle/遮挡 ablation，确认模型确实使用输入模态。
- [ ] 分别报告感知、推理、grounding、幻觉、安全和效率。
- [ ] judge 经人工校准，避免模型家族自偏好和位置偏差。
- [ ] 图像文字 prompt injection 与敏感属性推断有红队测试。

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| 换图答案不变 | image ablation/counterfactual | 语言先验主导、gate 接近零 [[3]](#ref-3) |
| OCR 小字丢失 | 视觉 token 数与分辨率 | patch 过粗、resize 破坏细节 |
| 多图对话张冠李戴 | placeholder 对齐审计 | `<image_N>` 与 embeddings 错位 |
| loss 正常但 grounding 差 | 连接器容量 | projector 过弱、encoder/LLM 冻结过度 |
| 训练崩溃或静默降质 | 形状/位置编码单测 | mask 与 position ids 未同步 |
| judge 分数与人工背离 | judge 模态能力 | text-only judge、自家族偏好 |
| 长序列显存爆炸 | token 数统计 | 高分辨率 patch 数二次增长 |
| 输出遵循图中恶意指令 | 信任边界设计 | OCR 内容被当作高优先级指令 |

## 讨论：效度威胁与结论边界

### Construct validity

- “看图回答正确”可能来自语言先验或数据集伪相关，counterfactual 设计才是感知的
  操作化定义；
- VQA 准确率混合了感知、知识与推理三种构造；
- judge 对长而自信的视觉描述有系统性偏好。

### Internal validity

- 同时改变分辨率、token 数与训练数据时，无法归因性能变化；
- 训练/评估预处理不一致（resize、crop、归一化）会制造虚假差距；
- 视觉 encoder 的预训练数据若与 benchmark 重叠，形成 contamination。

### External validity

- 英文-centric caption 数据上学到的对齐不保证跨语言、跨文化成立；
- 特定分辨率/tile 策略下的结论不能外推到任意宽高比与真实照片分布；
- 安全红队结论仅覆盖被测的攻击面，图像注入的变体空间是开放的。

论文式表述应报告：counterfactual 一致率、分轴置信区间、预处理流水线版本与 judge
校准协议，而不是单一的 benchmark 平均分。

## 面试要点速记

**高频问题与答题要点**

1. **Q：三种连接结构怎么选？** 要点：linear projector（LLaVA）最简，冻结
   两塔阶段够用；Q-Former/Flamingo 式 cross-attention 压缩 token 数或按需
   读取，但结构侵入；VQ 统一 token 支持自回归生成，代价是量化损失。
2. **Q：视觉 token 数与成本？** 要点：\((H/P)(W/P)\)；全拼接 attention 为
   \(O((N_t+N_v)^2)\)；高分辨率靠 tiles + 缩略图，但必须保序保几何。
3. **Q：多模态幻觉如何操作化？** 要点：counterfactual image pairs（换图不
   改问）、遮挡 ablation、image-shuffle 一致性；“回答合理”≠看见证据。
4. **Q：CLIP 的目标一句话？** 要点：batch 内 InfoNCE 对比，让匹配图文对的
   相似度高于所有负对，学到对齐的联合表示空间。
5. **Q：图内 prompt injection 怎么防？** 要点：OCR 文本是不可信数据；系统
   指令 / 用户输入 / 图像文字三级信任边界必须显式建模。
6. **Q：MMMU 为什么几乎必测？** 要点：大学水平，11,500 题、6 学科、30 主题、
   183 子领域，同时考知识与推理；Llama 3.1/Gemini 1.5/Claude 3/Qwen2.5-VL/
   DeepSeek-VL2 发布时均测。
7. **Q：循环评估（circular eval）为什么能防位置偏差？** 要点：选择题得分受
   选项排列位置影响；把正确答案轮换到每个选项位置、全部答对才计对，
   MMBench 以此消除位置偏差。
8. **Q：多模态 RL 的落地形态？** 要点：GRPO 及序列级变体 GSPO 已用于多模态
   推理与 GUI 计算机使用 agent 训练；OSWorld 公开案例 62.2%→72.6%。

**必背数字**

- patch 数公式；GQA 级压缩思想在视觉 token（resampler 查询数）上的同构；
  面试必答：多模态评估必须分感知/推理/grounding/幻觉/安全五轴报告。

**工业界参照**

- MMMU：11,500 题，6 学科/30 主题/183 子领域，大学水平，选择题+问答，
  规则化评分（单词/短语匹配）。
- MMBench：3,217 题，80% 互联网 + 20% 开源数据；过滤"纯文本也能答对"与
  "所有 VLM 都答错"的题；感知+推理 2 大维度、20 子能力，循环评估防位置偏差。
- MathVista：6,141 题 = 3 个新建 + 9 个数学 QA + 19 个需数学推理的 VQA
  数据集；五大任务：图形 QA/几何问题/数学应用题/教科书 QA/视觉 QA。
- DocVQA：50k，UCSF Industry Documents Library（烟草/食物/药物/石油化学
  文档）众包 QA；2015–2017 年代数据集被 MLLM 复用的典型。
- VSI-Bench：空间智能评测，SOTA 中最好的 Gemini-1.5 Pro 仅 48.8%。
- OSWorld：GUI 计算机使用 agent，GRPO/GSPO 多模态 RL 公开案例
  62.2%→72.6%。

## 行业现状与最新进展（2024–2026）

### 多模态评测谱系：从感知到生成

多模态评测在 2023 年集中爆发，2024 年起收敛为"必测组合 + 分轴补测"格局；
大量评测复用 CV/旧数据（2007 DocVQA 口径、2015 LibriSpeech），
contamination 风险必须显式审计。

| Benchmark | 年份/来源 | 规模 | 覆盖面 | 关键设计 |
| --- | --- | --- | --- | --- |
| MMMU | 2023 | 11,500 题 | 6 学科/30 主题/183 子领域，大学水平 | 选择题+问答；规则化评分（单词/短语匹配） |
| MMBench | 2023，上海AI实验室+浙大 | 3,217 题 | 感知+推理 2 大维度、20 子能力 | 循环评估防位置偏差；80% 互联网+20% 开源数据；过滤纯文本可答/全员答错的题 |
| MathVista | 2023，UCLA+UW+微软 | 6,141 题 | 3 新建 + 9 数学 QA + 19 数学推理 VQA | 五大任务：图形 QA/几何/应用题/教科书 QA/视觉 QA |
| DocVQA | 2015–2017（MLLM 复用） | 50k | 烟草/食物/药物/石油化学文档 | UCSF Industry Documents Library；众包 QA |
| EgoSchema | 2023，伯克利 | 5k+ | Ego4D 第一视角视频，人类活动与行为理解 | GPT-4/Bard/Claude 生成 QA；五选一 |
| VBench | 2024，南洋理工+上海AI实验室 | 文生视频 ~1600、图生视频 1118 | 视频生成 16 个维度 | +1 人类偏好注释集；行业口径称"生成模型的 MMMU" |

模型选测实践：MMMU 几乎必选——Llama 3.1/Gemini 1.5/Claude 3/Qwen2.5-VL/
DeepSeek-VL2 均测；其余百花齐放（AI2 Diagram/ChartQA/DocVQA/MathVista/
EgoSchema/LibriSpeech）。这与本讲"分轴报告、不要单一平均分"的评估框架一致。

### 原生多模态趋势

- GPT-4 引入图文输入：视觉问答/图表解释/截图分析/文档理解成为标配能力。
- GPT-4o：端到端实时多模态交互；产品形态由文本助手向综合交互入口升级。
- o3/o4-mini：将推理延伸到工具使用与多模态，"推理 + 感知"合流。
- 趋势解读：从"视觉 encoder + LLM 拼接"走向原生联合训练，但 projector/
  cross-attention 谱系仍是多数开源模型（LLaVA 系/Q-Former 系）的主干。

### 空间智能缺口

VSI-Bench（李飞飞/谢赛宁）测空间智能：SOTA MLLM 普遍不足，最好的
Gemini-1.5 Pro 仅 48.8%。结论：训练数据需加入空间知识，可提升空间生成与
具身智能表现。这是"感知 token 接入 ≠ 空间理解"的直接证据，呼应本讲的
counterfactual 诊断：模型可能背题而不建立空间表征。

### 多模态 RL 落地

GRPO 及序列级变体 GSPO 已用于多模态推理与 GUI 计算机使用 agent 训练；
OSWorld 公开案例 62.2%→72.6%。奖励来自可验证信号（界面状态变化、答案
匹配），与本讲"verifier 分量：OCR/grounding/answer/safety"的思路一致。

**对本讲学习者的启示**：行业演进没有推翻本讲的三层框架（token 化 → 连接
结构 → 对齐阶段），而是给它加了两端——前端是原生多模态预训练（GPT-4o
路线），后端是可验证奖励的多模态 RL（GRPO/GSPO 路线）。学习顺序不变：先
掌握 projector/Q-Former/cross-attention 的取舍与形状审计，再理解评测谱系
（MMMU/MMBench/MathVista/DocVQA/EgoSchema/VBench）各自测什么、防什么偏差，
最后把空间智能（VSI-Bench 48.8% 缺口）与 GUI agent（OSWorld 62.2%→72.6%）
当作对齐与评测框架的迁移练习。

## 大厂面试真题与答题框架

高频面试题（公开面经风格），答题框架均可口述 3–5 步。

**题目 1：视觉编码器 → LLM 的对齐方案怎么选：投影层 vs cross-attention vs 早融合？**
- 考点：连接结构谱系、成本-灵活性取舍。
- 答题框架：
  1. 线性/MLP projector（LLaVA）：参数最少，冻结两塔的表示对齐阶段够用；
  2. resampler/Q-Former：固定数量 learned queries 压缩可变视觉 token，
     控上下文预算，GQA 思想的同构；
  3. gated cross-attention（Flamingo）：text 按需读视觉 memory，支持交错
     多图少样本，但结构侵入、部署复杂；
  4. 早融合/原生多模态：从头联合训练（GPT-4o 路线），上限高、数据与算力
     门槛高；
  5. 收尾：按训练阶段与预算选型，复杂度不是目的。
- 加分项：形状审计（placeholder 数 = 视觉 embeddings 数）；gate 近零初始化
  防破坏语言能力，但需 image ablation 查"模型忽略图像"。
- 踩坑：把"加了 projector"说成"已完成对齐"；忽视 (H/P)(W/P) 二次增长的
  token 成本。

**题目 2：多模态指令数据怎么构造？如何防止图文错配？**
- 考点：指令对齐数据契约、数据治理。
- 答题框架：
  1. 来源：GPT-4 生成的指令跟随数据（LLaVA 路线）+ OCR/grounding/图表/
     文档/多轮对话混合，控制 caption 与开放问答比例；
  2. 配对审计：图像与文本 augmentation 不得独立 shuffle；每对带 hash/
     provenance/license；
  3. 过滤：剔除"纯文本也能答对"与"所有 VLM 都答错"的题（MMBench 同款
     思路）；
  4. 多图引用：`<image_1>/<image_2>` 与 embeddings 位置一一对应；
  5. response-only 计损，prompt 与视觉 token 屏蔽。
- 加分项：提 benchmark 图片混入预训练 crawl 的 contamination，拆分按近重复
  簇完成。
- 踩坑：caption 占比过高导致模型只学会描述图像；空图/坏文件/极端宽高比
  无测试。

**题目 3：循环评估（circular eval）为什么能防位置偏差？**
- 考点：评测方法论、选择题评分。
- 答题框架：
  1. 现象：MLLM 对选择题选项位置敏感，固定位置命中率虚高；
  2. 方案：把正确答案轮换到每个选项位置，生成多版本题；
  3. 判定：所有位置都答对才计对，任一版答错即整题失败；
  4. 效果：消除位置先验，得分更接近真实能力（MMBench 3,217 题采用）；
  5. 延伸：开放式 judge 同理需随机化候选顺序、隐藏模型身份。
- 加分项：指出这是 counterfactual 思想在选项维度的应用。
- 踩坑：只报单次准确率不报位置一致性；用 text-only judge 校验。

**题目 4：图像 token 数与上下文预算怎么权衡？**
- 考点：tokenization 成本模型。
- 答题框架：
  1. 公式：N_img=(H/P)(W/P)，随分辨率二次增长；
  2. 全拼接 attention 为 O((N_text+N_img)^2)，与长上下文竞争；
  3. 压缩手段：tiles+缩略图、resampler 固定 queries、加粗 patch；
  4. 任务分层：OCR/grounding 要细粒度，闲聊对话可粗；动态分辨率按需分配；
  5. 报告：视觉 token 数、首 token 延迟、峰值显存一并给出。
- 加分项：保 tile 顺序与几何，否则空间关系/OCR 退化。
- 踩坑：训练 resize/crop 与评估不一致，制造虚假差距。

**题目 5：如何证明多模态模型真的"看图"了，而不是靠语言先验答题？**
- 考点：counterfactual 诊断协议。
- 答题框架：
  1. counterfactual image pairs：换图不换问，答案应随证据变化；
  2. 遮挡关键区域：性能应显著下降；
  3. image ablation：清零/打乱视觉输入，输出应退化；
  4. 对照 benchmark 伪相关：VQA 准确率混合感知/知识/推理，需分轴报告；
  5. 结论：报 counterfactual 一致率，而非单一平均分。
- 加分项：联系 VSI-Bench——最好的 Gemini-1.5 Pro 仅 48.8%，说明空间题
  靠先验背不住。
- 踩坑：用开放式"回答合理"当感知证据；judge 系统性偏好长而自信的描述。

**题目 6：多模态偏好对齐（DPO/PPO）的数据契约与纯文本有何不同？**
- 考点：偏好与安全对齐、数据契约。
- 答题框架：
  1. chosen/rejected 必须共用同一图像与 prompt；
  2. annotator 必须真正看到所有模态，纯文本 judge 无法判断图像事实；
  3. 覆盖敏感图像/身份推断/医疗建议/版权/深伪维度；
  4. 图中 prompt injection：OCR 内容按不可信数据处理；
  5. 分轴报告幻觉与安全，不做单一平均。
- 加分项：提 GRPO/GSPO 用可验证奖励（答案匹配/界面状态）替代部分偏好
  数据，OSWorld 公开案例 62.2%→72.6%。
- 踩坑：偏好标注只看文本不看图；视觉 encoder 预训练数据与 benchmark 重叠
  未审计。

## 系统设计题

**设计题 1：设计一个中文文档理解 MLLM（扫描合同/发票/表格，OCR-free）**
- 需求澄清：文档类型与版面复杂度；端到端 OCR-free 还是保留 OCR pipeline
  兜底；延迟与分辨率上限；是否需要 grounding（定位到页/区域）。
- 规模估算：参照 DocVQA 口径 50k 众包 QA 起步，中文需自建同量级数据
  （10^4–10^5，量级）；高分辨率扫描件 patch 数随分辨率二次增长，须
  tiles+缩略图控制视觉 token 预算。
- 架构：动态分辨率 ViT（保序 tiles+二维位置编码）→ MLP projector →
  中文 LLM；三阶段：表示对齐（image-caption + OCR 监督）→ 指令对齐
  （文档 QA/表格/多页多轮）→ 偏好与安全（图内注入红队）。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 取舍 |
| --- | --- | --- | --- |
| 连接器 | linear projector | Q-Former 压缩 | A 简单保细节；B 控 token 但丢小字 |
| 分辨率 | 固定低清 | tiles+缩略图 | A 批处理简单；B 保 OCR 但实现复杂 |
| 计损 | 全序列 | response-only | A 简单；B 防模型学会复述 prompt |

- 评测方案：DocVQA + 自建中文集（counterfactual 换页/改数字）；OCR exact
  match 说明大小写/Unicode/版面 normalization；image ablation 证明非语言
  先验；报告视觉 token 数与延迟。
- 追问预案：小字丢失→查 patch 粒度与 resize；多页引用错位→placeholder
  审计；答案不随图变→counterfactual 一致率定位语言先验。

**设计题 2：为具身智能场景设计空间智能评测与训练方案**
- 需求澄清：目标是评测还是训练改进（或两者）；空间能力粒度（物体位置/
  度量估计/路径规划）；是否绑定具体机器人平台。
- 规模估算：VSI-Bench 显示 SOTA 中最好的 Gemini-1.5 Pro 仅 48.8%，缺口
  即价值；自建评测集量级数千题（量级，视任务粒度）。
- 架构：评测端用视频/多视角输入 + counterfactual 空间问题（移动物体后
  重问）；训练端在数据中显式加入空间知识（相对位置/度量/导航轨迹监督），
  再接 projector → LLM 主干。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 取舍 |
| --- | --- | --- | --- |
| 输入 | 单帧 | 视频/多视角 | A 便宜；B 保时序但 token 爆炸 |
| 空间监督 | 无（靠涌现） | 显式空间知识数据 | A 零成本；B 有 VSI-Bench 结论支持 |
| 评测 | 开放式描述 | counterfactual 问答 | A 难评； B 可规则化评分 |

- 评测方案：分轴报告（定位/计数/度量/导航）；遮挡与换景 ablation；与
  VSI-Bench 数字直接对齐比较。
- 追问预案：模型背题→换场景泛化测试；视频 token 超预算→下采样/分块 +
  保序审计；如何支撑空间生成→离散 VQ token 统一建模路线。

**设计题 3：用多模态 RL 训练一个 GUI 计算机使用 agent**
- 需求澄清：操作系统/浏览器范围；动作空间（点击坐标/键盘/滚动）；成功
  率与安全约束；是否允许在线探索。
- 规模估算：参照 OSWorld 公开案例：GRPO/GSPO 训练后 62.2%→72.6%；
  rollout 为截图序列 + 动作，视觉 token 预算决定屏幕分辨率上限。
- 架构：截图 → 视觉 encoder → projector → LLM（动作自回归生成）；奖励 =
  可验证界面状态变化（目标达成比对）；GRPO 组内相对优势，长轨迹换序列级
  GSPO。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 取舍 |
| --- | --- | --- | --- |
| 奖励 | 稀疏终态 | 过程奖励 | A 简单可信；B 信号密但易 hack |
| 算法 | GRPO | GSPO（序列级） | A 通用；B 长轨迹更稳（公开案例均用） |
| 视觉输入 | 全分辨率 | 降采样截图 | A 保小字；B 控 token 与延迟 |

- 评测方案：OSWorld 成功率 + 分解到步（定位/执行/恢复）；counterfactual
  任务（换 UI 主题答案应不变）；页面内 prompt injection 红队。
- 追问预案：奖励 hack（假成功）→界面状态哈希校验；长 horizon 信用分配→
  GSPO/过程奖励；屏幕 OCR 注入→信任边界分级。

## 代码实现题

**实现题 1：视觉 patch → projector → LLM embedding 的前向骨架**

考察点：形状一致性、placeholder 替换、position ids 同步。

```python
import torch
import torch.nn as nn

class MultimodalProjector(nn.Module):
    def __init__(self, d_v: int, d_llm: int):
        super().__init__()
        self.net = nn.Linear(d_v, d_llm)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # visual_tokens: [B, Nv, Dv] -> [B, Nv, Dllm]
        return self.net(visual_tokens)

def merge_embeddings(
    input_ids: torch.Tensor,          # [B, Nt]，含 <image> 占位符
    text_embeds: torch.Tensor,        # [B, Nt, Dllm]
    visual_tokens: torch.Tensor,      # [B, Nv, Dv]
    image_token_id: int,
    projector: MultimodalProjector,
):
    image_embeds = projector(visual_tokens)            # [B, Nv, Dllm]
    B, Nt, _ = text_embeds.shape
    merged = text_embeds.clone()
    for b in range(B):
        img = image_embeds[b]
        # 每个 <image> 占位符替换为一个视觉 embedding，保序且一一对应
        pos = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
        assert pos.numel() == img.shape[0], (
            f"placeholder 数 {pos.numel()} != 视觉 token 数 {img.shape[0]}"
        )
        merged[b, pos] = img
    # position ids 按拼接后真实序列长度重算，防止插入视觉 token 后错位
    position_ids = torch.arange(Nt, device=text_embeds.device).expand(B, Nt)
    return merged, position_ids
```

验收标准：placeholder 数与视觉 token 数不一致时 assert 抛错；多图（多个
`<image>` 段）保序替换；零图像/单图/高分辨率 tiles 三个用例形状通过。

**实现题 2：多模态 loss mask——图文交错序列只对文本 response 计损**

考察点：response-only loss、视觉与 prompt 屏蔽。

```python
import torch

def build_multimodal_labels(
    input_ids: torch.Tensor,    # [B, N]，含 <image> 占位与特殊 token
    prompt_lens: torch.Tensor,  # [B]，prompt（含全部视觉 token）长度
    pad_token_id: int,
    vision_token_id: int = None,
    ignore_index: int = -100,
):
    B, N = input_ids.shape
    labels = input_ids.clone()
    for b in range(B):
        # 1) 视觉占位 token 不计损
        if vision_token_id is not None:
            labels[b, input_ids[b] == vision_token_id] = ignore_index
        # 2) prompt 段（含图像占位）不计损，只保留 response
        labels[b, : int(prompt_lens[b])] = ignore_index
        # 3) padding 段不计损
        labels[b, input_ids[b] == pad_token_id] = ignore_index
    return labels

def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                         ignore_index: int = -100):
    # logits: [B, N, V]; labels: [B, N]
    flat_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    mask = labels.ne(ignore_index).reshape(-1)
    return (flat_loss * mask).sum() / mask.sum().clamp(min=1), mask.sum()
```

验收标准：纯 prompt 序列有效 token 数为 0、loss 为 0；视觉 token 位置全为
-100；多图交错样本与单图样本的有效 token 数符合手算。

**实现题 3：MMMU 式选择题自动评分脚本（规则化评分 + 循环评估）**

考察点：规则化评分（单词/短语匹配）、位置偏差消除。

```python
import re

def normalize(text: str) -> str:
    # 大小写、多余空白、标点统一；MMMU 规则化评分为单词/短语匹配
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()

def extract_choice(response: str, options: dict):
    # 优先匹配选项字母；其次短语匹配选项文本
    resp = normalize(response)
    for letter in options:
        if re.search(rf"\b{letter.lower()}\b", resp):
            return letter
    for letter, text in options.items():
        if normalize(text) and normalize(text) in resp:
            return letter
    return None

def circular_score(responses: list, options: list, gold: str):
    # 循环评估：正确答案轮换到每个选项位置，全部答对才计对
    if len(responses) != len(options):
        raise ValueError("每个轮换版本需要一条 response")
    for resp, opt in zip(responses, options):
        pred = extract_choice(resp, opt)
        if pred is None or normalize(opt[pred]) != normalize(opt[gold]):
            return 0.0
    return 1.0

def circular_accuracy(dataset: list) -> float:
    # dataset 每项含 n 个轮换版 responses + options + gold
    scores = [circular_score(it["responses"], it["options"], it["gold"])
              for it in dataset]
    return sum(scores) / len(scores)
```

验收标准：字母与选项文本两种表达都能匹配；未匹配返回 None 不误判；轮换
版本中任一版答错即整题记 0。

## 小结

多模态模型的第一性问题是如何把高带宽感知输入压缩成可用 token，并在计算预算内连接
语言模型。projector 简洁，cross-attention 灵活，统一离散 tokens 便于生成；它们
对应不同的结构和训练代价。表示对齐、指令对齐与安全对齐是不同阶段，最终评估必须
证明模型看到了正确证据，而不只是输出了语言上合理的答案。

## 参考文献

<a id="ref-1"></a>[1] A. Dosovitskiy, L. Beyer, A. Kolesnikov, et al. “An Image
is Worth 16x16 Words: Transformers for Image Recognition at Scale.” *ICLR*,
2021. [link](https://arxiv.org/abs/2010.11929)

<a id="ref-2"></a>[2] A. Radford, J. W. Kim, C. Hallacy, et al. “Learning
Transferable Visual Models From Natural Language Supervision.” *ICML*, 2021.
[link](https://arxiv.org/abs/2103.00020)

<a id="ref-3"></a>[3] J.-B. Alayrac, J. Donahue, P. Luc, et al. “Flamingo:
a Visual Language Model for Few-Shot Learning.” *NeurIPS*, 2022.
[link](https://arxiv.org/abs/2204.14198)

<a id="ref-4"></a>[4] J. Li, D. Li, S. Savarese, et al. “BLIP-2: Bootstrapping
Language-Image Pre-training with Frozen Image Encoders and Large Language
Models.” *ICML*, 2023. [link](https://arxiv.org/abs/2301.12597)

<a id="ref-5"></a>[5] H. Liu, C. Li, Q. Wu, et al. “Visual Instruction
Tuning.” *NeurIPS*, 2023. [link](https://arxiv.org/abs/2304.08485)

<a id="ref-6"></a>[6] H. Laurençon, L. Saulnier, L. Tronchon, et al. “OBELICS:
An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents.” *EMNLP*,
2023. [link](https://arxiv.org/abs/2306.16527)

<a id="ref-7"></a>[7] A. van den Oord, O. Vinyals, K. Kavukcuoglu. “Neural
Discrete Representation Learning.” *NeurIPS*, 2017.
[link](https://arxiv.org/abs/1711.00937)

<a id="ref-8"></a>[8] P. Esser, R. Rombach, B. Ommer. “Taming Transformers for
High-Resolution Image Synthesis.” *CVPR*, 2021.
[link](https://arxiv.org/abs/2012.09841)

## 延伸阅读与复现材料

- Stanford CS336, [Lecture 17 — Alignment and multimodality](https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py)
- [Alignment 主题导航](../experiments/topics/alignment.md)
- 本仓库 A4/A5 实现与报告（用于数据治理和训练契约映射）
- [Stanford CS336 课程主页](https://cs336.stanford.edu)（访问日期 2026-10-04）
- [CS336 lectures 仓库](https://github.com/stanford-cs336/lectures)（访问日期 2026-10-04）
