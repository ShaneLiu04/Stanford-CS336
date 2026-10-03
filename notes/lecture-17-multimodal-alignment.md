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
- 文档性质：原创中文自学综述，非课程提交
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

## 面试备考（Interview Prep）

> 多模态 LLM 是面试的进阶架构题：面试官常从「三种连接结构怎么选」切入，追到
> 「视觉 token 数随分辨率二次增长」「CLIP 的 InfoNCE」「多模态幻觉怎么操作化诊断」
> 「图内 prompt injection 怎么防」。核心是理解「token 化 → 连接结构 → 对齐阶段 → 评估证据」
> 四层，并牢记评估必须证明模型「看见了证据」而非「输出了合理答案」。下面按
> 「一页速览 → 高频题 → 手撕 → 追问」四层组织。

### 一页速览卡（面试前 1 分钟）

**核心主张**：多模态 LLM 的第一性问题是把高带宽感知输入压缩成有限 token、在预算内接入 LLM；
评估的核心论点是「回答合理」无法区分真实感知与语言先验，必须用 counterfactual 对证明证据读取。

**必背数字与公式**

- 图像 patch 数 \(N_{\text{img}}=(H/P)(W/P)\)，随分辨率**二次**增长。
- 全序列 self-attention \(O((N_{\text{text}}+N_{\text{img}})^2)\)；cross-attention 约 \(O(N_t N_v)\)。
- CLIP InfoNCE \(-\frac1B\sum_i\log\frac{\exp(s(v_i,t_i)/\tau)}{\sum_j\exp(s(v_i,t_j)/\tau)}\)。
- VQ 量化 \(k^\star=\arg\min_k\|z_e(x)-e_k\|_2^2\)。

**三句话答高频**

1. linear projector 最简（LLaVA）；cross-attention 按需读取（Flamingo）；VQ 统一 token 支持生成。
2. 视觉 token 随分辨率二次增长，高分辨率靠 tiles + 缩略图，但必须保序保几何。
3. 幻觉诊断用 counterfactual pairs + 遮挡 ablation；「回答合理」≠「看见证据」。

### 高频面试题与答题框架

**Q1：三种连接结构怎么选？**

- **linear/MLP projector**：参数少、训练快，冻结 encoder/LLM 的第一阶段表示对齐；LLaVA 证明极简结构 + 高质量指令数据即可（GPT-4 生成指令数据低成本撬动）。
- **Q-Former / resampler**：用固定数量 learned queries 压缩可变视觉 token，控制 LLM 上下文成本；**cross-attention（Flamingo）**：在若干 LLM 层插入，让 text query 按需读视觉 memory，支持交错图文少样本，但结构侵入、预训练/部署复杂。
- **VQ 离散 token**：与语言 token 统一自回归建模、便于生成，代价是量化误差 + 大 codebook 训练。

**Q2：视觉 token 数与成本？**

- \(N_{\text{img}}=(H/P)(W/P)\)，提高分辨率二次增加 token 数；全拼接 attention \(O((N_t+N_v)^2)\)。
- 动态分辨率把大图切 tiles + 加 thumbnail/二维位置编码，但必须保留 tile 顺序与几何，否则 OCR/图表/空间关系退化。
- 取舍：token 少便宜但丢小字/计数/定位，token 多保细节但显存/延迟/长上下文竞争加剧。

**Q3：ViT 的 patch tokenization？**

- 把图像切成不重叠 \(P\times P\) patch，每个 patch 展平 + 线性映射为 \(d_v\) 维，再经 ViT 得视觉 tokens \(Z_v\in\mathbb R^{N_{\text{img}}\times d_v}\)。
- 确立「图像即 token 序列」范式，是多模态接入 LLM 的基础。

**Q4：CLIP 的目标（InfoNCE）？**

- batch 内对比学习：让匹配 image-text 对的相似度高于所有负对，学到可迁移的对齐表示空间。
- 用 4 亿 image-caption 对，自然语言监督即可产生强视觉表示；InfoNCE 成为多模态表示对齐的标准构件。

**Q5：多模态对齐的三阶段？**

- **表示对齐**：用 image-caption 对让 projector/encoder 输出落入 LLM 可用表示空间（caption loss 或 InfoNCE）。
- **多模态指令微调**：图像问答/OCR/grounding/图表/文档/多轮，response-only loss；数据混合要控制 caption/OCR/开放问答比例，避免只学会描述图像。
- **偏好与安全对齐**：RM/DPO/PPO，但偏好标注必须让 annotator 真正看到所有模态（纯文本 judge 无法判图像事实）。

**Q6：Flamingo 的 gated cross-attention？为什么 gate 初始化接近零？**

- 在若干 LLM 层插入 cross-attention，让 text queries 按需读视觉 memory；成本 \(O(N_t N_v)\)，视觉 memory 可复用。
- gated 以接近零的 gate 初始化，减少接入新模态时破坏原语言能力；但 gate 过小会「忽略图像」，需 image ablation 检查。

**Q7：Q-Former 的作用？**

- 以少量 learned queries 从冻结视觉 encoder 提取与语言最相关的表示，把可变视觉 token 压缩成固定长度，控制 LLM 上下文成本。
- 属于「resampler」一类：用固定数量查询压缩，代价是压缩可能丢失细粒度信息。

**Q8：多模态幻觉如何操作化诊断？**

- 不能只用开放式「回答合理」；要 counterfactual image pairs（换图不改问）、遮挡关键区域、交换对象属性、image-shuffle 一致性。
- 验证输出随证据变化：换图答案不变说明语言先验主导、没真正读图。

**Q9：VQ-VAE / VQGAN 的离散 token？**

- VQ-VAE 用 codebook 量化把连续观测转离散隐变量；VQGAN 加感知损失 + 对抗训练，使高分辨率保真。
- 离散 tokens 可与语言 token 统一自回归建模、便于生成；代价是量化误差与大 codebook 训练。

**Q10：图内 prompt injection 怎么防？**

- 图中 OCR 文本是不可信数据，不能被当作高优先级指令；要显式建模「系统指令 / 用户输入 / 图像文字」三级信任边界。
- 红队测试：图像文字注入、敏感属性推断、身份推断、医疗/定位、深伪。

### 手撕要点（视觉 token 数与形状审计）

面试让「算视觉 token 数」或「审计形状」时，按公式写：

```text
图像 patch 数: N_img = (H/P) * (W/P)     （随分辨率二次增长）
全序列 attention: O((N_text + N_img)^2)
cross-attention:   O(N_t * N_v)

最小接口:
  vision_encoder(pixel_values[B,C,H,W]) -> visual_tokens[B,Nv,Dv]
  projector(visual_tokens) -> multimodal_tokens[B,Nm,Dllm]
  merge(input_ids, multimodal_tokens, image_positions)
    -> inputs_embeds[B,Nt+Nm,Dllm], attention_mask, labels
```

**三个必踩坑**

1. **`<image>` 占位数与视觉 embeddings 数必须一致**：不一致会静默错位。
2. **projector 输出 token 数变化后要同步 attention mask 与 position ids**。
3. **训练/评估的 resize/crop/归一化必须一致**：不一致会破坏位置/文字、制造虚假差距。

### 高频追问与陷阱

| 追问 | 正确方向 |
| --- | --- |
| 换图答案不变说明什么？ | 语言先验主导、gate 接近零或没真正读图 |
| 用 text-only judge 评视觉事实可以吗？ | 否，会得虚假高分，须多模态 judge + 人工校准 |
| benchmark 图片会污染吗？ | 会，视觉 encoder 预训练数据可能与 benchmark 重叠 |
| 高分辨率一定好吗？ | 否，token 二次增长，需 tiles + 缩略图权衡 |
| 「回答合理」够吗？ | 否，须 counterfactual 对证明证据读取 |

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
