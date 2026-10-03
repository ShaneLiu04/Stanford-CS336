---
title: "Lecture 03 — Architectures & Hyperparameters"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-04-06"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf"
  - "../assignments/spring2026/assignment1-basics/"
---

# Lecture 03 — Architectures 与 Hyperparameters：从建模假设到可训练系统

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述，非课程提交
- 适用对象：自学者、模型架构工程师与 LLM 研究者

## 摘要

Transformer architecture 不是若干模块的机械叠加，而是一组相互耦合的统计与系统设计：
normalization 决定深层 residual stream 的优化稳定性，position mechanism 决定长度归纳偏置，
attention/FFN 比例决定参数与计算分配，初始化、learning rate、batch size 和 precision
共同决定可训练区域。本文以 decoder-only language model 为中心，系统梳理 embedding、
residual path、RMSNorm、causal multi-head attention、RoPE 与 SwiGLU 的数学结构，
并推导参数量、FLOPs、activation 与超参数耦合关系。在工程层面，本文给出 shape contract、
初始化统计、数值稳定、checkpoint compatibility、单元测试与 profiler 方法；在研究层面，
讨论 depth/width、head dimension、context/vocabulary、normalization 与 position encoding
的 controlled ablation、scaling transfer 和效度威胁。目标是使读者能从建模假设解释架构，
从资源约束选择配置，并设计可复核而非只比较最终 loss 的实验。

**关键词：** Decoder-only Transformer；RMSNorm；Pre-Norm；RoPE；SwiGLU；
Initialization；Hyperparameter Scaling；Residual Stream；Ablation Study

## 本文贡献

1. 建立“architecture—optimization—systems”三层统一视角；
2. 推导 attention、FFN、embedding/head 的参数与计算分配；
3. 解释 normalization、residual scaling、初始化和 depth 的稳定性耦合；
4. 给出 RoPE 相对位置性质、SwiGLU 参数匹配和 context/vocabulary trade-off；
5. 提供面向工程验证与研究消融的 checklist、故障排查、效度分析与文献路线。

> 现代 decoder-only Transformer 看似有大量变体，但已有一组稳健起点：pre-norm、RMSNorm、
> RoPE、SwiGLU、bias-free linear layers。它们是经验默认值，不是数学定律；公平比较必须控制参数量、
> token budget、初始化和系统效率。

## 学习目标

1. 从 next-token objective 推导 decoder-only Transformer 的数据流；
2. 解释 residual、pre-norm、RMSNorm、RoPE、SwiGLU 的作用和公式；
3. 追踪完整模型中每个 tensor 的 shape；
4. 推导模型参数量与主要 FLOPs，理解 depth/width/FFN/head/vocab 的耦合；
5. 区分架构超参数、优化超参数和系统超参数；
6. 设计参数匹配、token 对齐、可复现的 architecture ablation。

## 先修知识

- Lecture 1：token IDs、\(V\)、next-token loss；
- Lecture 2：Linear、einsum、FLOPs、memory；
- softmax、矩阵乘法、基本梯度下降。

## 核心概念与公式推导路线

本讲从 decoder-only 数据流出发，依次推导 RMSNorm、scaled dot-product attention、RoPE 的相对位置
性质、SwiGLU 的参数匹配，以及整模参数/FLOPs。核心判断框架是：任何 architecture choice 都要同时
回答表示能力、优化稳定性、shape/复杂度与 hardware efficiency 四个问题。

## 相关工作与架构演进

Transformer 用 self-attention 取代 recurrence，使 token 间依赖可并行计算
[[1]](#ref-1)。GPT 系列进一步证明 decoder-only next-token pretraining 能统一多任务生成：
GPT-2 强调 zero-shot transfer [[2]](#ref-2)，GPT-3 展示 in-context learning 与规模效应
[[3]](#ref-3)。现代开源模型（如 LLaMA）把 pre-norm、RMSNorm、RoPE 与 SwiGLU
组合为稳定且高效的 decoder backbone [[4]](#ref-4)。

架构细节具有明确优化后果。RMSNorm 去掉均值中心化以降低成本
[[5]](#ref-5)；理论分析表明 pre-norm 能改善初始化时的梯度尺度，使深层 Transformer
更易训练 [[6]](#ref-6)。RoPE 用复数/二维旋转编码相对位置
[[7]](#ref-7)，ALiBi 则直接向 attention logits 加距离 bias，体现不同的长度归纳偏置
[[8]](#ref-8)。Gated linear units 的系统比较发现 SwiGLU 等变体在相近计算预算下优于
传统 ReLU/GELU FFN [[9]](#ref-9)。

超参数也不能脱离规模讨论。Chinchilla 说明 model size 与 training tokens 需联合分配
[[10]](#ref-10)；\(\mu\)P 尝试让 learning rate、初始化与其他超参跨 width transfer
[[11]](#ref-11)。这些工作共同说明：架构创新若没有 parameter/compute matching、
稳定训练与 controlled comparison，单点 loss 改善很难归因。

## 1. Decoder-only Transformer 全景

输入 token IDs \(X\in\{0,\ldots,V-1\}^{B\times T}\)，模型输出
\(Z\in\mathbb R^{B\times T\times V}\)。第 \(t\) 个位置的 logits 只允许依赖
\(x_{\le t}\)，用于预测 \(x_{t+1}\)。

```text
token IDs [B,T]
  -> token embedding [B,T,d]
  -> L × {pre-norm causal attention + residual
          pre-norm SwiGLU + residual} [B,T,d]
  -> final RMSNorm [B,T,d]
  -> LM head [B,T,V]
```

训练目标通常把连续 token array 采样成：

\[
x=(u_s,\ldots,u_{s+T-1}),\qquad
y=(u_{s+1},\ldots,u_{s+T}).
\]

如果忘记 shift，模型会学习复制当前 token，loss 看似快速下降但任务已错误。

## 2. Embedding、Linear 与初始化

Embedding 矩阵 \(E\in\mathbb R^{V\times d}\)，查表：

\[
H_{bt:}=E_{X_{bt}:},\qquad H\in\mathbb R^{B\times T\times d}.
\]

它不应实现成 one-hot \([B,T,V]\) 乘矩阵，因为会创建巨大稀疏中间量。LM head
\(W_U\in\mathbb R^{V\times d}\)：

\[
Z_{btv}=\sum_j H_{btj}(W_U)_{vj}.
\]

Input/output weight tying 令 \(W_U=E\)，减少 \(Vd\) 参数，但初始化必须兼顾“被查表的表示”和
“输出分类器”两种角色。本仓库实验中直接共享原 std=1 embedding 会退化，改用 std=0.02 后恢复并
提高参数效率，说明 sharing 本身不是无条件收益。

对于 linear，仓库使用 fan-in/fan-out truncated normal：

\[
\sigma=\sqrt{\frac{2}{d_{\rm in}+d_{\rm out}}}.
\]

初始化的目标是控制 signal / gradient scale；不同 residual depth、norm placement 与 tying
会改变合适的尺度。

### 2.1 方差传播与 residual scaling

若 \(x_i\) 独立、零均值、方差 \(q\)，线性层 \(y_j=\sum_i W_{ji}x_i\)，则近似

\[
\operatorname{Var}(y_j)
=d_{\text{in}}\operatorname{Var}(W_{ji})q.
\]

令 \(\operatorname{Var}(W)\propto1/d_{\text{in}}\) 可保持单层方差量级；但 residual stack
\(x_{l+1}=x_l+f_l(x_l)\) 会累积支路方差。若各层独立同尺度，未控制时 residual variance
可随 depth 增长。常见工程策略包括：

- 对 residual output projection 乘 \(1/\sqrt{L}\) 或相关 depth scaling；
- pre-norm 保持进入子层的尺度稳定；
- 使用 DeepNorm/\(\mu\)P 等有理论约束的 parameterization；
- 在初始化后实测每层 residual RMS、activation/gradient norm。

“使用标准正态初始化”不是完整描述；必须记录 distribution、truncation、std、哪些矩阵使用
特殊 scaling，以及 weight tying 后是否重新初始化。

### 2.2 Weight tying 的统计与接口约束

共享 \(E=W_U\) 节省 \(Vd\) 参数，并让输入/输出 token geometry 关联；但同一矩阵同时接收
sparse embedding gradients 与 dense LM-head gradients。需检查：

- state dict 只保存一个 Parameter 还是重复 alias；
- optimizer 是否重复更新 tied parameter；
- initialization 是否适合 output logits scale；
- checkpoint load 后 alias 是否仍保持；
- vocabulary resize 是否同步修改两端；
- distributed wrapping 是否把 alias 错误分片两次。

Tying 的公平比较应匹配总参数或至少同时报告 parameter efficiency；否则 untied 模型容量更大，
直接比较 loss 混合了 sharing inductive bias 与 parameter count。

## 3. Normalization 与 Residual Path

### 3.1 LayerNorm 与 RMSNorm

LayerNorm 对每个样本的 feature 维做均值/方差归一化并配以可学习平移
[[12]](#ref-12)：

\[
\operatorname{LN}(x)=
\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
\]

RMSNorm 不减均值：

\[
\operatorname{RMS}(x)=\sqrt{\frac1d\sum_{i=1}^d x_i^2+\epsilon},\qquad
\operatorname{RMSNorm}(x)=g\odot\frac{x}{\operatorname{RMS}(x)}.
\]

输入/输出 shape 都是 `[..., d]`，reduction 在最后一维。RMSNorm 参数仅 \(d\) 个 gain，
计算略简化；平方 reduction 应在 fp32 中进行以减小低精度风险。

### 3.2 Pre-norm 与 post-norm

Pre-norm block：

\[
y=x+\operatorname{Attn}(\operatorname{Norm}(x)),\qquad
z=y+\operatorname{FFN}(\operatorname{Norm}(y)).
\]

Post-norm block：

\[
y=\operatorname{Norm}(x+\operatorname{Attn}(x)),\qquad
z=\operatorname{Norm}(y+\operatorname{FFN}(y)).
\]

Pre-norm 提供跨层近似 identity 的 residual gradient path，通常更易训练深网络；post-norm
可能有不同表示性质，但对深度与初始化更敏感。仓库 full-budget 实验中 post-norm best loss 1.385，
略差于 pre-norm baseline；这是特定规模证据，不应上升为普遍定理。

移除 normalization 不只是“少一个小模块”。本仓库 40M-token baseline 约 1.637，而无
RMSNorm 为 9.256；降 LR 与延长预算仍未恢复，说明 residual stream scale 已失稳。

### 3.3 为什么 pre-norm 更容易优化

对 pre-norm residual block \(x_{l+1}=x_l+F_l(N(x_l))\)，Jacobian 为

\[
\frac{\partial x_{l+1}}{\partial x_l}
=I+\frac{\partial F_l}{\partial N}\frac{\partial N}{\partial x_l}.
\]

Identity 项为梯度提供不经过子层的直接路径。Post-norm 则把 normalization Jacobian 放在
residual addition 之后，连续多层相乘时更依赖初始化与 warm-up。该推导解释“更容易训练”
的机制，但不是 post-norm 必然更差的证明；DeepNorm 通过对 residual 分支与 norm gain
的联合缩放，把 post-norm Transformer 稳定扩展到千层量级 [[13]](#ref-13)，
说明稳定性来自 scale 的可控性，而非 norm 的位置本身。

Normalization 实现还受数值细节影响：

- \(\epsilon\) 太大改变低 RMS activation，太小可能在 fp16 underflow；
- reduction 应 fp32 accumulate；
- gain 初始值通常为 1；
- padding/masked positions 是否参与 norm 取决于数据布局；
- fused norm kernel 必须与 reference 的 dtype/cast 顺序一致。

## 4. Causal Multi-head Self-attention

令 \(d=hd_h\)。一次 projection 后：

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
\]

由 \([B,T,d]\) reshape 为 \([B,h,T,d_h]\)。每个 head：

\[
S_{bhij}=\frac{Q_{bhi:}K_{bhj:}^{\top}}{\sqrt{d_h}}+M_{ij},
\]

\[
A=\operatorname{softmax}(S,\text{dim}=j),\qquad
O=AV.
\]

Causal mask：

\[
M_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
\]

缩放来自方差：若 \(q_k,k_k\) 独立、零均值、方差约 1，则
\(\operatorname{Var}(q^\top k)\approx d_h\)；除以 \(\sqrt{d_h}\) 把 logit scale 拉回
常数量级，避免 softmax 过早饱和。

Shape 路径：

```text
X          [B,T,d]
Q,K,V      [B,h,T,d_h]
scores     [B,h,T,T]
mask       [T,T] 或可广播到 [B,h,T,T]
weights    [B,h,T,T]
head out   [B,h,T,d_h]
concat     [B,T,d]
output     [B,T,d]
```

Projection 为 \(O(BTd^2)\)，score/value 为 \(O(BT^2d)\)，显式 attention matrix 为
\(O(BhT^2)\) memory。

### 4.1 Mask、softmax 与数值稳定

实现顺序应为 `scores * scale → mask → softmax`。若用有限负数代替 \(-\infty\)，其幅度需结合
dtype：fp16 中过大负数可能 underflow/表示异常，过小又泄漏 future probability。
Stable softmax 对每行减最大值：

\[
\operatorname{softmax}(s)_j
=\frac{\exp(s_j-\max_k s_k)}{\sum_i\exp(s_i-\max_k s_k)}.
\]

必须沿 key 轴归一化。若一整行都被 mask，`-inf - (-inf)` 会产生 NaN；causal attention
通常保证至少当前位置有效，但 arbitrary masks 需显式处理。

### 4.2 Head 数不是独立自由度

在固定 \(d\) 下增加 heads 会减小 \(d_h=d/h\)，QKVO 参数量近似不变，但：

- attention matrix 数量随 \(h\) 增加；
- kernel shape/并行度变化；
- \(d_h\) 太小可能限制单 head 表示；
- tensor cores 偏好特定 alignment；
- RoPE frequency 维度与 KV cache layout 改变。

因此 head sweep 应固定 \(d\) 并报告 \(d_h\)、latency、memory，而不是只把 heads 当“更多容量”。
MQA 让所有 query heads 共享单一 K/V head [[14]](#ref-14)，GQA 将其推广为分组共享
（组内 heads 共享 K/V）[[15]](#ref-15)；二者以极小的质量代价大幅降低 decode KV
cache 与 bandwidth，属于 inference-aware architecture choice，将在 Lecture 4/10
进一步讨论。

## 5. Position：RoPE

没有位置机制时，self-attention 本身对 token 排列是 permutation equivariant；causal mask 提供
“过去/未来”顺序，却不充分表达距离。RoPE 对每对 Q/K 维度施加位置相关旋转。

对二维子向量和频率 \(\omega_i=\theta^{-2i/d_h}\)：

\[
R_m^{(i)}=
\begin{bmatrix}
\cos(m\omega_i)&-\sin(m\omega_i)\\
\sin(m\omega_i)& \cos(m\omega_i)
\end{bmatrix}.
\]

\[
\tilde q_m=R_mq_m,\qquad \tilde k_n=R_nk_n.
\]

因为旋转矩阵正交：

\[
\tilde q_m^\top\tilde k_n
=q_m^\top R_m^\top R_n k_n
=q_m^\top R_{n-m}k_n,
\]

点积自然依赖相对位移 \(n-m\)。RoPE 通常只施加到 Q/K，不施加 V；维度需为偶数，且 A1
实现使用每个 head 的 \(d_h\)，不是整个 \(d\)。

缓存 sin/cos 的 shape 在仓库中逻辑上为 `[2, context_length, d_h/2]`。超出训练长度的外推
并非天然可靠；改变 \(\theta\)、frequency scaling 或 context curriculum 都需实验验证。

仓库 NoPE full-budget loss 1.439，差于 RoPE baseline，说明 causal mask 不能替代显式距离信息。

### 5.1 Context extrapolation 不是免费性质

训练只观察 \(m<T_{\text{train}}\) 的 rotation phases。推理扩展到更长位置时，高频分量会更快
绕圈，模型还会遇到未训练的 phase combinations。常见扩展方法包括 position interpolation
（把位置坐标线性压缩回训练区间，配少量 continued training）[[16]](#ref-16)、
NTK-aware/YaRN 式 frequency scaling（按 NTK 视角调整 RoPE base 与频谱插值）
[[17]](#ref-17)，或 long-context continued pretraining；
它们改变 frequency spectrum，必须同时评估 short-context regression、long-context retrieval
与 perplexity。

研究时至少区分：

- **长度外推**：直接输入更长序列；
- **位置插值**：把长位置压回训练区间；
- **继续训练**：在新长度分布上更新参数；
- **任务外推**：needle/retrieval 成功不等于 general long-context reasoning。

## 6. FFN 与 SwiGLU

普通两层 FFN：

\[
\operatorname{FFN}(x)=W_2\phi(W_1x),
\]

约有 \(2df\) 参数。SwiGLU：

\[
\operatorname{SwiGLU}(x)
=W_2\left(\operatorname{SiLU}(W_1x)\odot W_3x\right),
\]

\[
\operatorname{SiLU}(u)=u\sigma(u).
\]

它有 \(3df\) 参数和两个 width-\(f\) 上投影。要与标准 FFN 参数匹配，应令：

\[
3df_{\rm GLU}\approx2d(4d)
\Rightarrow f_{\rm GLU}\approx\frac83d.
\]

因此“都设 \(f=4d\)”不是公平 activation ablation。工程上还常把 \(f\) round 到 128/256 的倍数，
以获得更合适的 GEMM shape。

本仓库 SwiGLU 与参数近似匹配的 SiLU FFN 在该小规模最终 loss 接近；结论是“本 setting 差异小”，
不是“gating 无效”。

## 7. 架构与超参数如何耦合

### 7.1 Depth 与 width

忽略 embedding，参数量近似：

\[
P\approx L(4d^2+3df).
\]

若 \(f=cd\)，则 \(P\approx L(4+3c)d^2\)。固定 \(P\) 时，加深意味着缩小 \(d\)。
质量之外还要考虑：

- 深度增加 sequential dependency，降低 layer-level 并行；
- 宽度影响 GEMM shape 与硬件利用率；
- residual stability 随深度更敏感；
- activation memory 近似随 \(LBTd\)；
- pipeline parallelism 更喜欢可切分的层数。

经验 aspect ratio（如 \(d/L\)）只能作为起点，不能脱离参数量和硬件照搬。

### 7.2 Heads 与 head dimension

通常 \(d=hd_h\)。固定 \(d\) 改变 \(h\) 时，dense QKVO 参数基本不变，但会改变：

- 每个 head 的表示维度；
- RoPE 频率维度；
- attention matrix 的 head 轴与 kernel layout；
- KV cache 的组织方式。

常见 \(d_h\) 约 64–128 是经验/硬件折中，不是表达能力定律。`d % h == 0` 是基本 shape 约束。

### 7.3 Vocabulary 与 context

- \(V\) 增大：token 序列可能缩短，但 embedding/head 参数和 logits 计算增大；
- \(T\) 增大：可利用更长历史，但 standard attention 为 \(T^2\)；
- Tokenizer、\(V\)、\(T\) 和训练 token budget 必须联合定义。

### 7.4 优化与系统超参数

不要把所有旋钮都称“模型架构”：

- 架构：\(L,d,h,f\)、norm、position、FFN、tying；
- 优化：LR、betas、weight decay、warmup、clip、global batch；
- 系统：physical batch、accumulation、dtype、compile、parallelism；
- 数据：tokenizer、mixture、context packing。

同一 LR 对不同 depth/width/batch 未必公平；同一 FLOPs 在不同 kernel shape 下 wall-clock 也不同。

### 7.5 Hyperparameter transfer 与搜索策略

改变 width/depth 后，直接复用 learning rate、初始化 std 和 warm-up 可能使“架构比较”
实际变成“某架构的超参更合适”。三种常见策略：

1. **每配置独立调参**：公平但成本高，适合最终结论；
2. **固定 recipe**：测 robustness/工程便利，不能声称 intrinsic 最优；
3. **scaling parameterization（如 \(\mu\)P）**：用理论缩放规则转移超参，但需严格遵守
   parameter classes 和 width change 定义。

推荐两阶段设计：

- 小 budget screening：宽范围 LR、少量 seeds，排除明显不稳定配置；
- full budget confirmation：候选架构各自使用预注册 tuning budget，多 seeds、固定 tokens/FLOPs。

选择 best validation loss 会引入 winner's curse；应保留所有 sweep runs，并在 held-out seed
或独立 validation 上确认。若 tuning budget 不相等，必须在报告中披露。

### 7.6 Architecture ablation 的最低规范

一项可归因消融至少固定：

- tokenizer/vocab、train/validation split、data order；
- parameter budget 或 training FLOPs（说明选择哪一个）；
- optimizer family、batch、schedule 和 precision；
- evaluation interval 与 checkpoint selection；
- seeds 和 stopping rule。

同时记录 parameters、tokens/s、peak memory、wall-clock 与 time-to-loss。某架构最终 loss
略优但慢 2×，对 compute-constrained setting 未必更好。

## 8. 参数量与复杂度汇总

| 组件 | 参数量 | Forward FLOPs（batch \(B\)） | Activation shape |
|---|---:|---:|---|
| Embedding | \(Vd\) | lookup | `[B,T,d]` |
| QKVO | \(4d^2\) / layer | \(8BTd^2\) | QKV `[B,h,T,d_h]` |
| Attention mixing | 0 | \(4BT^2d\) | scores `[B,h,T,T]` |
| SwiGLU | \(3df\) / layer | \(6BTdf\) | branches `[B,T,f]` |
| 2×RMSNorm | \(2d\) / layer | \(O(BTd)\) | same as input |
| Final RMSNorm | \(d\) | \(O(BTd)\) | `[B,T,d]` |
| LM head | \(Vd\)（untied） | \(2BTdV\) | logits `[B,T,V]` |

完整模型近似：

\[
P=2Vd+L(4d^2+3df+2d)+d
\]

（untied、无 bias），forward FLOPs：

\[
F=B\left[L(8Td^2+4T^2d+6Tdf)+2TdV\right].
\]

公式忽略 softmax、RoPE、norm 和逐元素 gate 的低阶 FLOPs，但这些操作可能因 memory-bound 在
wall-clock 中并非完全可忽略。

## 9. 实现映射（本仓库）

| 概念 | 路径 / 符号 | 测试 / 实验 |
|---|---|---|
| Embedding / Linear | `assignments/spring2026/assignment1-basics/cs336_basics/model.py` | `test_embedding`、`test_linear` |
| RMSNorm fp32 reduction | 同文件：`RMSNorm.forward` | `test_rmsnorm` |
| RoPE cache/rotation | 同文件：`RotaryEmbedding` | `test_rope` |
| SDPA + causal mask | 同文件：`scaled_dot_product_attention`、`CausalMultiHeadSelfAttention` | attention snapshots |
| SwiGLU / SiLU | 同文件：`SwiGLU`、`SiLUFFN` | `test_swiglu`、ablation |
| Pre/post norm | 同文件：`TransformerBlock.forward` | `use_post_norm` |
| 完整 LM / tying | 同文件：`BasicsTransformerLM` | model snapshot、tied experiments |
| 模型超参数 | `.../cs336_basics/training.py`：`TrainConfig` | config 固化 |
| Shape 契约 | `.../tests/adapters.py` | state-dict keys 与输入输出 |
| 消融证据 | `.../report/main.tex` 第 7.3、7.5 节 | RMSNorm、NoPE、SwiGLU、tying |

## 10. 易错点与反例

1. **Mask 方向反了。** Query \(i\) 应允许 key \(j\le i\)。
2. **在 softmax 后 mask。** 未来 token 已参与归一化；必须在 softmax 前设 \(-\infty\)。
3. **softmax 轴错。** 应沿 key 轴，不是 head/query 轴。
4. **漏除 \(\sqrt{d_h}\)。** logits 方差随 head dimension 增长。
5. **RoPE 用整个 \(d\) 而非 \(d_h\)。** A1 接口按 head 旋转。
6. **RoPE 也旋转 V。** 标准做法只旋转 Q/K。
7. **把 RMSNorm 当 LayerNorm。** RMSNorm 不减均值，通常无 bias。
8. **把 pre-norm 写成 `Norm(x + Sublayer(x))`。** 那是 post-norm 结构。
9. **SwiGLU 与普通 FFN 都用 \(4d\) 后宣称公平。** 参数/FLOPs 不匹配。
10. **将 NoPE 仍能训练解释为“不需要位置”。** Causal boundary 泄露部分顺序信号，但距离表达受限。
11. **只比较相同步数。** Batch 或 context 不同会导致 tokens/FLOPs 不同。
12. **根据单 seed 小实验宣布普遍最佳架构。** 差异可能小于 seed noise 或系统误差。

## 11. 实践 Checklist

- [ ] 用纸写完整 `[B,T] → [B,T,V]` shape 流。
- [ ] 检查 `d % num_heads == 0` 且 RoPE dimension 为偶数。
- [ ] 构造 causal leakage test：改未来 token 不应影响过去 logits。
- [ ] 用大 logits 验证 stable softmax。
- [ ] 让 tiny model 过拟合一个 batch，排除 shift/mask/optimizer 错误。
- [ ] 手算参数量并与 PyTorch 统计对齐。
- [ ] 架构消融固定 tokenizer、数据顺序、seed、token budget、optimizer。
- [ ] SwiGLU/SiLU 比较时匹配参数量或 FLOPs，并明确匹配方式。
- [ ] 报告 train/validation loss、tokens/s、peak VRAM、参数量。
- [ ] 重要结论至少多 seed；细小差异给 uncertainty。
- [ ] Tying 时单独检查初始化和 state-dict alias。
- [ ] 把默认值记录为 baseline，不把它包装成理论最优。

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| loss 近似 \(\log V\) 且不降 | shift/mask/optimizer | 标签未右移、未来泄漏或未 step |
| 训练初期 NaN | init/LR/norm dtype | residual scale、fp16 reduction |
| 深层模型比浅层差很多 | pre/post norm、warm-up | 梯度路径不稳定 |
| attention 输出全相同 | softmax 轴、mask | 沿 query/head 归一化 |
| 长上下文突然退化 | RoPE cache/外推 | 超出训练长度、frequency mismatch |
| SwiGLU 参数暴涨 | hidden-size matching | 沿用 \(4d\) 未按 \(8d/3\) 调整 |
| checkpoint 后 tying 失效 | parameter alias | load 时创建了两份 Parameter |
| GPU utilization 下降 | GEMM shape/heads | width/head dimension 不友好 |
| 消融结论反复 | seed/tuning budget | 差异小于 noise、超参不公平 |
| PPL 看似更低但不可比 | tokenizer/data | vocabulary 或 bytes/token 不同 |

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| 训练发散/loss 尖峰 | 初始化尺度与学习率 | 残差流未缩放、lr 与深度不匹配 |
| loss 停在高位平台 | normalization 位置 | pre-LN/post-LN 选择与初始化不配套 |
| 参数量与论文对不上 | 统计口径（embedding、bias、tied） | 未说明是否计 embedding 或权重共享 |
| 长序列外推差 | RoPE base 与训练长度 | base 过小；需 position interpolation/YaRN |
| batch 增大吞吐不升 | attention 二次项与激活显存 | 序列长度主导，考虑 FlashAttention |
| 微调后基座能力受损 | 学习率与全参数更新 | 改用 LoRA 或更小 lr |
| 生成重复退化 | 采样参数与数据检查 | 温度/重复惩罚设置；排除训练数据退化 |
| 深层网络不收敛 | 残差路径与 gamma 初始化 | DeepNorm 类缩放未按深度调整 |

## 12. 作业关联

- `linear`, `embedding`, `rmsnorm`：最小模块与初始化；
- `positionwise_feedforward`, `rope`：现代 FFN 与相对位置信息；
- `softmax`, `scaled_dot_product_attention`, `multihead_self_attention`：数值和 shape 核心；
- `transformer_block`, `transformer_lm`：pre-norm decoder-only 模型；
- `transformer_accounting`：参数/FLOPs/内存；
- `layer_norm_ablation`, `pre_norm_ablation`, `no_pos_emb`, `swiglu_ablation`：受控架构实验；
- `learning_rate`, `batch_size_experiment`：架构不能脱离优化 recipe 判断；
- `leaderboard` 扩展中的 weight tying：参数效率与初始化耦合。

## 13. 讨论：效度威胁与架构结论边界

### 14.1 Construct validity

- parameter count、training FLOPs、wall-clock 与 inference cost 是不同预算；
- validation loss 不等价于 downstream capability、robustness 或 safety；
- context benchmark 的 retrieval 成功不等价于普遍 long-context reasoning；
- “相同 hidden size”不代表不同 FFN/attention 结构参数或 FLOPs 相同。

### 14.2 Internal validity

- 架构变更常同时改变初始化、LR optimum、kernel shape 与可用 batch；
- best-of-sweep 会产生 selection bias，尤其不同配置 tuning runs 数量不等；
- 相同步数而非相同 tokens/FLOPs 的比较混入训练预算；
- 单 seed 差异可能小于 initialization/data-order variance；
- early stopping/checkpoint selection 必须统一。

### 14.3 External validity

- 小模型上 SwiGLU/NoPE/tying 排名未必迁移到 billion-scale；
- 单一语言/领域的最佳 vocabulary/context 不代表多语言/code；
- 单 GPU hardware efficiency 不代表多卡 communication efficiency；
- 训练吞吐最佳的架构未必 inference latency/energy 最佳。

### 14.4 研究结论的推荐表述

避免：“Pre-norm 永远优于 post-norm。”
推荐：“在固定参数、token budget、optimizer 与三 seeds 的本实验中，pre-norm 达到更低
validation loss 和更稳定 gradient norm；该结论限于当前规模与 tuning budget。”

## 面试要点速记

**高频问题与答题要点**

1. **Q：pre-LN vs post-LN？** 要点：pre-LN 残差流直通、深层训练稳定；post-LN
   表达略强但对 warmup 与初始化敏感。现代大模型默认 pre-LN/RMSNorm。
2. **Q：RoPE 相对绝对位置编码的优势？** 要点：只依赖相对位置、通过旋转内积
   实现、外推性质更好；超长上下文需增大 base 或用 PI/YaRN。
3. **Q：为什么 SwiGLU 取代 ReLU FFN？** 要点：门控乘性调制提升表达；相同参数
   预算下 loss 更优；三矩阵结构（约 8d²/层，保持与双矩阵 FFN 等参）。
4. **Q：非嵌入参数量怎么估？** 要点：每层 attention 4d² + FFN 8d² = 12d²，
   总量 ≈ **12·l·d²**；embedding 另计 l·v（tied 可省输出侧）。

**必背数字**

- N≈12ld²（非嵌入）；RoPE base 10k → 长上下文 500k 级；RMSNorm 去均值中心化、
  计算更省；QK-norm 抑制 logit 增长与注意力熵坍缩。

## 14. 结论与本讲小结

现代 LM block 的核心是 causal attention 与 token-wise FFN，经 residual stream 反复组合。
Pre-norm/RMSNorm 保护优化稳定性，RoPE 把相对位置注入 QK 点积，SwiGLU 用门控提高 FFN 表达。
架构选择会同时改变参数、FLOPs、activation、kernel shape 和可训练性；MQA/GQA 等
inference-aware 变体与长度外推方法（position interpolation、YaRN）进一步把“架构”
与部署成本、上下文能力耦合起来。好的实验不是“换模块看一次 loss”，而是参数/预算匹配、
可追溯、多指标的 controlled comparison。

## 参考文献

<a id="ref-1"></a>[1] A. Vaswani et al. “Attention Is All You Need.”
*NeurIPS*, 2017. https://arxiv.org/abs/1706.03762

<a id="ref-2"></a>[2] A. Radford et al. “Language Models are Unsupervised
Multitask Learners.” OpenAI Technical Report, 2019.
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

<a id="ref-3"></a>[3] T. Brown et al. “Language Models are Few-Shot Learners.”
*NeurIPS*, 2020. https://arxiv.org/abs/2005.14165

<a id="ref-4"></a>[4] H. Touvron et al. “LLaMA: Open and Efficient Foundation
Language Models.” arXiv:2302.13971, 2023. https://arxiv.org/abs/2302.13971

<a id="ref-5"></a>[5] B. Zhang, R. Sennrich. “Root Mean Square Layer
Normalization.” *NeurIPS*, 2019. https://arxiv.org/abs/1910.07467

<a id="ref-6"></a>[6] R. Xiong et al. “On Layer Normalization in the
Transformer Architecture.” *ICML*, 2020. https://arxiv.org/abs/2002.04745

<a id="ref-7"></a>[7] J. Su et al. “RoFormer: Enhanced Transformer with
Rotary Position Embedding.” *Neurocomputing*, 2024.
https://arxiv.org/abs/2104.09864

<a id="ref-8"></a>[8] O. Press, N. A. Smith, M. Lewis. “Train Short, Test Long:
Attention with Linear Biases Enables Input Length Extrapolation.” *ICLR*, 2022.
https://openreview.net/forum?id=R8sQPpGCv0

<a id="ref-9"></a>[9] N. Shazeer. “GLU Variants Improve Transformer.”
arXiv:2002.05202, 2020. https://arxiv.org/abs/2002.05202

<a id="ref-10"></a>[10] J. Hoffmann et al. “Training Compute-Optimal Large
Language Models.” arXiv:2203.15556, 2022. https://arxiv.org/abs/2203.15556

<a id="ref-11"></a>[11] G. Yang et al. “Tensor Programs V: Tuning Large Neural
Networks via Zero-Shot Hyperparameter Transfer.” arXiv:2203.03466, 2022.
https://arxiv.org/abs/2203.03466

<a id="ref-12"></a>[12] J. L. Ba, J. R. Kiros, and G. E. Hinton. “Layer
Normalization.” arXiv:1607.06450, 2016. [link](https://arxiv.org/abs/1607.06450)

<a id="ref-13"></a>[13] H. Wang, S. Ma, L. Dong, et al. “DeepNet: Scaling
Transformers to 1,000 Layers.” arXiv:2203.00555, 2022.
[link](https://arxiv.org/abs/2203.00555)

<a id="ref-14"></a>[14] N. Shazeer. “Fast Transformer Decoding: One Write-Head
Is All You Need.” arXiv:1911.02150, 2019. [link](https://arxiv.org/abs/1911.02150)

<a id="ref-15"></a>[15] J. Ainslie, J. Lee-Thorp, M. de Jong, et al. “GQA:
Training Generalized Multi-Query Transformer Models from Multi-Head
Checkpoints.” *EMNLP*, 2023. [link](https://arxiv.org/abs/2305.13245)

<a id="ref-16"></a>[16] S. Chen, S. Wong, L. Chen, and Y. Tian. “Extending
Context Window of Large Language Models via Positional Interpolation.”
arXiv:2306.15595, 2023. [link](https://arxiv.org/abs/2306.15595)

<a id="ref-17"></a>[17] B. Peng, J. Quesnelle, H. Fan, and E. Shippole.
“YaRN: Efficient Context Window Extension of Large Language Models.”
*ICLR*, 2024. [link](https://arxiv.org/abs/2309.00071)

## 延伸阅读与复现材料

- Stanford CS336, [Spring 2026 Lecture 3](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)
- [Tokenization & Basics 主题导航](../experiments/topics/tokenization-and-basics.md)
- [A1 实验报告](../assignments/spring2026/assignment1-basics/report/main.pdf)
- [Lecture 04 — Attention Alternatives & MoE](lecture-04-attention-moe.md)
