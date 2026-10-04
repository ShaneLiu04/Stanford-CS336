---
title: "Lecture 04 — Attention Alternatives & Mixture of Experts"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-04-08"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf"
  - "../assignments/spring2026/assignment1-basics/"
---

# Lecture 04 — Attention Alternatives 与 Mixture of Experts：选择性计算的统一视角

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
- 适用对象：自学者、架构/系统工程师与 LLM 研究者

## 摘要

标准 Transformer 依赖 full self-attention 与 dense feed-forward network（FFN），其优势是
统一、并行和表达能力强，但在长上下文与大参数规模下分别面临 \(O(T^2)\) token mixing、
KV-cache 增长和 \(O(P)\) dense activation 成本。本文以 **selective computation** 为统一视角，
系统比较 sparse/local attention、linear attention、state-space/gated recurrent models、
MQA/GQA 与 Mixture of Experts（MoE）。我们不仅讨论渐近复杂度，还分析 memory traffic、
kernel regularity、state/KV cache、routing capacity、load balancing、all-to-all communication
和 failure modes。进一步给出 parameter/FLOP matching、quality-latency-memory 多目标评估、
controlled ablation 与效度威胁，说明“理论线性”“稀疏激活”与“实际更快”之间为何存在显著鸿沟。

**关键词：** Sparse Attention；Linear Attention；State Space Model；MQA；GQA；
Mixture of Experts；Routing；Load Balancing；All-to-All；Selective Computation

## 本文贡献

1. 用 selective computation 统一 token-axis sparsity 与 parameter-axis sparsity；
2. 推导 full/local/linear attention 的计算、状态和表达边界；
3. 区分 MHA/MQA/GQA 的训练计算与 autoregressive KV-cache 成本；
4. 系统分析 MoE routing、capacity、auxiliary loss、expert parallel 与通信瓶颈；
5. 提供架构研究的匹配预算、benchmark protocol、故障排查与论文级结论边界。

> Attention alternatives 决定“每个 token 看哪些历史”，MoE 决定“每个 token 使用哪些参数”。
> 两条线都在做 selective computation：减少每个 token 的激活计算，同时接受近似误差、路由、
> 状态管理、负载均衡与通信复杂度。

## 学习目标

1. 说明 full attention 的计算、内存与 KV-cache 瓶颈；
2. 推导 linear attention 的 recurrent / parallel dual form；
3. 比较 local/sliding-window、hybrid attention、state-space / gated recurrent alternatives；
4. 推导 Top-\(k\) MoE routing、active parameters、capacity 与 auxiliary losses；
5. 区分“总参数多”“每 token 激活参数少”“实际 wall-clock 快”；
6. 从 shape、FLOPs、memory、communication 和训练稳定性评估稀疏架构。

## 先修知识

- Lecture 2 的 roofline、memory accounting；
- Lecture 3 的 causal attention、SwiGLU、\([B,h,T,d_h]\) shape；
- softmax、矩阵乘法、基本分布式 all-to-all 概念（可边学边补）。

## 核心概念与公式推导路线

本讲用 selective computation 统一两条主线：attention alternatives 稀疏或压缩历史，MoE 稀疏激活
参数。公式部分从 full attention 成本出发，推导 linear attention 的 recurrent state、GQA KV-cache
缩减、Top-\(k\) routing 的 active/total parameters、capacity 与 load-balancing objective。

## 相关工作与技术谱系

Transformer 的 full attention 提供全局 content-based interaction [[1]](#ref-1)，但长序列
促成多条替代路线。Longformer 以 sliding window + global tokens 处理长文档
[[2]](#ref-2)，BigBird 将 local/random/global sparsity 与理论表达能力联系
[[3]](#ref-3)，Reformer 用 LSH attention 近似内容邻居 [[4]](#ref-4)。
Linear Transformer 与 Performer 分别通过 kernel factorization 和随机特征把 attention
改写为线性时间/状态更新 [[5]](#ref-5)[[6]](#ref-6)。RetNet 与 Mamba 进一步把
parallel training、recurrent inference 或 selective state-space dynamics 结合
[[7]](#ref-7)[[8]](#ref-8)。

Inference 侧，Multi-Query Attention（MQA）共享 K/V heads 以压缩 KV cache
[[9]](#ref-9)，Grouped-Query Attention（GQA）在 MHA 质量与 MQA 成本间折中
[[10]](#ref-10)。Parameter scaling 侧，GShard 与 Switch Transformer 将每 token 只路由到
少数 experts，使 total parameters 与 active compute 解耦
[[11]](#ref-11)[[12]](#ref-12)。Mixtral 展示 open-weight sparse MoE 的实用性
[[13]](#ref-13)，MegaBlocks 则从 block-sparse kernels 解决 routing 后的不规则计算
[[14]](#ref-14)。Expert Choice routing 反转选择方向、由每个 expert 挑选固定数量
tokens 以天然均衡负载 [[15]](#ref-15)；DeepSeekMoE 通过 fine-grained expert
segmentation 与 shared experts 提升 expert specialization [[16]](#ref-16)。

这些工作并非单一“替代 Transformer”的线性演进：不同方法优化 training FLOPs、decode
latency、memory、context quality 或 parameter capacity 中的不同目标，必须在相同约束下比较。

## 1. 统一视角：Selective Computation

Dense Transformer 对每个 token：

- 与所有过去 tokens 交互；
- 经过同一组 dense FFN 参数。

长上下文和大容量分别形成两个扩展问题：

| 问题 | Dense 成本 | Selective 方向 |
|---|---|---|
| 看多少历史 | attention \(O(T^2d)\) | sparse/local/linear/recurrent/hybrid |
| 用多少参数 | FFN 每 token 激活全部 \(O(df)\) | MoE Top-\(k\) experts |

选择性计算不等于“免费稀疏”。如果稀疏 kernel 不规则、router 造成 all-to-all、expert 负载不均，
理论 FLOPs 降低仍可能没有 wall-clock 收益。

## 2. Full Attention 的三个瓶颈

标准单 head：

\[
Y=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
\]

对于 batch \(B\)、heads \(h\)、长度 \(T\)、head dim \(d_h\)：

- score/value FLOPs：\(O(BhT^2d_h)=O(BT^2d)\)；
- 显式 score/probability memory：\(O(BhT^2)\)；
- autoregressive decode 的 KV cache：每层约 \(2BTd\) elements。

训练时 FlashAttention 能通过 tiling 避免把完整 \(T\times T\) matrix 写回 HBM
[[17]](#ref-17)，但不会改变
全连接 attention 的 \(O(T^2d)\) 算术量。推理时每生成一个 token，query 仍需读取历史 K/V，
总 decode attention 工作随输出长度近似二次增长。

因此要区分：

1. **exact systems optimization**：数学结果不变，如 FlashAttention；
2. **architectural approximation / sparsity**：改变允许的交互或状态，如 local/linear attention。

### 2.1 Training 与 decode 的瓶颈不同

Training 一次处理 \(T\) 个 queries，大矩阵乘可获得高 tensor-core utilization；主要问题是
\(T^2\) arithmetic 与 activation IO。Autoregressive decode 每步只有少量 queries，
需要读取长度 \(T\) 的 KV cache，常呈 matrix-vector / small-batch GEMM，memory bandwidth
成为主导。于是：

- FlashAttention 显著改善 training IO，但不消除 decode 读取历史 KV；
- MQA/GQA 主要优化 decode cache/bandwidth，训练 FLOPs 改善较小；
- recurrent/SSM 固定 state 对 decode 有吸引力，但训练 kernel 必须高效 scan；
- sparse attention 若 pattern 不规则，training/decode 都可能被 indexing/launch overhead 抵消。

比较架构时应分别报告 prefill throughput、time-to-first-token、inter-token latency、
KV/state bytes/request 与训练 tokens/s。

### 2.2 Exact、approximate 与 restricted 三类方法

- **Exact**：FlashAttention、memory-efficient SDPA；输出与 full softmax attention 数学等价。
- **Approximate**：random feature/low-rank/LSH 近似 full attention，存在 estimator error。
- **Restricted**：local/block sparse 直接修改 connectivity，未被连接的 token 权重严格为零。
- **Compressed state**：linear recurrence/SSM 将历史映射到固定 state，信息容量受限。

这四类方法的 correctness benchmark 不同：exact 方法测数值误差；approximate 方法测近似误差；
restricted 方法测 task-induced connectivity；state 方法测 memory capacity 与长程遗忘。

## 3. Sparse 与 Local Attention

### 3.1 Sliding-window attention

每个 query \(i\) 只看最近 \(w\) 个 keys：

\[
\mathcal N(i)=\{j:\max(0,i-w+1)\le j\le i\}.
\]

复杂度从 \(O(T^2d)\) 降为 \(O(Twd)\)，attention state 从 \(O(T^2)\) 降为 \(O(Tw)\)。
但单层 receptive field 仅 \(w\)；堆叠 \(L\) 层后信息最远约传播 \(L(w-1)\) 个位置，
且路径长度增加。

### 3.2 Global + local / block sparse

可为少数 global tokens 提供全局连接，其他 tokens 只局部连接；或按 blocks 设计固定稀疏图。
理论复杂度下降只有在 kernel 能跳过 masked blocks 时才成立。构造 dense \(T\times T\) 后再把大部分
位置 mask 掉，FLOPs 和内存几乎没省。

### 3.3 Hybrid layers

现代模型常交替 full attention 与 local/recurrent layers：

- local 层低成本处理局部模式；
- 周期性 full 层重建全局通信；
- recurrent/state-space 层提供压缩历史。

评估时要报告“全局信息的最大/平均路径长度”，不能只报 asymptotic FLOPs。

### 3.4 Connectivity、dilation 与信息路径

Sparse pattern 可视为有向图：token 是节点，attention edge 表示一层可读取关系。
经过 \(L\) 层后，可访问范围由图的 \(L\)-hop closure 决定。Sliding window 的 receptive
field 线性增长；dilated pattern 可指数扩大范围；global tokens 将图直径降到常数级，
但形成 bottleneck。

设计 pattern 时同时问：

- 任意两 token 最短路径多长；
- 每层 degree/edges 数；
- pattern 是否 static，能否 block-sparse 编译；
- document boundary 是否被跨越；
- global token 是否承载过多信息；
- retrieval target 是否落在 connectivity 内。

BigBird 的理论结果依赖 local/random/global edges 的组合 [[3]](#ref-3)；它不意味着任意
稀疏 mask 都保留 full attention 表达能力。

## 4. Linear Attention：从二次矩阵到状态

### 4.1 Kernel factorization

若 attention similarity 可写为

\[
\operatorname{sim}(q,k)=\phi(q)^\top\phi(k),
\]

且 \(\phi(\cdot)\) 产生非负 features，则归一化 causal linear attention 可写成

\[
y_t=
\frac{\phi(q_t)^\top\left(\sum_{i\le t}\phi(k_i)v_i^\top\right)}
{\phi(q_t)^\top\left(\sum_{i\le t}\phi(k_i)\right)+\epsilon}.
\]

定义状态：

\[
S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad
z_t=z_{t-1}+\phi(k_t),
\]

\[
y_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t+\epsilon}.
\]

这利用结合律把 \((QK^\top)V\) 改写为 \(Q(K^\top V)\)，无需显式 \(T\times T\)。

### 4.2 Shape 与复杂度

令 feature dimension 为 \(r\)，value dimension 为 \(d_v\)：

```text
phi(q_t), phi(k_t)    [B,h,r]
v_t                   [B,h,d_v]
S_t                   [B,h,r,d_v]
z_t                   [B,h,r]
y_t                   [B,h,d_v]
```

每 token 更新/读取约 \(O(rd_v)\)，整段 \(O(Trd_v)\)，state memory \(O(rd_v)\)，不随
decode context length 增长。若 \(r\approx d_h\)，可理解为 \(O(Td_h^2)\)。

### 4.3 代价

Softmax attention 的动态、内容相关归一化很难被有限维 feature map 完全复现。固定状态
\(S_t\) 会把任意长历史压缩到有限容量，可能丢失精确 retrieval。并行训练与 recurrent decode
虽在数学上可形成 dual form，数值稳定、gating、chunking 和 kernel 实现仍决定实际性能。

### 4.4 Feature map、因果 scan 与数值问题

常见 \(\phi\) 包括 `elu(x)+1`、随机特征近似 softmax kernel 等。不同 feature map 改变：

- 是否保证非负 denominator；
- approximation bias/variance；
- feature dimension \(r\)；
- state update 的稳定范围；
- 是否可用 tensor-core-friendly GEMM。

Causal prefix sums 具有结合结构，可用 parallel scan 训练；decode 则递归更新 \(S_t,z_t\)。
训练并行算法必须避免顺序 Python loop，否则理论 \(O(T)\) 仍可能比 fused \(O(T^2)\) attention 慢。

Denominator 很小时输出会爆炸；长期累积还可能产生 state scale drift。工程上需监控
\(\|S_t\|\)、denominator quantiles、NaN/Inf，并考虑 normalization、decay、chunk reset
和 fp32 accumulation。

## 5. State-space / Gated Recurrent Alternatives

一个抽象 recurrent state update：

\[
S_t=A_t\odot S_{t-1}+B_t,\qquad y_t=C_t(S_t),
\]

其中 input-dependent decay/gate \(A_t\) 控制遗忘，\(B_t\) 写入新信息。Mamba-2
[[18]](#ref-18)、Gated DeltaNet [[19]](#ref-19) 等方法的具体参数化不同，但共同目标是：

- 训练时用 scan / dual form 并行；
- decode 时维护固定大小 state，单 token 近线性或常数上下文成本；
- 用 gating / delta update 提升有限状态的选择性。

以 delta-rule 风格的矩阵记忆为例：

\[
S_t=S_{t-1}
+\beta_t\left(v_t-S_{t-1}k_t\right)k_t^\top.
\]

\(S_{t-1}k_t\) 是当前记忆对 key 的预测，更新只写入 prediction error。若 key 未归一化或 gate
失控，state 可能不稳定；这类模型不能只凭线性复杂度断言优于 attention。

### 5.1 从连续 SSM 到 selective recurrence

线性 continuous-time state-space model 写作

\[
\dot h(t)=Ah(t)+Bx(t),\qquad y(t)=Ch(t).
\]

离散化后得到 \(h_t=\bar A h_{t-1}+\bar B x_t\)。经典线性时不变系统可转成 convolution
并行训练；selective SSM 让 \(\bar A,\bar B,C\) 依赖输入，提高内容选择能力，但失去简单
固定 convolution，需要 specialized scan kernel。Mamba 的核心贡献既是 selective
parameterization，也是使其可在 GPU memory hierarchy 上高效执行的 scan 系统
[[8]](#ref-8)。

固定 state size 带来 decode 内存优势，也意味着任意历史必须压缩进有限维状态。对需要精确
copy/retrieval 的任务，attention 的显式 addressable memory 与 SSM 的 compressed memory
具有根本差异。

### 5.2 应如何比较

- **Recall quality**：needle/retrieval 任务是否需要精确 token-level history；
- **Training throughput**：scan kernel 是否高效；
- **Decode latency/state size**：固定 state 是否真正优于 KV cache；
- **Hybrid ratio**：多少 full-attention layers 才维持质量；
- **长度外推**：state saturation 与训练长度之外行为。

复杂度更好是必要但非充分条件。

## 6. Multi-query / Grouped-query 与 KV Cache

这类方法没有消除 attention 的 \(T^2\) 训练计算，但减少 K/V heads：

- MHA：Q/K/V 都有 \(h_q\) heads；
- MQA：所有 query heads 共享 1 组 K/V；
- GQA：\(h_q\) query heads 分成 \(h_{kv}\) 组共享 K/V。

KV cache 每层从 MHA 的

\[
2BT h_qd_h
\]

elements 降为

\[
2BT h_{kv}d_h.
\]

缩减比约 \(h_q/h_{kv}\)。Q heads 仍为 \([B,h_q,T,d_h]\)，K/V 为
\([B,h_{kv},T,d_h]\)，计算前按 group 映射而不是物理复制，否则节省会被破坏。

它主要解决 autoregressive serving 的 memory bandwidth / capacity；训练质量与 head sharing
程度仍需验证。

### 6.1 Cache accounting 与 serving capacity

对 \(L\) 层、dtype bytes \(s\)，单 request KV cache 近似

\[
M_{\mathrm{KV}}=2LT h_{kv}d_hs.
\]

例如 \(L=32,T=8192,h_q=32,d_h=128,\) bf16：
MHA \(h_{kv}=32\) 每 request 约 4 GiB；GQA \(h_{kv}=8\) 约 1 GiB；MQA 约 128 MiB。
实际还包含 allocator/page metadata、padding 和 beam/sample copies，但数量级说明 KV heads
直接决定可并发 requests。

### 6.2 从 MHA checkpoint 转 GQA

GQA 可从 MHA K/V heads 分组平均或 pooling 初始化，再 continued pretraining 恢复质量
[[10]](#ref-10)。公平评估需同时比较：

- prefill/decode latency 与 throughput；
- KV bytes/request 和 maximum concurrency；
- perplexity/downstream loss；
- continued-training tokens/compute；
- long-context 与 head specialization 退化。

直接从头训练与 checkpoint conversion 是不同研究问题，不能混为同一“GQA 效果”。

## 7. Mixture of Experts（MoE）

### 7.1 从 dense FFN 到 routed experts

Sparsely-gated MoE 的核心思想是以可学习 router 实现 conditional computation，
让总参数容量增长快于每 token 计算 [[20]](#ref-20)。

设有 \(E\) 个 FFN experts \(f_e(x)\)，router logits：

\[
r(x)=W_rx\in\mathbb R^E,\qquad p(x)=\operatorname{softmax}(r(x)).
\]

Top-\(k\) 集合 \(\mathcal T_k(x)\) 后，输出可写成：

\[
y=\sum_{e\in\mathcal T_k(x)}\tilde p_e(x)f_e(x),
\qquad
\tilde p_e=\frac{p_e}{\sum_{j\in\mathcal T_k}p_j}.
\]

输入若为 \([B,T,d]\)，先 flatten 为 \(N=BT\) tokens：

```text
router logits       [N,E]
top-k indices       [N,k]
top-k weights       [N,k]
dispatch            tokens -> per-expert ragged batches
expert outputs      [N,k,d]
combine             [N,d] -> [B,T,d]
```

Expert 通常是 SwiGLU/FFN。总 expert 参数约 \(E\cdot3df_e\)，每 token 激活约
\(k\cdot3df_e\)，因此可在较小 active FLOPs 下拥有很大总容量。

### 7.2 “参数多但计算少”的精确定义

令 shared dense 部分参数 \(P_s\)，每个 expert 参数 \(P_e\)：

\[
P_{\rm total}=P_s+EP_e,\qquad
P_{\rm active/token}\approx P_s+kP_e.
\]

若 \(E=64,k=2\)，expert 总参数是单 expert 的 64 倍，但每 token 只调用 2 个。注意：

- checkpoint / model storage 按 \(P_{\rm total}\)；
- optimizer states 也按被训练的总参数分布存储；
- forward FLOPs 近似按 active parameters；
- router、dispatch、padding、all-to-all 产生额外成本；
- 每个 expert 看到的 tokens 变少，可能 undertrain。

所以“671B total / 37B active”这类描述必须同时给 total 与 active，不能只用一个数字比较 dense 模型。

## 8. Routing、Capacity 与 Load Balancing

### 8.1 Token-choice 与容量

Top-\(k\) token-choice 让每个 token 选择 experts，易导致热门 expert overload。平均每 expert
分配量为 \(Nk/E\)。常定义 capacity：

\[
C=\left\lceil c\frac{Nk}{E}\right\rceil,
\]

其中 \(c>1\) 是 capacity factor。超过容量的 token 可能被 drop、reroute 或 padding 策略处理。
增大 \(c\) 减少 drop，却浪费计算/内存。

### 8.2 辅助负载均衡

令

\[
f_e=\frac{1}{Nk}\sum_x\mathbf 1[e\in\mathcal T_k(x)]
\]

为实际 dispatch fraction，

\[
P_e=\frac1N\sum_x p_e(x)
\]

为平均 router probability。一类辅助损失：

\[
\mathcal L_{\rm balance}
=\alpha E\sum_{e=1}^{E}f_eP_e.
\]

它鼓励均匀利用，但会把“模型真正偏好的 specialization”拉向均匀，过强时损害主任务。
还可用 router z-loss 抑制极大 logits，例如

\[
\mathcal L_z=\beta\frac1N\sum_x
\left(\log\sum_e e^{r_e(x)}\right)^2.
\]

不同实现的 balance loss 定义不同；阅读论文或代码时不要只看名称。

### 8.3 Shared 与 fine-grained experts

- **Shared experts**：所有 tokens 总会经过，承载通用模式，降低 routed experts 的重复学习；
- **Fine-grained experts**：把大 expert 拆成更多小 experts，Top-\(k\) 组合更灵活；
  DeepSeekMoE 的消融显示，两者结合可在相同计算预算下取得比常规粒度 MoE 更优的质量
  [[16]](#ref-16)；
- 代价：更小 GEMM、更高 routing metadata、更多通信，hardware efficiency 可能下降。

MoE 的设计目标不是最大化 expert 数，而是在 specialization 与大矩阵效率之间找平衡。

### 8.4 Routing gradient、collapse 与 dropless MoE

`topk` index selection 离散、不可微；常见实现让 selected expert weights 的连续部分接收梯度，
并用 auxiliary loss 训练 router。早期随机偏差可能形成正反馈：

\[
\text{更多 tokens}\rightarrow\text{expert 学得更快}\rightarrow
\text{router 更偏好该 expert}\rightarrow\text{更多 tokens},
\]

最终 router collapse。可缓解方法包括 router noise、temperature、capacity、load-balance loss、
expert-choice routing 或 shared experts。

Capacity-based MoE 会 drop/reroute overflow tokens；dropless MoE 保留所有 assignments，
通过 block-sparse/ragged kernels 处理不均衡 batch [[14]](#ref-14)。Dropless 消除语义性
token drop，却不能消除 straggler：最忙 expert 仍决定 step 尾延迟。

### 8.5 Token-choice、Expert-choice 与 batch priority

- **Token-choice**：每 token 选 experts，语义直观但 expert load 无保证；
- **Expert-choice**：每 expert 选固定数量 tokens，负载天然平衡，但 token 可能被多个或零个
  experts 选择 [[15]](#ref-15)；
- **Batch-priority routing**：容量不足时优先保留 router score 高的 assignments，
  比按输入顺序截断更合理 [[21]](#ref-21)。

除 auxiliary loss 外，DeepSeek-V3 采用 per-expert bias 的动态调整实现 auxiliary-loss-free
的负载均衡：bias 只影响 routing 决策、不进入 router 的梯度通路，从而在不惩罚 specialization
的前提下抑制过载 [[21]](#ref-21)。

比较 routing policy 时必须报告 coverage（多少 token 获得 0/1/多 expert）、负载方差、
drop/reroute rate 与 quality，不只看平均 utilization。

## 9. MoE 的系统代价

Expert parallelism 把 experts 分到不同 devices。一次 MoE 层通常：

1. 每台设备本地算 router；
2. 按目标 expert 对 tokens pack；
3. all-to-all 发送 token activations；
4. 本地 expert GEMM；
5. all-to-all 返回输出并 combine。

通信量粗略随 routed activation：

\[
O(Nkd\cdot \text{bytes/element}),
\]

但实际延迟还取决于设备拓扑、token 分布、消息碎片和 straggler。某 expert overload 会让其所在设备
成为全局尾延迟。即使理论 expert FLOPs 与 dense baseline 相同，MoE 也可能因小 batch per expert
降低 GEMM utilization。

评估 MoE 至少报告：

- total / active parameters；
- router entropy、每 expert token counts、drop rate；
- capacity factor 与 load-balance loss；
- expert GEMM sizes、all-to-all time、tokens/s；
- 训练/推理 quality，而非仅 perplexity 或仅速度。

### 9.1 Expert parallel communication accounting

若每个 rank 原有 \(N/P\) tokens、每 token 路由 \(k\) 个 experts、activation width \(d\)、
元素 \(s\) bytes，一次 dispatch 的 global payload 量级为 \(Nkds\)，返回再一次同量级。
每 rank 实际跨网络 bytes 依 expert placement 与本地命中率而异，但至少需要记录：

\[
T_{\text{MoE}}\approx
T_{\text{router}}+T_{\text{pack}}+T_{\text{A2A-fwd}}
+T_{\text{expert}}+T_{\text{A2A-bwd}}+T_{\text{combine}}.
\]

All-to-all 对小消息和不均匀 split 敏感；通信还可能与 expert GEMM overlap。只用总 bytes
无法解释 latency，应结合 topology（NVLink/PCIe/InfiniBand）、消息数与最大 rank load。

### 9.2 Expert batch size 与 padding waste

平均 expert batch 为 \(Nk/E\)，但真实分布有方差。小 expert batch 会降低 GEMM utilization；
固定 capacity padding 又执行无效 FLOPs。可报告：

- expert token count 的 mean/std/max/Gini；
- padded slots / real assignments；
- 每 expert GEMM 的 \(m,n,k\)；
- straggler rank idle time。

Fine-grained experts 增强组合性，却让 \(m\) 更小；MegaBlocks 一类系统通过 block-sparse
reordering 恢复大块计算 [[14]](#ref-14)。

### 9.3 MoE inference 并不只支付 active weights

单 request 只激活 \(k\) experts，但 server 仍需让所有 experts 驻留显存或跨层/跨设备取权重。
低 batch 时不同 tokens 路由分散，可能读取大量 expert weights；高并发能提高复用，却加剧
load imbalance。评估应报告 model residency、weight bandwidth、batch routing diversity、
tail latency 与并发吞吐。

## 10. 架构评估与研究方法

### 10.1 先声明预算匹配方式

Attention alternative / MoE 比较至少有四种公平性：

1. 相同 total parameters；
2. 相同 active parameters/FLOPs；
3. 相同 training wall-clock/energy；
4. 相同 inference latency/memory。

这些约束通常不能同时满足。MoE 在 total parameters 匹配下 active compute 更低；
在 active compute 匹配下 total storage 更大。报告必须明确主要约束，并把其他量作为结果列出。

### 10.2 多目标 Pareto 而非单一 loss

建议同时报告：

- pretraining/validation loss 与 downstream tasks；
- training tokens/s、MFU、peak memory；
- prefill/decode latency、throughput、KV/state bytes；
- total/active parameters、checkpoint size；
- communication、router entropy、drop rate；
- long-context retrieval 与 short-context regression。

架构 A loss 略低但 decode 慢 3×，是否“更好”取决于 deployment objective。使用 Pareto frontier
比把异构指标加成一个任意分数更诚实。

### 10.3 Controlled ablation matrix

```text
attention: full / local / linear / hybrid
kv heads: MHA / GQA / MQA
ffn: dense / top-1 / top-2 MoE
budget: parameter-matched / FLOP-matched
seeds: >= 3
```

固定 tokenizer、data order、training tokens、optimizer family 与 evaluation。每个候选允许相同
tuning budget，或明确测试“单一 shared recipe 下的 robustness”。先用小预算排除 NaN、router
collapse 和 connectivity bug，再做 full-budget confirmation。

### 10.4 研究假设示例

> 在固定 active FLOPs 和 total training tokens 下，Top-2 MoE 会降低 validation loss，
> 但在单机 PCIe 环境中 all-to-all/dispatch 使 tokens/s 低于等计算 dense FFN；
> shared expert 可降低 router collapse，但减少 specialization capacity。

该假设同时给出 quality 与 systems 机制，允许被 loss、router statistics 和 profiler 证伪。

## 11. Shape 与复杂度汇总

| 方法 | 训练序列计算 | Decode state/cache | 主要风险 |
|---|---:|---:|---|
| Full MHA | \(O(T^2d)\) | \(O(Th_{kv}d_h)\) / layer | 长上下文二次成本 |
| FlashAttention | 仍 \(O(T^2d)\) | 同 full | 只优化 I/O，不改连接 |
| Sliding window \(w\) | \(O(Twd)\) | \(O(wh_{kv}d_h)\) | 远距路径变长 |
| Linear attention | \(O(Trd_v)\) | \(O(rd_v)\) | 有限状态、近似 softmax |
| Recurrent/SSM | 常见 \(O(Td^2)\) 或结构化线性 | 固定 state | scan/kernel、精确 recall |
| GQA | attention 仍 \(O(T^2d)\) | \(O(Th_{kv}d_h)\) | K/V sharing 质量折中 |
| Dense FFN | \(O(Tdf)\) | 无序列 cache | 每 token 激活全部参数 |
| Top-\(k\) MoE | 约 \(O(Tkdf_e)\) + routing | 总权重很大 | balance、capacity、all-to-all |

Asymptotic notation 会隐藏常数、kernel shape 和通信。实际选择必须结合 context、batch、硬件与质量。

## 12. 实现映射（本仓库）

本仓库 A1 实现的是 dense baseline；没有把本讲的 MoE/linear-attention 扩展伪装成官方 A1 要求。

| 本讲概念 | 仓库基线位置 | 如何扩展 / 验证 |
|---|---|---|
| Full causal attention | `assignments/spring2026/assignment1-basics/cs336_basics/model.py`：`scaled_dot_product_attention` | 替换前先做输出/causal leakage baseline |
| MHA Q/K/V shape | 同文件：`CausalMultiHeadSelfAttention` | GQA 需拆 `num_heads` 与 `num_kv_heads` |
| Dense FFN | 同文件：`SwiGLU` | MoE expert 可复用相同 FFN 接口 |
| Transformer block | 同文件：`TransformerBlock` | 仅把部分层的 `ffn` 换成 routed module |
| Resource accounting | `.../report/main.tex` 第 3.3、4.3 节 | 加入 active/total 参数、routing/通信成本 |
| Shape tests | `.../tests/test_model.py`、`tests/adapters.py` | 扩展测试不要修改官方 adapter 契约 |
| Training metrics | `.../cs336_basics/training.py` | 追加 expert load/drop/router entropy |
| A2 systems 入口 | `assignments/spring2026/assignment2-systems/` | 区分 exact attention kernel 与架构近似 |

一个安全的自学扩展顺序：

1. 复制最小实验配置，不改官方 adapter；
2. 先实现 single-device Top-1/Top-2 router 和等宽 experts；
3. 做 dense-equivalence 特例测试（\(E=1,k=1\)）；
4. 再测 routing/load；最后才考虑 expert parallel all-to-all。

## 13. 易错点与反例

1. **把 FlashAttention 称为 linear attention。** 前者保持 exact \(T^2\) 算术，后者改变公式/状态。
2. **创建 dense mask 后宣称 sparse FLOPs。** 没有稀疏 kernel 就没跳过计算。
3. **只比较大 \(T\) 的渐近复杂度。** 小/中 context 下常数与 GEMM efficiency 可反转结论。
4. **认为 recurrent state 能无损记住无限历史。** 固定维状态必然形成信息瓶颈。
5. **把 GQA 当训练 attention 线性化。** 它主要缩小 KV cache，不消除 query-key 二次交互。
6. **MoE 只报 total parameters。** 每 token compute 由 active parameters 决定。
7. **MoE 只报 active parameters。** 存储、optimizer、通信仍受 total parameters 影响。
8. **Top-\(k\) 后不重归一化却默认权重和为 1。** 两种定义均可，但必须明确。
9. **忽略 capacity overflow。** Dropped tokens 会静默改变模型语义。
10. **balance loss 越强越好。** 过强会压制 specialization。
11. **物理复制 GQA K/V 到所有 query heads。** 可能抵消 memory savings。
12. **把 expert 当语义标签。** 路由 specialization 未必稳定对应“数学/代码/语言”。
13. **在 A1 official tests 中强塞扩展。** 会破坏接口可比性，应保持独立模块/配置。
14. **把 auxiliary-loss-free 等价于“没有均衡约束”。** V3 用可学习 bias 动态
    修正路由并配序列级损失补偿，约束仍在，只是不进主损失梯度通路。
15. **把 MLA 当成 GQA 的极端情形。** MLA 是对 KV 做低秩压缩、cache 存压缩
    表示；混同会导致 KV cache 估算与 kernel 设计双双出错。

## 14. 实践 Checklist

- [ ] 先测 dense attention 的 FLOPs、peak memory、latency baseline。
- [ ] 明确优化是 exact kernel 还是 architecture change。
- [ ] 写出方法的训练并行形态与 autoregressive recurrent 形态。
- [ ] 对 sparse pattern 画连接图，检查信息传播路径。
- [ ] 对 GQA 分别记录 `num_query_heads`、`num_kv_heads` 与 cache bytes。
- [ ] MoE 同时报 total / active parameters 和 active FLOPs。
- [ ] 记录每 expert token count、router entropy、drop rate、capacity factor。
- [ ] 检查 \(E=1,k=1\) 与 dense expert 的数值等价。
- [ ] 对 load-balance coefficient 做小范围 sweep，不只看主 loss。
- [ ] Profile dispatch/pack/all-to-all/expert GEMM/combine 的时间占比。
- [ ] 固定 token/FLOP/wall-clock 三种预算分别比较，避免单一指标误导。
- [ ] 长上下文实验包含 retrieval、language modeling quality 与真实 memory/latency。

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| sparse attention 不加速 | profiler/kernel | dense mask、irregular gather |
| linear attention NaN | denominator/state norm | feature map 非正、累计漂移 |
| 长距 retrieval 失败 | connectivity/state capacity | window 不可达、状态饱和 |
| GQA 显存未下降 | tensor storage/expand | 物理复制 K/V heads |
| decode 慢但 prefill 快 | KV bytes/request | memory bandwidth / small GEMM |
| router 迅速单 expert | token counts/entropy | positive feedback、aux loss 弱 |
| MoE loss 正常但吞吐低 | expert batch/A2A | 小 GEMM、padding、straggler |
| token 被静默丢弃 | capacity/drop rate | capacity factor 太低 |
| balance loss 降、主 loss 升 | coefficient | 过度均匀抑制 specialization |
| 多卡偶发 hang | all-to-all splits | rank 间 token-count metadata 不一致 |

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| MoE 训练 loss 尖峰/崩溃 | aux loss 权重与 router 初始化 | 负载均衡约束过弱、router z-loss 缺失 |
| MoE 推理不见加速 | expert 数与容量因子 | 容量溢出、all-to-all 未与计算 overlap |
| expert 负载严重不均 | 路由分布与溢出率统计 | aux loss 系数不合适、token 分布漂移 |
| linear attention 长程回忆差 | 状态维度与门控设计 | 核函数近似误差、状态容量不足 |
| KV cache 显存过大 | head 分组数（MQA/GQA） | KV head 未压缩或未量化 |
| attention 输出数值错 | causal mask 与 softmax 缩放 | mask 错位、\(1/\sqrt{d_k}\) 缺失 |
| fused kernel 与参考不符 | online softmax 累计量 | 区分数值误差与实现错误（LSE 复用） |
| 稀疏化后质量骤降 | 稀疏模式与任务匹配 | 局部窗口过小、模式对检索任务不适配 |

## 15. 作业关联

- A1 `scaled_dot_product_attention` 是所有 alternatives 的正确性基线；
- `multihead_self_attention` 的 Q/K/V shape 是 GQA/MQA 扩展起点；
- `transformer_accounting` 给出 full attention 的 \(T^2\) 对照；
- `positionwise_feedforward` / `SwiGLU` 可作为每个 expert 的基础；
- A1 architecture extensions 应与官方接口隔离，不属于题面必需；
- A2 的 FlashAttention 是 exact systems optimization，不等同于本讲的近似 attention；
- 后续 inference 主题会继续讨论 KV cache、decode bandwidth 与 batching；
- 后续 parallelism 主题会解释 expert parallelism 的 all-to-all 成本。

## 16. 讨论：效度威胁与结论边界

### 17.1 Construct validity

- theoretical sparsity 不等于 executed sparse FLOPs；
- active parameters 不等于 model memory/optimizer state；
- perplexity 不充分衡量 retrieval、reasoning、tail latency 或 energy；
- needle benchmark 不代表自然长文档建模；
- router entropy 高不一定 load 平衡，token counts 均匀也不保证 expert specialization。

### 17.2 Internal validity

- 方法若使用不同 parameter/FLOP/tuning budget，无法归因架构本身；
- capacity drop、padding 和 reroute 若未记录，会静默改变 objective；
- sparse/SSM kernel maturity 差异可能主导 wall-clock；
- MoE 每 expert 实际 tokens 不同，等总 tokens 不等每参数训练充分；
- checkpoint conversion 与 from-scratch training 不能直接混比。

### 17.3 External validity

- synthetic retrieval、固定 context 的结论未必迁移到 code、dialogue 或 multimodal；
- 单机 NVLink 结果不代表多节点 all-to-all；
- training throughput 最佳方法未必 decode/serving 最佳；
- 小模型 router behavior 未必外推到百 experts/billion parameters；
- 当前实现成熟度不应被误解为算法上限。

### 17.4 推荐的学术表述

避免：“Linear attention 比 Transformer 更高效。”
推荐：“在固定模型规模、序列分布与 GPU 上，该 linear-attention kernel 在 \(T\ge X\)
取得 Y× training throughput，并以 Z 的 retrieval degradation 为代价；短序列与 decode
regime 的结论不同。”

## 面试要点速记

**高频问题与答题要点**

1. **Q：FlashAttention 到底快在哪？** 要点：不减少 FLOPs；tiling + online
   softmax 把 HBM 读写从 O(N²) 降到 O(N)（IO-aware），显存同步降为 O(N)，
   从而解锁长上下文。
2. **Q：MQA/GQA 与 KV cache？** 要点：cache = 2·l·h_kv·d_h·seq·B·bytes；
   GQA 把 h_kv 从 h 降到 h/g，容量与带宽同比例下降，质量损失小。
3. **Q：MoE 如何解耦参数量与计算量？** 要点：参数 ∝ E·d²，每 token 只激活
   top-k 专家；代价是全量参数显存与 all-to-all 通信；负载靠 aux loss +
   容量因子约束。
4. **Q：linear attention 牺牲了什么？** 要点：固定大小状态限制精确 recall；
   门控/chunkwise（SSD 类）补回部分能力；检索型任务仍吃亏。
5. **Q：auxiliary-loss-free 怎么做到不伤 specialization？** 要点：per-expert
   可学习 bias 只修正路由选择、不进损失与梯度通路；均衡靠动态调整而非
   “拉平”惩罚；V3 官方口径负载均衡率 99.5% 以上，并配序列级损失补偿。
6. **Q：MLA 与 GQA 的本质区别？** 要点：GQA 是共享 KV head，MLA 是对 KV 做
   低秩压缩、cache 存压缩表示；MLA 官方口径 KV cache -93.3%、最大吞吐
   +576%，代价是 kernel 与实现复杂度更高。
7. **Q：SSM 什么时候优于 attention？** 要点：decode 固定 state、长上下文
   成本不增；精确 recall 任务吃亏；2024 口径 Transformer 仍占 74%、替代
   范式约 22%，混合层配置（如 Griffin 线性递归+局部注意力）最有前景。

**必背数字**

- KV cache 公式；FA 显存 O(N)；MoE aux loss 权重典型 0.01 量级、容量因子
  1–1.5；DeepSeek 级细粒度专家 + 共享专家是当前主流配置。

**工业界参照（面试引用）**

- DeepSeek-V3：每层 1 共享专家 + 256 路由专家、每 token 激活 8 个路由专家；
  671B 总参数 / 约 37B 激活；组级路由 8 组、token 先 top-2 组再组内选专家。
- 负载均衡演进：V2 三重辅助损失 + 设备级 token 丢弃 → V3 auxiliary-loss-free
  （可学习偏置仅影响选择不影响损失），官方口径负载均衡率 99.5% 以上。
- 消融官方口径：细粒度分割精度 +12%、共享专家 +7% 性能增益；较 GShard 参数
  规模 -33%；DeepSeek-MoE 较 dense 计算效率约 7×（0.24B vs 1.89B 激活量级）。
- V3 推理：EP64 纯专家并行、跨节点 IB + 片内 NVLink 混合通信，32-GPU 集群
  设备利用率约 98%；据报道吞吐约 8.2k tokens/s/GPU 量级；部署最大 4 节点。
- MLA（DeepSeek-V2）：KV cache -93.3%、最大吞吐 +576%（官方口径）。
- 架构格局（2024 全景报告口径）：Transformer 74%、替代范式约 22%。

## 行业现状与最新进展（2024–2026）

### DeepSeek-V3：细粒度 + 共享专家 + auxiliary-loss-free

- 每层 1 个共享专家 + 256 个路由专家，每 token 激活 8 个路由专家；
  总参数 671B、激活约 37B——本讲 "total vs active" 的最大规模公开实例。
- 组级路由：8 个组、token 先选 top-2 组再组内选专家，是官方的负载均衡策略。
- 负载均衡迭代：V2 用 160 路由专家 + 2 共享、三重辅助损失 + 设备级 token
  丢弃；V3 升级为 auxiliary-loss-free——可学习偏置项动态修正路由概率，
  仅影响选择不影响损失，官方口径负载均衡率 99.5% 以上，配序列级损失补偿，
  摒弃 token 丢弃。
- 细粒度消融（官方口径）：细粒度分割精度 +12%，共享专家 +7% 性能增益；
  较同等条件 dense 计算效率提升约 7×（0.24B vs 1.89B 激活参数量级）；
  较 GShard 参数规模缩减 33%；V3 256 专家下激活组合空间约 4.4 万亿种。
- 推理侧：EP64 纯专家并行；跨节点 IB + 片内 NVLink 混合通信，32-GPU 集群
  设备利用率约 98%；据报道推理吞吐约 8.2k tokens/s/GPU 量级；部署最大 4 节点。

### MLA 与 KV 压缩谱系

MLA（DeepSeek-V2）对 KV cache 做低秩压缩，官方口径 KV cache 减少 93.3%、
最大吞吐 +576%。谱系：MHA → MQA（共享 1 组 K/V）→ GQA（分组共享）→
MLA（低秩压缩、cache 存压缩表示），压缩力度依次增强，kernel 与实现复杂度
也依次上升。本讲第 6 节的 cache 公式是统一度量工具。

### SSM / 混合架构路线与格局

- Falcon-Mamba 7B 在基准上优于同规模 Transformer（2024 口径）。
- AI21 Mamba-Transformer 混合在知识/推理基准优于 8B Transformer，
  且推理生成快至 8×。
- Google Griffin（线性递归 + 局部注意力）用约 1/6 的 Llama-2 训练数据量
  达到可比效果。
- MOHAWK 蒸馏法把 Transformer 教师蒸馏到 SSM 学生（Phi-Mamba 仅 3B tokens，
  不到原 SOTA Mamba 训练量的 1%），大幅降低架构迁移成本。
- 格局判断（2024 全景报告口径）：Transformer 仍占 74%，替代范式约 22%，
  混合是最有前景方向。

### 对照表：本讲概念 ↔ 工业界实践

| 本讲概念 | 工业界实践（2024–2026） | 关键数字（官方/公开口径） |
|---|---|---|
| Top-\(k\) routing | V3 组级路由：先 top-2 组再组内选 | 256 路由专家 / 激活 8 / 8 组 |
| Shared experts | V3 每层 1 共享专家，承载通用模式 | +7% 性能增益 |
| Fine-grained experts | 细粒度 + 共享成主流配置 | 精度 +12%；较 GShard 参数 -33% |
| Auxiliary loss | V2 三重 aux loss + token 丢弃 → V3 aux-loss-free bias | 均衡率 99.5% 以上 |
| Total vs active | V3：671B total / 约 37B active | 组合空间约 4.4 万亿种 |
| KV cache 压缩 | MHA → GQA → MLA 谱系 | MLA：cache -93.3%、吞吐 +576% |
| Expert parallel | V3 EP64、跨节点 IB + 片内 NVLink | 32-GPU 设备利用率约 98% |
| Hybrid layers | SSM/注意力混合：Griffin、AI21 混合 | Griffin 约 1/6 数据量；生成快至 8× |

**对本讲学习者的启示**：本讲的 selective computation 视角正是工业界主线——
MoE 在参数轴稀疏激活（V3：671B/37B），SSM/混合在时间轴压缩历史
（Griffin、Falcon-Mamba），MLA 压缩 KV cache（-93.3%）。学习时应把每个工业
数字还原成本讲的公式口径（total/active、cache bytes、all-to-all 量级），
并注意"官方口径"数字来自技术报告的最优配置，迁移到自己的场景必须重新做
parameter/FLOP 预算匹配。

## 大厂面试真题与答题框架

**题目 1：为什么 auxiliary-loss-free 偏置比辅助损失好？**
- 考点：负载均衡机制、aux loss 的副作用、选择通路与梯度通路分离。
- 答题框架：
  1. aux loss 目标是鼓励均匀利用，但本质是"拉平"压力；
  2. 副作用：过强会把 specialization 拉向均匀、损害主任务，需要调权重；
  3. aux-loss-free：每专家一个可学习 bias，动态修正路由概率，
     只影响选择、不进损失与梯度通路；
  4. 均衡由动态调整实现而非损失惩罚，specialization 得以保留；
  5. 用数字收尾：V3 官方口径均衡率 99.5% 以上，配序列级损失补偿，
     并摒弃 token 丢弃。
- 加分项：讲清 V2（三重 aux loss + 设备级 token 丢弃）→ V3（bias +
  序列级补偿）的演进逻辑；指出 bias 更新步长/频率是超参。
- 踩坑：说"aux-loss-free 就是没有均衡约束"；把 bias 说成损失项。

**题目 2：MoE 为什么"总参数↑激活参数不变"？代价是什么？**
- 考点：total vs active 解耦、显存/optimizer/通信成本。
- 答题框架：
  1. 公式：\(P_{\rm total}=P_s+EP_e\)，\(P_{\rm active}\approx P_s+kP_e\)；
  2. 增大 \(E\) 提升 capacity，\(k\) 不变则 active FLOPs 不变；
  3. 代价按 total 收费：checkpoint、optimizer states、显存驻留；
  4. 通信：dispatch/combine 两次 all-to-all，量级 \(O(Nkds)\)；
  5. 数字例：V3 671B total / 约 37B active；对比 dense 671B 每 token 全量激活。
- 加分项：推理 server 仍需全量 expert 驻留；低 batch 路由分散增加权重读取；
  组合空间约 4.4 万亿种说明 capacity 提升。
- 踩坑：只报 total 或只报 active；声称推理只需激活参数的显存。

**题目 3：MLA 为什么用低秩压缩 KV？比 GQA 强在哪？**
- 考点：KV cache 结构、MHA→MQA→GQA→MLA 谱系、decode 带宽瓶颈。
- 答题框架：
  1. decode 瓶颈是 KV cache 带宽与容量；
  2. GQA 减少 KV head 数，是"共享"式压缩，质量有折中；
  3. MLA 把 KV 投到低秩表示，cache 存压缩表示而非完整 K/V；
  4. 官方口径：KV cache -93.3%、最大吞吐 +576%；
  5. 代价：attention 计算路径需重推导、kernel 更复杂。
- 加分项：指出谱系的压缩力度递增与实现复杂度递增是对偶的。
- 踩坑：把 MLA 等同于 GQA 极端情形；忽略训练侧结构变化。

**题目 4：为什么细粒度专家 + 共享专家成为主流？**
- 考点：DeepSeekMoE 设计逻辑、specialization 与组合性。
- 答题框架：
  1. 细粒度：大 expert 拆小，同预算下 top-k 组合数指数增长，更灵活；
  2. 共享：通用模式固定走共享 expert，路由 expert 专注 specialization，
     减少重复学习；
  3. 官方口径：细粒度 +12%、共享 +7%；较 GShard 参数 -33%；较 dense
     效率约 7×（0.24B vs 1.89B 激活量级）；
  4. 代价：GEMM 更小、routing metadata 与通信更多，hardware efficiency 下降，
     需要 block-sparse kernel 与 EP 补偿。
- 加分项：V3 256 专家组合空间约 4.4 万亿种；引 MegaBlocks 缓解小 GEMM。
- 踩坑：认为专家越多越好；忽略小 GEMM 利用率损失。

**题目 5：Transformer 会被 SSM 取代吗？**
- 考点：SSM 优劣、hybrid 层配置、格局判断。
- 答题框架：
  1. SSM 优势：固定 state、decode 上下文成本不增、训练可 parallel scan；
  2. 弱点：压缩 state 对精确 recall 不利；
  3. 证据（2024 口径）：Falcon-Mamba 7B 优于同规模 Transformer；AI21 混合
     知识/推理优于 8B Transformer 且生成快至 8×；Griffin 约 1/6 Llama-2
     数据量可比；MOHAWK 蒸馏（Phi-Mamba 仅 3B tokens，不到 SOTA Mamba
     训练量 1%）降低迁移成本；
  4. 格局：Transformer 74% vs 替代约 22%，混合最有前景；
  5. 结论：不是取代，而是按任务混合层配置。
- 加分项：指出"kernel 成熟度"与"算法上限"是两回事。
- 踩坑：用渐近复杂度直接断言优劣；只看单点基准。

**题目 6：token 丢弃为什么被摒弃？**
- 考点：capacity overflow 的语义破坏、均衡策略配套。
- 答题框架：
  1. token-choice 负载无保证 → capacity → overflow 处理；
  2. drop 静默改变模型语义、reroute 引入偏差；
  3. V2 用设备级 token 丢弃；V3 改为 bias 动态均衡 + 序列级损失补偿，
     摒弃 token 丢弃；
  4. dropless 后 straggler 问题仍在，靠 EP 与 kernel 缓解。
- 加分项：提 batch-priority routing 作为容量不足时的中间方案。
- 踩坑：把 drop 率低等同负载均衡好；忽略溢出统计要按层报告。

## 系统设计题

**设计题 1：为 1T token 预算设计 MoE 配置**
- 需求澄清：训练算力预算（GPU 数 × 时长）、质量对标 dense 规模、部署约束
  （节点上限、时延）、EP 拓扑与互连。
- 规模估算：以 V3 为锚点——256 路由专家 + 1 共享、激活 8、671B/约 37B。
  激活对标 dense 30B 级时，每 expert 参数约为
  \((P_{\rm active}-P_{\rm shared-attn})/8\) 量级；1T tokens × 37B 激活 ×
  约 6 FLOPs/参数 ≈ \(2.2\times10^{23}\) FLOPs 量级，除以集群 MFU 反推
  GPU 时。
- 架构：细粒度专家 + 1 共享专家；组级路由（8 组 top-2）或 token-choice +
  aux-loss-free bias；EP 度取决于专家显存与 all-to-all 带宽，参照 V3 EP64、
  跨节点 IB + 片内 NVLink、部署最大 4 节点。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 取舍 |
|---|---|---|---|
| 专家数 | 128（GEMM 更大） | 256（组合空间约 4.4 万亿种） | 利用率 vs 组合性 |
| 共享专家 | 0 | 1（官方口径 +7%） | 激活预算 vs 质量 |
| 均衡 | aux loss（0.01 量级权重） | aux-loss-free bias（均衡率 99.5%+） | 简单 vs 不伤 specialization |
| top-k | 2 | 8 | 通信/激活成本 vs 质量 |

- 评测方案：同预算 dense baseline；报 validation loss、router entropy、
  每 expert token count 分布、all-to-all 时间占比、tokens/s；对专家数与
  共享专家做 controlled ablation。
- 追问预案：负载严重不均 → bias 步长/组级路由；小 GEMM 拖累 MFU →
  block-sparse kernel 或减少专家数；跨节点通信主导 → EP 度与 NVLink/IB
  拓扑对齐（V3 参照：设备利用率约 98%）。

**设计题 2：把 70B dense 迁移到 MoE 的 A/B 方案**
- 需求澄清：质量是否不得低于原 dense 70B？推理时延/吞吐目标？可承受的
  continued pretraining token 预算？
- 规模估算：激活对标 70B、total 按激活的约 18× 放大（参照 V3 671B/37B
  比例），总参数数百 B 量级；continued pretraining 需显著 tokens 恢复质量
  （参照 GQA 从 MHA checkpoint 转换的经验）。
- 架构：
  - A 方案（细粒度）：把 dense FFN 切分为小 expert 初始化 + 1 共享专家
    承载通用模式；官方口径细粒度 +12%、共享 +7%；
  - B 方案（粗粒度）：复制 dense FFN 为 expert 加扰动，GShard 风格，实现
    简单但参数效率低（细粒度口径较 GShard 参数 -33% 可作参照）。
- trade-off 表：

| 维度 | A：细粒度+共享 | B：粗粒度复制 |
|---|---|---|
| 初始化质量 | 切分后需恢复 | 起点即 dense 等价 |
| 参数效率 | 高（口径 -33% vs GShard） | 低 |
| 实现复杂度 | 高（routing metadata 多） | 低 |
| 恢复训练预算 | 较长 | 较短 |

- 评测方案：A/B 同预算比较 validation loss、downstream、推理吞吐与显存；
  记录 router collapse 早期信号（entropy、单 expert token 占比）。
- 追问预案：router collapse → router noise / z-loss / bias 均衡；
  expert undertrain → 延长 continued training、监控每 expert token 数。

**设计题 3：为长上下文 serving 选 KV 压缩方案**
- 需求澄清：上下文长度分布、目标并发、质量红线（retrieval 任务占比）。
- 规模估算：用本讲第 6.1 节公式，\(L=32,T=8192,h_q=32,d_h=128\) bf16：
  MHA 每 request 约 4 GiB；GQA \(h_{kv}=8\) 约 1 GiB；MQA 约 128 MiB；
  MLA 官方口径 cache -93.3%。
- 架构：谱系选择 MHA → GQA → MQA → MLA；或混合层（SSM + 局部注意力，
  参照 Griffin 线性递归 + 局部注意力、AI21 混合生成快至 8×）。
- trade-off 表：

| 方案 | cache 缩减 | 质量 | 实现复杂度 |
|---|---|---|---|
| GQA \(h_{kv}=8\) | 4× | 小损失 | 低 |
| MQA | \(h_q\) 倍 | 有折中 | 低 |
| MLA | 93.3%（官方口径） | 官方口径下吞吐 +576% | 高（kernel 重推导） |
| SSM 混合层 | 固定 state | 精确 recall 弱 | 中（scan kernel） |

- 评测方案：TTFT / inter-token latency、并发容量、needle/retrieval、
  长文档建模 loss；分 prefill 与 decode 报告。
- 追问预案：recall 退化 → 提高混合配置中 attention 层比例；MLA kernel
  不可用 → 退 GQA 并量化 cache。

## 代码实现题

**代码题 1：top-k 组级路由 Gate 前向**
- 题目：实现 V3 风格组级路由：E=256 专家分 8 组、token 先 top-2 组再
  组内选专家，共激活 8 个路由专家。
- 考察点：top-k、离散选择与可微权重分离、组约束实现。

```python
def group_route(router_logits, n_groups=8, topk_groups=2,
                experts_per_group=32, k=8):
    # router_logits: [N, E]，E = n_groups * experts_per_group
    N, E = router_logits.shape
    probs = torch.softmax(router_logits, dim=-1)
    # 组得分取组内概率和（示意性设计，可换 top 内最大值）
    group_score = probs.view(N, n_groups, experts_per_group).sum(-1)  # [N, G]
    top_g = group_score.topk(topk_groups, dim=-1).indices            # [N, 2]
    g_idx = torch.arange(E, device=probs.device) // experts_per_group
    group_allowed = torch.isin(g_idx, top_g.reshape(-1))             # [E]
    masked_probs = probs * group_allowed.unsqueeze(0)                # 仅组内可被选
    topk_w, topk_i = masked_probs.topk(k, dim=-1)                    # [N, k]
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)              # 重归一化
    return topk_i, topk_w
```

- 验收标准：shape 为 [N, k]；全部被选专家落在 top-2 组；权重和为 1；
  \(E=1,k=1\) 时退化为 dense 等价；可在选择前加 bias 且不进梯度。

**代码题 2：auxiliary-loss-free 偏置更新逻辑**
- 题目：实现 per-expert 偏置：只影响选择、不影响损失；过载专家偏置下调、
  欠载上调。
- 考察点：选择通路与梯度通路分离、负载统计驱动更新。

```python
def select_with_bias(router_probs, bias, k=8):
    # router_probs: [N, E]（带梯度）；bias: [E]（不进梯度图）
    with torch.no_grad():
        _, sel_i = (router_probs.detach() + bias).topk(k, dim=-1)
    # 门控权重仍取 router_probs 原值参与前向与反向
    gate_w = router_probs.gather(1, sel_i)
    gate_w = gate_w / gate_w.sum(dim=-1, keepdim=True)
    return sel_i, gate_w

def update_bias(bias, sel_i, capacity_per_expert, lr=1e-3):
    # sel_i: [N, k] 本步实际分配；capacity_per_expert: [E] 目标负载
    with torch.no_grad():
        load = torch.bincount(sel_i.flatten(),
                              minlength=bias.numel()).float()
        bias -= lr * torch.sign(load - capacity_per_expert)
```

- 验收标准：bias 不出现在 autograd 图；负载高于目标的专家 bias 下降、
  低于目标上升；选择通路与权重通路分离；随步数负载均衡率上升。

**代码题 3：MoE 前向（含共享专家与加权聚合）**
- 题目：单卡 MoE 层前向：1 个共享专家 + E 个路由专家 top-k 加权聚合。
- 考察点：dispatch/combine、top-k 权重重归一化、dense-equivalence 测试。

```python
class MoELayer(nn.Module):
    def __init__(self, d, d_ff, E=8, k=2):
        super().__init__()
        self.shared = SwiGLU(d, d_ff)                 # 共享专家全程激活
        self.experts = nn.ModuleList([SwiGLU(d, d_ff) for _ in range(E)])
        self.router = nn.Linear(d, E)
        self.E, self.k = E, k

    def forward(self, x):                             # x: [B, T, d]
        B, T, d = x.shape
        flat = x.reshape(B * T, d)                    # [N, d]
        probs = torch.softmax(self.router(flat), dim=-1)
        w, idx = probs.topk(self.k, dim=-1)           # [N, k]
        w = w / w.sum(dim=-1, keepdim=True)           # top-k 重归一化
        out = self.shared(flat)                       # [N, d]
        for e in range(self.E):
            sel = (idx == e)                          # [N, k]
            tok_mask = sel.any(dim=-1)
            if not tok_mask.any():
                continue
            w_e = (sel.float() * w).sum(dim=-1)       # 每 token 在该专家的权重
            expert = self.experts[e]
            expert_out = expert(flat[tok_mask])
            out[tok_mask] = out[tok_mask] + w_e[tok_mask].unsqueeze(-1) * expert_out
        return out.reshape(B, T, d)
```

- 验收标准：\(E=1,k=1\) 且 router 近均匀时与 dense 等价；输出 shape
  [B, T, d]；每 token 路由权重和为 1；可从 idx 直接统计 expert 负载。

## 17. 结论与本讲小结

Full attention 的价值是动态、精确地访问历史，代价是长上下文的二次计算与线性 KV cache。
Local/sparse attention 限制连接，linear/recurrent 方法压缩历史，hybrid 架构在质量与效率间折中。
MoE 通过 Top-\(k\) routing 解耦 total capacity 与 active compute，却把难题转移到负载均衡、
expert 训练和跨设备通信；shared/fine-grained experts 与 auxiliary-loss-free 的 bias 调节
是当前缓解 routing 病态的主要方向。判断一种稀疏架构，必须同时看质量、active/total 参数、
真实 kernel、内存和通信，而不能只看渐近 FLOPs。

## 参考文献

<a id="ref-1"></a>[1] A. Vaswani et al. “Attention Is All You Need.”
*NeurIPS*, 2017. https://arxiv.org/abs/1706.03762

<a id="ref-2"></a>[2] I. Beltagy, M. E. Peters, A. Cohan. “Longformer:
The Long-Document Transformer.” arXiv:2004.05150, 2020.
https://arxiv.org/abs/2004.05150

<a id="ref-3"></a>[3] M. Zaheer et al. “Big Bird: Transformers for Longer
Sequences.” *NeurIPS*, 2020. https://arxiv.org/abs/2007.14062

<a id="ref-4"></a>[4] N. Kitaev, Ł. Kaiser, A. Levskaya. “Reformer:
The Efficient Transformer.” *ICLR*, 2020. https://arxiv.org/abs/2001.04451

<a id="ref-5"></a>[5] A. Katharopoulos et al. “Transformers are RNNs:
Fast Autoregressive Transformers with Linear Attention.” *ICML*, 2020.
https://arxiv.org/abs/2006.16236

<a id="ref-6"></a>[6] K. Choromanski et al. “Rethinking Attention with Performers.”
*ICLR*, 2021. https://arxiv.org/abs/2009.14794

<a id="ref-7"></a>[7] Y. Sun et al. “Retentive Network: A Successor to
Transformer for Large Language Models.” arXiv:2307.08621, 2023.
https://arxiv.org/abs/2307.08621

<a id="ref-8"></a>[8] A. Gu, T. Dao. “Mamba: Linear-Time Sequence Modeling
with Selective State Spaces.” *COLM*, 2024. https://arxiv.org/abs/2312.00752

<a id="ref-9"></a>[9] N. Shazeer. “Fast Transformer Decoding:
One Write-Head is All You Need.” arXiv:1911.02150, 2019.
https://arxiv.org/abs/1911.02150

<a id="ref-10"></a>[10] J. Ainslie et al. “GQA: Training Generalized
Multi-Query Transformer Models from Multi-Head Checkpoints.” *EMNLP*, 2023.
https://arxiv.org/abs/2305.13245

<a id="ref-11"></a>[11] D. Lepikhin et al. “GShard: Scaling Giant Models
with Conditional Computation and Automatic Sharding.” *ICLR*, 2021.
https://arxiv.org/abs/2006.16668

<a id="ref-12"></a>[12] W. Fedus, B. Zoph, N. Shazeer. “Switch Transformers:
Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.”
*JMLR*, 2022. https://arxiv.org/abs/2101.03961

<a id="ref-13"></a>[13] A. Q. Jiang et al. “Mixtral of Experts.”
arXiv:2401.04088, 2024. https://arxiv.org/abs/2401.04088

<a id="ref-14"></a>[14] T. Gale et al. “MegaBlocks: Efficient Sparse Training
with Mixture-of-Experts.” *MLSys*, 2023. https://arxiv.org/abs/2211.15841

<a id="ref-15"></a>[15] Y. Zhou, T. Lei, H. Liu, et al. “Mixture-of-Experts
with Expert Choice Routing.” *NeurIPS*, 2022.
[link](https://arxiv.org/abs/2202.09368)

<a id="ref-16"></a>[16] D. Dai, C. Deng, C. Zhao, et al. “DeepSeekMoE:
Towards Ultimate Expert Specialization in Mixture-of-Experts Language
Models.” arXiv:2401.06066, 2024. [link](https://arxiv.org/abs/2401.06066)

<a id="ref-17"></a>[17] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré.
“FlashAttention: Fast and Memory-Efficient Exact Attention with
IO-Awareness.” *NeurIPS*, 2022. [link](https://arxiv.org/abs/2205.14135)

<a id="ref-18"></a>[18] T. Dao, A. Gu. “Transformers are SSMs: Generalized
Models and Efficient Algorithms Through Structured State Space Duality.”
*ICML*, 2024. [link](https://arxiv.org/abs/2405.21060)

<a id="ref-19"></a>[19] S. Yang, J. Kautz, A. Hatamizadeh. “Gated Delta
Networks: Improving Mamba2 with Delta Rule.” *ICML*, 2025.
[link](https://arxiv.org/abs/2412.06464)

<a id="ref-20"></a>[20] N. Shazeer et al. “Outrageously Large Neural Networks:
The Sparsely-Gated Mixture-of-Experts Layer.” *ICLR*, 2017.
https://arxiv.org/abs/1701.06538

<a id="ref-21"></a>[21] DeepSeek-AI. “DeepSeek-V3 Technical Report.”
arXiv:2412.19437, 2024. [link](https://arxiv.org/abs/2412.19437)

## 延伸阅读与复现材料

- Stanford CS336, [Spring 2026 Lecture 4](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)
- [Tokenization & Basics 主题导航](../experiments/topics/tokenization-and-basics.md)
- [Systems 主题导航](../experiments/topics/systems.md)
- [Lecture 10 — Inference](lecture-10-inference.md)
- Dao, Gu, [Transformers are SSMs / SSD](https://arxiv.org/abs/2405.21060)
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)（访问日期 2026-10-04）
- DeepSeek-AI, [DeepSeek-V2（MLA）](https://arxiv.org/abs/2405.04434)（访问日期 2026-10-04）
- Jiang et al., [Mixtral of Experts](https://arxiv.org/abs/2401.04088)（访问日期 2026-10-04）
- De et al., [Griffin: Mixing Gated Linear Recurrences with Local Attention](https://arxiv.org/abs/2402.19427)（访问日期 2026-10-04）
