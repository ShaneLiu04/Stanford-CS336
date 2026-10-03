---
title: "Lecture 04 — Attention Alternatives & Mixture of Experts"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-04-08"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf"
  - "../assignments/assignment1-basics/"
---

# Lecture 04 — Attention Alternatives 与 Mixture of Experts：选择性计算的统一视角

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述，非课程提交
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
| Full causal attention | `assignments/assignment1-basics/cs336_basics/model.py`：`scaled_dot_product_attention` | 替换前先做输出/causal leakage baseline |
| MHA Q/K/V shape | 同文件：`CausalMultiHeadSelfAttention` | GQA 需拆 `num_heads` 与 `num_kv_heads` |
| Dense FFN | 同文件：`SwiGLU` | MoE expert 可复用相同 FFN 接口 |
| Transformer block | 同文件：`TransformerBlock` | 仅把部分层的 `ffn` 换成 routed module |
| Resource accounting | `.../report/main.tex` 第 3.3、4.3 节 | 加入 active/total 参数、routing/通信成本 |
| Shape tests | `.../tests/test_model.py`、`tests/adapters.py` | 扩展测试不要修改官方 adapter 契约 |
| Training metrics | `.../cs336_basics/training.py` | 追加 expert load/drop/router entropy |
| A2 systems 入口 | `assignments/assignment2-systems/` | 区分 exact attention kernel 与架构近似 |

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
| 长距 retrieval 失败 / 长程回忆差 | connectivity/state capacity | window 不可达、状态饱和、核近似误差 |
| GQA 显存未下降 | tensor storage/expand | 物理复制 K/V heads |
| decode 慢但 prefill 快 | KV bytes/request | memory bandwidth / small GEMM |
| KV cache 显存过大 | head 分组数（MQA/GQA） | KV head 未压缩或未量化 |
| attention 输出数值错 | causal mask 与 softmax 缩放 | mask 错位、\(1/\sqrt{d_k}\) 缺失 |
| fused kernel 与参考不符 | online softmax 累计量 | 区分数值误差与实现错误（LSE 复用） |
| 稀疏化后质量骤降 | 稀疏模式与任务匹配 | 局部窗口过小、模式对检索任务不适配 |
| router 迅速单 expert / 训练 loss 尖峰 | token counts/entropy / aux loss | positive feedback、负载均衡约束过弱、router z-loss 缺失 |
| expert 负载严重不均 | 路由分布与溢出率统计 | aux loss 系数不合适、token 分布漂移 |
| MoE loss 正常但吞吐低 / 推理不见加速 | expert batch/A2A / 容量因子 | 小 GEMM、padding、straggler、容量溢出、未 overlap |
| token 被静默丢弃 | capacity/drop rate | capacity factor 太低 |
| balance loss 降、主 loss 升 | coefficient | 过度均匀抑制 specialization |
| 多卡偶发 hang | all-to-all splits | rank 间 token-count metadata 不一致 |

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

### 16.1 Construct validity

- theoretical sparsity 不等于 executed sparse FLOPs；
- active parameters 不等于 model memory/optimizer state；
- perplexity 不充分衡量 retrieval、reasoning、tail latency 或 energy；
- needle benchmark 不代表自然长文档建模；
- router entropy 高不一定 load 平衡，token counts 均匀也不保证 expert specialization。

### 16.2 Internal validity

- 方法若使用不同 parameter/FLOP/tuning budget，无法归因架构本身；
- capacity drop、padding 和 reroute 若未记录，会静默改变 objective；
- sparse/SSM kernel maturity 差异可能主导 wall-clock；
- MoE 每 expert 实际 tokens 不同，等总 tokens 不等每参数训练充分；
- checkpoint conversion 与 from-scratch training 不能直接混比。

### 16.3 External validity

- synthetic retrieval、固定 context 的结论未必迁移到 code、dialogue 或 multimodal；
- 单机 NVLink 结果不代表多节点 all-to-all；
- training throughput 最佳方法未必 decode/serving 最佳；
- 小模型 router behavior 未必外推到百 experts/billion parameters；
- 当前实现成熟度不应被误解为算法上限。

### 16.4 推荐的学术表述

避免：“Linear attention 比 Transformer 更高效。”
推荐：“在固定模型规模、序列分布与 GPU 上，该 linear-attention kernel 在 \(T\ge X\)
取得 Y× training throughput，并以 Z 的 retrieval degradation 为代价；短序列与 decode
regime 的结论不同。”

## 17. 面试备考（Interview Prep）

> 本讲是「高效注意力 + MoE」的面试重灾区：面试官常从「FlashAttention 为什么快」切入，追到
> 「MQA/GQA 的 KV cache 显存」「MoE 的 total vs active 参数」「router collapse 怎么解决」。
> 关键是区分 **exact 系统优化（改 IO 不改数学）** 与 **架构近似（改连接/状态）**，别把
> 渐近复杂度当成实际速度。下面按「一页速览 → 高频题 → 手撕 → 追问」四层组织。

### 17.1 一页速览卡（面试前 1 分钟）

**核心主张**：attention alternatives 与 MoE 都在做 selective computation——前者稀疏/压缩
「看多少历史」，后者稀疏「用多少参数」，都以近似、路由、状态管理与通信为代价。

**必背数字与公式**

- KV cache（单 request）：\(M_{\rm KV}=2\,L\,T\,h_{kv}\,d_h\,s\)；GQA 缩减比 \(h_q/h_{kv}\)。
- MoE：\(P_{\rm total}=P_s+EP_e\)，\(P_{\rm active/token}\approx P_s+kP_e\)（如 671B total / 37B active）。
- capacity \(C=\lceil c\,\frac{Nk}{E}\rceil\)；balance loss \(\mathcal L=\alpha E\sum_e f_e P_e\)。
- FlashAttention 仍是 \(O(T^2d)\) FLOPs，只把 HBM 读写降到 \(O(N)\)。
- linear attention state \(S_t=S_{t-1}+\phi(k_t)v_t^\top\)，state 内存 \(O(rd_v)\)。

**三句话答高频**

1. FlashAttention 是 exact 优化（不减少 FLOPs，只减 IO）；local/linear/SSM 是架构近似（改连接或状态）。
2. MQA/GQA 只优化 decode 的 KV cache 带宽与容量，不消除训练的 \(T^2\) 交互。
3. MoE 用 total 参数换 active compute，代价是全量存储、负载均衡与 all-to-all 通信。

### 17.2 高频面试题与答题框架

**Q1：FlashAttention 快在哪？为什么不减少 FLOPs？**

- **本质**：它是 exact systems optimization——数学结果与 full softmax attention 完全一致。
- **为什么快**：tiling + online softmax 分块更新 row max 与归一化分母，避免把 \(T^2\) 的 score 矩阵写回 HBM；显存与 IO 从 \(O(T^2)\) 降到 \(O(N)\)。
- **边界**：算术量仍是 \(O(T^2d)\)；训练 IO 改善了，但 decode 仍需读取历史 KV，不解决 decode 带宽问题。

**Q2：MQA / GQA 解决什么？KV cache 显存怎么算？**

- **解决**：decode 阶段 KV cache 的显存与带宽——让多个 query heads 共享 K/V，压缩 cache 容量。
- **公式**：MHA 每层 `2BT·h_q·d_h` elements，GQA/MQA 降到 `2BT·h_kv·d_h`，缩减比 \(h_q/h_{kv}\)。
- **代价**：K/V head 少可能略损质量；GQA 是 MHA 质量与 MQA 成本间的折中（Ainslie et al. 2023）。

**Q3：具体算一下：一个 7B 模型在 8K 上下文下的 KV cache 显存？**

- 设 `L=32, T=8192, h_q=32, d_h=128, bf16(s=2)`：
- MHA（`h_kv=32`）：\(2\times32\times8192\times32\times128\times2\approx4.3\text{ GiB}\) / request。
- GQA（`h_kv=8`）：约 `1.07 GiB`；MQA（`h_kv=1`）：约 `134 MiB`。
- 结论：KV heads 数直接决定可并发 requests 数；GQA/MQA 是 serving capacity 的关键杠杆。

**Q4：MoE 如何解耦「参数量」与「计算量」？**

- **定义**：\(P_{\rm total}=P_s+EP_e\)（存储/optimizer 按 total），\(P_{\rm active}\approx P_s+kP_e\)（forward FLOPs 按 active）。
- **例子**：`E=64, k=2` → expert 总参数是单 expert 的 64 倍，但每 token 只调用 2 个；DeepSeek-V3 671B total / 37B active。
- **陷阱**：不能只报一个数——total 决定 checkpoint/优化器/存储，active 决定每 token FLOPs，两个都要给。

**Q5：MoE 的 load balancing 怎么做？router collapse 是什么？**

- **问题**：Top-\(k\) token-choice 让 token 自由选 expert，热门 expert 过载 → 正反馈「更多 token → 学更快 → 更被偏好」→ collapse。
- **缓解**：auxiliary balance loss \(\alpha E\sum f_e P_e\)、capacity factor 截断、router z-loss、expert-choice routing、shared experts、DeepSeek-V3 的 auxiliary-loss-free bias。
- **陷阱**：balance loss 过强会压制真实 specialization；负载均匀 ≠ expert 各司其职。

**Q6：linear attention 的原理与代价？**

- **原理**：把 softmax 相似度写成 kernel 内积 \(\phi(q)^\top\phi(k)\)，用结合律把 \((QK^\top)V\) 改写为 \(Q(K^\top V)\)，维护状态 \(S_t=S_{t-1}+\phi(k_t)v_t^\top\)，无需显式 \(T^2\)。
- **复杂度**：每 token \(O(rd_v)\)，state 内存 \(O(rd_v)\) 不随上下文增长。
- **代价**：固定状态压缩任意长历史，精确 retrieval 受损；并行 scan 若实现差，理论 \(O(T)\) 反而不如 fused \(O(T^2)\) 快。

**Q7：SSM / Mamba 的核心思想？**

- 连续 SSM \(\dot h=Ah+Bx, y=Ch\) 离散化得 \(h_t=\bar A h_{t-1}+\bar B x_t\)；固定 state 让 decode 上下文成本近常数。
- Mamba 的关键：让 \(\bar A,\bar B,C\) 依赖输入（selective），配合高效 GPU scan kernel，兼顾并行训练与 recurrent 推理。
- **代价**：固定状态是 compressed memory，精确 copy/retrieval 不如 attention 的 addressable memory。

**Q8：稀疏注意力（sliding window）为什么不保证加速？**

- sliding window 把计算降到 \(O(Twd)\)，但只有当 kernel 真正跳过 masked block 时才省算力。
- 若构造 dense \(T\times T\) 再 mask 大部分位置，FLOPs/内存几乎没省；sparse gather 不规则还会引入 indexing/launch overhead。
- 且单层 receptive field 只有 \(w\)，堆 \(L\) 层最远传播 \(L(w-1)\)，长程检索能力受限。

**Q9：MoE 的通信开销（all-to-all）？**

- expert parallelism 下，一次 MoE 层 = router → pack → all-to-all dispatch → expert GEMM → all-to-all combine。
- 通信量 \(\sim Nkds\)（routed activation），往返各一次；实际延迟受拓扑、token 分布、消息碎片与 straggler 影响。
- 最忙 expert 决定尾延迟；理论 expert FLOPs 与 dense 相同时，也可能因 per-expert 小 batch 降低 GEMM utilization。

**Q10：token-choice 与 expert-choice routing 的区别？**

- **token-choice**：每个 token 选 top-k expert，语义直观，但 expert load 无保证（需 capacity/aux loss 约束）。
- **expert-choice**：每个 expert 选固定数量 token，负载天然均衡，但 token 可能被 0 个或多个 expert 选中。
- 折中还有 batch-priority routing（容量不足优先保留高 router score 的 assignment）。

### 17.3 手撕要点（KV cache 与 linear attention）

面试常让「算 KV cache」或「写 linear attention 的状态更新」：

```python
# 1. KV cache 显存（字节）
def kv_cache_bytes(L, T, h_kv, d_h, dtype_bytes):
    return 2 * L * T * h_kv * d_h * dtype_bytes   # K 和 V 各一份

# 2. linear attention 的 recurrent 更新
import torch
def linear_attn_step(S, z, phi_k, v):
    S = S + phi_k.unsqueeze(-1) * v.unsqueeze(-2)  # [B,h,r,d_v] += 外积
    z = z + phi_k                                  # [B,h,r] 归一化分母
    return S, z
def linear_attn_out(q, S, z, phi):
    y = (phi(q) @ S) / (phi(q) @ z.unsqueeze(-1) + 1e-6)
    return y
```

**三个必踩坑**

1. **KV cache 别忘了乘 2**：K 和 V 各一份；bf16 时 `dtype_bytes=2`。
2. **GQA 用 group 映射而非物理复制**：否则显存节省被物理复制抵消。
3. **linear attention 的 denominator 会漂移**：监控 \(\|S\|\) 与 denominator quantiles，防止 NaN。

### 17.4 高频追问与陷阱

| 追问 | 正确方向 |
| --- | --- |
| FlashAttention 和 linear attention 一样吗？ | 否，FA 保持 exact \(T^2\) 算术，linear 改公式/状态 |
| GQA 能加速训练吗？ | 否，主要降 decode KV cache 带宽，训练 FLOPs 改善小 |
| MoE 报 671B 还是 37B？ | 两个都要报：total 决定存储/优化器，active 决定每 token FLOPs |
| balance loss 越大越好吗？ | 否，过强压制 specialization，主任务受损 |
| recurrent 状态能记住无限历史吗？ | 否，固定维状态必然有信息瓶颈 |
| 稀疏 FLOPs 一定更快吗？ | 否，没有稀疏 kernel、irregular gather、小 GEMM 都会抵消 |
| MoE 推理只付 active 权重吗？ | 否，所有 experts 需驻留显存或跨层/设备取权重 |

## 18. 结论与本讲小结

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
