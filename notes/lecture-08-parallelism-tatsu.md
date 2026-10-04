---
title: "Lecture 08 — TP, PP, FSDP & Multi-Dimensional Parallelism"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-04-22"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf"
  - "../experiments/topics/systems.md"
  - "../assignments/spring2026/assignment2-systems/"
---

# Lecture 08 — TP、PP、FSDP 与多维并行：容量、通信与拓扑的联合设计

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
- 适用对象：自学者、分布式系统工程师与研究者

## 摘要

当模型参数、optimizer state、activation 或单步计算超过单设备能力时，需要沿数据、张量、
层、序列或 experts 等不同维度切分。不同 parallelism 改变的不仅是存储位置，还包括计算图、
collective 类型、同步频率、数值顺序与 checkpoint 语义。本文系统推导 Tensor Parallelism
（TP）的 column/row sharding、Pipeline Parallelism（PP）的 microbatch schedule 与 bubble、
ZeRO/FSDP 的 parameter/gradient/optimizer state lifecycle，以及 FSDP×TP 的二维组合。
进一步讨论 sequence/context/expert parallelism、DeviceMesh/topology mapping、prefetch、
mixed precision、distributed checkpoint、fault tolerance 与自动并行 cost model。在工程层面，
本文给出 memory/communication 账本、deadlock 排查和 profiler protocol；在研究层面，强调
fixed-quality scaling、Pareto frontier 与跨拓扑效度边界。

**关键词：** Tensor Parallelism；Pipeline Parallelism；ZeRO；FSDP；Device Mesh；
Sequence Parallelism；Context Parallelism；3D Parallelism；Distributed Checkpoint

## 本文贡献

1. 统一 DP/TP/PP/FSDP/CP/EP 的 sharding axis 与 collective 语义；
2. 推导 Megatron-style MLP/attention TP 和 pipeline bubble；
3. 对 ZeRO stages 与 FSDP parameter lifecycle 做精确 memory/communication accounting；
4. 解释 2D/3D parallelism 的 topology-aware device mesh 设计；
5. 提供工程诊断、实验矩阵、效度威胁与论文式性能报告规范。

## 学习目标

1. 分清 data、tensor、pipeline 与 fully-sharded data parallel 切的是哪个维度；
2. 为每种策略写出局部 tensor shape、常驻内存、collective 和通信量；
3. 理解 column/row-parallel linear、pipeline bubble、FSDP 参数生命周期；
4. 在给定 GPU 数与拓扑下组合 FSDP×TP，而不是追求单一并行方式。

## 先修知识

- Lecture 07：DDP、collective 通信量模型与 ZeRO 谱系（本讲 ZeRO/FSDP 的直接前置）。
- Lecture 04：MQA/GQA 与 KV cache（并行策略与注意力结构的耦合）。
- Lecture 02：参数/激活显存账本，用于各并行的显存-通信权衡。

## 相关工作与并行谱系

Megatron-LM 将 Transformer 的 attention/MLP 按 hidden/head dimensions 做 tensor
parallel，并通过 column/row pairing 减少 collectives [[1]](#ref-1)。GPipe 用 microbatch
pipeline schedule 扩展深模型 [[2]](#ref-2)，PipeDream 系列进一步研究 1F1B、weight
stashing 与 memory-efficient pipeline schedules [[3]](#ref-3)[[4]](#ref-4)。

ZeRO 将 optimizer、gradient、parameter state 分阶段分片 [[5]](#ref-5)，FSDP 把该思想
集成到 PyTorch module/autograd/checkpoint 生态 [[6]](#ref-6)。大规模训练通常组合 data、
tensor、pipeline 和 sequence parallelism；Megatron-LM 的系统研究表明 topology-aware 3D
mapping 与通信/计算 overlap 对千卡效率至关重要 [[7]](#ref-7)。Mesh TensorFlow、GShard、
Alpa 等工作则把 device mesh、tensor layout 或自动 cost model 提升为编译/规划问题
[[8]](#ref-8)[[9]](#ref-9)[[10]](#ref-10)。

这些方法解决不同 bottleneck：TP 降低单层宽度，PP 切深度，FSDP 切状态，context parallel
切序列。不存在不依赖模型、batch、context 和 topology 的“最佳并行策略”。

## 1. 一张选择图

| 策略 | 切分维度 | 主要通信 | 解决的主要问题 |
|---|---|---|---|
| DDP | batch | gradient all-reduce | 增加数据吞吐 |
| FSDP/ZeRO | batch + model states | parameter all-gather、gradient reduce-scatter | 参数状态容量 |
| TP | hidden/head/FFN width | 每层 all-reduce/all-gather/reduce-scatter | 单层太宽 |
| PP | layer/depth | stage 间 activation/gradient P2P | 模型太深 |

策略并非互斥。总 GPU 数常分解为

\[
G=d\times t\times p,
\]

分别是 data/FSDP、tensor、pipeline parallel degree。若还有 expert/context parallel，会继续增加网格维度。

### 1.1 Sharding axis 决定 communication

| 维度 | 被切 tensor/工作 | 典型 collective |
|---|---|---|
| data/FSDP | batch、model states | all-reduce / all-gather / reduce-scatter |
| tensor | hidden、heads、FFN | all-reduce / all-gather / reduce-scatter |
| sequence parallel | non-TP region 的 sequence | all-gather / reduce-scatter |
| context parallel | attention sequence/KV | ring P2P / all-to-all |
| pipeline | layers | activation/gradient P2P |
| expert | experts/tokens | all-to-all |

选择并行方式前先找“最大不可放下对象”：parameters、optimizer、activation、single-layer compute、
KV/context 或 expert set。错误地沿无关维度切分只会增加通信。

### 1.2 Local shape 是最可靠的推理工具

对每个 tensor 标记 global shape、mesh dimensions、placement（replicate/shard/partial）。
例如 `[B,T,d]` 在 data×tensor mesh 上可能是 `[B/dp,T,d/tp]`；若某 matmul 产生 partial
sum，就必须在使用前 reduce。只记“用了 TP=8”不足以审计正确性。

## 2. Tensor Parallelism（TP）

设 linear：

\[
Y=XW,\quad X\in\mathbb{R}^{M\times K},\
W\in\mathbb{R}^{K\times N},\
Y\in\mathbb{R}^{M\times N}.
\]

### Column-parallel linear

按输出维切 \(W=[W_0,\ldots,W_{t-1}]\)，每 rank：

\[
W_r:[K,N/t],\qquad Y_r=XW_r:[M,N/t].
\]

若下一个操作接受分片 activation，可保持 \(Y_r\) 而不通信；若需要完整 \(Y\)，做 all-gather，结果大小 \(MN\) 元素。backward 中 \(dX=\sum_r dY_rW_r^\top\)，若需要复制的 \(dX\)，需 all-reduce。

### Row-parallel linear

按输入维切

\[
W_r:[K/t,N],\qquad X_r:[M,K/t].
\]

每 rank 产生 partial output：

\[
\widetilde Y_r=X_rW_r:[M,N],\qquad
Y=\sum_r\widetilde Y_r.
\]

forward 需要 all-reduce，或 reduce-scatter 后保持输出分片。backward 对 \(dX_r=dY W_r^\top\) 可自然保持分片。

### MLP 的经典配对

对 \(XW_1\rightarrow\phi\rightarrow W_2\)，将扩张层 \(W_1\) column-parallel，使中间 \(d_{\mathrm{ff}}/t\) 保持分片；收缩层 \(W_2\) row-parallel，最后规约 partial output。中间 activation 不必 all-gather，通常每个 Transformer block 只需少数大 collective。

设 tokens \(T=B N\)、model width \(h\)。若输出复制，row-parallel 规约的 activation tensor 为 \(Th\) 元素；ring all-reduce 每 rank 网络量约

\[
2\frac{t-1}{t}Ths_a\ \text{bytes},
\]

其中 \(s_a\) 是 activation 每元素 bytes。TP 通信随 tokens 增长且每层发生，因此要求高带宽、低延迟互联。

### Attention 中的 TP

若 heads 数可被 \(t\) 整除，每 rank 持有 \(H/t\) 个 heads，Q/K/V 与 attention 内部计算天然局部。output projection 通常 row-parallel，再规约。GQA/MQA 的 KV head 数可能小于 TP degree，需复制 KV heads 或采用不同切分；“hidden size 可整除”并不足以保证可行。

### Sequence Parallelism

Megatron-style TP 中 LayerNorm/Dropout/residual 等非 TP 算子若保持 `[B,T,h]` 复制，会浪费
activation memory。Sequence Parallelism 在这些区域沿 token 维切分
[[11]](#ref-11)：

```text
TP region:       hidden/head shard
reduce-scatter
SP region:       token shard, full hidden locally
all-gather
next TP region
```

它不等同于 long-context Context Parallelism：SP 主要消除 TP group 内 replicated activations；
CP 则让 attention sequence/KV 本身跨设备。

### Vocab-parallel embedding 与 cross-entropy

词表 \(V\) 可沿 TP ranks 切分，每 rank 保存 \(V/t\) rows。Embedding lookup 需要判断 token
是否落在本地 shard，再 all-reduce partial embeddings；LM head 产生 vocab-sharded logits。
Cross-entropy 可通过 distributed max/sum 和 target-logit reduction 计算，无需 all-gather
完整 `[B,T,V]`，避免大词表通信和显存。

## 3. Pipeline Parallelism（PP）

把 \(L\) 层切成 \(p\) 个连续 stages，每个 stage 只保存约 \(L/p\) 层。相邻 stages 传 activation，backward 反向传 activation gradient。

若 microbatch activation shape 为 \([b_\mu,N,h]\)，每个 stage boundary 的单向消息量约

\[
b_\mu N h s_a.
\]

通信不包含整层参数，能跨较慢链路；但计算存在 pipeline bubble。

### Microbatch 与 bubble

把一个 minibatch 切为 \(m\) 个 microbatches。仅 forward 的 GPipe 式流水，理想利用率约：

\[
U\approx\frac{m}{m+p-1},\qquad
\text{bubble fraction}\approx\frac{p-1}{m+p-1}.
\]

训练还要调度 forward/backward。1F1B 在 warmup 后交替执行，可降低峰值 activation 和 bubble，但公式取决于 schedule、是否 interleaved、stage 时间是否均衡。

增大 \(m\) 减少 bubble，却会缩小每个 microbatch 的 GEMM，可能降低 GPU 利用率，并增加调度/P2P 开销。

### Stage balance

按“层数相等”切分不一定平衡：embedding、LM head、不同 attention/MLP shape、checkpoint 和网络链路都会造成 stage skew。step 速度由最慢 stage 决定：

\[
T_{\text{steady}}\gtrsim m\cdot\max_s T_s.
\]

需用实测 per-layer time 和 memory 做 partition。

### 3.1 GPipe、1F1B 与 interleaving

- **GPipe schedule**：先所有 forward，再所有 backward；简单、weight version 一致，
  但需保存更多 microbatch activations。
- **1F1B**：warmup 后 forward/backward 交替；降低 activation peak，常用于同步 pipeline。
- **Interleaved 1F1B**：每 rank 持多个 virtual stages，缩短 bubble，但增加 P2P/调度复杂度。
- **PipeDream-style async**：减少 bubble，但不同 microbatches 可能看到不同 weight versions，
  需要 weight stashing 或延迟更新 [[3]](#ref-3)。

同步训练通常在一个 minibatch 的所有 microbatches backward 完成后统一 optimizer step，
避免 weight inconsistency。Schedule correctness 需检查 activation/gradient tag、microbatch ID、
forward/backward pairing 与 RNG。

### 3.2 Activation transport 与 recomputation

Stage boundary 需保存或发送 activation。Activation checkpointing 可在 stage 内重算；
pipeline 还可选择发送低精度 activation、recompute boundary 或 offload。每种方案改变：

- P2P bytes；
- stage memory；
- recompute FLOPs；
- bubble/overlap；
- numerical error。

Stage balance 应同时考虑 recompute 与 communication，不只 forward layer time。

### 3.3 Pipeline 与 global batch

Global batch 常为

\[
B_{\mathrm{global}}=b_\mu\times m\times d,
\]

其中 \(d\) 是 data-parallel degree。为减 bubble 增大 \(m\) 会改变 global batch，除非同步减小
microbatch size；但过小 \(b_\mu\) 又降低 GEMM efficiency。因此 pipeline schedule 与 optimizer
batch recipe 紧密耦合。

## 4. ZeRO 与 FSDP

DDP 的内存冗余来自每 rank 都保存完整状态。粗略分级：

- ZeRO-1：分片 optimizer states；
- ZeRO-2：再分片 gradients；
- ZeRO-3/FSDP：再分片 parameters。

设参数元素 \(P\)，FP32 参数/梯度和两份 Adam moments。忽略 activation/buffer：

\[
\text{DDP}\approx16P\text{ bytes/rank},
\]

\[
\text{ZeRO-1}\approx 8P+\frac{8P}{f},
\]

\[
\text{ZeRO-2}\approx 4P+\frac{12P}{f},
\]

\[
\text{ZeRO-3/FSDP}\approx\frac{16P}{f}.
\]

具体实现若有 BF16 compute copy、FP32 master weight、padding、prefetch 和 flat buffer，账本会不同，不能机械套公式。

### 状态仍放不下：Offload 与 ZeRO-Infinity

当 \(16P/f\) 仍超出设备显存时，ZeRO-Infinity 进一步把 optimizer states、parameters、
activations 按需分层卸载到 CPU memory 甚至 NVMe，用 bandwidth-centric partitioning
与异步 prefetch 隐藏 PCIe/网络延迟 [[12]](#ref-12)。其代价是：CPU/NVMe 带宽成为新
bottleneck、failure 恢复更复杂、throughput 显著低于纯 GPU 训练。offload 是“能训”
与“训得快”之间的权衡，评估应报告 time-to-quality 与 energy，而不是只报显存数字。

### FSDP 生命周期

对每个 FSDP unit：

1. 常驻 FP32 parameter shard，约为 full parameter 的 \(1/f\)；
2. forward 前 all-gather full compute parameter；
3. 执行该 unit；
4. 按策略 reshard/free full parameter；
5. backward 前必要时再次 all-gather；
6. 计算 full gradient 后 reduce-scatter；
7. optimizer 更新 local shard。

若一个 unit 参数大小为 \(S\) bytes：

- 一次 all-gather 每 rank 发送/接收量量级 \((f-1)S/f\)；
- 一次 reduce-scatter 同量级；
- 若 forward 后 reshard，backward 还要再 all-gather；
- 因此训练每 step 常见总量约 \(3(f-1)S/f\)，具体取决于保留策略。

DDP ring all-reduce 约 \(2(f-1)S/f\)。FSDP 常以更多/更细通信换显存；若 full parameter 从 forward 保留到 backward，可少一次 gather，却提高峰值内存。

### Prefetch 与 wrapping

- unit 太大：full parameter 峰值高，通信启动晚；
- unit 太小：collective 太多，被 latency 支配；
- backward prefetch 可把下一 unit all-gather 与当前 unit backward 重叠；
- 同时 materialize 多个 full units 会提高峰值；
- wrapping 应尊重 tied weights、模块依赖和参数共享。

### Flat parameters、original parameters 与 alias

FSDP 可把多个 parameters flatten/pad 成通信友好的 shard；这减少 collective metadata，
但改变 parameter view/lifetime。`use_orig_params` 类模式保留用户可见 Parameter 语义，
便于 per-parameter optimizer、冻结与 tied weights，却增加实现复杂度。

Tied embedding/LM head 若跨 FSDP units，会破坏 alias 或重复 all-gather。Wrap policy 需要让
共享 parameter 属于兼容 unit，并在 load checkpoint 后重新验证 identity/storage。

### Mixed precision 三种 dtype

至少区分：

- parameter/master dtype；
- compute/communication parameter dtype；
- reduce gradient dtype。

BF16 all-gather 降低 bandwidth，但 optimizer 可能仍需 FP32 shard；reduce-scatter 用低精度会引入
跨 rank 累加误差。Buffer dtype（如 norm statistics）还可能独立配置。

### Distributed checkpoint

Checkpoint 可保存 full、local shard 或 sharded state dict。Full 易迁移但 rank0 memory/IO 高；
sharded 并行写入高效，但恢复需要 metadata、world-size reshard 与原始 parameter mapping。
完整恢复还包括 optimizer shards、scheduler、RNG、sampler、global step。只保存 model shards
不能严格 resume。

## 5. 仓库教学型 FSDP

实现：`assignments/spring2026/assignment2-systems/cs336_systems/fsdp.py`

它只分片课程自定义 `Linear`/`Embedding` weights：

- 参数 padding 后切成等长 FP32 master shards；
- forward pre-hook 用 all-gather materialize full weight；
- 可用 BF16/FP16 compute/communication copy；
- `finish_gradient_synchronization()` 同步 gradient 并恢复 shard；
- replicated 参数仍做 all-reduce average。

必须注意它是教学实现，不等价于生产 FSDP：

- gathered parameter 会保留到 backward 完成；
- 当前代码对 full gradient 先 all-reduce 再切片，而非真正 reduce-scatter；
- 不 overlap reduce-scatter/backward；
- 因此正确性可验证，但峰值和通信性能不能代表生产系统。

这也是阅读实现的重要原则：根据实际 collective 和生命周期核算，而不是根据类名推断性质。

## 6. 二维 FSDP × TP

令总 GPU \(G=f t\)。把 rank 排成 \([f,t]\) 网格：

- **TP group**：同一 data replica 内宽度切分，大小 \(t\)；
- **FSDP group**：跨 replicas 对同一 TP shard 做状态分片，大小 \(f\)。

每 rank 持有约 \(1/(ft)\) 的参数状态；但通信分两类：

1. TP collective：每层、高频、依赖链短，消息多为 activation；
2. FSDP collective：按 unit gather/scatter parameter/gradient shard。

### 粗略通信核算

若完整模型参数 \(P\) elements，TP 后每 rank 对应参数 shard 为 \(P/t\)。FSDP 在大小 \(f\) 的组内处理该 shard：

\[
V_{\text{FSDP,rank}}
\approx c\frac{f-1}{f}\frac{P}{t}s_p,
\]

其中 \(c\) 常为 2 或 3，取决于 parameter 是否在 backward 前重新 gather。

TP 以每层 activation 为主。若每层有 \(k\) 次大小 \(A_l\) 的 ring all-reduce：

\[
V_{\text{TP,rank}}
\approx\sum_l k\cdot2\frac{t-1}{t}A_ls_a.
\]

因此 TP degree 增大虽继续分参数，却提高高频 activation communication；FSDP degree 增大主要改变参数通信和 data-parallel compute 分摊。

### 拓扑映射

通常把 TP group 放在 NVLink/NVSwitch 等最快域内，把 FSDP/DP 跨节点，因为 TP 每层都在关键路径上。若一节点 8 GPU、总 64 GPU，一个常见候选是 \(t=8,f=8\)，但仍需比较：

- 模型单层是否能在 \(t<8\) 放下；
- 跨节点 bandwidth 是否承受 FSDP；
- batch 是否足以供 \(f\) 个 data shards；
- heads、hidden、FFN dimensions 是否可整除；
- activation 与 parameter 哪类通信占主导。

不要默认“节点内一定全做 TP”：对小模型，较小 TP 可减少通信并提高 GEMM shape。

### Context Parallelism（CP）

超长 sequence 的 activation/KV 单卡放不下时，可沿 \(T\) 切分。Attention 仍需 query shard
访问全体 K/V，可采用 ring attention：K/V blocks 沿 CP ranks 轮转，每步做本地 attention tile，
用 online softmax 合并统计 [[13]](#ref-13)。通信约线性于 KV bytes，计算仍是全 attention 的 \(T^2\) 主项，
但 memory 分摊。

CP 与 sequence parallel 不同：CP 切 attention context 本身；SP 主要切 LayerNorm/Dropout 等
replicated activation。USP 进一步把 SP 与 ring-attention 风格的 CP 统一为可组合的
sequence parallelism 框架 [[14]](#ref-14)。Causal ring 需正确处理 global positions 与 block mask。

### 3D/4D parallelism

大训练常有 mesh `[DP/FSDP, PP, TP, CP]`，MoE 再加 EP。总 world size：

\[
G=d\times p\times t\times c\;(\times e).
\]

每一维必须映射到 topology：高频 TP/CP 通常放 node 内；PP P2P 可跨相邻 nodes；
DP/FSDP collective 粒度大但频率较低。映射问题是 graph partitioning/cost modeling，
不能只按 rank 连续编号。

## 7. 何时选择哪种并行

- 模型能放单卡、目标是吞吐：先 DDP；
- 参数状态放不下：FSDP/ZeRO；
- 单层本身放不下或 GEMM 太大：TP；
- 深度导致整模型放不下，且跨 stage activation 可接受：PP；
- 大规模训练：先满足容量，再把高频通信映射到最快链路，组合 2D/3D。

决策时同时比较：

\[
T_{\text{step}}\approx
\max(T_{\text{compute}},T_{\text{overlapped comm}})
+T_{\text{exposed comm}}+T_{\text{bubble}}.
\]

只比较总通信 bytes 会漏掉依赖与 overlap。

## 8. 代码与实验映射

A2 包含 `fsdp_calcs`、`tp_calcs`、`fsdp_tp_calcs`，以及教学 FSDP correctness/accounting。仓库报告：

- 对 TP/2D 的结论是公式推导，没有多卡实测；
- 对 FSDP 只有 CPU/Gloo 数值测试；
- 单张 RTX 6000D 无法生成 2-GPU NCCL timeline；
- 后续应补 all-gather/reduce-scatter trace、prefetch 峰值、不同 wrap 粒度和 topology-aware group mapping。

建议实验矩阵：

1. 固定 global batch，对 DDP/FSDP 比 step time 与 peak memory；
2. 扫 FSDP unit size、reshard、prefetch、communication dtype；
3. 扫 TP degree，记录 GEMM shape 与每层 collective；
4. 扫 PP stages/microbatches，记录 bubble 和 stage skew；
5. 对每个 2D 网格分别记录 TP/FSDP exposed time，而不只给总吞吐。

## 9. Cost Model 与研究方法

### 9.1 容量约束先于速度优化

候选策略必须先满足每 rank：

\[
M_{\text{state}}+M_{\text{activation}}+M_{\text{full-unit peak}}
+M_{\text{workspace}}+M_{\text{communication buffer}}
<M_{\text{device}}.
\]

再估算 compute、collective、P2P、bubble 和 overlap。若 cost model 只算平均 memory，会漏掉
FSDP prefetch/PP warmup 的瞬时峰值。

### 9.2 自动并行与搜索空间

并行计划可视为：

- graph nodes 的 device/stage partition；
- tensor dimensions 的 mesh placement；
- collective insertion；
- schedule/prefetch/checkpoint 决策。

搜索空间巨大。Mesh TensorFlow/GShard 用 layout rules，Alpa 用 inter/intra-operator cost
model 自动搜索 [[8]](#ref-8)[[9]](#ref-9)[[10]](#ref-10)。自动计划仍依赖准确 profile，
且 compiler 预测误差/动态 shape 会导致次优。

### 9.3 Benchmark matrix

至少扫描 world size、nodes、DP/TP/PP/FSDP/CP degree、microbatch、sequence、wrap/bucket。
每点记录：

- per-rank peak memory 与 OOM；
- tokens/s、step time、MFU；
- compute/collective/P2P/bubble/exposed time；
- load/stage imbalance；
- loss/gradient correctness；
- time-to-quality 与 checkpoint overhead。

强扩展固定 global work，弱扩展固定 per-rank work；二者不能混成一条 scaling efficiency。

### 9.4 预注册假设示例

> 在 2 nodes×8 GPUs、单层可放入 4-way TP 时，TP=4 + FSDP=4 比 TP=8 + FSDP=2
> 更快，因为后者增加每层 node-local TP collective，而参数 all-gather 仍能被 backward
> overlap；但 TP=4 可能提高 per-rank parameter peak。

该假设可由 memory、timeline 和 step-time 同时证伪。

## 10. 易错点

- 把 TP 当成“不通信的模型切分”；
- column/row parallel 维度切反，导致每层都 all-gather；
- hidden 可整除但 attention heads/KV heads 不可整除；
- PP microbatch 太小，bubble 下降却 GEMM 效率崩溃；
- FSDP all-gather 后没有及时 reshard，峰值接近 DDP；
- forward 后释放参数，却忘记 backward 需要重新 gather；
- mixed-precision communication 覆盖 FP32 master shard；
- tied parameters 被不同 FSDP units 分割；
- checkpoint 只保存一个 rank 的 shard，无法恢复完整分布式状态；
- 把教学实现的 all-reduce+slice 误称为真正 reduce-scatter；
- 2D rank groups 构造不一致导致 collective deadlock；
- 忽略网络拓扑，只按 GPU 编号连续分组。
- 把 EP 的 all-to-all 当作可任意跨慢链路：dispatch/combine 数据相关且不均匀，须与 NVLink/IB 拓扑共置并用容量因子兜底。
- 只背 ZeRO stage 名称不看通信语义：ZeRO-3/FSDP 通信约 3S 而非 DDP 的 2Φ，offload 还会把瓶颈转移到 CPU/NVMe 带宽。

## 11. Checklist

- [ ] 写明每个并行维度及 process group
- [ ] 列出每层输入、权重、输出的 global/local shape
- [ ] 标注 forward/backward 的每个 collective
- [ ] 分别核算 parameter、gradient、optimizer、activation memory
- [ ] 通信量注明 per-rank/aggregate、元素/bytes、SUM/AVG
- [ ] TP 检查 hidden/head/KV-head/FFN 可整除性
- [ ] PP 记录 microbatch、schedule、bubble、stage skew
- [ ] FSDP 记录 gather/reshard/prefetch 生命周期
- [ ] 正确性与性能分别在合适 backend 验证
- [ ] 2D groups 映射到真实 NVLink/跨节点拓扑

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| TP output 不一致 | shard axis/collective | row/column 切反 |
| TP 显存未降 | activation gather | 过早 all-gather、未 SP |
| PP 吞吐低 | timeline/stage time | bubble、stage skew |
| PP 显存高 | schedule/activation | GPipe 保存过多 microbatches |
| FSDP 峰值接近 DDP | memory snapshot | unit 太大、未 reshard、prefetch 重叠 |
| FSDP 通信很多 | wrap policy | units 太小、latency-bound |
| mixed precision 发散 | master/reduce dtype | 低精度累加 |
| tied weight 错误 | wrap/state dict alias | 跨 units 或 load 后 alias 丢失 |
| checkpoint 无法换 world size | state-dict type | 缺 sharding metadata/reshard |
| 2D/3D hang | process groups | mesh rank 映射不一致 |
| 多节点骤降 | topology trace | 高频 TP/CP 跨慢链路 |

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| TP 提速不明显 | 通信量 \(3S{+}2S\) 与互联域 | TP 组跨节点、all-reduce 未 overlap |
| PP 气泡占比大 | microbatch 数与调度 | microbatch 不足、未用 1F1B/interleaved |
| FSDP all-gather 慢 | prefetch 与分片大小 | 未开 backward prefetch、分片过碎 |
| FSDP 仍 OOM | wrap 策略与 reshard_after_forward | wrap 粒度不当、未及时 reshard、未 offload |
| 3D 组合后不收敛 | 梯度裁剪的归约顺序 | 裁剪须在完整梯度上跨组归约后进行 |
| ZeRO-3 训练停滞 | 通信串行化 | 通信未与计算重叠、prefetch 失效 |

## 12. 讨论：效度威胁与结论边界

### Construct validity
- theoretical memory lower bound 不含 full-unit peak/workspace；
- total communication bytes 不等于 exposed communication；
- MFU/throughput 不等于 time-to-quality；
- weak scaling 不代表固定训练任务加速。

### Internal validity
- 并行度变化常同时改变 global batch、microbatch 与 optimizer recipe；
- stage/wrap/bucket tuning budget 不等会造成 selection bias；
- 只测 rank0 隐藏 straggler/peak；
- profiler、checkpoint、compile 与 first-step overhead 口径不一致。

### External validity
- 单节点 NVLink mapping 不外推跨节点；
- 小模型 layer balance 不外推大模型；
- 教学 FSDP correctness 不代表生产 memory/performance；
- 一个 topology 的最优 mesh 不泛化到 TPU/其他 GPU cluster。

论文式结论必须限定 model、mesh、topology、dtype、batch/context、schedule、software 与
measurement protocol，并公开 OOM/hang/failed configurations。

## 面试要点速记

**高频问题与答题要点**

1. **Q：TP 的通信量与适用边界？** 要点：每层前向 2 次、反向 2 次 all-reduce
   （合计 ~4S 量级）；必须限制在 NVLink 单机域内，跨节点会被通信淹没。
2. **Q：PP 气泡占比公式？** 要点：**(p−1)/(m+p−1)**；增大 microbatch 数 m、
   用 1F1B/interleaved schedule 降气泡。
3. **Q：FSDP 与 TP 怎么选？** 要点：FSDP（=ZeRO-3）通信 3S、参数全分片，
   适合跨机大模型；TP 通信频繁但算子级并行无参数复制，适合单机多卡；
   超大规模组合为 3D（数据×流水×张量）。
4. **Q：3D 并行下梯度裁剪的顺序？** 要点：所有并行组归约完成后，在完整梯度
   上裁剪，再分发回各分片；顺序错误是最常见的静默 bug。
5. **Q：EP（专家并行）的通信形态与 all-reduce 有何不同？** 要点：token 路由产生
   不均匀 all-to-all（dispatch + combine 各一轮），数据相关、天然不均；
   DeepSeek-V3 用 EP64（256 路由专家/64 卡）+ PP，IB 与 NVLink 混合通信流
   与计算时空重叠，据报道 32-GPU 集群设备利用率约 98%。
6. **Q：SP 与 CP 的区别？** 要点：SP 切 TP region 外 LayerNorm/Dropout 的
   复制激活（Megatron SP+TP 长序列激活约降 5×）；CP 切 attention 序列/KV
   本身（ring attention）；Ulysses 支持 1M tokens（激活 O(n) 优化）。
7. **Q：FSDP 与 ZeRO-3 什么关系？FSDP2 改了什么？** 要点：FSDP≈ZeRO-3 全参数
   分片 + 自动 bucket；FSDP2（torch 2.4）per-parameter sharding + 动态重分片，
   消除 flat buffer 的 alias/生命周期问题。
8. **Q：ZeRO-3/FSDP 为什么能跨机、TP 不能？** 要点：DDP ring all-reduce
   每 step 约 2Φ、FSDP 约 3S——低频、可与计算 overlap；TP 每层前向+反向
   各 2 次 all-reduce，随 tokens/TP size 增长且在关键路径上。

**必背数字**

- TP 4S（层内高频）/ PP 气泡 (p−1)/(m+p−1) / FSDP 3S；三者的量纲都是
  “每步每参数字节数”，可直接进 cost model。

**工业界参照（2024–2026 检索口径）**

- Megatron-LM 3D 并行官方启动模板：`--tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 --data-parallel-size 8`（TP×PP×DP=64 GPU 基准配方）。
- DeepSeek-V3 训练用 EP64（256 个路由专家分布 64 卡）+ 流水线并行；跨节点 IB 与片内 NVLink 混合通信流、传输与计算时空重叠，据报道 32-GPU 集群设备利用率约 98%。
- Megatron SP+TP 联合使长序列激活显存约降 5×；Ulysses 序列并行支持 1M tokens 长序列（激活 O(n) 优化）。
- 通信量口径：DDP ring all-reduce 每 step 约 2Φ；TP 每层前向+反向各 2 次 all-reduce（随 tokens 与 TP size 增长）→ 只适合机内高带宽域。
- FSDP≈ZeRO-3 全参数分片 + 自动 bucket；FSDP2（torch 2.4）per-parameter sharding + 动态重分片；DeepSpeed ZeRO-Offload/Infinity 支持 CPU/NVMe 卸载，据报道单卡可训千亿参数。

## 行业现状与最新进展（2024–2026）

### 1.1 3D 并行的业界标准配方

超大模型（GPT-3、LLaMA-70B 级）训练的通用组合业界已收敛为：**机内 TP（NVLink 高速域）+ 跨机 PP（按层切分、设备间传激活）+ 外层 DP（同步多个 TP+PP 副本）**，外加 SP/CP 组成完整方案（业界总结为“3D 并行 + 序列并行”）。Megatron-LM 官方示例模板：

```bash
--tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 --data-parallel-size 8
```

TP 按行/列切分 attention 与 FFN 矩阵；PP 把 L 层划分到多 GPU；DP 在最外层复制并同步。

| 本讲概念 | 工业界实践/数字 | 口径 |
|---|---|---|
| TP（column/row sharding） | 机内 NVLink 域，典型 TP=4–8；每层前向+反向各 2 次 all-reduce | 通信量口径 |
| PP（microbatch + bubble） | 跨机按层切分；1F1B/interleaved 缓解 (p−1)/(m+p−1) 气泡 | 教学推导 |
| DP 外层 | ring all-reduce 每 step 约 2Φ，低频、可 overlap | 通信量口径 |
| ZeRO/FSDP | FSDP≈ZeRO-3；FSDP2（torch 2.4）per-parameter sharding + 动态重分片 | 官方版本口径 |
| EP（MoE） | DeepSeek-V3 EP64（256 专家/64 卡）+ PP；IB+NVLink 混合流，据报道 32-GPU 集群利用率约 98% | DeepSeek-V3 报告 |
| SP/CP | Megatron SP+TP 激活约降 5×；Ulysses 支持 1M tokens | 据报道 |

### 1.2 MoE 时代的专家并行（EP）

DeepSpeed-MoE 提供标准 EP 实现。DeepSeek-V3 的训练配方是 **EP64——256 个路由专家分布 64 卡——叠加流水线并行**；关键工程点：跨节点 IB 与片内 NVLink 混合通信流，数据传输与计算时空重叠，据报道 32-GPU 集群设备利用率约 98%。EP 的通信形态是 token 级 all-to-all（dispatch + combine 各一轮），与 DP 的 all-reduce 本质不同：数据相关、天然不均匀，需要容量因子与负载均衡损失兜底。

### 1.3 序列并行与上下文并行的工业落地

- SP 与 TP 联合：Megatron SP+TP 使长序列激活显存约降 5×；代码入口 `megatron/core/tensor_parallel/layers.py`（2025-05 主分支合并 sequence_parallel）。
- 长上下文：Ulysses 序列并行支持 1M tokens 长序列（激活 O(n) 优化）；USP 把 SP 与 ring-attention 风格 CP 统一（本讲 §6）。
- 数据并行一侧的演进：FSDP≈ZeRO-3 全参数分片 + 自动 bucket + 异构分片；FSDP2（torch 2.4）per-parameter sharding + 动态重分片；ZeRO-Offload/Infinity 支持 CPU/NVMe 卸载，据报道单卡可训千亿参数——“能训”与“训得快”的边界持续外推。

**对本讲学习者的启示**：工业配方不是另起炉灶——“TP 机内、PP 跨机、DP/ZeRO 外层、MoE 加 EP、长序列加 SP/CP”正是本讲通信量账本（TP 每层各 2 次 all-reduce、DDP 约 2Φ、FSDP 3S、气泡 (p−1)/(m+p−1)）的第一性推导。学任何新框架（FSDP2、DeepSpeed-MoE、Ulysses）都先问四个问题：切哪个维度、用什么 collective、频率多高、落在哪条链路。

## 大厂面试真题与答题框架

以下均为**高频面试题（公开面经风格）**，不指向任何具体公司或年份。

**题目 1：PP 气泡占比公式是什么？如何缓解？**
- 考点：气泡推导、microbatch schedule、与 batch recipe 的耦合。
- 答题框架：1) 画 p 级 m 微批时间线，气泡段 = (p−1) 个 step 单位；2) 总时间 ≈ (m+p−1)(t_f+t_b)，占比 = **(p−1)/(m+p−1)**；3) 增大 m 缓解——但 global batch = b_μ·m·d，m 受 optimizer recipe 耦合限制；4) 1F1B 降激活峰值（不降稳态气泡），interleaved 1F1B 用 virtual stages 缩短气泡；5) 与 DP/TP 组合摊薄。
- 加分项：指出 stage skew 会让公式失效（step 时间由最慢 stage 决定，T ≥ m·max T_s）。
- 踩坑：说“1F1B 消除气泡”（只降峰值）；忽略 m 太大后 GEMM 效率与 P2P 开销。

**题目 2：TP 为什么只在机内做？**
- 考点：TP 通信频率/量 vs DP、ZeRO 的量级对比。
- 答题框架：1) TP 每层前向+反向各 2 次 all-reduce，通信随 tokens 与 TP size 增长，且在计算关键路径上；2) 对比 DDP ring all-reduce 每 step 约 2Φ、FSDP 约 3S——低频、可 overlap；3) NVLink 与跨节点 IB 带宽/延迟差数量级；4) 结论：TP 限 NVLink 域，跨机交给 PP/DP/FSDP。
- 加分项：引 Megatron 模板 TP=4；补 SP 切掉 LayerNorm 区域复制激活（SP+TP 激活约降 5×）。
- 踩坑：只说“机内带宽高”而不给通信量账本。

**题目 3：PP 各 stage 切分不均怎么办？**
- 考点：stage balance、partition 依据。
- 答题框架：1) 实测 per-layer forward/backward time 与 memory；2) embedding、LM head、不同 attention/MLP shape 单独计权；3) 非均匀层数切分（按累计成本做整数划分）；4) 把 recompute 与 stage 间通信并入成本；5) interleaved virtual stages 缓解残余 skew。
- 加分项：skew 与 bubble 叠加分析——最慢 stage 决定 T_steady ≥ m·max T_s。
- 踩坑：按层数平均切；用 FLOPs 代替实测时间（忽略 embedding/LM head 与 recompute）。

**题目 4：MoE 的 EP 通信是怎么发生的？**
- 考点：all-to-all dispatch/combine、与 all-reduce 的本质区别。
- 答题框架：1) router 为每 token 选 top-k 专家 → token 必须物理到达专家所在卡：dispatch all-to-all；2) 专家计算后结果回原 rank：combine all-to-all；3) 通信量数据相关、天然不均——容量因子与负载均衡损失兜底；4) 工程实践：跨节点 IB 与片内 NVLink 混合通信流、与计算时空重叠（DeepSeek-V3 EP64：256 路由专家/64 卡 + PP，据报道 32-GPU 集群利用率约 98%）。
- 加分项：DeepSpeed-MoE 是标准 EP 参考实现；EP 与 DP 组合时先 EP 域 combine 再 DP 归约。
- 踩坑：把 all-to-all 等同 all-reduce（前者每消息目的 rank 不同、不均匀）。

**题目 5：ZeRO-3/FSDP 的通信量与 offload 的代价？**
- 考点：2Φ vs 3S、offload 是“能训”不是“训得快”。
- 答题框架：1) DDP ring all-reduce 每 step 约 2Φ；2) FSDP 每 unit all-gather + reduce-scatter（+backward 前 re-gather）≈ 3S——更多通信换显存；3) ZeRO-1/2/3 分片阶梯（optimizer/gradient/parameter）；4) ZeRO-Offload/Infinity 卸载到 CPU/NVMe，据报道单卡可训千亿参数，但吞吐显著低于纯 GPU——新瓶颈是 PCIe/NVMe 带宽；5) FSDP2（torch 2.4）per-parameter sharding + 动态重分片改善工程性。
- 加分项：prefetch 把下一 unit all-gather 与当前 backward 重叠；unit 粒度的 latency/bandwidth 权衡。
- 踩坑：说 ZeRO-3“省通信”；把 offload 当免费午餐。

**题目 6：SP 和 CP 各解决什么问题？**
- 考点：序列维度的两类切分不可混用。
- 答题框架：1) SP 切 TP region 外 LayerNorm/Dropout 的复制激活，与 TP 联合（Megatron SP+TP 长序列激活约降 5×）；2) CP 切 attention 序列/KV 本身（ring attention + online softmax）；3) 长上下文必须 CP：Ulysses 支持 1M tokens（激活 O(n) 优化）；4) 选择依据：短序列用 SP 消冗余，超长上下文必须 CP。
- 加分项：Megatron SP 代码入口 `tensor_parallel/layers.py`（2025-05 主分支合并）；USP 统一框架。
- 踩坑：把 SP 当长上下文方案；混淆 SP（消复制）与 CP（切 KV）。

## 系统设计题

**设计题 1：为 DeepSeek-V3 量级 MoE（256 路由专家）设计多机并行布局**
- 需求澄清：训练还是推理？节点拓扑（8 GPU/节点、机内 NVLink、跨节点 IB）？目标 tokens/s 与 MFU？global batch 与激活参数预算？
- 规模估算：EP64 = 256 路由专家分布 64 卡（每卡 4 专家）+ PP 沿层切分 + DP 外层复制；参照口径：IB 与 NVLink 混合通信流、传输与计算时空重叠，据报道 32-GPU 集群设备利用率约 98%；SP+TP 使长序列激活约降 5×。
- 架构：mesh `[DP, PP, EP, TP/SP]`——TP/SP 放机内 NVLink 域，PP 跨相邻节点传激活，EP 占 64 卡专家域（token all-to-all），DP 最外层 all-reduce（每 step 约 2Φ）。
- trade-off：

| 决策 | 收益 | 代价 |
|---|---|---|
| EP 增大 | 单卡专家显存↓ | all-to-all 跨节点占比↑、延迟敏感 |
| PP 级数增大 | 单卡层数↓ | 气泡 (p−1)/(m+p−1)↑ |
| TP 超出机内 | 无 | 每层 all-reduce 跨慢链路，不可接受 |
| DP 增大 | 吞吐近线性 | 每 step 约 2Φ all-reduce |

- 评测方案：tokens/s、MFU、per-rank peak memory、all-to-all exposed time（目标≈0，靠重叠）、PP bubble 实测 vs 公式、专家负载分布（最大/均值比）。
- 追问预案：专家负载不均 → 容量因子 + 辅助损失；EP+DP 梯度归约顺序（先 EP 域 combine 再 DP all-reduce）；故障恢复用 sharded checkpoint 并行 IO。

**设计题 2：为 405B dense 模型设计 1024-GPU 训练布局**
- 需求澄清：dense BF16；heads/KV-heads/FFN 对 TP 的可整除性；单层能否在 t≤8 放下；global batch 与 microbatch 耦合。
- 规模估算：混合精度全账本约 16P bytes（本讲 §4 口径），405B → 总状态约 16×405e9 ≈ 6.5 TB 量级；ZeRO-3 分片后每 rank 约 16P/f；若走 3D 并行则参数随 TP/PP 切分、DP 不复制参数。
- 架构：以 Megatron 模板 TP=4 × PP=2 × DP=8（64 GPU）为单元，1024 GPU = 外层 DP 扩到 128；机内叠 SP 消 LayerNorm 区域复制激活。
- trade-off：

| 方案 | 每 rank 状态 | 高频通信 | 适用 |
|---|---|---|---|
| 纯 DP | 16P | 每 step 2Φ | 放不下，排除 |
| ZeRO-3/FSDP f=1024 | 16P/f | 3S、可 overlap | 状态容量优先 |
| 3D（TP4×PP2×DP128）+ SP | 随 TP/PP 切分 | TP 机内 4S 级 + PP P2P + DP 2Φ | 大带宽集群首选 |

- 评测方案：step time、MFU、per-rank peak memory、TP exposed time（机内应≈0）、PP bubble 实测 vs (p−1)/(m+p−1)、多组配置 Pareto。
- 追问预案：梯度裁剪须全组归约后在完整梯度上做；interleaved 1F1B 降气泡的代价是 P2P 变多；FSDP×TP 替代（t=8, f=128）何时更优。

**设计题 3：大规模训练的故障恢复与弹性训练（checkpoint 策略）**
- 需求澄清：集群规模与节点 MTBF？允许的 RPO（损失步数）/RTO（恢复时长）？存储带宽与是否可用 CPU/NVMe 缓冲？
- 规模估算：千卡级集群节点故障常态化；完整恢复需 model shards + optimizer shards + RNG + sampler + global step（本讲 §4）；sharded checkpoint 并行写入，避免单 rank IO 峰值。
- 架构：分层策略——sharded state dict 并行 IO + 异步后台写（不阻塞训练 step）+ 定期与 on-event 双触发；ZeRO-Offload/Infinity 的 CPU/NVMe 层级可复用为 checkpoint 缓冲，但与 offload 争同一 IO 通道。
- trade-off：

| 策略 | RPO | RTO | IO 开销 |
|---|---|---|---|
| full checkpoint 单点写 | 步数间隔 | 大（单点 IO） | rank0 峰值高 |
| sharded 并行写 | 同 | 小（并行 IO） | 需 metadata/reshard |
| 高频异步写 | 小 | 中 | 常驻带宽占用 |

- 评测方案：checkpoint overhead 占 step 时间比例、RTO 实测、恢复后 loss 曲线连续性、换 world size 的 reshard 成功率。
- 追问预案：换 world size 需 sharding metadata 与原始 parameter mapping；FSDP2 动态重分片对弹性更友好；恢复时 DP/EP process group 重建一致性。

## 代码实现题

**代码题 1：1F1B 调度模拟器**
- 题目：给定 p 个流水级、m 个 micro-batch 与每 op 时长，输出每 stage 的 `(start, end, op, microbatch)` 时间线与气泡占比。
- 考察点：warmup/steady/cooldown 序列、依赖建模、气泡公式验证。

```python
def simulate_1f1b(p: int, m: int, t_f: float = 1.0, t_b: float = 1.0):
    """同步 1F1B 模拟：返回 (timeline, makespan, bubble_fraction)。"""
    seqs = []
    for s in range(p):
        w = max(0, min(m, p - 1 - s))            # warmup forward 数
        ops = [("F", k) for k in range(w)]
        for i in range(m - w):                   # 稳态：1F1B 交替
            ops += [("F", w + i), ("B", i)]
        ops += [("B", k) for k in range(m - w, m)]   # cooldown backward
        seqs.append(ops)

    end, idx, clock = {}, [0] * p, [0.0] * p
    timeline = [[] for _ in range(p)]
    done, total = 0, sum(len(x) for x in seqs)
    while done < total:
        progressed = False
        for s in range(p):
            while idx[s] < len(seqs[s]):
                op, k = seqs[s][idx[s]]
                if op == "F":
                    if s > 0 and ("F", s - 1, k) not in end:
                        break                    # 等上游 forward
                    dep = end[("F", s - 1, k)] if s > 0 else 0.0
                    dur = t_f
                else:
                    if ("F", s, k) not in end:
                        break                    # backward 需先有 forward
                    if s < p - 1 and ("B", s + 1, k) not in end:
                        break                    # 等下游 backward
                    dep = end[("B", s + 1, k)] if s < p - 1 else 0.0
                    dur = t_b
                start = max(clock[s], dep)
                end[(op, s, k)] = start + dur
                clock[s] = start + dur
                timeline[s].append((start, start + dur, op, k))
                idx[s] += 1
                done += 1
                progressed = True
        if not progressed:
            raise RuntimeError("deadlock: 依赖不满足，检查 schedule")
    makespan = max(clock)
    bubble = 1.0 - m * (t_f + t_b) / makespan    # 每 stage 满载 m*(tf+tb)
    return timeline, makespan, bubble
```

- 验收标准：p=2、m=4、t_f=t_b=1 → makespan=10、bubble=0.2=(p−1)/(m+p−1)；m 增大时 bubble→0、p 增大时上升；时间线内同 stage 事件不重叠、F(k,s) 晚于 F(k,s−1)。

**代码题 2：列并行/行并行线性层前向（含 all-reduce 位置）**
- 题目：实现 ColumnParallelLinear 与 RowParallelLinear，组成 MLP（W1 列切 → gelu → W2 行切），标注唯一通信点。
- 考察点：切分维度选择、partial sum、all-reduce 位置、中间激活保持分片。

```python
import torch
import torch.distributed as dist

class ColumnParallelLinear(torch.nn.Module):
    """Y=XW 按输出维 N 切 t 份：每 rank 持 [K, N/t]，前向无通信。"""
    def __init__(self, in_f: int, out_f: int, world_size: int):
        super().__init__()
        assert out_f % world_size == 0
        self.weight = torch.nn.Parameter(
            torch.randn(out_f // world_size, in_f))
    def forward(self, x):                    # x: [M, K]（复制）
        return x @ self.weight.t()           # [M, N/t]，无需通信

class RowParallelLinear(torch.nn.Module):
    """Y=XW 按输入维 K 切 t 份：每 rank 持 [K/t, N]，输出 partial sum。"""
    def __init__(self, in_f: int, out_f: int, world_size: int):
        super().__init__()
        assert in_f % world_size == 0
        self.weight = torch.nn.Parameter(
            torch.randn(out_f, in_f // world_size))
    def forward(self, x_shard):              # x_shard: [M, K/t]
        y = x_shard @ self.weight.t()        # [M, N] partial sum
        dist.all_reduce(y)                   # 唯一通信点：块内 1 次 all-reduce
        return y

def tp_mlp(x, w1, w2):
    h = w1(x)                                # [M, d_ff/t] 保持分片
    h = torch.nn.functional.gelu(h)          # 逐元素，本地计算
    return w2(h)                             # 行切 + all-reduce
```

- 验收标准：torchrun 2 卡下与单卡 reference `allclose`（rtol=1e-5）；中间激活 shape 为 [M, d_ff/t]；profiler 中每个 MLP 块前向仅 1 次 all-reduce。

**代码题 3：EP all-to-all 通信量模拟器**
- 题目：给定 rank 数、每 rank token 数、专家数、top-k，输出 all-to-all 发送矩阵与 dispatch+combine 的 token 副本总数。
- 考察点：路由不均匀、all-to-all 与 all-reduce 的区别、EP 规模换算。

```python
import hashlib

def route(token_id: int, num_experts: int, top_k: int):
    h = hashlib.sha256(str(token_id).encode()).digest()
    return sorted({b % num_experts for b in h})[:top_k]

def ep_allto_all(num_ranks: int, tokens_per_rank: int,
                 num_experts: int, top_k: int):
    assert num_experts % num_ranks == 0
    epr = num_experts // num_ranks          # 每 rank 持专家数（EP64/256 → 4）
    send = [[0] * num_ranks for _ in range(num_ranks)]
    for src in range(num_ranks):
        for t in range(tokens_per_rank):
            for e in route(src * tokens_per_rank + t, num_experts, top_k):
                send[src][e // epr] += 1    # 目标 rank = 专家所属 rank
    copies = sum(map(sum, send))
    return send, copies, 2 * copies         # dispatch + combine 各一轮
```

- 验收标准：每行和 = tokens_per_rank×top_k；num_ranks=64、num_experts=256 时 epr=4（与 DeepSeek-V3 EP64 口径一致）；top_k=1 时每 token 恰有一个目的 rank。

## 13. 小结

TP 切宽度并高频通信 activation，PP 切深度并承受 bubble，FSDP 切参数状态并按模块 gather/scatter。二维并行的关键不是把公式相乘，而是把两类不同频率、不同依赖的通信放到合适互联上，并用局部 shape、生命周期和实测 timeline 验证。当状态仍超出显存时，offload 与 context parallelism 把边界进一步外推，但分别受限于卸载带宽与全量 attention 计算。

## 参考文献

<a id="ref-1"></a>[1] M. Shoeybi et al. “Megatron-LM: Training Multi-Billion
Parameter Language Models Using Model Parallelism.” arXiv:1909.08053, 2019.
https://arxiv.org/abs/1909.08053

<a id="ref-2"></a>[2] Y. Huang et al. “GPipe: Efficient Training of Giant
Neural Networks using Pipeline Parallelism.” *NeurIPS*, 2019.
https://arxiv.org/abs/1811.06965

<a id="ref-3"></a>[3] A. Harlap et al. “PipeDream: Generalized Pipeline
Parallelism for DNN Training.” *SOSP*, 2019.
https://doi.org/10.1145/3341301.3359646

<a id="ref-4"></a>[4] D. Narayanan et al. “Memory-Efficient Pipeline-Parallel
DNN Training.” *ICML*, 2021. https://arxiv.org/abs/2006.09503

<a id="ref-5"></a>[5] S. Rajbhandari et al. “ZeRO: Memory Optimizations Toward
Training Trillion Parameter Models.” *SC*, 2020.
https://arxiv.org/abs/1910.02054

<a id="ref-6"></a>[6] Y. Zhao et al. “PyTorch FSDP: Experiences on Scaling
Fully Sharded Data Parallel.” *VLDB*, 2023.
https://arxiv.org/abs/2304.11277

<a id="ref-7"></a>[7] D. Narayanan et al. “Efficient Large-Scale Language Model
Training on GPU Clusters Using Megatron-LM.” *SC*, 2021.
https://arxiv.org/abs/2104.04473

<a id="ref-8"></a>[8] N. Shazeer et al. “Mesh-TensorFlow: Deep Learning for
Supercomputers.” arXiv:1811.02084, 2018. https://arxiv.org/abs/1811.02084

<a id="ref-9"></a>[9] D. Lepikhin et al. “GShard: Scaling Giant Models with
Conditional Computation and Automatic Sharding.” *ICLR*, 2021.
https://arxiv.org/abs/2006.16668

<a id="ref-10"></a>[10] L. Zheng et al. “Alpa: Automating Inter- and
Intra-Operator Parallelism for Distributed Deep Learning.” *OSDI*, 2022.
https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin

<a id="ref-11"></a>[11] V. A. Korthikanti et al. “Reducing Activation
Recomputation in Large Transformer Models.” *MLSys*, 2023.
https://arxiv.org/abs/2205.05198

<a id="ref-12"></a>[12] S. Rajbhandari, O. Ruwase, J. Rasley, et al.
“ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep
Learning.” *SC*, 2022. [link](https://arxiv.org/abs/2104.07857)

<a id="ref-13"></a>[13] H. Liu et al. “Ring Attention with Blockwise
Transformers for Near-Infinite Context.” *ICLR*, 2024.
https://arxiv.org/abs/2310.01889

<a id="ref-14"></a>[14] J. Fang, S. Zhao. “USP: A Unified Sequence
Parallelism Approach for Long Context Generative AI.” arXiv:2405.07719,
2024. [link](https://arxiv.org/abs/2405.07719)

## 延伸阅读与复现材料

- [CS336 Lecture 8 官方讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)
- [CS336 Lecture 7 可执行讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [A2 Systems 官方题面](../assignments/spring2026/assignment2-systems/cs336_assignment2_systems.pdf)
- 本仓库：[A2 FSDP/并行策略报告](../assignments/spring2026/assignment2-systems/report/writeup.pdf)
- [Megatron-LM（NVIDIA 官方仓库）](https://github.com/NVIDIA/Megatron-LM)（访问日期 2026-10-04）
- [DeepSpeed（微软官方仓库）](https://github.com/microsoft/DeepSpeed)（访问日期 2026-10-04）
- [DeepSeek-V3 技术报告（arXiv:2412.19437）](https://arxiv.org/abs/2412.19437)（访问日期 2026-10-04）
- [ZeRO 论文（arXiv:1910.02054）](https://arxiv.org/abs/1910.02054)（访问日期 2026-10-04）
