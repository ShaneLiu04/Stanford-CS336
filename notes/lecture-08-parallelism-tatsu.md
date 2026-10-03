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
- 文档性质：原创中文自学综述，非课程提交
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

**必背数字**

- TP 4S（层内高频）/ PP 气泡 (p−1)/(m+p−1) / FSDP 3S；三者的量纲都是
  “每步每参数字节数”，可直接进 cost model。

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
